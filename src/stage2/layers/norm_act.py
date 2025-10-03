from functools import partial

import torch
import torch.nn as nn
from loguru import logger
from timm.layers import create_act, create_norm

from src.utilities.logging import once

from .ops.triton_rms_norm import TritonRMSNorm2dFunc


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class Qwen3NextRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Qwen3Next is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class TritonRMSNorm2d(nn.LayerNorm):
    def zero_out(self):
        nn.init.constant_(self.weight, 0)
        nn.init.constant_(self.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_numel = x.numel()
        if input_numel >= 1 << 31:
            num_chunks = (input_numel - 1) // (1 << 31) + 1
            output = []
            for x_chunk in x.chunk(num_chunks, dim=2):
                output.append(
                    TritonRMSNorm2dFunc.apply(
                        x_chunk.contiguous(), self.weight, self.bias, self.eps
                    )
                )
            output = torch.cat(output, dim=2)
            return output
        else:
            return TritonRMSNorm2dFunc.apply(  # type: ignore
                x.contiguous(), self.weight, self.bias, self.eps
            )


class AdaptiveGroupNorm(nn.Module):
    def __init__(self, in_chan, cond_chan, num_groups=32, eps=1e-6):
        super().__init__()
        self.gn = nn.GroupNorm(
            num_groups=num_groups, num_channels=in_chan, eps=eps, affine=False
        )
        self.gamma = nn.Linear(cond_chan, in_chan)
        self.beta = nn.Linear(cond_chan, in_chan)
        self.eps = eps

    def forward(self, x, z: torch.Tensor):
        B, C, _, _ = x.shape
        fz = z.flatten(2)

        # calcuate var for scale
        scale = fz.var(dim=-1) + self.eps  # not unbias
        scale = scale.sqrt()
        scale = self.gamma(scale).view(B, C, 1, 1)

        # calculate mean for bias
        bias = fz.mean(dim=-1)
        bias = self.beta(bias).view(B, C, 1, 1)

        x = self.gn(x)
        x = scale * x + bias

        return x


# * --- Activations --- * #


class PolyNormAct(torch.nn.Module):
    """
    A trainable activation function introduced in https://arxiv.org/html/2411.03884v1.
    The code is copied from https://github.com/BryceZhuo/PolyCom?tab=readme-ov-file/README.md
    taken from https://huggingface.co/Motif-Technologies/Motif-2.6B/blob/main/modeling_motif.py#L26
    """

    def __init__(self, eps=1e-6):
        super(PolyNormAct, self).__init__()
        self.weight = torch.nn.Parameter(torch.ones(3) / 3)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.eps = eps

    @torch.compile
    def _norm(self, x):
        return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (
            self.weight[0] * self._norm(x**3)
            + self.weight[1] * self._norm(x**2)
            + self.weight[2] * self._norm(x)
            + self.bias
        )


class SwiGLUAct(nn.Module):
    def __init__(self, alpha: float = 1.702, limit: float = 7.0):
        super().__init__()
        self.alpha = alpha
        self.limit = limit

    def forward(self, x_glu, x_linear):
        alpha, limit = self.alpha, self.limit
        # x_glu, x_linear = x[..., ::2], x[..., 1::2]
        # Clamp the input values
        x_glu = x_glu.clamp(min=None, max=limit)
        x_linear = x_linear.clamp(min=-limit, max=limit)
        out_glu = x_glu * torch.sigmoid(alpha * x_glu)
        # Note we add an extra bias of 1 to the linear layer
        return out_glu * (x_linear + 1)


# Norm registration


class ActLayerMeta(nn.Module):
    def __init__(self, act_layer, name: str, **kwargs):
        super().__init__()
        self.act_layer = act_layer
        self.name = name
        self.kwargs = kwargs

    def forward(self, x):
        return self.act_layer(x, **self.kwargs)

    def extra_repr(self):
        words = self.name.split("_")
        first_cap_name = "_".join([w.capitalize() for w in words])
        return f"{first_cap_name}()"


@once
def _register_new_acts():
    new_acts = {
        "swiglu": SwiGLUAct,
        "poly_norm": PolyNormAct,
    }
    create_act._ACT_LAYER_DEFAULT.update(new_acts)
    create_act._ACT_LAYER_ME.update(new_acts)
    logger.debug(f"[Timm registered new acts]: 'swiglu', 'poly_norm'")

    try:
        from fla.modules.activations import fast_gelu_impl, swiglu, swish

        get_act_layer = lambda act, name: partial(
            ActLayerMeta, name=name, act_layer=act
        )

        fla_acts = {
            "fla_swish": get_act_layer(swish, "fla_swish"),
            "fla_silu": get_act_layer(swish, "fla_silu"),
            "fla_fast_gelu": get_act_layer(fast_gelu_impl, "fla_fast_gelu"),
            "fla_swiglu": get_act_layer(swiglu, "fla_swiglu"),
        }

        create_act._ACT_LAYER_DEFAULT.update(fla_acts)
        create_act._ACT_LAYER_ME.update(fla_acts)

        fla_act_names = list(fla_acts.keys())
        logger.debug(f"[Timm registered FLA acts]: {', '.join(fla_act_names)}")
    except ImportError:
        logger.debug("[FLA not available, skipping FLA activation registration]")


@once
def _register_new_norms():
    try:
        from flash_attn.ops.rms_norm import RMSNorm as FlashRMSNorm

        create_norm._NORM_MAP.update({"flashrmsnorm": FlashRMSNorm})  # type: ignore
    except ImportError:
        pass

    try:
        # from fla.modules import LayerNorm, RMSNorm
        import fla.modules

        create_norm._NORM_MAP.update(
            {"flarmsnorm": fla.modules.RMSNorm, "flalayernorm": fla.modules.LayerNorm}
        )
    except ImportError:
        pass

    create_norm._NORM_MAP["zeromeanrmsnorm"] = Qwen3NextRMSNorm  # type: ignore
    create_norm._NORM_MAP["tritonrmsnorm2d"] = TritonRMSNorm2d  # type: ignore
    create_norm._NORM_MAP["adaptivegroupnorm"] = AdaptiveGroupNorm  # type: ignore
    logger.debug(
        f"[Timm registered new norms]: 'zeromeanrmsnorm', 'flashrmsnorm', 'tritonrmsnorm2d'"
    )


_register_new_acts()
_register_new_norms()


if __name__ == "__main__":
    """
        python -m src.stage2.layers.norm_act
    """
    # import torch

    # x = torch.randn(1, 3, 224, 224).cuda()
    # model = create_norm.create_norm_layer("triton_rmsnorm2d", 3).cuda()
    # y = model(x)
    # print(y.shape)

    # # backward
    # y.sum().backward()
    # for p in model.parameters():
    #     print(p.grad.shape)

    layer = create_act.create_act_layer("fla_silu")
    x = torch.randn(1, 256, 256).cuda()
    y = layer(x)

    y2 = nn.SiLU()(x)

    print(torch.isclose(y, y2))
