import torch
import torch.nn as nn
from timm.layers import create_act, create_norm
from timm.layers.create_act import get_act_layer
from timm.layers.create_norm import get_norm_layer

from src.utilities.logging import log, once


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


# Register custom norm
create_norm._NORM_MAP["zeromeanrmsnorm"] = Qwen3NextRMSNorm


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


@once
def _register_new_acts():
    new_acts = {
        "swiglu": SwiGLUAct,
        "poly_norm": PolyNormAct,
    }
    create_act._ACT_LAYER_DEFAULT.update(new_acts)
    create_act._ACT_LAYER_ME.update(new_acts)


@once
def _register_new_norms():
    try:
        from flash_attn.ops.rms_norm import RMSNorm as FlashRMSNorm

        create_norm._NORM_MAP.update({"flash_rms_norm": FlashRMSNorm})  # type: ignore
    except ImportError:
        pass


_register_new_acts()
_register_new_norms()
log(
    "[NormActRegister]: Register activation ('swiglu' and 'polynorm_act'), ",
    "normalization ('flash_rms_norm')",
    level="debug",
)
