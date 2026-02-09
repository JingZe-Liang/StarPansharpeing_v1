from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from timm.layers import create_act, create_norm, create_norm_layer

from src.utilities.logging import once

try:
    from .rmsnorm_triton import TritonRMSNorm2dFunc
except Exception as e:
    logger.warning("TritonRMSNorm2dFunc not available. Falling back to torch.nn.LayerNorm")
    logger.warning(f"Exception: {e}")
    TritonRMSNorm2dFunc = None  # type: ignore


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, bias: bool = False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        y = self._norm(x.float()).type_as(x) * self.weight
        if self.bias is not None:
            y = y + self.bias
        return y

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class RMSNorm2d(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, bias: bool = False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            x(Tensor): Shape [B, C, H, W]
        """
        y = self._norm(x.float()).type_as(x) * self.weight.view(1, -1, 1, 1)
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1)
        return y

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + self.eps)


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
        assert TritonRMSNorm2dFunc is not None, "TritonRMSNorm2dFunc is not available."

        input_numel = x.numel()
        if input_numel >= 1 << 31:
            num_chunks = (input_numel - 1) // (1 << 31) + 1
            output = []
            for x_chunk in x.chunk(num_chunks, dim=2):
                output.append(TritonRMSNorm2dFunc.apply(x_chunk.contiguous(), self.weight, self.bias, self.eps))
            output = torch.cat(output, dim=2)
            return output
        else:
            return TritonRMSNorm2dFunc.apply(  # type: ignore
                x.contiguous(), self.weight, self.bias, self.eps
            )


class AdaptiveGroupNorm(nn.Module):
    def __init__(self, in_chan, cond_chan, num_groups=32, eps=1e-6):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=in_chan, eps=eps, affine=False)
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


# GatedNorm from Qwen team


class PreAffineRMSNorm(nn.Module):
    """
    PreAffine + RMSNorm
    x -> (lambda * x) -> RMSNorm -> *weight (+bias optional)

    PreAffine is a learnable per-channel scaling applied BEFORE RMSNorm.
    """

    def __init__(self, dim: int, eps: float = 1e-6, bias: bool = False, force_fp32=False):
        super().__init__()
        self.pre_affine = nn.Parameter(torch.ones(dim))  # λ1 in the paper's spirit
        self.norm = create_norm_layer("rmsnormfp32" if force_fp32 else "rmsnorm", dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.pre_affine
        return self.norm(x)


class PreAffineRMSNorm2d(nn.Module):
    """
    PreAffine + RMSNorm2d
    x -> (lambda * x) -> RMSNorm2d -> *weight (+bias optional)
    """

    def __init__(self, dim: int, eps: float = 1e-6, bias: bool = False, force_fp32=False):
        super().__init__()
        self.pre_affine = nn.Parameter(torch.ones(dim))
        self.norm = create_norm_layer("rmsnorm2dfp32" if force_fp32 else "rmsnorm2d", dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.pre_affine.view(1, -1, 1, 1)
        return self.norm(x)


class _GatedNorm(nn.Module):
    """
    GatedNorm as a POST-NORM plugin:
      y = Norm(x)
      gate = sigmoid(W_up(swish(W_down(y))))
      y' = gate ⊙ y

    Low-rank bottleneck: W_down: dim->rank, W_up: rank->dim

    B, T, D = 2, 4, 128
    x = torch.randn(B, T, D)

    pre = PreAffineRMSNorm(D)
    gn = GatedRMSNorm(D, rank=16)
    both = PreAffineGatedRMSNorm(D, rank=16)

    y_pre = pre(x)
    y_gn = gn(x)
    y_both = both(x)

    print(y_pre.shape, y_gn.shape, y_both.shape)
    """

    def __init__(self, dim: int, rank: int = 16, dropout: float = 0.0):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.normal_(self.down.weight, mean=0.0, std=1e-2)
        nn.init.normal_(self.up.weight, mean=0.0, std=1e-2)
        nn.init.constant_(self.up.bias, 2.0)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        g = self.up(F.silu(self.down(y)))  # swish/silu
        g = torch.sigmoid(g)
        g = self.dropout(g)
        return g * y


class _GatedNorm2d(nn.Module):
    def __init__(self, dim: int, rank: int = 16, dropout: float = 0.0):
        super().__init__()
        self.down = nn.Conv2d(dim, rank, kernel_size=1, bias=False)
        self.up = nn.Conv2d(rank, dim, kernel_size=1, bias=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.normal_(self.down.weight, mean=0.0, std=1e-2)
        nn.init.normal_(self.up.weight, mean=0.0, std=1e-2)
        assert self.up.bias is not None
        nn.init.constant_(self.up.bias, 2.0)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        g = self.up(F.silu(self.down(y)))
        g = torch.sigmoid(g)
        g = self.dropout(g)
        return g * y


class GatedRMSNorm(nn.Module):
    """
    RMSNorm + GatedNorm
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        rank: int = 16,
        dropout: float = 0.0,
        bias: bool = False,
        force_fp32: bool = False,
    ):
        super().__init__()
        self.norm = create_norm_layer("rmsnormfp32" if force_fp32 else "rmsnorm", dim)
        self.gate = _GatedNorm(dim=dim, rank=rank, dropout=dropout)
        self.force_fp32 = force_fp32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        if self.force_fp32 and in_dtype != torch.float32:
            x = x.float()
        y = self.norm(x)
        y = self.gate(y)
        if self.force_fp32 and in_dtype != torch.float32:
            y = y.to(dtype=in_dtype)
        return y


class PreAffineGatedRMSNorm(nn.Module):
    """
    PreAffine + RMSNorm + GatedNorm (full chain)
    Useful if you want to test "all together".
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        rank: int = 16,
        dropout: float = 0.0,
        bias: bool = False,
        force_fp32: bool = False,
    ):
        super().__init__()
        self.pre = nn.Parameter(torch.ones(dim))
        self.norm = create_norm_layer("rmsnormfp32" if force_fp32 else "rmsnorm", dim)
        self.gate = _GatedNorm(dim=dim, rank=rank, dropout=dropout)
        self.force_fp32 = force_fp32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        if self.force_fp32 and in_dtype != torch.float32:
            x = x.float()
        x = x * self.pre
        y = self.norm(x)
        y = self.gate(y)
        if self.force_fp32 and in_dtype != torch.float32:
            y = y.to(dtype=in_dtype)
        return y


class GatedRMSNorm2d(nn.Module):
    """
    RMSNorm2d + GatedNorm2d
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        rank: int = 16,
        dropout: float = 0.0,
        bias: bool = False,
        force_fp32: bool = False,
    ):
        super().__init__()
        self.norm = create_norm_layer("rmsnorm2dfp32" if force_fp32 else "rmsnorm2d", dim)
        self.gate = _GatedNorm2d(dim=dim, rank=rank, dropout=dropout)
        self.force_fp32 = force_fp32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        if self.force_fp32 and in_dtype != torch.float32:
            x = x.float()
        y = self.norm(x)
        y = self.gate(y)
        if self.force_fp32 and in_dtype != torch.float32:
            y = y.to(dtype=in_dtype)
        return y


class PreAffineGatedRMSNorm2d(nn.Module):
    """
    PreAffine + RMSNorm2d + GatedNorm2d (full chain)
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        rank: int = 16,
        dropout: float = 0.0,
        bias: bool = False,
        force_fp32: bool = False,
    ):
        super().__init__()
        self.pre = nn.Parameter(torch.ones(dim))
        self.norm = create_norm_layer("rmsnorm2dfp32" if force_fp32 else "rmsnorm2d", dim)
        self.gate = _GatedNorm2d(dim=dim, rank=rank, dropout=dropout)
        self.force_fp32 = force_fp32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        if self.force_fp32 and in_dtype != torch.float32:
            x = x.float()
        x = x * self.pre.view(1, -1, 1, 1)
        y = self.norm(x)
        y = self.gate(y)
        if self.force_fp32 and in_dtype != torch.float32:
            y = y.to(dtype=in_dtype)
        return y


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
    create_act._ACT_LAYER_DEFAULT.update(new_acts)  # type: ignore
    create_act._ACT_LAYER_ME.update(new_acts)  # type: ignore
    logger.debug(f"[Timm registered new acts]: 'swiglu', 'poly_norm'")

    try:
        from fla.modules.activations import fast_gelu_impl, swiglu, swish

        get_act_layer = lambda act, name: partial(ActLayerMeta, name=name, act_layer=act)

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
    except ImportError as e:
        logger.warning(f"Flash Attention not available, skipping Flash RMSNorm: {e}.")

    try:
        import fla.modules

        class FlaRMSNorm(fla.modules.RMSNorm):
            def __init__(
                self,
                hidden_size: int,
                elementwise_affine: bool = True,
                bias: bool = False,
                eps: float = 1e-5,
                **kwargs,
            ):
                super().__init__(
                    hidden_size,
                    elementwise_affine=elementwise_affine,
                    bias=bias,
                    eps=eps,
                )

        class FlaLayerNorm(fla.modules.LayerNorm):
            def __init__(
                self,
                hidden_size: int,
                elementwise_affine: bool = True,
                bias: bool = False,
                eps: float = 1e-5,
                **kwargs,
            ):
                super().__init__(
                    hidden_size,
                    elementwise_affine=elementwise_affine,
                    bias=bias,
                    eps=eps,
                )

        create_norm._NORM_MAP.update(  # type: ignore
            {"flarmsnorm": FlaRMSNorm, "flalayernorm": FlaLayerNorm}
        )
        logger.debug(f'[Timm regsitered new norms]: "flarmsnorm", "flalayernorm"')
    except ImportError:
        logger.debug("FLA not available, skipping FLA norms registration")

    create_norm._NORM_MAP["zeromeanrmsnorm"] = Qwen3NextRMSNorm  # type: ignore
    create_norm._NORM_MAP["tritonrmsnorm2d"] = TritonRMSNorm2d  # type: ignore
    create_norm._NORM_MAP["adaptivegroupnorm"] = AdaptiveGroupNorm  # type: ignore
    create_norm._NORM_MAP["gatedrmsnorm"] = GatedRMSNorm  # type: ignore
    create_norm._NORM_MAP["preaffinegatedrmsnorm"] = PreAffineGatedRMSNorm  # type: ignore
    create_norm._NORM_MAP["gatedrmsnorm2d"] = GatedRMSNorm2d  # type: ignore
    create_norm._NORM_MAP["preaffinegatedrmsnorm2d"] = PreAffineGatedRMSNorm2d  # type: ignore
    logger.debug(f"[Timm registered new norms]: 'zeromeanrmsnorm', 'flashrmsnorm', 'tritonrmsnorm2d'")


_register_new_acts()
_register_new_norms()


if __name__ == "__main__":
    """
        python -m src.stage1.cosmos.modules.norm
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

    # layer = create_act.create_act_layer("fla_silu")
    # x = torch.randn(1, 256, 256).cuda()
    # y = layer(x)

    # y2 = nn.SiLU()(x)

    # print(torch.isclose(y, y2))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    x1d = torch.randn(2, 64, 128, device=device, dtype=dtype)
    x2d = torch.randn(2, 128, 16, 16, device=device, dtype=dtype)

    norm_1 = create_norm.create_norm_layer("gatedrmsnorm", 128, rank=16, force_fp32=True).to(device)
    norm_3 = create_norm.create_norm_layer("gatedrmsnorm2d", 128, rank=16, force_fp32=True).to(device)
    norm_4 = create_norm.create_norm_layer("preaffinegatedrmsnorm2d", 128, rank=16, force_fp32=True).to(device)

    y1 = norm_1(x1d)
    y2 = norm_2(x1d)
    y3 = norm_3(x2d)
    y4 = norm_4(x2d)

    print("gatedrmsnorm:", y1.shape, y1.dtype)
    print("preaffinegatedrmsnorm:", y2.shape, y2.dtype)
    print("gatedrmsnorm2d:", y3.shape, y3.dtype)
    print("preaffinegatedrmsnorm2d:", y4.shape, y4.dtype)
