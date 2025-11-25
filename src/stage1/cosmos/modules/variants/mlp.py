import functools
import os
import warnings
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from loguru import logger
from timm.layers.helpers import to_2tuple

from ..norm import SwiGLUAct

# fmt: off
XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import SwiGLU as SwiGLUXformers
        from xformers.ops import unbind
        from xformers.ops.swiglu_op import swiglu, swiglu_packed
        XFORMERS_AVAILABLE = True
    else:
        warnings.warn("xFormers is disabled (SwiGLU)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False

FLA_ENABLED = os.environ.get("FLA_DISABLED") is None
try:
    if FLA_ENABLED:
        from fla.modules.fused_norm_gate import (
            FusedLayerNormGatedLinear,
            FusedLayerNormSwishGateLinear,  # lin(ln(x) * silu(g))
            FusedRMSNormGatedLinear,
            FusedRMSNormSwishGateLinear,
        )
        from fla.modules.mlp import swiglu_linear
        FLA_AVAILABLE = True
    else:
        warnings.warn('FLA is disabled (fused FFNs).')
except ImportError:
    FLA_AVAILABLE = False
logger.debug(f"[Mlp Fused Kernels]: XFORMERS_AVAILABLE: {XFORMERS_AVAILABLE}, FLA_AVAILABLE: {FLA_AVAILABLE}")
# fmt: on


class SwiGLU(nn.Module):
    """SwiGLU
    NOTE: GluMLP above can implement SwiGLU, but this impl has split fc1 and
    better matches some other common impl which makes mapping checkpoints simpler.
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer: type[nn.Module] = nn.SiLU,
        norm_layer: type[nn.Module] | None = None,
        bias=True,
        drop=0.0,
        use_conv=False,
        is_fused: str | None = "xformers",
        packed_weights=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)  # type: ignore
        drop_probs: tuple[float, float] = to_2tuple(drop)  # type: ignore
        self.use_conv = use_conv
        self.is_fused = is_fused is not None
        self.fused_type = is_fused
        if self.is_fused:
            assert XFORMERS_AVAILABLE or FLA_AVAILABLE, f"Xformers and FLA is not installed, can not use fused FFN."
            assert norm_layer is None, "Fused FFN does not support norm layer, please set norm_layer to None."

        # is fused need to flatten and permute
        linear_layer = functools.partial(nn.Conv2d, kernel_size=1) if (use_conv and not is_fused) else nn.Linear
        # split g, x. not packed
        self.packed_weights = packed_weights
        if packed_weights:
            self.w12 = linear_layer(in_features, hidden_features * 2, bias=bias[0])
        else:
            self.w12 = None
            self.w1 = linear_layer(in_features, hidden_features, bias=bias[0])
            self.w2 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.w3 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

        self._xformers_swiglu_op = None  # allow for xformers.ops to dispatcher the op
        self.init_weights()

    def init_weights(self):
        # override init of fc1 w/ gate portion set to weight near zero, bias=1
        if self.packed_weights:
            assert self.w12 is not None
            c = self.w12.weight.shape[0] // 2  # hidden size
            if self.w12.bias is not None:
                nn.init.ones_(self.w12.bias[:c])
            nn.init.normal_(self.w12.weight, std=1e-6)
        else:
            if self.w1.bias is not None:
                nn.init.ones_(self.w1.bias)
            nn.init.normal_(self.w1.weight, std=1e-6)

    def _get_x12(self, x):
        if self.packed_weights:
            x1, x2 = self.w12(x).chunk(2, dim=1 if self.use_conv else -1)
        else:
            x1 = self.w1(x)
            x2 = self.w2(x)
        return x1, x2

    def math_forward(self, x):
        x1, x2 = self._get_x12(x)
        x = self.act(x1) * x2  # silu(x1) * x2

        x = self.drop1(x)
        x = self.norm(x)
        x = self.w3(x)
        x = self.drop2(x)
        return x

    def _to_1d(self, x):
        back_shape = None
        if x.ndim == 4:  # is an image
            h, w = x.shape[-2:]
            x = x.flatten(2).transpose(1, 2)  # bs, l, c
            back_shape = {"pattern": "b (h w) c -> b c h w", "h": h, "w": w}

        return x, back_shape

    def xformers_fused_forward(self, x):
        x, back_shape = self._to_1d(x)
        assert x.ndim == 3

        w3, b3 = self.w3.weight, self.w3.bias
        if self.packed_weights:
            w12 = self.w12.weight
            b12 = self.w12.bias
            if self._xformers_swiglu_op is not None:
                w12 = w12.view([2, w12.shape[0] // 2, w12.shape[1]])
                if b12 is not None:
                    b12 = b12.view([2, b12.shape[0] // 2])
                return swiglu_packed(x, w12, b12, w3, b3, op=self._xformers_swiglu_op)
            else:
                # splits the weights
                w1, w2 = unbind(w12.view([2, w12.shape[0] // 2, w12.shape[1]]), dim=0)
                if b12 is not None:
                    b1, b2 = unbind(b12.view([2, b12.shape[0] // 2]), dim=0)
                else:
                    b1, b2 = None, None
        else:
            w1, w2 = self.w1.weight, self.w2.weight
            b1, b2 = self.w1.bias, self.w2.bias

        x = swiglu(x, w1, b1, w2, b2, w3, b3)
        x = self.drop2(x)

        if back_shape is not None:
            x = rearrange(x, **back_shape)
        return x

    def fla_fused_forward(self, x):
        x, back_shape = self._to_1d(x)
        x1, x2 = self._get_x12(x)
        x = swiglu_linear(x1, x2, self.w3.weight, self.w3.bias)
        x = self.drop2(x)
        if back_shape is not None:
            x = rearrange(x, **back_shape)
        return x

    def forward(self, x, **kwargs):
        if self.fused_type is None:
            # logger.debug("use eager version")
            return self.math_forward(x)
        elif self.fused_type == "xformers":
            # logger.debug("use xformers version")
            return self.xformers_fused_forward(x)
        else:
            # logger.debug("use fla version")
            return self.fla_fused_forward(x)


class ClipSwiGLUMlp(SwiGLU):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer: type[nn.Module] = nn.SiLU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        mlp_bias=True,
        use_conv=False,
    ):
        super().__init__(
            in_features,
            hidden_features,
            out_features,
            act_layer,
            norm_layer,
            bias,
            drop,
            use_conv,
            is_fused=None,
            packed_weights=False,
        )
        self.mlp_bias = nn.Parameter(torch.zeros(self.w3.weight.shape[0])) if mlp_bias else None

    def forward(self, x, **kwargs):
        x_gate = self.w1(x)
        x = self.w2(x)
        if isinstance(self.act, SwiGLUAct):
            x = self.act(x_gate, x)
        else:
            x = self.act(x_gate) * x
        x = self.drop1(x)
        x = self.norm(x)
        x = self.w3(x)
        if self.mlp_bias is not None:
            if not self.use_conv:
                bias = self.mlp_bias
            else:
                bias = self.mlp_bias[..., None, None]
            x += bias
        x = self.drop2(x)
        return x


def test_fla_fused_ln_lin_kernel():
    x1 = torch.randn(1, 512, 32).cuda()
    x2 = torch.randn(1, 512, 32).cuda()
    layernorm = nn.LayerNorm(32, bias=False).cuda()
    lin = nn.Linear(32, 32).cuda()

    # x = norm(x1) * silu(x2) or norm(x1 * silu(x2))
    # x = lin(x)

    lgl = FusedLayerNormSwishGateLinear(32, True).cuda()
    x_lgl_prenorm = lgl(x1, x2, lin.weight, lin.bias, prenorm=False)
    x_lgl_postnorm = lgl(x1, x2, lin.weight, lin.bias, prenorm=True)

    # math version: prenorm
    layernorm.weight.data.copy_(lgl.weight.data)
    xx = layernorm(x1) * F.silu(x2)
    xx = lin(xx)
    print(xx.shape)
    print(x_lgl_prenorm - xx)

    # mathversion: postnorm
    xx = layernorm(x1 * F.silu(x2))
    xx = lin(xx)
    print(xx.shape)
    print(x_lgl_postnorm[0] - xx)


if __name__ == "__main__":
    """
    python -m src.stage2.layers.mlp
    """
    import time

    mlp = SwiGLU(256, 1024, 256, is_fused=None, use_conv=True).cuda()
    x = torch.randn(1, 256, 32, 32).cuda()

    xformers_out = mlp(x)
    print(xformers_out.shape)
    # print(f"Xformers costs time {time.time() - t1}")

    # mlp.fused_type = "fla"
    # fla_out = mlp(x)

    # mlp.fused_type = None
    # mlp.is_fused = False
    # math_out = mlp(x)

    # print(xformers_out - math_out)
    # print(fla_out - math_out)
