import math
from functools import partial, wraps
from inspect import isclass
from typing import Any, Callable, Literal, Optional, Tuple, TypeVar, Union

import numpy as np
import torch
import torch._dynamo
import torch._functorch.config
import torch.nn as nn
import torch.nn.functional as F
from accelerate.state import PartialState
from einops import rearrange
from natten import na2d
from timm.layers import DropPath
from torch.utils.checkpoint import checkpoint
from transformers.activations import ACT2FN

from src.stage1.MoEs.deepseek_moe.moe_layer import DeepseekECMoE
from src.stage1.MoEs.deepseek_moe.moe_layer import DeepseekV2MoE as DeepSeekTCMoE
from src.utilities.logging import log_print

from .utils import (
    Normalize,
    extract_needed_kwargs,
    gelu_nonlinear,
    nonlinearity,
    unit_magnitude_normalize,
    val2tuple,
)

compile_forward_fn = True
# options
compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "default"
compile_full_graph = True
epilogue_fusion = True
shape_padding = True
if compile_forward_fn:
    _compile_decorator = torch.compile(
        mode=compile_mode,
        fullgraph=compile_full_graph,
        # options={
        #     "max_autotune": False,
        #     "triton.cudagraphs": True,
        #     "shape_padding": shape_padding,
        #     "epilogue_fusion": epilogue_fusion,
        # },
    )
    log_print("will compile the forward function and disable donated buffer", "debug")
    torch._functorch.config.donated_buffer = False  # for adaptive weighting
    torch._dynamo.config.cache_size_limit = 20
else:

    def _null_decorator(**any_kwargs):
        def _inner_decorator(func):
            return func

        return _inner_decorator

    def _null_decorator_no_any_kwgs(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    _compile_decorator = _null_decorator_no_any_kwgs
    log_print("not compile the forward function", "debug")


# utilities


def get_same_padding(
    kernel_size: Union[int, tuple[int, ...]],
):
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2


# * --- Activation Functions --- #

# register activation function here
REGISTERED_ACT_DICT: dict[str, type | partial] = {
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "hswish": nn.Hardswish,
    "silu": nn.SiLU,
    "gelu": partial(nn.GELU, approximate="tanh"),
}


def build_act(name: str, **kwargs) -> Optional[nn.Module]:
    if name in REGISTERED_ACT_DICT:
        act_cls = REGISTERED_ACT_DICT[name]
        kwargs = extract_needed_kwargs(kwargs, act_cls)
        return act_cls(**kwargs)
    else:
        return None


# * --- Convolutional Layers --- #


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        use_bias: bool = False,
        dropout: float = 0,
        norm: str = "bn2d",
        act_func: str = "relu",
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()

        padding = get_same_padding(kernel_size)
        padding *= dilation

        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=use_bias,
            padding_mode=padding_mode,
        )
        self.norm = Normalize(in_channels=out_channels, norm_type=norm)
        self.act = build_act(act_func)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


# * --- GLUMBConv --- #


class MPConv(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )

    def forward(self, x, gain=1):
        w = self.weight.to(torch.float32)
        if self.training:
            with torch.no_grad():
                self.weight.copy_(
                    unit_magnitude_normalize(w)
                )  # forced weight normalization
        w = unit_magnitude_normalize(w)  # traditional weight normalization
        w = w * (gain / np.sqrt(w[0].numel()))  # magnitude-preserving scaling
        w = w.to(x.dtype)

        # if w.ndim == 2:
        #     return x @ w.t()
        # assert w.ndim == 4

        return self._conv_forward(x, w, self.bias)


# * --- Residaul block builder --- #


class ResidualBlock(nn.Module):
    def __init__(
        self,
        main: Optional[nn.Module],
        shortcut: Optional[nn.Module],
        post_act=None,
        pre_norm: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.pre_norm = pre_norm
        self.main = main
        self.shortcut = shortcut
        self.post_act = build_act(post_act)

    @_compile_decorator
    def forward_main(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm is None:
            return self.main(x)
        else:
            return self.main(self.pre_norm(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.main is None:
            res = x
        elif self.shortcut is None:
            res = self.forward_main(x)
        else:
            res = self.forward_main(x) + self.shortcut(x)
            if self.post_act:
                res = self.post_act(res)
        return res


# TODO: this version causes the norm is high
class GLUMBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        use_bias=False,
        norm=(None, None, "ln2d"),
        act_func=("silu", "silu", None),
    ):
        super().__init__()
        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act_func = val2tuple(act_func, 3)

        mid_channels = (
            round(in_channels * expand_ratio) if mid_channels is None else mid_channels
        )

        self.glu_act = build_act(act_func[1], inplace=False)
        self.inverted_conv = ConvLayer(
            in_channels,
            mid_channels * 2,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.depth_conv = ConvLayer(
            mid_channels * 2,
            mid_channels * 2,
            kernel_size,
            stride=stride,
            groups=mid_channels * 2,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=None,
        )
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            use_bias=use_bias[2],
            norm=norm[2],
            act_func=act_func[2],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inverted_conv(x)
        x = self.depth_conv(x)

        x, gate = torch.chunk(x, 2, dim=1)
        gate = self.glu_act(gate)
        x = x * gate

        x = self.point_conv(x)
        return x


# * --- Attentions --- #


class AttnBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        act_checkpoint: bool,
        use_residual_factor: bool = False,
        norm_type: str = "gn",
        norm_groups: int = 32,
        sdpa: bool = False,
    ):
        super().__init__()
        self.sdpa = sdpa
        log_print(f"[Attention Block]: use sdpa: {self.sdpa}", "debug")

        self.norm = Normalize(in_channels, norm_type=norm_type, num_groups=norm_groups)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

        self.act_checkpoint = act_checkpoint
        self.use_residual_factor = use_residual_factor
        if self.use_residual_factor:
            self.residual_factor = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

    def forward_fn_math(self, x: torch.Tensor) -> torch.Tensor:
        # TODO (freda): Consider reusing implementations in Attn `imaginaire`,
        # since than one is gonna be based on TransformerEngine's attn op,
        # w/c could ease CP implementations.
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # [b, l, c]
        k = k.reshape(b, c, h * w)
        w_ = torch.bmm(q, k)  # [b, l, l]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)
        if self.use_residual_factor:
            h_ = h_ * self.residual_factor

        return x + h_

    def forward_fn_sdpa(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b (h w) c")
        k = rearrange(k, "b c h w -> b (h w) c")
        v = rearrange(v, "b c h w -> b (h w) c")

        # sdpa
        h_ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        h_ = rearrange(h_, "b (h w) c -> b c h w", h=h, w=w)
        h_ = self.proj_out(h_)
        if self.use_residual_factor:
            h_ = h_ * self.residual_factor
        return x + h_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fn = self.forward_fn_math if not self.sdpa else self.forward_fn_sdpa
        if self.act_checkpoint:
            return checkpoint(fn, x, use_reentrant=True)
        return fn(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", heads=self.heads, qkv=3
        )
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(
            out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w
        )
        return self.to_out(out)


class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""

    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)


class LiteMLA(nn.Module):
    r"""Lightweight multi-scale linear attention"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: Optional[int] = None,
        heads_ratio: float = 1.0,
        dim=8,
        use_bias=False,
        norm=(None, "bn2d"),
        act_func=(None, None),
        kernel_func="relu",
        scales: tuple[int, ...] = (5,),
        eps=1.0e-15,
    ):
        super().__init__()
        self.eps = eps
        heads = int(in_channels // dim * heads_ratio) if heads is None else heads

        total_dim = heads * dim

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.dim = dim
        self.qkv = ConvLayer(
            in_channels,
            3 * total_dim,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        3 * total_dim,
                        3 * total_dim,
                        scale,
                        padding=get_same_padding(scale),
                        groups=3 * total_dim,
                        bias=use_bias[0],
                    ),
                    nn.Conv2d(
                        3 * total_dim,
                        3 * total_dim,
                        1,
                        groups=3 * heads,
                        bias=use_bias[0],
                    ),
                )
                for scale in scales
            ]
        )
        self.kernel_func = build_act(kernel_func, inplace=False)

        self.proj = ConvLayer(
            total_dim * (1 + len(scales)),
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    @torch.autocast(device_type="cuda", enabled=False)
    def relu_linear_att(self, qkv: torch.Tensor) -> torch.Tensor:
        B, _, H, W = list(qkv.size())

        if qkv.dtype == torch.float16:
            qkv = qkv.float()

        qkv = torch.reshape(qkv, (B, -1, 3 * self.dim, H * W))
        q, k, v = (
            qkv[:, :, 0 : self.dim],
            qkv[:, :, self.dim : 2 * self.dim],
            qkv[:, :, 2 * self.dim :],
        )

        # lightweight linear attention
        q = self.kernel_func(q)
        k = self.kernel_func(k)

        # linear matmul
        trans_k = k.transpose(-1, -2)

        v = F.pad(v, (0, 0, 0, 1), mode="constant", value=1)
        vk = torch.matmul(v, trans_k)
        out = torch.matmul(vk, q)
        if out.dtype == torch.bfloat16:
            out = out.float()
        out = out[:, :, :-1] / (out[:, :, -1:] + self.eps)

        out = torch.reshape(out, (B, -1, H, W))
        return out

    @torch.autocast(device_type="cuda", enabled=False)
    def relu_quadratic_att(self, qkv: torch.Tensor) -> torch.Tensor:
        B, _, H, W = list(qkv.size())

        qkv = torch.reshape(qkv, (B, -1, 3 * self.dim, H * W))
        q, k, v = (
            qkv[:, :, 0 : self.dim],
            qkv[:, :, self.dim : 2 * self.dim],
            qkv[:, :, 2 * self.dim :],
        )

        q = self.kernel_func(q)
        k = self.kernel_func(k)

        att_map = torch.matmul(k.transpose(-1, -2), q)  # b h n n
        original_dtype = att_map.dtype
        if original_dtype in [torch.float16, torch.bfloat16]:
            att_map = att_map.float()
        att_map = att_map / (
            torch.sum(att_map, dim=2, keepdim=True) + self.eps
        )  # b h n n
        att_map = att_map.to(original_dtype)
        out = torch.matmul(v, att_map)  # b h d n

        out = torch.reshape(out, (B, -1, H, W))
        return out

    @_compile_decorator
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # generate multi-scale q, k, v
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
        qkv = torch.cat(multi_scale_qkv, dim=1)

        H, W = list(qkv.size())[-2:]
        if H * W > self.dim:
            out = self.relu_linear_att(qkv).to(qkv.dtype)
        else:
            out = self.relu_quadratic_att(qkv)
        out = self.proj(out)

        return out


class NattenAttention(nn.Module):
    def __init__(
        self,
        in_channels: int,
        act_checkpoint: bool = False,
        norm_type: str = "gn",
        norm_groups: int = 32,
        heads: int = 8,
        ksize: int = 8,
        stride: int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        self.norm = Normalize(in_channels, norm_type=norm_type, num_groups=norm_groups)
        self.qkv = nn.Conv2d(
            in_channels, in_channels * 3, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

        self.heads = heads
        self.ksize = ksize
        self.stride = stride
        self.dilation = dilation

        self.act_checkpoint = act_checkpoint

    def forward(self, x):
        qkv = self.qkv(self.norm(x))  # [bs, c, h, w]
        qkv = rearrange(qkv, "b (qkv h c) x y -> qkv b x y h c", qkv=3, h=self.heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        out = na2d(  # [bs, x, y, h, c]
            q,
            k,
            v,
            kernel_size=self.ksize,
            stride=self.stride,
            dilation=self.dilation,
            is_causal=False,
        )
        out = rearrange(out, "b x y h c -> b (h c) x y")  # [bs, h*c, x, y]
        out = self.proj_out(out)

        return out


def make_attn(in_channels, attn_type="vanilla", act_checkpoint=False):
    log_print(f"making attention of type '{attn_type}' with {in_channels=}", "debug")

    if attn_type.startswith("attn"):
        sdpa = attn_type.endswith("sdpa")
        return AttnBlock(in_channels, act_checkpoint=act_checkpoint, sdpa=sdpa)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    elif attn_type == "lite_mla":
        return LiteMLA(
            in_channels,
            in_channels,
            dim=32,
            act_func="silu",
            norm=(None, "rms_triton"),
            heads_ratio=1.0,
            scales=(5,),  # 5 scale, or () for no scale
        )
    elif attn_type == "linear":
        return LinAttnBlock(in_channels)
    else:
        raise ValueError(
            f"attn_type {attn_type} is not supported, "
            "supported types are: ['vanilla', 'linear', 'lite_mla', 'none']"
        )


# * --- Upsample and Downsample --- #


def resample_norm_keep(x, x_resampled):
    return x_resampled * torch.norm(x) / torch.norm(x_resampled)


class UpsampleRepeatConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        padding_mode: str = "zeros",
        norm_type: str | None = None,
        norm_keep: bool = False,
    ):
        super().__init__()
        self.norm_keep = norm_keep
        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode=padding_mode,
        )
        # self.norm = Normalize(
        #     in_channels=in_channels, num_groups=32, norm_type=norm_type
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
        x_resp = self.conv(x)
        if self.norm_keep:
            x_resp = resample_norm_keep(x, x_resp)
        return x_resp


class DownsamplePadConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        padding_mode: str = "constant",
        padding_in_conv: bool = False,
        norm_type: Optional[str] = None,
        norm_keep: bool = False,
    ):
        # Zihan NOTE: using pad (left and right) align the center of the pixel when downsampling
        # but (may?) cause the boundary artifact when upsampling

        super().__init__()
        self.padding_mode = padding_mode
        self.padding_in_conv = padding_in_conv
        self.norm_keep = norm_keep

        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=2,
            padding=0 if not padding_in_conv else 1,
            padding_mode=padding_mode
            if padding_in_conv
            else "zeros",  # 'zeros' as default
        )

        # self.norm = Normalize(
        #     in_channels=in_channels, num_groups=32, norm_type=norm_type
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LDM VAE also use the unsymmetric padding
        if not self.padding_in_conv:  # cosmos manually pad
            # to align on the center of the downsampled image pixels
            pad = (0, 1, 0, 1)  # lower and righter pads, why? inductive bias?
            if self.padding_mode not in ("constant", "zeros"):
                x = F.pad(x, pad, mode=self.padding_mode)
            else:
                x = F.pad(x, pad, mode="constant", value=0)

        x_resp = self.conv(x)
        if self.norm_keep:
            x_resp = resample_norm_keep(x, x_resp)
        return x_resp


class ConvPixelUnshuffleDownSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        factor: int,
        padding_mode: str = "zeros",
        norm_keep: bool = False,
    ):
        super().__init__()
        self.factor = factor
        self.norm_keep = norm_keep
        out_ratio = factor**2
        assert out_channels % out_ratio == 0
        self.conv = nn.Conv2d(
            in_channels,
            out_channels // out_ratio,
            kernel_size,
            padding=get_same_padding(kernel_size),
            padding_mode=padding_mode,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = F.pixel_unshuffle(x, self.factor)
        if self.norm_keep:
            x = resample_norm_keep(x, x)
        return x


class PixelUnshuffleChannelAveragingDownSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
        # group_size: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        assert in_channels * factor**2 % out_channels == 0
        self.group_size = in_channels * factor**2 // out_channels
        # hidden = out_channels * group_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pixel_unshuffle(
            x, self.factor
        )  # c * factor ** 2 -> hidden = out_c * group_size
        B, C, H, W = x.shape
        x = x.view(B, self.out_channels, self.group_size, H, W)
        x = x.mean(dim=2)
        return x


class ConvPixelShuffleUpSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        factor: int,
        padding_mode: str = "zeros",
        norm_type: Optional[str] = None,
    ):
        super().__init__()
        self.factor = factor
        out_ratio = factor**2
        self.norm = Normalize(
            in_channels=in_channels, num_groups=32, norm_type=norm_type
        )
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * out_ratio,
            kernel_size,
            padding=get_same_padding(kernel_size),
            padding_mode=padding_mode,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(self.norm(x))
        x = F.pixel_shuffle(x, self.factor)
        return x


class InterpolateConvUpSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        factor: int,
        mode: str = "nearest",
        padding_mode: str = "zeros",
        norm_type: Optional[str] = None,
        norm_keep: bool = False,
    ) -> None:
        super().__init__()
        self.factor = factor
        self.mode = mode
        self.norm_keep = norm_keep
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            1,
            padding=get_same_padding(kernel_size),
            padding_mode=padding_mode,
        )
        # self.norm = Normalize(
        #     in_channels=in_channels, num_groups=32, norm_type=self.norm_type
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(x, scale_factor=self.factor, mode=self.mode)
        x_resp = self.conv(x)
        if self.norm_keep:
            x_resp = resample_norm_keep(x, x_resp)
        return x_resp


class ChannelDuplicatingPixelUnshuffleUpSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        assert out_channels * factor**2 % in_channels == 0
        self.repeats = out_channels * factor**2 // in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        x = F.pixel_shuffle(x, self.factor)
        return x


# * --- Upsample and downsample entries --- #


def build_upsample_block(
    block_type: str,
    in_channels: int,
    out_channels: int,
    shortcut: Optional[str],
    padding_mode: str = "zeros",
    norm_type: str | None = None,  # deprecated
    norm_keep: bool = False,
) -> nn.Module:
    log_print(
        f"[build_upsample_block] block_type: {block_type}, "
        f"in_channels: {in_channels}, "
        f"out_channels: {out_channels}, "
        f"shortcut: {shortcut}, "
        f"padding_mode: {padding_mode}, "
        f"norm keep: {norm_keep}",
        "debug",
    )

    if block_type == "ConvPixelShuffle":
        block = ConvPixelShuffleUpSampleLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            factor=2,
            padding_mode=padding_mode,
        )
    elif block_type == "RepeatConv":
        block = UpsampleRepeatConv(
            in_channels, padding_mode=padding_mode, norm_keep=norm_keep
        )
    elif block_type == "InterpolateConv":
        block = InterpolateConvUpSampleLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            factor=2,
            padding_mode=padding_mode,
            mode="nearest",
            norm_keep=norm_keep,
        )
    else:
        raise ValueError(f"block_type {block_type} is not supported for upsampling")

    if shortcut is None:
        pass
    elif shortcut == "duplicating":
        shortcut_block = ChannelDuplicatingPixelUnshuffleUpSampleLayer(
            in_channels=in_channels, out_channels=out_channels, factor=2
        )
        block = ResidualBlock(block, shortcut_block)
    else:
        raise ValueError(f"shortcut {shortcut} is not supported for upsample")
    return block


def build_downsample_block(
    block_type: str,
    in_channels: int,
    out_channels: int,
    shortcut: Optional[str],
    padding_mode: str = "zeros",
    *,
    padconv_use_manually_pad: bool = True,  # for the compatibility with cosmos checkpoints
    norm_type: str | None = None,
    norm_keep: bool = False,
) -> nn.Module:
    log_print(
        f"[build_downsample_block] block_type: {block_type}, "
        f"in_channels: {in_channels}, "
        f"out_channels: {out_channels}, "
        f"shortcut: {shortcut}, "
        f"padding_mode: {padding_mode} "
        f"padconv_use_manually_pad: {padconv_use_manually_pad}, "
        f"norm keep: {norm_keep}, ",
        "debug",
    )

    if block_type == "Conv":
        block = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            padding_mode=padding_mode,
            bias=True,
        )
    elif block_type == "PadConv":
        block = DownsamplePadConv(
            in_channels=in_channels,
            padding_in_conv=not padconv_use_manually_pad,
            padding_mode=padding_mode,
            norm_keep=norm_keep,
        )
    elif block_type == "ConvPixelUnshuffle":
        block = ConvPixelUnshuffleDownSampleLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            factor=2,
            padding_mode=padding_mode,
        )
    else:
        raise ValueError(f"block_type {block_type} is not supported for downsampling")

    if shortcut is None:
        pass
    elif shortcut == "averaging":
        shortcut_block = PixelUnshuffleChannelAveragingDownSampleLayer(
            in_channels=in_channels, out_channels=out_channels, factor=2
        )
        block = ResidualBlock(block, shortcut_block)
    else:
        raise ValueError(f"shortcut {shortcut} is not supported for downsample")
    return block


# * --- ffns --- #


class GLUFeedForward(nn.Module):
    def __init__(
        self,
        hidden_size,
        intermediate_size,
        hidden_act="gelu_pytorch_tanh",  # gelu_fast, gelu
        conv_cls=nn.Conv2d,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = conv_cls(self.hidden_size, self.intermediate_size, 1, 1, 0)
        self.up_proj = conv_cls(self.hidden_size, self.intermediate_size, 1, 1, 0)
        self.down_proj = conv_cls(self.intermediate_size, self.hidden_size, 1, 1, 0)
        self.act_fn = (
            ACT2FN[hidden_act] if isinstance(hidden_act, str) else hidden_act()
        )

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class LlamaFFN(nn.Module):
    def __init__(self, hidden_size, intermediate_size, conv_cls=nn.Conv2d):
        super().__init__()
        self.w1 = conv_cls(hidden_size, intermediate_size, 1, 1, 0)
        self.w2 = conv_cls(intermediate_size, hidden_size, 1, 1, 0)
        self.w3 = conv_cls(hidden_size, intermediate_size, 1, 1, 0)

        self.weight_init()

    def weight_init(self):
        torch.nn.init.trunc_normal_(self.w1.weight, std=0.02)
        torch.nn.init.trunc_normal_(self.w2.weight, std=0.02)
        torch.nn.init.trunc_normal_(self.w3.weight, std=0.02)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x) * self.w3(x)))


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=partial(nn.GELU, approximate="tanh"),
        norm_layer=None,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = (bias, bias)
        drop_probs = (drop, drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1)

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = (
            act_layer()
            if isclass(act_layer) or isinstance(act_layer, partial)
            else act_layer
        )
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# * --- Input and output convs with different bands images --- #


class DiffBandsInputConvIn(nn.Module):
    def __init__(
        self,
        band_lst: list[int],
        hidden_dim: int,
        basic_module: Literal["conv", "resnet", "inv_bottleneck", "moe"] = "conv",
        padding_mode: str = "zeros",
        check_grads: bool = True,
    ):
        super().__init__()

        self.band_lst = band_lst
        self.hidden_dim = hidden_dim
        self.is_ddp = PartialState().use_distributed or check_grads

        self.in_modules = nn.ModuleDict()
        for c in band_lst:
            if basic_module == "conv":
                module = nn.Conv2d(
                    in_channels=c,
                    out_channels=hidden_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode=padding_mode,
                )
            elif basic_module == "mobile":
                module = nn.Sequential(
                    nn.Conv2d(c, hidden_dim, 1, 1, 0),
                    nn.Conv2d(
                        hidden_dim,
                        hidden_dim,
                        3,
                        1,
                        1,
                        padding_mode=padding_mode,
                        groups=hidden_dim,
                    ),
                )
            elif basic_module == "inv_bottleneck":
                module = ResidualBlock(
                    GLUMBConv(
                        in_channels=c,
                        out_channels=hidden_dim,
                        kernel_size=3,
                        stride=1,
                        expand_ratio=4,
                        use_bias=True,
                        norm=(None, None, "rms_triton"),
                        act_func=("silu", "silu", None),
                    ),
                    nn.Conv2d(c, hidden_dim, kernel_size=1, stride=1),
                )
            elif basic_module == "resnet":
                module = ResnetBlock(
                    in_channels=c,
                    out_channels=hidden_dim,
                    num_groups=1,
                    dropout=0.0,
                    act_checkpoint=False,
                    padding_mode=padding_mode,
                    norm_type="gn",
                )
            elif basic_module == "moe":
                module = nn.Sequential(
                    nn.Conv2d(
                        in_channels=c,
                        out_channels=hidden_dim,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        padding_mode="zeros",
                    ),
                    MoE2DBlock(
                        in_channels=hidden_dim,
                        hidden_channels=hidden_dim,
                        n_experts=4,
                        n_selected=1,
                        n_shared_experts=1,
                        moe_type="tc",
                        act_checkpoint=False,
                    ),
                    # to patcher dim
                    # nn.Conv2d(c, hidden_dim, kernel_size=1, stride=1, padding=0),
                )
            else:
                raise ValueError(
                    f"[DiffBandsInputConvIn] Unknown basic_module={basic_module}"
                )

            self.in_modules["conv_in_{}".format(c)] = module

            log_print(
                f"[DiffBandsInputConvIn] set conv to hidden module and buffer for channel {c}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c_ = x.shape[1]
        module = getattr(self.in_modules, "conv_in_{}".format(c_))
        if module is None:
            raise ValueError(
                f"[DiffBandsInputConvIn] no module for channel {c_}, please check the channel list"
            )
        h = module(x)

        if self.training and self.is_ddp:
            for c in self.band_lst:
                if c != c_:
                    m = self.in_modules["conv_in_{}".format(c)]
                    dummy_loss = sum(p.sum() * 0.0 for p in m.parameters())
                    h = h + dummy_loss

        return h


class DiffBandsInputConvOut(nn.Module):
    def __init__(
        self,
        band_lst: list[int],
        hidden_dim: int,
        basic_module: str = "conv",
        padding_mode: str = "zeros",
        check_grads: bool = True,
    ):
        super().__init__()

        self.band_lst = band_lst
        self.hidden_dim = hidden_dim
        self.basic_module = basic_module
        self.is_ddp = PartialState().use_distributed or check_grads

        self.in_modules = nn.ModuleDict()
        for c in band_lst:
            if basic_module == "conv":
                module = nn.Conv2d(
                    in_channels=hidden_dim,
                    out_channels=c,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode=padding_mode,
                )
            elif basic_module == "mobile":
                module = nn.Sequential(
                    nn.Conv2d(
                        hidden_dim,
                        hidden_dim,
                        3,
                        1,
                        1,
                        padding_mode=padding_mode,
                        groups=hidden_dim,
                    ),
                    nn.Conv2d(hidden_dim, c, 1, 1, 0),
                )
            elif basic_module == "inv_bottleneck":
                module = ResidualBlock(
                    GLUMBConv(
                        in_channels=hidden_dim,
                        out_channels=c,
                        kernel_size=3,
                        stride=1,
                        expand_ratio=4,
                        use_bias=True,
                        norm=(None, None, "rms_triton"),
                        act_func=("silu", "silu", None),
                    ),
                    nn.Conv2d(hidden_dim, c, kernel_size=1, stride=1, padding=0),
                )
            elif basic_module == "resnet":
                module = ResnetBlock(
                    in_channels=hidden_dim,
                    out_channels=c,
                    dropout=0.0,
                    act_checkpoint=False,
                    num_groups=1,
                    norm_type="gn",
                )
            elif basic_module == "moe":
                module = nn.Sequential(
                    nn.Conv2d(
                        in_channels=hidden_dim,
                        out_channels=hidden_dim,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        padding_mode=padding_mode,
                    ),
                    MoE2DBlock(
                        in_channels=hidden_dim,
                        hidden_channels=hidden_dim,
                        n_experts=4,
                        n_selected=1,
                        n_shared_experts=1,
                        moe_type="tc",
                        act_checkpoint=False,
                    ),
                    # to patcher dim
                    nn.Conv2d(hidden_dim, c, kernel_size=1, stride=1, padding=0),
                )
            else:
                raise ValueError(
                    f"[DiffBandsInputConvIn] Unknown basic_module={basic_module}"
                )

            self.in_modules["conv_out_{}".format(c)] = module

            log_print(
                f"[DiffBandsInputConvOut] set conv to hidden module for channel {c}"
            )

        self.out_channel = None

    def forward(self, x: torch.Tensor, out_channel: int) -> torch.Tensor:
        self.out_channel = out_channel

        module = getattr(self.in_modules, f"conv_out_{out_channel}", None)
        if module is None:
            raise ValueError(
                f"[DiffBandsInputConvOut] No module for out_channel={out_channel}. Available: {list(self.in_modules.keys())}",
            )
        h = module(x)

        if self.training and self.is_ddp:
            for c in self.band_lst:
                if c != out_channel:
                    m = self.in_modules["conv_out_{}".format(c)]
                    dummy_loss = sum(p.sum() * 0.0 for p in m.parameters())
                    h = h + dummy_loss

        return h

    @property
    def weight(self):
        # used to get the weight of the conv_out module for GAN loss
        assert self.out_channel is not None, (
            "out_channel is not set, please call forward first"
        )
        module = getattr(self.in_modules, f"conv_out_{self.out_channel}", None)
        if module is None:
            raise ValueError(
                f"[DiffBandsInputConvOut] No module for out_channel={self.out_channel}. Available: {list(self.in_modules.keys())}",
            )

        match self.basic_module:
            case "conv":
                return module.weight
            case "resnet":
                return module.conv2.weight
            case "moe":
                return module[-1].weight
            case "inv_bottleneck":
                return module.main.depth_conv.conv.weight
            case "mobile":
                return module[-1].weight
            case _:
                raise ValueError(
                    f"[DiffBandsInputConvOut] Unknown basic_module={self.basic_module}. Available: conv, resnet, moe, inv_bottleneck"
                )


# * --- MoEs 2D --- #


class MoE2DBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        n_experts: int = 4,
        n_selected: int = 1,
        n_shared_experts: int = 1,
        n_token_ec: int = 64 * 64,  # 2d image, 64x64
        moe_type: Literal["tc", "ec", "tc+ec"] = "tc",
        act_checkpoint: bool = False,
    ):
        super().__init__()
        self.moe_type = moe_type
        self.act_checkpoint = act_checkpoint

        moe_tc_fn = partial(
            DeepSeekTCMoE,
            num_experts_per_tok=n_selected,
            n_routed_experts=n_experts,
            hidden_size=in_channels,
            moe_intermediate_size=hidden_channels,
            n_shared_experts=n_shared_experts,
        )
        moe_ec_fn = partial(
            DeepseekECMoE,
            expert_capacity_per_batch=n_token_ec,
            n_routed_experts=n_experts,
            moe_intermediate_size=hidden_channels,
            hidden_size=in_channels,
            n_shared_experts=n_shared_experts,
        )

        if self.moe_type == "tc":  # token choice
            self.moe = nn.ModuleDict({"moe_tc": moe_tc_fn()})
        elif self.moe_type == "ec":  # expert choice
            assert n_token_ec is not None, "n_token_ec should be set for expert choice"
            self.moe = nn.ModuleDict({"moe_ed": moe_ec_fn()})
        elif self.moe_type == "tc+ec":  # token choice + expert choice
            assert n_token_ec is not None, "n_token_ec should be set for expert choice"
            self.moe = nn.ModuleDict({"moe_tc": moe_tc_fn(), "moe_ec": moe_ec_fn()})
        else:
            raise ValueError(f"[MoE2DBlockTC] Unknown moe_type={self.moe_type}")

    @_compile_decorator
    def _forward_fn(self, x: torch.Tensor):
        # x: (B, C, H, W)
        h, w = x.shape[-2:]
        x = rearrange(x, "b c h w -> b (h w) c")
        if self.moe_type == "tc":
            x = self.moe["moe_tc"](x)
        elif self.moe_type == "ec":
            x = self.moe["moe_ec"](x)
        else:
            x_ec = self.moe["moe_ec"](x)
            x_tc = self.moe["moe_tc"](x)
            x = x_ec + x_tc

        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        return x

    def forward(self, x):
        x = x.contiguous()
        if self.training and self.act_checkpoint:
            return checkpoint(self._forward_fn, x, use_reentrant=True)
        return self._forward_fn(x)


class ResnetBlockMoE2D(nn.Module):
    # variant for ResnetBlock
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        drop_out: float = 0.0,
        n_experts: int = 4,
        n_selected: int = 1,
        n_shared_experts: int = 1,
        n_token_ec: int = 64 * 64,  # 2d image, 64x64
        hidden_factor: int = 4,
        moe_type: Literal["tc", "ec", "tc+ec"] = "tc",
        act_checkpoint: bool = False,
        padding_mode: str = "zeros",
        norm_type: str = "gn",
        norm_groups: int = 32,
        token_mixer_type: Literal["res_block", "dico_block", "convnext"] = "res_block",
        **resnet_kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.token_mixer_type = token_mixer_type

        log_print(f"[ResnetBlockMoE2D] using token mixer: {token_mixer_type}", "debug")
        if token_mixer_type == "res_block":
            self.token_mixer = ResnetBlock(
                in_channels=in_channels,
                out_channels=self.out_channels,
                dropout=drop_out,
                act_checkpoint=act_checkpoint,
                use_residual_factor=False,
                padding_mode=padding_mode,
                norm_type=norm_type,
                **resnet_kwargs,
            )
        elif token_mixer_type == "dico_block":
            self.token_mixer = DiCoBlock(
                in_channels=in_channels,
                out_channels=self.out_channels,
                hidden_channels=int(hidden_factor * self.out_channels),
                dropout=drop_out,
                act_checkpoint=act_checkpoint,
                use_residual=False,
                padding_mode=padding_mode,
                norm_type=norm_type,
                use_ffn=False,  # moe serves as ffn
                norm_groups=resnet_kwargs.get("num_groups", 32),
            )
        elif token_mixer_type == "convnext":
            self.token_mixer = ConvNeXtBlock(
                dim=in_channels,
                hidden_dim=int(hidden_factor * self.out_channels),
                out_dim=self.out_channels,
                act_checkpoint=act_checkpoint,
                drop_path=drop_out,
                norm_type=norm_type,
            )
        else:
            raise ValueError(
                f"[ResnetBlockMoE2D] Unknown token_mixer_type={token_mixer_type}, "
                "supported types are: ['res_block', 'dico_block', 'convnext']"
            )
        self.moe_prenorm = Normalize(
            self.out_channels, norm_type=norm_type, num_groups=norm_groups
        )
        self.moe = MoE2DBlock(
            in_channels=self.out_channels,
            hidden_channels=int(hidden_factor * self.out_channels),
            act_checkpoint=act_checkpoint,
            n_experts=n_experts,
            n_selected=n_selected,
            n_shared_experts=n_shared_experts,
            n_token_ec=n_token_ec,
            moe_type=moe_type,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # token mixer
        h = self.token_mixer(x)  # shortcut is in the block
        # ffn
        h = self.moe(self.moe_prenorm(h)) + h
        return h


# * --- Dico Block --- #


class DiCoCompactChannelAttention(nn.Module):
    def __init__(
        self,
        in_channels: int,
        conv_cls: nn.Module = nn.Conv2d,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.global_avg = nn.AdaptiveAvgPool2d(1)
        self.body = conv_cls(in_channels, in_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.global_avg(x)
        h = self.body(h)
        h = self.sigmoid(h) * x
        return h


@_compile_decorator
def modulation(x, shift, scale):
    return x * (scale + 1) + shift


class DiCoBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int | None = None,
        dropout: float = 0.0,
        norm_type: str = "gn",
        norm_groups: int = 32,
        use_residual: bool = False,
        padding_mode: str = "zeros",
        act_checkpoint: bool = False,
        use_ffn: bool = True,
        conv_type: str = "conv",
        ffn_type: str = "glu",
    ):
        super().__init__()
        self.use_residual = use_residual
        self.act_checkpoint = act_checkpoint
        self.use_ffn = use_ffn
        if out_channels is None:
            out_channels = in_channels

        log_print(
            f"[Dico block]: in: {in_channels} "
            f"out: {out_channels} "
            f"hidden: {hidden_channels} "
            f"conv type: {conv_type} ",
            "debug",
        )

        self.norm = Normalize(in_channels, norm_type=norm_type, num_groups=norm_groups)
        # self.dropout = nn.Dropout(dropout)
        conv_cls = (
            MPConv if conv_type == "mpconv" else nn.Conv2d
        )  # mpconv cannot support the torch.compile

        self.body = nn.Sequential(
            # point conv
            conv_cls(in_channels, out_channels, kernel_size=1, stride=1),
            # depth conv
            conv_cls(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode=padding_mode,
                groups=out_channels,
            ),
            nn.GELU(approximate="tanh"),
            # nn.SiLU(),
        )
        self.body_out = conv_cls(out_channels, out_channels, kernel_size=1, stride=1)

        # cca
        self.cca = DiCoCompactChannelAttention(out_channels, conv_cls=conv_cls)

        # ffn
        if self.use_ffn:
            if ffn_type == "mlp":
                self.ffn = Mlp(
                    in_features=out_channels,
                    hidden_features=hidden_channels,
                    out_features=out_channels,
                    act_layer=partial(nn.GELU, approximate="tanh"),
                    norm_layer=None,
                    bias=True,
                )
            elif ffn_type == "glu":
                self.ffn = GLUFeedForward(
                    out_channels,
                    hidden_channels,
                    conv_cls=nn.Conv2d,
                    hidden_act="gelu_pytorch_tanh",
                )
            elif ffn_type == "llama":
                self.ffn = LlamaFFN(
                    hidden_size=out_channels,
                    intermediate_size=hidden_channels,
                    conv_cls=nn.Conv2d,
                )
            else:
                raise ValueError(f"Unknown ffn_type: {ffn_type}")
            self.norm_ffn = Normalize(
                out_channels, norm_type=norm_type, num_groups=norm_groups
            )

        self.nin_shortcut = (
            conv_cls(in_channels, out_channels, kernel_size=1, stride=1)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.weight_init()

    def weight_init(self):
        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1.0)
                if m.bias is not None:
                    m.bias.data.zero_()

    @_compile_decorator
    def forward_fn(self, x: torch.Tensor) -> torch.Tensor:
        h = x

        # token mixer
        h = self.body(self.norm(h))
        h = self.cca(h)
        h = self.body_out(h)
        h = h + self.nin_shortcut(x)

        # ffn
        if self.use_ffn:
            h = self.ffn(self.norm_ffn(h)) + h

        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_checkpoint and self.training:
            return checkpoint(self.forward_fn, x, use_reentrant=True)

        return self.forward_fn(x)


# * --- Blocks --- #


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int | None = None,
        dropout: float,
        use_residual_factor: bool = False,
        act_type: tuple = ("gelu", "gelu"),
        nin_shortcut_norm: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        padding_mode = kwargs.get("padding_mode", "zeros")
        norm_type = kwargs.get("norm_type", "gn")
        gn_norm_groups = kwargs.get("num_groups", 32)
        self.use_dico_cca = kwargs.get("use_dico_cca", False)

        self.norm1 = Normalize(
            in_channels, num_groups=gn_norm_groups, norm_type=norm_type
        )
        self.act1 = ACT2FN[act_type[0]]
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode=padding_mode,
        )
        self.norm2 = Normalize(
            out_channels, num_groups=gn_norm_groups, norm_type=norm_type
        )
        self.act2 = ACT2FN[act_type[1]]
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode=padding_mode,
        )
        if in_channels != out_channels:
            if nin_shortcut_norm:
                self.nin_shortcut = nn.Sequential(
                    Normalize(  # type: ignore
                        in_channels, num_groups=gn_norm_groups, norm_type=norm_type
                    ),
                    nn.Conv2d(
                        in_channels, out_channels, kernel_size=1, stride=1, padding=0
                    ),
                )
            else:
                self.nin_shortcut = nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )
        else:
            self.nin_shortcut = nn.Identity()
        self.act_checkpoint = kwargs.get("act_checkpoint", False)
        self.use_residual_factor = use_residual_factor
        if use_residual_factor:
            self.residual_factor = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        if self.use_dico_cca:
            self.dico_cca = DiCoCompactChannelAttention(out_channels)

    @_compile_decorator
    def forward_fn(
        self,
        x: torch.Tensor,
        slots: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = self.act1(h)
        h = self.conv1(h)
        if self.use_dico_cca:
            h = self.dico_cca(h)

        h = self.norm2(h)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)

        x = self.nin_shortcut(x)
        if self.use_residual_factor:
            h = h * self.residual_factor
        res = x + h
        return res

    def forward(
        self,
        x: torch.Tensor,
        slots: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # slots, t not used, compacted with ResnetBlockSlotsInjected
        # torch.compiler.cudagraph_mark_step_begin()
        if self.act_checkpoint and self.training:
            return checkpoint(
                self.forward_fn, x, use_reentrant=True
            )  # .clone()  # type: ignore
        return self.forward_fn(x)  # .clone()


class ResnetBlockSlotsInjected(ResnetBlock):
    def __init__(
        self,
        *,
        in_channels: int,
        slot_dim: int,
        time_dim: int,
        out_channels: int | None = None,
        dropout: float = 0.0,
        act_checkpoint: bool = False,
        use_residual_factor: bool = False,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            act_checkpoint=act_checkpoint,
            use_residual_factor=use_residual_factor,
        )
        self.in_channels = in_channels
        self.slot_dim = slot_dim
        self.time_dim = time_dim
        self.time_to_hidden = nn.Linear(time_dim, in_channels)
        self.slot_to_hidden = nn.Conv2d(
            slot_dim,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="zeros",
        )
        self.slots_t_to_mod = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(in_channels, in_channels * 2, 3, 1, 1),
        )

        # zero out the last conv for condition
        self.slots_t_to_mod[-1].weight.data.zero_()
        self.slots_t_to_mod[-1].bias.data.zero_()

    def forward_fn(self, x, slots, t):
        # interpolate 2d
        slots = F.interpolate(slots, size=x.shape[-2:], mode="nearest")
        slots = self.slot_to_hidden(slots)
        t = self.time_to_hidden(t)
        slots_t = slots + t[..., None, None]
        scale, shift = self.slots_t_to_mod(slots_t).chunk(2, dim=1)

        # base forward
        h = x
        h = self.norm1(h)
        h = h * (1 + scale) + shift

        h = nonlinearity(h)
        h = self.conv1(h)

        # base forward
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        x = self.nin_shortcut(x)

        if self.use_residual_factor:
            h = h * self.residual_factor

        return x + h

    def forward(
        self, x: torch.Tensor, slots: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        if self.act_checkpoint and self.training:
            return checkpoint(self.forward_fn, x, slots, t, use_reentrant=True)  # type: ignore
        return self.forward_fn(x, slots, t)


# * --- Convnext blocks --- #


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim,
        out_dim,
        norm_type="gn",
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        act_checkpoint=False,
        padding_mode="zeros",
        num_groups=32,
    ):
        super().__init__()

        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim, padding_mode=padding_mode
        )
        self.norm = Normalize(dim, norm_type=norm_type, num_groups=num_groups)

        self.pwconv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(hidden_dim, out_dim, kernel_size=1)

        # Layer Scaling
        self.gamma = (
            nn.Parameter(
                layer_scale_init_value * torch.ones((out_dim)), requires_grad=True
            )
            if layer_scale_init_value > 0
            else None
        )
        if dim != out_dim:
            self.nin_shortcut = nn.Conv2d(dim, out_dim, kernel_size=1, stride=1)
        else:
            self.nin_shortcut = nn.Identity()

        # Stochastic Depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.act_checkpoint = act_checkpoint

    @_compile_decorator
    def forward_fn(self, x):
        input = x

        x = self.dwconv(x)
        x = self.norm(x)

        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        # Layer Scaling
        if self.gamma is not None:
            x = self.gamma.view(-1, 1, 1) * x

        x = self.nin_shortcut(input) + self.drop_path(x)

        return x

    def forward(self, x):
        if self.act_checkpoint and self.training:
            return checkpoint(self.forward_fn, x, use_reentrant=True)  # type: ignore
        return self.forward_fn(x)


# * --- Diffusion blocks --- #


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(
        self, hidden_size, frequency_embedding_size=256, time_scale: float = 1.0
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.register_buffer(
            "time_scale",
            torch.tensor(time_scale),
        )

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(
            t * self.time_scale, self.frequency_embedding_size
        )
        t_emb = self.mlp(t_freq.to(self.mlp[0].weight.dtype))
        return t_emb


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(
        self,
        hidden_size,
        patch_size,
        out_channels,
        z_channels,
        t_channels,
    ):
        super().__init__()
        self.norm_final = Normalize(hidden_size)
        self.conv = nn.Conv2d(
            hidden_size, patch_size * patch_size * out_channels, 3, 1, 1
        )
        self.t_embd = nn.Linear(t_channels, hidden_size, bias=True)
        self.z_embd = nn.Conv2d(z_channels, hidden_size, 1, 1)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Conv2d(hidden_size, 2 * hidden_size, 1, 1)
        )

    def forward(self, x, z, t):
        c = (
            self.z_embd(F.interpolate(z, size=x.shape[-2:], mode="nearest"))
            + self.t_embd(t)[..., None, None]
        )
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.norm_final(x) * (1 + scale) + shift
        x = self.conv(x)
        return x
