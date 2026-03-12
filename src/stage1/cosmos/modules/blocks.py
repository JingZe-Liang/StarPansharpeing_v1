import math
from functools import partial
from inspect import isclass
from typing import (
    Annotated,
    Any,
    Callable,
    Literal,
    Optional,
    no_type_check,
)

import lazy_loader
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.state import PartialState
from einops import rearrange
from einops.layers.torch import Rearrange
from loguru import logger
from timm.layers import LayerScale2d, create_act_layer, create_conv2d, create_norm
from timm.layers.drop import DropPath
from timm.layers.weight_init import lecun_normal_
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from transformers.activations import ACT2FN

from src.stage1.MoEs.deepseek_moe.moe_layer import DeepseekECMoE
from src.stage1.MoEs.deepseek_moe.moe_layer import DeepseekV2MoE as DeepSeekTCMoE
from src.utilities.network_utils import compile_decorator, model_compiled_flag, safe_init_weights

from .utils import (
    Normalize,
    extract_needed_kwargs,
    nonlinearity,
    unit_magnitude_normalize,
    val2tuple,
)

natten = lazy_loader.load("natten")

type AdaptiveConvMode = Literal[
    "slice",
    "interp",
    "interp_proj",
    "mix",
    "sitok",
    "sitok_film",
    "sitok_pointwise",
    "cross_attn",
]
type AdaptiveLinearMode = Literal["slice", "interp", "sitok", "sitok_film", "sitok_pointwise"]
type ChannelMixRouterCondition = Literal[
    "none",
    "per_channel_mean",
    "per_channel_dw_pool",
    "per_channel_mean_dw_pool",
]
type SitokReduceMode = Literal["none", "sum", "mean", "pointwise"]
type KernelNormDim = Literal["c_in", "c_out"]
type SitokOutputVariant = Literal["film", "pointwise"]


# * --- Utilities --- #


def get_same_padding(kernel_size: int | tuple[int, ...]):
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    elif isinstance(kernel_size, int):
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2
    else:
        raise ValueError(f"kernel_size should be int or tuple, got {type(kernel_size)}")


# Create constant tuple of norm classes to avoid recreating it every time
_NORM_CLASSES = tuple(
    create_norm.get_norm_layer(f)
    for f in ["batchnorm2d", "rmsnorm", "rmsnorm2d", "layernorm", "layernorm2d", "groupnorm"]
)


def block_basic_init(
    module: nn.Module,
    name: str = "",
    dim: int | None = None,
    init_type: str | None = "trunc_normal",
    mode: str = "fan_in",
    nonlinearity: str = "relu",
    trunc_bounds: tuple[float, float] = (-2.0, 2.0),
    trunc_std: float | None = None,
):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        if init_type in (None, "trunc_normal"):
            if trunc_std is None:
                if dim is not None:
                    std = math.sqrt(2 / dim)
                else:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    std = math.sqrt(2 / fan_in)
            else:
                std = trunc_std
            nn.init.trunc_normal_(module.weight, std=std, a=trunc_bounds[0], b=trunc_bounds[1])
        elif init_type == "lecun_normal":
            # LeCun normal: std = sqrt(1 / fan_in), suitable for SELU activation
            # fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
            # std = trunc_std if trunc_std is not None else math.sqrt(1.0 / fan_in)
            # nn.init.trunc_normal_(module.weight, std=std, a=trunc_bounds[0], b=trunc_bounds[1])
            lecun_normal_(module.weight)
        elif init_type == "kaiming_normal":
            nn.init.kaiming_normal_(module.weight, mode=mode, nonlinearity=nonlinearity)
        elif init_type == "kaiming_uniform":
            nn.init.kaiming_uniform_(module.weight, mode=mode, nonlinearity=nonlinearity)
        elif init_type == "xavier_normal":
            nn.init.xavier_normal_(module.weight)
        elif init_type == "xavier_uniform":
            nn.init.xavier_uniform_(module.weight)
        else:
            raise ValueError(f"init_type {init_type} is not supported")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, _NORM_CLASSES):
        if hasattr(module, "weight") and module.weight is not None:
            nn.init.ones_(module.weight)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.zeros_(module.bias)


def _to_conv_channels_last_memformat(x: Tensor) -> Tensor:
    if (not model_compiled_flag) or (not x.is_cuda) or x.ndim != 4:
        return x
    if x.is_contiguous(memory_format=torch.channels_last):
        return x
    return x.contiguous(memory_format=torch.channels_last)


def _can_use_activation_checkpoint(training: bool) -> bool:
    return training and torch.is_grad_enabled() and (not torch.is_inference_mode_enabled())


def _pad2d_like_conv(x: Tensor, pad: int, padding_mode: str) -> Tensor:
    """
    Padding in nn.Conv2d will cause the channels_last memory format to
    contiguous format which makes the torch.compile runtime fail at different
    tensor stride.

    Fix:
        after the padding, mannually cast the memory format to channels_last.
    """
    if pad == 0:
        return x

    padding = (pad, pad, pad, pad)
    if padding_mode in ("zeros", "constant"):
        x = F.pad(x, padding, mode="constant", value=0.0)
    else:
        x = F.pad(x, padding, mode=padding_mode)

    return _to_conv_channels_last_memformat(x)


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
        act_func: str | None = "relu",
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
        self.act = create_act_layer(act_func)

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
                self.weight.copy_(unit_magnitude_normalize(w))  # forced weight normalization
        w = unit_magnitude_normalize(w)  # traditional weight normalization
        w = w * (gain / np.sqrt(w[0].numel()))  # magnitude-preserving scaling
        w = w.to(x.dtype)

        # if w.ndim == 2:
        #     return x @ w.t()
        # assert w.ndim == 4

        return self._conv_forward(x, w, self.bias)


# * --- Residual block builder --- #


class ResidualBlock(nn.Module):
    def __init__(
        self,
        main: nn.Module,
        shortcut: nn.Module,
        post_act=None,
        pre_norm=None,
    ):
        super().__init__()

        self.pre_norm = pre_norm
        self.main = main
        self.shortcut = shortcut
        self.post_act = create_act_layer(post_act)

    # @compile_decorator
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

        mid_channels = round(in_channels * expand_ratio) if mid_channels is None else mid_channels

        # self.glu_act = build_act(act_func[1], inplace=False)
        self.glu_act = create_act_layer(act_func[1], inplace=False)
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
        logger.debug(f"[Attention Block]: use sdpa: {self.sdpa}")

        self.norm = Normalize(in_channels, norm_type=norm_type, num_groups=norm_groups)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

        self.act_checkpoint = act_checkpoint
        self.use_residual_factor = use_residual_factor
        if self.use_residual_factor:
            self.residual_factor = nn.Parameter(torch.ones(1, in_channels, 1, 1) * 1e-5)

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
        if self.act_checkpoint and _can_use_activation_checkpoint(self.training):
            return checkpoint(fn, x, use_reentrant=False)
        return fn(x)

    def init_weights(self):
        self.apply(block_basic_init)


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
        q, k, v = rearrange(qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", heads=self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w)
        return self.to_out(out)

    def init_weights(self):
        self.apply(block_basic_init)


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
        # self.kernel_func = build_act(kernel_func, inplace=False)
        self.kernel_func = create_act_layer(kernel_func, inplace=False)

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
        att_map = att_map / (torch.sum(att_map, dim=2, keepdim=True) + self.eps)  # b h n n
        att_map = att_map.to(original_dtype)
        out = torch.matmul(v, att_map)  # b h d n

        out = torch.reshape(out, (B, -1, H, W))
        return out

    # @compile_decorator
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

    def init_weights(self):
        self.apply(block_basic_init)


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
        self.qkv = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

        self.heads = heads
        self.ksize = ksize
        self.stride = stride
        self.dilation = dilation

        self.act_checkpoint = act_checkpoint
        assert natten is not None, "Please install natten to use attention"

    def forward(self, x):
        qkv = self.qkv(self.norm(x))  # [bs, c, h, w]
        qkv = rearrange(qkv, "b (qkv h c) x y -> qkv b x y h c", qkv=3, h=self.heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # [bs, x, y, h, c]
        out = natten.na2d(  # type: ignore
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

    def init_weights(self):
        self.apply(block_basic_init)


def make_attn(in_channels, attn_type="vanilla", act_checkpoint=False):
    logger.debug(f"making attention of type '{attn_type}' with {in_channels=}")

    if attn_type is None:
        return nn.Identity(in_channels)

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
            f"attn_type {attn_type} is not supported, supported types are: ['vanilla', 'linear', 'lite_mla', 'none']"
        )


# * --- FFNs --- #


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
        self.act_fn = ACT2FN[hidden_act] if isinstance(hidden_act, str) else hidden_act()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class LlamaFFN(nn.Module):
    def __init__(self, hidden_size, intermediate_size, conv_cls=nn.Conv2d):
        super().__init__()
        self.w1 = conv_cls(hidden_size, intermediate_size, 1, 1, 0)
        self.w2 = conv_cls(intermediate_size, hidden_size, 1, 1, 0)
        self.w3 = conv_cls(hidden_size, intermediate_size, 1, 1, 0)

    @safe_init_weights
    def init_weights(self):
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
        self.hidden_features = hidden_features = hidden_features or in_features

        bias = (bias, bias)
        drop_probs = (drop, drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1)

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer() if isclass(act_layer) or isinstance(act_layer, partial) else act_layer
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
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

    @safe_init_weights
    def init_weights(self):
        block_basic_init(self)


# * --- Input and output convs with different bands images --- #


def _create_conv_in_module(basic_module: str, c: int, hidden_dim: int, padding_mode, norm_type: str = "rmsnorm2d"):
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
            norm_type=norm_type,
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
        raise ValueError(f"[DiffBandsInputConvIn] Unknown basic_module={basic_module}")
    return module


def _create_conv_out_module(
    basic_module: str,
    c: int,
    hidden_dim: int,
    padding_mode: str,
    norm_type: str = "rmsnorm2d",
):
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
            nn.Conv2d(hidden_dim, c, kernel_size=1, stride=1),
        )
    elif basic_module == "resnet":
        module = ResnetBlock(
            in_channels=hidden_dim,
            out_channels=c,
            dropout=0.0,
            act_checkpoint=False,
            num_groups=1,
            norm_type=norm_type,
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
            nn.Conv2d(hidden_dim, c, kernel_size=1, stride=1, padding=0),
        )
    else:
        raise ValueError(f"[DiffBandsInputConvOut] Unknown basic_module={basic_module}")
    return module


class DiffBandsInputConvIn(nn.Module):
    def __init__(
        self,
        band_lst: list[int],
        hidden_dim: int,
        basic_module: Literal["conv", "resnet", "inv_bottleneck", "moe"] = "conv",
        norm_type: str = "rmsnorm2d",
        padding_mode: str = "zeros",
        check_grads: bool = True,
    ):
        super().__init__()
        self.band_lst = band_lst
        self.hidden_dim = hidden_dim
        self.is_ddp = PartialState().use_distributed or check_grads
        self._in_module_partial_kwargs: dict[str, Any] = {
            "basic_module": basic_module,
            "hidden_dim": hidden_dim,
            "padding_mode": padding_mode,
            "norm_type": norm_type,
        }

        self.in_modules = nn.ModuleDict()
        for c in band_lst:
            module = _create_conv_in_module(c=c, **self._in_module_partial_kwargs)
            self.in_modules["conv_in_{}".format(c)] = module
            logger.debug(f"[DiffBandsInputConvIn] set conv to hidden module and buffer for channel {c}")

    def add_or_drop_modules(self, add_chans: list[int] | None = None, drop_chans: list[int] | None = None):
        """
        Add or remove convolution modules for specific channels.

        Args:
            add_chans: List of channel numbers to add modules for
            drop_chans: List of channel numbers to remove modules for
        """
        if drop_chans is not None:
            for drop_chan in drop_chans:
                if drop_chan in self.band_lst:
                    # Remove module
                    conv_key = f"conv_in_{drop_chan}"
                    if conv_key in self.in_modules:
                        del self.in_modules[conv_key]

                    # Update band list
                    self.band_lst = [c for c in self.band_lst if c != drop_chan]

        if add_chans is not None:
            for add_chan in add_chans:
                if add_chan not in self.band_lst:
                    # Add module using saved kwargs
                    conv_key = f"conv_in_{add_chan}"
                    self.in_modules[conv_key] = _create_conv_in_module(c=add_chan, **self._in_module_partial_kwargs)

                    # Update band list
                    self.band_lst.append(add_chan)
                    self.band_lst.sort()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c_ = x.shape[1]
        module = getattr(self.in_modules, "conv_in_{}".format(c_))
        if module is None:
            raise ValueError(f"[DiffBandsInputConvIn] no module for channel {c_}, please check the channel list")
        h = module(x)

        if self.training and self.is_ddp:
            for c in self.band_lst:
                if c != c_:
                    m = self.in_modules["conv_in_{}".format(c)]
                    dummy_loss = sum(p.sum() * 0.0 for p in m.parameters())
                    h = h + dummy_loss

        return h

    @safe_init_weights
    def init_weights(self):
        # init: kaiming uniform
        for name, module in self.in_modules.items():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, a=np.sqrt(5))
                if module.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(module.bias, -bound, bound)


class DiffBandsInputConvOut(nn.Module):
    def __init__(
        self,
        band_lst: list[int],
        hidden_dim: int,
        basic_module: str = "conv",
        padding_mode: str = "zeros",
        norm_type: str = "rmsnorm2d",
        check_grads: bool = True,
    ):
        super().__init__()

        self.band_lst = band_lst
        self.hidden_dim = hidden_dim
        self.basic_module = basic_module
        self.is_ddp = PartialState().use_distributed or check_grads
        self._out_module_partial_kwargs: dict[str, Any] = {
            "basic_module": basic_module,
            "hidden_dim": hidden_dim,
            "padding_mode": padding_mode,
            "norm_type": norm_type,
        }

        self.in_modules = nn.ModuleDict()
        for c in band_lst:
            module = _create_conv_out_module(c=c, **self._out_module_partial_kwargs)
            self.in_modules["conv_out_{}".format(c)] = module
            logger.debug(f"[DiffBandsInputConvOut] set conv to hidden module for channel {c}")

        self.out_channel: int | None = None

    def add_or_drop_modules(self, add_chans: list[int] | None = None, drop_chans: list[int] | None = None):
        """
        Add or remove convolution modules for specific channels.

        Args:
            add_chans: List of channel numbers to add modules for
            drop_chans: List of channel numbers to remove modules for
        """
        if drop_chans is not None:
            for drop_chan in drop_chans:
                if drop_chan in self.band_lst:
                    # Remove module
                    conv_key = f"conv_out_{drop_chan}"
                    if conv_key in self.in_modules:
                        del self.in_modules[conv_key]

                    # Update band list
                    self.band_lst = [c for c in self.band_lst if c != drop_chan]

        if add_chans is not None:
            for add_chan in add_chans:
                if add_chan not in self.band_lst:
                    # Add module using saved kwargs
                    conv_key = f"conv_out_{add_chan}"
                    self.in_modules[conv_key] = _create_conv_out_module(c=add_chan, **self._out_module_partial_kwargs)

                    # Update band list
                    self.band_lst.append(add_chan)
                    self.band_lst.sort()

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
        assert self.out_channel is not None, "out_channel is not set, please call forward first"
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

    @safe_init_weights
    def init_weights(self):
        # init: kaiming uniform
        for name, module in self.in_modules.items():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, a=np.sqrt(5))
                if module.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(module.bias, -bound, bound)


# * --- Nested Diffbands conv --- #


def _kernel_norm(
    w: Annotated[Tensor, "c_out c_in k k"],
    kernel_norm: str | None,
    dim: KernelNormDim = "c_in",
) -> Annotated[Tensor, "c_out c_in k k"]:
    if kernel_norm is None:
        return w

    if kernel_norm == "softmax":
        # Apply softmax on input channel dimension
        dim_ = 1 if dim == "c_in" else 0
        w = F.softmax(w, dim=dim_)
        return w

    elif kernel_norm == "layernorm":
        w_shape = w.shape
        if dim == "c_in":
            w = w.reshape(*w.shape[:2], -1).permute(0, 2, 1)  # c_out, (k*k), c_in
            w = F.layer_norm(w, [w.shape[-1]])  # normalize on c_in dimension
            return w.permute(0, 2, 1).reshape(w_shape)  # c_out, c_in, k, k
        else:
            w = w.reshape(*w.shape[2:], -1).permute(1, 2, 0)  # c_in, (k*k) c_out
            w = F.layer_norm(w, [w.shape[-1]])  # normalize on c_out dimension
            return w.permute(2, 1, 0).reshape(w_shape)  # c_out, c_in, k, k

    elif kernel_norm == "weight_std":
        dim_ = 1 if dim == "c_in" else 0
        # Weight standardization: normalize to zero mean and unit std
        mean = w.mean(dim=dim_, keepdim=True)  # mean over c_in dimension
        std = w.std(dim=dim_, keepdim=True) + 1e-5  # std over c_in dimension
        return (w - mean) / std

    else:
        raise ValueError(f"Unknown kernel_norm type: {kernel_norm}")


def _normalized_channel_coords(n: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    if n <= 1:
        return torch.zeros((n,), device=device, dtype=dtype)
    return torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)


def _sincos_channel_index_embedding(
    n_channels: int,
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    base: float = 10000.0,
) -> Tensor:
    if dim <= 0:
        raise ValueError(f"{dim=} must be > 0")
    if n_channels <= 0:
        raise ValueError(f"{n_channels=} must be > 0")
    if base <= 0:
        raise ValueError(f"{base=} must be > 0")

    pos = torch.arange(n_channels, device=device, dtype=dtype)
    half_dim = (dim + 1) // 2
    div_term = torch.exp(torch.arange(half_dim, device=device, dtype=dtype) * (-math.log(base) / dim))

    emb = torch.zeros((n_channels, dim), device=device, dtype=dtype)
    sin_inp = pos[:, None] * div_term[None, :]
    emb[:, 0::2] = torch.sin(sin_inp)[:, : emb[:, 0::2].shape[1]]
    if dim > 1:
        emb[:, 1::2] = torch.cos(sin_inp)[:, : emb[:, 1::2].shape[1]]
    return emb


class _ChannelMixRouter(nn.Module):
    def __init__(
        self,
        *,
        base_channels: int,
        condition: ChannelMixRouterCondition = "none",
        hidden_dim: int = 128,
        temperature: float = 1.0,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"{temperature=} must be > 0")

        self.base_channels = base_channels
        self.condition = condition
        self.temperature = float(temperature)

        match condition:
            case "none":
                in_features = 1
            case "per_channel_mean" | "per_channel_dw_pool":
                in_features = 2
            case "per_channel_mean_dw_pool":
                in_features = 3
            case _:
                raise ValueError(f"Unknown condition: {condition}")
        if hidden_dim <= 0:
            self.proj = nn.Linear(in_features, base_channels, bias=use_bias)
        else:
            self.proj = nn.Sequential(
                nn.Linear(in_features, hidden_dim, bias=use_bias),
                nn.SiLU(),
                nn.Linear(hidden_dim, base_channels, bias=use_bias),
            )

    def forward(
        self,
        coords: Tensor,
        *,
        channel_mean: Tensor | None = None,
        channel_dw_pool: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            coords: (c,) normalized coordinates in [0, 1]
            channel_mean: if condition == 'per_channel_mean', expects (b, c)
            channel_dw_pool: if condition includes 'dw_pool', expects (b, c)

        Returns:
            mixing weights:
              - (c, base_channels) when condition == 'none'
              - (b, c, base_channels) when condition == 'per_channel_mean'
        """
        if coords.ndim != 1:
            raise ValueError(f"coords must be 1D, got shape {tuple(coords.shape)}")

        match self.condition:
            case "none":
                x = coords[:, None]
            case "per_channel_mean":
                if channel_mean is None:
                    raise ValueError("channel_mean is required when condition='per_channel_mean'")
                if channel_mean.ndim != 2 or channel_mean.shape[1] != coords.shape[0]:
                    raise ValueError(
                        f"channel_mean must be (b, c={coords.shape[0]}), got shape {tuple(channel_mean.shape)}"
                    )
                x = torch.stack(
                    [
                        coords[None, :].expand(channel_mean.shape[0], -1),
                        channel_mean,
                    ],
                    dim=-1,
                )
            case "per_channel_dw_pool":
                if channel_dw_pool is None:
                    raise ValueError("channel_dw_pool is required when condition='per_channel_dw_pool'")
                if channel_dw_pool.ndim != 2 or channel_dw_pool.shape[1] != coords.shape[0]:
                    raise ValueError(
                        f"channel_dw_pool must be (b, c={coords.shape[0]}), got shape {tuple(channel_dw_pool.shape)}"
                    )
                x = torch.stack(
                    [
                        coords[None, :].expand(channel_dw_pool.shape[0], -1),
                        channel_dw_pool,
                    ],
                    dim=-1,
                )
            case "per_channel_mean_dw_pool":
                if channel_mean is None:
                    raise ValueError("channel_mean is required when condition='per_channel_mean_dw_pool'")
                if channel_dw_pool is None:
                    raise ValueError("channel_dw_pool is required when condition='per_channel_mean_dw_pool'")
                if channel_mean.ndim != 2 or channel_mean.shape[1] != coords.shape[0]:
                    raise ValueError(
                        f"channel_mean must be (b, c={coords.shape[0]}), got shape {tuple(channel_mean.shape)}"
                    )
                if channel_dw_pool.ndim != 2 or channel_dw_pool.shape[1] != coords.shape[0]:
                    raise ValueError(
                        f"channel_dw_pool must be (b, c={coords.shape[0]}), got shape {tuple(channel_dw_pool.shape)}"
                    )
                x = torch.stack(
                    [
                        coords[None, :].expand(channel_mean.shape[0], -1),
                        channel_mean,
                        channel_dw_pool,
                    ],
                    dim=-1,
                )
            case _:
                raise ValueError(f"Unknown condition: {self.condition}")

        logits = self.proj(x)
        return F.softmax(logits / self.temperature, dim=-1)


class AdaptiveInputConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        padding: int | None = None,
        use_bias: bool = False,
        mode: AdaptiveConvMode = "slice",
        k_hidden: int | None = None,
        kernel_norm: str | None = None,
        router_condition: ChannelMixRouterCondition = "none",
        router_hidden_dim: int = 128,
        router_temperature: float = 1.0,
        always_use_router: bool = False,
        router_dw_kernel_size: int = 3,
        sitok_reduce: SitokReduceMode = "none",
        sitok_embed_scale: float = 1.0,
        sitok_embed_base: float = 10000.0,
        cross_attn_pool_size: int = 4,
        cross_attn_embed_dim: int = 64,
    ):
        """
        Adaptive 2D convolution with a variable number of input channels.

        This layer is meant for inputs where the spectral/channel count can vary between runs
        (e.g., hyperspectral data with different band counts). It keeps a single "base" kernel
        (initialized with `in_channels`) and adapts it to the runtime `x.shape[1]` by slicing,
        interpolation, mixing, or a SiTok-style per-channel shared kernel.

        Key args:
        - `in_channels` / `out_channels`: base kernel in/out channels; runtime input channels come from `x.shape[1]`.
        - `mode`:
          - `"slice"`: use `weight[:, :in_channels]` (only makes sense when runtime `C_in <= base_in`).
          - `"interp"`: interpolate the kernel along the input-channel dimension to match runtime `C_in`.
          - `"interp_proj"`: project (two linear layers) then interpolate to runtime `C_in`.
          - `"mix"`: router-generated coefficients mix base input channels into runtime `C_in` (requires `groups=1`).
          - `"sitok"`: apply the SAME kernel to each input channel independently and add a sinusoidal channel-index
            embedding; `sitok_reduce` controls whether to aggregate across channels (requires `groups=1`).
        - `always_use_router`: if False and runtime `C_in == base_in`, use the native conv path (faster).
        - `kernel_norm`: passed to `_kernel_norm` (normalized over `"c_in"`).
        - `router_*`: used only in `mode="mix"`; optionally conditions on per-channel statistics.
        - `sitok_*`: used only in `mode="sitok"`; controls channel-index embedding scale/frequency and reduction.

        Shapes: input `x` is `(B, C_in, H, W)`. Output channels are `out_channels`, except when
        `mode="sitok" and sitok_reduce="none"`, where output is `(B, C_in * out_channels, H, W)`.
        """
        super().__init__()
        if mode in ("sitok_film", "sitok_pointwise"):
            mode = "sitok"
        conv_groups = groups
        if mode == "sitok":
            if groups != 1:
                raise ValueError("AdaptiveInputConvLayer(mode='sitok') currently requires groups=1")
            conv_groups = 1
        elif mode == "cross_attn":
            if groups != 1:
                raise ValueError("AdaptiveInputConvLayer(mode='cross_attn') currently requires groups=1")
            conv_groups = out_channels
        conv_kwargs = dict(stride=stride, groups=conv_groups, dilation=dilation, bias=use_bias)
        if padding is not None:
            # if padding not set, the create_conv2d will use same padding
            conv_kwargs["padding"] = padding

        conv_in_channels = 1 if mode == "sitok" else (out_channels if mode == "cross_attn" else in_channels)
        self.conv = create_conv2d(conv_in_channels, out_channels, kernel_size, **conv_kwargs)
        self.mode = mode
        self.kernel_norm = kernel_norm
        self.always_use_router = always_use_router
        self.sitok_reduce = sitok_reduce
        self.sitok_embed_scale = float(sitok_embed_scale)
        self.sitok_embed_base = float(sitok_embed_base)
        self.cross_attn_pool_size = int(cross_attn_pool_size)
        self.cross_attn_embed_dim = int(cross_attn_embed_dim)
        self.sitok_reduce_head: nn.Module | None = None
        self.cross_attn_band_proj: nn.Linear | None = None
        self.cross_attn_query: nn.Parameter | None = None

        if mode == "interp_proj":
            # (bs, c, k1, k2) img -> (c_out, c_in, k1, k2) kernel
            # kernel -> (c_out, k1*k2, c_in) -> Linear(in_channels, k_hidden) -> k_hidden -> c_in -> (c_out, k1*k2, c_in)
            k_hidden = k_hidden or in_channels
            self.kernel_proj = nn.ModuleList(
                [
                    nn.Linear(in_channels, k_hidden, bias=use_bias),
                    nn.Linear(k_hidden, in_channels),
                ]
            )
        elif mode == "mix":
            if groups != 1:
                raise ValueError("AdaptiveInputConvLayer(mode='mix') currently requires groups=1")
            self.in_router = _ChannelMixRouter(
                base_channels=in_channels,
                condition=router_condition,
                hidden_dim=router_hidden_dim,
                temperature=router_temperature,
                use_bias=use_bias,
            )
            if "dw_pool" in router_condition:
                if router_dw_kernel_size <= 0 or (router_dw_kernel_size % 2) != 1:
                    raise ValueError(f"{router_dw_kernel_size=} must be a positive odd number")
                k = router_dw_kernel_size
                kernel = torch.zeros((k, k), dtype=torch.float32)
                kernel[k // 2, k // 2] = 1.0
                self.router_dw_kernel = nn.Parameter(kernel)
        elif mode == "sitok":
            if sitok_reduce not in ("none", "sum", "mean", "pointwise"):
                raise ValueError(f"Unknown sitok_reduce: {sitok_reduce}")
            if self.sitok_embed_scale < 0:
                raise ValueError(f"{sitok_embed_scale=} must be >= 0")
            if self.sitok_embed_base <= 0:
                raise ValueError(f"{sitok_embed_base=} must be > 0")
            if sitok_reduce == "pointwise":
                self.sitok_reduce_head = nn.Sequential(
                    nn.Linear(out_channels, out_channels, bias=True),
                    nn.SiLU(),
                    nn.Linear(out_channels, out_channels, bias=True),
                )
        elif mode == "cross_attn":
            if self.cross_attn_pool_size <= 0:
                raise ValueError(f"{cross_attn_pool_size=} must be > 0")
            if self.cross_attn_embed_dim <= 0:
                raise ValueError(f"{cross_attn_embed_dim=} must be > 0")
            pooled_dim = self.cross_attn_pool_size * self.cross_attn_pool_size
            self.cross_attn_band_proj = nn.Linear(pooled_dim, self.cross_attn_embed_dim, bias=use_bias)
            self.cross_attn_query = nn.Parameter(torch.empty(1, out_channels, self.cross_attn_embed_dim))
            nn.init.trunc_normal_(self.cross_attn_query, std=0.02)

        self.forward_mappings: dict[str, Callable[..., Tensor]] = {
            "slice": self._slice_forward,
            "interp": self._interp_forward,
            "interp_proj": self._interp_proj_forward,
            "mix": self._mix_forward,
            "sitok": self._sitok_forward,
            "cross_attn": self._cross_attn_forward,
        }

    def _forward_conv_with_wb(self, x, w, b):
        w = w.to(device=x.device, dtype=x.dtype)
        if b is not None:
            b = b.to(device=x.device, dtype=x.dtype)
        x = nn.functional.conv2d(
            x,
            w,
            b,
            self.conv.stride,
            self.conv.padding,
            self.conv.dilation,
            self.conv.groups,
        )
        return x

    def _slice_forward(self, x):
        w = self.conv.weight[:, : x.shape[1]]
        w = _kernel_norm(w, self.kernel_norm, "c_in")
        x = self._forward_conv_with_wb(x, w, self.conv.bias)
        return x

    def _interp_forward(self, x, w: Tensor | None = None):
        in_channels = x.shape[1]
        if w is None:
            w = self.conv.weight
        assert w is not None

        c_out, c_in, k1, k2 = w.shape
        # c_in -> in_channels
        w = rearrange(w, "c_out c_in k1 k2 -> k1 k2 c_out c_in")
        w = torch.nn.functional.interpolate(w, size=(c_out, in_channels), mode="bicubic", align_corners=False)
        w = rearrange(w, "k1 k2 c_out c_in -> c_out c_in k1 k2")
        # Conv
        w = _kernel_norm(w, self.kernel_norm, "c_in")
        x = self._forward_conv_with_wb(x, w, self.conv.bias)
        return x

    def _interp_proj_forward(self, x):
        in_channels = x.shape[1]
        lin1, lin2 = self.kernel_proj

        # weights
        w: Tensor = self.conv.weight
        c_out, c_in, k1, k2 = w.shape

        # weight projection
        w = rearrange(w, "c_out c_in k1 k2 -> c_out (k1 k2) c_in")
        w = lin1(w)  # c_out, (k1*k2), k_hidden
        w = lin2(w)  # c_out, (k1*k2), c_in
        w = rearrange(w, "c_out (k1 k2) c_in -> c_out c_in k1 k2", k1=k1, k2=k2)

        # Interpolation
        return self._interp_forward(x, w)

    def _mix_forward(self, x: torch.Tensor) -> torch.Tensor:
        in_channels = x.shape[1]
        base_in = self.conv.weight.shape[1]
        if base_in == 0:
            raise ValueError("base_in_channels must be > 0")

        coords = _normalized_channel_coords(in_channels, device=x.device, dtype=x.dtype)

        channel_mean: torch.Tensor | None = None
        channel_dw_pool: torch.Tensor | None = None
        if self.in_router.condition in ("per_channel_mean", "per_channel_mean_dw_pool"):
            channel_mean = x.mean(dim=(2, 3))
        if self.in_router.condition in ("per_channel_dw_pool", "per_channel_mean_dw_pool"):
            if not hasattr(self, "router_dw_kernel"):
                raise RuntimeError("router_dw_kernel is missing while condition requires dw_pool")
            k = int(self.router_dw_kernel.shape[0])
            # NOTE: avoid branching on x.is_cuda during torch.compile; keep the layout stable for Inductor/AOTAutograd.
            w_dw = self.router_dw_kernel.to(dtype=x.dtype, device=x.device)[None, None].repeat(in_channels, 1, 1, 1)
            dw = F.conv2d(x, w_dw, bias=None, stride=1, padding=k // 2, groups=in_channels)
            channel_dw_pool = dw.abs().mean(dim=(2, 3))

        coeff = self.in_router(coords, channel_mean=channel_mean, channel_dw_pool=channel_dw_pool)

        w_base = self.conv.weight
        k1, k2 = w_base.shape[-2:]
        w_base_flat = rearrange(w_base, "c_out c_in k1 k2 -> c_out c_in (k1 k2)")

        if coeff.ndim == 2:
            # (c_in, base_in)
            w_flat = torch.einsum("ocp,ic->oip", w_base_flat, coeff)
            w = rearrange(w_flat, "c_out c_in (k1 k2) -> c_out c_in k1 k2", k1=k1, k2=k2)
            w = _kernel_norm(w, self.kernel_norm, "c_in")
            y = self._forward_conv_with_wb(x, w, self.conv.bias)
            return y

        # Per-sample dynamic weights: (b, c_in, base_in)
        bsz = x.shape[0]
        w_flat_b = torch.einsum("ocp,bic->boip", w_base_flat, coeff)
        w_b = rearrange(w_flat_b, "b c_out c_in (k1 k2) -> b c_out c_in k1 k2", k1=k1, k2=k2)
        w_b_flat = w_b.reshape(bsz * w_b.shape[1], in_channels, k1, k2)
        w_b_flat = _kernel_norm(w_b_flat, self.kernel_norm, "c_in")
        w_b = w_b_flat.reshape(bsz, -1, in_channels, k1, k2)

        bias = self.conv.bias
        bias_b = bias.repeat(bsz) if bias is not None else None

        # Force a stable NCHW contiguous layout for the grouped-conv trick, then restore channels_last on output.
        x_cf = x.contiguous()
        x_g = x_cf.reshape(1, bsz * in_channels, *x.shape[2:])
        w_g = w_b.reshape(bsz * w_b.shape[1], in_channels, k1, k2)
        y = F.conv2d(  # type: ignore
            x_g,
            w_g,
            bias_b,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=bsz,
        )
        y = y.reshape(bsz, -1, *y.shape[2:])
        return y

    def _sitok_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply a shared spatial kernel to variable-length spectral inputs in a SiTok-style way.

        The same normalized 2D kernel is reused for every spectral band, while an optional
        sinusoidal channel-index embedding provides spectral-position information. The reduction
        mode controls whether band-wise outputs are kept explicitly or collapsed into a fixed-width
        output tensor:

        - ``"none"``:
          apply the shared kernel to every input band independently and keep all band outputs.
          This produces ``(B, C_in * C_out, H_out, W_out)`` and is the most expressive, but it
          materializes the largest activation tensor.
        - ``"pointwise"``:
          compute a learned spectral mixing matrix from the channel-index embeddings, mix the raw
          input bands into ``C_out`` channels with ``einsum``, then apply the shared kernel once.
          This avoids materializing ``(B, C_in, C_out, H, W)`` while still allowing per-output
          spectral weighting.
        - ``"sum"`` / ``"mean"``:
          first reduce the spectral axis to a single aggregated map and then apply the shared
          kernel once. Because convolution is linear and shared across bands, this is equivalent to
          summing or averaging per-band outputs, but is much cheaper in memory and compute.

        Args:
            x: Input tensor of shape ``(B, C_in, H, W)``.

        Returns:
            Output tensor of shape ``(B, C_in * C_out, H_out, W_out)`` when
            ``sitok_reduce == "none"``, otherwise ``(B, C_out, H_out, W_out)``.
        """
        bsz, in_channels, h, w = x.shape
        w_shared = _kernel_norm(self.conv.weight, self.kernel_norm, "c_in")
        c_out = self.conv.weight.shape[0]
        ch_emb_scaled: Tensor | None = None
        if self.sitok_embed_scale != 0 or self.sitok_reduce == "pointwise":
            ch_emb = _sincos_channel_index_embedding(
                in_channels,
                c_out,
                device=x.device,
                dtype=x.dtype,
                base=self.sitok_embed_base,
            )
            ch_emb_scaled = ch_emb * self.sitok_embed_scale

        if self.sitok_reduce == "none":
            w_rep = w_shared.repeat(in_channels, 1, 1, 1)
            b_rep: Tensor | None
            if self.conv.bias is None:
                b_rep = None
            else:
                b_rep = self.conv.bias.repeat(in_channels)
            y = F.conv2d(  # type: ignore
                x,
                w_rep,
                b_rep,
                stride=self.conv.stride,
                padding=self.conv.padding,
                dilation=self.conv.dilation,
                groups=in_channels,
            )
            y = y.reshape(bsz, in_channels, c_out, y.shape[-2], y.shape[-1])
            if ch_emb_scaled is not None and self.sitok_embed_scale != 0:
                y = y + ch_emb_scaled[None, :, :, None, None]
            y = y.reshape(bsz, in_channels * c_out, y.shape[-2], y.shape[-1])
            return y

        if self.sitok_reduce == "pointwise":
            if self.sitok_reduce_head is None:
                raise RuntimeError("sitok_reduce_head is missing while sitok_reduce='pointwise'")
            if ch_emb_scaled is None:
                raise RuntimeError("ch_emb_scaled is missing while sitok_reduce='pointwise'")
            logits = self.sitok_reduce_head(ch_emb_scaled).float()
            weights = logits.softmax(dim=0).to(dtype=x.dtype)
            x_mix = torch.einsum("bchw,cd->bdhw", x, weights)

            bias_eff = self.conv.bias
            if self.sitok_embed_scale != 0:
                bias_add = (weights.float() * ch_emb_scaled.float()).sum(dim=0).to(dtype=x.dtype)
                if bias_eff is None:
                    bias_eff = bias_add
                else:
                    bias_eff = bias_eff + bias_add

            y = F.conv2d(  # type: ignore
                x_mix,
                w_shared,
                bias_eff,
                stride=self.conv.stride,
                padding=self.conv.padding,
                dilation=self.conv.dilation,
                groups=c_out,
            )
            return y

        # first mean_out c -> 1, then conv
        if self.sitok_reduce in ("sum", "mean"):
            if self.sitok_reduce == "sum":
                x_agg = x.sum(dim=1, keepdim=True)
            else:
                x_agg = x.mean(dim=1, keepdim=True)

            bias_eff = self.conv.bias
            if bias_eff is not None and self.sitok_reduce == "sum":
                bias_eff = bias_eff * in_channels

            if ch_emb_scaled is not None and self.sitok_embed_scale != 0:
                emb_agg = (ch_emb_scaled.sum(dim=0) if self.sitok_reduce == "sum" else ch_emb_scaled.mean(dim=0)).to(
                    dtype=x.dtype
                )
                if bias_eff is None:
                    bias_eff = emb_agg
                else:
                    bias_eff = bias_eff + emb_agg

            y = F.conv2d(  # type: ignore
                x_agg,
                w_shared,
                bias_eff,
                stride=self.conv.stride,
                padding=self.conv.padding,
                dilation=self.conv.dilation,
                groups=1,
            )
            return y

        y = None
        raise ValueError(f"Unknown sitok_reduce: {self.sitok_reduce}")

    def _cross_attn_project_band_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if self.cross_attn_band_proj is None:
            raise RuntimeError("cross_attn_band_proj is missing while mode='cross_attn'")
        p = int(self.cross_attn_pool_size)
        x_pool = F.adaptive_avg_pool2d(x, output_size=(p, p))
        band_desc = rearrange(x_pool, "b c p1 p2 -> b c (p1 p2)")
        proj_dtype = self.cross_attn_band_proj.weight.dtype
        return self.cross_attn_band_proj(band_desc.to(dtype=proj_dtype))

    def _cross_attn_alpha(self, band_tokens: torch.Tensor, x_dtype: torch.dtype) -> torch.Tensor:
        if self.cross_attn_query is None:
            raise RuntimeError("cross_attn_query is missing while mode='cross_attn'")
        query = self.cross_attn_query.expand(band_tokens.shape[0], -1, -1).contiguous()
        band_tokens_t = band_tokens.transpose(1, 2).contiguous()
        logits = torch.matmul(query, band_tokens_t)
        logits = logits * (query.shape[-1] ** -0.5)
        alpha = logits.softmax(dim=-1).transpose(1, 2)
        return alpha.to(dtype=x_dtype)

    def _cross_attn_spectral_mix(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        bsz, _, height, width = x.shape
        x_flat = x.reshape(bsz, x.shape[1], height * width).contiguous()
        alpha_t = alpha.transpose(1, 2).contiguous()
        x_mix = torch.bmm(alpha_t, x_flat)
        return x_mix.reshape(bsz, alpha_t.shape[1], height, width)

    def _cross_attn_forward(self, x: torch.Tensor) -> torch.Tensor:
        band_tokens = self._cross_attn_project_band_tokens(x)
        alpha = self._cross_attn_alpha(band_tokens, x.dtype)
        x_mix = self._cross_attn_spectral_mix(x, alpha)
        w_shared = _kernel_norm(self.conv.weight, self.kernel_norm, "c_in")
        return self._forward_conv_with_wb(x_mix, w_shared, self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_channels = x.shape[1]

        if self.mode in ("sitok", "cross_attn"):
            return self.forward_mappings[self.mode](x)

        # Native case
        if (not self.always_use_router) and in_channels == self.conv.weight.shape[1]:
            w = _kernel_norm(self.conv.weight, self.kernel_norm, "c_in")
            return self._forward_conv_with_wb(x, w, self.conv.bias)

        # Adaptive cases
        return self.forward_mappings[self.mode](x)

    @property
    def weight(self):
        return self.conv.weight

    @safe_init_weights
    def init_weights(self):
        def _inner(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)  # uniformly init
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(_inner)
        if self.cross_attn_query is not None:
            nn.init.trunc_normal_(self.cross_attn_query, std=0.02)


class AdaptiveOutputConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        padding: int | None = None,
        use_bias: bool = False,
        mode: AdaptiveConvMode = "slice",
        k_hidden: int | None = None,
        kernel_norm: str | None = None,
        router_condition: ChannelMixRouterCondition = "none",
        router_hidden_dim: int = 128,
        router_temperature: float = 1.0,
        router_dw_kernel_size: int = 3,
        sitok_embed_dim: int = 16,
        sitok_hidden_dim: int = 64,
        sitok_basis_dim: int = 32,
        sitok_embed_scale: float = 1.0,
        sitok_embed_base: float = 10000.0,
        cross_attn_pool_size: int = 4,
        cross_attn_embed_dim: int = 64,
    ):
        """
        Adaptive 2D convolution with a variable number of output channels.

        Unlike `AdaptiveInputConvLayer`, this layer keeps the input channel count fixed to the initialized
        `in_channels`, while allowing the output channels to be selected at runtime via `forward(..., out_channels=...)`.
        The constructor `out_channels` defines the base kernel size and is also the default when `out_channels=None`.

        Key args:
        - `in_channels`: input channels (must match `x.shape[1]`).
        - `out_channels`: base output channels (and the default output channels).
        - `mode`:
          - `"slice"`: slice `weight[:out_channels]` / `bias[:out_channels]` (requires `out_channels <= base_out`).
          - `"interp"`: interpolate kernel (and bias) along the output-channel dimension to the target `out_channels`.
          - `"interp_proj"`: project (two linear layers) then interpolate to the target `out_channels`.
          - `"mix"`: router-generated coefficients mix base output channels into target `out_channels` (requires `groups=1`).
          - `"sitok_film"`: shared conv -> base feature map, then FiLM `(gamma, beta)` per output channel from a
            channel-index embedding (requires `groups=1`).
          - `"sitok_pointwise"`: shared conv -> `D` basis features, then per-output-channel basis weights and bias from
            a channel-index embedding (requires `groups=1`).
          - `"sitok"`: backward-compatible alias for `"sitok_film"`.
        - `kernel_norm`: passed to `_kernel_norm` (normalized over `"c_out"`).
        - `router_*`: used only in `mode="mix"`; optionally conditions on input statistics.
        - `sitok_*`: used only in `mode in {"sitok_film","sitok_pointwise"}`; `sitok_basis_dim` only applies to pointwise.

        Shapes: input `x` is `(B, C_in, H, W)`, output is `(B, out_channels, H, W)`.
        """
        super().__init__()
        if mode == "sitok":
            mode = "sitok_film"
        self.mode = mode
        self.default_out_channels = out_channels
        self.cross_attn_pool_size = int(cross_attn_pool_size)
        self.cross_attn_embed_dim = int(cross_attn_embed_dim)

        conv_out_channels = out_channels
        conv_groups = groups
        if mode == "sitok_film":
            conv_out_channels = 1
        elif mode == "sitok_pointwise":
            if sitok_basis_dim <= 0:
                raise ValueError(f"{sitok_basis_dim=} must be > 0")
            conv_out_channels = int(sitok_basis_dim)
        elif mode == "cross_attn":
            if groups != 1:
                raise ValueError("AdaptiveOutputConvLayer(mode='cross_attn') currently requires groups=1")
            conv_out_channels = in_channels
            conv_groups = in_channels
        conv_kwargs = dict(
            stride=stride,
            groups=conv_groups,
            dilation=dilation,
            bias=use_bias,
        )
        if padding is not None:
            conv_kwargs["padding"] = padding

        self.conv = create_conv2d(
            in_channels,
            conv_out_channels,
            kernel_size,
            **conv_kwargs,
        )
        self.kernel_norm = kernel_norm
        self.cross_attn_kv_proj: nn.Linear | None = None
        self.cross_attn_q_proj: nn.Linear | None = None
        self.cross_attn_pw_head: nn.Linear | None = None

        if mode == "interp_proj":
            # For output channels, we project the output channel dimension
            # (c_out, c_in, k1, k2) kernel -> (c_out, k1*k2, c_in) -> Linear(c_out, k_hidden) -> k_hidden -> c_out -> (c_out, k1*k2, c_in)
            k_hidden = k_hidden or out_channels
            self.kernel_proj = nn.ModuleList(
                [
                    nn.Linear(out_channels, k_hidden, bias=use_bias),
                    nn.Linear(k_hidden, out_channels),
                ]
            )
        elif mode == "mix":
            if groups != 1:
                raise ValueError("AdaptiveOutputConvLayer(mode='mix') currently requires groups=1")
            self.out_router = _ChannelMixRouter(
                base_channels=out_channels,
                condition=router_condition,
                hidden_dim=router_hidden_dim,
                temperature=router_temperature,
                use_bias=use_bias,
            )
            if "dw_pool" in router_condition:
                if router_dw_kernel_size <= 0 or (router_dw_kernel_size % 2) != 1:
                    raise ValueError(f"{router_dw_kernel_size=} must be a positive odd number")
                k = router_dw_kernel_size
                kernel = torch.zeros((k, k), dtype=torch.float32)
                kernel[k // 2, k // 2] = 1.0
                self.router_dw_kernel = nn.Parameter(kernel)
        elif mode in ("sitok_film", "sitok_pointwise"):
            if groups != 1:
                raise ValueError("AdaptiveOutputConvLayer(mode='sitok_*') currently requires groups=1")
            if sitok_embed_dim <= 0:
                raise ValueError(f"{sitok_embed_dim=} must be > 0")
            if sitok_hidden_dim < 0:
                raise ValueError(f"{sitok_hidden_dim=} must be >= 0")
            if sitok_embed_scale < 0:
                raise ValueError(f"{sitok_embed_scale=} must be >= 0")
            if sitok_embed_base <= 0:
                raise ValueError(f"{sitok_embed_base=} must be > 0")

            self.sitok_embed_dim = int(sitok_embed_dim)
            self.sitok_embed_scale = float(sitok_embed_scale)
            self.sitok_embed_base = float(sitok_embed_base)

            if mode == "sitok_film":
                self.sitok_variant = "film"
                head_out_dim = 2  # gamma, beta
            else:
                self.sitok_variant = "pointwise"
                self.sitok_basis_dim = int(sitok_basis_dim)
                head_out_dim = self.sitok_basis_dim + 1  # weights over basis + bias

            if sitok_hidden_dim == 0:
                self.sitok_head = nn.Linear(self.sitok_embed_dim, head_out_dim, bias=True)
            else:
                self.sitok_head = nn.Sequential(
                    nn.Linear(self.sitok_embed_dim, sitok_hidden_dim),
                    nn.SiLU(),
                    nn.Linear(sitok_hidden_dim, head_out_dim),
                )

            last = self.sitok_head[-1] if isinstance(self.sitok_head, nn.Sequential) else self.sitok_head
            assert isinstance(last, nn.Linear)
            nn.init.trunc_normal_(last.weight, std=0.02)
            nn.init.zeros_(last.bias)
            if self.sitok_variant == "film":
                last.bias.data[0] = 1.0
        elif mode == "cross_attn":
            if self.cross_attn_pool_size <= 0:
                raise ValueError(f"{cross_attn_pool_size=} must be > 0")
            if self.cross_attn_embed_dim <= 0:
                raise ValueError(f"{cross_attn_embed_dim=} must be > 0")
            self.cross_attn_kv_proj = nn.Linear(in_channels, self.cross_attn_embed_dim, bias=use_bias)
            self.cross_attn_q_proj = nn.Linear(self.cross_attn_embed_dim, self.cross_attn_embed_dim, bias=use_bias)
            self.cross_attn_pw_head = nn.Linear(self.cross_attn_embed_dim, in_channels + 1, bias=True)

        # Initialize forward mappings
        self.forward_mappings: dict[str, Callable[..., Tensor]] = {
            "slice": self._slice_forward,
            "interp": self._interp_forward,
            "interp_proj": self._interp_proj_forward,
            "mix": self._mix_forward,
            "sitok_film": self._sitok_film_forward,
            "sitok_pointwise": self._sitok_pointwise_forward,
            "sitok": self._sitok_film_forward,  # backward-compatible alias
            "cross_attn": self._cross_attn_forward,
        }

    def _forward_conv_with_wb(self, x, w, b):
        """Perform convolution with custom weights and bias"""
        return nn.functional.conv2d(
            x,
            w,
            b,
            self.conv.stride,
            self.conv.padding,
            self.conv.dilation,
            self.conv.groups,
        )

    def _slice_forward(self, x: torch.Tensor, out_channels: int) -> torch.Tensor:
        """Forward method for slice mode"""
        assert out_channels <= self.conv.out_channels, (
            "In slice mode, out_channels must be less than or equal to the original out_channels, "
            f"got {out_channels=} > {self.conv.out_channels=}"
        )

        w = self.conv.weight[:out_channels]
        w = _kernel_norm(w, self.kernel_norm, "c_out")
        b = self.conv.bias[:out_channels] if self.conv.bias is not None else None
        return self._forward_conv_with_wb(x, w, b)

    def _interp_forward(
        self,
        x: torch.Tensor,
        out_channels: int,
        w: torch.Tensor | None = None,
        b: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward method for interpolation mode"""
        c_in = x.shape[1]
        if w is None:
            c_out, _, k1, k2 = self.conv.weight.shape
            w = self.conv.weight

        # Interpolate weights from c_out to out_channels
        w = rearrange(w, "c_out c_in k1 k2 -> k1 k2 c_out c_in")
        w = torch.nn.functional.interpolate(w, size=(out_channels, c_in), mode="bicubic", align_corners=False)
        w = rearrange(w, "k1 k2 c_out c_in -> c_out c_in k1 k2")

        # Interpolate bias if present
        b = None
        if self.conv.bias is not None:
            # (b) -> (1,1,b)
            b = torch.nn.functional.interpolate(
                self.conv.bias[None, None],
                size=(out_channels,),
                mode="linear",
                align_corners=False,
            )[0, 0]

        # Apply kernel normalization
        w = _kernel_norm(w, self.kernel_norm, "c_out")
        return self._forward_conv_with_wb(x, w, b)

    def _interp_proj_forward(self, x: torch.Tensor, out_channels: int) -> torch.Tensor:
        """Forward method for interpolation with projection mode"""
        c_out, c_in, k1, k2 = self.conv.weight.shape

        # weight projection - similar to input layer but for output channels
        w = rearrange(self.conv.weight, "c_out c_in k1 k2 -> c_in (k1 k2) c_out")
        lin1, lin2 = self.kernel_proj
        w = lin1(w)  # c_in, (k1*k2), k_hidden
        w = lin2(w)  # c_in, (k1*k2), c_out
        w = rearrange(w, "c_in (k1 k2) c_out -> k1 k2 c_out c_in", k1=k1, k2=k2)

        # Interpolate from c_out to out_channels
        w = torch.nn.functional.interpolate(w, size=(out_channels, c_in), mode="bicubic", align_corners=False)
        w = rearrange(w, "k1 k2 c_out c_in -> c_out c_in k1 k2")

        # Interpolate bias if present
        b = None
        if self.conv.bias is not None:
            b = torch.nn.functional.interpolate(
                self.conv.bias[None, None],
                size=(out_channels,),
                mode="linear",
                align_corners=False,
            )[0, 0]

        # Apply kernel normalization
        w = _kernel_norm(w, self.kernel_norm, "c_out")

        return self._forward_conv_with_wb(x, w, b)

    def _mix_forward(self, x: torch.Tensor, out_channels: int) -> torch.Tensor:
        base_out = self.conv.weight.shape[0]
        if base_out == 0:
            raise ValueError("base_out_channels must be > 0")

        coords = _normalized_channel_coords(out_channels, device=x.device, dtype=x.dtype)
        channel_mean: torch.Tensor | None = None
        channel_dw_pool: torch.Tensor | None = None
        if self.out_router.condition in ("per_channel_mean", "per_channel_mean_dw_pool"):
            channel_mean = x.mean(dim=(2, 3)).mean(dim=1, keepdim=True).repeat(1, out_channels)
        if self.out_router.condition in ("per_channel_dw_pool", "per_channel_mean_dw_pool"):
            if not hasattr(self, "router_dw_kernel"):
                raise RuntimeError("router_dw_kernel is missing while condition requires dw_pool")
            k = int(self.router_dw_kernel.shape[0])
            w_dw = self.router_dw_kernel.to(dtype=x.dtype, device=x.device)[None, None].repeat(x.shape[1], 1, 1, 1)
            dw = F.conv2d(x, w_dw, bias=None, stride=1, padding=k // 2, groups=x.shape[1])
            channel_dw_pool = dw.abs().mean(dim=(2, 3)).mean(dim=1, keepdim=True).repeat(1, out_channels)

        coeff = self.out_router(coords, channel_mean=channel_mean, channel_dw_pool=channel_dw_pool)

        w_base = self.conv.weight
        k1, k2 = w_base.shape[-2:]
        w_base_flat = rearrange(w_base, "c_out c_in k1 k2 -> c_out c_in (k1 k2)")
        if coeff.ndim == 2:
            w_flat = torch.einsum("oip,uo->uip", w_base_flat, coeff)
            w = rearrange(w_flat, "c_out c_in (k1 k2) -> c_out c_in k1 k2", k1=k1, k2=k2)

            b = self.conv.bias
            if b is not None:
                b = torch.einsum("o,uo->u", b, coeff)

            w = _kernel_norm(w, self.kernel_norm, "c_out")
            y = self._forward_conv_with_wb(x, w, b)
            return y

        # Per-sample dynamic weights: (b, out_channels, base_out)
        bsz = x.shape[0]
        w_flat_b = torch.einsum("oip,buo->buip", w_base_flat, coeff)
        w_b = rearrange(w_flat_b, "b c_out c_in (k1 k2) -> b c_out c_in k1 k2", k1=k1, k2=k2)
        w_b_flat = w_b.reshape(bsz * w_b.shape[1], x.shape[1], k1, k2)
        w_b_flat = _kernel_norm(w_b_flat, self.kernel_norm, "c_out")
        w_b = w_b_flat.reshape(bsz, -1, x.shape[1], k1, k2)

        bias = self.conv.bias
        if bias is None:
            bias_b = None
        else:
            bias_b = torch.einsum("o,buo->bu", bias, coeff).reshape(-1)

        x_cf = x.contiguous()
        x_g = x_cf.reshape(1, bsz * x.shape[1], *x.shape[2:])
        w_g = w_b.reshape(bsz * out_channels, x.shape[1], k1, k2)
        y = F.conv2d(  # type: ignore
            x_g,
            w_g,
            bias_b,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=bsz,
        )
        y = y.reshape(bsz, out_channels, *y.shape[2:])
        return y

    def _sitok_film_forward(self, x: torch.Tensor, out_channels: int) -> torch.Tensor:
        """SiTok output (FiLM): shared conv -> base, per-channel (gamma, beta) from channel index embedding."""
        y_base = self._forward_conv_with_wb(x, self.conv.weight, self.conv.bias)
        if out_channels == 1:
            # return _to_conv_channels_last_memformat(y_base)
            return y_base

        emb = _sincos_channel_index_embedding(
            out_channels,
            self.sitok_embed_dim,
            device=y_base.device,
            dtype=y_base.dtype,
            base=self.sitok_embed_base,
        )
        film = self.sitok_head(emb)  # (out_channels, 2)
        gamma = film[:, 0]
        beta = film[:, 1]
        if self.sitok_embed_scale != 1.0:
            gamma = 1.0 + (gamma - 1.0) * self.sitok_embed_scale
            beta = beta * self.sitok_embed_scale

        y = y_base * gamma[None, :, None, None] + beta[None, :, None, None]
        return y

    def _sitok_pointwise_forward(self, x: torch.Tensor, out_channels: int) -> torch.Tensor:
        """
        SiTok output (dynamic pointwise):
          - shared conv generates basis feature map h (B, D, H, W)
          - channel index embedding -> weights over basis (out_channels, D) and bias (out_channels,)
          - y[b,o,h,w] = sum_d h[b,d,h,w] * w[o,d] + b[o]
        """
        h = self._forward_conv_with_wb(x, self.conv.weight, self.conv.bias)  # (b, D, h, w)
        emb = _sincos_channel_index_embedding(
            out_channels,
            self.sitok_embed_dim,
            device=h.device,
            dtype=h.dtype,
            base=self.sitok_embed_base,
        )
        params = self.sitok_head(emb)  # (out_channels, D+1)
        w = params[:, : self.sitok_basis_dim]
        b = params[:, self.sitok_basis_dim]
        if self.sitok_embed_scale != 1.0:
            w = w * self.sitok_embed_scale
            b = b * self.sitok_embed_scale

        y = torch.einsum("bdhw,od->bohw", h, w) + b[None, :, None, None]
        return y

    def _cross_attn_project_kv_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if self.cross_attn_kv_proj is None:
            raise RuntimeError("cross_attn_kv_proj is missing while mode='cross_attn'")
        p = int(self.cross_attn_pool_size)
        x_pool = F.adaptive_avg_pool2d(x, output_size=(p, p))
        kv_tokens = rearrange(x_pool, "b c p1 p2 -> b (p1 p2) c")
        proj_dtype = self.cross_attn_kv_proj.weight.dtype
        return self.cross_attn_kv_proj(kv_tokens.to(dtype=proj_dtype))

    def _cross_attn_output_queries(
        self,
        batch_size: int,
        out_channels: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if self.cross_attn_q_proj is None:
            raise RuntimeError("cross_attn_q_proj is missing while mode='cross_attn'")
        q_dtype = self.cross_attn_q_proj.weight.dtype
        emb = _sincos_channel_index_embedding(
            out_channels,
            self.cross_attn_embed_dim,
            device=device,
            dtype=q_dtype,
        )
        q = self.cross_attn_q_proj(emb)
        return q.unsqueeze(0).expand(batch_size, -1, -1)

    def _cross_attn_dynamic_pointwise(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if self.cross_attn_pw_head is None:
            raise RuntimeError("cross_attn_pw_head is missing while mode='cross_attn'")
        params = self.cross_attn_pw_head(z)
        w_pw = params[..., : h.shape[1]].to(dtype=h.dtype)
        b_pw = params[..., h.shape[1]].to(dtype=h.dtype)
        bsz, _, height, width = h.shape
        h_flat = h.reshape(bsz, h.shape[1], height * width).contiguous()
        y = torch.bmm(w_pw.contiguous(), h_flat)
        y = y.reshape(bsz, w_pw.shape[1], height, width)
        return y + b_pw[:, :, None, None]

    def _cross_attn_forward(self, x: torch.Tensor, out_channels: int) -> torch.Tensor:
        w_shared = _kernel_norm(self.conv.weight, self.kernel_norm, "c_out")
        h = self._forward_conv_with_wb(x, w_shared, self.conv.bias)
        kv = self._cross_attn_project_kv_tokens(h)
        q = self._cross_attn_output_queries(h.shape[0], out_channels, device=h.device)
        z = F.scaled_dot_product_attention(q.unsqueeze(1), kv.unsqueeze(1), kv.unsqueeze(1), dropout_p=0.0)
        return self._cross_attn_dynamic_pointwise(h, z.squeeze(1))

    def forward(self, x: torch.Tensor, out_channels: Optional[int] = None) -> torch.Tensor:
        if out_channels is None:
            out_channels = (
                self.default_out_channels
                if (self.mode.startswith("sitok") or self.mode == "cross_attn")
                else self.conv.out_channels
            )

        if self.mode in ("sitok_film", "sitok_pointwise", "cross_attn"):
            return self.forward_mappings[self.mode](x, out_channels)

        # Native case - if output channels match, use direct convolution
        if out_channels == self.conv.weight.shape[0]:
            w = self.conv.weight
            w = _kernel_norm(w, self.kernel_norm, "c_out")
            return self._forward_conv_with_wb(x, w, self.conv.bias)

        # Adaptive cases - use mapping
        return self.forward_mappings[self.mode](x, out_channels)

    @property
    def weight(self):
        return self.conv.weight

    @safe_init_weights
    def init_weights(self, zero_out=False):
        # init conv
        if zero_out:
            nn.init.zeros_(self.conv.weight)
        else:
            nn.init.xavier_uniform_(self.conv.weight)

        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

        if self.mode == "interp_proj":
            for lin in self.kernel_proj:
                torch.nn.init.trunc_normal_(lin.weight, std=0.02)
                if lin.bias is not None:
                    torch.nn.init.zeros_(lin.bias)
        elif self.mode == "cross_attn":
            assert self.cross_attn_kv_proj is not None
            assert self.cross_attn_q_proj is not None
            assert self.cross_attn_pw_head is not None
            for linear in (self.cross_attn_kv_proj, self.cross_attn_q_proj, self.cross_attn_pw_head):
                torch.nn.init.trunc_normal_(linear.weight, std=0.02)
                if linear.bias is not None:
                    torch.nn.init.zeros_(linear.bias)
        elif self.mode.startswith("sitok"):
            for m in self.sitok_head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            if getattr(self, "sitok_variant", None) == "film":
                last = self.sitok_head[-1] if isinstance(self.sitok_head, nn.Sequential) else self.sitok_head
                assert isinstance(last, nn.Linear)
                last.bias.data[0] = 1.0


class AdaptiveInputLinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mode: AdaptiveLinearMode = "slice",
        *,
        sitok_group_size: int | None = None,
        sitok_reduce: SitokReduceMode = "none",
        sitok_embed_scale: float = 1.0,
        sitok_embed_base: float = 10000.0,
    ) -> None:
        super().__init__()
        if mode in ("slice_lin", "interp_lin"):
            mode = "slice" if mode == "slice_lin" else "interp"

        self.mode = mode
        self.sitok_group_size = sitok_group_size
        self.sitok_reduce = sitok_reduce
        self.sitok_embed_scale = float(sitok_embed_scale)
        self.sitok_embed_base = float(sitok_embed_base)
        self.sitok_reduce_head: nn.Module | None = None

        if mode == "sitok":
            if sitok_group_size is None or sitok_group_size <= 0:
                raise ValueError(f"{sitok_group_size=} must be a positive int when mode='sitok'")
            if sitok_reduce not in ("none", "sum", "mean", "pointwise"):
                raise ValueError(f"Unknown sitok_reduce: {sitok_reduce}")
            if self.sitok_embed_scale < 0:
                raise ValueError(f"{sitok_embed_scale=} must be >= 0")
            if self.sitok_embed_base <= 0:
                raise ValueError(f"{sitok_embed_base=} must be > 0")
            self.linear = nn.Linear(int(sitok_group_size), out_features, bias=bias)
            if sitok_reduce == "pointwise":
                self.sitok_reduce_head = nn.Linear(out_features, out_features, bias=True)
        else:
            self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.forward_mappings: dict[str, Callable[..., Tensor]] = {
            "slice": self._slice_forward,
            "interp": self._interp_forward,
            "sitok": self._sitok_forward,
        }

    def _sitok_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sitok_group_size is None:
            raise RuntimeError("sitok_group_size is missing while mode='sitok'")
        group_size = int(self.sitok_group_size)
        total_in = x.shape[-1]
        if total_in % group_size != 0:
            raise ValueError(f"Expected last dim {total_in} divisible by {group_size=}")
        n_channels = total_in // group_size

        w = self.linear.weight
        b = self.linear.bias

        ch_emb_scaled: Tensor | None = None
        if self.sitok_embed_scale != 0 or self.sitok_reduce == "pointwise":
            ch_emb = _sincos_channel_index_embedding(
                n_channels,
                w.shape[0],
                device=x.device,
                dtype=x.dtype,
                base=self.sitok_embed_base,
            )
            ch_emb_scaled = ch_emb * self.sitok_embed_scale

        x_cg = x.reshape(*x.shape[:-1], n_channels, group_size)

        if self.sitok_reduce == "none":
            y = F.linear(x_cg, w, b)
            if ch_emb_scaled is not None and self.sitok_embed_scale != 0:
                y = y + ch_emb_scaled
            y = y.reshape(*x.shape[:-1], n_channels * w.shape[0])
            return y

        if self.sitok_reduce == "pointwise":
            if self.sitok_reduce_head is None:
                raise RuntimeError("sitok_reduce_head is missing while sitok_reduce='pointwise'")
            if ch_emb_scaled is None:
                raise RuntimeError("ch_emb_scaled is missing while sitok_reduce='pointwise'")
            logits = self.sitok_reduce_head(ch_emb_scaled).float()
            weights = logits.softmax(dim=0).to(dtype=x.dtype)  # (c_in, c_out)
            x_mix = torch.einsum("...cg,cd->...dg", x_cg, weights)

            y = (x_mix * w).sum(dim=-1)
            bias_eff = b
            if self.sitok_embed_scale != 0:
                bias_add = torch.einsum("cd,cd->d", weights.float(), ch_emb_scaled.float()).to(dtype=x.dtype)
                if bias_eff is None:
                    bias_eff = bias_add
                else:
                    bias_eff = bias_eff + bias_add
            if bias_eff is not None:
                y = y + bias_eff
            return y

        if self.sitok_reduce == "sum":
            x_agg = x_cg.sum(dim=-2)
            bias_eff = b
            if bias_eff is not None:
                bias_eff = bias_eff * n_channels
            if ch_emb_scaled is not None and self.sitok_embed_scale != 0:
                emb_agg = ch_emb_scaled.sum(dim=0).to(dtype=x.dtype)
                if bias_eff is None:
                    bias_eff = emb_agg
                else:
                    bias_eff = bias_eff + emb_agg
            return F.linear(x_agg, w, bias_eff)

        if self.sitok_reduce == "mean":
            x_agg = x_cg.mean(dim=-2)
            bias_eff = b
            if ch_emb_scaled is not None and self.sitok_embed_scale != 0:
                emb_agg = ch_emb_scaled.mean(dim=0).to(dtype=x.dtype)
                if bias_eff is None:
                    bias_eff = emb_agg
                else:
                    bias_eff = bias_eff + emb_agg
            return F.linear(x_agg, w, bias_eff)

        raise ValueError(f"Unknown sitok_reduce: {self.sitok_reduce}")

    def _slice_forward(self, x: torch.Tensor) -> torch.Tensor:
        in_features = x.shape[-1]
        w = self.linear.weight[:, :in_features]
        b = self.linear.bias
        return F.linear(x, w, b)

    def _interp_forward(self, x: torch.Tensor) -> torch.Tensor:
        in_features = x.shape[-1]
        out_f, _ = (w := self.linear.weight).shape
        w_i = w.unsqueeze(0).unsqueeze(0).contiguous()  # 1, 1, out_f, in_f
        w_i = torch.nn.functional.interpolate(w_i, size=(out_f, in_features), mode="bicubic", align_corners=False)
        w = w_i.squeeze(0, 1)
        b = self.linear.bias
        return F.linear(x, w, b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "sitok":
            return self.forward_mappings[self.mode](x)

        in_features = x.shape[-1]
        if in_features == self.linear.weight.shape[1]:
            return self.linear(x)

        return self.forward_mappings[self.mode](x)

    def init_weight(self) -> None:
        nn.init.trunc_normal_(self.linear.weight, 0.02)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
        if self.sitok_reduce_head is not None:
            for m in self.sitok_reduce_head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)


class AdaptiveOutputLinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mode: AdaptiveLinearMode = "slice",
        *,
        sitok_group_size: int | None = None,
        sitok_embed_dim: int = 16,
        sitok_hidden_dim: int = 64,
        sitok_basis_dim: int = 32,
        sitok_embed_scale: float = 1.0,
        sitok_embed_base: float = 10000.0,
    ):
        super().__init__()
        if mode == "sitok":
            mode = "sitok_film"

        self.mode = mode
        self.sitok_group_size = sitok_group_size

        self.sitok_embed_dim = int(sitok_embed_dim)
        self.sitok_hidden_dim = int(sitok_hidden_dim)
        self.sitok_basis_dim = int(sitok_basis_dim)
        self.sitok_embed_scale = float(sitok_embed_scale)
        self.sitok_embed_base = float(sitok_embed_base)
        self.sitok_variant: SitokOutputVariant | None = None
        self.sitok_head: nn.Module | None = None

        if mode in ("sitok_film", "sitok_pointwise"):
            if sitok_group_size is None or sitok_group_size <= 0:
                raise ValueError(f"{sitok_group_size=} must be a positive int when mode='sitok_*'")
            if self.sitok_embed_dim <= 0:
                raise ValueError(f"{sitok_embed_dim=} must be > 0")
            if self.sitok_hidden_dim < 0:
                raise ValueError(f"{sitok_hidden_dim=} must be >= 0")
            if self.sitok_embed_scale < 0:
                raise ValueError(f"{sitok_embed_scale=} must be >= 0")
            if self.sitok_embed_base <= 0:
                raise ValueError(f"{sitok_embed_base=} must be > 0")

            group_size_int = int(sitok_group_size)
            if mode == "sitok_film":
                self.sitok_variant = "film"
                self.linear = nn.Linear(in_features, group_size_int, bias=bias)
                head_out_dim = 2 * group_size_int
            else:
                self.sitok_variant = "pointwise"
                if sitok_basis_dim <= 0:
                    raise ValueError(f"{sitok_basis_dim=} must be > 0")
                self.linear = nn.Linear(in_features, int(sitok_basis_dim), bias=bias)
                head_out_dim = group_size_int * (int(sitok_basis_dim) + 1)

            if self.sitok_hidden_dim == 0:
                self.sitok_head = nn.Linear(self.sitok_embed_dim, head_out_dim, bias=True)
            else:
                self.sitok_head = nn.Sequential(
                    nn.Linear(self.sitok_embed_dim, self.sitok_hidden_dim),
                    nn.SiLU(),
                    nn.Linear(self.sitok_hidden_dim, head_out_dim),
                )
        else:
            assert mode in ("slice", "interp"), f'mode must be one of "slice", "interp", "sitok_*"'
            self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.forward_mappings: dict[str, Callable[..., Tensor]] = {
            "slice": self._slice_forward,
            "interp": self._interp_forward,
            "sitok_film": self._sitok_film_forward,
            "sitok_pointwise": self._sitok_pointwise_forward,
            "sitok": self._sitok_film_forward,
        }

    def _sitok_film_forward(self, x: torch.Tensor, out_features: int) -> torch.Tensor:
        if self.sitok_group_size is None:
            raise RuntimeError("sitok_group_size is missing while mode='sitok_*'")
        if self.sitok_head is None:
            raise RuntimeError("sitok_head is missing while mode='sitok_*'")

        group_size = int(self.sitok_group_size)
        if out_features % group_size != 0:
            raise ValueError(f"Expected {out_features=} divisible by {group_size=}")
        n_channels = out_features // group_size

        emb = _sincos_channel_index_embedding(
            n_channels,
            self.sitok_embed_dim,
            device=x.device,
            dtype=x.dtype,
            base=self.sitok_embed_base,
        )
        params = self.sitok_head(emb)

        base = self.linear(x)
        gamma = params[:, :group_size]
        beta = params[:, group_size:]
        if self.sitok_embed_scale != 1.0:
            gamma = 1.0 + (gamma - 1.0) * self.sitok_embed_scale
            beta = beta * self.sitok_embed_scale
        y = base.unsqueeze(-2) * gamma + beta
        return y.reshape(*x.shape[:-1], out_features)

    def _sitok_pointwise_forward(self, x: torch.Tensor, out_features: int) -> torch.Tensor:
        if self.sitok_group_size is None:
            raise RuntimeError("sitok_group_size is missing while mode='sitok_*'")
        if self.sitok_head is None:
            raise RuntimeError("sitok_head is missing while mode='sitok_*'")

        group_size = int(self.sitok_group_size)
        if out_features % group_size != 0:
            raise ValueError(f"Expected {out_features=} divisible by {group_size=}")
        n_channels = out_features // group_size

        emb = _sincos_channel_index_embedding(
            n_channels,
            self.sitok_embed_dim,
            device=x.device,
            dtype=x.dtype,
            base=self.sitok_embed_base,
        )
        params = self.sitok_head(emb)

        basis_dim = int(self.sitok_basis_dim)
        h = self.linear(x)
        w_flat = params[:, : group_size * basis_dim].reshape(n_channels, group_size, basis_dim)
        b = params[:, group_size * basis_dim :].reshape(n_channels, group_size)
        if self.sitok_embed_scale != 1.0:
            w_flat = w_flat * self.sitok_embed_scale
            b = b * self.sitok_embed_scale
        y = torch.einsum("...d,cgd->...cg", h, w_flat) + b
        return y.reshape(*x.shape[:-1], out_features)

    def _slice_forward(self, x: torch.Tensor, out_features: int) -> torch.Tensor:
        w = self.linear.weight[:out_features]
        b = self.linear.bias
        if b is not None:
            b = b[:out_features]
        return F.linear(x, w, b)

    def _interp_forward(self, x: torch.Tensor, out_features: int) -> torch.Tensor:
        w = self.linear.weight
        b = self.linear.bias

        out_f, in_f = w.shape
        w_i = w.unsqueeze(0).unsqueeze(0).contiguous()  # 1, 1, out_f, in_f
        w_i = torch.nn.functional.interpolate(w_i, size=(out_features, in_f), mode="bicubic", align_corners=False)
        w = w_i.squeeze(0, 1)

        if b is not None:
            b = torch.nn.functional.interpolate(
                b[None, :, None].contiguous(),  # (1, out_f, 1)
                size=(out_features,),
                mode="linear",
                align_corners=False,
            )[0, :, 0]
        return F.linear(x, w, b)

    def forward(self, x: torch.Tensor, out_features: int | None = None) -> torch.Tensor:
        if self.mode in ("sitok_film", "sitok_pointwise", "sitok"):
            if out_features is None:
                raise ValueError("AdaptiveOutputLinearLayer(mode='sitok_*') requires explicit out_features")
            return self.forward_mappings[self.mode](x, out_features)

        if out_features is None:
            out_features = self.linear.out_features

        if out_features == self.linear.weight.shape[0]:
            return self.linear(x)
        return self.forward_mappings[self.mode](x, out_features)

    @safe_init_weights
    def init_weights(self, zero_out=False):
        if zero_out:
            nn.init.zeros_(self.linear.weight)
        else:
            nn.init.trunc_normal_(self.linear.weight, 0.02)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
        if self.mode in ("sitok_film", "sitok_pointwise"):
            assert self.sitok_head is not None
            for m in self.sitok_head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            if self.sitok_variant == "film":
                last = self.sitok_head[-1] if isinstance(self.sitok_head, nn.Sequential) else self.sitok_head
                assert isinstance(last, nn.Linear)
                assert self.sitok_group_size is not None
                last.bias.data[: int(self.sitok_group_size)] = 1.0

    @property
    def weight(self):
        """Get the weight tensor of the underlying linear layer."""
        return self.linear.weight

    @property
    def bias(self):
        """Get the bias tensor of the underlying linear layer."""
        return self.linear.bias


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

    # @compile_decorator
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
        if self.act_checkpoint and _can_use_activation_checkpoint(self.training):
            return checkpoint(self._forward_fn, x, use_reentrant=False)
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

        logger.debug(f"[ResnetBlockMoE2D] using token mixer: {token_mixer_type}", "debug")
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
                padding_mode=padding_mode,
            )
        else:
            raise ValueError(
                f"[ResnetBlockMoE2D] Unknown token_mixer_type={token_mixer_type}, "
                "supported types are: ['res_block', 'dico_block', 'convnext']"
            )
        self.moe_prenorm = Normalize(self.out_channels, norm_type=norm_type, num_groups=norm_groups)
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
    def __init__(self, in_channels: int, conv_cls=nn.Conv2d):
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


@compile_decorator
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

        logger.debug(
            f"[Dico block]: in: {in_channels} out: {out_channels} hidden: {hidden_channels} conv type: {conv_type}"
        )

        self.norm = Normalize(in_channels, norm_type=norm_type, num_groups=norm_groups)
        # self.dropout = nn.Dropout(dropout)
        conv_cls = MPConv if conv_type == "mpconv" else nn.Conv2d  # mpconv cannot support the torch.compile

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
            self.norm_ffn = Normalize(out_channels, norm_type=norm_type, num_groups=norm_groups)

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
        if self.act_checkpoint and _can_use_activation_checkpoint(self.training):
            return checkpoint(self.forward_fn, x, use_reentrant=False)

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
        act_type: str | tuple[str, str] = ("silu", "silu"),  # Check: may gelu cause large activation values?
        nin_shortcut_norm: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels = in_channels if out_channels is None else out_channels
        self.norm_type = norm_type = kwargs.get("norm_type", "gn")
        gn_norm_groups = kwargs.get("num_groups", 32)
        self.padding_mode = kwargs.get("padding_mode", "reflect")
        self.layer_idx = kwargs.get("layer_idx", None)

        self.use_dico_cca = kwargs.get("use_dico_cca", False)
        if self.use_dico_cca:
            logger.warning(f"Using dico cca, may cause large memory usage")
        if isinstance(act_type, str):
            act_type = (act_type, act_type)

        self.norm1 = Normalize(in_channels, num_groups=gn_norm_groups, norm_type=norm_type)
        self.act1 = ACT2FN[act_type[0]]
        self.conv1 = nn.Conv2d(
            in_channels, self.out_channels, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode
        )
        self.norm2 = Normalize(out_channels, num_groups=gn_norm_groups, norm_type=norm_type)
        self.act2 = ACT2FN[act_type[1]]
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(
            self.out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode
        )
        self._conv_pad = 1

        if in_channels != out_channels:
            if nin_shortcut_norm:
                self.nin_shortcut = nn.Sequential(
                    Normalize(in_channels, num_groups=gn_norm_groups, norm_type=norm_type),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                )
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.nin_shortcut = nn.Identity()

        self.act_checkpoint = kwargs.get("act_checkpoint", False)
        self.use_residual_factor = use_residual_factor
        if use_residual_factor:
            self.layer_scale = LayerScale2d(out_channels, init_values=1e-2)
        else:
            self.layer_scale = nn.Identity()

        if self.use_dico_cca:
            self.dico_cca = DiCoCompactChannelAttention(out_channels)

    @compile_decorator
    def forward_fn(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
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
        res = x + self.layer_scale(h)
        return res

    def forward(
        self,
        x: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if self.act_checkpoint and _can_use_activation_checkpoint(self.training):
            return checkpoint(self.forward_fn, x, use_reentrant=False)
        return self.forward_fn(x)

    def init_weights(self):
        def _inner(m):
            block_basic_init(m, init_type="lecun_normal")

        self.apply(_inner)


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

    def init_weights(self):
        super().init_weights()
        nn.init.zeros_(self.slots_t_to_mod[-1].weight)
        nn.init.zeros_(self.slots_t_to_mod[-1].bias)

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

        x = self._forward_shortcut(x)

        if self.use_residual_factor:
            h = h * self.residual_factor

        return x + h

    def forward(self, x: torch.Tensor, slots: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.act_checkpoint and _can_use_activation_checkpoint(self.training):
            return checkpoint(self.forward_fn, x, slots, t, use_reentrant=False)  # type: ignore
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

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, padding_mode=padding_mode)
        self.padding_mode = padding_mode
        self.norm = Normalize(dim, norm_type=norm_type, num_groups=num_groups)

        self.pwconv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(hidden_dim, out_dim, kernel_size=1)

        # Layer Scaling
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((out_dim)), requires_grad=True)
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

    # @compile_decorator
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
        if self.act_checkpoint and _can_use_activation_checkpoint(self.training):
            return checkpoint(self.forward_fn, x, use_reentrant=False)  # type: ignore
        return self.forward_fn(x)

    def init_weights(self):
        self.apply(partial(block_basic_init, init_type="lecun_normal"))


# * --- Diffusion blocks --- #


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256, time_scale: float = 1.0):
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
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t * self.time_scale, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq.to(self.mlp[0].weight.dtype))
        return t_emb

    def init_weights(self):
        self.apply(block_basic_init)


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
        self.conv = nn.Conv2d(hidden_size, patch_size * patch_size * out_channels, 3, 1, 1)
        self.t_embd = nn.Linear(t_channels, hidden_size, bias=True)
        self.z_embd = nn.Conv2d(z_channels, hidden_size, 1, 1)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Conv2d(hidden_size, 2 * hidden_size, 1, 1))

    def forward(self, x, z, t):
        c = self.z_embd(F.interpolate(z, size=x.shape[-2:], mode="nearest")) + self.t_embd(t)[..., None, None]
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.norm_final(x) * (1 + scale) + shift
        x = self.conv(x)
        return x

    def init_weights(self):
        self.apply(block_basic_init)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)


if __name__ == "__main__":
    """
    export LOVELY_TENSORS=1
    python -m src.stage1.cosmos.modules.blocks
    """
    adaptive_in_layer = AdaptiveInputConvLayer(
        in_channels=256,
        out_channels=16,
        mode="interp",
        kernel_size=8,
        stride=8,
        padding=0,
        kernel_norm="layernorm",
    )
    x = torch.randn(1, 128, 64, 64)
    y = adaptive_in_layer(x)
    print(y.shape)

    print("-" * 60)

    adaptive_out_layer = AdaptiveOutputConvLayer(
        in_channels=386,
        out_channels=32,
        mode="interp_proj",
        kernel_size=3,
        stride=1,
        padding=1,
    )
    x = torch.randn(1, 386, 64, 64)
    y = adaptive_out_layer(x, out_channels=16)
    print(y.shape)
