import math
from typing import Any, Callable, Optional, no_type_check, Union, Tuple

import torch
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import (
    ConvMlp,
    GluMlp,
    LayerScale,
    LayerScale2d,
    Mlp,
    SwiGLU,
    create_act_layer,
    create_conv2d,
    create_norm,
    create_norm_act_layer,
    create_norm_layer,
    get_act_layer,
    get_norm_layer,
)
from timm.models.convnext import ConvNeXtBlock
from timm.layers import EcaModule, CecaModule
from timm.layers.drop import DropPath
from timm.layers.weight_init import trunc_normal_
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from timm.layers.helpers import to_2tuple, to_3tuple

from src.utilities.network_utils import safe_init_weights, compile_decorator

from .attention import (
    Attention,
    LiteLA,
    LiteMLA,
    NatAttention1d,
    NatAttention2d,
    Qwen3SdpaAttention,
    SoftmaxAttention2D,
)
from .conv import (
    GLUMBConv,
    MBConv,
    MbConvLNBlock,
    MBStem,
    ConvLayer,
    SEModule_,
    CoordAttnModule_,
)
from .cross_attn import CrossAttention, SoftmaxCrossAttention2D
from .functional import ConditionalBlock, ResidualBlock
from .mlp import ClipSwiGLUMlp
from .mlp import SwiGLU as SwiGLU_Custom


def _pick_num_groups(num_channels: int, max_groups: int) -> int:
    for groups in range(min(max_groups, num_channels), 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1


def _resize_like(source: Tensor, target: Tensor) -> Tensor:
    if source.shape[-2:] == target.shape[-2:]:
        return source
    return F.interpolate(source, size=target.shape[-2:], mode="bilinear", align_corners=False)


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        mid_channels: Optional[int] = None,
        expand_ratio: float = 1,
        use_bias: bool | tuple = False,
        norm: tuple[Optional[str], ...] = ("bn2d", "bn2d"),
        act_func: tuple[Optional[str], ...] = ("relu6", None),
        use_ca: bool = False,
    ):
        super().__init__()
        use_bias = to_2tuple(use_bias)
        norm = to_2tuple(norm)
        act_func = to_2tuple(act_func)

        mid_channels = round(in_channels * expand_ratio) if mid_channels is None else mid_channels

        self.conv1 = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.conv2 = ConvLayer(
            mid_channels,
            out_channels,
            kernel_size,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )
        self.ca = nn.Identity()
        if use_ca:
            self.ca = CecaModule(out_channels)

    @compile_decorator
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.ca(x)
        return x


class ResBlockCondition(ResBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        mid_channels: Optional[int] = None,
        expand_ratio: float = 1,
        use_bias: bool | tuple = False,
        use_ca: bool = False,
        norm: tuple[Optional[str], ...] = ("bn2d", "bn2d"),
        act_func: tuple[Optional[str], ...] = ("relu6", None),
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            mid_channels,
            expand_ratio,
            use_bias,
            norm,
            act_func,
        )
        mid_channels = round(in_channels * expand_ratio) if mid_channels is None else mid_channels
        cond_layer = nn.Sequential(
            create_conv2d(cond_channels, mid_channels, kernel_size=1),
            create_norm_act_layer("layernorm2dfp32", mid_channels, act_layer="silu", eps=1e-6),
            create_conv2d(mid_channels, mid_channels * 2, kernel_size=3, groups=mid_channels),
        )
        self.conv2 = ConditionalBlock(
            self.conv2,
            cond_layer,
            condition_types="adaln2",
            process_cond_before="interpolate_as_x",
        )
        self.ca = nn.Identity()
        if use_ca:
            self.ca = CecaModule(channels=out_channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x, cond)
        x = self.ca(x)
        return x


class GLUResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        gate_kernel_size: int = 3,
        stride: int = 1,
        mid_channels: Optional[int] = None,
        expand_ratio: float = 1,
        use_bias: bool | tuple = False,
        norm: tuple[Optional[str], ...] = (None, "trms2d"),
        act_func: tuple[Optional[str], ...] = ("silu", None),
    ):
        super().__init__()
        use_bias = to_3tuple(use_bias)
        norm = to_3tuple(norm)
        act_func = to_3tuple(act_func)

        mid_channels = round(in_channels * expand_ratio) if mid_channels is None else mid_channels

        self.conv1 = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=None,
        )
        self.conv_gate = ConvLayer(
            in_channels,
            mid_channels,
            gate_kernel_size,
            stride,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.conv2 = ConvLayer(
            mid_channels,
            out_channels,
            kernel_size,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    @compile_decorator
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x) * self.conv_gate(x)
        x = self.conv2(x)
        return x


class ChannelAttentionResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        mid_channels: Optional[int] = None,
        expand_ratio: float = 1,
        use_bias: bool | tuple = False,
        norm: tuple[Optional[str], ...] = ("bn2d", "bn2d"),
        act_func: tuple[Optional[str], ...] = ("relu6", None),
        channel_attention_operation: str = "SEModule",
        channel_attention_position: int = 2,
    ):
        super().__init__()
        use_bias = to_2tuple(use_bias)
        norm = to_2tuple(norm)
        act_func = to_2tuple(act_func)

        mid_channels = round(in_channels * expand_ratio) if mid_channels is None else mid_channels

        self.conv1 = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.conv2 = ConvLayer(
            mid_channels,
            out_channels,
            kernel_size,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )
        if channel_attention_operation == "SEModule":
            self.channel_attention = SEModule_(out_channels, reduction=4)
        elif channel_attention_operation == "CoordAttnModule":
            self.channel_attention = CoordAttnModule_(out_channels, out_channels, groups=4)
        elif channel_attention_operation == "CECAModule":
            self.channel_attention = CecaModule(out_channels, 3, 2, 1)
        else:
            raise ValueError(f"channel_attention_operation {channel_attention_operation} is not supported")
        self.channel_attention_position = channel_attention_position

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        if self.channel_attention_position == 1:
            x = self.channel_attention(x)
        x = self.conv2(x)
        if self.channel_attention_position == 2:
            x = self.channel_attention(x)
        return x


class TimeCondResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int | None,
        cond_channels: int | None,
        norm_layer: str = "groupnorm",
        expand_ratio: float = 1.0,
        use_channel_attention: bool = False,
        dropout: float = 0.0,
        num_groups: int = 32,
    ) -> None:
        super().__init__()
        mid_chans = round(in_channels * expand_ratio)
        self.act = nn.SiLU()
        self.norm1 = create_norm_layer(norm_layer, in_channels)
        self.norm2 = create_norm_layer(norm_layer, mid_chans)
        self.conv1 = nn.Conv2d(in_channels, mid_chans, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(mid_chans, out_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.use_ca = use_channel_attention
        if use_channel_attention:
            self.ca = CecaModule(out_channels, 3, 2, 1)
        else:
            self.ca = nn.Identity()

        # FiLM-style conditioning: produces (scale, shift)
        self.time_proj = self.cond_proj = None
        if time_embed_dim is not None:
            self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, mid_chans * 2))
        if cond_channels is not None:
            self.cond_proj = nn.Sequential(nn.SiLU(), nn.Conv2d(cond_channels, mid_chans * 2, kernel_size=1))

        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x: Tensor, temb: Tensor | None, cond: Tensor | None) -> Tensor:
        # ========= Conv1 ==========
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        # ========= Conv2 ==========
        h = self.norm2(h)

        # conditioning
        if self.time_proj is not None:
            if temb is None:
                raise ValueError("temb must be provided when time embedding is enabled")
            # t_params: [B, C*2, 1, 1]
            t_scale, t_shift = self.time_proj(self.act(temb))[:, :, None, None].chunk(2, dim=1)
            h = h * (1 + t_scale) + t_shift
        if self.cond_proj is not None and cond is not None:
            cond = _resize_like(cond, h)
            # c_params: [B, C*2, H, W]
            c_scale, c_shift = self.cond_proj(cond).chunk(2, dim=1)
            h = h * (1 + c_scale) + c_shift
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        # ====== Channel Attention ========
        h = self.ca(h)

        if self.skip is not None:
            x = self.skip(x)

        return x + h


class TimeCondNatBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        time_embed_dim: int | None,
        cond_channels: int | None,
        k_size: int = 8,
        stride: int = 2,
        dilation: int = 2,
        num_heads: int = 8,
        ffn_ratio: float | int = 2,
        qkv_bias: bool = True,
        qk_norm: str = "layernorm2d",
        norm_layer: str = "layernorm2d",
        norm_eps: float = 1e-6,
        drop_path: float = 0.0,
        latent_cond_type: str = "adaln3",
        ms_cond_chans: int | None = None,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        self.requires_cond = cond_channels is not None
        self.cond_channels = cond_channels
        self.time_proj = None if time_embed_dim is None else nn.Linear(time_embed_dim, out_channels)
        if cond_channels is None:
            self.block = Spatial2DNATBlock(
                dim=out_channels,
                k_size=k_size,
                stride=stride,
                dilation=dilation,
                n_heads=num_heads,
                ffn_ratio=ffn_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                norm_layer=norm_layer,
                norm_eps=norm_eps,
                drop_path=drop_path,
            )
        else:
            self.block = Spatial2DNATBlockConditional(
                dim=out_channels,
                cond_chs=cond_channels,
                ms_cond_chans=ms_cond_chans,
                k_size=k_size,
                stride=stride,
                dilation=dilation,
                n_heads=num_heads,
                ffn_ratio=ffn_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                norm_layer=norm_layer,
                norm_eps=norm_eps,
                drop_path=drop_path,
                latent_cond_type=latent_cond_type,
            )

    def forward(self, x: Tensor, temb: Tensor | None, cond: Tensor | None) -> Tensor:
        if self.in_proj is not None:
            x = self.in_proj(x)

        if self.time_proj is None:
            if temb is not None:
                raise ValueError("temb must be None when time embedding is disabled.")
        else:
            if temb is None:
                raise ValueError("temb must be provided when time embedding is enabled.")
            x = x + self.time_proj(temb)[:, :, None, None]

        if self.requires_cond:
            if cond is None:
                raise ValueError("cond must be provided when conditional NAT block is enabled.")
            return self.block(x, cond)

        return self.block(x)


class _TimeCondAdapter(nn.Module):
    def __init__(self, block: nn.Module, *, expect_cond: bool) -> None:
        super().__init__()
        self.block = block
        self.expect_cond = expect_cond

    def forward(self, x: Tensor, temb: Tensor | None, cond: Tensor | None) -> Tensor:
        _ = temb
        if self.expect_cond:
            if cond is None:
                raise ValueError("cond must be provided for conditional blocks.")
            return self.block(x, cond)
        return self.block(x)


class TimeCondConvNextBlock(ConvNeXtBlock):
    def __init__(
        self,
        in_chs: int,
        time_embed_dim: int | None,
        cond_channels: int | None,
        out_chs: Optional[int] = None,
        kernel_size: int = 7,
        stride: int = 1,
        dilation: Union[int, Tuple[int, int]] = (1, 1),
        mlp_ratio: float = 4,
        conv_mlp: bool = False,
        conv_bias: bool = True,
        use_grn: bool = False,
        use_channel_attention: bool = False,
        ls_init_value: Optional[float] = 1e-6,
        act_layer: Union[str, Callable] = "gelu",
        norm_layer: Optional[Callable] = None,
        drop_path: float = 0.0,
        device=None,
        dtype=None,
    ):
        actual_out_chs = out_chs or in_chs

        super().__init__(
            actual_out_chs,
            actual_out_chs,
            kernel_size,
            stride,
            dilation,
            mlp_ratio,
            conv_mlp,
            conv_bias,
            use_grn,
            ls_init_value,
            act_layer,
            norm_layer,
            drop_path,
            device,
            dtype,
        )
        self.in_proj = nn.Conv2d(in_chs, actual_out_chs, kernel_size=1) if in_chs != actual_out_chs else None
        self.use_ca = use_channel_attention
        self.cond_channels = cond_channels
        self.time_embed_dim = time_embed_dim
        self.act = nn.SiLU()

        if use_channel_attention:
            self.ca = CecaModule(actual_out_chs, 3, 2, 1)

        # FiLM-style conditioning: produces (scale, shift)
        self.time_proj = self.cond_proj = None
        if time_embed_dim is not None:
            self.time_proj = nn.Linear(time_embed_dim, actual_out_chs * 2)
        if cond_channels is not None:
            self.cond_proj = nn.Conv2d(cond_channels, actual_out_chs * 2, kernel_size=1)

    def forward(self, x: torch.Tensor, temb: Tensor | None = None, cond: Tensor | None = None) -> torch.Tensor:
        """Forward pass."""
        if self.in_proj is not None:
            x = self.in_proj(x)

        shortcut = x
        x = self.conv_dw(x)

        if self.use_conv_mlp:
            x = self.norm(x)
        else:
            x = x.permute(0, 2, 3, 1)
            x = self.norm(x)
            x = x.permute(0, 3, 1, 2)

        # Scale and Shift components for FiLM
        if self.time_proj is not None:
            if temb is None:
                raise ValueError("temb must be provided when time embedding is enabled")
            # t_params: [B, C*2, 1, 1]
            t_params = self.time_proj(self.act(temb))[:, :, None, None]
            t_scale, t_shift = t_params.chunk(2, dim=1)
            x = x * (1.0 + t_scale) + t_shift

        if self.cond_proj is not None and cond is not None:
            cond = _resize_like(cond, x)
            # c_params: [B, C*2, H, W]
            c_params = self.cond_proj(cond)
            c_scale, c_shift = c_params.chunk(2, dim=1)
            x = x * (1.0 + c_scale) + c_shift

        if self.use_conv_mlp:
            x = self.mlp(x)
        else:
            x = x.permute(0, 2, 3, 1)
            x = self.mlp(x)
            x = x.permute(0, 3, 1, 2)

        # add channel attention
        if self.use_ca:
            x = self.ca(x)

        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))

        x = self.drop_path(x) + self.shortcut(shortcut)
        return x


class Spatial2DNATBlock(nn.Module):
    def __init__(
        self,
        dim,
        k_size=8,
        stride=2,
        dilation=2,
        n_heads=8,
        ffn_ratio: float | int = 2,
        qkv_bias=True,
        qk_norm="layernorm2d",
        norm_layer="layernorm2d",
        norm_eps: float = 1e-6,
        drop_path: float = 0.0,
        **_kwargs,
    ) -> None:
        super().__init__()
        self.grad_checkpointing = False
        qk_norm_layer = get_norm_layer(qk_norm) if qk_norm is not None else None
        norm_layer = get_norm_layer(norm_layer)
        norm_layer_fn: Callable[..., nn.Module] = norm_layer
        self.attn = NatAttention2d(
            dim,
            k_size,
            stride,
            dilation,
            n_heads,
            qkv_bias,
            qk_norm=qk_norm_layer,
            norm_layer=norm_layer,
        )
        self.cffn = ConvMlp(
            dim,
            int(dim * ffn_ratio),
            dim,
            norm_layer=norm_layer,
            act_layer=get_act_layer("silu"),
        )
        self.ls1 = LayerScale2d(dim, 0.1)
        self.ls2 = LayerScale2d(dim, 0.1)

        self.drop_path = DropPath(drop_path)

        self.norm1: nn.Module
        self.norm2: nn.Module
        if norm_eps is None:
            self.norm1 = norm_layer_fn(dim)
            self.norm2 = norm_layer_fn(dim)
        else:
            self.norm1 = norm_layer_fn(dim, eps=norm_eps)
            self.norm2 = norm_layer_fn(dim, eps=norm_eps)

    # @compile_decorator
    def forward(self, x, *args, **kwargs):
        def _closure(x):
            x = x + self.drop_path(self.ls1(self.attn(self.norm1(x))))
            x = x + self.drop_path(self.ls2(self.cffn(self.norm2(x))))
            return x

        # checkpointing
        if self.grad_checkpointing and self.training:
            x = checkpoint(_closure, x, use_reentrant=False)
        else:
            x = _closure(x)
        return x

    @safe_init_weights
    def init_weights(self):
        norms = [get_norm_layer(n) for n in ["layernorm2d", "rmsnorm2d"]]

        def _module_applied(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, tuple(norms)):
                nn.init.constant_(m.weight, 1.0)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_module_applied)


class Spatial2DNATBlockConditional(Spatial2DNATBlock):
    def __init__(
        self,
        dim,
        cond_chs: int,
        ms_cond_chans=None,
        k_size=8,
        stride=2,
        dilation=2,
        n_heads=8,
        ffn_ratio: float | int = 2,
        qkv_bias=True,
        qk_norm="layernorm2d",
        norm_layer="layernorm2d",
        norm_eps: float = 1e-6,
        drop_path: float = 0.0,
        latent_cond_type: str = "adaln3",
        **_kwargs,
    ) -> None:
        super().__init__(
            dim,
            k_size,
            stride,
            dilation,
            n_heads,
            ffn_ratio,
            qkv_bias,
            qk_norm,
            norm_layer,
            norm_eps,
            drop_path,
        )

        if ms_cond_chans is not None:
            self.ms_conv_before_add = create_conv2d(ms_cond_chans, dim, 3)

        mod_factor = 2
        self.latent_cond_type = latent_cond_type
        assert latent_cond_type in ("adaln3", "adaln6"), "latent_cond_type must be adaln3 or adaln6"
        self.modulation = nn.Sequential(
            create_conv2d(cond_chs, dim // mod_factor, 1, bias=True),
            create_norm_act_layer(norm_layer, dim // mod_factor, act_layer="silu", eps=norm_eps),
            create_conv2d(
                dim // mod_factor,
                dim * 3 if latent_cond_type == "adaln3" else dim * 6,
                3,
                stride=1,
                bias=False,
                groups=dim // 2,
            ),
        )

    def _interp_as(self, x, tgt_sz: tuple | torch.Size):
        return F.interpolate(x, size=tgt_sz, mode="bilinear", align_corners=False)

    def _modulate(self, x, scale, shift):
        return x * (scale + 1) + shift

    # @compile_decorator
    def forward(self, x, latent, ms_cond=None, **kwargs):
        def _closure(x, latent, ms_cond=None):
            if hasattr(self, "ms_conv_before_add") and ms_cond is not None:
                ms_cond = self._interp_as(ms_cond, x.shape[2:])
                x = x + self.ms_conv_before_add(ms_cond)

            latent_cond = self._interp_as(latent, x.shape[2:])
            if self.latent_cond_type == "adaln3":
                sh_a, sc_a, g_a = self.modulation(latent_cond).chunk(3, dim=1)
                sh_f, sc_f, g_f = 0.0, 0.0, 1.0
            else:
                sh_a, sc_a, g_a, sh_f, sc_f, g_f = self.modulation(latent_cond).chunk(6, dim=1)

            # NAT attention
            y_attn = self.ls1(self.attn(self._modulate(self.norm1(x), sc_a, sh_a))) * g_a
            x = x + self.drop_path(y_attn)
            # ConvFFN
            y_ffn = self.ls2(self.cffn(self._modulate(self.norm2(x), sc_f, sh_f))) * g_f
            x = x + self.drop_path(y_ffn)
            return x

        # checkpointing
        if self.grad_checkpointing and self.training:
            x = checkpoint(_closure, x, latent, ms_cond, use_reentrant=False)
        else:
            x = _closure(x, latent, ms_cond)
        return x


class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        num_heads=8,
        attn_type="1d",
        qkv_bias=True,
        kernel_size=8,
        stride=2,
        dilation=2,
        qk_norm: type[nn.Module] | None | bool = None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        mlp_norm_layer=None,
        act_layer=nn.SiLU,
        layer_scale_value=1e-5,
        use_layerscale=False,
        mlp_type: str = "swiglu",
        is_causal=False,
    ):
        super().__init__()
        self.grad_checkpointing = False
        self.norm1 = norm_layer(dim)

        # Attention
        if attn_type == "1d":
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm is not None,
                norm_layer=norm_layer,
                attn_drop=attn_drop,
                proj_drop=drop,
                attn_type="sdpa",
                is_causal=is_causal,
            )
        elif attn_type == "1d_gated":
            self.attn = Qwen3SdpaAttention(
                dim,
                num_heads,
                num_heads,
                use_qk_norm=qk_norm is not None,
                qkv_bias=qkv_bias,
                headwise_attn_output_gate=True,
                elementwise_attn_output_gate=False,
            )
        elif attn_type == "nat_1d":
            # NatAttention expects norm layer object, not boolean
            nat_qk_norm = qk_norm if isinstance(qk_norm, (type, nn.Module)) else (norm_layer if qk_norm else None)
            self.attn = NatAttention1d(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=nat_qk_norm,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                norm_layer=norm_layer,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        elif attn_type == "nat_2d":
            # NatAttention expects norm layer object, not boolean
            nat_qk_norm = qk_norm if isinstance(qk_norm, (type, nn.Module)) else (norm_layer if qk_norm else None)
            self.attn = NatAttention2d(
                dim,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=nat_qk_norm,
                norm_layer=norm_layer,
                proj_bias=True,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        self.norm2 = norm_layer(dim)

        # Mlp
        hidden_dim = int(dim * mlp_ratio)
        if mlp_type == "swiglu_custom":
            self.mlp = ClipSwiGLUMlp(
                in_features=dim,
                hidden_features=hidden_dim,
                act_layer=act_layer,
                drop=drop,
                norm_layer=mlp_norm_layer,
                use_conv=attn_type != "1d",
            )
        elif mlp_type == "mlp":
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=hidden_dim,
                act_layer=act_layer,
                drop=drop,
                norm_layer=mlp_norm_layer,
                use_conv=attn_type != "1d",
            )
        elif mlp_type == "glumlp":
            self.mlp = GluMlp(
                in_features=dim,
                hidden_features=hidden_dim,
                act_layer=nn.SiLU,
                drop=drop,
                norm_layer=mlp_norm_layer,
                use_conv=attn_type != "1d",
            )
        elif mlp_type == "swiglu":
            self.mlp = SwiGLU(
                in_features=dim,
                hidden_features=hidden_dim,
                act_layer=nn.SiLU,
                norm_layer=None,
                bias=True,
                drop=drop,
            )
        else:
            raise ValueError(f"mlp_type {mlp_type} is not supported")

        self.ls1 = LayerScale(dim, layer_scale_value) if use_layerscale else nn.Identity()
        self.ls2 = LayerScale(dim, layer_scale_value) if use_layerscale else nn.Identity()

        self.drop_path = DropPath(drop_path)

    def forward_(self, x, mask=None, pe=None):
        x = x + self.drop_path(self.ls1(self.attn(self.norm1(x), mask=mask, rope=pe)))
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        return x

    def forward(self, x, mask=None, pe=None, **kwargs):
        if self.grad_checkpointing and self.training:
            return checkpoint(self.forward_, x, mask, pe, use_reentrant=False)
        else:
            return self.forward_(x, mask, pe)


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        ffn_ratio=4.0,
        ctx_dim=None,
        n_q_heads=8,
        n_kv_heads=None,
        qkv_bias=False,
        qk_norm: str | None = None,
        norm_layer: str = "rmsnorm",
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path: float = 0.0,
        init_values=1e-5,
        self_attn_use_gated=True,
        q_pos: bool = False,
        q_len: int | None = None,
        use_mem_prenorm=False,
        sa_gate_type: str | None = None,
        ca_gate_type: str | None = None,
    ):
        super().__init__()
        self.grad_checkpoint = False
        self.q_pos = q_pos
        if self.q_pos:
            assert q_len is not None, "q_len must be specified if q_pos is True"
            self.q_pos_embed = nn.Parameter(torch.zeros(1, q_len, dim))
        if self_attn_use_gated:
            self.self_attention = Qwen3SdpaAttention(
                dim,
                n_q_heads,
                n_kv_heads or n_q_heads,
                use_qk_norm=qk_norm is not None,
                qkv_bias=qkv_bias,
                headwise_attn_output_gate=sa_gate_type == "head_wise",
                elementwise_attn_output_gate=sa_gate_type == "element_wise",
            )
        else:
            self.self_attention = nn.MultiheadAttention(
                dim, n_q_heads, dropout=attn_drop, bias=qkv_bias, batch_first=True
            )
        self.cross_attention = CrossAttention(
            dim,
            ctx_dim,
            n_q_heads,
            n_kv_heads,
            qk_norm,
            qkv_bias,
            attn_drop,
            proj_drop,
            gate_type=ca_gate_type,
        )
        self.ffn = ClipSwiGLUMlp(dim, int(ffn_ratio * dim), proj_drop, norm_layer=get_norm_layer(norm_layer))

        self.prenorm_q = create_norm_layer(norm_layer, dim)
        self.prenorm_kv = create_norm_layer(norm_layer, dim) if use_mem_prenorm else nn.Identity()

        self.ls_sa = LayerScale(dim, init_values=init_values)
        self.ls_ca = LayerScale(dim, init_values=0.8)
        self.ls_ffn = LayerScale(dim, init_values=init_values)

        self.drop_path = DropPath(drop_path)

    def _with_pos_embed(self, x, pos: Tensor | None = None):
        if pos is not None:
            return x + pos
        else:
            return x

    def forward(self, xq, mem, rope=None):
        def closure_(xq, mem):
            xq = self.prenorm_q(xq)
            mem = self.prenorm_kv(mem)

            sa_x = self.self_attention(xq, xq, xq, is_causal=False, need_weights=False)
            if isinstance(sa_x, tuple):
                sa_x = sa_x[0]
            x = self.drop_path(self.ls_sa(sa_x)) + xq

            ca_x = self.cross_attention(x, mem, None)  # do not apply rope in cross-attention # fmt: skip
            x = self.drop_path(self.ls_ca(ca_x)) + x

            x = self.drop_path(self.ls_ffn(self.ffn(x))) + x

            return x

        if self.grad_checkpoint and self.training:
            x = checkpoint(closure_, xq, mem, use_reentrant=False)
        else:
            x = closure_(xq, mem)

        return x


class LiteLA_GLUMB_Block(nn.Module):
    def __init__(
        self,
        hidden_size,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        ffn_drop: float = 0.0,
        qk_norm: bool = True,
        mlp_acts=("silu", "silu", None),
        linear_head_dim: int = 32,
        norm_type: str = "flashrmsnorm",
        mlp_type="glu_mb",
    ):
        super().__init__()
        self.grad_checkpointing = False
        from .mlp import GLUMBConvMlp

        self.hidden_size = hidden_size
        self.norm1 = create_norm_layer(norm_type, hidden_size, eps=1e-6)
        self_num_heads = hidden_size // linear_head_dim
        self.attn = LiteLA(hidden_size, hidden_size, heads=self_num_heads, eps=1e-8, qk_norm=qk_norm)
        self.norm2 = create_norm_layer(norm_type, hidden_size, eps=1e-6)
        if mlp_type == "glu_mb":
            self.mlp = GLUMBConvMlp(
                in_channels=hidden_size,
                out_channels=hidden_size,
                expand_ratio=mlp_ratio,
                use_bias=(True, True, False),
                norm=(None, None, None),
                act_func=mlp_acts,
            )
        elif mlp_type == "swiglu":
            self.mlp = ClipSwiGLUMlp(
                in_features=hidden_size,
                hidden_features=int(hidden_size * mlp_ratio),
                out_features=hidden_size,
                norm_layer=get_norm_layer(norm_type),
                drop=ffn_drop,
                use_conv=False,
            )
        else:  # mlp
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=int(hidden_size * mlp_ratio),
                out_features=hidden_size,
                norm_layer=get_norm_layer(norm_type),
                drop=ffn_drop,
                use_conv=False,
            )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

    @compile_decorator
    def forward(self, x, mask=None, HW=None, pe=None, **kwargs):
        def _closure(x, HW):
            x = x + self.drop_path(self.attn(self.norm1(x), HW=HW))
            x = x + self.drop_path(self.mlp(self.norm2(x), HW=HW))
            return x

        if self.grad_checkpointing and self.training:
            return checkpoint(_closure, x, HW, use_reentrant=False)
        else:
            return _closure(x, HW)


class GhostBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        ratio: int = 2,
        dw_kernel_size: int = 3,
        stride: int = 1,
        use_bias: bool = False,
        norm: Optional[str] = "bn2d",
        act_func: Optional[str] = "relu6",
        cond_channels: int | None = None,
    ) -> None:
        super().__init__()
        if ratio < 1:
            raise ValueError("ratio must be >= 1.")

        self.out_channels = out_channels
        primary_channels = int(math.ceil(out_channels / ratio))
        cheap_channels = out_channels - primary_channels

        self.primary_conv = ConvLayer(
            in_channels,
            primary_channels,
            kernel_size=kernel_size,
            stride=stride,
            use_bias=use_bias,
            norm=norm,
            act_func=act_func,
        )

        self.cheap_conv = None
        if cheap_channels > 0:
            self.cheap_conv = ConvLayer(
                primary_channels,
                cheap_channels,
                kernel_size=dw_kernel_size,
                stride=1,
                groups=primary_channels,
                use_bias=use_bias,
                norm=norm,
                act_func=act_func,
            )

        self.cond = None
        if cond_channels is not None:
            cond_layer = nn.Sequential(
                create_conv2d(cond_channels, out_channels, kernel_size=1),
                create_norm_act_layer("layernorm2dfp32", out_channels, act_layer="silu", eps=1e-6),
                create_conv2d(out_channels, out_channels * 2, kernel_size=3, groups=out_channels),
            )
            self.cond = ConditionalBlock(
                main=nn.Identity(),
                condition_module=cond_layer,
                condition_types="adaln2",
                process_cond_before="interpolate_as_x",
            )

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        x_primary = self.primary_conv(x)
        if self.cheap_conv is not None:
            x_cheap = self.cheap_conv(x_primary)
            x_out = torch.cat([x_primary, x_cheap], dim=1)
        else:
            x_out = x_primary

        x_out = x_out[:, : self.out_channels, :, :]
        if self.cond is not None and cond is not None:
            x_out = self.cond(x_out, cond)
        return x_out


IdentityLayer = nn.Identity  # alias


class EfficientViTBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        heads_ratio: float = 1.0,
        dim=32,
        expand_ratio: float = 4,
        scales: tuple[int, ...] = (5,),
        norm: str = "bn2d",
        act_func: str = "hswish",
        context_module: str = "LiteMLA",
        local_module: str = "MBConv",
        norm_qk: bool = False,
        **kwargs,
    ):
        super(EfficientViTBlock, self).__init__()

        # Context Module
        # keep the output channels same as input channels
        if context_module == "LiteMLA":
            self.context_module = ResidualBlock(
                LiteMLA(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    heads_ratio=heads_ratio,
                    dim=dim,
                    norm=(None, norm),
                    scales=scales,
                    norm_qk=norm_qk,
                ),
                IdentityLayer(),
            )
        elif context_module == "SoftmaxAttention":
            self.context_module = ResidualBlock(
                SoftmaxAttention2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    heads_ratio=heads_ratio,
                    dim=dim,
                    norm=(None, norm),
                ),
                IdentityLayer(),
            )
        elif context_module == "Spatial2DNAT":
            # Spatial2DNAT has residual already
            self.context_module = Spatial2DNATBlock(
                dim=in_channels,
                qk_norm=None if not norm_qk else "layernorm2d",
                **kwargs,
            )
        else:
            raise ValueError(f"context_module {context_module} is not supported")

        if local_module == "MBConv":
            self.local_module = ResidualBlock(
                MBConv(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    expand_ratio=expand_ratio,
                    use_bias=(True, True, False),
                    norm=(None, None, norm),
                    act_func=(act_func, act_func, None),
                ),
                IdentityLayer(),
            )
        elif local_module == "GLUMBConv":
            self.local_module = ResidualBlock(
                GLUMBConv(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    expand_ratio=expand_ratio,
                    use_bias=(True, True, False),
                    norm=(None, None, norm),
                    act_func=(act_func, act_func, None),
                ),
                IdentityLayer(),
            )
        else:
            raise NotImplementedError(f"local_module {local_module} is not supported")

        out_channels = out_channels or in_channels
        self.proj = nn.Conv2d(in_channels, out_channels, 1)

    # @compile_decorator
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.context_module(x)
        x = self.local_module(x)
        return self.proj(x)


class TTT(nn.Module):
    r"""Test-Time Training block for ViT^3 model.
        - https://arxiv.org/abs/2512.01643

    This block implements test-time inner training of two parallel sub-modules:
        1. Simplified SwiGLU inner module, i.e., SwiGLU with identity output layer
        2. 3x3 depth-wise convolution (3x3dwc) inner module

    Note:
        The TTT inner loss is a per-head / per-sample vector-valued loss (shape [B, num_heads]).
        The torch.autograd.backward only supports scalar losses, so here we implement a hand-derived
        backward (closed-form gradient expressions) that directly computes parameter gradients.
        Alternative efficient implementations are welcome and appreciated.

    Args:
        dim (int): Number of input channels
        num_heads (int): Number of attention heads
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, num_heads, qkv_bias=True, **kwargs):
        super().__init__()
        head_dim = dim // num_heads
        self.dim = dim
        self.num_heads = num_heads

        self.qkv = nn.Linear(dim, dim * 3 + head_dim * 3, bias=qkv_bias)
        self.w1 = nn.Parameter(torch.zeros(1, self.num_heads, head_dim, head_dim))
        self.w2 = nn.Parameter(torch.zeros(1, self.num_heads, head_dim, head_dim))
        self.w3 = nn.Parameter(torch.zeros(head_dim, 1, 3, 3))
        trunc_normal_(self.w1, std=0.02)
        trunc_normal_(self.w2, std=0.02)
        trunc_normal_(self.w3, std=0.02)
        self.proj = nn.Linear(dim + head_dim, dim)

        equivalent_head_dim = 9
        self.scale = equivalent_head_dim**-0.5
        # The equivalent head_dim of 3x3dwc branch is 1x(3x3)=9 (1 channel, 3x3 kernel)
        # We used this equivalent_head_dim to compute self.scale in our earlier experiments
        # Using self.scale=head_dim**-0.5 (head_dim of simplified SwiGLU branch) leads to similar performance

    def inner_train_simplified_swiglu(self, k, v, w1, w2, lr=1.0):
        """
        Args:
            k (torch.Tensor): Key tensor of shape [B, num_heads, N, head_dim]
            v (torch.Tensor): Value tensor of shape [B, num_heads, N, head_dim]
            w1 (torch.Tensor): First weight matrix of shape [1, num_heads, head_dim, head_dim]
            w2 (torch.Tensor): Second weight matrix of shape [1, num_heads, head_dim, head_dim]
            lr (float, optional): Learning rate for inner-loop update. Default: 1.0

        Returns:
            tuple: Updated w1 and w2
        """
        # --- Forward ---
        z1 = k @ w1
        z2 = k @ w2
        sig = F.sigmoid(z2)
        a = z2 * sig
        # v_hat = a
        # l = (v_hat * v).sum(dim=3).mean(dim=2) * self.scale
        # Notably, v_hat and l are not computed here because
        # they are unnecessary for deriving the gradient expression below.
        # We directly compute e = dl/dv_hat for the backward pass.

        # --- Backward ---
        e = -v / float(v.shape[2]) * self.scale
        g1 = k.transpose(-2, -1) @ (e * a)
        g2 = k.transpose(-2, -1) @ (e * z1 * (sig * (1.0 + z2 * (1.0 - sig))))

        # --- Clip gradient (for stability) ---
        g1 = g1 / (g1.norm(dim=-2, keepdim=True) + 1.0)
        g2 = g2 / (g2.norm(dim=-2, keepdim=True) + 1.0)

        # --- Step ---
        w1, w2 = w1 - lr * g1, w2 - lr * g2
        return w1, w2

    def inner_train_3x3dwc(self, k, v, w, lr=1.0, implementation="prod"):
        """
        Args:
            k (torch.Tensor): Spatial key tensor of shape [B, C, H, W]
            v (torch.Tensor): Spatial value tensor of shape [B, C, H, W]
            w (torch.Tensor): 3x3 convolution weights of shape [C, 1, 3, 3]
            lr (float, optional): Learning rate for inner-loop update. Default: 1.0
            implementation (str, optional): Implementation method, 'conv' or 'prod'. Default: 'prod'

        Returns:
            torch.Tensor: Updated convolution weights
        """
        # --- Forward ---
        # v_hat = F.conv2d(k, w, padding=1, groups=C)
        # l = - (v_hat * v).mean(dim=[-2, -1]) * self.scale
        # Notably, v_hat and l are not computed here because
        # they are unnecessary for deriving the gradient expression below.
        # We directly compute e = dl/dv_hat for the backward pass.

        # --- Backward ---
        # Two equivalent implementations. The 'prod' implementation appears to be slightly faster
        B, C, H, W = k.shape
        e = -v / float(v.shape[2] * v.shape[3]) * self.scale
        if implementation == "conv":
            g = F.conv2d(k.reshape(1, B * C, H, W), e.reshape(B * C, 1, H, W), padding=1, groups=B * C)
            g = g.transpose(0, 1)
        elif implementation == "prod":
            k = F.pad(k, (1, 1, 1, 1))
            outs = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ys = 1 + dy
                    xs = 1 + dx
                    dot = (k[:, :, ys : ys + H, xs : xs + W] * e).sum(dim=(-2, -1))
                    outs.append(dot)
            g = torch.stack(outs, dim=-1).reshape(B * C, 1, 3, 3)
        else:
            raise NotImplementedError

        # --- Clip gradient (for stability) ---
        g = g / (g.norm(dim=[-2, -1], keepdim=True) + 1.0)

        # --- Step ---
        w = w.repeat(B, 1, 1, 1) - lr * g
        return w

    def forward(self, x, h, w, rope=None):
        """
        Args:
            x (torch.Tensor): Input features with shape of (B, N, C)
            h (int): Feature map height
            w (int): Feature map width
            rope (nn.Module, optional): Rotary Position Embedding
        """
        b, n, c = x.shape
        d = c // self.num_heads

        # Prepare q/k/v
        q1, k1, v1, q2, k2, v2 = torch.split(self.qkv(x), [c, c, c, d, d, d], dim=-1)
        if rope is not None:
            q1 = rope(q1.reshape(b, h, w, c)).reshape(b, n, self.num_heads, d).transpose(1, 2)
            k1 = rope(k1.reshape(b, h, w, c)).reshape(b, n, self.num_heads, d).transpose(1, 2)
        else:
            q1 = q1.reshape(b, n, self.num_heads, d).transpose(1, 2)
            k1 = k1.reshape(b, n, self.num_heads, d).transpose(1, 2)
        v1 = v1.reshape(b, n, self.num_heads, d).transpose(1, 2)
        q2 = q2.reshape(b, h, w, d).permute(0, 3, 1, 2)
        k2 = k2.reshape(b, h, w, d).permute(0, 3, 1, 2)
        v2 = v2.reshape(b, h, w, d).permute(0, 3, 1, 2)

        # Inner training using (k, v)
        w1, w2 = self.inner_train_simplified_swiglu(k1, v1, self.w1, self.w2)
        w3 = self.inner_train_3x3dwc(k2, v2, self.w3, implementation="prod")

        # Apply updated inner module to q
        x1 = (q1 @ w1) * F.silu(q1 @ w2)
        x1 = x1.transpose(1, 2).reshape(b, n, c)
        x2 = F.conv2d(q2.reshape(1, b * d, h, w), w3, padding=1, groups=b * d)
        x2 = x2.reshape(b, d, n).transpose(1, 2)

        # Output proj
        x = torch.cat([x1, x2], dim=-1)
        x = self.proj(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_heads={self.num_heads}"


# *==============================================================
# * Interface
# *==============================================================


def _create_conv_out_blk(in_channels: int, out_channels: int):
    if in_channels == out_channels:
        return nn.Identity()
    else:
        return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)


def build_spatial_block(
    block_type: str,
    *,
    in_channels: int,
    out_channels: int,
    cond_channels: int | None,
    use_time_block: bool,
    time_embed_dim: int | None,
    dropout: float,
    num_groups: int,
    resblock_cfg: dict[str, Any] | None = None,
    nat_cfg: dict[str, Any] | None = None,
    convnext_cfg: dict[str, Any] | None = None,
) -> nn.Module:
    block_key = block_type.lower()

    # cfgs
    resblock_cfg = resblock_cfg or {}
    nat_cfg = nat_cfg or {}
    convnext_cfg = convnext_cfg or {}

    if block_key in {"resblock", "res"}:
        resblock_norm_layer = str(resblock_cfg.get("norm_layer", "layernorm2dfp32"))
        resblock_act_layer = str(resblock_cfg.get("act_layer", "relu6"))
        use_ca = bool(resblock_cfg.get("use_ca", False))
        expand_ratio = float(resblock_cfg.get("expand_ratio", 2))
        if use_time_block:
            return TimeCondResBlock(
                in_channels,
                out_channels,
                time_embed_dim=time_embed_dim,
                cond_channels=cond_channels,
                dropout=dropout,
                num_groups=num_groups,
                use_channel_attention=use_ca,
                norm_layer=resblock_norm_layer,
                expand_ratio=expand_ratio,
            )
        else:
            base_block = build_block(
                f"ResBlock@{expand_ratio}",
                in_channels,
                out_channels,
                norm=resblock_norm_layer,
                act=resblock_act_layer,
                cond_channels=cond_channels,
                use_ca=use_ca,
            )
            return _TimeCondAdapter(base_block, expect_cond=cond_channels is not None)
    elif block_key in {"natblock", "nat"}:
        nat_k_size = int(nat_cfg.get("k_size", 8))
        nat_stride = int(nat_cfg.get("stride", 2))
        nat_dilation = int(nat_cfg.get("dilation", 2))
        nat_num_heads = int(nat_cfg.get("num_heads", 8))
        nat_ffn_ratio = nat_cfg.get("ffn_ratio", 2)
        nat_qkv_bias = bool(nat_cfg.get("qkv_bias", True))
        nat_qk_norm = str(nat_cfg.get("qk_norm", "layernorm2dfp32"))
        nat_norm_layer = str(nat_cfg.get("norm_layer", "layernorm2dfp32"))
        nat_norm_eps = float(nat_cfg.get("norm_eps", 1e-6))
        nat_drop_path = float(nat_cfg.get("drop_path", 0.0))
        nat_latent_cond_type = str(nat_cfg.get("latent_cond_type", "adaln3"))
        nat_ms_cond_chans = nat_cfg.get("ms_cond_chans", None)
        if nat_ms_cond_chans is not None:
            nat_ms_cond_chans = int(nat_ms_cond_chans)
        return TimeCondNatBlock(
            in_channels,
            out_channels,
            time_embed_dim=time_embed_dim,
            cond_channels=cond_channels,
            k_size=nat_k_size,
            stride=nat_stride,
            dilation=nat_dilation,
            num_heads=nat_num_heads,
            ffn_ratio=nat_ffn_ratio,
            qkv_bias=nat_qkv_bias,
            qk_norm=nat_qk_norm,
            norm_layer=nat_norm_layer,
            norm_eps=nat_norm_eps,
            drop_path=nat_drop_path,
            latent_cond_type=nat_latent_cond_type,
            ms_cond_chans=nat_ms_cond_chans,
        )
    elif block_key in {"convnext", "convnextblock"}:
        convnext_kernel_size = int(convnext_cfg.get("kernel_size", 7))
        convnext_mlp_ratio = float(convnext_cfg.get("mlp_ratio", 4.0))
        convnext_use_grn = bool(convnext_cfg.get("use_grn", False))
        convnext_use_ca = bool(convnext_cfg.get("use_channel_attention", False))
        convnext_drop_path = float(convnext_cfg.get("drop_path", 0.0))
        convnext_ls_init = convnext_cfg.get("ls_init_value", 1e-6)
        convnext_conv_mlp = bool(convnext_cfg.get("conv_mlp", False))
        convnext_norm_layer = convnext_cfg.get("norm_layer", None)
        convnext_act_layer = str(convnext_cfg.get("act_layer", "gelu"))

        return TimeCondConvNextBlock(
            in_chs=in_channels,
            time_embed_dim=time_embed_dim if use_time_block else None,
            cond_channels=cond_channels,
            out_chs=out_channels,
            kernel_size=convnext_kernel_size,
            stride=1,
            mlp_ratio=convnext_mlp_ratio,
            conv_mlp=convnext_conv_mlp,
            use_grn=convnext_use_grn,
            use_channel_attention=convnext_use_ca,
            ls_init_value=convnext_ls_init,
            act_layer=convnext_act_layer,
            norm_layer=convnext_norm_layer,
            drop_path=convnext_drop_path,
        )
    else:
        raise ValueError(f"Unsupported block_type: {block_type}")


def build_block(
    block_type: str,
    in_channels: int,
    out_channels: int,
    norm: Optional[str],
    act: Optional[str],
    cond_channels: Optional[int] = None,
    **block_kwargs,
) -> nn.Module:
    """
    Composition of ResidualBlock with different main and shortcut blocks.
    """
    cfg = block_type.split("@")
    block_name = cfg[0]
    if block_name == "ResBlock":
        expand_ratio = int(cfg[1]) if len(cfg) > 1 else 1
        if cond_channels is not None:
            # ResBlock@2
            main_block = ResBlockCondition(
                in_channels=in_channels,
                out_channels=out_channels,
                cond_channels=cond_channels,
                kernel_size=3,
                stride=1,
                use_bias=(True, False),
                norm=(None, norm),
                act_func=(act, None),
                expand_ratio=expand_ratio,
                use_ca=block_kwargs.get("use_ca", False),
            )
        else:
            main_block = ResBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                use_bias=(True, False),
                norm=(None, norm),
                act_func=(act, None),
                expand_ratio=expand_ratio,
                use_ca=block_kwargs.get("use_ca", False),
            )
        block = ResidualBlock(
            main_block,
            _create_conv_out_blk(in_channels, out_channels),
        )
    elif block_name == "GLUResBlock":
        # GLUResBlock@3
        main_block = GLUResBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            gate_kernel_size=int(cfg[1]),
            stride=1,
            use_bias=(True, False),
            norm=(None, norm),
            act_func=(act, None),
        )
        block = ResidualBlock(
            main_block,
            _create_conv_out_blk(in_channels, out_channels),
        )
    elif block_name == "ChannelAttentionResBlock":
        # ChannelAttentionResBlock@SEModule@2
        main_block = ChannelAttentionResBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            use_bias=(True, False),
            norm=(None, norm),
            act_func=(act, None),
            channel_attention_operation=cfg[1],
            channel_attention_position=int(cfg[2]) if len(cfg) > 2 else 2,
        )
        block = ResidualBlock(
            main_block,
            _create_conv_out_blk(in_channels, out_channels),
        )
    elif block_name == "GLUMBConv":
        # GLUMBConv@4
        main_block = GLUMBConv(
            in_channels=in_channels,
            out_channels=out_channels,
            expand_ratio=float(cfg[1]),
            use_bias=(True, True, False),
            norm=(None, None, norm),
            act_func=(act, act, None),
        )
        block = ResidualBlock(
            main_block,
            _create_conv_out_blk(in_channels, out_channels),
        )
    elif block_name == "EViTGLU":
        # assert in_channels == out_channels
        norm_name = norm or "bn2d"
        act_name = act or "hswish"
        block = EfficientViTBlock(
            in_channels,
            out_channels,
            norm=norm_name,
            act_func=act_name,
            local_module="GLUMBConv",
            scales=(),
        )
    elif block_name == "EViTNormQKGLU":
        # assert in_channels == out_channels
        norm_name = norm or "bn2d"
        act_name = act or "hswish"
        block = EfficientViTBlock(
            in_channels,
            out_channels,
            norm=norm_name,
            act_func=act_name,
            local_module="GLUMBConv",
            scales=(),
            norm_qk=True,
        )
    elif block_name == "EViTS5GLU":
        # assert in_channels == out_channels
        norm_name = norm or "bn2d"
        act_name = act or "hswish"
        block = EfficientViTBlock(
            in_channels,
            out_channels,
            norm=norm_name,
            act_func=act_name,
            local_module="GLUMBConv",
            scales=(5,),
        )
    elif block_name == "ViTGLU":
        # assert in_channels == out_channels
        norm_name = norm or "bn2d"
        act_name = act or "hswish"
        block = EfficientViTBlock(
            in_channels,
            out_channels,
            norm=norm_name,
            act_func=act_name,
            context_module="SoftmaxAttention",
            local_module="GLUMBConv",
        )
    elif block_name == "ViTNAT":
        # assert in_channels == out_channels
        # ViTNAT@8@1@1@8@4
        if len(cfg) == 1:
            # k_size, stride, dilation, n_heads, ffn_ratio
            cfg = ["ViTNAT", "8", "1", "1", "8", "4"]  # defaults
        norm_name = norm or "bn2d"
        act_name = act or "hswish"
        block = EfficientViTBlock(
            in_channels,
            out_channels,
            norm=norm_name,
            act_func=act_name,
            context_module="Spatial2DNAT",
            local_module="GLUMBConv",
            k_size=int(cfg[1]),
            stride=int(cfg[2]),
            dilation=int(cfg[3]),
            n_heads=int(cfg[4]),
            ffn_ratio=int(cfg[5]),
        )
    elif block_name == "ConvNext":
        # ConvNext@7@4@1
        block = TimeCondConvNextBlock(
            in_chs=in_channels,
            time_embed_dim=None,
            cond_channels=cond_channels,
            out_chs=out_channels,
            kernel_size=int(cfg[1]),
            stride=1,
            mlp_ratio=float(cfg[2]),
            conv_mlp=False,
            use_grn=bool(int(cfg[3])),
        )
    elif block_name == "SoftmaxCrossAttention":
        block = SoftmaxCrossAttention2D(
            q_in_channels=in_channels,
            kv_in_channels=int(cfg[1]),
            out_channels=out_channels,
            norm=(None, norm),
        )

    elif block_name == "GhostBlock":
        ratio = int(cfg[1]) if len(cfg) > 1 else 2
        kernel_size = int(cfg[2]) if len(cfg) > 2 else 1
        dw_kernel_size = int(cfg[3]) if len(cfg) > 3 else 3
        main_block = GhostBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            ratio=ratio,
            dw_kernel_size=dw_kernel_size,
            norm=norm,
            act_func=act,
            cond_channels=cond_channels,
        )
        block = ResidualBlock(
            main_block,
            _create_conv_out_blk(in_channels, out_channels),
        )
    else:
        raise ValueError(f"block_name {block_name} is not supported")
    return block


if __name__ == "__main__":
    # Test the build block function
    ...
