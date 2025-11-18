from typing import Any, Callable, Optional, no_type_check

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
from timm.layers.drop import DropPath
from timm.models.convnext import ConvNeXtBlock
from torch import Tensor
from torch.utils.checkpoint import checkpoint

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
    ChannelAttentionResBlock,
    GLUMBConv,
    GLUResBlock,
    MBConv,
    MbConvLNBlock,
    MBStem,
    ResBlock,
    ResBlockCondition,
)
from .cross_attn import CrossAttention, SoftmaxCrossAttention2D
from .functional import ConditionalBlock, ResidualBlock
from .mlp import ClipSwiGLUMlp
from .mlp import SwiGLU as SwiGLU_Custom

# Attentions and Blocks


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
            nat_qk_norm = (
                qk_norm
                if isinstance(qk_norm, (type, nn.Module))
                else (norm_layer if qk_norm else None)
            )
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
            nat_qk_norm = (
                qk_norm
                if isinstance(qk_norm, (type, nn.Module))
                else (norm_layer if qk_norm else None)
            )
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

        self.ls1 = (
            LayerScale(dim, layer_scale_value) if use_layerscale else nn.Identity()
        )
        self.ls2 = (
            LayerScale(dim, layer_scale_value) if use_layerscale else nn.Identity()
        )

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


class MbConvStages(nn.Module):
    """MobileConv for stage 1 and stage 2 of ViTamin"""

    def __init__(
        self,
        in_chans: int,
        stem_width: int,
        embed_dim: list[int],
        depths: list[int],
        cond_width: int,
        stride: int = 1,
        drop_path: float = 0.0,
        kernel_size: int = 3,
        norm_layer: str = "layernorm2d",
        norm_eps: float = 1e-6,
        act_layer: str = "gelu",
        expand_ratio: float = 4.0,
        **kwargs,
    ):
        super().__init__()
        self.stem = MBStem(
            in_chs=in_chans,
            out_chs=stem_width,
        )
        stages = {}
        self.num_stages = len(embed_dim)
        for s, dim in enumerate(embed_dim):  # stage
            stage_in_chs = embed_dim[s - 1] if s > 0 else stem_width
            blocks = [
                MbConvLNBlock(
                    in_chs=stage_in_chs if d == 0 else dim,
                    out_chs=dim,
                    cond_chs=stem_width,
                    stride=stride,  # 2 if d == 0 else 1,
                    drop_path=drop_path,
                    norm_layer=norm_layer,
                    norm_eps=norm_eps,
                    act_layer=act_layer,
                    expand_ratio=expand_ratio,
                )
                for d in range(depths[s])
            ]
            stages[f"stage_{s}"] = nn.ModuleList(blocks)
        self.stages = nn.ModuleDict(stages)

    def forward(self, x, cond=None):
        x = self.stem(x)

        # interpolate the condition
        if cond is not None:
            cond = torch.nn.functional.interpolate(
                cond, size=x.shape[2:], mode="bilinear", align_corners=False
            )

        # stages
        for stage in self.stages.values():
            for block in stage:  # type: ignore
                x = block(x, cond)

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
        drop_path: float = 0.0,
        **_kwargs,
    ) -> None:
        super().__init__()
        self.grad_checkpointing = False
        qk_norm = get_norm_layer(qk_norm)
        norm_layer = get_norm_layer(norm_layer)
        self.attn = NatAttention2d(
            dim,
            k_size,
            stride,
            dilation,
            n_heads,
            qkv_bias,
            qk_norm=qk_norm,
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

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

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
            drop_path,
        )

        if ms_cond_chans is not None:
            self.ms_conv_before_add = create_conv2d(ms_cond_chans, dim, 3)

        mod_factor = 2
        self.latent_cond_type = latent_cond_type
        assert latent_cond_type in ("adaln3", "adaln6"), (
            "latent_cond_type must be adaln3 or adaln6"
        )
        self.modulation = nn.Sequential(
            create_conv2d(cond_chs, dim // mod_factor, 1, bias=True),
            create_norm_act_layer(
                norm_layer, dim // mod_factor, act_layer="silu", eps=1e-6
            ),
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
                sh_a, sc_a, g_a, sh_f, sc_f, g_f = self.modulation(latent_cond).chunk(
                    6, dim=1
                )

            # NAT attention
            y_attn = (
                self.ls1(self.attn(self._modulate(self.norm1(x), sc_a, sh_a))) * g_a
            )
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
        self.ffn = ClipSwiGLUMlp(
            dim, int(ffn_ratio * dim), proj_drop, norm_layer=get_norm_layer(norm_layer)
        )

        self.prenorm_q = create_norm_layer(norm_layer, dim)
        self.prenorm_kv = (
            create_norm_layer(norm_layer, dim) if use_mem_prenorm else nn.Identity()
        )

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
        self.attn = LiteLA(
            hidden_size, hidden_size, heads=self_num_heads, eps=1e-8, qk_norm=qk_norm
        )
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
        self.scale_shift_table = nn.Parameter(
            torch.randn(6, hidden_size) / hidden_size**0.5
        )

    def forward(self, x, mask=None, HW=None, pe=None, **kwargs):
        def _closure(x, HW):
            x = x + self.drop_path(self.attn(self.norm1(x), HW=HW))
            x = x + self.drop_path(self.mlp(self.norm2(x), HW=HW))
            return x

        if self.grad_checkpointing and self.training:
            return checkpoint(_closure, x, HW, use_reentrant=False)
        else:
            return _closure(x, HW)


# * --- interface --- #

IdentityLayer = nn.Identity  # alias


# effiecient vit interface
class EfficientViTBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
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
                    norm_qk=norm_qk,
                ),
                IdentityLayer(),
            )
        elif context_module == "Spatial2DNAT":
            self.context_module = Spatial2DNATBlock(
                dim=in_channels,
                qk_norm=None if not norm_qk else "layernorm2d",
                **kwargs,
            )
            # local module should be identity usually
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.context_module(x)
        x = self.local_module(x)
        return x


# *==============================================================
# * Interface
# *==============================================================


def build_block(
    block_type: str,
    in_channels: int,
    out_channels: int,
    norm: Optional[str],
    act: Optional[str],
    cond_channels: Optional[int] = None,
) -> nn.Module:
    """
    Composition of ResidualBlock with different main and shortcut blocks.
    """
    cfg = block_type.split("@")
    block_name = cfg[0]
    if block_name == "ResBlock":
        assert in_channels == out_channels
        if cond_channels is not None:
            main_block = ResBlockCondition(
                in_channels=in_channels,
                out_channels=out_channels,
                cond_channels=cond_channels,
                kernel_size=3,
                stride=1,
                use_bias=(True, False),
                norm=(None, norm),
                act_func=(act, None),
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
            )
        block = ResidualBlock(main_block, IdentityLayer())
    elif block_name == "GLUResBlock":
        assert in_channels == out_channels
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
        block = ResidualBlock(main_block, IdentityLayer())
    elif block_name == "ChannelAttentionResBlock":
        assert in_channels == out_channels
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
        block = ResidualBlock(main_block, IdentityLayer())
    elif block_name == "GLUMBConv":
        assert in_channels == out_channels
        # GLUMBConv@4
        main_block = GLUMBConv(
            in_channels=in_channels,
            out_channels=out_channels,
            expand_ratio=float(cfg[1]),
            use_bias=(True, True, False),
            norm=(None, None, norm),
            act_func=(act, act, None),
        )
        block = ResidualBlock(main_block, IdentityLayer())
    elif block_name == "EViTGLU":
        assert in_channels == out_channels
        block = EfficientViTBlock(
            in_channels, norm=norm, act_func=act, local_module="GLUMBConv", scales=()
        )
    elif block_name == "EViTNormQKGLU":
        assert in_channels == out_channels
        block = EfficientViTBlock(
            in_channels,
            norm=norm,
            act_func=act,
            local_module="GLUMBConv",
            scales=(),
            norm_qk=True,
        )
    elif block_name == "EViTS5GLU":
        assert in_channels == out_channels
        block = EfficientViTBlock(
            in_channels, norm=norm, act_func=act, local_module="GLUMBConv", scales=(5,)
        )
    elif block_name == "ViTGLU":
        assert in_channels == out_channels
        block = EfficientViTBlock(
            in_channels,
            norm=norm,
            act_func=act,
            context_module="SoftmaxAttention",
            local_module="GLUMBConv",
        )
    elif block_name == "ViTNAT":
        assert in_channels == out_channels
        # ViTNAT@8@1@1@8@4
        if len(cfg) == 1:
            # k_size, stride, dilation, n_heads, ffn_ratio
            cfg = ["ViTNAT", "8", "1", "1", "8", "4"]  # defaults
        block = EfficientViTBlock(
            in_channels,
            norm=norm,
            act_func=act,
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
        block = ConvNeXtBlock(
            in_channels,
            in_channels,
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
    else:
        raise ValueError(f"block_name {block_name} is not supported")
    return block


if __name__ == "__main__":
    # Test the build block function
    ...
