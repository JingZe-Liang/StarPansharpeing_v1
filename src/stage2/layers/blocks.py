import torch as th
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import (
    ConvMlp,
    LayerScale2d,
    create_conv2d,
    create_norm,
    create_norm_act_layer,
    create_norm_layer,
    get_act_layer,
    get_norm_layer,
)
from timm.layers.drop import DropPath
from timm.layers.mlp import Mlp
from timm.models.convnext import ConvNeXtStage
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .attention import Attention, NatAttention1d, NatAttention2d, Qwen3SdpaAttention
from .conv import MbConvLNBlock, Stem
from .cross_attn import CrossAttention
from .layerscale import LayerScale
from .mlp import ClipSwiGLUMlp, SwiGLU


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
        qk_norm: type[nn.Module] | None = None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        mlp_norm_layer=nn.LayerNorm,
        act_layer=nn.SiLU,
        layer_scale_value=1e-3,
        use_layerscale=False,
    ):
        super().__init__()
        self.grad_checkpointing = False
        self.norm1 = norm_layer(dim)
        if attn_type == "1d":
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                norm_layer=norm_layer,
                attn_drop=attn_drop,
                proj_drop=drop,
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
            self.attn = NatAttention1d(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                norm_layer=norm_layer,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        elif attn_type == "nat_2d":
            self.attn = NatAttention2d(
                dim,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                norm_layer=norm_layer,
                proj_bias=True,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        self.norm2 = norm_layer(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = ClipSwiGLUMlp(
            in_features=dim,
            hidden_features=hidden_dim,
            act_layer=act_layer,
            drop=drop,
            norm_layer=mlp_norm_layer,
            use_conv=attn_type != "1d",
        )
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

    def forward(self, x, mask=None, pe=None):
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
        self.stem = Stem(
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
            cond = th.nn.functional.interpolate(
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
            act_layer=get_act_layer("gelu_tanh"),
        )
        self.ls1 = LayerScale2d(dim, init_values=1e-5, inplace=True)
        self.ls2 = LayerScale2d(dim, init_values=1e-5, inplace=True)

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
        self.modulation = nn.Sequential(
            create_conv2d(cond_chs, dim // mod_factor, 1, bias=True),
            create_norm_act_layer(
                norm_layer, dim // mod_factor, act_layer="silu", eps=1e-6
            ),
            create_conv2d(
                dim // mod_factor,
                dim * 6,
                3,
                stride=1,
                bias=False,
                groups=dim // 2,
            ),
        )

    def _interp_as(self, x, tgt_sz: tuple | th.Size):
        return F.interpolate(x, size=tgt_sz, mode="bilinear", align_corners=False)

    def _modulate(self, x, scale, shift):
        return x * (scale + 1) + shift

    def forward(self, x, latent, ms_cond=None):
        def _closure(x, latent, ms_cond=None):
            if hasattr(self, "ms_conv_before_add") and ms_cond is not None:
                ms_cond = self._interp_as(ms_cond, x.shape[2:])
                x = x + self.ms_conv_before_add(ms_cond)

            latent_cond = self._interp_as(latent, x.shape[2:])
            sh_a, sc_a, g_a, sh_f, sc_f, g_f = self.modulation(latent_cond).chunk(
                6, dim=1
            )
            y_attn = self.ls1(self.attn(self._modulate(self.norm1(x), sc_a, sh_a))) * g_a  # fmt: skip
            x = x + self.drop_path(y_attn)
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
            self.q_pos_embed = nn.Parameter(th.zeros(1, q_len, dim))
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


if __name__ == "__main__":
    ca = CrossAttentionBlock(
        256,
        256,
        n_q_heads=8,
        n_kv_heads=8,
        sa_gate_type="head_wise",
        ca_gate_type="head_wise",
    )
    q = th.randn(2, 1024, 256)
    mem = th.randn(2, 4096, 256)
    print(ca(q, mem).shape)
