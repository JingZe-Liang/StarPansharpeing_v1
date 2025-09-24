import torch as th
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import (
    ConvMlp,
    LayerScale2d,
    create_norm,
    get_act_layer,
    get_norm_layer,
)
from timm.layers.drop import DropPath
from timm.layers.mlp import Mlp
from timm.models.convnext import ConvNeXtStage
from torch.utils.checkpoint import checkpoint

from .attention import Attention, NatAttention1d, NatAttention2d
from .conv import MbConvLNBlock, Stem
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
        ffn_ratio=4 / 3,
        qkv_bias=True,
        qk_norm="layernorm2d",
        norm_layer="layernorm2d",
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

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        def _closure(x):
            x = x + self.ls1(self.attn(self.norm1(x)))
            x = x + self.ls2(self.cffn(self.norm2(x)))
            return x

        # checkpointing
        if self.grad_checkpointing and self.training:
            x = checkpoint(_closure, x, use_reentrant=False)
        else:
            x = _closure(x)
        return x
