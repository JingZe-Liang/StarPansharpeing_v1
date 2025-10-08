"""
Cross Attention sink with learnable tokens
"""

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn.layers.rotary import (
    RotaryEmbedding as FlashAttnRotaryEmbedding,  # type: ignore
)
from timm.layers import create_norm_layer, get_norm_layer
from timm.layers.attention import AttentionRope as Attention_
from timm.layers.pos_embed import resample_abs_pos_embed
from timm.layers.pos_embed_sincos import (
    RotaryEmbeddingCat,
    apply_rot_embed_cat,
    get_mixed_freqs,
    get_mixed_grid,
)
from torch import Tensor
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .rope import get_1d_sincos_pos_embed_from_grid, get_2d_sincos_pos_embed
from .variants.cross_attn import CrossAttention
from .variants.mlp import SwiGLU


class Attention(Attention_):
    config = SimpleNamespace(
        {
            "_attn_implementation": "flash_attention_2",
        }
    )  # for flash_attention_2 config

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qkv_fused: bool = True,
        num_prefix_tokens: int = 0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_head_dim: Optional[int] = None,
        norm_layer: Type[nn.Module] | str | None = None,
        qk_norm: bool = False,
        scale_norm: bool = False,
        proj_bias: bool = True,
        attn_type: str = "sdpa",
    ):
        super().__init__(
            dim,
            num_heads,
            qkv_bias,
            qkv_fused,
            num_prefix_tokens,
            attn_drop,
            proj_drop,
            attn_head_dim,
            get_norm_layer(norm_layer) if isinstance(norm_layer, str) else norm_layer,
            qk_norm,
            scale_norm,
            proj_bias,
        )
        self.attn_implem = attn_type
        self.is_causal = False

    def forward(
        self, x, rope: Tensor | None = None, mask: BlockMask | Tensor | None = None
    ):
        B, N, C = x.shape

        if self.qkv is not None:
            qkv = self.qkv(x)
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
        else:
            q = (
                self.q_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
            )  # B, num_heads, N, C
            k = self.k_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
            v = self.v_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)

        q, k = self.q_norm(q), self.k_norm(k)

        if rope is not None:
            npt = self.num_prefix_tokens
            # (bs, nhead, n, head_dim)
            q = torch.cat(
                [q[:, :, :npt, :], apply_rot_embed_cat(q[:, :, npt:, :], rope)], dim=2
            ).type_as(v)
            k = torch.cat(
                [k[:, :, :npt, :], apply_rot_embed_cat(k[:, :, npt:, :], rope)], dim=2
            ).type_as(v)

        if self.attn_implem != "flex_attention" and isinstance(mask, BlockMask):
            mask = mask.to_dense()
        x, _ = ALL_ATTENTION_FUNCTIONS[self.attn_implem](
            self, q, k, v, attention_mask=mask, dropout=self.attn_drop.p
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


def resample_1d_pe(pe: Tensor, target_len: int) -> Tensor:
    if pe.shape[1] < target_len:
        # (l, dim)
        pe = F.interpolate(
            pe.transpose(0, 1)[None, ..., None],  # (1, dim, l, 1)
            size=(target_len, 1),
            mode="bicubic",
            align_corners=False,
        )
        pe = pe.squeeze(0, -1).transpose(0, 1)  # (l, dim)
    elif pe.shape[1] > target_len:
        pe = pe[:target_len]
    return pe


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        ctx_dim,
        n_q_heads,
        n_kv_heads=None,
        qkv_bias=False,
        norm_layer="rmsnorm",
        attn_drop=0.0,
        proj_drop=0.0,
        ffn_drop=0.0,
        mlp_ratio=4,
        use_gate=True,
        norm_eps=1e-6,
        fused_type=None,
        use_sa=True,
        attn_type="sdpa",
    ):
        super().__init__()
        self.use_sa = use_sa
        self.self_attention = (
            Attention(
                dim,
                num_heads=n_q_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                qk_norm=True,
                norm_layer=norm_layer,
                attn_type=attn_type,
            )
            if self.use_sa
            else nn.Identity()
        )
        self.cross_attention = CrossAttention(
            dim,
            ctx_dim=ctx_dim,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            qk_norm=norm_layer,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            use_gate=use_gate,
            attn_implem=attn_type,
        )
        self.ffn = SwiGLU(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            norm_layer=None,
            bias=True,
            drop=ffn_drop,
            is_fused=fused_type,
        )

        ls = torch.empty(3, dim).fill_(1e-5)
        self.layer_scales = nn.ParameterList(
            [nn.Parameter(ls[i][None, None]) for i in range(3)]  # (1, 1, dim)
        )

    def forward(self, x, ctx, mask=None, rope=None):
        if self.use_sa:
            x = self.layer_scales[0] * self.self_attention(x, rope=rope) + x
        x = (
            self.layer_scales[1] * self.cross_attention(x, ctx, rope=rope, mask=mask)
            + x
        )
        x = self.layer_scales[2] * self.ffn(x) + x
        return x


@dataclass
class CrossTransformer1DConfig:
    # Transformer configs
    dim: int = 512
    depth: int = 8
    heads: int = 8
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    dropout: float = 0.0
    attention_dropout: float = 0.0
    norm_layer: str = "rmsnorm"
    drop_path: float = 0.0
    attn_type: str = "sdpa"
    # Cross attention specific parameters
    n_kv_heads: Optional[int] = None
    qk_norm: str | None = "rmsnorm"
    use_gate: bool = True
    norm_eps: float = 1e-6
    # Additional transformer parameters
    mlp_dropout: float = 0.0
    act_layer: str = "gelu"
    use_rope: bool = False
    rope_theta: float = 10000.0
    # Sequence specific parameters
    q_seq_len: int = 512
    patch_size: int = 1
    ctx_latent_size: tuple[int, int] = (32, 32)
    query_strategy: str = "2d_mean"  # 2d_mean, learnable
    # Dimensions
    ctx_dim: int = 256
    out_dim: Optional[int] = None


class ContextTransformer1D(nn.Module):
    def __init__(self, cfg: CrossTransformer1DConfig):
        super().__init__()
        self.cfg = cfg

        # Transformer blocks
        self.blocks = nn.ModuleList()
        for i in range(cfg.depth):
            block = CrossAttentionBlock(
                dim=cfg.dim,
                ctx_dim=cfg.ctx_dim,
                n_q_heads=cfg.heads,
                n_kv_heads=cfg.n_kv_heads,
                qkv_bias=cfg.qkv_bias,
                attn_drop=cfg.attention_dropout,
                proj_drop=cfg.dropout,
                use_gate=cfg.use_gate,
                use_sa=True if i != 0 else False,
                attn_type=cfg.attn_type,
            )
            self.blocks.append(block)

        self.projections = nn.ModuleDict(
            {
                "query_proj": nn.Linear(cfg.ctx_dim, cfg.dim)
                if cfg.query_strategy != "learnable"
                else nn.Identity(),
                "ctx_proj": nn.Linear(cfg.ctx_dim, cfg.dim),
            }
        )

        # Query latent
        if cfg.query_strategy == "learnable":
            self.q_latent = nn.Parameter(torch.randn(1, cfg.dim))

        # PEs
        query_latent_pe = get_1d_sincos_pos_embed_from_grid(  # (l, dim)
            cfg.dim, torch.arange(0, cfg.q_seq_len)
        )
        ctx_latent_pe = get_2d_sincos_pos_embed(  # (l, dim)
            cfg.dim, cfg.ctx_latent_size, pe_interpolation=1.0
        )
        self.register_buffer(
            "query_latent_pe", torch.as_tensor(query_latent_pe), persistent=False
        )
        self.register_buffer(
            "ctx_latent_pe", torch.as_tensor(ctx_latent_pe), persistent=False
        )
        self.query_latent_pe: nn.Buffer
        self.ctx_latent_pe: nn.Buffer

        self.rope = RotaryEmbeddingCat(
            dim=cfg.dim // cfg.heads * 2,  # since this class uses div 4.
            temperature=cfg.rope_theta,
            max_res=cfg.q_seq_len,
            feat_shape=[cfg.q_seq_len],
        )
        rope_cat = self.rope.get_embed()
        self.register_buffer("rope_cat", rope_cat, persistent=False)
        self.rope_cat: nn.Buffer

        # Projection out
        self.projections["out_proj"] = nn.Sequential(
            create_norm_layer(cfg.norm_layer, cfg.dim, eps=cfg.norm_eps),
            nn.Linear(cfg.dim, cfg.out_dim or cfg.ctx_dim),
        )

    def _get_queries(self, x):
        H, W = x.shape[-2:]
        N = self.cfg.q_seq_len
        if self.cfg.query_strategy == "learnable":
            q = self.q_latent[None].repeat(x.shape[0], N, 1)  # (bs, qN, dim)
        elif self.cfg.query_strategy == "2d_mean":
            q = x.mean(dim=(-2, -1))[:, None].repeat(1, N, 1)  # (bs, qN, dim)
        else:
            raise ValueError(f"Unknown query strategy: {self.cfg.query_strategy}")
        q = self.projections["query_proj"](q)
        return q

    def _with_pos_embed(self, q, x, x_hw: tuple | None = None):
        # Query
        # is 1d learnable pe
        q_pe = self.query_latent_pe[None].to(q.dtype)
        if q.shape[1] != q_pe.shape[1]:
            q_pe = resample_1d_pe(q_pe, target_len=q.shape[1])
        q = q + q_pe

        # 2D latent
        x_pe = self.ctx_latent_pe[None, :, :].to(x.dtype)
        breakpoint()
        if x.shape[1] != self.ctx_latent_pe.shape[1]:
            if x_hw is None:
                hw = math.sqrt(x.shape[1])
                assert hw.is_integer(), (
                    f"Cannot resample 2D PE with non-square number of tokens: {x.shape[1]}"
                )
                x_hw = (int(hw), int(hw))
            x_pe = resample_abs_pos_embed(  # type: ignore
                x_pe,  # (1, l, dim)
                num_prefix_tokens=0,  # TODO: add register tokens support
                new_size=x_hw,
                old_size=self.cfg.ctx_latent_size,
            )
        x = x + x_pe

        return q, x

    def forward(self, x):
        hw = x.shape[-2:]
        q = self._get_queries(x)

        # proj ctx
        x = self.projections["ctx_proj"](
            x.flatten(2).permute(0, 2, 1)
        )  # (bs, ctxN, dim)

        # Add PEs
        q, x = self._with_pos_embed(q, x, x_hw=hw)

        # Cross attention blocks
        rope_cat = self.rope_cat
        for blk in self.blocks:
            x = blk(q, x, mask=None, rope=rope_cat)

        # Output projection
        x = self.projections["out_proj"](x)

        return x


if __name__ == "__main__":
    """
    python -m src.stage1.cosmos.modules.transformer
    """
    import sys

    print("Testing transformer modules...")

    # Test Attention module (without rope first)
    # print("\n=== Testing Attention module ===")
    # attention = Attention(dim=256, num_heads=8, attn_type="sdpa")
    # x = torch.randn(2, 100, 256)

    # # Test without rope first
    # out = attention(x, rope=None)
    # assert out.shape == (2, 100, 256), f"Expected (2, 100, 256), got {out.shape}"
    # print("✓ Attention module test without rope passed")

    # # Test with rope using proper configuration like in ContextTransformer1D
    # from timm.layers.pos_embed_sincos import RotaryEmbeddingCat

    # cfg_head_dim = 256 // 8  # 32
    # rope = RotaryEmbeddingCat(dim=cfg_head_dim * 2, temperature=10000.0, max_res=100)
    # rope_cat = rope.get_embed(shape=(100,))  # (seq_len, head_dim)
    # print(f"Created rope with shape: {rope_cat.shape}")

    # out_with_rope = attention(x, rope=rope_cat)
    # assert out_with_rope.shape == (2, 100, 256), (
    #     f"Expected (2, 100, 256), got {out_with_rope.shape}"
    # )
    # print("✓ Attention module test with rope passed")

    # # Test CrossAttentionBlock
    # print("\n=== Testing CrossAttentionBlock ===")
    # ca_block = CrossAttentionBlock(
    #     dim=256,
    #     ctx_dim=256,
    #     n_q_heads=8,
    #     n_kv_heads=8,
    #     mlp_ratio=4,
    #     use_sa=True,
    # )
    # x = torch.randn(2, 100, 256)
    # ctx = torch.randn(2, 150, 256)

    # out = ca_block(x, ctx)
    # assert out.shape == (2, 100, 256), f"Expected (2, 100, 256), got {out.shape}"
    # print("✓ CrossAttentionBlock test passed")

    # Test ContextTransformer1D
    print("\n=== Testing ContextTransformer1D ===")
    cfg = CrossTransformer1DConfig(
        dim=256,
        depth=4,
        heads=8,
        ctx_dim=256,
        out_dim=256,
        q_seq_len=100,
        ctx_latent_size=(16, 16),  # 256 tokens
        query_strategy="2d_mean",
        attn_type="sdpa",
    )

    model = ContextTransformer1D(cfg).to("cuda", torch.bfloat16)
    x = torch.randn(
        2, 256, 16, 16, dtype=torch.bfloat16
    ).cuda()  # (batch, ctx_tokens, ctx_dim)

    out = model(x)
    assert out.shape == (2, 100, 256), f"Expected (2, 100, 256), got {out.shape}"
    print("✓ ContextTransformer1D test passed")

    # Test different query strategies
    print("\n=== Testing different query strategies ===")
    cfg_learnable = CrossTransformer1DConfig(
        dim=256,
        depth=2,
        heads=8,
        ctx_dim=256,
        out_dim=256,
        q_seq_len=50,
        ctx_latent_size=(16, 16),
        query_strategy="learnable",
    )

    model_learnable = ContextTransformer1D(cfg_learnable).to("cuda", torch.bfloat16)
    x = torch.randn(1, 256, 32, 32).to("cuda", torch.bfloat16)

    out_learnable = model_learnable(x)
    assert out_learnable.shape == (1, 50, 256), (
        f"Expected (1, 50, 256), got {out_learnable.shape}"
    )
    print("✓ Learnable query strategy test passed")

    # Test resample_1d_pe function
    print("\n=== Testing resample_1d_pe function ===")
    pe = torch.randn(100, 128)

    # Test upsampling
    pe_up = resample_1d_pe(pe, target_len=150)
    assert pe_up.shape == (150, 128), f"Expected (150, 128), got {pe_up.shape}"

    # Test downsampling
    pe_down = resample_1d_pe(pe, target_len=50)
    assert pe_down.shape == (50, 128), f"Expected (50, 128), got {pe_down.shape}"

    # Test same length
    pe_same = resample_1d_pe(pe, target_len=100)
    assert pe_same.shape == (100, 128), f"Expected (100, 128), got {pe_same.shape}"
    assert torch.allclose(pe, pe_same), "Same length should return identical tensor"

    print("✓ resample_1d_pe function test passed")

    print("\n🎉 All tests passed successfully!")
