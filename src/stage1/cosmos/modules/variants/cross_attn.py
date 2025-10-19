from types import SimpleNamespace
from typing import Callable

import torch
import torch.nn as nn
from jaxtyping import Float
from timm.layers import create_norm
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Adapted from transformers.models.glm.modular_glm.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Removes the interleaving of cos and sin from GLM

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


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
create_norm._NORM_MAP["zeromeanrmsnorm"] = Qwen3NextRMSNorm  # type: ignore


class CrossAttention(nn.Module):
    config = SimpleNamespace(
        _attn_implementation="flash_attention_2"
    )  # for flash_attention_2 config

    def __init__(
        self,
        dim: int,
        ctx_dim: int | None = None,
        n_q_heads=8,
        n_kv_heads=None,
        qk_norm: str | None = "zero_mean_rmsnorm",
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        use_gate=True,
        norm_eps=1e-6,
        attn_implem="sdpa",
    ):
        super().__init__()
        self.ctx_dim = ctx_dim if ctx_dim is not None else dim
        assert self.ctx_dim is not None, (
            "ctx_dim must be specified if different from dim"
        )

        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads = n_kv_heads or n_q_heads
        assert dim % n_q_heads == 0, "dim should be divisible by n_q_heads"
        assert self.ctx_dim % n_kv_heads == 0, "dim should be divisible by n_kv_heads"
        assert n_q_heads % self.n_kv_heads == 0, (
            "n_q_heads should be divisible by n_kv_heads for Grouped-Query Attention"
        )

        self.head_dim = dim // n_q_heads
        self.scale = self.head_dim**-0.5

        q_dim = (
            n_q_heads * self.head_dim if not use_gate else n_q_heads * self.head_dim * 2
        )
        kv_dim = n_kv_heads * self.head_dim
        self.q_proj = nn.Linear(dim, q_dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(self.ctx_dim, kv_dim * 2, bias=qkv_bias)

        norm = create_norm.get_norm_layer(qk_norm) if qk_norm else None
        if norm is not None:
            self.q_norm = norm(self.head_dim, eps=norm_eps)
            self.k_norm = norm(self.head_dim, eps=norm_eps)

        self.out_proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_drop = attn_drop

        self.is_causal = False
        self._attn_implementation = (
            attn_implem  # flash_attention_2, spda, flex_attention
        )

    def _qkv_proj(
        self, x, context, rope: Callable | tuple[Tensor, Tensor] | None = None
    ):
        bl_shape = x.shape[:-1]  # (B, L)
        qg = self.q_proj(x)
        qg = qg.view(*bl_shape, -1, self.head_dim * 2)
        q_states, gate = qg.chunk(2, dim=-1)
        q_states = self.q_norm(q_states).transpose(1, 2)  # (B, n_q_heads, L, head_dim)
        gate = gate.reshape(*bl_shape, -1)  # (B, L, C)

        b_ctxl_shape = context.shape[:-1]  # (B, L2)
        kv_states = self.kv_proj(context)
        kv_states = kv_states.view(
            *b_ctxl_shape, -1, self.head_dim * 2
        )  # (B, L2, n_kv_heads, head_dim*2)
        k_states, v_states = kv_states.chunk(2, dim=-1)
        k_states = self.k_norm(k_states).transpose(1, 2)
        v_states = v_states.transpose(1, 2)

        if callable(rope):
            q_states, k_states = rope(q_states, k_states)
        elif isinstance(rope, tuple):
            cos, sin = rope
            q_states, k_states = apply_rotary_pos_emb(q_states, k_states, cos, sin)

        return q_states, k_states, v_states, gate

    def _attention(self, q, k, v, mask: Tensor | BlockMask | None = None):
        if self._attn_implementation == "flex_attention":
            # not compiled: transformers == 4.57.0
            assert isinstance(mask, BlockMask) or mask is None, (
                f"mask must be BlockMask or None, got {type(mask)}"
            )
            attn_out, attn_weights = flex_attention(
                q, k, v, block_mask=mask, return_lse=False, enable_gqa=True
            )
        else:
            attn_interface = ALL_ATTENTION_FUNCTIONS[self._attn_implementation]
            attn_out, attn_weights = attn_interface(
                self,
                q,
                k,
                v,
                mask,
                dropout=self.attn_drop if self.training else 0.0,
                is_causal=self.is_causal,
                scaling=self.scale,
            )
        return attn_out, attn_weights

    def forward(
        self,
        x: Float[Tensor, "b l c"],
        context: Float[Tensor, "b l2 c"],
        rope: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]  # callable rope class
        | tuple[Tensor, Tensor]  # tuple of cos, sin
        | None = None,
        mask: Tensor | BlockMask | None = None,
    ):
        B, N1, C = x.shape
        q_hidden_shape = (B, N1)

        q, k, v, g = self._qkv_proj(x, context, rope)
        attn_out, _ = self._attention(q, k, v, mask)
        attn_out = attn_out.reshape(*q_hidden_shape, -1)
        attn_out = attn_out * g.sigmoid()
        out = self.out_proj(attn_out)

        return out


# * --- Test --- #
import pytest


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize(["seq_len_q", "seq_len_kv"], [[128, 256], [256, 512]])
@pytest.mark.parametrize("dim", [32, 64, 128])
def test_forward(batch_size, seq_len_q, seq_len_kv, dim):
    n_q_heads = 8
    n_kv_heads = 8

    model = CrossAttention(dim, n_q_heads=n_q_heads, n_kv_heads=n_kv_heads)
    x = torch.randn(batch_size, seq_len_q, dim)
    context = torch.randn(batch_size, seq_len_kv, dim)

    output = model(x, context)
    assert output.shape == (batch_size, seq_len_q, dim)
