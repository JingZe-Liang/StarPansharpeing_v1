from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from timm.layers import create_norm
from torch import Tensor
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .conv import ConvLayer, LinearLayer


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


class CrossAttention(nn.Module):
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
        gate_type: str | None = None,
        norm_eps=1e-6,
        rope_type: str | None = None,
    ):
        super().__init__()
        self.ctx_dim = ctx_dim if ctx_dim is not None else dim
        assert self.ctx_dim is not None, "ctx_dim must be specified if different from dim"

        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads = n_kv_heads or n_q_heads
        assert dim % n_q_heads == 0, "dim should be divisible by n_q_heads"
        assert self.ctx_dim % n_kv_heads == 0, "dim should be divisible by n_kv_heads"
        assert n_q_heads % self.n_kv_heads == 0, (
            "n_q_heads should be divisible by n_kv_heads for Grouped-Query Attention"
        )

        self.head_dim = dim // n_q_heads
        self.scale = self.head_dim**-0.5
        self.rope_type = rope_type

        self.gate_type = gate_type
        if gate_type is None:
            q_dim = n_q_heads * self.head_dim
        elif gate_type == "element_wise":
            q_dim = n_q_heads * self.head_dim * 2
        elif gate_type == "head_wise":
            q_dim = n_q_heads * self.head_dim + n_q_heads
        else:
            raise ValueError(f"Unsupported gate type: {gate_type}")

        kv_dim = n_kv_heads * self.head_dim
        self.q_proj = nn.Linear(dim, q_dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(self.ctx_dim, kv_dim * 2, bias=qkv_bias)

        norm = create_norm.get_norm_layer(qk_norm) if qk_norm else None
        if norm is not None:
            self.q_norm = norm(self.head_dim, eps=norm_eps)
            self.k_norm = norm(self.head_dim, eps=norm_eps)
        else:
            self.q_norm = self.k_norm = nn.Identity()

        self.out_proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_drop = attn_drop

        self.is_causal = False
        self._attn_implementation = "sdpa"

    def _rope(self, q, k, rope: Callable | tuple[Tensor, Tensor] | None = None):
        if self.rope_type == "rope_q":
            raise NotImplementedError("rope_q not implemented")
        else:
            if callable(rope):
                q, k = rope(q, k)
            elif isinstance(rope, tuple):
                cos, sin = rope
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
        return q, k

    def _qkv_proj(self, x, context, rope: Callable | tuple[Tensor, Tensor] | None = None):
        bl_shape = x.shape[:-1]  # (B, L)

        # query and gate
        qg = self.q_proj(x)
        if self.gate_type is None:
            q_states = qg.view(*bl_shape, -1, self.head_dim)
            gate = None
        elif self.gate_type == "element_wise":
            qg = qg.view(*bl_shape, -1, self.head_dim * 2)
            q_states, gate = qg.chunk(2, dim=-1)
        else:
            qg = qg.view(*bl_shape, -1, self.head_dim + 1)
            q_states, gate = qg.split([self.head_dim, 1], dim=-1)
            gate = gate.view(*bl_shape, -1, 1)  # (B, L, n_q_heads, 1)
        q_states = self.q_norm(q_states).transpose(1, 2)  # (B, n_q_heads, L, head_dim)

        # context key and value
        b_ctxl_shape = context.shape[:-1]  # (B, L2)
        kv_states = self.kv_proj(context)
        kv_states = kv_states.view(*b_ctxl_shape, -1, self.head_dim * 2)  # (B, L2, n_kv_heads, head_dim*2)
        k_states, v_states = kv_states.chunk(2, dim=-1)
        k_states = self.k_norm(k_states).transpose(1, 2)
        v_states = v_states.transpose(1, 2)

        # rope
        q_states, k_states = self._rope(q_states, k_states, rope)

        return q_states, k_states, v_states, gate

    def _attention(self, q, k, v, mask=None):
        attn_interface = ALL_ATTENTION_FUNCTIONS[self._attn_implementation]
        attn_out, attn_weights = attn_interface(
            self,
            q,
            k,
            v,
            mask,
            dropout_p=self.attn_drop if self.training else 0.0,
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
        mask: Tensor | None = None,
    ):
        B, N1, C = x.shape
        q_hidden_shape = (B, N1)

        q, k, v, g = self._qkv_proj(x, context, rope)
        attn_out, _ = self._attention(q, k, v, mask)
        if g is not None:
            attn_out = attn_out * g.sigmoid()
        attn_out = attn_out.reshape(*q_hidden_shape, -1)
        out = self.out_proj(attn_out)

        return out


class SoftmaxCrossAttention2D(nn.Module):
    """
    input q is 2d image but the kv context is 1d.
    """

    def __init__(
        self,
        q_in_channels: int,
        kv_in_channels: int,
        out_channels: int,
        head_dim: int = 32,
        use_bias: bool = False,
        norm: tuple[Optional[str], Optional[str]] = (None, None),
        act_func: tuple[Optional[str], Optional[str]] = (None, None),
    ):
        super().__init__()
        assert q_in_channels % head_dim == 0, "q_in_channels must be divisible by head_dim"
        assert kv_in_channels % head_dim == 0, "kv_in_channels must be divisible by head_dim"
        assert out_channels % head_dim == 0, "out_channels must be divisible by head_dim"

        self.q_in_channels = q_in_channels
        self.kv_in_channels = kv_in_channels
        self.out_channels = out_channels
        self.head_dim = head_dim
        self.num_heads = out_channels // head_dim

        self.q = ConvLayer(
            q_in_channels,
            out_channels,
            1,
            use_bias=use_bias,
            norm=norm[0],
            act_func=act_func[0],
        )
        self.kv = LinearLayer(
            kv_in_channels,
            out_channels * 2,
            use_bias=use_bias,
            norm=norm[0],
            act_func=act_func[0],
            try_squeeze=False,
        )
        self.proj = ConvLayer(
            out_channels,
            out_channels,
            1,
            use_bias=use_bias,
            norm=norm[1],
            act_func=act_func[1],
        )

    def forward(self, x, cond, mask=None):
        # query: img tokens; key/value: condition; mask: if padding tokens
        # cond: (B, N, C)
        B, _, H, W = x.shape
        N = cond.shape[1]

        q = (
            self.q(x).reshape(B, -1, self.head_dim, H * W).transpose(2, 3)
        )  # (B, C, H, W) -> (B, num_heads, head_dim, H * W) -> (B, num_heads, H * W, head_dim)
        kv = (
            self.kv(cond).view(B, N, -1, 2 * self.head_dim).transpose(1, 2)
        )  # (B, N, 2 * C) -> (B, N, num_heads, 2 * head_dim) -> (B, num_heads, N, 2 * head_dim)
        k, v = kv[..., 0 : self.head_dim], kv[..., self.head_dim :]

        if mask is not None and mask.ndim == 2:
            raise NotImplementedError("mask is not supported")
            # mask = (1 - mask.to(x.dtype)) * -10000.0
            # mask = mask[:, None, None].repeat(1, self.num_heads, 1, 1)
        x = F.scaled_dot_product_attention(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False,
        )
        x = x.transpose(2, 3)  # (B, num_heads, H * W, head_dim) -> (B, num_heads, head_dim, H * W)
        x = x.reshape(B, -1, H, W)
        x = self.proj(x)
        return x


# * --- Test --- #
import pytest


@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize(["seq_len_q", "seq_len_kv"], [[128, 256], [256, 512]])
@pytest.mark.parametrize("dim", [32, 64, 128])
@pytest.mark.parametrize("gate_type", ["element_wise", "head_wise", None])
def test_forward(batch_size, seq_len_q, seq_len_kv, dim, gate_type):
    n_q_heads = 8
    n_kv_heads = 8

    model = CrossAttention(dim, n_q_heads=n_q_heads, n_kv_heads=n_kv_heads, gate_type=gate_type)
    x = torch.randn(batch_size, seq_len_q, dim)
    context = torch.randn(batch_size, seq_len_kv, dim)

    output = model(x, context)
    assert output.shape == (batch_size, seq_len_q, dim)
