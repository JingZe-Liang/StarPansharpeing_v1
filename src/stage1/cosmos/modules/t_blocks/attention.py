# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
From: https://github.com/naver-ai/rope-vit/blob/main/self-attn/rope_self_attn.py
This code was originally obtained from:
https://github.com/meta-llama/codellama/blob/main/llama/model.py
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from loguru import logger
from timm.layers import apply_rot_embed_cat

from .ada_norms import get_norm_layer

# JVP support
try:
    from jvp_flash_attention.jvp_attention import JVPAttn
    from jvp_flash_attention.jvp_attention import attention as jvp_attention

    JVP_FLASH_ATTN_ENABLED = True
except ImportError:
    JVP_FLASH_ATTN_ENABLED = False


def _jvp_math_attention(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
):
    """JVP-compatible math attention implementation"""
    # Use math backend for JVP compatibility
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=dropout,
            is_causal=getattr(module, "is_causal", False),
        ), None


def _jvp_flash_attention(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
):
    """JVP-compatible flash attention implementation"""
    if not JVP_FLASH_ATTN_ENABLED:
        logger.warning("JVP Flash Attention not available, falling back to math attention")
        return _jvp_math_attention(module, query, key, value, attention_mask, dropout)

    if dropout > 0:
        logger.warning("Dropout is not supported for JVP Flash Attention, setting to 0")

    try:
        x = jvp_attention(query, key, value, attn_mask=attention_mask)
        return x, None
    except Exception as e:
        logger.warning(f"JVP Flash Attention failed: {e}, falling back to math attention")
        return _jvp_math_attention(module, query, key, value, attention_mask, dropout)


#######################
### Basic Attention ###
#######################


class Attention(nn.Module):
    # Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py
    r"""
    A self / cross attention layer.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: Optional[int] = None,
        out_dim: int | None = None,
        dropout: float = 0.0,
        attn_drop: float = 0.0,
        bias: bool = False,
        qk_norm: Optional[str] = "layernorm",
        cross_attention_norm: Optional[str] = None,
        out_bias: bool = True,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        is_causal: bool = False,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        *,
        delta_t_aware: bool = False,
        delta_t_dim: int | None = None,
        jvp: bool = False,
    ):
        super().__init__()

        # Compute dims
        if out_dim is None:
            out_dim = dim_head * heads if dim_head is not None else query_dim
        inner_dim = out_dim if dim_head is None else dim_head * heads
        is_cross_attention = cross_attention_dim is not None
        cross_attention_dim = cross_attention_dim or query_dim
        if dim_head is None:
            dim_head = inner_dim // heads

        if out_dim is not None:
            assert heads * dim_head == inner_dim

        # Args
        self.is_causal = is_causal
        self.heads = heads
        self.inner_dim = inner_dim
        self.dim_head = dim_head
        self.rescale_output_factor = rescale_output_factor
        self.attn_drop = attn_drop
        self.residual_connection = residual_connection

        self.norm_q = get_norm_layer(
            qk_norm,
            dim_head,
            heads=heads,
            eps=eps,
            elementwise_affine=elementwise_affine,
        )
        self.norm_k = get_norm_layer(
            qk_norm,
            dim_head,
            heads=heads,
            eps=eps,
            elementwise_affine=elementwise_affine,
        )
        self.norm_cross = get_norm_layer(
            cross_attention_norm if is_cross_attention else None,
            cross_attention_dim,
            eps=eps,
        )

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=bias)

        self.delta_t_aware = delta_t_aware
        if delta_t_aware:
            self.to_q_delta = nn.Linear(delta_t_dim or query_dim, inner_dim, bias=bias)
            self.to_k_delta = nn.Linear(delta_t_dim or query_dim, inner_dim, bias=bias)
            self.to_v_delta = nn.Linear(delta_t_dim or query_dim, inner_dim, bias=bias)

        self.out_proj = nn.Linear(inner_dim, out_dim, bias=out_bias)
        self.out_drop = nn.Dropout(dropout)

        self._cache_attn_mask = None
        self.jvp = jvp
        self._attention_functions = {
            "sdpa": self._sdpa_attention,
            "jvp_math": _jvp_math_attention,
            "jvp_flash": _jvp_flash_attention,
        }

        if jvp and not JVP_FLASH_ATTN_ENABLED:
            logger.warning(
                "JVP Flash Attention is not enabled. Please install it by running `pip install jvp_flash_attention` "
                "or the flash attention will not be used and fall back to math attention kernel."
            )
            self._attn_impl = "jvp_math"
        elif jvp and JVP_FLASH_ATTN_ENABLED:
            self._attn_impl = "jvp_flash"
        else:
            self._attn_impl = "sdpa"

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states=None,  # For compatibility with transformers
        delta_emb: Optional[torch.Tensor] = None,
        rope: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert encoder_hidden_states is None, "encoder_hidden_states should be None"
        residual = hidden_states

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = rearrange(hidden_states, "B C H W -> B (H W) C")

        batch_size, sequence_length, _ = (
            hidden_states.shape if cross_hidden_states is None else cross_hidden_states.shape
        )

        if not self.training and self._cache_attn_mask is not None:
            # If eval and have cached attention mask, use it
            attention_mask = self._cache_attn_mask
        elif attention_mask is not None:
            attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            attention_mask = attention_mask.view(batch_size, self.heads, -1, attention_mask.shape[-1])

        ######## Q, K, V #########
        query = self.to_q(hidden_states)
        if cross_hidden_states is None:
            cross_hidden_states = hidden_states
        elif self.norm_cross:
            cross_hidden_states = self.norm_cross_hidden_states(cross_hidden_states)
        key = self.to_k(cross_hidden_states)
        value = self.to_v(cross_hidden_states)

        # Add delta t awareness
        if self.delta_t_aware and delta_emb is not None:
            delta_query = self.to_q_delta(delta_emb)[:, None]
            delta_key = self.to_k_delta(delta_emb)[:, None]
            delta_value = self.to_v_delta(delta_emb)[:, None]

            query = query + delta_query
            key = key + delta_key
            value = value + delta_value

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads

        # bs, nheads, seq_len, head_dim
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        # rope
        if rope is not None:
            assert tuple(rope.shape[-2:]) == (sequence_length, head_dim * 2), (
                f"rope shape {tuple(rope.shape[-2:])} does not match needed shape {(sequence_length, head_dim * 2)}"
            )
            query = apply_rot_embed_cat(query, rope)
            key = apply_rot_embed_cat(key, rope)

        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        # output of sdp: (batch, num_heads, seq_len, head_dim)
        hidden_states = self._process_attn(query, key, value, attn_mask=attention_mask)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = self.out_proj(hidden_states)
        hidden_states = self.out_drop(hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if self.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / self.rescale_output_factor

        return hidden_states

    def _sdpa_attention(self, query, key, value, attn_mask, *args, **kwargs):
        """Standard SDPA attention (non-JVP)"""
        return F.scaled_dot_product_attention(  # pylint: disable=not-callable
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop,
            is_causal=self.is_causal,
        )

    def _process_attn(self, query, key, value, attn_mask):
        """Process attention with JVP support"""
        if self.jvp:
            # Use JVP-compatible attention
            attention_fn = self._attention_functions[self._attn_impl]
            output, _ = attention_fn(
                self,
                query,
                key,
                value,
                attn_mask,
                self.attn_drop if self.training else 0.0,
            )
            return output
        else:
            # Use standard SDPA
            return self._sdpa_attention(query, key, value, attn_mask)

    def prepare_attention_mask(
        self,
        attention_mask: torch.Tensor,
        target_length: int,
        batch_size: int,
        out_dim: int = 3,
    ) -> torch.Tensor:
        r"""
        Prepare the attention mask for the attention computation.

        Args:
            attention_mask (`torch.Tensor`):
                The attention mask to prepare.
            target_length (`int`):
                The target length of the attention mask. This is the length of the attention mask after padding.
            batch_size (`int`):
                The batch size, which is used to repeat the attention mask.
            out_dim (`int`, *optional*, defaults to `3`):
                The output dimension of the attention mask. Can be either `3` or `4`.

        Returns:
            `torch.Tensor`: The prepared attention mask.
        """
        head_size = self.heads
        if attention_mask is None:
            return attention_mask

        current_length: int = attention_mask.shape[-1]
        if current_length != target_length:
            attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)

        if out_dim == 3:
            if attention_mask.shape[0] < batch_size * head_size:
                attention_mask = attention_mask.repeat_interleave(head_size, dim=0)
        elif out_dim == 4:
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.repeat_interleave(head_size, dim=1)

        return attention_mask
