import os
from collections import OrderedDict
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from timm.models.vitamin import GeGluMlp
from torch import Tensor, nn

from .rope import apply_rotary_emb

# check the version of pytorch,
# if pytorch version >= 2.2.0, then flash_attention can be used
if torch.__version__ >= "2.2.0":
    HAS_FLASH_ATTENTION_V2 = True
    # print("flash_attention v2 can be used.")
else:
    HAS_FLASH_ATTENTION_V2 = False
    # print("flash_attention v2 is not supported.")


# *==============================================================
# * Self-Attention
# *==============================================================


class AttentionCustom(nn.Module):
    def __init__(
        self,
        dim,
        n_head,
        resid_dropout_p,
        use_rope=False,
        use_qk_norm=False,
        use_flash_attn=False,
        no_bias=False,
        attn_dropout_p=0,
    ):
        """
        This custom attention block supports the following modifications:
        - ROPE
        - QK Norm
        - Flash attention
        Currently, the dimension of the key and value is the same as the query.
        """
        super().__init__()

        self.use_rope = use_rope
        self.use_qk_norm = use_qk_norm
        self.use_flash_attn = use_flash_attn
        if self.use_flash_attn:
            print("Using flash attention!")
        # flash attention can be switched to normal attention for inference
        # rasie error only when training and use_flash_attn is True and
        # flash attention is not installed
        assert HAS_FLASH_ATTENTION_V2 or (not self.use_flash_attn) or (not self.training), (
            "Flash attention is not installed and cannot be used when training"
        )
        assert dim % n_head == 0
        self.dim = dim
        self.head_dim = dim // n_head
        self.n_head = n_head
        total_qkv_dim = 3 * self.n_head * self.head_dim

        # key, query, value projections for all heads, but in a batch
        self.q_proj = nn.Linear(dim, dim, bias=not no_bias)
        self.k_proj = nn.Linear(dim, dim, bias=not no_bias)
        self.v_proj = nn.Linear(dim, dim, bias=not no_bias)
        self.wo = nn.Linear(dim, dim, bias=not no_bias)

        # regularization
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout = nn.Dropout(resid_dropout_p)

        if self.use_qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        freqs_cis: torch.Tensor | None = None,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ):
        """
        The q, k, v will be projected into multiple heads.
        """
        seqlen, bsz, _ = query.shape

        # rearrange, (L, B, D) -> (B, L, D)
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)

        xq = self.q_proj(query)
        xk = self.k_proj(key)
        xv = self.v_proj(value)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_head, self.head_dim)

        if self.use_qk_norm:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        if self.use_rope:
            xq = apply_rotary_emb(xq, freqs_cis)
            xk = apply_rotary_emb(xk, freqs_cis)
        else:
            assert freqs_cis is None, (
                "Attention Module is not using ROPE but freqs_cis is not None. Check your setting!"
            )

        # (B, L, H, D) -> (B, H, L, D)
        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))

        if self.use_flash_attn:
            with torch.nn.attention.sdpa_kernel(backends="flash"):
                output = F.scaled_dot_product_attention(
                    xq,
                    xk,
                    xv,
                    attn_mask=attn_mask,
                    is_causal=is_causal,  # is_causal=False is for KV cache
                    dropout_p=self.attn_dropout_p if self.training else 0,
                )
        else:
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                attn_mask=attn_mask,
                is_causal=is_causal,  # is_causal=False is for KV cache
                dropout_p=self.attn_dropout_p if self.training else 0,
            )

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        output = self.resid_dropout(self.wo(output))

        # rearrange, (B, L, D) -> (L, B, D)
        return output.transpose(0, 1)


class SelfAttentionCustom(nn.Module):
    def __init__(
        self,
        dim,
        n_head,
        resid_dropout_p,
        use_rope=False,
        use_qk_norm=False,
        no_bias=False,
        use_flash_attn=False,
        attn_dropout_p=0,
    ):
        """
        This custom attention block supports the following modifications:
        - ROPE
        - QK Norm
        - Flash attention
        """
        super().__init__()

        self.use_rope = use_rope
        self.use_qk_norm = use_qk_norm
        self.use_flash_attn = use_flash_attn
        # flash attention can be switched to normal attention for inference
        # rasie error only when training and use_flash_attn is True and
        # flash attention is not installed
        assert HAS_FLASH_ATTENTION_V2 or (not use_flash_attn) or (not self.training), (
            "Flash attention is not installed and cannot be used when training"
        )
        assert dim % n_head == 0

        if self.use_flash_attn:
            print("Using flash attention!")

        self.dim = dim
        self.head_dim = dim // n_head
        self.n_head = n_head
        total_qkv_dim = 3 * self.n_head * self.head_dim

        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(dim, total_qkv_dim, bias=not no_bias)
        self.wo = nn.Linear(dim, dim, bias=not no_bias)

        # regularization
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout = nn.Dropout(resid_dropout_p)

        if self.use_qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor = None,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ):
        seqlen, bsz, _ = x.shape

        # rearrange, (L, B, D) -> (B, L, D)
        x = x.transpose(0, 1)

        xq, xk, xv = self.wqkv(x).split([self.dim, self.dim, self.dim], dim=-1)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_head, self.head_dim)

        if self.use_qk_norm:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        if self.use_rope:
            xq = apply_rotary_emb(xq, freqs_cis)
            xk = apply_rotary_emb(xk, freqs_cis)
        else:
            assert freqs_cis is None, (
                "Attention Module is not using ROPE but freqs_cis is not None. Check your setting!"
            )

        # (B, L, H, D) -> (B, H, L, D)
        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))

        if self.use_flash_attn:
            # Shape: (batch_size, num_heads, seq_length, head_dim)
            with torch.nn.attention.sdpa_kernel(backends="flash"):
                output = F.scaled_dot_product_attention(
                    xq,
                    xk,
                    xv,
                    attn_mask=mask,
                    is_causal=is_causal,  # is_causal=False is for KV cache
                    dropout_p=self.attn_dropout_p if self.training else 0,
                )
        else:
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                attn_mask=mask,
                is_causal=is_causal,  # is_causal=False is for KV cache
                dropout_p=self.attn_dropout_p if self.training else 0,
            )

        # (B, H, L, D) -> (B, L, H*D)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        output = self.resid_dropout(self.wo(output))

        # rearrange back, (B, L, D) -> (L, B, D)
        return output.transpose(0, 1)


class AttnProjection(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, norm_layer=nn.LayerNorm, mlp_ratio=2):
        super().__init__()
        assert out_dim % in_dim == 0 or in_dim % out_dim == 0
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.norm1 = norm_layer(in_dim)
        # self.attn = CausalAttention(in_dim, out_dim, num_heads)
        self.attn = nn.MultiheadAttention(
            in_dim,
            num_heads=num_heads,
        )
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm3 = norm_layer(in_dim)

        self.norm2 = norm_layer(out_dim)
        hidden_dim = int(out_dim * mlp_ratio)
        self.mlp = GeGluMlp(in_features=out_dim, hidden_features=hidden_dim)

    def forward(self, x):
        x = self.proj(self.norm3(x)) + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# *==============================================================
# * ViT Encoders
# *==============================================================
