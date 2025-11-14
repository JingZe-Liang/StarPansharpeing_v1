import os
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import natten
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict
from einops import rearrange
from jaxtyping import Float
from loguru import logger
from timm.layers import (
    DropPath,
    LayerScale,
    PatchEmbed,
    create_act_layer,
    create_norm_act_layer,
    create_norm_layer,
    get_norm_layer,
)
from timm.layers.attention import AttentionRope as Attention_
from timm.layers.create_act import create_act_layer
from timm.layers.create_conv2d import create_conv2d
from timm.layers.create_norm import create_norm_layer
from timm.layers.helpers import to_2tuple
from timm.layers.pos_embed import resample_abs_pos_embed, resample_abs_pos_embed_nhwc
from timm.layers.pos_embed_sincos import (
    RotaryEmbeddingCat,
    RotaryEmbeddingDinoV3,
    apply_rot_embed_cat,
    create_rope_embed,
    get_mixed_freqs,
    get_mixed_grid,
)
from torch import Tensor
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)
from torch.utils.checkpoint import checkpoint
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from typing_extensions import Annotated

from .conv import ConvLayer, GLUMBConv, MBConv, ResBlock

IdentityLayer = nn.Identity  # alias


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


def _float16_clip_value(x, dtype: torch.dtype | None = None):
    is_f16 = (dtype or torch.get_autocast_dtype("cuda")) in (
        torch.float16,
        torch.bfloat16,
    )
    if is_f16:
        x = x.clip(-65504, 65504)

    return x


# *==============================================================
# * Naive Attention
# *==============================================================


class Attention(Attention_):
    config = EasyDict(
        _attn_implementation="flash_attention_2"
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
        norm_layer: type[nn.Module] | str | None = None,
        qk_norm: bool = True,
        scale_norm: bool = False,
        proj_bias: bool = True,
        attn_type: str = "sdpa",
        is_causal: bool = False,
        rotate_half=False,
    ):
        norm_layer = (
            get_norm_layer(norm_layer) if isinstance(norm_layer, str) else norm_layer
        )
        super().__init__(
            dim,
            num_heads,
            qkv_bias,
            qkv_fused,
            num_prefix_tokens,
            attn_drop,
            proj_drop,
            attn_head_dim,
            norm_layer,
            qk_norm,
            scale_norm,
            proj_bias,
            rotate_half,
        )
        self.attn_implem = attn_type
        self.is_causal = is_causal

        self._all_attention_functions = ALL_ATTENTION_FUNCTIONS

    def forward(
        self,
        x,
        rope: Tensor | None = None,
        mask: BlockMask | Tensor | None = None,
        delta_t_emb: Tensor | None = None,
    ):
        B, N, C = x.shape

        if self.qkv is not None:
            qkv = self.qkv(x)
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
        else:
            # B, num_heads, N, C
            q = self.q_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
            k = self.k_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
            v = self.v_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)

        # QK-norm
        q, k = self.q_norm(q), self.k_norm(k)

        if rope is not None and torch.is_tensor(rope):
            npt = self.num_prefix_tokens
            # (bs, nhead, n, head_dim)
            if rope.shape[-2] + npt != N:
                logger.warning(f"Rope shape mismatch: {rope.shape[-2]} != {N}")
                rope_q = rope[:, :, :N]  # N is the sequence length
                rope_k = rope[:, :, :N]  # Q and K have same length in self-attention
            else:
                rope_q = rope
                rope_k = rope

            # Rotate them
            q = torch.cat(
                [
                    # nope tokens
                    q[:, :, :npt, :],
                    # rope tokens
                    apply_rot_embed_cat(q[:, :, npt:, :], rope_q, self.rotate_half),
                ],
                dim=2,
            ).type_as(v)
            k = torch.cat(
                [
                    k[:, :, :npt, :],
                    apply_rot_embed_cat(k[:, :, npt:, :], rope_k, self.rotate_half),
                ],
                dim=2,
            ).type_as(v)
        elif callable(rope):
            # rope is callable module, not support reg tokens
            prefixed_tokens = self.num_prefix_tokens
            q_prefixed, q_patches = q[:, :, :prefixed_tokens], q[:, :, prefixed_tokens:]
            k_prefixed, k_patches = k[:, :, :prefixed_tokens], k[:, :, prefixed_tokens:]
            q_patches = rope(q_patches, transpose=True)
            k_patches = rope(k_patches, transpose=True)
            q = torch.cat([q_prefixed, q_patches], dim=2).type_as(v)
            k = torch.cat([k_prefixed, k_patches], dim=2).type_as(v)

        if self.attn_implem != "flex_attention" and isinstance(mask, BlockMask):
            mask = mask.to_dense()

        attention_function_ = self._all_attention_functions.get(self.attn_implem, None)
        assert attention_function_ is not None, f"Attention implementation {self.attn_implem} not found in available attention functions."  # fmt: skip
        x, _ = attention_function_(
            self, q, k, v, attention_mask=mask, dropout=self.attn_drop.p
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class SoftmaxAttention2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: Optional[int] = None,
        heads_ratio: float = 1.0,
        dim=32,
        use_bias=False,
        norm=(None, "bn2d"),
        act_func=(None, None),
    ):
        super().__init__()
        heads = heads or int(out_channels // dim * heads_ratio)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dim = dim

        total_dim = heads * dim

        use_bias = to_2tuple(use_bias)
        norm = to_2tuple(norm)
        act_func = to_2tuple(act_func)

        self.qkv = ConvLayer(
            in_channels,
            3 * total_dim,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )

        self.proj = ConvLayer(
            total_dim,
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    def attn_matmul(self, qkv: torch.Tensor) -> torch.Tensor:
        B, _, H, W = list(qkv.size())

        qkv = torch.reshape(
            qkv,
            (
                B,
                -1,
                3 * self.dim,
                H * W,
            ),
        )
        qkv = torch.transpose(qkv, -1, -2)
        q, k, v = (
            qkv[..., 0 : self.dim],
            qkv[..., self.dim : 2 * self.dim],
            qkv[..., 2 * self.dim :],
        )

        out = F.scaled_dot_product_attention(
            q.contiguous(), k.contiguous(), v.contiguous()
        )

        out = torch.transpose(out, -1, -2)
        out = torch.reshape(out, (B, -1, H, W))
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.qkv(x)
        x = self.attn_matmul(x)
        x = self.proj(x)

        return x


# *==============================================================
# * Gated Attention from Qwen3
# * https://arxiv.org/abs/2505.06708
# * https://github.com/qiuzh20/gated_attention/blob/main/modeling_qwen3.py#L224
# *==============================================================


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen3SdpaAttention(nn.Module):
    """
    Qwen3 attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `Qwen3Attention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 2,
        use_qk_norm: bool = False,
        qkv_bias: bool = True,
        rms_norm_eps: float = 1e-6,
        head_dim: Optional[int] = None,
        attention_dropout: float = 0.0,
        max_position_embeddings: int = 8192,
        rope_theta: float = 10000.0,
        headwise_attn_output_gate: bool = False,
        elementwise_attn_output_gate: bool = False,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = head_dim or self.hidden_size // self.num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.is_causal = True
        self.attention_dropout = attention_dropout
        self.use_qk_norm = use_qk_norm
        self.headwise_attn_output_gate = headwise_attn_output_gate
        self.elementwise_attn_output_gate = elementwise_attn_output_gate

        if self.headwise_attn_output_gate:
            self.q_proj = nn.Linear(
                self.hidden_size,
                self.num_heads * self.head_dim + self.num_heads,
                bias=qkv_bias,
            )
        elif self.elementwise_attn_output_gate:
            self.q_proj = nn.Linear(
                self.hidden_size,
                self.num_heads * self.head_dim * 2,
                bias=qkv_bias,
            )
        else:
            self.q_proj = nn.Linear(
                self.hidden_size, self.num_heads * self.head_dim, bias=qkv_bias
            )

        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=qkv_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=qkv_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=qkv_bias
        )
        if self.use_qk_norm:
            self.q_norm = Qwen3RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = Qwen3RMSNorm(self.head_dim, eps=rms_norm_eps)

        # self.rotary_emb = Qwen3RotaryEmbedding(config=self.config)

    # Adapted from Qwen3Attention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        # position_ids: Optional[torch.LongTensor] = None,
        # past_key_value: Optional[Cache] = None,
        # output_attentions: bool = False,
        # use_cache: bool = False,
        # cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # c=nh * hd + nh; nh * hd * 2; nd * hd
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if self.headwise_attn_output_gate:
            # bs, l, nh * hd + nh -> bs, l, n_kvh, (nh * hd + nh) / n_kvh
            # if n_kvh = hd -> bs, l, nh, hd + 1

            # n_kvg = nh // n_kvh
            # if not -> bs, l, n_kvh, n_kvg * hd + n_kvg
            query_states = query_states.view(bsz, q_len, self.num_key_value_heads, -1)
            # q: bs, l, n_kvh, n_kvg * hd
            # g: bs, l, n_kvh, n_kvg
            query_states, gate_score = torch.split(
                query_states,
                [self.head_dim * self.num_key_value_groups, self.num_key_value_groups],
                dim=-1,
            )
            # g: bs, l, n_kvh * n_kvg, 1 = bs, l, nh, 1
            gate_score = gate_score.reshape(bsz, q_len, -1, 1)
            # q: bs, l, n_kvh * n_kvg, hd = bs, l, nh, hd
            query_states = query_states.reshape(
                bsz, q_len, -1, self.head_dim
            ).transpose(1, 2)
        elif self.elementwise_attn_output_gate:
            # q: bs, l, n_kvh, n_kvg * hd * 2
            query_states = query_states.view(bsz, q_len, self.num_key_value_heads, -1)
            # q: bs, l, n_kvh, n_kvg * hd
            # g: bs, l, n_kvh, n_kvg * hd
            query_states, gate_score = torch.split(
                query_states,
                [
                    self.head_dim * self.num_key_value_groups,
                    self.head_dim * self.num_key_value_groups,
                ],
                dim=-1,
            )
            # g: bs, l, n_kvh * n_kvg, hd = bs, l, nh, hd
            gate_score = gate_score.reshape(bsz, q_len, -1, self.head_dim)
            # q: bs, l, n_kvh * n_kvg, hd = bs, l, nh, hd
            query_states = query_states.reshape(
                bsz, q_len, -1, self.head_dim
            ).transpose(1, 2)
        else:
            query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(
                1, 2
            )
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        if position_embeddings is not None:
            # q, k: (bs, nh, s, hd)
            if isinstance(position_embeddings, tuple):
                cos, sin = position_embeddings
                query_states, key_states = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin
                )
            elif torch.is_tensor(position_embeddings):
                query_states = apply_rot_embed_cat(
                    query_states, position_embeddings, half=True
                )
                key_states = apply_rot_embed_cat(
                    key_states, position_embeddings, half=True
                )

        # if past_key_value is not None:
        #     cache_kwargs = {
        #         "sin": sin,
        #         "cos": cos,
        #         "cache_position": cache_position,
        #     }  # Specific to RoPE models
        #     key_states, value_states = past_key_value.update(
        #         key_states, value_states, self.layer_idx, cache_kwargs
        #     )

        # key_states: bs, head, q_len, head_dim
        # key_states = repeat_kv(key_states, self.num_key_value_groups)
        # value_states = repeat_kv(value_states, self.num_key_value_groups)

        # causal_mask = attention_mask
        # if attention_mask is not None:  # no matter the length, we just slice it
        #     causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()
        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
        # is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # attn_mask=causal_mask,
            # is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()

        if self.headwise_attn_output_gate or self.elementwise_attn_output_gate:
            attn_output = attn_output * torch.sigmoid(gate_score)

        attn_output = attn_output.view(bsz, q_len, self.num_heads * self.head_dim)

        attn_output = self.o_proj(attn_output)

        return attn_output  # , None, past_key_value


# *==============================================================
# * Natten Attention
# *==============================================================


class NatAttention1d(Attention_):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        qk_norm: type[nn.Module] | None = None,
        kernel_size=8,
        stride=2,
        dilation=2,
        dtype=torch.bfloat16,
        **block_kwargs,
    ):
        super().__init__(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            **block_kwargs,
        )
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.dtype = dtype

        if qk_norm:
            self.q_norm = qk_norm(dim)
            self.k_norm = qk_norm(dim)
        else:
            self.q_norm = self.k_norm = nn.Identity()

    def forward(self, x, mask=None, rope: Callable | None = None):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, C)
        q, k, v = qkv.unbind(2)
        dtype = q.dtype

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.type(self.dtype)
        k = k.type(self.dtype)
        v = v.type(self.dtype)

        # (bs, seq_len, nh, hd)
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).to(dtype)
        k = k.reshape(B, N, self.num_heads, C // self.num_heads).to(dtype)
        v = v.reshape(B, N, self.num_heads, C // self.num_heads).to(dtype)

        use_fp32_attention = getattr(
            self, "fp32_attention", False
        )  # necessary for NAN loss
        if use_fp32_attention:
            q, k, v = q.float(), k.float(), v.float()

        # sdpa:
        # (bs, bh, seq_len, hd)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # RoPE
        if rope is not None:
            q, k = rope(q, k)
        elif torch.is_tensor(rope):
            q = apply_rot_embed_cat(q, rope)
            k = apply_rot_embed_cat(k, rope)

        x = natten.na1d(  # (bs, h, w, nh, hd)  # type: ignore
            q,
            k,
            v,
            kernel_size=self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            is_causal=False,
            torch_compile=False,
        )
        x = x.transpose(1, 2)

        x = x.view(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = _float16_clip_value(x, self.dtype)

        return x


class NatAttention2d(nn.Module):
    def __init__(
        self,
        dim,
        kernel_size=8,
        stride=2,
        dilation=2,
        num_heads=8,
        qkv_bias=True,
        qk_norm: type[nn.Module] | None = None,
        norm_layer=None,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        scale_norm: bool = False,
        torch_compile=False,
        dtype=torch.bfloat16,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.torch_compile = torch_compile
        self.dtype = dtype
        self.is_causal = False

        self.qkv = nn.Conv2d(dim, dim * 3, 3, 1, 1, groups=dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = (
            norm_layer(dim) if scale_norm and norm_layer is not None else nn.Identity()
        )
        self.proj = nn.Conv2d(dim, dim, 1, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        if qk_norm is not None:
            self.q_norm = qk_norm(dim)
            self.k_norm = qk_norm(dim)
        else:
            self.q_norm = self.k_norm = nn.Identity()

    def _apply_rope(self, q: Float[Tensor, "b h w nh hd"], k, rope):
        if rope is None:
            return q, k

        h, w = q.shape[1:3]
        q = rearrange(q, "b h w nh hd -> b nh (h w) hd")
        k = rearrange(k, "b h w nh hd -> b nh (h w) hd")

        if callable(rope):
            q, k = rope(q, k)
        elif torch.is_tensor(rope):
            # b, nh, s, hd
            q = apply_rot_embed_cat(q, rope)
            k = apply_rot_embed_cat(k, rope)

        q = rearrange(q, "b nh (h w) hd -> b h w nh hd", h=h, w=w)
        k = rearrange(k, "b nh (h w) hd -> b h w nh hd", h=h, w=w)

        return q, k

    def forward(self, x: Float[Tensor, "b c h w"], mask=None, rope=None):
        B, C, H, W = x.shape

        qkv = self.qkv(x)
        qkv = rearrange(qkv, "b (qkv c) h w -> b qkv c h w", qkv=3)
        q, k, v = qkv.unbind(1)
        dtype = q.dtype

        q = self.q_norm(q)  # (bs, c, h, w)
        k = self.k_norm(k)

        q = rearrange(q, "b (nh hd) h w -> b h w nh hd", nh=self.num_heads)
        k = rearrange(k, "b (nh hd) h w -> b h w nh hd", nh=self.num_heads)
        v = rearrange(v, "b (nh hd) h w -> b h w nh hd", nh=self.num_heads)

        q, k = self._apply_rope(q.float(), k.float(), rope)

        q = q.to(self.dtype)
        k = k.to(self.dtype)
        v = v.to(self.dtype)

        x = natten.na2d(  # (bs, h, w, nh, hd)  # type: ignore
            q,
            k,
            v,
            kernel_size=self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            is_causal=False,
            torch_compile=self.torch_compile,
        )

        x = rearrange(x, "b h w nh hd -> b (nh hd) h w")
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = _float16_clip_value(x, self.dtype)

        return x.to(dtype)


# *==============================================================
# * Lite Linear Attention
# *==============================================================


# single scale linear attention
class LiteLA(Attention_):
    PAD_VAL = 1

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: Optional[int] = None,
        heads_ratio: float = 1.0,
        dim=32,
        eps=1e-15,
        use_bias=False,
        qk_norm=False,
        norm_eps=1e-5,
    ):
        heads = heads or int(out_dim // dim * heads_ratio)
        super().__init__(in_dim, num_heads=heads, qkv_bias=use_bias)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.dim = out_dim // heads  # TODO: need some change
        self.eps = eps

        self.kernel_func = nn.ReLU(inplace=False)
        if qk_norm:
            try:
                self.q_norm = create_norm_layer("flashrmsnorm", in_dim, eps=norm_eps)
                self.k_norm = create_norm_layer("flashrmsnorm", in_dim, eps=norm_eps)
            except Exception as e:
                self.q_norm = create_norm_layer("rmsnorm", in_dim, eps=norm_eps)
                self.k_norm = create_norm_layer("rmsnorm", in_dim, eps=norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    @torch.autocast("cuda", enabled=bool(int(os.getenv("AUTOCAST_LINEAR_ATTN", False))))
    def attn_matmul(self, q, k, v: torch.Tensor) -> torch.Tensor:
        # lightweight linear attention
        q = self.kernel_func(q)  # B, h, h_d, N
        k = self.kernel_func(k)

        q, k, v = q.float(), k.float(), v.float()

        v = F.pad(v, (0, 0, 0, 1), mode="constant", value=LiteLA.PAD_VAL)
        vk = torch.matmul(v, k)
        out = torch.matmul(vk, q)

        if out.dtype in [torch.float16, torch.bfloat16]:
            out = out.float()
        out = out[:, :, :-1] / (out[:, :, -1:] + self.eps)

        return out

    def forward(
        self, x: torch.Tensor, mask=None, HW=None, block_id=None
    ) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, C)
        q, k, v = qkv.unbind(2)  # B, N, 3, C --> B, N, C
        dtype = q.dtype

        q = self.q_norm(q).transpose(-1, -2)  # (B, N, C) -> (B, C, N)
        k = self.k_norm(k).transpose(-1, -2)  # (B, N, C) -> (B, C, N)
        v = v.transpose(-1, -2)

        q = q.reshape(B, C // self.dim, self.dim, N)  # (B, h, h_d, N)
        k = k.reshape(B, C // self.dim, self.dim, N).transpose(-1, -2)  # (B, h, N, h_d)
        v = v.reshape(B, C // self.dim, self.dim, N)  # (B, h, h_d, N)

        out = self.attn_matmul(q, k, v).to(dtype)

        out = out.view(B, C, N).permute(0, 2, 1)  # B, N, C
        out = self.proj(out)

        if not self.training:
            _float16_clip_value(out, dtype)

        return out

    @property
    def module_str(self) -> str:
        _str = type(self).__name__ + "("
        eps = f"{self.eps:.1E}"
        _str += (
            f"i={self.in_dim},o={self.out_dim},h={self.heads},d={self.dim},eps={eps}"
        )
        return _str

    def __repr__(self):
        return f"EPS{self.eps}-" + super().__repr__()


# multi-scale linear attention
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
        norm_qk: bool = False,
    ):
        super(LiteMLA, self).__init__()
        self.eps = eps
        heads = int(in_channels // dim * heads_ratio) if heads is None else heads

        self.total_dim = total_dim = heads * dim

        use_bias = to_2tuple(use_bias)
        norm = to_2tuple(norm)
        act_func = to_2tuple(act_func)

        self.dim = dim
        self.qkv = ConvLayer(
            in_channels,
            3 * total_dim,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.norm_qk = norm_qk
        if norm_qk:
            self.norm_q = create_norm_layer("tritonrmsnorm2d", total_dim)
            self.norm_k = create_norm_layer("tritonrmsnorm2d", total_dim)
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    create_conv2d(
                        3 * total_dim,
                        3 * total_dim,
                        scale,
                        groups=3 * total_dim,
                        bias=use_bias[0],
                    ),
                    create_conv2d(
                        3 * total_dim,
                        3 * total_dim,
                        1,
                        groups=3 * total_dim,
                        bias=use_bias[0],
                    ),
                )
                for scale in scales
            ]
        )
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

        if self.norm_qk:
            q, k, v = (
                qkv[:, : self.total_dim],
                qkv[:, self.total_dim : 2 * self.total_dim],
                qkv[:, 2 * self.total_dim :],
            )
            q, k = self.norm_q(q), self.norm_k(k)
            q, k, v = (
                q.reshape(B, -1, self.dim, H * W),
                k.reshape(B, -1, self.dim, H * W),
                v.reshape(B, -1, self.dim, H * W),
            )
        else:
            qkv = torch.reshape(
                qkv,
                (
                    B,
                    -1,
                    3 * self.dim,
                    H * W,
                ),
            )
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

        if self.norm_qk:
            q, k, v = (
                qkv[:, : self.total_dim],
                qkv[:, self.total_dim : 2 * self.total_dim],
                qkv[:, 2 * self.total_dim :],
            )
            q, k = self.norm_q(q), self.norm_k(k)
            q, k, v = (
                q.reshape(B, -1, self.dim, H * W),
                k.reshape(B, -1, self.dim, H * W),
                v.reshape(B, -1, self.dim, H * W),
            )
        else:
            qkv = torch.reshape(
                qkv,
                (
                    B,
                    -1,
                    3 * self.dim,
                    H * W,
                ),
            )
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
        att_map = att_map / (
            torch.sum(att_map, dim=2, keepdim=True) + self.eps
        )  # b h n n
        att_map = att_map.to(original_dtype)
        out = torch.matmul(v, att_map)  # b h d n

        out = torch.reshape(out, (B, -1, H, W))
        return out

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


class ReLULinearAttention(LiteMLA):
    "relu linear attention used in efficientvit"

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: Optional[int] = None,
        heads_ratio: float = 1.0,
        dim=32,
        use_bias=False,
        norm=(None, "ln2d"),
        act_func=(None, None),
        kernel_func="relu",
        eps=1.0e-8,
        norm_qk: bool = False,
    ):
        nn.Module.__init__(self)
        self.eps = eps
        heads = int(in_channels // dim * heads_ratio) if heads is None else heads

        self.total_dim = total_dim = heads * dim

        use_bias = to_2tuple(use_bias)
        norm = to_2tuple(norm)
        act_func = to_2tuple(act_func)

        self.dim = dim
        self.qkv = ConvLayer(
            in_channels,
            3 * total_dim,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.norm_qk = norm_qk
        if norm_qk:
            self.norm_q = create_norm_layer("tritonrmsnorm2d", total_dim)
            self.norm_k = create_norm_layer("tritonrmsnorm2d", total_dim)

        self.kernel_func = create_act_layer(kernel_func, inplace=False)

        self.proj = ConvLayer(
            total_dim,
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # generate multi-scale q, k, v
        qkv = self.qkv(x)

        H, W = list(qkv.size())[-2:]
        if H * W > self.dim:
            out = self.relu_linear_att(qkv).to(qkv.dtype)
        else:
            out = self.relu_quadratic_att(qkv)
        out = self.proj(out)

        return out


# *==============================================================
# * Fla DGN Attention
# *==============================================================


...

if __name__ == "__main__":
    """
        python -m src.stage2.layers.attention
    """
    # Test LiteLA
    print("Testing LiteLA...")
    lite_la = LiteLA(32, 32, heads_ratio=1.0)
    lite_la = lite_la.cuda()
    x = torch.randn(1, 196, 32).cuda()
    y = lite_la(x)
    print(f"LiteLA output shape: {y.shape}")

    # Test LiteMLA
    print("\nTesting LiteMLA...")
    lite_mla = LiteMLA(32, 32, heads_ratio=1.0, dim=8, scales=(5,))
    lite_mla = lite_mla.cuda()
    x_2d = torch.randn(1, 32, 14, 14).cuda()
    y_mla = lite_mla(x_2d)
    print(f"LiteMLA output shape: {y_mla.shape}")

    # Test ReLULinearAttention
    print("\nTesting ReLULinearAttention...")
    relu_la = ReLULinearAttention(32, 32, heads_ratio=1.0, dim=8)
    relu_la = relu_la.cuda()
    y_relu = relu_la(x_2d)
    print(f"ReLULinearAttention output shape: {y_relu.shape}")

    print("\nAll tests completed successfully!")
