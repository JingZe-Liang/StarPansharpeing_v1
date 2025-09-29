from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import natten
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from sympy import use
from timm.layers.attention import Attention as Attention_
from torch import Tensor

from src.utilities.logging import log


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


class Attention(Attention_):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        qk_norm: type[nn.Module] | None = None,
        **block_kwargs,
    ):
        super().__init__(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            **block_kwargs,
        )

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

        q = q.reshape(B, N, self.num_heads, C // self.num_heads).to(dtype)
        k = k.reshape(B, N, self.num_heads, C // self.num_heads).to(dtype)
        v = v.reshape(B, N, self.num_heads, C // self.num_heads).to(dtype)

        # RoPE
        if rope is not None:
            q, k = rope(q, k)

        use_fp32_attention = getattr(
            self, "fp32_attention", False
        )  # necessary for NAN loss
        if use_fp32_attention:
            q, k, v = q.float(), k.float(), v.float()

        # sdpa
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if mask is not None and mask.ndim == 2:
            mask = (1 - mask.to(x.dtype)) * -10000.0
            mask = mask[:, None, None].repeat(1, self.num_heads, 1, 1)

        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
        )
        x = x.transpose(1, 2)

        x = x.view(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = _float16_clip_value(x)

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


@dataclass
class Qwen3Config:
    """Configuration for Qwen3 SDPA Attention module."""


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
        # if layer_idx is None:
        #     log(
        #         f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
        #         "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
        #         "when creating this class.",
        #         warn_once=True,
        #     )

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
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
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

        # RoPE
        if rope is not None:
            q, k = rope(q, k)

        use_fp32_attention = getattr(
            self, "fp32_attention", False
        )  # necessary for NAN loss
        if use_fp32_attention:
            q, k, v = q.float(), k.float(), v.float()

        # sdpa
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

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
        if rope:
            h, w = q.shape[1:3]
            q = rearrange(q, "b h w nh hd -> b (h w) nh hd")
            k = rearrange(k, "b h w nh hd -> b (h w) nh hd")

            q, k = rope(q, k)

            q = rearrange(q, "b (h w) nh hd -> b h w nh hd", h=h, w=w)
            k = rearrange(k, "b (h w) nh hd -> b h w nh hd", h=h, w=w)

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


if __name__ == "__main__":
    g_attn = Qwen3SdpaAttention(
        Qwen3Config(
            hidden_size=256,
            num_attention_heads=8,
            num_key_value_heads=8,
            use_qk_norm=True,
        )
    )
    x = torch.randn(1, 384, 256)
    print(g_attn(x).shape)
