"""
Self / Cross Attention sink with learnable tokens
Features:
1. Mingtok tokenizer with downsample / upsample
2. Masked Autoencoder (no drop tokens)
"""

from functools import partial
import math
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, List, Optional, Tuple, Type, Union, cast, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from einx import get_at
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
from timm.layers.pos_embed import resample_abs_pos_embed, resample_abs_pos_embed_nhwc
from timm.layers.pos_embed_sincos import (
    RotaryEmbeddingCat,
    RotaryEmbeddingDinoV3,
    apply_rot_embed_cat,
    create_rope_embed,
    get_mixed_freqs,
    get_mixed_grid,
)
from timm.models.naflexvit import NaFlexRopeIterator, get_init_weights_vit, named_apply
from timm.models.vision_transformer import VisionTransformer
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)
from torch.utils.checkpoint import checkpoint
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from typing_extensions import Annotated

from src.utilities.network_utils import (
    compile_decorator,
    model_compiled_flag,
    null_decorator,
)

from .value_residual import mix_value_residual
from .blocks import (
    AdaptiveInputConvLayer,
    AdaptiveOutputConvLayer,
    AdaptiveOutputLinearLayer,
    block_basic_init,
)
from .mask import random_masking_mae, random_masking_no_drop
from .patching import (
    AdaptivePatchEmbedding,
    AdaptiveProgressivePatchEmbedding,
    AdaptiveProgressivePatchUnembedding,
    create_unpatcher,
)
from .resample import MingtokDownsampleShortCut, MingtokUpsampleAverage
from .rope import (
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed,
    resample_1d_pe,
)
from .variants.cross_attn import CrossAttention
from .variants.mlp import SwiGLU
from .moe import MoEFFN

JVP_FLASH_ATTN_ENABLED = False
try:
    from jvp_flash_attention.jvp_attention import JVPAttn
    from jvp_flash_attention.jvp_attention import attention as jvp_attention

except ImportError:
    JVP_FLASH_ATTN_ENABLED = False


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads for Grouped Query Attention (GQA).

    This is the equivalent of torch.repeat_interleave(x, n_rep, dim=1).
    The hidden states go from (batch, num_kv_heads, seqlen, head_dim) to
    (batch, num_attention_heads, seqlen, head_dim).
    """
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def _jvp_math_attention(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
):
    # assert dropout == 0, "Dropout is not supported for JVP Attention"
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attention_mask,
            dropout,
        ), None


def _jvp_flash_attention(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
):
    assert dropout == 0, "Dropout is not supported for JVP Attention"
    x = jvp_attention(query, key, value, attn_mask=attention_mask)
    return x, None


class Attention(Attention_):
    config = SimpleNamespace(_attn_implementation="flash_attention_2")  # for flash_attention_2 config

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
        is_causal: bool = False,
        jvp=False,
        delta_t_aware: bool = False,
        rotate_half=False,
        v_residual: bool = False,
    ):
        norm_layer = get_norm_layer(norm_layer) if isinstance(norm_layer, str) else norm_layer
        if norm_layer is None:
            norm_layer = nn.Identity
        norm_layer = cast(Type[nn.Module], norm_layer)
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
        self.jvp = jvp
        self.delta_t_aware = delta_t_aware
        if delta_t_aware:
            self.qkv_delta_t = nn.Linear(dim, dim * 3)

        self._all_attention_functions = ALL_ATTENTION_FUNCTIONS
        if jvp and not JVP_FLASH_ATTN_ENABLED:
            logger.warning(
                "JVP Flash Attention is not enabled. Please install it by running `pip install jvp_flash_attention` "
                "or the flash attention will not be used and fall back to math attention kernel."
            )
            self._all_attention_functions["jvp_math_attention"] = _jvp_math_attention
            self.attn_implem = "jvp_math_attention"
        elif jvp and JVP_FLASH_ATTN_ENABLED:
            self._all_attention_functions["jvp_flash_attention"] = _jvp_flash_attention
            self.attn_implem = "jvp_flash_attention"

        self.v_residual: bool = v_residual
        if self.v_residual:
            self.v_residual_lamb1 = nn.Parameter(torch.tensor(0.5))
            self.v_residual_lamb2 = nn.Parameter(torch.tensor(0.5))

    def project_v(self, x: Tensor, *, delta_t_emb: Tensor | None = None) -> Tensor:
        B, N, C = x.shape

        if self.qkv is not None:
            qkv = self.qkv(x)
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            _, _, v = qkv.unbind(0)  # B, num_heads, N, head_dim
        else:
            assert self.v_proj is not None
            v = self.v_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)

        if self.delta_t_aware and delta_t_emb is not None:
            qkv_delta = self.qkv_delta_t(delta_t_emb)
            _, _, vd = qkv_delta.reshape(B, -1, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            v = v + vd

        return v

    @compile_decorator
    def forward(
        self,
        x,
        rope: Tensor | None = None,
        attention_mask: BlockMask | Tensor | None = None,
        delta_t_emb: Tensor | None = None,
        v_residual_v1: Tensor | None = None,
    ):
        B, N, C = x.shape

        if self.qkv is not None:
            qkv = self.qkv(x)
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
        else:
            # B, num_heads, N, C
            assert self.q_proj is not None
            assert self.k_proj is not None
            assert self.v_proj is not None
            q = self.q_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
            k = self.k_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
            v = self.v_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)

        # Meanflow delta-time (t-r) awareness
        if self.delta_t_aware and delta_t_emb is not None:
            qkv_delta = self.qkv_delta_t(delta_t_emb)
            qd, kd, vd = qkv_delta.reshape(B, -1, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = q + qd, k + kd, v + vd

        if self.v_residual:
            if v_residual_v1 is None:
                raise ValueError("v_residual requires v_residual_v1")
            v = mix_value_residual(v, v_residual_v1, self.v_residual_lamb1, self.v_residual_lamb2)

        # QK-norm
        q, k = self.q_norm(q), self.k_norm(k)

        if rope is not None and torch.is_tensor(rope):
            npt = self.num_prefix_tokens
            # (bs, nhead, n, head_dim)
            if rope.shape[-2] + npt != N:
                logger.warning(f"Rope shape mismatch: {rope.shape[-2]} != {N}")
                rope_q = torch.narrow(rope, -2, 0, N)
                rope_k = torch.narrow(rope, -2, 0, N)
                # rope_q = rope[:, :, :N]  # N is the sequence length
                # rope_k = rope[:, :, :N]  # Q and K have same length in self-attention
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
            prefixed_tokens = self.num_prefix_tokens
            q_prefixed, q_patches = q[:, :, :prefixed_tokens], q[:, :, prefixed_tokens:]
            k_prefixed, k_patches = k[:, :, :prefixed_tokens], k[:, :, prefixed_tokens:]
            q_patches = rope(q_patches, transpose=True)
            k_patches = rope(k_patches, transpose=True)
            q = torch.cat([q_prefixed, q_patches], dim=2)
            k = torch.cat([k_prefixed, k_patches], dim=2)
        elif rope is not None:
            raise ValueError("Invalid rope type")

        if self.attn_implem != "flex_attention" and isinstance(attention_mask, BlockMask):
            attention_mask = attention_mask.to_dense()

        # flex attention will be compiled inside
        attention_function_ = self._all_attention_functions.get(self.attn_implem, None)
        assert attention_function_ is not None, f"Attention implementation {self.attn_implem} not found in available attention functions."  # fmt: skip
        x, _ = attention_function_(self, q, k, v, attention_mask=attention_mask, dropout=self.attn_drop.p)

        if x.ndim == 4:
            # Most SDPA/flash attention implementations return [B, heads, N, head_dim].
            if x.shape[1] == self.num_heads:
                x = x.transpose(1, 2).reshape(B, N, C)
            elif x.shape[2] == self.num_heads:
                x = x.reshape(B, N, C)
            else:
                raise ValueError(f"Unexpected attention output shape {tuple(x.shape)} for num_heads={self.num_heads}")
        elif x.ndim != 3:
            raise ValueError(f"Unexpected attention output ndim={x.ndim}, shape={tuple(x.shape)}")

        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class GatedAttention(nn.Module):
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
        norm_layer: str | Callable | None = "layernorm",
        use_qk_norm: bool = False,
        qkv_bias: bool = True,
        rms_norm_eps: float = 1e-6,
        head_dim: Optional[int] = None,
        attention_dropout: float = 0.0,
        max_position_embeddings: int = 8192,
        rope_theta: float = 10000.0,
        headwise_attn_output_gate: bool = False,
        elementwise_attn_output_gate: bool = False,
        is_causal: bool = False,
        num_prefix_tokens: int = 0,
        attn_type: str = "sdpa",
        layer_idx: Optional[int] = None,
        v_residual: bool = False,
        *,
        jvp=False,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_prefix_tokens = num_prefix_tokens
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = head_dim or self.hidden_size // self.num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.is_causal = is_causal
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
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=qkv_bias)

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
        self.proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=qkv_bias)
        if self.use_qk_norm:
            norm_layer = norm_layer or "layernorm"
            self.q_norm = create_norm_layer(norm_layer, self.head_dim, eps=rms_norm_eps)
            self.k_norm = create_norm_layer(norm_layer, self.head_dim, eps=rms_norm_eps)

        # self.rotary_emb = Qwen3RotaryEmbedding(config=self.config)

        self.attn_implem = attn_type
        self._all_attention_functions = ALL_ATTENTION_FUNCTIONS
        if jvp and not JVP_FLASH_ATTN_ENABLED:
            logger.warning(
                "JVP Flash Attention is not enabled. Please install it by running `pip install jvp_flash_attention` "
                "or the flash attention will not be used and fall back to math attention kernel."
            )
            self._all_attention_functions["jvp_math_attention"] = _jvp_math_attention
            self.attn_implem = "jvp_math_attention"
        elif jvp and JVP_FLASH_ATTN_ENABLED:
            self._all_attention_functions["jvp_flash_attention"] = _jvp_flash_attention
            self.attn_implem = "jvp_flash_attention"

        self.v_residual: bool = v_residual
        if self.v_residual:
            self.v_residual_lamb1 = nn.Parameter(torch.tensor(0.5))
            self.v_residual_lamb2 = nn.Parameter(torch.tensor(0.5))

    def project_v(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()
        value_states = self.v_proj(hidden_states)
        return value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    # Adapted from Qwen3Attention.forward
    @compile_decorator
    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        # position_ids: Optional[torch.LongTensor] = None,
        # past_key_value: Optional[Cache] = None,
        # output_attentions: bool = False,
        # use_cache: bool = False,
        # cache_position: Optional[torch.LongTensor] = None,
        rope: Annotated[torch.Tensor, "Rope embedding concat"] | None = None,  # necessary, but kept here for BC
        v_residual_v1: torch.Tensor | None = None,
    ):
        bsz, q_len, _ = hidden_states.size()

        # c=nh * hd + nh; nh * hd * 2; nd * hd
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        ###### Attention gate
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
            query_states = query_states.reshape(bsz, q_len, -1, self.head_dim).transpose(1, 2)
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
            query_states = query_states.reshape(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        else:
            gate_score = None  # no gate when neither headwise nor elementwise gate is used
            query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        kv_len = key_states.shape[2]

        if self.v_residual:
            if v_residual_v1 is None:
                raise ValueError("v_residual requires v_residual_v1")
            value_states = mix_value_residual(value_states, v_residual_v1, self.v_residual_lamb1, self.v_residual_lamb2)

        ###### QK-norm
        if self.use_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        ####### Rope
        if rope is not None:  # (seq_max_l, head_dim)
            npt = self.num_prefix_tokens
            N = hidden_states.shape[-2]
            # (bs, nhead, n, head_dim) or (n, head_dim)
            if rope.shape[-2] + npt != N:
                logger.warning(f"Rope shape mismatch: {rope.shape[-2]} != {N}")
                rope_q = torch.narrow(rope, -2, 0, N)
                rope_k = torch.narrow(rope, -2, 0, N)
            else:
                rope_q = rope
                rope_k = rope

            query_states = torch.cat(
                [
                    query_states[:, :, :npt, :],
                    apply_rot_embed_cat(query_states[:, :, npt:, :], rope_q),
                ],
                dim=2,
            ).type_as(value_states)
            key_states = torch.cat(
                [
                    key_states[:, :, :npt, :],
                    apply_rot_embed_cat(key_states[:, :, npt:, :], rope_k),
                ],
                dim=2,
            ).type_as(value_states)

        # if past_key_value is not None:
        #     cache_kwargs = {
        #         "sin": sin,
        #         "cos": cos,
        #         "cache_position": cache_position,
        #     }  # Specific to RoPE models
        #     key_states, value_states = past_key_value.update(
        #         key_states, value_states, self.layer_idx, cache_kwargs
        #     )

        # GQA: repeat K/V to match Q heads when num_key_value_heads != num_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # causal_mask = attn_mask
        # if attn_mask is not None:  # no matter the length, we just slice it
        #     causal_mask = attn_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda":
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()
        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
        # is_causal = True if causal_mask is None and q_len > 1 else False

        # attn_output = torch.nn.functional.scaled_dot_product_attention(
        #     query_states,
        #     key_states,
        #     value_states,
        #     dropout_p=self.attention_dropout if self.training else 0.0,
        #     is_causal=self.is_causal,
        #     enable_gqa=True,
        #     attn_mask=attn_mask,
        # )

        # Jvp supported flash attention
        attention_function_ = self._all_attention_functions.get(self.attn_implem, None)
        assert attention_function_ is not None, (
            f"Attention implementation {self.attn_implem} not found in available attention functions."
        )
        attn_output, _ = attention_function_(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=attn_mask,
            dropout=self.attention_dropout if self.training else 0.0,
        )

        if self.headwise_attn_output_gate or self.elementwise_attn_output_gate:
            if gate_score is None:
                raise ValueError("gate_score must be provided when attention output gate is enabled")
            attn_output = attn_output * torch.sigmoid(gate_score)

        # attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.proj(attn_output)  # compactible with timm attentions

        return attn_output


class _HC_Attention(nn.Module):
    def __init__(
        self,
        norm: nn.Module,
        attention: nn.Module,
        layer_scale: nn.Module,
        drop: nn.Module,
    ) -> None:
        super().__init__()
        self.norm = norm
        self.attention = attention
        self.layer_scale = layer_scale
        self.drop = drop

    def forward(self, x: Tensor, rope: Tensor | None = None, v_residual_v1: Tensor | None = None) -> Tensor:
        x = self.norm(x)
        x = self.attention(x, rope=rope, v_residual_v1=v_residual_v1)
        x = self.layer_scale(x)
        x = self.drop(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        n_kv_heads=None,
        qkv_bias=False,
        norm_layer="layernorm",
        attn_drop=0.0,
        drop_path=0.0,
        proj_drop=0.0,
        fused_type=None,
        attn_type="sdpa",
        ffn_type="swiglu",
        qk_norm=True,
        is_causal=False,
        mlp_ratio=4,
        ffn_drop=0.0,
        num_prefix_tokens=0,
        use_gate=False,
        layer_idx=None,
        jvp=False,
        v_residual: bool = False,
        moe_kwargs: dict[str, Any] | None = None,
        hyperconnection_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.use_gate = use_gate
        self.layer_idx = layer_idx

        self.sa: GatedAttention | Attention
        if self.use_gate:
            self.sa = GatedAttention(
                dim,
                n_heads,
                n_kv_heads or n_heads,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                use_qk_norm=qk_norm,
                elementwise_attn_output_gate=False,
                is_causal=is_causal,
                num_prefix_tokens=num_prefix_tokens,
                attn_type=attn_type,
                layer_idx=layer_idx,
                v_residual=v_residual,
                jvp=jvp,
            )
        else:
            self.sa = Attention(
                dim,
                num_heads=n_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                norm_layer=norm_layer,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                attn_type=attn_type,
                num_prefix_tokens=num_prefix_tokens,
                is_causal=is_causal,
                v_residual=v_residual,
                jvp=jvp,
            )

        if ffn_type == "swiglu":
            self.ffn = SwiGLU(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                norm_layer=None,
                bias=True,
                drop=ffn_drop,
                is_fused=fused_type,
            )
        elif ffn_type == "moe":
            moe_kwargs = moe_kwargs or {}
            # num_layers is required by MoEFFN/TransformerConfig
            num_layers = moe_kwargs.pop("num_layers", 12)

            self.ffn = MoEFFN(
                dim=dim,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                num_layers=num_layers,
                layer_idx=layer_idx if layer_idx is not None else 0,
                ffn_drop=ffn_drop,
                fused_type=fused_type,
                assume_bsh=True,
                **moe_kwargs,
            )

        self.norm1 = create_norm_layer(norm_layer, dim)
        self.norm2 = create_norm_layer(norm_layer, dim)
        self.ls1 = LayerScale(dim, 1e-2)
        self.ls2 = LayerScale(dim, 1e-2)
        self.dp1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.dp2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        logger.debug(
            f"Layer {layer_idx} uses Attention {self.sa.__class__.__name__} - is_causal={is_causal} - "
            f"attn_type={attn_type} norm_layer={norm_layer} mlp_ratio={mlp_ratio} - "
            f"FFN {self.ffn.__class__.__name__} with fused_type={fused_type}"
        )

        self._forward_type = "attention_block" if hyperconnection_kwargs is None else "attention_block_mhc"

        # Hyper-connection mHC from Deepseek
        self.hyperconnection_kwargs = hyperconnection_kwargs
        if hyperconnection_kwargs is not None:
            init_hc_fn = hyperconnection_kwargs.get("init_hc")
            assert init_hc_fn is not None and layer_idx is not None
            init_hc_kwargs = dict(
                mhc=hyperconnection_kwargs.get("mhc", False),
                sinkhorn_iters=hyperconnection_kwargs.get("sinkhorn_iters", 10),
                sinkhorn_tau=hyperconnection_kwargs.get("sinkhorn_tau", 0.05),
                mhc_h_res_proj=hyperconnection_kwargs.get("mhc_h_res_proj", "sinkhorn"),
                ns_steps=hyperconnection_kwargs.get("ns_steps", 5),
                ns_eps=hyperconnection_kwargs.get("ns_eps", 1e-7),
                ns_coeffs=hyperconnection_kwargs.get("ns_coeffs", (3.0, -3.2, 1.2)),
            )
            self.hc_attn = init_hc_fn(
                dim=dim,
                branch=_HC_Attention(self.norm1, self.sa, self.ls1, self.dp1),
                layer_index=layer_idx * 2,
                **init_hc_kwargs,
            )
            self.hc_mlp = init_hc_fn(
                dim=dim,
                branch=nn.Sequential(self.norm2, self.ffn, self.ls2, self.dp2),
                layer_index=layer_idx * 2 + 1,
                **init_hc_kwargs,
            )

    def _forward_attention_block(self, x, rope=None, v_residual_v1: Tensor | None = None):
        x = x + self.dp1(self.ls1(self.sa(self.norm1(x), rope=rope, v_residual_v1=v_residual_v1)))
        x = x + self.dp2(self.ls2(self.ffn(self.norm2(x))))
        return x

    def _forward_attention_block_mhc(self, x, rope=None, v_residual_v1: Tensor | None = None):
        x = self.hc_attn(x, rope=rope, v_residual_v1=v_residual_v1)
        x = self.hc_mlp(x)
        return x

    def forward(self, *args, **kwargs):
        if self._forward_type == "attention_block":
            return self._forward_attention_block(*args, **kwargs)
        elif self._forward_type == "attention_block_mhc":
            return self._forward_attention_block_mhc(*args, **kwargs)
        else:
            raise ValueError(f"Unknown forward type: {self._forward_type}")


class AttentionBlockCondition(AttentionBlock):
    def __init__(
        self,
        dim,
        n_heads,
        time_embed_dim,
        cxt_embed_dim,
        n_kv_heads=None,
        qkv_bias=False,
        norm_layer="layernorm",
        attn_drop=0.0,
        drop_path=0.0,
        proj_drop=0.0,
        fused_type=None,
        attn_type="sdpa",
        qk_norm=True,
        is_causal=False,
        mlp_ratio=4,
        ffn_drop=0.0,
        num_prefix_tokens=0,
        use_gate=False,
        layer_idx=None,
        jvp=False,
        fuse_t_z=True,
    ):
        super().__init__(
            dim,
            n_heads,
            n_kv_heads,
            qkv_bias,
            norm_layer,
            attn_drop,
            drop_path,
            proj_drop,
            fused_type,
            attn_type,
            qk_norm,
            is_causal,
            mlp_ratio,
            ffn_drop,
            num_prefix_tokens,
            use_gate,
            layer_idx,
            jvp,
        )
        self.time_embed_dim = time_embed_dim
        self.ctx_embed_dim = cxt_embed_dim

        ## Add time embedding, context embedding
        self.fuse_t_z = fuse_t_z
        if not fuse_t_z:
            self.t_proj = nn.Sequential(nn.SiLU(), nn.Linear(self.time_embed_dim, dim))
            self.ctx_proj = nn.Sequential(nn.SiLU(), nn.Linear(self.ctx_embed_dim, dim))
        else:
            self.cond_proj = nn.Sequential(
                create_norm_act_layer("layernorm", dim, "silu"),
                nn.Linear(dim, dim * 2),
            )

        self._forward_type = "condition_attention_block"

    def _get_hw(self, x, inp_shape=None):
        if inp_shape is not None:
            h, w = inp_shape[-2:]
        else:
            L = x.shape[1]
            h = w = int(math.sqrt(L))

        return h, w

    def _interp_z_to_x(self, x, z, inp_shape=None):
        if z.shape[1] == x.shape[1]:
            return z

        h, w = self._get_hw(x, inp_shape)
        z_h, z_w = self._get_hw(z, None)

        # to 2d
        z_2d = rearrange(z, "b (zh zw) c -> b c zh zw", zh=z_h, zw=z_w)
        x_2d = rearrange(x, "b (xh xw) c -> b c xh xw", xh=h, xw=w)
        # interpolate
        z_2d_interp = F.interpolate(
            z_2d,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )
        # to 1d
        z_1d = rearrange(z_2d_interp, "b c zh zw -> b (zh zw) c")
        return z_1d

    @compile_decorator
    def _forward_condition_attention_block(
        self, x, c, t=None, inp_shape=None, rope=None, v_residual_v1: Tensor | None = None
    ):
        if not self.fuse_t_z:
            # t, c not fused, should be passed both
            assert t is not None and c is not None
            # interpolate z into x
            z_emb = self.ctx_proj(c)
            z_emb = self._interp_z_to_x(x, z_emb, inp_shape)
            t_emb = self.t_proj(t)[..., None, None]
            cond_emb = z_emb + t_emb
        else:
            # c is passed only, t and z are fused together
            cond_emb = self.ctx_proj(c)

        cond_emb = self.cond_proj(cond_emb)
        c_scale, c_shift = torch.chunk(cond_emb, 2, dim=-1)

        # AdaLN
        x = self.norm1(x)
        x = x * (1 + c_scale) + c_shift

        # Attention + FFN
        x = x + self.dp1(self.ls1(self.sa(x, rope=rope, v_residual_v1=v_residual_v1)))
        x = x + self.dp2(self.ls2(self.ffn(self.norm2(x))))
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        ctx_dim,
        n_q_heads,
        n_kv_heads=None,
        qkv_bias=False,
        norm_layer="layernorm",
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
        x = self.layer_scales[1] * self.cross_attention(x, ctx, rope=rope, mask=mask) + x  # fmt: skip
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
    norm_layer: str = "layernorm"
    drop_path: float = 0.0
    attn_type: str = "sdpa"
    # Cross attention specific parameters
    n_kv_heads: Optional[int] = None
    qk_norm: str | None = "layernorm"
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
                "query_proj": nn.Linear(cfg.ctx_dim, cfg.dim) if cfg.query_strategy != "learnable" else nn.Identity(),
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
        self.register_buffer("query_latent_pe", torch.as_tensor(query_latent_pe), persistent=False)
        self.register_buffer("ctx_latent_pe", torch.as_tensor(ctx_latent_pe), persistent=False)
        self.query_latent_pe: nn.Buffer
        self.ctx_latent_pe: nn.Buffer

        self.rope = RotaryEmbeddingCat(
            dim=cfg.dim // cfg.heads * 2,  # since this class uses div 4.
            temperature=cfg.rope_theta,
            max_res=cfg.q_seq_len,
            in_pixels=False,
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
        if x.shape[1] != self.ctx_latent_pe.shape[1]:
            if x_hw is None:
                hw = math.sqrt(x.shape[1])
                assert hw.is_integer(), f"Cannot resample 2D PE with non-square number of tokens: {x.shape[1]}"
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
        x = self.projections["ctx_proj"](x.flatten(2).permute(0, 2, 1))  # (bs, ctxN, dim)

        # Add PEs
        q, x = self._with_pos_embed(q, x, x_hw=hw)

        # Cross attention blocks
        rope_cat = self.rope_cat
        for blk in self.blocks:
            x = blk(q, x, mask=None, rope=rope_cat)

        # Output projection
        x = self.projections["out_proj"](x)

        return x


class TransformerTokenizer(nn.Module):
    def __init__(
        self,
        in_chan,
        embed_dim,
        out_chan=None,
        img_size=384,
        patch_size=16,
        out_patch_size=1,
        depth=12,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        dropout=0.0,
        attention_dropout=0.0,
        norm_layer="layernorm",
        patch_embeder_with_norm=False,
        drop_path=0.0,
        attn_type="sdpa",
        attn_blk_type="AttentionBlock",
        n_reg_tokens: int = 0,
        projections={"input": None, "output": None},
        additional_pe=False,
        pe_type="rope",  # ['learn', 'rope']
        rope_kwargs={"temperature": 10000.0},
        head: str | None = None,
        other_blk_kwargs: dict = {},
        # MAE masks
        mask_train_ratio=0.0,  # MAE encoder
        mask_refill: bool = False,  # MAE decoder
        mask_drop_type: str = "mae",  # MAE drop, or no drop
        last_norm: str | None = None,
        is_causal: bool = False,
        patcher_type: str = "patch_embedder",
        with_cls_token: bool = False,
        # mHC
        hc_streams: int = -1,  # -1 means no HC, only residual connection
        hc_implem: str = "naive",
        hc_other_kwargs: dict = {"sinkhorn_iters": 10, "sinkhorn_tau": 0.05},
        # others
        patch_prog_dims: list[int] | None = None,
        unpatch_prog_dims: list[int] | None = None,
        v_residual: bool = False,
    ):
        super().__init__()
        self.patch_embed: nn.Module
        self.in_chan = in_chan
        self.embedded_dim = embed_dim
        self.n_reg_tokens: int = n_reg_tokens
        self.v_residual: bool = v_residual
        self.num_prefix_tokens: int = n_reg_tokens + int(with_cls_token)  # cls + reg
        self.grad_checkpointing: bool = False

        self._build_hyperconnection(hc_streams, hc_implem, hc_other_kwargs)
        self._build_patch_embedding(
            patcher_type,
            img_size,
            patch_size,
            in_chan,
            embed_dim,
            patch_prog_dims,
            get_norm_layer(norm_layer) if patch_embeder_with_norm else None,
        )
        self._build_projections(projections, in_chan, embed_dim)
        self._build_positional_embedding(
            pe_type,
            additional_pe,
            rope_kwargs,
            patch_size,
            img_size,
            embed_dim,
            num_heads,
            depth,
            with_cls_token,
        )
        self._build_transformer_layers(
            drop_path,
            depth,
            embed_dim,
            norm_layer,
            mlp_ratio,
            attn_blk_type,
            num_heads,
            qkv_bias,
            attention_dropout,
            dropout,
            attn_type,
            is_causal,
            self.num_prefix_tokens,
            hc_other_kwargs,
            {**other_blk_kwargs, "v_residual": v_residual},
        )
        self._build_cls_reg_tokens(n_reg_tokens, with_cls_token, embed_dim)
        self._build_head(head, out_chan, out_patch_size, norm_layer, embed_dim, unpatch_prog_dims)
        self._build_mask_tokens(mask_train_ratio, mask_refill, mask_drop_type, embed_dim)
        self._build_last_norm(last_norm, embed_dim)

        # self.init_weights()

    def _get_value_residual_input(self, block: nn.Module, x: Tensor) -> Tensor:
        block_any = cast(Any, block)
        if hasattr(block_any, "hc_attn"):
            branch_input, _, _ = block_any.hc_attn.width_connection(x)
            return block_any.norm1(branch_input)
        return block_any.norm1(x)

    def _compute_value_residual_v1(self, x: Tensor) -> Tensor:
        if len(self.layers) == 0:
            raise ValueError("v_residual requires depth >= 1")
        first_block_any = cast(Any, self.layers[0])
        x_in = self._get_value_residual_input(first_block_any, x)
        if not hasattr(first_block_any, "sa") or not hasattr(first_block_any.sa, "project_v"):
            raise ValueError("v_residual requires an AttentionBlock whose self-attention supports project_v()")
        return first_block_any.sa.project_v(x_in)

    def _build_hyperconnection(self, hc_streams: int = 1, implem: str = "naive", hc_other_kwargs: dict = {}):
        self._hc_implem = implem
        self._hc_streams = hc_streams
        self.hc_other_kwargs = hc_other_kwargs

        self.init_hc_fn = None
        if hc_streams < 0:
            self._use_hc = False
            return
        else:
            assert hc_streams >= 1
            assert implem in ("naive", "cuda")
            logger.info(
                f"Using <green>Hyperconnection</green> - this is experimental, with kwargs: "
                f"{self._hc_implem=}, {self._hc_streams=}, {self.hc_other_kwargs=}"
            )

            self._use_hc = True

            from .mHC import (
                HyperConnections,
                get_init_and_expand_reduce_stream_functions,
                HyperConnectionsCUDA,
                get_init_and_expand_reduce_stream_functions_cuda,
            )

            hc_cls, init_expand_reduce_fm = {
                "naive": (HyperConnections, get_init_and_expand_reduce_stream_functions),
                "cuda": (HyperConnectionsCUDA, get_init_and_expand_reduce_stream_functions_cuda),
            }[implem]

            self.init_hc_fn, self.hc_expand_stream, self.reduce_stream = init_expand_reduce_fm(hc_streams)

    def _build_patch_embedding(
        self,
        patcher_type: str,
        img_size: int,
        patch_size: int,
        in_chan: int,
        embed_dim: int,
        patch_prog_dims: list[int] | None = None,
        norm_layer: type[nn.Module] | None = None,
    ):
        # Patch embedding
        if patcher_type in ("patch_embedder", "progressive_patch_embedder"):
            if patcher_type == "progressive_patch_embedder":
                raise NotImplementedError("Bug here.")

                assert patch_prog_dims is not None, "patch_prog_dims must be provided for progressive patch embedding"
                patch_embed = AdaptiveProgressivePatchEmbedding(
                    img_size=(img_size, img_size),  # placeholder, not used
                    patch_size=patch_size,
                    progressive_dims=patch_prog_dims,
                    in_chans=in_chan,
                    embed_dim=embed_dim,
                    output_fmt="NLC",
                )
            else:
                patch_embed = AdaptivePatchEmbedding(
                    img_size=(img_size, img_size),  # placeholder, not used
                    patch_size=patch_size,
                    in_chans=in_chan,
                    embed_dim=embed_dim,
                    output_fmt="NLC",
                    norm_layer=norm_layer,
                )
            self.patch_embed = patch_embed
            self.patch_size = patch_size
            self.grid_size = patch_embed.grid_size
            self.n_patches = patch_embed.num_patches
        elif patcher_type == "linear":
            # Input tensor is 1D Tensor
            self.patch_embed = nn.Linear(in_chan, embed_dim)
            self.patch_size = 1
            self.grid_size = (img_size, img_size)
            self.n_patches = img_size * img_size
        else:
            raise ValueError(f"Unknown patcher type: {patcher_type}")

    def _build_projections(self, projections: dict, in_chan: int, embed_dim: int):
        # Projections
        # Encoder: forward output downsample layer
        # Decoder: forward input upsample layer
        input_proj_type = projections.get("input", None)  # decoder
        output_proj_type = projections.get("output", None)  # encoder
        input_proj = output_proj = nn.Identity()
        if output_proj_type == "ds_shortcut":
            output_proj = MingtokDownsampleShortCut(in_chan, embed_dim)
        if input_proj_type == "us_average":
            input_proj = MingtokUpsampleAverage(in_chan, embed_dim)
        self.projections = nn.ModuleDict({"input_proj": input_proj, "output_proj": output_proj})

    def _build_transformer_layers(
        self,
        drop_path: float,
        depth: int,
        embed_dim: int,
        norm_layer: str,
        mlp_ratio: float,
        attn_blk_type: str,
        num_heads: int,
        qkv_bias: bool,
        attention_dropout: float,
        dropout: float,
        attn_type: str,
        is_causal: bool,
        num_prefix_tokens: int,
        hc_kwargs: dict,
        other_blk_kwargs: dict,
    ):
        # Layers
        layers = []
        drop_path_ps = torch.linspace(0, drop_path, depth).tolist()
        block_cls = {
            "AttentionBlock": AttentionBlock,
            "AttentionBlockCondition": AttentionBlockCondition,
        }[attn_blk_type]

        if attn_blk_type == "AttentionBlock" and self.init_hc_fn is not None and self._use_hc:
            hyperconnection_kwargs = {"init_hc": self.init_hc_fn, **hc_kwargs}
            other_blk_kwargs.update(hyperconnection_kwargs=hyperconnection_kwargs)
        elif self.init_hc_fn is not None:
            logger.warning(f"Using block_type={attn_blk_type}, not support hyper connection, ignore hc_kwargs")

        if other_blk_kwargs.get("v_residual", False):
            logger.info("Using attention value residual")

        for i in range(depth):
            block = block_cls(
                dim=embed_dim,
                n_heads=num_heads,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                attn_drop=attention_dropout,
                proj_drop=dropout,
                attn_type=attn_type,
                drop_path=drop_path_ps[i],
                is_causal=is_causal,
                mlp_ratio=mlp_ratio,
                ffn_drop=dropout,
                layer_idx=i,
                num_prefix_tokens=num_prefix_tokens,
                **other_blk_kwargs,
            )
            layers.append(block)
        self.layers = nn.ModuleList(layers)

    def _build_positional_embedding(
        self,
        pe_type: str,
        additional_pe: bool,
        rope_kwargs: dict,
        patch_size: int,
        img_size: int,
        embed_dim: int,
        num_heads: int,
        depth: int,
        with_cls_token: bool,
    ):
        # Positional embeddings
        self.pe_type: str = pe_type
        self.rope: RotaryEmbeddingCat | RotaryEmbeddingDinoV3 | None = None
        self.pe: nn.Parameter | None = None
        self._rope_is_mixed = False
        self.additional_pe = additional_pe

        if pe_type == "learn" or additional_pe:
            # [cls_token + n_reg_tokens + grid_size*grid_size, embed_dim]
            pe_2d = get_2d_sincos_pos_embed(
                embed_dim,
                grid_size=self.grid_size,
                pe_interpolation=1.0,
                cls_token=self.num_prefix_tokens != 0,
                extra_tokens=self.num_prefix_tokens,
            )
            # `get_2d_sincos_pos_embed()` 来自 numpy，内部会用到 float64；
            # 这里若不显式指定 dtype，会把可学习参数 `self.pe` 变成 float64，导致模型参数混合精度并触发 foreach/dynamo 的 dtype 检查错误。
            pe_2d = torch.as_tensor(pe_2d, dtype=torch.float32)
            self.pe = nn.Parameter(pe_2d, requires_grad=True)

        if pe_type.startswith("rope"):
            # rope, rope_cat, rope_mixed, rope_dinov3
            if pe_type == "rope":
                _rope_type_create = "cat"
            else:
                _rope_type_create = pe_type.split("_")[1]
            _rope_kwargs: dict[str, Any] = dict(
                dim=embed_dim,
                num_heads=num_heads,
                device="cuda",
                dtype=torch.float32,
                # from timm repo usage
                in_pixels=rope_kwargs.get("in_pixels", False) if _rope_type_create != "dinov3" else None,
                temperature=rope_kwargs.get("rope_theta", 10000.0),
            )
            if _rope_type_create == "dinov3":
                _rope_kwargs.pop("in_pixels", None)

            if _rope_type_create == "mixed":
                _rope_kwargs["depth"] = depth
                self._rope_is_mixed = True
            logger.log("NOTE", f"rope kwargs: {_rope_kwargs}")
            with torch.autocast("cuda", enabled=False):
                self.rope = create_rope_embed(_rope_type_create, **_rope_kwargs)

    def _build_cls_reg_tokens(self, n_reg_tokens: int, with_cls_token: bool, embed_dim: int):
        # Register tokens
        self.reg_tokens: nn.Parameter | None = None
        if n_reg_tokens > 0:
            self.reg_tokens = nn.Parameter(torch.zeros(1, n_reg_tokens, embed_dim))
            nn.init.trunc_normal_(self.reg_tokens, std=0.02)

        # Class token
        self.with_cls_token: bool = with_cls_token
        if self.with_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _build_head(
        self,
        head: str | None,
        out_chan: int | None,
        out_patch_size: int,
        norm_layer: str,
        embed_dim: int,
        unpatch_prog_dims: list[int] | None = None,
    ):
        # Head
        self.head = nn.Identity()
        self.head_required: bool = False
        self.head_type: str | None = head
        self.out_chan: int = 0 if out_chan is None else out_chan
        self.out_patch_size: int = out_patch_size

        if head is not None:
            assert out_chan is not None, "out_chan must be specified if head is used"
            assert out_patch_size is not None, "out_patch_size must be specified if head is used"
            self.out_chan = out_chan
            self.head_required = True
            if head == "linear":
                self.head = nn.Linear(embed_dim, out_chan * (out_patch_size**2))
            elif head == "norm_linear":
                self.head = nn.Sequential(
                    create_norm_layer(norm_layer, embed_dim),
                    nn.Linear(embed_dim, out_chan * (out_patch_size**2)),
                )
            elif head == "adaptive_linear":
                self.head = AdaptiveOutputLinearLayer(embed_dim, out_chan * (out_patch_size**2), mode="interp")
            elif head == "adaptive_unpatcher":
                assert unpatch_prog_dims is not None, "unpatch_prog_dims must be provided for adaptive unpatcher head"
                assert out_patch_size > 1, f"out_patch_size must be >1 for unpatcher head"
                self.head = AdaptiveProgressivePatchUnembedding(
                    embed_dim,
                    out_chan,
                    unpatch_prog_dims,
                    out_patch_size,
                    adaptive_mode="interp",
                )
            else:
                raise ValueError(f"Unknown head type: {head}")
        else:
            # head is None, but the out_chan is not None
            if out_chan is not None:
                logger.warning(
                    f"[Transformer Tokenizer] out_chan is specified but head is not. "
                    "This is not recommended. Please set 'head'."
                )

    def _build_mask_tokens(
        self,
        mask_train_ratio: float,
        mask_refill: bool,
        mask_drop_type: str,
        embed_dim: int,
    ):
        # Mask token
        self.mask_train_ratio = mask_train_ratio
        self.mask_training = mask_train_ratio > 0.0 or mask_refill
        self.mask_drop_type = mask_drop_type
        if self.mask_training:
            # Different from the original MAE that differ the encoder and decoder,
            # encoder takes ONLY UNMASKED tokens, and the decoder add them back
            # Here, we only mask the input tokens with a learnable mask token but
            # do not drop the masked tokens.

            # TODO: RoPE, theoretically support, but I haven't implemented it yet.
            assert not self.pe_type.startswith("rope"), "Rope PE not supported for mask training"
            self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.mask_token, std=0.02)
            logger.info(f"[Transformer Tokenizer] Mask training enabled with ratio {mask_train_ratio}")

    def _build_last_norm(self, last_norm: str | None, embed_dim: int):
        # Last norm
        self.last_norm = None
        if last_norm is not None:
            self.last_norm = create_norm_layer(last_norm, embed_dim)

    def init_weights(self):
        # Initialize weights
        named_apply(partial(block_basic_init, init_type="trunc_normal"), self)
        # named_apply(get_init_weights_vit("jax"), self)
        logger.debug("[Transformer Tokenizer]: Model initialized with MoCo Vit weights")

    def _get_masked_x(self, x):
        if self.mask_train_ratio > 0:
            # is MAE encoder
            if self.mask_drop_type == "mae":
                x, mask, ids_restore = random_masking_mae(x, self.mask_train_ratio)
            else:
                x, mask, ids_restore = random_masking_no_drop(x, self.mask_train_ratio, self.mask_token)
        else:
            x, mask, ids_restore = x, None, None

        return x, mask, ids_restore

    def _refill_masked_x(self, x, mask, ids_restore):
        """the original x are masked and dropped"""
        mask_learned = getattr(self, "mask_token", None)
        assert mask_learned is not None, "mask_token must be init when using this class as MAE decoder"
        ids_restore = ids_restore[..., None].repeat(1, 1, x.shape[2])
        mask_learned_expand = mask_learned.repeat(x.shape[0], ids_restore.shape[1] + self.n_reg_tokens - x.shape[1], 1)
        x_ = torch.cat([x[:, self.n_reg_tokens :], mask_learned_expand], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore)  # unshuffle
        x = torch.cat([x[:, : self.n_reg_tokens], x_], dim=1)
        return x

    def _jepa_apply_masks(self, x, masks: list[Tensor]):
        all_x = []
        for m in masks:
            mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
            all_x += [torch.gather(x, dim=1, index=mask_keep)]
        return torch.cat(all_x, dim=0)

    def _apply_ijepa_mask(self, x, masks, rope):
        # Apply masks
        prefixed_tokens, x = x[:, : self.num_prefix_tokens], x[:, self.num_prefix_tokens :]

        # Apply token masks
        x = self._jepa_apply_masks(x, masks=masks)

        # Re-attach prefix tokens
        # Note: In IJEPA, typically we process multiple masks, so batch size increases.
        # We need to repeat prefix tokens to match the new batch size.
        # IMPORTANT: `_jepa_apply_masks` concatenates masks on batch dim as:
        #   [mask0(B), mask1(B), ..., maskK(B)]
        # so prefix tokens must be repeated in the same order (repeat whole batch per mask),
        # not interleaved per sample.
        n_masks = len(masks)
        if n_masks > 1:
            prefixed_tokens = prefixed_tokens.repeat(n_masks, 1, 1)

        x = torch.cat([prefixed_tokens, x], dim=1)

        # Apply RoPE masks
        if rope is not None:
            rope_masked = []
            for m in masks:
                # m: [B, S_masked] indices
                assert not self._rope_is_mixed, "mixed rope is not supported in JEPA training"
                rope_masked += [get_at("[S] ropeD, B S_masked -> B 1 S_masked ropeD", rope, m)]
            rope = torch.cat(rope_masked, dim=0)  # [B*n_masks, 1, S_masked, ropeD]

        return x, rope

    def _get_naflex_rope_embed(
        self,
        B: int,
        grid_sizes: list[tuple[int, int]],
        l_cur: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        """Variable grid_sizes in a batch of images: naflex mode"""
        assert self.rope is not None
        assert len(grid_sizes) == B, f"{len(grid_sizes)=} != {B=}"
        # hp, wp = grid_size
        # grid_size -> list of batch indices
        size_to_indices: dict[tuple, list[int]] = {}
        unique_sizes: list[tuple] = []  # unique grid size tuples
        # Do not support the naflex mode
        for bi, grid_size in enumerate(grid_sizes):  # [(hp, wp)] * x.shape[0]
            if grid_size not in size_to_indices:
                size_to_indices[grid_size] = []
                unique_sizes.append(grid_size)
            size_to_indices[grid_size].append(bi)

        if self._rope_is_mixed:
            rope_embeds = NaFlexRopeIterator(
                self.rope,
                size_to_indices,
                unique_sizes,
                batch_size=B,
                seq_len=l_cur,
                dtype=dtype,
                device=device,
            )

        # create the rope embeddings with different grid sizes
        rope_embeds = torch.zeros(B, l_cur, self.rope.dim * 2, dtype=dtype, device=device)
        get_batch_embeds = getattr(self.rope, "get_batch_embeds", None)
        if callable(get_batch_embeds):
            # Rope cat
            unique_embeds = get_batch_embeds(unique_sizes)
            for grid_size, embed, batch_indices in zip(unique_sizes, unique_embeds, size_to_indices.values()):
                h, w = grid_size
                actual_len = h * w
                for bi in batch_indices:
                    rope_embeds[bi, :actual_len] = embed[:actual_len]
        else:
            # Rope dinov3
            # Generate each unique size separately and assign
            for grid_size, batch_indices in size_to_indices.items():
                rope_embed = self.rope.get_embed(shape=[grid_size[0], grid_size[1]])
                h, w = grid_size
                actual_len = h * w
                for bi in batch_indices:
                    rope_embeds[bi, :actual_len] = rope_embed[:actual_len]

        return rope_embeds

    def _get_standard_rope_embed(self, grid_size: tuple[int, int], l_cur: int):
        """Per-batch generate the rope embeddings"""
        assert self.rope is not None
        with torch.autocast(device_type="cuda", enabled=False):
            rope_embeds = self.rope.get_embed(shape=[grid_size[0], grid_size[1]])
        # logger.log("NOTE", f"create the rope embeds shaped as {rope_embeds.shape}")
        return rope_embeds  # [seq_len, dim_h]

    def _with_pos_embed(self, x, hw: tuple | None = None):
        if hw is not None:
            hp, wp = hw[0], hw[1]
        else:
            hp = wp = math.sqrt(x.shape[1])
            assert hp.is_integer() and wp.is_integer(), (
                f"Cannot resample 2D PE with non-square number of tokens: {x.shape[1]}"
            )
        hp, wp = int(hp), int(wp)
        l_cur = hp * wp
        rope_ret = None

        if getattr(self, "pe", None) is not None:
            assert self.pe is not None, f"Positional encoding is None"
            pe_1lc = self.pe[None]
            if l_cur != self.n_patches:
                pe_1lc = resample_abs_pos_embed(  # type: ignore
                    pe_1lc,  # (1, l, dim)
                    num_prefix_tokens=self.num_prefix_tokens,
                    new_size=(hp, wp),
                    old_size=self.grid_size,
                )
            x = x + pe_1lc

        if self.pe_type.startswith("rope") and self.rope is not None:
            # NOTE: not supports the naflex mode yet.
            rope_ret = self._get_standard_rope_embed((hp, wp), l_cur)

        return x, rope_ret

    def _forward_proj_out(self, x):
        # Encoder proj out: image -> backbone -> output proj
        x = self.projections["output_proj"](x)
        return x

    def _forward_proj_in(self, x):
        # Decoder proj in: latent -> input proj -> backbone -> reconstruction
        x = self.projections["input_proj"](x)
        return x

    def _forward_get_tokens(self, x, hw=None):
        if x.ndim == 4:
            bs, c, h, w = x.shape
            if hw is not None:
                h, w = hw
            x = self.patch_embed(x)  # (bs, n, dim)
        elif x.ndim == 3:
            bs, l, c = x.shape
            if hw is None:
                h = w = int(math.sqrt(l))
            else:
                h, w = hw
            x = self.patch_embed(x.contiguous())
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")

        # Register tokens get no PE
        if self.n_reg_tokens > 0 and self.reg_tokens is not None:
            reg_tokens = self.reg_tokens.repeat(bs, 1, 1)  # (bs, n_reg, dim)
            x = torch.cat([reg_tokens, x], dim=1)  # (bs, n+n_reg, dim)

        if self.with_cls_token:
            assert getattr(self, "cls_token", None) is not None
            cls_tokens = self.cls_token.repeat(bs, 1, 1)
            x = torch.cat([cls_tokens, x], dim=1)

        x, rope = self._with_pos_embed(x, hw=(h // self.patch_size, w // self.patch_size))

        return x, rope

    def forward_features(
        self,
        x: Float[Tensor, "b c h w or b l c"],
        hw: tuple[int, int] | None = None,
        get_intermidates: list[int] | None = None,
        masks: list[Tensor] | Tensor | None = None,
    ):
        dtype = x.dtype

        # Get grid size
        if hw is None:
            # x grid size
            if x.ndim == 4:
                gx_h, gx_w = (
                    x.shape[-2] // self.patch_size,
                    x.shape[-1] // self.patch_size,
                )
            else:
                gx_h, gx_w = (
                    int(math.sqrt(x.shape[1])) // self.patch_size,
                    int(math.sqrt(x.shape[1])) // self.patch_size,
                )
        else:
            gx_h, gx_w = hw[0] // self.patch_size, hw[1] // self.patch_size

        # Embed patches
        x, rope = self._forward_get_tokens(x, hw)  # (bs, n+n_reg, dim)

        # HC streaming embeddings
        if self._use_hc:
            x = self.hc_expand_stream(x)

        if masks is not None:
            if not isinstance(masks, list):
                masks = [masks]
            x, rope = self._apply_ijepa_mask(x, masks, rope)

        # Projection in
        x = self._forward_proj_in(x)

        v_residual_v1: Tensor | None = None
        if self.v_residual:
            x = x.type(dtype)
            v_residual_v1 = self._compute_value_residual_v1(x)

        # Layers
        intermidates = []
        index = get_intermidates or []
        for i, blk in enumerate(self.layers):
            # Rope
            rope_ = None
            if self._rope_is_mixed:
                # need to regenerate rope for each layer
                if isinstance(rope, NaFlexRopeIterator):
                    rope_ = next(rope)
                elif torch.is_tensor(rope):
                    # is standard rope [bs, depth, nheads, seq_len, dim]
                    rope_ = rope[:, i]
            else:
                rope_ = rope

            # Blocks
            x = x.type(dtype)
            if self.grad_checkpointing and self.training:
                if self.v_residual:
                    x = checkpoint(blk, x, rope_, v_residual_v1, use_reentrant=False)
                else:
                    x = checkpoint(blk, x, rope_, use_reentrant=False)
            else:
                if self.v_residual:
                    x = blk(x, rope=rope_, v_residual_v1=v_residual_v1)
                else:
                    x = blk(x, rope=rope_)

            if i in index:
                # Only return spatial tokens, drop all prefix tokens (cls + reg)
                intermidates.append(x[:, self.num_prefix_tokens :])

        # HC reduce stream
        if self._use_hc:
            x = self.reduce_stream(x)

        # Projection out
        x = self._forward_proj_out(x)

        # Norm
        x = x if self.last_norm is None else self.last_norm(x)
        cls_reg_tokens = x[:, : self.num_prefix_tokens]
        x_norm = x[:, self.num_prefix_tokens :, :]  # remove all prefix tokens (cls + reg)

        out = {
            "x_norm": x,
            "x_norm_patch_tokens": x_norm,  # spatial patch tokens
            "cls_tokens": cls_reg_tokens[:, :1] if self.with_cls_token else None,
            "reg_tokens": cls_reg_tokens[:, 1:] if self.n_reg_tokens > 0 else None,
            "grid_size": (gx_h, gx_w),
            "intermidates": intermidates if len(intermidates) > 0 else None,
        }
        return out

    def _to_output(
        self,
        x,
        grid_size: tuple[int, int] | torch.Tensor | torch.Size,
        ret_2d_tokens=False,
        out_shape: torch.Size | tuple | int | None = None,
    ):
        # Reshape into 2d img
        reshape_flag = self.head_required and ret_2d_tokens
        if isinstance(out_shape, (torch.Size, tuple)):
            out_shape = out_shape[1]

        def to_out_2d(x):
            if ret_2d_tokens:
                x = rearrange(
                    x,
                    "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
                    h=grid_size[0],
                    w=grid_size[1],
                    p1=self.out_patch_size,
                    p2=self.out_patch_size,
                )
            return x

        if not self.head_required and ret_2d_tokens:
            x = rearrange(x, "... (h w) c -> ... c h w", h=grid_size[0], w=grid_size[1])
        else:
            if self.head_type in ("linear", "norm_linear"):
                x = self.head(x)  # (bs, n, out_chan)
                x = to_out_2d(x)
            elif self.head_type == "adaptive_linear":
                assert out_shape is not None
                x = self.head(x, out_shape * self.out_patch_size**2)
                x = to_out_2d(x)
            elif self.head_type == "adaptive_unpatcher":
                # Adaptive unpatcher
                x = self.head(x, out_shape)
                if not ret_2d_tokens:
                    logger.warning("Adaptive unpatcher alwarys return 2d tokens")
        return x

    def forward(
        self,
        x: torch.Tensor,
        *,
        ret_2d_tokens=False,
        ret_all=True,
        get_intermidates: list[int] | None = None,
        out_shape: torch.Size | tuple | None = None,
        masks: list[Tensor] | Tensor | None = None,
    ) -> Tensor | dict[str, Any]:
        out = self.forward_features(x, get_intermidates=get_intermidates, masks=masks)  # (bs, n, dim)

        x = out["x_norm_patch_tokens"]
        x = self._to_output(
            x,
            grid_size=out["grid_size"],
            ret_2d_tokens=ret_2d_tokens,
            out_shape=out_shape,
        )

        if not ret_all and get_intermidates is None:
            return x
        else:
            out["head_out"] = x
            return out

    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    def get_last_layer(self):
        if self.head_type == "linear":
            assert isinstance(self.head, nn.Linear)
            return self.head.weight
        elif self.head_type == "norm_linear":
            if not isinstance(self.head, nn.Sequential):
                raise TypeError("head must be nn.Sequential when head_type='norm_linear'")
            return self.head[0].weight
        elif self.head_type == "adaptive_linear":
            assert isinstance(self.head, AdaptiveOutputLinearLayer)
            return self.head.linear.weight
        elif self.head_type == "adaptive_unpatcher":
            assert isinstance(self.head, AdaptiveProgressivePatchUnembedding)
            return self.head.unpatchers[-1].weight
        else:
            raise ValueError(f"Unsupported head type: {self.head_type} to get last layer's weight")


if __name__ == "__main__":
    """
    LOVELY_TENSORS=1 python -m src.stage1.cosmos.modules.transformer
    """
    transformer = (
        TransformerTokenizer(
            3,
            512,
            3,
            patch_size=16,
            pe_type="rope_dinov3",
            drop_path=0.2,
            is_causal=False,
            mask_train_ratio=0.0,
            additional_pe=True,
            with_cls_token=True,
            n_reg_tokens=4,
            other_blk_kwargs={"use_gate": True},
            # hc_streams=2,
            # hc_implem="naive",
            v_residual=True,
        )
        .cuda()
        .to(dtype=torch.bfloat16)
    )
    x = torch.randn(2, 3, 224, 224).cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = transformer(x, ret_2d_tokens=True, ret_all=True)
    for k, v in out.items():
        print("{:<20}: {}".format(k, str(v)))
    # pass
