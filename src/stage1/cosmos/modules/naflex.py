from timm.layers.create_norm import create_norm_layer
import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast, TypedDict, Literal

import einops
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from easydict import EasyDict as edict
from einx import get_at
from loguru import logger
from timm.layers import apply_keep_indices_nlc, to_2tuple
from timm.layers.create_norm import get_norm_layer
from timm.layers.helpers import to_2tuple
from timm.layers.patch_embed import PatchEmbed
from timm.layers.pos_embed import resample_abs_pos_embed
from timm.layers import get_act_layer, Mlp, LayerScale
from timm.models import eva, naflexvit
from timm.models._manipulate import named_apply
from timm.models.eva import AttentionRope, DropPath, EvaAttention, GluMlp, Mlp, SwiGLU, apply_rot_embed_cat
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from timm.models.naflexvit import (
    Block,
    NaFlexEmbeds,
    NaFlexVit,
    # checkpoint,
    create_attention_mask,
    feature_take_indices,
    get_init_weights_vit,
)
from timm.layers.drop import calculate_drop_path_rates
from diffusers.models.embeddings import get_2d_sincos_pos_embed

from src.utilities.config_utils import dataclass_from_dict, function_config_to_basic_types
from src.utilities.network_utils import compile_decorator

from .norm import *  # register custom norms
from .transformer import GatedAttention
from .SLA import LinearCrossAttention, SparseLinearAttention

logger = logger.bind(_name_="Naflex")


class _DeviceDTypeKwargs(TypedDict, total=False):
    device: torch.device | None
    dtype: torch.dtype | None


def nepa_prediction_loss(h_in: Tensor, h_out: Tensor, shift: bool = True) -> Tensor:
    """
    Next Embedding Prediction loss (negative cosine similarity).

    This loss encourages the model to predict the next position's input embedding
    from the current position's output hidden state, similar to autoregressive
    language modeling but in latent space.

    Args:
        h_in:  [B, T, D] input embeddings (target, will be detached)
        h_out: [B, T, D] output hidden states (prediction)
        shift: if True, predict h_in[i+1] from h_out[i] (next token prediction)
               if False, predict h_in[i] from h_out[i] (position-wise matching)

    Returns:
        loss: scalar, negative cosine similarity in range [-1, 1]
    """
    # Detach target to prevent gradient flow
    h_in = h_in.detach()

    if shift:
        # Next token prediction: h_out[i] predicts h_in[i+1]
        p = h_out[:, :-1, :]  # positions 0 to T-2
        z = h_in[:, 1:, :]  # positions 1 to T-1
    else:
        # Position-wise matching
        p = h_out
        z = h_in

    # L2 normalize along feature dimension
    p = torch.nn.functional.normalize(p, dim=-1)
    z = torch.nn.functional.normalize(z, dim=-1)

    # Negative cosine similarity (minimize to maximize similarity)
    loss = -(p * z).sum(dim=-1).mean()
    return loss


class GatedAttentionTimmWrapped(GatedAttention):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qkv_fused: bool = True,
        qkv_bias_separate: bool = False,
        num_prefix_tokens: int = 1,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_head_dim: Optional[int] = None,
        norm_layer: Optional[Callable] = None,
        qk_norm: bool = False,
        scale_norm: bool = True,
        rotate_half: bool = False,
        is_causal: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__(
            dim,
            num_heads,
            num_heads,
            norm_layer,
            qk_norm,
            qkv_bias,
            num_prefix_tokens=num_prefix_tokens,
            attention_dropout=attn_drop,
            headwise_attn_output_gate=True,
            elementwise_attn_output_gate=False,
            is_causal=is_causal,
        )


def _apply_linear_attn_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if attn_mask is None:
        return q, k, v

    mask = attn_mask == 0
    valid_k = mask.any(dim=-2)  # [B, 1, Lk]
    valid_q = mask.any(dim=-1)  # [B, 1, Lq]

    q = q * valid_q.unsqueeze(1).to(dtype=q.dtype).unsqueeze(-1)
    k = k * valid_k.unsqueeze(1).to(dtype=k.dtype).unsqueeze(-1)
    v = v * valid_k.unsqueeze(1).to(dtype=v.dtype).unsqueeze(-1)
    return q, k, v


class EvaLinearAttention(nn.Module):
    """Eva-style attention with SLA linear attention backend."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qkv_fused: bool = True,
        qkv_bias_separate: bool = False,
        num_prefix_tokens: int = 1,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_head_dim: Optional[int] = None,
        norm_layer: Optional[Callable] = None,
        qk_norm: bool = False,
        scale_norm: bool = True,
        rotate_half: bool = False,
        feature_map: str = "softmax",
        use_bf16: bool = True,
        tie_feature_map_qk: bool = True,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        dd: _DeviceDTypeKwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.LayerNorm
        if scale_norm or qk_norm:
            assert norm_layer is not None, "norm_layer must be provided if qk_norm or scale_norm is True"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        self.head_dim = head_dim
        attn_dim = head_dim * self.num_heads
        self.num_prefix_tokens = num_prefix_tokens
        self.qkv_bias_separate = qkv_bias_separate
        self.rotate_half = rotate_half

        if qkv_fused:
            self.qkv = nn.Linear(dim, attn_dim * 3, bias=False, **dd)
            self.q_proj = self.k_proj = self.v_proj = None
            if qkv_bias:
                self.q_bias = nn.Parameter(torch.zeros(attn_dim, **dd))
                self.register_buffer("k_bias", torch.zeros(attn_dim, **dd), persistent=False)
                self.v_bias = nn.Parameter(torch.zeros(attn_dim, **dd))
            else:
                self.q_bias = self.k_bias = self.v_bias = None
        else:
            self.q_proj = nn.Linear(dim, attn_dim, bias=qkv_bias, **dd)
            self.k_proj = nn.Linear(dim, attn_dim, bias=False, **dd)
            self.v_proj = nn.Linear(dim, attn_dim, bias=qkv_bias, **dd)
            self.qkv = None
            self.q_bias = self.k_bias = self.v_bias = None

        self.q_norm = norm_layer(self.head_dim, **dd) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, **dd) if qk_norm else nn.Identity()
        self.attn = LinearCrossAttention(
            feature_map=feature_map,
            use_bf16=use_bf16,
            tie_feature_map_qk=tie_feature_map_qk,
            eps=eps,
        )
        self.norm = norm_layer(attn_dim, **dd) if scale_norm else nn.Identity()
        self.proj = nn.Linear(attn_dim, dim, **dd)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        rope: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, C = x.shape

        if self.qkv is not None:
            if self.q_bias is None:
                qkv = self.qkv(x)
            else:
                assert self.q_bias is not None
                assert self.k_bias is not None
                assert self.v_bias is not None
                qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
                if self.qkv_bias_separate:
                    qkv = self.qkv(x)
                    qkv += qkv_bias
                else:
                    qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
        else:
            assert self.q_proj is not None
            assert self.k_proj is not None
            assert self.v_proj is not None
            q = self.q_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
            k = self.k_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
            v = self.v_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)

        q, k = self.q_norm(q), self.k_norm(k)

        if rope is not None:
            npt = self.num_prefix_tokens
            half = getattr(self, "rotate_half", False)
            q = torch.cat([q[:, :, :npt, :], apply_rot_embed_cat(q[:, :, npt:, :], rope, half=half)], dim=2).type_as(v)
            k = torch.cat([k[:, :, :npt, :], apply_rot_embed_cat(k[:, :, npt:, :], rope, half=half)], dim=2).type_as(v)

        q, k, v = _apply_linear_attn_mask(q, k, v, attn_mask)
        x = self.attn(q, k, v)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EvaSparseLinearAttention(nn.Module):
    """Eva-style attention with SLA sparse linear attention backend."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qkv_fused: bool = True,
        qkv_bias_separate: bool = False,
        num_prefix_tokens: int = 1,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_head_dim: Optional[int] = None,
        norm_layer: Optional[Callable] = None,
        qk_norm: bool = False,
        scale_norm: bool = True,
        rotate_half: bool = False,
        topk: float = 0.25,
        feature_map: str = "softmax",
        BLKQ: int = 64,
        BLKK: int = 64,
        use_bf16: bool = True,
        tie_feature_map_qk: bool = True,
        device=None,
        dtype=None,
    ):
        dd: _DeviceDTypeKwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.LayerNorm
        if scale_norm or qk_norm:
            assert norm_layer is not None, "norm_layer must be provided if qk_norm or scale_norm is True"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        self.head_dim = head_dim
        attn_dim = head_dim * self.num_heads
        self.num_prefix_tokens = num_prefix_tokens
        self.qkv_bias_separate = qkv_bias_separate
        self.rotate_half = rotate_half

        if qkv_fused:
            self.qkv = nn.Linear(dim, attn_dim * 3, bias=False, **dd)
            self.q_proj = self.k_proj = self.v_proj = None
            if qkv_bias:
                self.q_bias = nn.Parameter(torch.zeros(attn_dim, **dd))
                self.register_buffer("k_bias", torch.zeros(attn_dim, **dd), persistent=False)
                self.v_bias = nn.Parameter(torch.zeros(attn_dim, **dd))
            else:
                self.q_bias = self.k_bias = self.v_bias = None
        else:
            self.q_proj = nn.Linear(dim, attn_dim, bias=qkv_bias, **dd)
            self.k_proj = nn.Linear(dim, attn_dim, bias=False, **dd)
            self.v_proj = nn.Linear(dim, attn_dim, bias=qkv_bias, **dd)
            self.qkv = None
            self.q_bias = self.k_bias = self.v_bias = None

        self.q_norm = norm_layer(self.head_dim, **dd) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, **dd) if qk_norm else nn.Identity()
        self.attn = SparseLinearAttention(
            head_dim,
            topk=topk,
            feature_map=feature_map,
            BLKQ=BLKQ,
            BLKK=BLKK,
            use_bf16=use_bf16,
            tie_feature_map_qk=tie_feature_map_qk,
        )
        self.norm = norm_layer(attn_dim, **dd) if scale_norm else nn.Identity()
        self.proj = nn.Linear(attn_dim, dim, **dd)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        rope: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, C = x.shape

        if self.qkv is not None:
            if self.q_bias is None:
                qkv = self.qkv(x)
            else:
                assert self.q_bias is not None
                assert self.k_bias is not None
                assert self.v_bias is not None
                qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
                if self.qkv_bias_separate:
                    qkv = self.qkv(x)
                    qkv += qkv_bias
                else:
                    qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
        else:
            assert self.q_proj is not None
            assert self.k_proj is not None
            assert self.v_proj is not None
            q = self.q_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
            k = self.k_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
            v = self.v_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)

        q, k = self.q_norm(q), self.k_norm(k)

        if rope is not None:
            npt = self.num_prefix_tokens
            half = getattr(self, "rotate_half", False)
            q = torch.cat([q[:, :, :npt, :], apply_rot_embed_cat(q[:, :, npt:, :], rope, half=half)], dim=2).type_as(v)
            k = torch.cat([k[:, :, :npt, :], apply_rot_embed_cat(k[:, :, npt:, :], rope, half=half)], dim=2).type_as(v)

        q, k, v = _apply_linear_attn_mask(q, k, v, attn_mask)
        x = self.attn(q, k, v)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class _HC_EvaAttention(nn.Module):
    """Helper class to wrap EvaBlock attention branch for mHC."""

    def __init__(
        self,
        norm: nn.Module,
        attention: nn.Module,
        gamma: nn.Parameter | None,
        drop_path: nn.Module,
    ) -> None:
        super().__init__()
        self.norm = norm
        self.attention = attention
        self.gamma = gamma
        self.drop_path = drop_path

    def forward(
        self,
        x: Tensor,
        rope: Tensor | None = None,
        attn_mask: Tensor | None = None,
        v_residual_v1: Tensor | None = None,
    ) -> Tensor:
        x_normed = self.norm(x)
        attn_out = self.attention(x_normed, rope=rope, attn_mask=attn_mask, v_residual_v1=v_residual_v1)
        if self.gamma is not None:
            attn_out = self.gamma * attn_out
        return self.drop_path(attn_out)


class _HC_EvaMLP(nn.Module):
    """Helper class to wrap EvaBlock MLP branch for mHC."""

    def __init__(
        self,
        norm: nn.Module,
        mlp: nn.Module,
        gamma: nn.Parameter | None,
        drop_path: nn.Module,
    ) -> None:
        super().__init__()
        self.norm = norm
        self.mlp = mlp
        self.gamma = gamma
        self.drop_path = drop_path

    def forward(self, x: Tensor) -> Tensor:
        x_normed = self.norm(x)
        mlp_out = self.mlp(x_normed)
        if self.gamma is not None:
            mlp_out = self.gamma * mlp_out
        return self.drop_path(mlp_out)


class EvaBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        qkv_fused: bool = True,
        mlp_ratio: float = 4.0,
        ffn_type: str | None = None,
        swiglu_mlp: bool = False,  # legacy args
        swiglu_align_to: int = 0,
        scale_mlp: bool = False,
        scale_attn_inner: bool = False,
        num_prefix_tokens: int = 1,
        attn_type: str = "eva",
        rotate_half: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        init_values: Optional[float] = None,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        attn_head_dim: Optional[int] = None,
        block_norm_type: Literal["pre", "post"] = "pre",
        is_causal: bool = False,
        v_residual: bool = False,
        layer_idx: int | None = None,
        hyperconnection_kwargs: dict[str, Any] | None = None,
        post_norm_alpha: float | None = None,
        post_norm_skip_first_layer: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs,
    ):
        """Initialize the EVA transformer block.

        Args:
          dim: Input dimension of the token embeddings
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias terms in query, key, value projections
            qkv_fused: Whether to use a single projection for query, key, value
            mlp_ratio: Ratio of MLP hidden dimension to input dimension
            swiglu_mlp: Whether to use SwiGLU activation in the MLP
            scale_mlp: Whether to use normalization in the MLP
            scale_attn_inner: Whether to use normalization within the attention mechanism
            num_prefix_tokens: Number of tokens at the beginning of the sequence (class tokens, etc.)
            attn_type: Type of attention module to use ('eva' or 'rope')
            proj_drop: Dropout rate for projection layers
            attn_drop: Dropout rate for attention matrix
            drop_path: Stochastic depth rate
            init_values: Initial value for LayerScale, None = no LayerScale
            act_layer: Activation layer constructor
            norm_layer: Normalization layer constructor
            attn_head_dim: Dimension of each attention head (if None, computed as dim // num_heads)
            v_residual: Whether to use value residual connections
            layer_idx: Layer index for mHC initialization
            hyperconnection_kwargs: Hyperconnection (mHC) configuration
        """
        dd: _DeviceDTypeKwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.layer_idx = layer_idx
        self.v_residual = v_residual
        self.block_norm_type = block_norm_type
        assert block_norm_type in ["pre", "post"], f"block_norm_type must be 'pre' or 'post', got {block_norm_type}"
        if self.block_norm_type == "post":
            # using Bytesdance SEED `keel` post-norm: arxiv, 2601.19895
            assert hyperconnection_kwargs is None, "hyperconnection_kwargs must be None when block_norm_type is 'post'"
            assert post_norm_alpha is not None, "post_norm_alpha must be provided for post-norm blocks"

        self.norm1 = norm_layer(dim, **dd)
        logger.trace(f"Layer {self.__class__.__name__} uses attention type {attn_type}")
        attn_cls = {
            "rope": AttentionRope,
            "eva": EvaAttention,
            "gated": GatedAttentionTimmWrapped,
            "linear": EvaLinearAttention,
            "sparse_linear": EvaSparseLinearAttention,
        }[attn_type]

        attn_kwargs = {}
        if attn_type == "gated":
            attn_kwargs["is_causal"] = is_causal
        elif attn_type == "linear":
            linear_kwargs = kwargs.get("linear_attn")
            if linear_kwargs is not None:
                attn_kwargs.update(linear_kwargs)
        elif attn_type == "sparse_linear":
            sparse_linear_kwargs = kwargs.get("sparse_linear_attn")
            if sparse_linear_kwargs is not None:
                attn_kwargs.update(sparse_linear_kwargs)

        self.attn = attn_cls(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qkv_fused=qkv_fused,
            num_prefix_tokens=num_prefix_tokens,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            attn_head_dim=attn_head_dim,
            norm_layer=norm_layer,
            scale_norm=scale_attn_inner,
            rotate_half=rotate_half,
            **attn_kwargs,
            **dd,
        )
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim, **dd)) if init_values is not None else None
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim, **dd)
        hidden_features = int(dim * mlp_ratio)
        if swiglu_mlp or ffn_type == "swiglu":
            if scale_mlp or swiglu_align_to:
                # when norm in SwiGLU used or alignment enabled, an impl with separate fc for gate & x is used
                self.mlp = SwiGLU(
                    in_features=dim,
                    hidden_features=hidden_features,
                    norm_layer=norm_layer if scale_mlp else None,
                    drop=proj_drop,
                    align_to=swiglu_align_to,
                    **dd,
                )
            else:
                # w/o any extra norm, an impl with packed weights is used
                self.mlp = GluMlp(
                    in_features=dim,
                    hidden_features=hidden_features * 2,
                    norm_layer=norm_layer if scale_mlp else None,
                    act_layer=nn.SiLU,
                    gate_last=False,
                    drop=proj_drop,
                    **dd,
                )

        elif ffn_type == "moe":
            moe_kwargs = dict(kwargs.get("moe", {}) or {})
            from .moe import MoEFFN

            num_layers = moe_kwargs.pop("num_layers", 12)
            self.mlp = MoEFFN(
                dim=dim,
                n_heads=num_heads,
                mlp_ratio=mlp_ratio,
                num_layers=num_layers,
                layer_idx=layer_idx if layer_idx is not None else 0,
                ffn_drop=proj_drop,
                fused_type=None,
                assume_bsh=True,
                **moe_kwargs,
            )

        elif ffn_type == "mlp":
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=hidden_features,
                act_layer=act_layer,
                norm_layer=norm_layer if scale_mlp else None,
                drop=proj_drop,
                **dd,
            )
        self.gamma_2 = nn.Parameter(init_values * torch.ones(dim, **dd)) if init_values is not None else None
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        if self.block_norm_type == "post":
            post_norm_alpha_value = float(post_norm_alpha) if post_norm_alpha is not None else 1.0
            self.register_buffer(
                "post_norm_alpha",
                torch.tensor(post_norm_alpha_value, **dd),
                persistent=False,
            )
            self.post_norm1 = norm_layer(dim, **dd)
            self.post_norm2 = norm_layer(dim, **dd)
            self._post_norm_skip_first_layer = bool(post_norm_skip_first_layer) and self.layer_idx == 0
        else:
            self.post_norm1 = nn.Identity()
            self.post_norm2 = nn.Identity()
            self._post_norm_skip_first_layer = False

        # Determine forward type
        self._forward_type = "eva_block" if hyperconnection_kwargs is None else "eva_block_mhc"
        self._forward_eva_block = (
            self._forward_eva_block_prenorm if self.block_norm_type == "pre" else self._forward_eva_block_postnorm
        )

        # Hyper-connection mHC
        self.hyperconnection_kwargs = hyperconnection_kwargs
        if hyperconnection_kwargs is not None:
            init_hc_fn = hyperconnection_kwargs.get("init_hc")
            assert init_hc_fn is not None and layer_idx is not None, "init_hc and layer_idx required for mHC"
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
                branch=_HC_EvaAttention(self.norm1, self.attn, self.gamma_1, self.drop_path1),
                layer_index=layer_idx * 2,
                **init_hc_kwargs,
            )
            self.hc_mlp = init_hc_fn(
                dim=dim,
                branch=_HC_EvaMLP(self.norm2, self.mlp, self.gamma_2, self.drop_path2),
                layer_index=layer_idx * 2 + 1,
                **init_hc_kwargs,
            )

    def _forward_eva_block_prenorm(
        self,
        x: torch.Tensor,
        rope: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        v_residual_v1: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.gamma_1 is None:
            x = x + self.drop_path1(self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask))
            x = x + self.drop_path2(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask))
            x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

    def _forward_eva_block_postnorm(
        self,
        x: torch.Tensor,
        rope: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        v_residual_v1: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self._post_norm_skip_first_layer:
            return self._forward_eva_block_prenorm(x, rope=rope, attn_mask=attn_mask, v_residual_v1=v_residual_v1)

        alpha = getattr(self, "post_norm_alpha", None)
        assert alpha is not None, "post_norm_alpha is required for post-norm blocks"
        if self.gamma_1 is None:
            attn_out = self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask)
        else:
            attn_out = self.gamma_1 * self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask)
        x = self.post_norm1(alpha * x + self.drop_path1(attn_out))

        if self.gamma_2 is None:
            mlp_out = self.mlp(self.norm2(x))
        else:
            mlp_out = self.gamma_2 * self.mlp(self.norm2(x))
        x = self.post_norm2(alpha * x + self.drop_path2(mlp_out))
        return x

    def _forward_eva_block_mhc(
        self,
        x: torch.Tensor,
        rope: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        v_residual_v1: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.hc_attn(x, rope=rope, attn_mask=attn_mask, v_residual_v1=v_residual_v1)
        x = self.hc_mlp(x)
        return x

    @compile_decorator
    def forward(
        self,
        x: torch.Tensor,
        rope: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        v_residual_v1: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self._forward_type == "eva_block":
            return self._forward_eva_block(x, rope, attn_mask, v_residual_v1)
        elif self._forward_type == "eva_block_mhc":
            return self._forward_eva_block_mhc(x, rope, attn_mask, v_residual_v1)
        else:
            raise ValueError(f"Unknown forward type: {self._forward_type}")


eva.EvaBlock = EvaBlock  # type: ignore[invalid-assignment]


def get_block_fn(cfg) -> Callable:
    """Get appropriate block function based on configuration.

    Returns a partially applied block constructor with EVA-specific
    or conflicting parameters pre-configured if needed.
    """

    def _wrap_with_layer_idx(block_cls: type[nn.Module], **fixed_kwargs: Any) -> Callable:
        layer_counter = {"idx": 0}

        def _factory(*args: Any, **kwargs: Any) -> nn.Module:
            if kwargs.get("layer_idx") is None:
                kwargs["layer_idx"] = layer_counter["idx"]
            layer_counter["idx"] += 1
            return block_cls(*args, **fixed_kwargs, **kwargs)

        return _factory

    # Check if we need EVA block features
    use_eva_features = (
        cfg.attn_type in ("eva", "rope", "gated", "linear", "sparse_linear")
        or cfg.rope_type not in ("", "none")  # Any ROPE type requires EVA blocks
        or cfg.swiglu_mlp
    )

    if use_eva_features:
        # Determine attention type based on rope_type if not explicitly set
        attn_type = cfg.attn_type
        if attn_type == "standard" and cfg.rope_type not in ("", "none"):
            attn_type = "rope"

        num_prefix_tokens = (1 if cfg.class_token else 0) + cfg.reg_tokens
        block_norm_type = "pre" if cfg.pre_norm else "post"
        post_norm_alpha = (
            cfg.post_norm_alpha
            if cfg.post_norm_alpha is not None
            else (cfg.depth * 2 if block_norm_type == "post" else None)
        )
        return _wrap_with_layer_idx(
            EvaBlock,
            attn_type=attn_type,
            swiglu_mlp=cfg.swiglu_mlp,
            scale_mlp=cfg.scale_mlp_norm,
            scale_attn_inner=cfg.scale_attn_inner_norm,
            qkv_fused=cfg.qkv_fused,
            num_prefix_tokens=num_prefix_tokens,
            is_causal=getattr(cfg, "is_causal", False),  # despite the 'nepa' pretrained task, `is_causal` is False
            block_norm_type=block_norm_type,
            post_norm_alpha=post_norm_alpha,
            post_norm_skip_first_layer=cfg.post_norm_skip_first_layer,
            linear_attn=cfg.linear_attn,
            sparse_linear_attn=cfg.sparse_linear_attn,
        )
    else:
        # Standard ViT block
        block_fn = cfg.block_fn or Block
        if cfg.scale_mlp_norm or cfg.scale_attn_inner_norm:
            # param names differ between EVA vs non-EVA block types
            block_fn = partial(
                block_fn,
                scale_mlp_norm=cfg.scale_mlp_norm,
                scale_attn_norm=cfg.scale_attn_inner_norm,
            )
        return block_fn


naflexvit.get_block_fn = get_block_fn  # type: ignore

# -------------- Naflex Config ------------------ #


@dataclass
class NaFlexVitCfg:
    """Configuration for FlexVit model.

    This dataclass contains the bulk of model configuration parameters,
    with core parameters (img_size, in_chans, num_classes, etc.) remaining
    as direct constructor arguments for API compatibility.
    """

    # Architecture parameters
    patch_size: int = 1
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    scale_mlp_norm: bool = False  # Apply scaling norm to MLP

    # Attention parameters
    qkv_bias: bool = True
    qk_norm: bool = True
    proj_bias: bool = True
    attn_drop_rate: float = 0.0
    scale_attn_inner_norm: bool = False  # Apply scaling norm to attn context
    linear_attn: dict[str, Any] | None = (
        None  # kwargs for EvaLinearAttention (feature_map/use_bf16/tie_feature_map_qk/eps)
    )
    sparse_linear_attn: dict[str, Any] | None = (
        None  # kwargs for EvaSparseLinearAttention (topk/feature_map/BLKQ/BLKK/use_bf16/tie_feature_map_qk)
    )

    # Regularization
    init_values: Optional[float] = None  # Layer-scale init values (layer-scale enabled if not None)
    drop_rate: float = 0.0  # Dropout rate for classifier
    pos_drop_rate: float = 0.0  # Dropout rate for position embeddings
    patch_drop_rate: float = 0.0  # Dropout rate for patch tokens
    proj_drop_rate: float = 0.0  # Dropout rate for linear projections
    drop_path_rate: float = 0.0  # Stochastic depth drop rate

    # Prefix token configuration
    class_token: bool = False  # Use class token
    reg_tokens: int = 0  # Number of register tokens

    # Position embedding configuration
    pos_embed: str = "learned"  # Type of position embedding ('learned', 'factorized', 'rope', 'none')
    # Grid size for position embedding initialization
    pos_embed_grid_size: Optional[Tuple[int, int]] = (16, 16)
    pos_embed_interp_mode: str = "bicubic"  # Interpolation mode for position embedding resizing
    pos_embed_ar_preserving: bool = False  # Whether to preserve aspect ratio during position embedding interpolation
    pos_embed_use_grid_sample: bool = False  # Whether to use grid_sample for naflex position embedding interpolation

    # ROPE specific configuration
    rope_type: str = (
        "axial"  # ROPE type: '' or 'none' for no ROPE, 'axial' for standard, 'mixed' for learnable frequencies
    )
    rope_temperature: float = 10000.0  # Temperature for ROPE frequency computation
    rope_ref_feat_shape: Optional[Tuple[int, int]] = None
    rope_grid_offset: float = 0.0  # Grid offset for non-pixel ROPE mode
    rope_grid_indexing: str = "ij"  # Grid indexing mode for ROPE ('ij' or 'xy')

    # Image processing
    dynamic_img_pad: bool = False  # Whether to enable dynamic padding for variable resolution

    # Other architecture choices
    pre_norm: bool = True  # Whether to apply normalization before attention/MLP layers (start of blocks)
    final_norm: bool = True  # Whether to apply final normalization before pooling and classifier (end of blocks)
    fc_norm: Optional[bool] = None  # Whether to normalize features before final classifier (after pooling)
    post_norm_alpha: float | None = None  # Keel post-norm alpha, defaults to 2 * depth when None
    post_norm_skip_first_layer: bool = True  # Skip post-norm on the first block (Keel init trick)

    # Global pooling setup
    global_pool: str = ""  # Type of global pooling for final sequence
    pool_include_prefix: bool = False  # Whether to include class/register prefix tokens in global pooling
    attn_pool_num_heads: Optional[int] = None  # Override num_heads for attention pool
    attn_pool_mlp_ratio: Optional[float] = None  # Override mlp_ratio for attention pool

    # Weight initialization
    weight_init: str = "jax"  # Weight initialization scheme
    fix_init: bool = True  # Apply weight initialization fix (scaling w/ layer index)

    # Embedding configuration
    embed_proj_type: str = "linear"  # Type of embedding layer ('conv' or 'linear')
    input_norm_layer: Optional[str] = None  # Normalization layer for embeddings input (before input projection)
    embed_norm_layer: Optional[str] = None  # Normalization layer for embeddings (after input projection)

    # Layer implementations
    norm_layer: Optional[str] = "rmsnorm"  # Normalization layer for transformer blocks
    act_layer: Optional[str] = None  # Activation layer for MLP blocks
    block_fn: Optional[str] = None  # Transformer block implementation class name
    mlp_layer: Optional[str] = None  # MLP implementation class name

    # EVA-specific parameters
    attn_type: str = "eva"  # Attention type: 'standard', 'eva', 'rope', 'gated', 'linear', 'sparse_linear'
    swiglu_mlp: bool = True  # Use SwiGLU MLP variant
    qkv_fused: bool = True  # Whether to use fused QKV projections

    # Variable patch size support
    enable_patch_interpolator: bool = True  # Enable dynamic patch size support

    # Tokenization related
    img_size: int = 32
    in_chans: int = 256
    out_chans: int = 16
    out_2d_latent: bool = True
    unpatch_size: Optional[int] = None  # if None, use patch_size
    compile_model: bool = False  # control torch.compile

    # for cross-attention
    cross_attn_tokens: int = -1
    cross_attn_ratio: float = 0.5

    # adaptive generation decoder
    is_first_cat_noise: bool = False

    # 'ijepa', 'lejepa', None for no pretrained task
    pretrained_type: Any = None  # Union[str, list[str]]

    # IBOT head cfgs
    ibot_n_prototypes: int = 65536
    ibot_head_hidden_dim: int = 2048
    ibot_bottleneck_dim: int = 256
    ibot_nlayers: int = 3

    # MAE head cfgs
    mae_decoder_dim: int = 768
    mae_decoder_pe_init_type: str = "trunc_normal"  # trunc_normal or sincos
    mae_decoder_depth: int = 8
    mae_mask_type: str = "kaiming"  # 'kaiming' or 'pixio'
    mae_mask_ratio: float = 0.8
    mae_decoder_head: str = "seperated"  # 'shared' or 'seperated'
    mae_pixio_mask_grid: int = 2
    mae_latent_size: int = 14  # latent size for mae

    # NePA (Next Embedding Prediction) cfgs
    nepa_is_causal: bool = True  # Whether to use causal attention for NePA
    nepa_shift_prediction: bool = True  # Whether to predict next position (shift by 1)
    # attn_mask_type: 'is_causal' (implicit, requires gated attn) or 'explicit' (default) or None (auto)
    nepa_attn_mask_type: Optional[str] = "explicit"
    is_causal: bool = False  # Global model causality (if True, applies to all tasks)

    # Hyper-connection (mHC) configuration
    hc_streams: int = -1  # -1 means no HC, only residual connection
    hc_implem: str = "naive"  # "naive" or "cuda"
    hc_other_kwargs: Any = None  # sinkhorn_iters, sinkhorn_tau, etc.

    # Value residual configuration
    v_residual: bool = False  # Whether to use attention value residual


def ffn_init_fn(module: nn.Module, d_model: int, d_ffn: int, layer_id: int | None = None):
    std = 1.0 / math.sqrt(d_model)
    torch.nn.init.trunc_normal_(module.layer1.weight, std=std, a=-3 * std, b=3 * std)  # ty: ignore error[invalid-argument-type]

    # scale init by depth as in https://arxiv.org/abs/1908.11365 -- worked slightly better.
    std = 1.0 / math.sqrt(d_ffn)
    if layer_id is not None:
        std = std / math.sqrt(2 * (layer_id + 1))
    torch.nn.init.trunc_normal_(module.layer2.weight, std=std, a=-3 * std, b=3 * std)  # ty: ignore error[invalid-argument-type]


def attention_init_fn(module, layer_id: int | None = None):
    std = 1.0 / math.sqrt(module.q_dim)
    torch.nn.init.trunc_normal_(module.q_proj.weight, std=std, a=-3 * std, b=3 * std)
    std = 1.0 / math.sqrt(module.ctx_dim)
    torch.nn.init.trunc_normal_(module.k_proj.weight, std=std, a=-3 * std, b=3 * std)
    torch.nn.init.trunc_normal_(module.v_proj.weight, std=std, a=-3 * std, b=3 * std)

    std = 1.0 / math.sqrt(module.inner_dim)
    torch.nn.init.trunc_normal_(module.output_proj.weight, std=std, a=-3 * std, b=3 * std)

    for layer in module.q_norm, module.k_norm, module.v_norm:
        if hasattr(layer, "init_weights"):
            layer.init_weights()


class Transformer(NaFlexVit):
    def __init__(self, cfg: NaFlexVitCfg):
        super().__init__(cfg, in_chans=cfg.in_chans, img_size=cfg.img_size)  # ty: ignore error[invalid-argument-type]
        self.cfg = cfg
        self._build_head(cfg)

        # Build hyperconnections if enabled
        self._use_hc = False
        self._hc_streams = cfg.hc_streams
        self.v_residual = cfg.v_residual
        if cfg.hc_streams > 0:
            self._build_hyperconnection(cfg.hc_streams, cfg.hc_implem, cfg.hc_other_kwargs or {})
            self._rebuild_blocks_with_hc(cfg)

        if cfg.compile_model:
            logger.log("NOTE", f"[Naflex Transformer]: Compiling model ...")
            for i in range(len(self.blocks)):
                self.blocks[i] = torch.compile(self.blocks[i])  # ty: ignore error[invalid-assignment]
            # self.head = torch.compile(self.head)

    def _build_hyperconnection(self, hc_streams: int, implem: str, hc_other_kwargs: dict):
        """Build hyperconnection (mHC) infrastructure."""
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
            if implem == "cuda":
                logger.warning(
                    "Using CUDA implementation for HyperConnections, this is un-tested and may not work correctly."
                )

            self.init_hc_fn, self.hc_expand_stream, self.reduce_stream = init_expand_reduce_fm(hc_streams)

    def _rebuild_blocks_with_hc(self, cfg: NaFlexVitCfg):
        """Rebuild blocks with hyperconnection kwargs."""
        if not self._use_hc or self.init_hc_fn is None:
            return

        hyperconnection_kwargs = {"init_hc": self.init_hc_fn, **self.hc_other_kwargs}
        block_fn = get_block_fn(cfg)
        norm_layer = get_norm_layer(cfg.norm_layer) or nn.LayerNorm
        act_layer = get_act_layer(cfg.act_layer) or nn.GELU
        dpr = calculate_drop_path_rates(cfg.drop_path_rate, cfg.depth)
        dd: _DeviceDTypeKwargs = {"device": None, "dtype": None}

        new_blocks = nn.ModuleList()
        for i in range(cfg.depth):
            blk = block_fn(
                dim=cfg.embed_dim,
                num_heads=cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio,
                qkv_bias=cfg.qkv_bias,
                qk_norm=cfg.qk_norm,
                proj_bias=cfg.proj_bias,
                init_values=cfg.init_values,
                proj_drop=cfg.proj_drop_rate,
                attn_drop=cfg.attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                v_residual=cfg.v_residual,
                layer_idx=i,
                hyperconnection_kwargs=hyperconnection_kwargs,
                **dd,
            )
            new_blocks.append(blk)
        self.blocks = new_blocks
        logger.info(f"[Naflex Transformer]: Rebuilt {cfg.depth} blocks with mHC support")

    def _build_head(self, cfg: NaFlexVitCfg):
        norm_layer = get_norm_layer(cfg.norm_layer) or nn.LayerNorm
        self.patch_size = cfg.patch_size
        self.unpatch_size = cfg.unpatch_size or cfg.patch_size
        self.head = nn.Sequential(
            norm_layer(cfg.embed_dim),
            nn.Linear(cfg.embed_dim, cfg.out_chans * self.unpatch_size**2, bias=True),
        )

    def unpatchify(self, x: torch.Tensor, hw=None):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.cfg.out_chans
        p = self.unpatch_size
        if hw is None:
            h = w = int(x.shape[1] ** 0.5)
            assert h * w == x.shape[1]
        else:
            h, w = hw

        x = einops.rearrange(
            x,
            "bs (h w) (p1 p2 c) -> bs c (h p1) (w p2)",
            h=h // p,
            w=w // p,
            p1=p,
            p2=p,
            c=c,
        )
        return x

    def _forward_after_backbone(self, x, hw: list | None):
        x = self.head(x)
        if self.cfg.out_2d_latent and hw is not None:
            x = self.unpatchify(x, hw)
        else:
            assert hw is None and self.unpatch_size == 1, (
                f"HW is not None or the unpatch_size is not 1, when force no patchify"
            )
            # Let the output be 1D tensor
        return x

    def _get_output_shape(self, x):
        hw = x.shape[-2:]
        if self.cfg.unpatch_size is not None:
            out_hw = (torch.tensor(hw) // self.patch_size * self.unpatch_size).tolist()
        else:
            out_hw = hw
        return out_hw

    def forward_features(
        self,
        patches: torch.Tensor,
        patch_coord: Optional[torch.Tensor] = None,
        patch_valid: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Any:
        """Override forward_features to add mHC stream expand and value residual."""
        naflex_mode = patch_coord is not None

        # Pass through patch & abs position embedding module with patch coordinate/type support
        embeds = self._forward_embeds(
            patches,
            patch_coord=patch_coord,
            patch_valid=patch_valid,
            attn_mask=attn_mask,
        )
        x = embeds["patches"]
        rope_embeds = embeds.get("rope_embeds", None)
        keep_indices = embeds.get("keep_indices", None)
        attn_mask = embeds.get("attn_mask", None)

        # mHC expansion
        if self._use_hc:
            x = self.hc_expand_stream(x)

        # Value residual initialization
        v_residual_v1 = None
        if self.v_residual:
            v_residual_v1 = x.clone()

        # Apply transformer blocks
        do_checkpointing = getattr(self, "grad_checkpointing", False) and not torch.jit.is_scripting()

        if self.rope_is_mixed and rope_embeds is not None:
            # Mixed mode with per-layer embeddings (list or iterator)
            for i, (blk, rope_embed) in enumerate(zip(self.blocks, rope_embeds)):
                if self.training and self.patch_drop is not None and keep_indices is not None:
                    # Apply patch dropout to rope_embed if needed
                    from timm.models.naflexvit import apply_keep_indices_nlc

                    rope_embed = apply_keep_indices_nlc(
                        x,
                        rope_embed,
                        keep_indices,
                        pos_embed_has_batch=naflex_mode,
                    )
                if do_checkpointing:
                    x = torch_checkpoint(
                        blk, x, rope=rope_embed, attn_mask=attn_mask, v_residual_v1=v_residual_v1, use_reentrant=False
                    )
                else:
                    x = blk(x, rope=rope_embed, attn_mask=attn_mask, v_residual_v1=v_residual_v1)
        elif rope_embeds is not None:
            # Axial ROPE mode with shared embeddings
            for blk in self.blocks:
                if do_checkpointing:
                    x = torch_checkpoint(
                        blk, x, rope=rope_embeds, attn_mask=attn_mask, v_residual_v1=v_residual_v1, use_reentrant=False
                    )
                else:
                    x = blk(x, rope=rope_embeds, attn_mask=attn_mask, v_residual_v1=v_residual_v1)
        else:
            for blk in self.blocks:
                if do_checkpointing:
                    x = torch_checkpoint(blk, x, attn_mask=attn_mask, v_residual_v1=v_residual_v1, use_reentrant=False)
                else:
                    x = blk(x, attn_mask=attn_mask, v_residual_v1=v_residual_v1)

        x = self.norm(x)

        if naflex_mode:
            return {
                "patches": x,
                "patch_valid": embeds["patch_valid"],
                "keep_indices": keep_indices,
            }

        return x

    def forward(self, x, output_type: str | None = None, **_ignored_kwargs):
        # Output HW
        if output_type in (None, "2d"):
            out_hw = self._get_output_shape(x)
        else:
            out_hw = None  # keep the output to be 1D tensor

        # Features with mHC stream handling
        x = self.forward_features(x)
        x = cast(torch.Tensor, x)

        # Reduce mHC streams if enabled
        if self._use_hc:
            x = self.reduce_stream(x)

        x = x[:, self.num_prefix_tokens :]

        # Head
        x = self._forward_after_backbone(x, out_hw)

        return x

    @classmethod
    @function_config_to_basic_types
    def create_model(cls, **overrides):
        cfg: NaFlexVitCfg = dataclass_from_dict(NaFlexVitCfg, overrides)

        # Support for NePA implicit causal mask
        # Configure attn_mask_type: 'is_causal' (implicit) or 'explicit'
        if cfg.pretrained_type is not None and "nepa" in cfg.pretrained_type and cfg.nepa_is_causal:
            mask_type = cfg.nepa_attn_mask_type

            # Auto-selection if None
            if mask_type is None:
                mask_type = "is_causal" if cfg.attn_type == "gated" else "explicit"

            if mask_type == "is_causal":
                if cfg.attn_type == "gated":
                    cfg.is_causal = True
                    logger.info("[NePA]: Enabled implicit causal attention (is_causal=True for GatedAttention)")
                else:
                    logger.warning(
                        f"[NePA]: 'is_causal' mask requested but attn_type='{cfg.attn_type}' "
                        "does not support it. Falling back to explicit mask."
                    )
                    cfg.is_causal = False
            else:
                # Explicit mode
                cfg.is_causal = False

        model = cls(cfg)
        return model

    def forward_intermediates(
        self,
        x: Union[torch.Tensor, dict[str, torch.Tensor]],
        indices: Optional[Union[int, List[int]]] = None,
        return_prefix_tokens: bool = False,
        norm: bool = False,
        stop_early: bool = False,
        output_fmt: str = "NCHW",
        intermediates_only: bool = False,
        output_dict: bool = False,
        patch_coord: Optional[torch.Tensor] = None,
        patch_valid: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]], dict[str, Any]]:
        """Override forward_intermediates to add mHC stream expand and value residual."""
        assert output_fmt in ("NCHW", "NLC"), "Output format must be one of NCHW or NLC."
        reshape = output_fmt == "NCHW"
        intermediates = []
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)

        if isinstance(x, dict):
            # Handle dictionary input from NaFlex collator
            patch_coord = x["patch_coord"]
            patch_valid = x["patch_valid"]
            patches = x["patches"]
        else:
            patches = x
            height, width = x.shape[-2:]
            H, W = self.embeds.dynamic_feat_size((height, width))

        # Pass through patch & abs position embedding module
        embeds = self._forward_embeds(
            patches,
            patch_coord=patch_coord,
            patch_valid=patch_valid,
            attn_mask=attn_mask,
        )
        x = embeds["patches"]
        rope_embeds = embeds.get("rope_embeds", None)
        keep_indices = embeds.get("keep_indices", None)
        attn_mask = embeds.get("attn_mask", None)
        naflex_mode = patch_coord is not None

        # mHC expansion
        if self._use_hc:
            x = self.hc_expand_stream(x)

        # Value residual initialization
        v_residual_v1 = None
        if self.v_residual:
            v_residual_v1 = x.clone()

        # Forward pass through blocks
        if torch.jit.is_scripting() or not stop_early:
            blocks: list[nn.Module] = list(self.blocks)
        else:
            blocks = list(self.blocks[: max_index + 1])  # ty: ignore error[invalid-argument-type]

        do_checkpointing = getattr(self, "grad_checkpointing", False) and not torch.jit.is_scripting()

        if self.rope_is_mixed and rope_embeds is not None:
            # Mixed mode with per-layer embeddings (list or iterator)
            for i, (blk, rope_embed) in enumerate(zip(self.blocks, rope_embeds)):
                if self.training and self.patch_drop is not None and keep_indices is not None:
                    from timm.models.naflexvit import apply_keep_indices_nlc

                    rope_embed = apply_keep_indices_nlc(
                        x,
                        rope_embed,
                        keep_indices,
                        pos_embed_has_batch=naflex_mode,
                    )
                if do_checkpointing:
                    x = torch_checkpoint(
                        blk, x, rope=rope_embed, attn_mask=attn_mask, v_residual_v1=v_residual_v1, use_reentrant=False
                    )
                else:
                    x = blk(x, rope=rope_embed, attn_mask=attn_mask, v_residual_v1=v_residual_v1)
                if i in take_indices:
                    intermediates.append(self.norm(x) if norm else x)
        else:
            for i, blk in enumerate(blocks):
                # Axial ROPE mode or no ROPE
                r = rope_embeds if rope_embeds is not None else None
                if do_checkpointing:
                    x = torch_checkpoint(
                        blk, x, rope=r, attn_mask=attn_mask, v_residual_v1=v_residual_v1, use_reentrant=False
                    )
                else:
                    x = blk(x, rope=r, attn_mask=attn_mask, v_residual_v1=v_residual_v1)
                if i in take_indices:
                    intermediates.append(self.norm(x) if norm else x)

        # Reduce mHC streams if enabled
        if self._use_hc:
            x = self.reduce_stream(x)
            intermediates = [self.reduce_stream(y) for y in intermediates]

        # Process intermediates
        if self.num_prefix_tokens:
            prefix_tokens = [y[:, 0 : self.num_prefix_tokens] for y in intermediates]
            intermediates = [y[:, self.num_prefix_tokens :] for y in intermediates]
        else:
            prefix_tokens = None

        if reshape:
            intermediates = [y.reshape(y.shape[0], H, W, -1).permute(0, 3, 1, 2).contiguous() for y in intermediates]

        # For dictionary output
        if output_dict:
            result_dict = {}
            result_dict["image_intermediates"] = intermediates
            if prefix_tokens is not None and return_prefix_tokens:
                result_dict["image_intermediates_prefix"] = prefix_tokens
            if not intermediates_only:
                x_final = self.norm(x)
                result_dict["image_features"] = x_final
            return result_dict

        if not torch.jit.is_scripting() and return_prefix_tokens and prefix_tokens is not None:
            intermediates = list(zip(intermediates, prefix_tokens))

        if intermediates_only:
            return intermediates

        x = self.norm(x)
        return x, intermediates

    def init_weights(self, mode: str = "jax", needs_reset: bool = True):
        super().init_weights(mode=mode, needs_reset=needs_reset)

        def rescale(p, layer_id):
            p.div_(math.sqrt(2.0 * layer_id))

        # Rescale the depth
        rescale_layer = True
        if rescale_layer:
            for layer_id, blk in enumerate(self.blocks, 1):
                rescale(blk.attn.proj.weight.data, layer_id)
                rescale(blk.mlp.fc2.weight.data, layer_id)
            logger.info("[Naflex Transformer]: Rescale the layers initialization for more stable training")


class IJEPANaFlexViT(Transformer):
    def __init__(self, cfg: NaFlexVitCfg):
        super().__init__(cfg)

        self.pretrained_type: str = cfg.pretrained_type

        # --------- Build the heads or decoders --------- #
        if "lejepa" in self.pretrained_type:
            self._build_lejepa_head(cfg)
        if "ibot" in self.pretrained_type:
            self._build_ibot_head(cfg)
        if "latent_mae" in self.pretrained_type or "pixel_mae" in self.pretrained_type:
            self._build_mae_decoder(cfg)
        if "nepa" in self.pretrained_type:
            self._build_nepa_head(cfg)

    def _build_mae_decoder(self, cfg: NaFlexVitCfg):
        """
        Latent MAE is more like JEPA,
            image x -> CNN encoder -> latent -> mask inside Naflex transformer (MAE encoder) ->
                -> merge masked tokens in  Naflex transformer (MAE decoder) -> predict the masked tokens (MSE loss)
            this type MAE works at latent, so called latent_mae

        Pixel MAE is more like the original pixel-space MAE,
            image x -> CNN encoder -> latent -> mask inside Naflex transformer (MAE encoder) ->
                -> merge masked tokens in  Naflex transformer (MAE decoder) -> CNN decoder -> reconstruct the image
            this type MAE works at pixel, so called pixel_mae
        """
        assert "latent_mae" in cfg.pretrained_type or "pixel_mae" in cfg.pretrained_type, "MAE decoder is not supported"
        embed_dim = cfg.mae_decoder_dim
        mae_decoder_depth = cfg.mae_decoder_depth
        block_fn = get_block_fn(cfg)
        dpr = calculate_drop_path_rates(cfg.drop_path_rate, cfg.depth)  # stochastic depth decay rule
        norm_layer = get_norm_layer(cfg.norm_layer) or nn.LayerNorm
        act_layer = get_act_layer(cfg.act_layer) or nn.GELU
        mlp_layer = cfg.mlp_layer or Mlp
        dd: _DeviceDTypeKwargs = {"device": None, "dtype": None}
        self.mae_decoder = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=cfg.num_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    qkv_bias=cfg.qkv_bias,
                    qk_norm=cfg.qk_norm,
                    proj_bias=cfg.proj_bias,
                    init_values=cfg.init_values,
                    proj_drop=cfg.proj_drop_rate,
                    attn_drop=cfg.attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    mlp_layer=mlp_layer,
                    **dd,
                )
                for i in range(mae_decoder_depth)
            ]
        )
        logger.debug("Build MAE decoder", name="MAE Naflex Transformer")

        # --------- Decoder embeddings, mask token, and decoder norm/predictor ----------- #

        # Init the decoder blocks
        init_mode = "timm"
        init_fn = get_init_weights_vit(mode=init_mode)
        # mae implement this using jax init fn
        self.mae_decoder.apply(init_fn)

        n_patches = (cfg.mae_latent_size // cfg.patch_size) ** 2
        # the pixio implementation fix the norm at the encoder's head
        # here we move the norm with mae embedder
        self.mae_embedder = nn.Sequential(
            create_norm_layer(cfg.norm_layer, cfg.embed_dim),
            nn.Linear(cfg.embed_dim, cfg.mae_decoder_dim, bias=True),
        )
        self.mae_mask_token = nn.Parameter(torch.zeros(1, 1, cfg.mae_decoder_dim))
        self.mae_pos_embed = nn.Parameter(torch.zeros(1, n_patches + self.num_prefix_tokens, cfg.mae_decoder_dim))
        if cfg.mae_decoder_head == "seperated":
            self.mae_head = nn.Sequential(
                norm_layer(cfg.mae_decoder_dim),
                nn.Linear(cfg.mae_decoder_dim, cfg.patch_size**2 * cfg.out_chans, bias=True),
            )

        # init weights
        pos_embed_init_type: str = cfg.mae_decoder_pe_init_type  # [trunc_norm, sincos]
        if pos_embed_init_type == "trunc_normal":
            nn.init.trunc_normal_(self.mae_pos_embed, std=0.02)
        elif pos_embed_init_type == "sincos":
            init_pe = get_2d_sincos_pos_embed(
                cfg.mae_decoder_dim,
                grid_size=cfg.img_size // cfg.patch_size,
                cls_token=True if self.num_prefix_tokens > 0 else False,
                extra_tokens=self.num_prefix_tokens,
                output_type="pt",
            )
            with torch.no_grad():
                self.mae_pos_embed.copy_(init_pe.unsqueeze(0))
        nn.init.normal_(self.mae_mask_token, std=0.02)
        logger.debug("Init the MAE decoder", name="MAE Naflex Transformer")

    def _build_lejepa_head(self, cfg):
        from src.stage1.self_supervised.lejepa_aug import create_lejepa_projector

        self.lejepa_projector = create_lejepa_projector(
            cfg.embed_dim,
            cfg.embed_dim,
            mean_out_hw=False,
            # mean_out_hw=not cfg.class_token,  # if use class, not mean out the spatial tokens
        )
        logger.info("[IJEPA Naflex Transformer]: Build LeJEPA head")

    def _build_ibot_head(self, cfg):
        from src.stage1.self_supervised.dino.layers.dino_head import DINOHead

        self.ibot_head = DINOHead(
            in_dim=cfg.embed_dim,
            out_dim=cfg.ibot_n_prototypes,
            hidden_dim=cfg.ibot_head_hidden_dim,
            bottleneck_dim=cfg.ibot_bottleneck_dim,
            nlayers=cfg.ibot_nlayers,
        )
        self.mask_token = nn.Parameter(torch.zeros(1, cfg.embed_dim))
        logger.info("[IBOT Naflex Transformer]: Build iBOT head")

    def _build_nepa_head(self, cfg: NaFlexVitCfg):
        """Build NePA (Next Embedding Prediction) pretraining components.

        NePA doesn't require an additional head - the loss is computed directly
        on the embedding space using cosine similarity between transformer output
        and (shifted) input embeddings.
        """
        self.nepa_is_causal = cfg.nepa_is_causal
        self.nepa_shift = cfg.nepa_shift_prediction
        logger.info(
            f"[NePA NaFlex Transformer]: Build NePA pretraining support "
            f"(causal={self.nepa_is_causal}, shift={self.nepa_shift}, nepa_attn_mask_type={cfg.nepa_attn_mask_type})"
        )

    def _forward_nepa_backbone(
        self,
        x: torch.Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass for NePA pretraining with optional causal attention.

        NePA (Next Embedding Prediction) trains the model to predict the next
        position's input embedding from the current position's output. This is
        similar to autoregressive language modeling but in latent space.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            h_in: [B, T, D] input embeddings before transformer blocks
            h_out: [B, T, D] output hidden states after transformer blocks
        """
        naflex_mode = False

        # 1. Get embeddings (before transformer blocks)
        embeds = self._forward_embeds(x, patch_coord=None, patch_valid=None, attn_mask=None, masks=None)
        h_in = embeds["patches"].clone()  # [B, T, D] - input embeddings
        x_forward = embeds["patches"]
        rope_embeds = embeds.get("rope_embeds", None)

        # 2. Create causal attention mask if enabled
        causal_mask = None

        # Check if we should use implicit causal implementation
        # Implicit mode is active if model.cfg.is_causal is True.
        # This was configured in create_model based on nepa_attn_mask_type.
        is_model_causal = getattr(self.cfg, "is_causal", True)

        # Only create explicit mask if NePA implies causal AND model is NOT natively causal
        if getattr(self, "nepa_is_causal", True) and not is_model_causal:
            seq_len = x_forward.shape[1]
            # Upper triangular mask: position i cannot attend to positions > i
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x_forward.device, dtype=x_forward.dtype),
                diagonal=1,
            )
            # Convert to attention mask format (0 = attend, -inf = mask)
            causal_mask = causal_mask * torch.finfo(x_forward.dtype).min

        # 3. Forward through transformer blocks with causal attention
        do_checkpointing = self.grad_checkpointing and self.training and not torch.jit.is_scripting()
        for blk in self.blocks:
            if do_checkpointing:
                x_forward = torch_checkpoint(blk, x_forward, rope_embeds, causal_mask, use_reentrant=False)
            else:
                x_forward = blk(x_forward, rope=rope_embeds, attn_mask=causal_mask)

        # 4. Apply final layer norm
        h_out = self.norm(x_forward)  # [B, T, D] - output hidden states

        return h_in, h_out

    def _prepare_masks(self, masks: Tensor | list[Tensor] | None = None) -> list[Tensor] | None:
        """Ensure the masks are a list of tensors."""
        if masks is None:
            return None
        if isinstance(masks, list):
            return list(masks)
        assert torch.is_tensor(masks)
        return [masks]

    def _ibot_apply_masks(self, x: Tensor, masks: Tensor) -> Tensor:
        assert masks.dtype is torch.bool
        x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
        return x

    def _jepa_apply_masks(self, x, masks: list[Tensor]):
        all_x = []
        for m in masks:
            mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
            all_x += [torch.gather(x, dim=1, index=mask_keep)]
        return torch.cat(all_x, dim=0)

    def _mae_apply_masks(self, x: Tensor, masks: Tensor) -> Tensor:
        """masks is a Tensor of indices of [B, S_masked]"""
        D = x.size(-1)
        assert masks.dtype in (torch.int32, torch.int64)
        x = torch.gather(x, dim=1, index=masks.unsqueeze(-1).repeat(1, 1, D))
        return x

    def _forward_embeds(
        self,
        x,
        patch_coord,
        patch_valid,
        attn_mask,
        masks: list[Tensor] | Tensor | None = None,
    ):
        """Forward pass through patch / abs pos / rope pos embeds and patch dropout

        IJEPA masking strategy: mask out the patches, and drop them.
            For this instance, `masks` is a list of Int32 mask indices shaped as [B, S_masked]
                e.g., [tensor([[ 14,  28,  42,  56,  70,  84,  98, 112, 126, 140, 154],
                        [ 30,  31,  32,  44,  45,  46,  58,  59,  60,  72,  73]])]
                that can be torch.gather indexed.
            RoPE actions:
            Equals at
            # axial rope: [S, headD * 2]
            # rope: [S, headD * 2] -> [B, 1, S, headD * 2]
            rope_embeds = rope_embeds[None, None].repeat(x.shape[0], 1, 1, 1)
            # masks -> [B, S_masked] -> [B, 1, S_masked, headD * 2]
            # gather -> [B, S_masked, headD * 2]
            m_repeated = m[:, None, :, None].repeat(
                1, 1, 1, rope_embeds.size(-1)
            )
            rope_masked += [rope_embeds.gather(-2, m_repeated)]

        IBOT masking strategy: mask the patches with a learnable token and not drop them.
            where(masks, mask_token, x)
        """
        naflex_mode = patch_coord is not None
        # patch embed, abs pos embed, returns global grid size as calculated from 'standard' NCHW batches
        x, grid_size = self.embeds(x, patch_coord=patch_coord, patch_valid=patch_valid)

        # Generate ROPE embeddings at model level
        rope_embeds = None
        if self.rope is not None:
            if patch_coord is not None:
                # NaFlex mode - variable grid sizes
                rope_embeds = self._generate_rope_naflex(x, patch_coord)
            elif grid_size is not None:
                # Standard mode - fixed grid size
                rope_embeds = self.rope.get_embed(shape=grid_size)  # ty: ignore error[call-non-callable]
            else:
                assert False, "Expected one of patch_coord or grid_size to be valid"

        # Apply patch dropout with coordinated updates
        keep_indices: Optional[torch.Tensor] = None
        if self.training and self.patch_drop is not None:
            x, keep_indices = self.patch_drop(x)  # ty: ignore error[call-non-callable]
            # keep_indices excludes prefix tokens, can use directly on patch_valid & rope embeds
            if patch_valid is not None:
                patch_valid = patch_valid.gather(1, keep_indices)
            if rope_embeds is not None and not self.rope_is_mixed:
                # Update ROPE embeddings to match dropped tokens (only for axial mode)
                # Batch dim already present in NaFlex mode, but will be added in standard mode.
                assert torch.is_tensor(rope_embeds)
                rope_embeds = apply_keep_indices_nlc(x, rope_embeds, keep_indices, pos_embed_has_batch=naflex_mode)
                if not naflex_mode:
                    # B, N, dim -> B, 1, N, dim. Need head dim added for standard mode, already added in NaFlex.
                    rope_embeds = rope_embeds.unsqueeze(1)

        # Create attention mask from patch_valid after patch dropout applied
        if attn_mask is None:
            attn_mask = create_attention_mask(patch_valid, num_prefix_tokens=self.num_prefix_tokens, dtype=x.dtype)

        # ------------ Apply masks ------------ #
        if masks is not None:
            # Apply masks
            prefixed_tokens, x = x[:, : self.num_prefix_tokens], x[:, self.num_prefix_tokens :]
            if "ibot" in self.pretrained_type:
                assert torch.is_tensor(masks)
                x = self._ibot_apply_masks(x, masks=masks)
            elif "ijepa" in self.pretrained_type:
                if not isinstance(masks, list):
                    raise TypeError("IJEPA expects a list of index tensors.")
                masks_list = masks
                x = self._jepa_apply_masks(x, masks=masks_list)  # ty: ignore error[invalid-argument-type]
            elif self._is_mae():
                assert torch.is_tensor(masks)
                x = self._mae_apply_masks(x, masks=masks)

            x = torch.cat([prefixed_tokens, x], dim=1)

            # Rope related, rope is applied in attention module
            if rope_embeds is not None:
                ##### IJEPA masking strategy: mask out
                if "ijepa" in self.pretrained_type:
                    rope_masked = []
                    for m in masks_list:
                        # m: [B, S_masked] indices
                        assert not self.rope_is_mixed, "mixed rope is not supported in JEPA training"
                        rope_masked += [get_at("[S] ropeD, B S_masked -> B 1 S_masked ropeD", rope_embeds, m)]
                    rope_embeds = torch.cat(rope_masked, dim=0)  # [B*n_masks, 1, S_masked, ropeD]
                elif self._is_mae():
                    # masks: [B, S_masked] indices, can be cat since the MAE has the same length token to keep
                    # for each sample in a batch
                    assert torch.is_tensor(masks), f"masks should be a tensor, got {type(masks)}"
                    rope_embeds = get_at("[S] ropeD, B S_masked -> B 1 S_masked ropeD", rope_embeds, masks)
                else:
                    # Do nothing to rope, only mask the embeded tokens
                    masks = cast(Tensor, masks)

        x = self.norm_pre(x)
        return {
            "patches": x,
            "patch_valid": patch_valid,
            "rope_embeds": rope_embeds,
            "attn_mask": attn_mask,
            "keep_indices": keep_indices,
        }

    def forward_features(
        self,
        patches: torch.Tensor,
        patch_coord: Optional[torch.Tensor] = None,
        patch_valid: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        masks: Optional[Tensor | List[Tensor]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """ """
        naflex_mode = patch_coord is not None

        # Pass through patch & abs position embedding module with patch coordinate/type support
        embeds = self._forward_embeds(
            patches,
            patch_coord=patch_coord,
            patch_valid=patch_valid,
            attn_mask=attn_mask,
            masks=masks,
        )
        x = embeds["patches"]
        rope_embeds = embeds.get("rope_embeds", None)
        keep_indices = embeds.get("keep_indices", None)
        attn_mask = embeds.get("attn_mask", None)

        # Apply transformer blocks with masked attention and/or ROPE if provided
        do_checkpointing = self.grad_checkpointing and not torch.jit.is_scripting()
        if self.rope_is_mixed and rope_embeds is not None:
            # Mixed mode with per-layer embeddings (list or iterator)
            for i, (blk, rope_embed) in enumerate(zip(self.blocks, rope_embeds)):
                if self.training and self.patch_drop is not None and keep_indices is not None:
                    # Apply patch dropout to rope_embed if needed (batch dim already present in naflex mode)
                    rope_embed = apply_keep_indices_nlc(
                        x,
                        rope_embed,
                        keep_indices,
                        pos_embed_has_batch=naflex_mode,
                    )
                if do_checkpointing:
                    x = torch_checkpoint(blk, x, rope_embed, attn_mask, use_reentrant=False)
                else:
                    x = blk(x, rope=rope_embed, attn_mask=attn_mask)
        elif rope_embeds is not None:
            # Axial ROPE mode with shared embeddings
            for blk in self.blocks:
                if do_checkpointing:
                    x = torch_checkpoint(blk, x, rope_embeds, attn_mask, use_reentrant=False)
                else:
                    x = blk(x, rope=rope_embeds, attn_mask=attn_mask)
        else:
            for blk in self.blocks:
                if do_checkpointing:
                    x = torch_checkpoint(blk, x, None, attn_mask, use_reentrant=False)
                else:
                    x = blk(x, attn_mask=attn_mask)

        x = self.norm(x)

        if naflex_mode:
            return {"patches": x, "patch_valid": embeds.get("patch_valid", None)}

        return x

    def forward(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        patch_coord: Optional[torch.Tensor] = None,
        patch_valid: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        output_type: str | None = None,
        masks: Tensor | list[Tensor] | None = None,  # IJEPA masks
    ):
        """Forward with JEPA masks support"""
        if masks is not None and self.pretrained_type == "ijepa":
            masks = self._prepare_masks(masks=masks)

        naflex_mode = isinstance(x, dict) or patch_coord is not None
        if naflex_mode:
            assert masks is None, "JEPA does not support naflex mode."

            if isinstance(x, dict):
                # Handle dictionary input from NaFlex collator, dict inputs take priority over args
                patches = x["patches"]
                if "patch_valid" in x:
                    patch_valid = x["patch_valid"]

                if "patch_coord" in x:
                    patch_coord = x["patch_coord"]

                if "attn_mask" in x:
                    attn_mask = x["attn_mask"]
            else:
                patches = x
            assert patch_coord is not None, "patch_coord is required in naflex mode"
            assert patch_valid is not None, "patch_valid is required in naflex mode"

            features = self.forward_features(
                patches=patches,
                patch_valid=patch_valid,
                patch_coord=patch_coord,
                attn_mask=attn_mask,
                ##### ! is naflex mode, do not input the jepa masks
            )

            # Pass patches & patch_valid to forward_head for masked pooling
            assert isinstance(features, dict)
            x = self.forward_head(**features)  # ty: ignore error[invalid-argument-type]
        else:
            # * This is the Tensor input x forward pass, not naflex mode ##################
            assert torch.is_tensor(x)

            if output_type in (None, "2d"):
                out_hw = self._get_output_shape(x)
            else:
                out_hw = None  # keep the output to be 1D tensor

            # Features
            x = self.forward_features(x, masks=masks)
            x = cast(torch.Tensor, x)
            x = x[:, self.num_prefix_tokens :]

            # Head
            x = self._forward_after_backbone(x, out_hw)

        return x

    def _forward_after_backbone(self, x: Tensor, hw: list | None) -> Tensor:
        head_out = self.head(x)
        if self.cfg.out_2d_latent and hw is not None:
            out = self.unpatchify(head_out, hw)
        else:
            assert hw is None and self.unpatch_size == 1, (
                f"HW is not None or the unpatch_size is not 1, when force no patchify"
            )
            # Let the output be 1D tensor
            out = head_out

        return out

    def _is_mae(self, pretrained_task: list[str] | None = None):
        pretrained_task = cast(list[str], pretrained_task or self.pretrained_type)
        return "latent_mae" in pretrained_task or "pixel_mae" in pretrained_task

    def _forward_pretrained_backbone(
        self,
        x: torch.Tensor,
        output_type: str = "1d",  # fixed it.
        masks: Optional[Tensor | List[Tensor]] = None,
        masks_indices: Optional[Tensor] = None,
        *,
        pretrained_task: list[str] | None = None,
        **kwargs,
    ):
        terms = {}
        pretrained_task = [] if pretrained_task is None else pretrained_task

        # if output_type in (None, "2d"):
        out_hw = self._get_output_shape(x)
        # else:
        #     out_hw = None  # keep the output to be 1D tensor

        # ---------- forward backbone ---------- #
        x = cast(torch.Tensor, self.forward_features(x, masks=masks))

        # ---------- forward different pretrained heads / decoders ---------- #
        ######### IJepa features ########
        if "ijepa" in self.pretrained_type and "ijepa" in pretrained_task:
            # x is the backbone's out
            terms["ijepa_feat"] = x[:, self.num_prefix_tokens :]

        ######### MAE decoders ########
        if self._is_mae(pretrained_task):
            ids_restore = kwargs.get("ids_restore", None)
            assert ids_restore is not None, "ids_restore is required for MAE decoder"

            # x is masked inside `forward_features`
            x_masked = x
            # 1. embed to decoder dim
            x_dec = self.mae_embedder(x_masked)
            # 2. add mask_token to the masked positions (following pixio pattern)
            x_cls_reg = x_dec[:, : self.num_prefix_tokens, :]
            x_patches_masked = x_dec[:, self.num_prefix_tokens :]
            mask_tokens = self.mae_mask_token.repeat(x.shape[0], ids_restore.shape[1] - x_patches_masked.shape[1], 1)

            # Concatenate visible tokens with mask tokens, then gather (unshuffle)
            x_dec_spatial = torch.cat([x_patches_masked, mask_tokens], dim=1)
            x_dec_spatial = torch.gather(
                x_dec_spatial, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_dec_spatial.shape[2])
            )

            # Append prefix tokens after gather operation
            x_dec = torch.cat([x_cls_reg, x_dec_spatial], dim=1)

            # 3. add decoder's positional embedding
            # Calculate spatial size excluding prefix tokens
            spatial_tokens = x_dec.shape[1] - self.num_prefix_tokens
            hw = int(spatial_tokens**0.5)
            abs_pe = resample_abs_pos_embed(
                self.mae_pos_embed, new_size=[hw, hw], num_prefix_tokens=self.num_prefix_tokens
            )
            x_dec = x_dec + abs_pe
            x_dec = x_dec.contiguous()
            # 4. call decoder blocks and head
            do_checkpointing = self.grad_checkpointing and self.training and not torch.jit.is_scripting()
            for blk in self.mae_decoder:
                if do_checkpointing:
                    x_dec = torch_checkpoint(blk, x_dec, None, None, use_reentrant=False)
                else:
                    x_dec = blk(x_dec)

            if self.cfg.mae_decoder_head == "shared":
                x_dec = self.head(x_dec)
            else:
                x_dec = self.mae_head(x_dec)
            # 5. unpatchify if needed
            # assert out_hw is not None
            p, c, h, w = self.unpatch_size, self.cfg.out_chans, out_hw[0], out_hw[1]
            x_dec_2d = self.unpatchify(x_dec[:, self.num_prefix_tokens :], hw=out_hw)
            # x_dec_2d = einops.rearrange(
            #     x_dec[:, self.num_prefix_tokens:],
            #     "bs (h w) (p1 p2 c) -> bs c (h p1) (w p2)",
            #     h=h // p,
            #     w=w // p,
            #     p1=p,
            #     p2=p,
            #     c=c
            # )
            # 6. form the output
            terms["mae_decode_out"] = x_dec[:, self.num_prefix_tokens :]
            terms["mae_decode_out_2d"] = x_dec_2d

        ######### Lejepa projector #########
        if hasattr(self, "lejepa_projector") and "lejepa" in self.cfg.pretrained_type and "lejepa" in pretrained_task:
            # NOTE: may use all spatial tokens to compute the lejepa loss?
            x_pool = self._pool(x)
            lejepa_proj = self.lejepa_projector(x_pool)  # x is the backbone's out
            terms["lejepa_proj"] = lejepa_proj
            # spatial tokens only
            terms["prefixed_tokens"] = x[:, : self.num_prefix_tokens]

            # Patch tokens projection for sigreg
            x_patches = x[:, self.num_prefix_tokens :]
            patch_t1d = einops.rearrange(x_patches, "b l c -> (b l) c")
            terms["lejepa_proj_patch"] = self.lejepa_projector(patch_t1d)

        ########## IBOT projector ##########
        if hasattr(self, "ibot_head") and "ibot" in self.cfg.pretrained_type and "ibot" in pretrained_task:
            terms["ibot_feat"] = x_tokens = x[:, self.num_prefix_tokens :]

            # Check mask shape alignment if masks provided
            if masks is not None and isinstance(masks, Tensor):
                B, N = masks.shape
                assert x_tokens.shape[:2] == (B, N), (
                    f"Mask shape {masks.shape} does not match feature shape {x_tokens.shape[:2]}"
                )

            if masks_indices is not None:
                # IBOT teacher does
                x_tokens = torch.index_select(x_tokens.flatten(0, 1), dim=0, index=masks_indices)
            terms["ibot_proj"] = self.ibot_head(x_tokens)

        ########## NePA (Next Embedding Prediction) ##########
        if hasattr(self, "nepa_is_causal") and "nepa" in self.cfg.pretrained_type and "nepa" in pretrained_task:
            # Use dedicated NePA forward pass with causal attention
            h_in, h_out = self._forward_nepa_backbone(kwargs.get("nepa_input", x))
            # Compute NePA loss
            shift = getattr(self, "nepa_shift", True)
            nepa_loss = nepa_prediction_loss(h_in, h_out, shift=shift)
            terms["nepa_loss"] = nepa_loss
            terms["nepa_h_in"] = h_in
            terms["nepa_h_out"] = h_out

        # Patch tokens
        x_tokens = x[:, self.num_prefix_tokens :]
        terms["x_patch_tokens"] = x_tokens

        # Return the 1d features as the backbone output
        # not forward by the head
        # terms: [prefixed_tokens, ijepa_feat, lejepa_proj, ibot_feat, ibot_proj, x_patch_tokens]
        return x_tokens, edict(terms)

    def forward_intermedieates(
        self,
        x: Union[torch.Tensor, Dict[str, torch.Tensor]],
        indices: Optional[Union[int, List[int]]] = None,
        return_prefix_tokens: bool = False,
        norm: bool = False,
        stop_early: bool = False,
        output_fmt: str = "NCHW",
        intermediates_only: bool = False,
        output_dict: bool = False,
        patch_coord: Optional[torch.Tensor] = None,
        patch_valid: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        masks: Optional[Tensor | List[Tensor]] = None,
    ):
        """Forward features that returns intermediates.

        Args:
            x: Input image tensor
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            return_prefix_tokens: Return both prefix and spatial intermediate tokens
            norm: Apply norm layer to all intermediates
            stop_early: Stop iterating over blocks when last desired intermediate hit
            output_fmt: Shape of intermediate feature outputs
            intermediates_only: Only return intermediate features
            output_dict: Return outputs as a dictionary with 'image_features' and 'image_intermediates' keys
            patch_coord: Optional patch coordinates [B, N, 2] for NaFlex mode
            patch_valid: Optional patch type indicators (1=patch, 0=padding) for NaFlex
            attn_mask: Optional attention mask for masked attention
        Returns:
            A tuple with (final_features, intermediates), a list of intermediate features, or a dictionary containing
            'image_features' and 'image_intermediates' (and optionally 'image_intermediates_prefix')
        """

        # Prepare JEPA masks
        if masks is not None:
            raise ValueError(f"Input masks are not supported for getting the intermidate features")
            masks = None

        return super().forward_intermediates(
            x,
            indices,
            return_prefix_tokens,
            norm,
            stop_early,
            output_fmt,
            intermediates_only,
            output_dict,
            patch_coord,
            patch_valid,
            attn_mask,
        )


class MAENaFlexViT(Transformer):
    def __init__(self): ...


class DINONaFlexVit(Transformer): ...


def __test_jepa_naflex():
    from src.stage1.self_supervised.ijepa.src.models.vision_transformer import (
        VisionTransformerPredictor,
        apply_masks,
        repeat_interleave_batch,
        vit_predictor,
    )
    from src.stage1.self_supervised.jepa_blockutils import MaskCollator

    x = torch.randn(2, 3, 224, 224)
    x = [xi for xi in x]
    collator = MaskCollator(
        patch_size=16,
        npred=4,
        nenc=1,
        min_keep=10,
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.15, 0.2),
    )
    x, mask_enc, mask_pred = collator(x)

    x = x.to("cuda", torch.bfloat16)
    mask_enc = [m.to("cuda", torch.int32) for m in mask_enc]
    mask_pred = [m.to("cuda", torch.int32) for m in mask_pred]

    # model
    cfg = NaFlexVitCfg(
        patch_size=16,
        embed_dim=768,
        depth=8,
        num_heads=8,
        pos_embed="learned",
        rope_type="axial",
        in_chans=3,
        out_chans=768,
        unpatch_size=1,
        reg_tokens=4,
    )
    model = IJEPANaFlexViT(cfg).to("cuda", torch.bfloat16)

    # predictor
    predictor = vit_predictor(
        num_patches=(224 // 16) ** 2,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=4,
        num_heads=8,
    ).to("cuda", torch.bfloat16)

    # Forward
    with torch.autocast("cuda", torch.bfloat16):
        # Target
        with torch.no_grad():
            h = model._forward_pretrained_backbone(x)[0]
            h = torch.layer_norm(h, (h.size(-1),))
            B = len(h)
            h = apply_masks(h, mask_pred)
            h_tgt = repeat_interleave_batch(h, B, repeat=len(mask_enc))
            print(h_tgt.shape)

        # Context
        h_ctx = model._forward_pretrained_backbone(x, jepa_masks=mask_enc)[0]
        print(h_ctx.shape)
        h_pred = predictor(h_ctx, mask_enc, mask_pred)
        print(h_pred.shape)

        # Loss
        loss = torch.nn.functional.smooth_l1_loss(h_pred, h_tgt)
        print(f"loss: {loss}")


if __name__ == "__main__":
    """
    python -m src.stage1.cosmos.modules.naflex
    """
    __test_jepa_naflex()
