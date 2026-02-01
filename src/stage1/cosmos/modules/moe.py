from __future__ import annotations

from typing import Any, Literal, cast

import torch
import torch.nn as nn
import torch.distributed as dist

try:
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.spec_utils import ModuleSpec
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
except ImportError:
    ProcessGroupCollection = None
    ModuleSpec = None
    TransformerConfig = None
    MoELayer = None
    MoESubmodules = None

from .variants.mlp import SwiGLU


# -----------------------------
# Literal-typed MoE config knobs
# -----------------------------
MoeTokenDispatcherType = Literal["allgather", "alltoall", "flex"]  # doc: allgather/alltoall/flex
MoeLoadBalancingType = Literal["aux_loss", "seq_aux_loss", "global_aux_loss", "sinkhorn", "none"]
MoeScoreFunction = Literal["softmax", "sigmoid"]


class SwiGLUExpert(SwiGLU):
    """Single expert implemented by your SwiGLU.

    IMPORTANT:
    - MoE experts are typically called with 2D input: [num_tokens, hidden].
    - Your xformers fused path asserts x.ndim == 3, so do NOT pass fused_type="xformers"
      unless you patch SwiGLU to support 2D -> 3D wrapping.
    """

    def __init__(self, hidden_size: int, ffn_hidden_size: int, ffn_drop: float, fused_type: str | None = None):
        super().__init__(
            in_features=hidden_size,
            hidden_features=ffn_hidden_size,
            out_features=hidden_size,
            norm_layer=None,
            bias=True,
            drop=ffn_drop,
            is_fused=fused_type,
        )


class MyExperts(nn.Module):
    """Custom experts module following Megatron MoE experts interface.

    It must implement:
      forward(permuted_local_hidden_states, tokens_per_expert, permuted_probs)
        -> (output, bias_or_none)

    Note:
    - TEGroupedMLP / SequentialMLP constructor signature is:
        (num_local_experts, config, submodules, pg_collection)
      We accept those args for compatibility and allow extra params via ModuleSpec.
      See Megatron's `experts` interface in `MoELayer`.
    """

    def __init__(
        self,
        num_local_experts: int,
        config: TransformerConfig,
        submodules: MoESubmodules | None = None,
        pg_collection: ProcessGroupCollection | None = None,
        *,
        ffn_drop: float = 0.0,
        fused_type: str | None = None,
    ):
        super().__init__()
        self.num_local_experts = num_local_experts
        self.config = config

        hidden = config.hidden_size
        ffn_hidden = config.ffn_hidden_size or hidden

        self.experts = nn.ModuleList(
            [
                SwiGLUExpert(hidden, ffn_hidden, ffn_drop=ffn_drop, fused_type=fused_type)
                for _ in range(num_local_experts)
            ]
        )

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,  # [num_local_tokens, hidden]
        tokens_per_expert: torch.Tensor,  # [num_local_experts]
        permuted_probs: torch.Tensor,  # usually [num_local_tokens] or [num_local_tokens, 1]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Normalize permuted_probs shape to 1D.
        if permuted_probs.ndim == 2 and permuted_probs.shape[-1] == 1:
            permuted_probs = permuted_probs.squeeze(-1)
        if permuted_probs.ndim != 1:
            raise ValueError(f"Expected permuted_probs to be 1D, got shape={tuple(permuted_probs.shape)}")

        # Split by expert. tokens_per_expert is small, so .tolist() overhead is acceptable.
        split_sizes = tokens_per_expert.to(dtype=torch.int64).tolist()
        chunks = permuted_local_hidden_states.split(split_sizes, dim=0)
        prob_chunks = permuted_probs.split(split_sizes, dim=0)

        out_chunks = []
        for expert_idx, (x_i, p_i) in enumerate(zip(chunks, prob_chunks)):
            if x_i.numel() == 0:
                out_chunks.append(x_i)
                continue

            # Apply routing probability either on input or output, controlled by config.
            if self.config.moe_apply_probs_on_input:
                x_i = x_i * p_i.to(dtype=x_i.dtype).unsqueeze(-1)
                y_i = self.experts[expert_idx](x_i)
            else:
                y_i = self.experts[expert_idx](x_i)
                y_i = y_i * p_i.to(dtype=y_i.dtype).unsqueeze(-1)

            out_chunks.append(y_i)

        return torch.cat(out_chunks, dim=0), None


class SharedExpertFFN(nn.Module):
    """A shared expert FFN (dense path) using SwiGLU.

    This is *not* the routed experts. It runs for all tokens.
    Megatron can schedule shared experts before/after router depending on config.
    """

    def __init__(self, hidden_size: int, shared_intermediate_size: int, ffn_drop: float, fused_type: str | None = None):
        super().__init__()
        self.ffn = SwiGLU(
            in_features=hidden_size,
            hidden_features=shared_intermediate_size,
            out_features=hidden_size,
            norm_layer=None,
            bias=True,
            drop=ffn_drop,
            is_fused=fused_type,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class _SingleProcessGroup:
    def size(self) -> int:
        return 1

    def rank(self) -> int:
        return 0


class MoEFFN(nn.Module):
    """MoE FFN wrapper that can be dropped into your Transformer block."""

    def __init__(
        self,
        *,
        dim: int,
        n_heads: int,
        mlp_ratio: float,
        num_layers: int,
        layer_idx: int,
        num_experts: int = 8,
        topk: int = 2,
        moe_token_dispatcher_type: MoeTokenDispatcherType = "alltoall",
        moe_router_load_balancing_type: MoeLoadBalancingType = "aux_loss",
        moe_aux_loss_coeff: float = 1e-2,
        moe_z_loss_coeff: float | None = None,
        score_function: MoeScoreFunction = "softmax",
        # DeepSeek / aux-loss-free knobs:
        aux_loss_free: bool = False,
        expert_bias_update_rate: float = 1e-3,
        # Shared expert knobs:
        shared_expert_intermediate_size: int | None = None,
        shared_expert_gate: bool = False,
        shared_expert_overlap: bool = False,
        # Expert impl knobs:
        ffn_drop: float = 0.0,
        fused_type: str | None = None,
        # Shapes / distributed:
        pg_collection: ProcessGroupCollection | None = None,
        assume_bsh: bool = True,
        **kwargs: Any,  # Allow passing other TransformerConfig args
    ):
        super().__init__()
        self.assume_bsh = assume_bsh

        # Megatron MoELayer expects a process group object with `.size()` even for world_size=1.
        # For single-process unit tests we allow `pg_collection.tp=None` and patch in a dummy group.
        if pg_collection is not None and getattr(pg_collection, "tp", None) is None:
            pg_collection.tp = cast(dist.ProcessGroup, _SingleProcessGroup())

        # Configure load balancing mode.
        if aux_loss_free:
            moe_router_load_balancing_type = "none"
        cfg = TransformerConfig(
            num_layers=num_layers,
            hidden_size=dim,
            num_attention_heads=n_heads,
            ffn_hidden_size=int(dim * mlp_ratio),
            # MoE:
            num_moe_experts=num_experts,
            moe_router_topk=topk,
            moe_token_dispatcher_type=moe_token_dispatcher_type,
            moe_router_load_balancing_type=moe_router_load_balancing_type,
            moe_aux_loss_coeff=0.0 if aux_loss_free else moe_aux_loss_coeff,
            moe_z_loss_coeff=moe_z_loss_coeff,
            moe_router_score_function=score_function,
            # Aux-loss-free expert bias routing:
            moe_router_enable_expert_bias=aux_loss_free,
            moe_router_bias_update_rate=expert_bias_update_rate,
            # Shared expert:
            moe_shared_expert_intermediate_size=shared_expert_intermediate_size,
            # moe_shared_expert_gate=shared_expert_gate,
            moe_shared_expert_overlap=shared_expert_overlap,
            **kwargs,
        )

        # Build experts via ModuleSpec (no placeholder, no post-hoc replacement).
        experts_spec = ModuleSpec(
            module=MyExperts,
            params={"ffn_drop": ffn_drop, "fused_type": fused_type},
        )
        submodules = MoESubmodules(experts=experts_spec)

        self.moe = MoELayer(config=cfg, submodules=submodules, layer_number=layer_idx, pg_collection=pg_collection)

        # Optional: an explicit shared FFN in your wrapper as well (only if you want),
        # separate from Megatron's internal shared expert scheduling.
        self.shared_ffn = None
        if shared_expert_intermediate_size is not None:
            self.shared_ffn = SharedExpertFFN(
                hidden_size=dim,
                shared_intermediate_size=shared_expert_intermediate_size,
                ffn_drop=ffn_drop,
                fused_type=None if fused_type == "xformers" else fused_type,  # keep it safe for 2D
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Your blocks commonly use [B,S,H]. Megatron MoE uses [S,B,H].
        if self.assume_bsh:
            x = x.transpose(0, 1).contiguous()  # [S,B,H]

        # Optional shared path (dense). If you only want Megatron-native shared experts,
        # you can remove this and rely on cfg.moe_shared_expert_* instead.
        if self.shared_ffn is not None:
            x = x + self.shared_ffn(x)

        out = self.moe(x)
        if isinstance(out, (tuple, list)):
            out, bias = out
            if bias is not None:
                out = out + bias

        if self.assume_bsh:
            out = out.transpose(0, 1).contiguous()  # [B,S,H]
        return out
