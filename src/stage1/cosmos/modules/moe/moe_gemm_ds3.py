from importlib import import_module
from typing import Any, Literal

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from .ops.grouped_gemm_swiglu import Gemm1SwiGLUFusionBackend, grouped_gemm1_swiglu_dispatch
from .ops.swiglu import SwiGLUBackend, swiglu_from_packed


gg: Any = None
try:
    gg = import_module("unsloth.kernels.moe.grouped_gemm.interface")
except Exception:
    gg = None

GroupedBackend = Literal["auto", "torch_grouped_mm", "torch__grouped_mm", "unsloth", "reference"]
ScoreFunction = Literal["sigmoid", "softmax"]


@dataclass(frozen=True)
class ExpertSlotLayout:
    num_ffn_experts: int
    num_zero_experts: int
    num_copy_experts: int
    num_constant_experts: int

    @property
    def total_num_experts(self) -> int:
        return self.num_ffn_experts + self.num_zero_experts + self.num_copy_experts + self.num_constant_experts

    @property
    def zero_start(self) -> int:
        return self.num_ffn_experts

    @property
    def copy_start(self) -> int:
        return self.zero_start + self.num_zero_experts

    @property
    def constant_start(self) -> int:
        return self.copy_start + self.num_copy_experts

    def is_ffn(self, expert_ids: torch.Tensor) -> torch.Tensor:
        return expert_ids < self.zero_start

    def is_copy(self, expert_ids: torch.Tensor) -> torch.Tensor:
        return (expert_ids >= self.copy_start) & (expert_ids < self.constant_start)

    def is_constant(self, expert_ids: torch.Tensor) -> torch.Tensor:
        return expert_ids >= self.constant_start

    def to_constant_local_index(self, expert_ids: torch.Tensor) -> torch.Tensor:
        return expert_ids - self.constant_start


def _make_offsets_from_counts(counts: torch.Tensor, device: torch.device) -> torch.Tensor:
    counts = counts.to(dtype=torch.int32, device=device).contiguous()
    return torch.cumsum(counts, dim=0, dtype=torch.int32)


def _grouped_gemm_reference(x_sorted: torch.Tensor, w_expert: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    x_sorted = x_sorted.view(-1, x_sorted.shape[-1])
    out = torch.zeros((x_sorted.shape[0], w_expert.shape[1]), dtype=x_sorted.dtype, device=x_sorted.device)
    start = 0
    for expert_idx, m_size in enumerate(counts.to(dtype=torch.int64).tolist()):
        size = int(m_size)
        if size <= 0:
            continue
        end = start + size
        out[start:end] = x_sorted[start:end] @ w_expert[expert_idx].transpose(0, 1)
        start = end
    return out


def _grouped_gemm_torch_functional(
    x_sorted: torch.Tensor, w_expert: torch.Tensor, counts: torch.Tensor
) -> torch.Tensor:
    if not hasattr(torch.nn.functional, "grouped_mm"):
        raise RuntimeError("torch.nn.functional.grouped_mm is not available.")
    offs = _make_offsets_from_counts(counts, device=x_sorted.device)
    w = w_expert.transpose(-2, -1).contiguous()
    return torch.nn.functional.grouped_mm(x_sorted.to(w.dtype), w, offs=offs)


def _grouped_gemm_torch_private(x_sorted: torch.Tensor, w_expert: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    if not hasattr(torch, "_grouped_mm"):
        raise RuntimeError("torch._grouped_mm is not available.")
    offs = _make_offsets_from_counts(counts, device=x_sorted.device)
    w = w_expert.transpose(-2, -1).contiguous()
    return torch._grouped_mm(x_sorted.to(w.dtype), w, offs)


def _grouped_gemm_unsloth(x_sorted: torch.Tensor, w_expert: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    if gg is None:
        raise RuntimeError("Unsloth grouped GEMM interface is not available.")
    if x_sorted.device.type != "cuda":
        raise RuntimeError("Unsloth grouped GEMM requires CUDA tensors.")

    m_sizes = counts.to(dtype=torch.int32, device=x_sorted.device).contiguous()
    gather_indices = torch.arange(x_sorted.shape[0], device=x_sorted.device, dtype=torch.int32)
    if hasattr(gg, "grouped_gemm"):
        try:
            return gg.grouped_gemm(
                X=x_sorted,
                W=w_expert,
                m_sizes=m_sizes,
                topk=1,
                gather_indices=gather_indices,
                permute_x=False,
                permute_y=False,
                autotune=True,
                is_first_gemm=True,
            )
        except TypeError:
            offs = torch.cumsum(m_sizes, dim=0, dtype=torch.int32)
            return gg.grouped_gemm(x_sorted, w_expert, offs=offs)
    if hasattr(gg, "grouped_mm"):
        offs = torch.cumsum(m_sizes, dim=0, dtype=torch.int32)
        return gg.grouped_mm(x_sorted, w_expert, offs=offs)

    candidates = [name for name in dir(gg) if "GEMM" in name or "Gemm" in name]
    if not candidates:
        raise RuntimeError("Cannot find grouped GEMM entry in Unsloth interface.")
    fn = getattr(gg, candidates[0])
    offs = torch.cumsum(m_sizes, dim=0, dtype=torch.int32)
    return fn.apply(x_sorted, w_expert, offs)


def grouped_gemm_dispatch(
    x_sorted: torch.Tensor,
    w_expert: torch.Tensor,
    counts: torch.Tensor,
    backend: GroupedBackend = "auto",
) -> torch.Tensor:
    if x_sorted.numel() == 0:
        return x_sorted.new_zeros((x_sorted.shape[0], w_expert.shape[1]))

    if backend == "reference":
        return _grouped_gemm_reference(x_sorted, w_expert, counts)
    if backend == "torch_grouped_mm":
        return _grouped_gemm_torch_functional(x_sorted, w_expert, counts)
    if backend == "torch__grouped_mm":
        return _grouped_gemm_torch_private(x_sorted, w_expert, counts)
    if backend == "unsloth":
        return _grouped_gemm_unsloth(x_sorted, w_expert, counts)

    errors: list[str] = []
    for impl in (_grouped_gemm_torch_functional, _grouped_gemm_torch_private, _grouped_gemm_unsloth):
        try:
            return impl(x_sorted, w_expert, counts)
        except Exception as exc:
            errors.append(f"{impl.__name__}: {exc}")
    try:
        return _grouped_gemm_reference(x_sorted, w_expert, counts)
    except Exception as exc:
        errors.append(f"_grouped_gemm_reference: {exc}")
        raise RuntimeError("No grouped GEMM backend is usable.\n" + "\n".join(errors)) from exc


class AuxLossAutoScaler(torch.autograd.Function):
    main_loss_backward_scale: torch.Tensor | None = None

    @staticmethod
    def forward(ctx, output: torch.Tensor, aux_loss: torch.Tensor):
        ctx.save_for_backward(aux_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (aux_loss,) = ctx.saved_tensors
        if AuxLossAutoScaler.main_loss_backward_scale is None:
            AuxLossAutoScaler.main_loss_backward_scale = torch.tensor(1.0, device=aux_loss.device)
        scale = AuxLossAutoScaler.main_loss_backward_scale
        return grad_output, torch.ones_like(aux_loss) * scale

    @staticmethod
    def set_loss_scale(scale: torch.Tensor) -> None:
        if AuxLossAutoScaler.main_loss_backward_scale is None:
            AuxLossAutoScaler.main_loss_backward_scale = scale
        else:
            AuxLossAutoScaler.main_loss_backward_scale.copy_(scale)


def router_z_loss(router_logits: torch.Tensor) -> torch.Tensor:
    logsum = torch.logsumexp(router_logits, dim=-1)
    return (logsum**2).mean()


def sequence_balance_loss(
    scores: torch.Tensor,
    topk_idx: torch.Tensor,
    *,
    num_experts: int,
    top_k: int,
    batch_size: int,
    seq_len: int,
    alpha: float,
) -> torch.Tensor:
    if alpha == 0.0:
        return scores.new_zeros(())

    scores_seq = scores.view(batch_size, seq_len, num_experts)
    topk_idx_seq = topk_idx.view(batch_size, seq_len, top_k)
    indicator = F.one_hot(topk_idx_seq, num_classes=num_experts).sum(dim=2).to(dtype=scores_seq.dtype)
    fi = indicator.mean(dim=1) * (num_experts / max(top_k, 1))
    normalized_scores = scores_seq / scores_seq.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    pi = normalized_scores.mean(dim=1)
    return alpha * (fi * pi).sum(dim=-1).mean()


class DeepSeekV3MoEGEMM(nn.Module):
    """DeepSeek-V3 style routed MoE FFN with grouped GEMM backends.

    Expected input/output shapes:
    - 3D input: ``x`` is ``[batch, seq, hidden]`` and output is ``[batch, seq, hidden]``.
    - 2D input: ``x`` is ``[tokens, hidden]`` and output is ``[tokens, hidden]``.

    Quick example: Attention + MoE block
    ------------------------------------
    ```python
    import torch
    import torch.nn as nn

    from .moe_gemm_ds3 import DeepSeekV3MoEGEMM


    class ToyAttentionMoEBlock(nn.Module):
        def __init__(self, hidden_size: int, num_heads: int) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(hidden_size)
            self.attn = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                dropout=0.0,
                batch_first=True,
            )
            self.norm2 = nn.LayerNorm(hidden_size)
            self.moe = DeepSeekV3MoEGEMM(
                hidden_size=hidden_size,
                intermediate_size=4 * hidden_size,
                num_experts=16,
                top_k=4,
                n_group=4,
                topk_group=2,
                grouped_backend="auto",
                swiglu_backend="torch",
                gemm1_swiglu_fusion_backend="off",
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: [B, S, H]
            h = self.norm1(x)
            attn_out, _ = self.attn(h, h, h, need_weights=False)  # [B, S, H]
            x = x + attn_out

            h = self.norm2(x)
            moe_out = self.moe(h)  # [B, S, H]
            x = x + moe_out
            return x


    B, S, H = 2, 128, 512
    x = torch.randn(B, S, H, device="cuda", dtype=torch.float16)
    block = ToyAttentionMoEBlock(hidden_size=H, num_heads=8).cuda().half()
    y = block(x)
    print(x.shape, y.shape)  # torch.Size([2, 128, 512]) torch.Size([2, 128, 512])
    ```

    Optional router logging:
    ```python
    y, router_logits, aux = block.moe(
        x,
        return_router_logits=True,
        return_aux_components=True,
    )
    # router_logits: [B*S, total_num_experts]
    # aux keys: {"z_loss", "seq_aux_loss", "aux_loss"}
    ```
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        num_shared_experts: int | None = None,
        num_zero_experts: int = 0,
        num_copy_experts: int = 0,
        num_constant_experts: int = 0,
        top_k: int = 8,
        n_group: int = 8,
        topk_group: int = 4,
        routed_scaling_factor: float = 1.0,
        norm_topk_prob: bool = True,
        score_function: ScoreFunction = "sigmoid",
        enable_expert_bias: bool = True,
        expert_bias_update_rate: float = 1e-3,
        update_expert_bias_on_forward: bool = True,
        bias_update_interval: int = 1,
        use_seq_aux_loss: bool = True,
        seq_aux_loss_coef: float = 1e-4,
        z_loss_coef: float = 0.0,
        enable_autoscale_aux: bool = True,
        grouped_backend: GroupedBackend = "auto",
        swiglu_backend: SwiGLUBackend = "torch",
        gemm1_swiglu_fusion_backend: Gemm1SwiGLUFusionBackend = "off",
    ):
        super().__init__()
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if num_experts <= 0:
            raise ValueError("num_experts must be positive.")
        if num_shared_experts is not None and num_shared_experts < 0:
            raise ValueError("n_shared_experts must be non-negative.")
        if num_zero_experts < 0:
            raise ValueError("num_zero_experts must be non-negative.")
        if num_copy_experts < 0:
            raise ValueError("num_copy_experts must be non-negative.")
        if num_constant_experts < 0:
            raise ValueError("num_constant_experts must be non-negative.")
        self.expert_slot_layout = ExpertSlotLayout(
            num_ffn_experts=num_experts,
            num_zero_experts=num_zero_experts,
            num_copy_experts=num_copy_experts,
            num_constant_experts=num_constant_experts,
        )
        total_num_experts = self.expert_slot_layout.total_num_experts
        if n_group <= 0 or total_num_experts % n_group != 0:
            raise ValueError("n_group must be positive and divide total_num_experts.")
        if topk_group <= 0 or topk_group > n_group:
            raise ValueError("topk_group must be in [1, n_group].")
        experts_per_group = total_num_experts // n_group
        if top_k > topk_group * experts_per_group:
            raise ValueError("top_k cannot exceed topk_group * experts_per_group.")
        if bias_update_interval <= 0:
            raise ValueError("bias_update_interval must be positive.")
        if gemm1_swiglu_fusion_backend not in {"off", "expert_loop", "triton", "auto"}:
            raise ValueError(f"Unsupported gemm1_swiglu_fusion_backend: {gemm1_swiglu_fusion_backend}")

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts if num_shared_experts not in {None, 0} else None
        self.num_zero_experts = num_zero_experts
        self.num_copy_experts = num_copy_experts
        self.num_constant_experts = num_constant_experts
        self.total_num_experts = total_num_experts
        self.top_k = top_k
        self.n_group = n_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.score_function = score_function
        self.grouped_backend = grouped_backend
        self.swiglu_backend = swiglu_backend
        self.gemm1_swiglu_fusion_backend = gemm1_swiglu_fusion_backend

        self.enable_expert_bias = enable_expert_bias
        self.expert_bias_update_rate = expert_bias_update_rate
        self.update_expert_bias_on_forward = update_expert_bias_on_forward
        self.bias_update_interval = bias_update_interval

        self.use_seq_aux_loss = use_seq_aux_loss
        self.seq_aux_loss_coef = seq_aux_loss_coef
        self.z_loss_coef = z_loss_coef
        self.enable_autoscale_aux = enable_autoscale_aux

        self.router = nn.Linear(hidden_size, self.total_num_experts, bias=False)
        self.w1 = nn.Parameter(torch.empty(num_experts, 2 * intermediate_size, hidden_size))
        self.w2 = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))
        nn.init.normal_(self.w1, std=0.02)
        nn.init.normal_(self.w2, std=0.02)
        self.constant_expert_vectors: nn.Parameter | None = None
        if self.num_constant_experts > 0:
            self.constant_expert_vectors = nn.Parameter(torch.empty(self.num_constant_experts, hidden_size))
            nn.init.normal_(self.constant_expert_vectors, std=0.02)

        self.shared_intermediate_size: int | None = None
        self.shared_w1: nn.Parameter | None = None
        self.shared_w2: nn.Parameter | None = None
        if self.num_shared_experts is not None:
            self.shared_intermediate_size = intermediate_size * self.num_shared_experts
            self.shared_w1 = nn.Parameter(torch.empty(2 * self.shared_intermediate_size, hidden_size))
            self.shared_w2 = nn.Parameter(torch.empty(hidden_size, self.shared_intermediate_size))
            nn.init.normal_(self.shared_w1, std=0.02)
            nn.init.normal_(self.shared_w2, std=0.02)

        if enable_expert_bias:
            self.register_buffer(
                "local_tokens_per_expert", torch.zeros(self.total_num_experts, dtype=torch.float32), persistent=False
            )
            self.register_buffer("expert_bias", torch.zeros(self.total_num_experts, dtype=torch.float32))
            self.register_buffer("_bias_update_step", torch.zeros((), dtype=torch.int64), persistent=False)
        else:
            self.local_tokens_per_expert = None
            self.expert_bias = None
            self._bias_update_step = None

    def set_expert_bias_update_rate(self, rate: float) -> None:
        self.expert_bias_update_rate = float(rate)

    def reset_expert_bias_stats(self) -> None:
        if self.local_tokens_per_expert is not None:
            self.local_tokens_per_expert.zero_()
        if self._bias_update_step is not None:
            self._bias_update_step.zero_()

    def update_expert_bias(
        self,
        *,
        all_reduce: bool = False,
        process_group: dist.ProcessGroup | None = None,
        reset_tokens: bool = True,
    ) -> None:
        if not self.enable_expert_bias or self.local_tokens_per_expert is None or self.expert_bias is None:
            return
        with torch.no_grad():
            tokens_per_expert = self.local_tokens_per_expert
            if all_reduce:
                if not dist.is_available() or not dist.is_initialized():
                    raise RuntimeError("all_reduce=True requires initialized torch.distributed.")
                tokens_per_expert = tokens_per_expert.clone()
                dist.all_reduce(tokens_per_expert, group=process_group)
            offset = tokens_per_expert.mean() - tokens_per_expert
            self.expert_bias.add_(torch.sign(offset) * self.expert_bias_update_rate)
            self.expert_bias.data = self.expert_bias.data.to(torch.float32)
            if reset_tokens:
                self.local_tokens_per_expert.zero_()

    def route_tokens_to_experts(
        self, router_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.score_function == "sigmoid":
            scores = torch.sigmoid(router_logits.float()).to(dtype=router_logits.dtype)
        elif self.score_function == "softmax":
            scores = torch.softmax(router_logits, dim=-1, dtype=torch.float32).to(dtype=router_logits.dtype)
        else:
            raise ValueError(f"Unsupported score_function: {self.score_function}")

        if self.expert_bias is not None:
            route_scores = scores + self.expert_bias.to(dtype=scores.dtype, device=scores.device)
        else:
            route_scores = scores

        num_tokens = route_scores.shape[0]
        experts_per_group = self.total_num_experts // self.n_group
        top2 = min(2, experts_per_group)
        group_scores = (
            route_scores.view(num_tokens, self.n_group, experts_per_group).topk(top2, dim=-1).values.sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False).indices
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1.0)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, experts_per_group)
            .reshape(num_tokens, self.total_num_experts)
            .to(torch.bool)
        )
        scores_for_choice = route_scores.masked_fill(~score_mask, 0.0)

        topk_idx = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False).indices
        topk_weights = scores.gather(1, topk_idx)
        if self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_idx, topk_weights, scores, scores_for_choice

    def _run_gemm1_and_swiglu(self, x_sorted: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        if self.gemm1_swiglu_fusion_backend != "off":
            return grouped_gemm1_swiglu_dispatch(
                x_sorted,
                self.w1,
                counts,
                backend=self.gemm1_swiglu_fusion_backend,
            )
        y1 = grouped_gemm_dispatch(x_sorted, self.w1, counts, backend=self.grouped_backend)
        return swiglu_from_packed(y1, self.intermediate_size, backend=self.swiglu_backend)

    def _run_ffn_experts(self, x_ffn: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
        if x_ffn.numel() == 0:
            return x_ffn.new_zeros((0, self.hidden_size))

        order = torch.argsort(expert_ids)
        x_sorted = x_ffn[order]
        expert_ids_sorted = expert_ids[order]
        counts = torch.bincount(expert_ids_sorted, minlength=self.num_experts).to(torch.int32)
        y_hidden = self._run_gemm1_and_swiglu(x_sorted, counts)
        y2 = grouped_gemm_dispatch(y_hidden, self.w2, counts, backend=self.grouped_backend)

        inv = torch.empty_like(order)
        inv[order] = torch.arange(order.numel(), device=order.device)
        return y2[inv]

    def _lookup_constant_expert_outputs(self, expert_ids: torch.Tensor) -> torch.Tensor:
        if self.constant_expert_vectors is None:
            raise RuntimeError("constant_expert_vectors is not initialized.")
        constant_local_idx = self.expert_slot_layout.to_constant_local_index(expert_ids)
        return self.constant_expert_vectors[constant_local_idx]

    def _dispatch_pair_outputs(self, x_rep: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
        pair_outputs = x_rep.new_zeros((x_rep.shape[0], self.hidden_size))

        ffn_mask = self.expert_slot_layout.is_ffn(expert_ids)
        if torch.any(ffn_mask):
            pair_outputs[ffn_mask] = self._run_ffn_experts(x_rep[ffn_mask], expert_ids[ffn_mask]).to(
                dtype=pair_outputs.dtype
            )

        copy_mask = self.expert_slot_layout.is_copy(expert_ids)
        if torch.any(copy_mask):
            pair_outputs[copy_mask] = x_rep[copy_mask]

        constant_mask = self.expert_slot_layout.is_constant(expert_ids)
        if torch.any(constant_mask):
            pair_outputs[constant_mask] = self._lookup_constant_expert_outputs(expert_ids[constant_mask]).to(
                dtype=pair_outputs.dtype
            )

        return pair_outputs

    def _run_shared_experts(self, x_flat: torch.Tensor) -> torch.Tensor | None:
        if self.shared_w1 is None or self.shared_w2 is None or self.shared_intermediate_size is None:
            return None
        y1 = F.linear(x_flat, self.shared_w1)
        y_hidden = swiglu_from_packed(y1, self.shared_intermediate_size, backend=self.swiglu_backend)
        return F.linear(y_hidden, self.shared_w2)

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_router_logits: bool = False,
        return_aux_components: bool = False,
    ):
        if x.dim() == 3:
            batch_size, seq_len, hidden_size = x.shape
            x_flat = x.reshape(-1, hidden_size)
        elif x.dim() == 2:
            batch_size = 1
            seq_len, hidden_size = x.shape
            x_flat = x
        else:
            raise ValueError(f"Expected x.ndim in {{2, 3}}, got shape={tuple(x.shape)}")

        total_tokens = x_flat.shape[0]
        router_logits = self.router(x_flat)
        topk_idx, topk_weights, scores, _ = self.route_tokens_to_experts(router_logits)

        x_rep = x_flat.repeat_interleave(self.top_k, dim=0)
        pair_expert_ids = topk_idx.reshape(-1)
        pair_weights = topk_weights.reshape(-1)
        counts = torch.bincount(pair_expert_ids, minlength=self.total_num_experts).to(torch.int32)
        pair_outputs = self._dispatch_pair_outputs(x_rep, pair_expert_ids)
        y = (pair_outputs * pair_weights.unsqueeze(-1)).view(total_tokens, self.top_k, hidden_size).sum(dim=1)

        shared_output = self._run_shared_experts(x_flat)
        if shared_output is not None:
            y = y + shared_output

        y = y.view_as(x)

        if (
            self.enable_expert_bias
            and self.local_tokens_per_expert is not None
            and self.training
            and torch.is_grad_enabled()
        ):
            with torch.no_grad():
                self.local_tokens_per_expert.add_(counts.to(dtype=self.local_tokens_per_expert.dtype))
                if self.update_expert_bias_on_forward and self._bias_update_step is not None:
                    self._bias_update_step.add_(1)
                    if int(self._bias_update_step.item()) % self.bias_update_interval == 0:
                        self.update_expert_bias(reset_tokens=True)

        z_loss_term = router_logits.new_zeros(())
        if self.z_loss_coef != 0.0:
            z_loss_term = router_z_loss(router_logits) * self.z_loss_coef

        seq_aux_loss_term = router_logits.new_zeros(())
        if self.use_seq_aux_loss and self.seq_aux_loss_coef != 0.0:
            seq_aux_loss_term = sequence_balance_loss(
                scores,
                topk_idx,
                num_experts=self.total_num_experts,
                top_k=self.top_k,
                batch_size=batch_size,
                seq_len=seq_len,
                alpha=self.seq_aux_loss_coef,
            )

        aux_loss = z_loss_term + seq_aux_loss_term
        if self.enable_autoscale_aux and (
            self.z_loss_coef != 0.0 or (self.use_seq_aux_loss and self.seq_aux_loss_coef != 0.0)
        ):
            y = AuxLossAutoScaler.apply(y, aux_loss)

        if return_router_logits and return_aux_components:
            return y, router_logits, {"z_loss": z_loss_term, "seq_aux_loss": seq_aux_loss_term, "aux_loss": aux_loss}
        if return_router_logits:
            return y, router_logits
        if return_aux_components:
            return y, {"z_loss": z_loss_term, "seq_aux_loss": seq_aux_loss_term, "aux_loss": aux_loss}
        return y


__all__ = [
    "AuxLossAutoScaler",
    "DeepSeekV3MoEGEMM",
    "grouped_gemm_dispatch",
    "router_z_loss",
    "sequence_balance_loss",
]
