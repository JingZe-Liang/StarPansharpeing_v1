from importlib import import_module
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


gg: Any = None
try:
    gg = import_module("unsloth.kernels.moe.grouped_gemm.interface")
except Exception:
    gg = None


def _torch_grouped_gemm_reference(
    x_sorted: torch.Tensor, w_expert: torch.Tensor, m_sizes: torch.Tensor
) -> torch.Tensor:
    x_sorted = x_sorted.view(-1, x_sorted.shape[-1])
    result = torch.zeros(
        (x_sorted.shape[0], w_expert.shape[1]),
        dtype=x_sorted.dtype,
        device=x_sorted.device,
    )
    start = 0
    for expert_idx, m_size in enumerate(m_sizes.tolist()):
        size = int(m_size)
        if size <= 0:
            continue
        end = start + size
        result[start:end] = x_sorted[start:end] @ w_expert[expert_idx].transpose(0, 1)
        start = end
    return result


def _call_grouped_gemm(x_sorted: torch.Tensor, w_expert: torch.Tensor, m_sizes: torch.Tensor) -> torch.Tensor:
    m_sizes = m_sizes.to(torch.int32).contiguous()
    if gg is None or x_sorted.device.type != "cuda":
        return _torch_grouped_gemm_reference(x_sorted, w_expert, m_sizes)

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
            offs = _make_offs_from_counts(m_sizes)
            return gg.grouped_gemm(x_sorted, w_expert, offs=offs)

    if hasattr(gg, "grouped_mm"):
        offs = _make_offs_from_counts(m_sizes)
        return gg.grouped_mm(x_sorted, w_expert, offs=offs)

    candidates = [name for name in dir(gg) if "GEMM" in name or "Gemm" in name]
    if not candidates:
        raise RuntimeError("Cannot find grouped GEMM entry in unsloth interface")
    fn = getattr(gg, candidates[0])
    offs = _make_offs_from_counts(m_sizes)
    return fn.apply(x_sorted, w_expert, offs)


def _make_offs_from_counts(counts: torch.Tensor) -> torch.Tensor:
    return torch.cumsum(counts, dim=0).to(torch.int32)


class MoEAuxLossAutoScaler(torch.autograd.Function):
    main_loss_backward_scale: torch.Tensor | None = None

    @staticmethod
    def forward(ctx, output: torch.Tensor, aux_loss: torch.Tensor):
        ctx.save_for_backward(aux_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (aux_loss,) = ctx.saved_tensors
        if MoEAuxLossAutoScaler.main_loss_backward_scale is None:
            MoEAuxLossAutoScaler.main_loss_backward_scale = torch.tensor(1.0, device=aux_loss.device)
        scale = MoEAuxLossAutoScaler.main_loss_backward_scale
        return grad_output, torch.ones_like(aux_loss) * scale

    @staticmethod
    def set_loss_scale(scale: torch.Tensor):
        if MoEAuxLossAutoScaler.main_loss_backward_scale is None:
            MoEAuxLossAutoScaler.main_loss_backward_scale = scale
        else:
            MoEAuxLossAutoScaler.main_loss_backward_scale.copy_(scale)


def router_z_loss(router_logits: torch.Tensor) -> torch.Tensor:
    z = torch.logsumexp(router_logits, dim=-1)  # (T,)
    return (z**2).mean()


def _topk_routing_with_optional_bias(
    router_logits: torch.Tensor,
    top_k: int,
    score_function: Literal["softmax", "sigmoid"],
    norm_topk_prob: bool,
    expert_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    bias = None
    if expert_bias is not None:
        bias = expert_bias.to(dtype=router_logits.dtype, device=router_logits.device)

    if score_function == "softmax":
        route_scores = router_logits if bias is None else router_logits + bias
        _, topk_idx = torch.topk(route_scores, top_k, dim=-1)
        topk_vals = torch.gather(router_logits, dim=-1, index=topk_idx)
        topk_w = F.softmax(topk_vals, dim=-1)
    elif score_function == "sigmoid":
        sigmoid_scores = torch.sigmoid(router_logits.float()).to(router_logits.dtype)
        if bias is None:
            topk_vals, topk_idx = torch.topk(sigmoid_scores, top_k, dim=-1)
        else:
            route_scores = sigmoid_scores + bias
            _, topk_idx = torch.topk(route_scores, top_k, dim=-1)
            topk_vals = torch.gather(sigmoid_scores, dim=-1, index=topk_idx)
        topk_w = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-20)
    else:
        raise ValueError(f"Unsupported score_function: {score_function}")

    if norm_topk_prob:
        topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-9)
    return topk_idx, topk_w


class MoEGEMMFused(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int = 2,
        score_function: Literal["softmax", "sigmoid"] = "softmax",
        norm_topk_prob: bool = False,
        aux_loss_coef: float = 0.0,
        z_loss_coef: float = 0.0,
        enable_autoscale_aux: bool = True,
        enable_expert_bias: bool = False,
        expert_bias_update_rate: float = 1e-3,
        update_expert_bias_on_forward: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.score_function = score_function
        self.norm_topk_prob = norm_topk_prob

        self.enable_expert_bias = enable_expert_bias
        self.expert_bias_update_rate = expert_bias_update_rate
        self.update_expert_bias_on_forward = update_expert_bias_on_forward
        self.aux_loss_coef = 0.0 if enable_expert_bias else aux_loss_coef
        self.z_loss_coef = 0.0 if enable_expert_bias else z_loss_coef
        self.enable_autoscale_aux = enable_autoscale_aux

        self.router = nn.Linear(hidden_size, num_experts, bias=False)

        self.w1 = nn.Parameter(torch.empty(num_experts, 2 * intermediate_size, hidden_size))
        self.w2 = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))
        nn.init.normal_(self.w1, std=0.02)
        nn.init.normal_(self.w2, std=0.02)

        if self.enable_expert_bias:
            self.register_buffer(
                "local_tokens_per_expert",
                torch.zeros(num_experts, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "expert_bias",
                torch.zeros(num_experts, dtype=torch.float32),
            )
        else:
            self.local_tokens_per_expert = None
            self.expert_bias = None

    def reset_expert_bias_stats(self) -> None:
        if self.local_tokens_per_expert is not None:
            self.local_tokens_per_expert.zero_()

    def update_expert_bias(
        self,
        *,
        all_reduce: bool = False,
        process_group: torch.distributed.ProcessGroup | None = None,
        reset_tokens: bool = True,
    ) -> None:
        if not self.enable_expert_bias or self.local_tokens_per_expert is None or self.expert_bias is None:
            return
        with torch.no_grad():
            tokens_per_expert = self.local_tokens_per_expert
            if all_reduce:
                tokens_per_expert = tokens_per_expert.clone()
                torch.distributed.all_reduce(tokens_per_expert, group=process_group)
            average_tokens = tokens_per_expert.mean()
            offset = average_tokens - tokens_per_expert
            self.expert_bias.add_(torch.sign(offset) * self.expert_bias_update_rate)
            if reset_tokens:
                self.local_tokens_per_expert.zero_()

    def forward(self, x: torch.Tensor, return_router_logits: bool = False):
        orig_shape = x.shape
        if x.dim() == 3:
            x = x.reshape(-1, x.size(-1))
        T, H = x.shape
        E = self.num_experts
        K = self.top_k

        router_logits = self.router(x)  # (T,E)

        topk_idx, topk_w = _topk_routing_with_optional_bias(
            router_logits=router_logits,
            top_k=K,
            score_function=self.score_function,
            norm_topk_prob=self.norm_topk_prob,
            expert_bias=self.expert_bias,
        )

        # expand pairs
        x_rep = x.repeat_interleave(K, dim=0)  # (T*K,H)
        exp_id = topk_idx.reshape(-1)  # (T*K,)
        exp_w = topk_w.reshape(-1)  # (T*K,)

        # sort by expert
        order = torch.argsort(exp_id)
        x_sorted = x_rep[order]
        exp_id_sorted = exp_id[order]
        exp_w_sorted = exp_w[order]

        counts = torch.bincount(exp_id_sorted, minlength=E).to(torch.int32)

        # expert MLP: 2 GEMMs
        y1 = _call_grouped_gemm(x_sorted, self.w1, counts)  # (T*K,2I)
        y_gate, y_up = torch.split(y1, self.intermediate_size, dim=-1)
        y_act = F.silu(y_gate) * y_up  # (T*K,I)
        y2 = _call_grouped_gemm(y_act, self.w2, counts)  # (T*K,H)

        # unsort back to pairs
        inv = torch.empty_like(order)
        inv[order] = torch.arange(order.numel(), device=order.device)
        y_pairs = y2[inv]  # (T*K,H)
        w_pairs = exp_w_sorted[inv].unsqueeze(-1)  # (T*K,1)

        # weighted sum over top-k
        y = (y_pairs * w_pairs).view(T, K, H).sum(dim=1)  # (T,H)
        y = y.reshape(orig_shape)

        if self.enable_expert_bias and self.local_tokens_per_expert is not None:
            if self.training and torch.is_grad_enabled():
                with torch.no_grad():
                    self.local_tokens_per_expert.add_(counts.to(dtype=self.local_tokens_per_expert.dtype))
                if self.update_expert_bias_on_forward:
                    self.update_expert_bias(reset_tokens=True)

        # ----- aux losses -----
        aux_loss = torch.zeros((), device=router_logits.device, dtype=router_logits.dtype)

        # z-loss（你关心的）
        if self.z_loss_coef != 0.0:
            aux_loss = aux_loss + self.z_loss_coef * router_z_loss(router_logits)

        # （可选）load balancing loss：这里留接口位
        # 你可以按 Megatron/DeepSpeed 的定义，把它写成：
        # aux_loss += self.aux_loss_coef * load_balance_loss(...)
        # 注：load-balance 依赖 token->expert 分配统计（topk_idx/topk_w），很好加
        # if self.aux_loss_coef != 0.0:
        #     aux_loss = aux_loss + self.aux_loss_coef * my_load_balance_loss(topk_idx, topk_w, E)

        # 自动挂载 aux loss：不用外面手动 `loss += aux_loss`
        if self.enable_autoscale_aux and (aux_loss.requires_grad or aux_loss != 0):
            y = MoEAuxLossAutoScaler.apply(y, aux_loss)

        if return_router_logits:
            return y, router_logits
        return y


def _reference_forward(model, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    orig_shape = x.shape
    if x.dim() == 3:
        x = x.reshape(-1, x.size(-1))

    total_tokens, hidden_size = x.shape
    top_k = model.top_k
    router_logits = model.router(x)
    topk_idx, topk_w = _topk_routing_with_optional_bias(
        router_logits=router_logits,
        top_k=top_k,
        score_function=model.score_function,
        norm_topk_prob=model.norm_topk_prob,
        expert_bias=model.expert_bias,
    )

    x_rep = x.repeat_interleave(top_k, dim=0)
    exp_id = topk_idx.reshape(-1)
    exp_w = topk_w.reshape(-1)
    order = torch.argsort(exp_id)

    x_sorted = x_rep[order]
    exp_id_sorted = exp_id[order]
    exp_w_sorted = exp_w[order]

    counts = torch.bincount(exp_id_sorted, minlength=model.num_experts).to(torch.int32)
    y1 = _torch_grouped_gemm_reference(x_sorted, model.w1, counts)
    y_gate, y_up = torch.split(y1, model.intermediate_size, dim=-1)
    y_act = F.silu(y_gate) * y_up
    y2 = _torch_grouped_gemm_reference(y_act, model.w2, counts)

    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.numel(), device=order.device)
    y_pairs = y2[inv]
    w_pairs = exp_w_sorted[inv].unsqueeze(-1)
    y = (y_pairs * w_pairs).view(total_tokens, top_k, hidden_size).sum(dim=1)
    return y.reshape(orig_shape), router_logits


def test_call_grouped_gemm_uses_unsloth_m_sizes_signature() -> None:
    if not torch.cuda.is_available():
        import pytest

        pytest.skip("CUDA is required for unsloth grouped_gemm signature test.")

    class _FakeGroupedGemm:
        def __init__(self) -> None:
            self.kwargs: dict[str, object] | None = None

        def grouped_gemm(self, **kwargs):  # noqa: ANN003
            self.kwargs = kwargs
            x = kwargs["X"]
            w = kwargs["W"]
            return torch.zeros((x.shape[0], w.shape[1]), device=x.device, dtype=x.dtype)

    fake = _FakeGroupedGemm()
    old_gg = gg
    try:
        globals()["gg"] = fake
        x_sorted = torch.randn(6, 4, device="cuda")
        w_expert = torch.randn(3, 5, 4, device="cuda")
        counts = torch.tensor([2, 1, 3], device="cuda", dtype=torch.int32)
        _ = _call_grouped_gemm(x_sorted, w_expert, counts)
        assert fake.kwargs is not None
        m_sizes = fake.kwargs["m_sizes"]
        assert isinstance(m_sizes, torch.Tensor)
        assert torch.equal(m_sizes, counts)
        assert fake.kwargs["topk"] == 1
        assert fake.kwargs["autotune"] is True
    finally:
        globals()["gg"] = old_gg


def test_standalone_fused_moe_matches_reference() -> None:
    old_gg = gg
    try:
        globals()["gg"] = None
        torch.manual_seed(0)
        model = MoEGEMMFused(
            hidden_size=8,
            intermediate_size=6,
            num_experts=4,
            top_k=2,
            norm_topk_prob=False,
            z_loss_coef=0.0,
            enable_autoscale_aux=False,
        )
        x = torch.randn(2, 3, 8)
        y, router_logits = model(x, return_router_logits=True)
        y_ref, router_logits_ref = _reference_forward(model, x)
        assert torch.allclose(router_logits, router_logits_ref, atol=1e-6, rtol=1e-6)
        assert torch.allclose(y, y_ref, atol=1e-5, rtol=1e-5)
    finally:
        globals()["gg"] = old_gg


def test_expert_bias_routing_prefers_positive_bias() -> None:
    old_gg = gg
    try:
        globals()["gg"] = None
        model = MoEGEMMFused(
            hidden_size=4,
            intermediate_size=4,
            num_experts=3,
            top_k=1,
            score_function="sigmoid",
            enable_expert_bias=True,
            update_expert_bias_on_forward=False,
            enable_autoscale_aux=False,
        )
        model.train()
        with torch.no_grad():
            model.router.weight.zero_()
            assert model.expert_bias is not None
            model.expert_bias.copy_(torch.tensor([0.0, 0.2, -0.1], dtype=torch.float32))
            model.reset_expert_bias_stats()

        x = torch.randn(2, 5, 4)
        _ = model(x)
        assert model.local_tokens_per_expert is not None
        assert torch.equal(model.local_tokens_per_expert, torch.tensor([0.0, 10.0, 0.0], dtype=torch.float32))
    finally:
        globals()["gg"] = old_gg


def test_update_expert_bias_matches_sign_rule() -> None:
    model = MoEGEMMFused(
        hidden_size=4,
        intermediate_size=4,
        num_experts=2,
        top_k=1,
        enable_expert_bias=True,
        update_expert_bias_on_forward=False,
        enable_autoscale_aux=False,
    )
    with torch.no_grad():
        assert model.local_tokens_per_expert is not None
        assert model.expert_bias is not None
        model.local_tokens_per_expert.copy_(torch.tensor([10.0, 0.0], dtype=torch.float32))
        model.expert_bias.zero_()

    model.update_expert_bias(reset_tokens=True)
    assert model.expert_bias is not None
    assert model.local_tokens_per_expert is not None
    expected = torch.tensor([-model.expert_bias_update_rate, model.expert_bias_update_rate], dtype=torch.float32)
    assert torch.allclose(model.expert_bias, expected, atol=1e-7, rtol=0.0)
    assert torch.equal(model.local_tokens_per_expert, torch.zeros(2, dtype=torch.float32))
