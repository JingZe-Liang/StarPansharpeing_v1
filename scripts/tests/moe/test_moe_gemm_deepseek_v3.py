from __future__ import annotations

from typing import Literal

import torch

from src.stage1.cosmos.modules.moe_gemm_deepseek_v3 import DeepSeekV3MoEGEMM, grouped_gemm_dispatch


def _build_model(
    *,
    hidden_size: int = 8,
    intermediate_size: int = 6,
    num_experts: int = 4,
    top_k: int = 2,
    n_group: int = 2,
    topk_group: int = 1,
    routed_scaling_factor: float = 1.0,
    norm_topk_prob: bool = True,
    score_function: Literal["sigmoid", "softmax"] = "sigmoid",
    enable_expert_bias: bool = True,
    expert_bias_update_rate: float = 1e-3,
    update_expert_bias_on_forward: bool = True,
    bias_update_interval: int = 1,
    use_seq_aux_loss: bool = True,
    seq_aux_loss_coef: float = 1e-4,
    z_loss_coef: float = 0.0,
    enable_autoscale_aux: bool = True,
    grouped_backend: Literal["auto", "torch_grouped_mm", "torch__grouped_mm", "unsloth", "reference"] = "reference",
) -> DeepSeekV3MoEGEMM:
    return DeepSeekV3MoEGEMM(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        n_group=n_group,
        topk_group=topk_group,
        routed_scaling_factor=routed_scaling_factor,
        norm_topk_prob=norm_topk_prob,
        score_function=score_function,
        enable_expert_bias=enable_expert_bias,
        expert_bias_update_rate=expert_bias_update_rate,
        update_expert_bias_on_forward=update_expert_bias_on_forward,
        bias_update_interval=bias_update_interval,
        use_seq_aux_loss=use_seq_aux_loss,
        seq_aux_loss_coef=seq_aux_loss_coef,
        z_loss_coef=z_loss_coef,
        enable_autoscale_aux=enable_autoscale_aux,
        grouped_backend=grouped_backend,
    )


def test_forward_shape_and_dtype() -> None:
    model = _build_model()
    x3 = torch.randn(2, 3, 8, dtype=torch.float32)
    y3 = model(x3)
    assert y3.shape == x3.shape
    assert y3.dtype == x3.dtype

    x2 = torch.randn(5, 8, dtype=torch.float32)
    y2 = model(x2)
    assert y2.shape == x2.shape
    assert y2.dtype == x2.dtype


def test_group_limited_routing_masking() -> None:
    model = _build_model(
        num_experts=8, top_k=2, n_group=4, topk_group=1, norm_topk_prob=False, score_function="sigmoid"
    )
    logits = torch.tensor([[-3.0, -3.0, -2.0, -2.0, 3.0, 2.5, -1.0, -1.0]], dtype=torch.float32)
    topk_idx, _, _, _ = model.route_tokens_to_experts(logits)
    experts_per_group = model.num_experts // model.n_group
    group_ids = torch.unique(topk_idx // experts_per_group)
    assert group_ids.numel() == 1
    assert group_ids.item() == 2


def test_bias_affects_selection_not_weight_source() -> None:
    model = _build_model(
        num_experts=3,
        top_k=1,
        n_group=1,
        topk_group=1,
        norm_topk_prob=False,
        score_function="sigmoid",
        enable_expert_bias=True,
    )
    with torch.no_grad():
        assert model.expert_bias is not None
        model.expert_bias.copy_(torch.tensor([0.0, 1.0, -1.0], dtype=torch.float32))

    logits = torch.zeros(1, 3, dtype=torch.float32)
    topk_idx, topk_w, scores, _ = model.route_tokens_to_experts(logits)
    assert topk_idx.item() == 1
    assert torch.allclose(topk_w.reshape(-1), scores[:, 1], atol=1e-7, rtol=0.0)
    assert torch.allclose(topk_w.reshape(-1), torch.tensor([0.5], dtype=torch.float32), atol=1e-7, rtol=0.0)


def test_bias_update_sign_rule() -> None:
    model = _build_model(num_experts=2, top_k=1, enable_expert_bias=True, update_expert_bias_on_forward=False)
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


def test_forward_auto_bias_update() -> None:
    model = _build_model(
        num_experts=2,
        top_k=1,
        n_group=1,
        topk_group=1,
        enable_expert_bias=True,
        score_function="sigmoid",
        norm_topk_prob=False,
        expert_bias_update_rate=0.1,
        bias_update_interval=2,
        update_expert_bias_on_forward=True,
    )
    model.train()
    with torch.no_grad():
        assert model.router.weight is not None
        model.router.weight.zero_()
        assert model.expert_bias is not None
        model.expert_bias.copy_(torch.tensor([0.2, 0.0], dtype=torch.float32))

    x = torch.randn(1, 4, 8)
    _ = model(x)
    bias_after_first = model.expert_bias.detach().clone()
    _ = model(x)
    bias_after_second = model.expert_bias.detach().clone()

    assert torch.allclose(bias_after_first, torch.tensor([0.2, 0.0], dtype=torch.float32), atol=1e-7, rtol=0.0)
    assert torch.allclose(bias_after_second, torch.tensor([0.1, 0.1], dtype=torch.float32), atol=1e-7, rtol=0.0)
    assert model.local_tokens_per_expert is not None
    assert torch.equal(model.local_tokens_per_expert, torch.zeros_like(model.local_tokens_per_expert))


def test_z_loss_and_seq_aux_attach_grad() -> None:
    torch.manual_seed(7)
    model_no_aux = _build_model(
        enable_autoscale_aux=False,
        z_loss_coef=5.0,
        use_seq_aux_loss=True,
        seq_aux_loss_coef=0.5,
        top_k=1,
    )
    x = torch.randn(2, 3, 8)
    y = model_no_aux(x)
    loss = y.sum()
    loss.backward()
    grad_no_aux = model_no_aux.router.weight.grad.detach().clone()

    torch.manual_seed(7)
    model_with_aux = _build_model(
        enable_autoscale_aux=True,
        z_loss_coef=5.0,
        use_seq_aux_loss=True,
        seq_aux_loss_coef=0.5,
        top_k=1,
    )
    x2 = x.detach().clone()
    y2 = model_with_aux(x2)
    loss2 = y2.sum()
    loss2.backward()
    grad_with_aux = model_with_aux.router.weight.grad.detach().clone()

    diff = (grad_no_aux - grad_with_aux).abs().max().item()
    assert diff > 1e-6


def test_backend_fallback_reference_equivalence() -> None:
    x_sorted = torch.randn(12, 6, dtype=torch.float32)
    w_expert = torch.randn(4, 10, 6, dtype=torch.float32)
    counts = torch.tensor([3, 4, 0, 5], dtype=torch.int32)

    y_ref = grouped_gemm_dispatch(x_sorted, w_expert, counts, backend="reference")
    try:
        y_auto = grouped_gemm_dispatch(x_sorted, w_expert, counts, backend="auto")
    except RuntimeError:
        return
    assert torch.allclose(y_auto, y_ref, atol=1e-5, rtol=1e-5)


def test_set_bias_update_rate_schedule() -> None:
    model = _build_model(num_experts=2, top_k=1, enable_expert_bias=True, update_expert_bias_on_forward=False)
    with torch.no_grad():
        assert model.local_tokens_per_expert is not None
        assert model.expert_bias is not None
        model.expert_bias.zero_()
        model.local_tokens_per_expert.copy_(torch.tensor([10.0, 0.0], dtype=torch.float32))
    model.update_expert_bias(reset_tokens=True)
    bias_after_first = model.expert_bias.detach().clone()

    model.set_expert_bias_update_rate(0.0)
    with torch.no_grad():
        assert model.local_tokens_per_expert is not None
        model.local_tokens_per_expert.copy_(torch.tensor([10.0, 0.0], dtype=torch.float32))
    model.update_expert_bias(reset_tokens=True)
    assert torch.allclose(model.expert_bias, bias_after_first, atol=1e-7, rtol=0.0)
