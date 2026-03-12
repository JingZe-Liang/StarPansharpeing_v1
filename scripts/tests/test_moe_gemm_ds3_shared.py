import torch
import torch.nn.functional as F

from src.stage1.cosmos.modules.moe.moe_gemm_ds3 import DeepSeekV3MoEGEMM, grouped_gemm_dispatch
from src.stage1.cosmos.modules.moe.ops.swiglu import swiglu_from_packed
from src.utilities.func.call import extract_needed_kwargs


def _build_model(
    *,
    num_shared_experts: int | None,
    num_zero_experts: int = 0,
    num_copy_experts: int = 0,
    num_constant_experts: int = 0,
    n_group: int = 2,
    topk_group: int = 1,
) -> DeepSeekV3MoEGEMM:
    return DeepSeekV3MoEGEMM(
        hidden_size=16,
        intermediate_size=24,
        num_experts=4,
        num_shared_experts=num_shared_experts,
        num_zero_experts=num_zero_experts,
        num_copy_experts=num_copy_experts,
        num_constant_experts=num_constant_experts,
        top_k=2,
        n_group=n_group,
        topk_group=topk_group,
        routed_scaling_factor=1.0,
        norm_topk_prob=True,
        score_function="sigmoid",
        enable_expert_bias=False,
        use_seq_aux_loss=False,
        seq_aux_loss_coef=0.0,
        z_loss_coef=0.0,
        enable_autoscale_aux=False,
        grouped_backend="reference",
        swiglu_backend="torch",
        gemm1_swiglu_fusion_backend="off",
    )


def _reference_forward(model: DeepSeekV3MoEGEMM, x: torch.Tensor) -> torch.Tensor:
    x_flat = x.reshape(-1, x.shape[-1]) if x.dim() == 3 else x
    total_tokens, hidden_size = x_flat.shape

    router_logits = model.router(x_flat)
    topk_idx, topk_weights, _, _ = model.route_tokens_to_experts(router_logits)

    x_rep = x_flat.repeat_interleave(model.top_k, dim=0)
    pair_expert_ids = topk_idx.reshape(-1)
    pair_weights = topk_weights.reshape(-1)
    pair_outputs = x_flat.new_zeros((pair_expert_ids.numel(), hidden_size))

    ffn_mask = pair_expert_ids < model.num_experts
    if torch.any(ffn_mask):
        ffn_expert_ids = pair_expert_ids[ffn_mask]
        order = torch.argsort(ffn_expert_ids)
        x_sorted = x_rep[ffn_mask][order]
        expert_ids_sorted = ffn_expert_ids[order]
        counts = torch.bincount(expert_ids_sorted, minlength=model.num_experts).to(torch.int32)
        y1 = grouped_gemm_dispatch(x_sorted, model.w1, counts, backend="reference")
        y_hidden = swiglu_from_packed(y1, model.intermediate_size, backend="torch")
        y2 = grouped_gemm_dispatch(y_hidden, model.w2, counts, backend="reference")

        inv = torch.empty_like(order)
        inv[order] = torch.arange(order.numel(), device=order.device)
        pair_outputs[ffn_mask] = y2[inv]

    copy_start = model.num_experts + model.num_zero_experts
    constant_start = copy_start + model.num_copy_experts
    copy_mask = (pair_expert_ids >= copy_start) & (pair_expert_ids < constant_start)
    if torch.any(copy_mask):
        pair_outputs[copy_mask] = x_rep[copy_mask]

    constant_mask = pair_expert_ids >= constant_start
    if torch.any(constant_mask):
        assert model.constant_expert_vectors is not None
        constant_idx = pair_expert_ids[constant_mask] - constant_start
        pair_outputs[constant_mask] = model.constant_expert_vectors[constant_idx]

    routed_output = (pair_outputs * pair_weights.unsqueeze(-1)).view(total_tokens, model.top_k, hidden_size).sum(dim=1)

    if model.shared_w1 is None or model.shared_w2 is None or model.shared_intermediate_size is None:
        return routed_output.view_as(x)

    shared_hidden = F.linear(x_flat, model.shared_w1)
    shared_hidden = swiglu_from_packed(shared_hidden, model.shared_intermediate_size, backend="torch")
    shared_output = F.linear(shared_hidden, model.shared_w2)
    return (routed_output + shared_output).view_as(x)


def test_extract_needed_kwargs_keeps_num_shared_experts() -> None:
    kwargs = {
        "hidden_size": 16,
        "intermediate_size": 24,
        "num_experts": 4,
        "num_shared_experts": 2,
        "unused_key": "ignored",
    }

    extracted = extract_needed_kwargs(kwargs, DeepSeekV3MoEGEMM)

    assert extracted["num_shared_experts"] == 2
    assert "unused_key" not in extracted


def test_shared_expert_forward_matches_reference() -> None:
    torch.manual_seed(0)
    model = _build_model(num_shared_experts=2)
    model.eval()
    x = torch.randn(3, 5, 16)

    y = model(x)
    y_ref = _reference_forward(model, x)

    assert torch.allclose(y, y_ref, atol=1e-5, rtol=1e-5)


def test_shared_expert_gradients_are_valid() -> None:
    torch.manual_seed(1)
    model = _build_model(num_shared_experts=2)
    model.train()
    x = torch.randn(2, 4, 16, requires_grad=True)

    y = model(x)
    loss = y.square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert model.shared_w1 is not None
    assert model.shared_w2 is not None
    assert model.shared_w1.grad is not None
    assert model.shared_w2.grad is not None
    assert torch.isfinite(model.shared_w1.grad).all()
    assert torch.isfinite(model.shared_w2.grad).all()


def test_shared_expert_forward_matches_reference_with_zero_compute_experts() -> None:
    torch.manual_seed(2)
    model = _build_model(
        num_shared_experts=2,
        num_zero_experts=1,
        num_copy_experts=1,
        num_constant_experts=1,
        n_group=1,
        topk_group=1,
    )
    model.eval()
    x = torch.randn(2, 3, 16)

    y = model(x)
    y_ref = _reference_forward(model, x)

    assert torch.allclose(y, y_ref, atol=1e-5, rtol=1e-5)
