import torch

from src.stage1.cosmos.modules.moe.moe_gemm_ds3 import DeepSeekV3MoEGEMM, grouped_gemm_dispatch
from src.stage1.cosmos.modules.moe.ops.swiglu import swiglu_from_packed
from src.utilities.func.call import extract_needed_kwargs


def _build_model(*, top_k: int = 1) -> DeepSeekV3MoEGEMM:
    return DeepSeekV3MoEGEMM(
        hidden_size=8,
        intermediate_size=12,
        num_experts=2,
        num_zero_experts=1,
        num_copy_experts=1,
        num_constant_experts=1,
        top_k=top_k,
        n_group=1,
        topk_group=1,
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


def _set_router_row_scales(model: DeepSeekV3MoEGEMM, row_scales: list[float]) -> None:
    assert len(row_scales) == model.total_num_experts
    with torch.no_grad():
        model.router.weight.zero_()
        for idx, scale in enumerate(row_scales):
            model.router.weight[idx].fill_(scale)


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

    return (
        (pair_outputs * pair_weights.unsqueeze(-1)).view(total_tokens, model.top_k, hidden_size).sum(dim=1).view_as(x)
    )


def test_mixed_zero_compute_routing_matches_reference() -> None:
    torch.manual_seed(0)
    model = _build_model(top_k=4)
    model.eval()
    with torch.no_grad():
        _set_router_row_scales(model, [1.5, -2.0, 1.2, 0.9, 0.6])
        assert model.constant_expert_vectors is not None
        model.constant_expert_vectors[0].copy_(torch.linspace(-0.4, 0.3, steps=model.hidden_size))

    x = torch.ones(2, 3, model.hidden_size)
    y = model(x)
    y_ref = _reference_forward(model, x)

    assert torch.allclose(y, y_ref, atol=1e-5, rtol=1e-5)


def test_zero_compute_only_path_skips_ffn_experts(monkeypatch) -> None:
    model = _build_model(top_k=3)
    model.eval()
    with torch.no_grad():
        _set_router_row_scales(model, [-2.0, -3.0, 1.3, 1.1, 0.9])
        assert model.constant_expert_vectors is not None
        model.constant_expert_vectors[0].fill_(0.25)

    def _raise_if_called(x_ffn: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
        raise AssertionError(f"FFN experts should not run, got {x_ffn.shape} and {expert_ids.shape}.")

    monkeypatch.setattr(model, "_run_ffn_experts", _raise_if_called)
    x = torch.ones(2, 2, model.hidden_size)
    y = model(x)
    y_ref = _reference_forward(model, x)

    assert torch.allclose(y, y_ref, atol=1e-5, rtol=1e-5)


def test_copy_expert_behaves_like_identity() -> None:
    model = _build_model(top_k=1)
    model.train()
    with torch.no_grad():
        _set_router_row_scales(model, [-2.0, -3.0, -1.0, 1.4, -0.5])

    x = torch.ones(2, 3, model.hidden_size, requires_grad=True)
    y = model(x)

    assert torch.allclose(y, x, atol=1e-6, rtol=1e-6)

    y.sum().backward()

    assert x.grad is not None
    assert torch.allclose(x.grad, torch.ones_like(x.grad), atol=1e-6, rtol=1e-6)


def test_constant_expert_outputs_learnable_vector_and_gradients() -> None:
    model = _build_model(top_k=1)
    model.train()
    with torch.no_grad():
        _set_router_row_scales(model, [-2.0, -3.0, -1.0, -0.5, 1.6])
        assert model.constant_expert_vectors is not None
        model.constant_expert_vectors[0].copy_(torch.linspace(-0.2, 0.5, steps=model.hidden_size))

    x = torch.ones(2, 3, model.hidden_size, requires_grad=True)
    y = model(x)
    expected = model.constant_expert_vectors[0].detach().view(1, 1, -1).expand_as(y)

    assert torch.allclose(y, expected, atol=1e-6, rtol=1e-6)

    loss = y.square().mean()
    loss.backward()

    assert model.constant_expert_vectors is not None
    assert model.constant_expert_vectors.grad is not None
    assert torch.isfinite(model.constant_expert_vectors.grad).all()
    assert torch.count_nonzero(model.constant_expert_vectors.grad).item() > 0


def test_zero_expert_outputs_zero() -> None:
    model = _build_model(top_k=1)
    model.eval()
    with torch.no_grad():
        _set_router_row_scales(model, [-2.0, -3.0, 1.7, -0.5, -1.0])

    x = torch.ones(2, 3, model.hidden_size)
    y = model(x)

    assert torch.allclose(y, torch.zeros_like(y), atol=1e-6, rtol=1e-6)


def test_extract_needed_kwargs_keeps_zero_compute_expert_args() -> None:
    kwargs = {
        "hidden_size": 8,
        "intermediate_size": 12,
        "num_experts": 2,
        "num_zero_experts": 1,
        "num_copy_experts": 2,
        "num_constant_experts": 3,
        "unused_key": "ignored",
    }

    extracted = extract_needed_kwargs(kwargs, DeepSeekV3MoEGEMM)

    assert extracted["num_zero_experts"] == 1
    assert extracted["num_copy_experts"] == 2
    assert extracted["num_constant_experts"] == 3
    assert "unused_key" not in extracted
