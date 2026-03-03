from __future__ import annotations

import time
from typing import Literal

import torch

from src.stage1.cosmos.modules.moe.moe_gemm_ds3 import DeepSeekV3MoEGEMM, grouped_gemm_dispatch
from src.stage1.cosmos.modules.moe.ops.grouped_gemm_swiglu import (
    TRITON_AVAILABLE as FUSED_TRITON_AVAILABLE,
    Gemm1SwiGLUFusionBackend,
)
from src.stage1.cosmos.modules.moe.ops.swiglu import swiglu_from_packed


def _build_model(
    *,
    grouped_backend: Literal["auto", "torch_grouped_mm", "torch__grouped_mm", "unsloth", "reference"] = "reference",
    swiglu_backend: Literal["auto", "triton", "torch"] = "torch",
    gemm1_swiglu_fusion_backend: Gemm1SwiGLUFusionBackend = "off",
) -> DeepSeekV3MoEGEMM:
    return DeepSeekV3MoEGEMM(
        hidden_size=32,
        intermediate_size=48,
        num_experts=8,
        top_k=2,
        n_group=4,
        topk_group=2,
        routed_scaling_factor=1.0,
        norm_topk_prob=True,
        score_function="sigmoid",
        enable_expert_bias=True,
        expert_bias_update_rate=1e-3,
        update_expert_bias_on_forward=False,
        bias_update_interval=1,
        use_seq_aux_loss=True,
        seq_aux_loss_coef=1e-4,
        z_loss_coef=1e-3,
        enable_autoscale_aux=False,
        grouped_backend=grouped_backend,
        swiglu_backend=swiglu_backend,
        gemm1_swiglu_fusion_backend=gemm1_swiglu_fusion_backend,
    )


def _reference_forward(model: DeepSeekV3MoEGEMM, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if x.dim() == 3:
        x_flat = x.reshape(-1, x.shape[-1])
    else:
        x_flat = x
    total_tokens, hidden = x_flat.shape

    router_logits = model.router(x_flat)
    topk_idx, topk_weights, _, _ = model.route_tokens_to_experts(router_logits)
    x_rep = x_flat.repeat_interleave(model.top_k, dim=0)
    exp_id = topk_idx.reshape(-1)
    exp_weight = topk_weights.reshape(-1)
    order = torch.argsort(exp_id)

    x_sorted = x_rep[order]
    exp_id_sorted = exp_id[order]
    exp_weight_sorted = exp_weight[order]
    counts = torch.bincount(exp_id_sorted, minlength=model.num_experts).to(torch.int32)

    y1 = grouped_gemm_dispatch(x_sorted, model.w1, counts, backend="reference")
    y_hidden = swiglu_from_packed(y1, model.intermediate_size, backend="torch")
    y2 = grouped_gemm_dispatch(y_hidden, model.w2, counts, backend="reference")

    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.numel(), device=order.device)
    y_pairs = y2[inv]
    pair_weights = exp_weight_sorted[inv].unsqueeze(-1)
    y = (y_pairs * pair_weights).view(total_tokens, model.top_k, hidden).sum(dim=1)
    return y.view_as(x), router_logits


def _measure_model_runtime_and_memory(
    model: DeepSeekV3MoEGEMM,
    x: torch.Tensor,
    *,
    warmup: int = 5,
    iters: int = 20,
) -> dict[str, float]:
    if x.device.type != "cuda":
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model(x)
        t1 = time.perf_counter()
        return {"time_ms": ((t1 - t0) / iters) * 1000.0, "peak_alloc_mb": 0.0, "peak_reserved_mb": 0.0}

    torch.cuda.reset_peak_memory_stats(device=x.device)
    torch.cuda.synchronize(device=x.device)
    for _ in range(warmup):
        _ = model(x)
    torch.cuda.synchronize(device=x.device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _ = model(x)
    end.record()
    torch.cuda.synchronize(device=x.device)

    return {
        "time_ms": start.elapsed_time(end) / iters,
        "peak_alloc_mb": torch.cuda.max_memory_allocated(device=x.device) / (1024.0**2),
        "peak_reserved_mb": torch.cuda.max_memory_reserved(device=x.device) / (1024.0**2),
    }


def _measure_model_fwd_bwd_runtime_and_memory(
    model: DeepSeekV3MoEGEMM,
    x: torch.Tensor,
    *,
    warmup: int = 5,
    iters: int = 20,
) -> dict[str, float]:
    def _step() -> None:
        model.zero_grad(set_to_none=True)
        x_in = x.detach().clone().requires_grad_(True)
        y = model(x_in)
        loss = y.float().square().mean()
        loss.backward()

    if x.device.type != "cuda":
        t0 = time.perf_counter()
        for _ in range(iters):
            _step()
        t1 = time.perf_counter()
        return {"time_ms": ((t1 - t0) / iters) * 1000.0, "peak_alloc_mb": 0.0, "peak_reserved_mb": 0.0}

    torch.cuda.reset_peak_memory_stats(device=x.device)
    torch.cuda.synchronize(device=x.device)
    for _ in range(warmup):
        _step()
    torch.cuda.synchronize(device=x.device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _step()
    end.record()
    torch.cuda.synchronize(device=x.device)

    return {
        "time_ms": start.elapsed_time(end) / iters,
        "peak_alloc_mb": torch.cuda.max_memory_allocated(device=x.device) / (1024.0**2),
        "peak_reserved_mb": torch.cuda.max_memory_reserved(device=x.device) / (1024.0**2),
    }


def test_moe_reference_matches_forward() -> None:
    torch.manual_seed(0)
    model = _build_model(grouped_backend="reference", swiglu_backend="torch")
    model.eval()
    x = torch.randn(3, 7, 32)

    y = model(x)
    y_ref, _ = _reference_forward(model, x)
    assert torch.allclose(y, y_ref, atol=1e-5, rtol=1e-5)


def test_moe_gradients_are_valid() -> None:
    torch.manual_seed(1)
    model = _build_model(grouped_backend="reference", swiglu_backend="torch")
    model.train()
    x = torch.randn(2, 5, 32, requires_grad=True)
    y = model(x)
    loss = y.square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert model.router.weight.grad is not None
    assert model.w1.grad is not None
    assert model.w2.grad is not None
    assert torch.isfinite(model.router.weight.grad).all()
    assert torch.isfinite(model.w1.grad).all()
    assert torch.isfinite(model.w2.grad).all()


def test_moe_parameter_shapes_and_counts() -> None:
    model = _build_model(grouped_backend="reference", swiglu_backend="torch")
    params = dict(model.named_parameters())
    assert "router.weight" in params
    assert "w1" in params
    assert "w2" in params

    assert params["router.weight"].shape == (model.num_experts, model.hidden_size)
    assert params["w1"].shape == (model.num_experts, 2 * model.intermediate_size, model.hidden_size)
    assert params["w2"].shape == (model.num_experts, model.hidden_size, model.intermediate_size)

    expected_numel = (
        model.num_experts * model.hidden_size
        + model.num_experts * 2 * model.intermediate_size * model.hidden_size
        + model.num_experts * model.hidden_size * model.intermediate_size
    )
    total_numel = sum(p.numel() for p in model.parameters())
    assert total_numel == expected_numel


def test_moe_fused_gemm1_swiglu_matches_non_fused() -> None:
    torch.manual_seed(3)
    model_base = _build_model(
        grouped_backend="reference",
        swiglu_backend="torch",
        gemm1_swiglu_fusion_backend="off",
    )
    model_fused = _build_model(
        grouped_backend="reference",
        swiglu_backend="torch",
        gemm1_swiglu_fusion_backend="expert_loop",
    )
    model_fused.load_state_dict(model_base.state_dict(), strict=True)
    model_base.eval()
    model_fused.eval()

    x = torch.randn(4, 11, 32)
    with torch.no_grad():
        y_base = model_base(x)
        y_fused = model_fused(x)
    assert torch.allclose(y_base, y_fused, atol=1e-5, rtol=1e-5)


def test_moe_fused_gemm1_swiglu_triton_matches_non_fused() -> None:
    if not torch.cuda.is_available() or not FUSED_TRITON_AVAILABLE:
        return

    torch.manual_seed(5)
    model_base = _build_model(
        grouped_backend="reference",
        swiglu_backend="torch",
        gemm1_swiglu_fusion_backend="off",
    ).cuda()
    model_triton = _build_model(
        grouped_backend="reference",
        swiglu_backend="torch",
        gemm1_swiglu_fusion_backend="triton",
    ).cuda()
    model_triton.load_state_dict(model_base.state_dict(), strict=True)
    model_base.half().eval()
    model_triton.half().eval()

    x = torch.randn(4, 11, 32, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        y_base = model_base(x)
        y_triton = model_triton(x)
    assert torch.allclose(y_base, y_triton, atol=2e-2, rtol=2e-2)


def test_moe_fused_gemm1_swiglu_triton_backward() -> None:
    if not torch.cuda.is_available() or not FUSED_TRITON_AVAILABLE:
        return

    torch.manual_seed(6)
    model_base = (
        _build_model(
            grouped_backend="reference",
            swiglu_backend="torch",
            gemm1_swiglu_fusion_backend="off",
        )
        .cuda()
        .half()
    )
    model_triton = (
        _build_model(
            grouped_backend="reference",
            swiglu_backend="torch",
            gemm1_swiglu_fusion_backend="triton",
        )
        .cuda()
        .half()
    )
    model_triton.load_state_dict(model_base.state_dict(), strict=True)
    model_base.train()
    model_triton.train()

    x_base = torch.randn(2, 7, 32, device="cuda", dtype=torch.float16, requires_grad=True)
    x_triton = x_base.detach().clone().requires_grad_(True)
    y_base = model_base(x_base)
    y_triton = model_triton(x_triton)
    assert torch.allclose(y_base, y_triton, atol=2e-2, rtol=2e-2)

    loss_base = y_base.float().square().mean()
    loss_triton = y_triton.float().square().mean()
    loss_base.backward()
    loss_triton.backward()

    assert x_base.grad is not None and x_triton.grad is not None
    assert model_base.w1.grad is not None and model_triton.w1.grad is not None
    assert model_base.w2.grad is not None and model_triton.w2.grad is not None
    assert torch.allclose(x_base.grad, x_triton.grad, atol=3e-2, rtol=3e-2)
    assert torch.allclose(model_base.w1.grad, model_triton.w1.grad, atol=3e-2, rtol=3e-2)
    assert torch.allclose(model_base.w2.grad, model_triton.w2.grad, atol=3e-2, rtol=3e-2)


def test_moe_perf_time_and_memory() -> None:
    if not torch.cuda.is_available():
        print("[REPORT][MoE] perf benchmark skipped (cuda unavailable)")
        return

    torch.manual_seed(2)
    x = torch.randn(16, 128, 32, device="cuda", dtype=torch.float16)

    model_torch = _build_model(grouped_backend="auto", swiglu_backend="torch").cuda().half().eval()
    model_triton = _build_model(grouped_backend="auto", swiglu_backend="triton").cuda().half().eval()
    with torch.no_grad():
        y_torch = model_torch(x)
        y_triton = model_triton(x)
    assert torch.allclose(y_torch, y_triton, atol=2e-2, rtol=2e-2)

    stat_torch = _measure_model_runtime_and_memory(model_torch, x)
    stat_triton = _measure_model_runtime_and_memory(model_triton, x)
    speedup = stat_torch["time_ms"] / max(stat_triton["time_ms"], 1e-12)

    print(
        f"[REPORT][MoE] input={tuple(x.shape)} dtype={x.dtype} "
        f"torch_time={stat_torch['time_ms']:.4f}ms torch_peak_alloc={stat_torch['peak_alloc_mb']:.2f}MB "
        f"torch_peak_reserved={stat_torch['peak_reserved_mb']:.2f}MB "
        f"triton_time={stat_triton['time_ms']:.4f}ms triton_peak_alloc={stat_triton['peak_alloc_mb']:.2f}MB "
        f"triton_peak_reserved={stat_triton['peak_reserved_mb']:.2f}MB "
        f"speedup={speedup:.3f}x"
    )

    assert stat_torch["time_ms"] > 0.0
    assert stat_triton["time_ms"] > 0.0
    assert stat_torch["peak_alloc_mb"] >= 0.0
    assert stat_triton["peak_alloc_mb"] >= 0.0


def test_moe_perf_fused_gemm1_swiglu_time_and_memory() -> None:
    if not torch.cuda.is_available() or not FUSED_TRITON_AVAILABLE:
        print("[REPORT][MoE-FusedW1-Triton] perf benchmark skipped (cuda or triton unavailable)")
        return

    torch.manual_seed(4)
    x = torch.randn(16, 128, 32, device="cuda", dtype=torch.float16)

    model_base = (
        _build_model(
            grouped_backend="auto",
            swiglu_backend="torch",
            gemm1_swiglu_fusion_backend="off",
        )
        .cuda()
        .half()
        .eval()
    )
    model_fused = (
        _build_model(
            grouped_backend="auto",
            swiglu_backend="torch",
            gemm1_swiglu_fusion_backend="triton",
        )
        .cuda()
        .half()
        .eval()
    )
    model_fused.load_state_dict(model_base.state_dict(), strict=True)

    with torch.no_grad():
        y_base = model_base(x)
        y_fused = model_fused(x)
    assert torch.allclose(y_base, y_fused, atol=2e-2, rtol=2e-2)

    stat_base = _measure_model_runtime_and_memory(model_base, x)
    stat_fused = _measure_model_runtime_and_memory(model_fused, x)
    speedup = stat_base["time_ms"] / max(stat_fused["time_ms"], 1e-12)

    print(
        f"[REPORT][MoE-FusedW1-Triton] input={tuple(x.shape)} dtype={x.dtype} "
        f"base_time={stat_base['time_ms']:.4f}ms base_peak_alloc={stat_base['peak_alloc_mb']:.2f}MB "
        f"base_peak_reserved={stat_base['peak_reserved_mb']:.2f}MB "
        f"fused_time={stat_fused['time_ms']:.4f}ms fused_peak_alloc={stat_fused['peak_alloc_mb']:.2f}MB "
        f"fused_peak_reserved={stat_fused['peak_reserved_mb']:.2f}MB "
        f"speedup={speedup:.3f}x"
    )


def test_moe_perf_fused_gemm1_swiglu_fwd_bwd_time_and_memory() -> None:
    if not torch.cuda.is_available() or not FUSED_TRITON_AVAILABLE:
        print("[REPORT][MoE-FusedW1-Triton-FWDBWD] perf benchmark skipped (cuda or triton unavailable)")
        return

    torch.manual_seed(7)
    x = torch.randn(16, 128, 32, device="cuda", dtype=torch.float16)

    model_base = (
        _build_model(
            grouped_backend="auto",
            swiglu_backend="torch",
            gemm1_swiglu_fusion_backend="off",
        )
        .cuda()
        .half()
        .train()
    )
    model_fused = (
        _build_model(
            grouped_backend="auto",
            swiglu_backend="torch",
            gemm1_swiglu_fusion_backend="triton",
        )
        .cuda()
        .half()
        .train()
    )
    model_fused.load_state_dict(model_base.state_dict(), strict=True)

    stat_base = _measure_model_fwd_bwd_runtime_and_memory(model_base, x)
    stat_fused = _measure_model_fwd_bwd_runtime_and_memory(model_fused, x)
    speedup = stat_base["time_ms"] / max(stat_fused["time_ms"], 1e-12)

    print(
        f"[REPORT][MoE-FusedW1-Triton-FWDBWD] input={tuple(x.shape)} dtype={x.dtype} "
        f"base_time={stat_base['time_ms']:.4f}ms base_peak_alloc={stat_base['peak_alloc_mb']:.2f}MB "
        f"base_peak_reserved={stat_base['peak_reserved_mb']:.2f}MB "
        f"fused_time={stat_fused['time_ms']:.4f}ms fused_peak_alloc={stat_fused['peak_alloc_mb']:.2f}MB "
        f"fused_peak_reserved={stat_fused['peak_reserved_mb']:.2f}MB speedup={speedup:.3f}x"
    )
