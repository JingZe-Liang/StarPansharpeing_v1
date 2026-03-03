from __future__ import annotations

import torch

from src.stage1.cosmos.modules.moe.ops.swiglu import TRITON_AVAILABLE, swiglu_dispatch, swiglu_from_packed


def test_swiglu_torch_dispatch_matches_math() -> None:
    gate = torch.randn(7, 9, dtype=torch.float32)
    up = torch.randn(7, 9, dtype=torch.float32)
    y_ref = torch.nn.functional.silu(gate) * up
    y = swiglu_dispatch(gate, up, backend="torch")
    assert torch.allclose(y, y_ref, atol=1e-7, rtol=0.0)


def test_swiglu_from_packed_shape_and_values() -> None:
    intermediate = 5
    packed = torch.randn(4, 2 * intermediate, dtype=torch.float32)
    gate, up = torch.split(packed, intermediate, dim=-1)
    y_ref = torch.nn.functional.silu(gate) * up
    y = swiglu_from_packed(packed, intermediate, backend="torch")
    assert y.shape == (4, intermediate)
    assert torch.allclose(y, y_ref, atol=1e-7, rtol=0.0)


def test_swiglu_auto_fallback_on_cpu() -> None:
    gate = torch.randn(8, 16, dtype=torch.float32)
    up = torch.randn(8, 16, dtype=torch.float32)
    y_auto = swiglu_dispatch(gate, up, backend="auto")
    y_torch = swiglu_dispatch(gate, up, backend="torch")
    assert torch.allclose(y_auto, y_torch, atol=1e-7, rtol=0.0)


def _benchmark_swiglu_forward_cuda(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    backend: str,
    warmup: int = 20,
    iters: int = 300,
) -> dict[str, float]:
    torch.cuda.synchronize(device=gate.device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = swiglu_dispatch(gate, up, backend=backend)  # type: ignore[arg-type]
    torch.cuda.synchronize(device=gate.device)

    torch.cuda.reset_peak_memory_stats(device=gate.device)
    start_alloc = torch.cuda.memory_allocated(device=gate.device)
    start_reserved = torch.cuda.memory_reserved(device=gate.device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        for _ in range(iters):
            _ = swiglu_dispatch(gate, up, backend=backend)  # type: ignore[arg-type]
    end.record()
    torch.cuda.synchronize(device=gate.device)

    end_alloc = torch.cuda.memory_allocated(device=gate.device)
    end_reserved = torch.cuda.memory_reserved(device=gate.device)
    peak_alloc = torch.cuda.max_memory_allocated(device=gate.device)
    peak_reserved = torch.cuda.max_memory_reserved(device=gate.device)
    time_ms = start.elapsed_time(end) / iters
    bytes_per_iter = float((gate.numel() + up.numel() + gate.numel()) * gate.element_size())
    bandwidth_gbps = (bytes_per_iter / 1e9) / max(time_ms / 1e3, 1e-12)
    return {
        "time_ms": time_ms,
        "peak_alloc_mb": peak_alloc / (1024.0**2),
        "peak_reserved_mb": peak_reserved / (1024.0**2),
        "delta_alloc_mb": (end_alloc - start_alloc) / (1024.0**2),
        "delta_reserved_mb": (end_reserved - start_reserved) / (1024.0**2),
        "bandwidth_gbps": bandwidth_gbps,
    }


def test_swiglu_triton_forward_backward_matches_torch() -> None:
    if not torch.cuda.is_available() or not TRITON_AVAILABLE:
        print("[REPORT][SwiGLU] triton benchmark skipped (cuda or triton unavailable)")
        return

    gate_a = torch.randn(256, 256, 128, device="cuda", dtype=torch.float16, requires_grad=True)
    up_a = torch.randn(256, 256, 128, device="cuda", dtype=torch.float16, requires_grad=True)
    gate_b = gate_a.detach().clone().requires_grad_(True)
    up_b = up_a.detach().clone().requires_grad_(True)

    y_triton = swiglu_dispatch(gate_a, up_a, backend="triton")
    y_torch = swiglu_dispatch(gate_b, up_b, backend="torch")
    assert torch.allclose(y_triton, y_torch, atol=1e-3, rtol=1e-3)

    grad = torch.randn_like(y_triton)
    y_triton.backward(grad)
    y_torch.backward(grad)

    assert gate_a.grad is not None and gate_b.grad is not None
    assert up_a.grad is not None and up_b.grad is not None
    assert torch.allclose(gate_a.grad, gate_b.grad, atol=2e-3, rtol=2e-3)
    assert torch.allclose(up_a.grad, up_b.grad, atol=2e-3, rtol=2e-3)

    # performance report (forward only)
    gate_eval = gate_a.detach()
    up_eval = up_a.detach()
    stat_torch = _benchmark_swiglu_forward_cuda(gate_eval, up_eval, backend="torch")
    stat_triton = _benchmark_swiglu_forward_cuda(gate_eval, up_eval, backend="triton")
    speedup = stat_torch["time_ms"] / max(stat_triton["time_ms"], 1e-12)
    print(
        f"[REPORT][SwiGLU] shape={tuple(gate_a.shape)} dtype={gate_a.dtype} "
        f"torch={stat_torch['time_ms']:.4f}ms torch_bw={stat_torch['bandwidth_gbps']:.1f}GB/s "
        f"torch_peak_alloc={stat_torch['peak_alloc_mb']:.3f}MB "
        f"torch_peak_reserved={stat_torch['peak_reserved_mb']:.3f}MB "
        f"torch_delta_alloc={stat_torch['delta_alloc_mb']:.3f}MB "
        f"torch_delta_reserved={stat_torch['delta_reserved_mb']:.3f}MB "
        f"triton={stat_triton['time_ms']:.4f}ms triton_bw={stat_triton['bandwidth_gbps']:.1f}GB/s "
        f"triton_peak_alloc={stat_triton['peak_alloc_mb']:.3f}MB "
        f"triton_peak_reserved={stat_triton['peak_reserved_mb']:.3f}MB "
        f"triton_delta_alloc={stat_triton['delta_alloc_mb']:.3f}MB "
        f"triton_delta_reserved={stat_triton['delta_reserved_mb']:.3f}MB "
        f"speedup={speedup:.3f}x"
    )
