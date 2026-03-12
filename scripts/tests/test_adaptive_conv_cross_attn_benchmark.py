import gc
from typing import Any

import pytest
import torch

from src.stage1.cosmos.modules.blocks import AdaptiveInputConvLayer, AdaptiveOutputConvLayer


def _cleanup_cuda_memory() -> None:
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    torch.cuda.synchronize()


def _time_cuda(fn, *, warmup: int = 3, iters: int = 8) -> tuple[float, float]:
    _cleanup_cuda_memory()
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    _cleanup_cuda_memory()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end) / iters
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024**2)
    return elapsed_ms, peak_mem_mb


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for adaptive conv benchmark tests")
def test_adaptive_input_conv_all_modes_benchmark_speed_and_mem() -> None:
    device = torch.device("cuda")
    cases: list[tuple[Any, dict[str, Any]]] = [
        ("slice", {}),
        ("interp", {}),
        ("interp_proj", {"k_hidden": 6}),
        ("mix", {"router_hidden_dim": 0, "always_use_router": True}),
        ("sitok", {"sitok_reduce": "none"}),
        ("sitok", {"sitok_reduce": "mean"}),
        ("sitok", {"sitok_reduce": "pointwise"}),
        ("cross_attn", {"cross_attn_pool_size": 2, "cross_attn_embed_dim": 16}),
    ]
    reports: list[str] = []

    for mode, extra_kwargs in cases:
        layer = AdaptiveInputConvLayer(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1,
            use_bias=False,
            mode=mode,
            **extra_kwargs,
        ).to(device)
        x = torch.randn(2, 220, 64, 64, device=device)

        def run_once() -> None:
            layer.zero_grad(set_to_none=True)
            y = layer(x)
            assert torch.isfinite(y).all()
            y.float().square().mean().backward()

        time_ms, peak_mem_mb = _time_cuda(run_once)
        reports.append(f"[adaptive-input] mode={mode} time_ms={time_ms:.3f} peak_mem_mb={peak_mem_mb:.1f}")
        del layer, x
        _cleanup_cuda_memory()

    print("\n".join(reports))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for adaptive conv benchmark tests")
def test_adaptive_output_conv_all_modes_benchmark_speed_and_mem() -> None:
    device = torch.device("cuda")
    cases: list[tuple[Any, dict[str, Any], int]] = [
        ("slice", {}, 220),
        ("interp", {}, 220),
        ("interp_proj", {"k_hidden": 6}, 220),
        ("mix", {"router_hidden_dim": 0}, 220),
        ("sitok_film", {"sitok_embed_dim": 8, "sitok_hidden_dim": 0}, 220),
        ("sitok_pointwise", {"sitok_embed_dim": 8, "sitok_hidden_dim": 0, "sitok_basis_dim": 4}, 220),
        ("cross_attn", {"cross_attn_pool_size": 4, "cross_attn_embed_dim": 64}, 220),
    ]
    reports: list[str] = []

    for mode, extra_kwargs, runtime_out_channels in cases:
        layer = AdaptiveOutputConvLayer(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1,
            use_bias=False,
            mode=mode,
            **extra_kwargs,
        ).to(device)
        x = torch.randn(2, 512, 128, 128, device=device)

        def run_once() -> None:
            layer.zero_grad(set_to_none=True)
            y = layer(x, out_channels=runtime_out_channels)
            assert torch.isfinite(y).all()
            y.float().square().mean().backward()

        time_ms, peak_mem_mb = _time_cuda(run_once)
        reports.append(f"[adaptive-output] mode={mode} time_ms={time_ms:.3f} peak_mem_mb={peak_mem_mb:.1f}")
        del layer, x
        _cleanup_cuda_memory()

    print("\n".join(reports))
