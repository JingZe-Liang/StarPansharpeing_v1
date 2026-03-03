import pytest
import torch

import importlib.util
from pathlib import Path


def _load_linear_attn_kernel_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "src" / "stage1" / "cosmos" / "modules" / "SLA" / "linear_attn_kernel.py"
    spec = importlib.util.spec_from_file_location("sla_linear_attn_kernel", module_path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_linear_attn_mod = _load_linear_attn_kernel_module()
linear_attention = _linear_attn_mod.linear_attention
linear_attention_reference = _linear_attn_mod.linear_attention_reference


def _make_positive_feature(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1)


def _time_cuda(fn, *, warmup: int = 10, iters: int = 30) -> float:
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end)
    return ms / iters


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("d", [64, 128])
def test_triton_linear_attention_matches_reference_self(dtype: torch.dtype, d: int) -> None:
    pytest.importorskip("triton")

    torch.manual_seed(0)
    device = torch.device("cuda")

    b, h, lq, lk = 2, 2, 256, 256

    q = _make_positive_feature(torch.randn((b, h, lq, d), device=device, dtype=torch.float32)).to(dtype)
    k = _make_positive_feature(torch.randn((b, h, lk, d), device=device, dtype=torch.float32)).to(dtype)
    v = torch.randn((b, h, lk, d), device=device, dtype=torch.float32).to(dtype)

    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)

    eps = 1e-5

    out_tri = linear_attention(q, k, v, eps=eps)
    out_ref = linear_attention_reference(q.float(), k.float(), v.float(), eps=eps).to(dtype)

    torch.testing.assert_close(out_tri, out_ref, rtol=2e-2, atol=2e-2)

    grad = torch.randn_like(out_tri)
    loss_tri = (out_tri * grad).sum()
    loss_ref = (out_ref * grad).sum()

    loss_tri.backward()

    q2 = q.detach().clone().requires_grad_(True)
    k2 = k.detach().clone().requires_grad_(True)
    v2 = v.detach().clone().requires_grad_(True)
    out_ref2 = linear_attention_reference(q2.float(), k2.float(), v2.float(), eps=eps).to(dtype)
    loss_ref2 = (out_ref2 * grad).sum()
    loss_ref2.backward()

    torch.testing.assert_close(q.grad, q2.grad, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(k.grad, k2.grad, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(v.grad, v2.grad, rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_triton_linear_attention_matches_reference_cross(dtype: torch.dtype) -> None:
    pytest.importorskip("triton")

    torch.manual_seed(0)
    device = torch.device("cuda")

    b, h, lq, lk, d = 2, 2, 192, 256, 64

    q = _make_positive_feature(torch.randn((b, h, lq, d), device=device, dtype=torch.float32)).to(dtype)
    k = _make_positive_feature(torch.randn((b, h, lk, d), device=device, dtype=torch.float32)).to(dtype)
    v = torch.randn((b, h, lk, d), device=device, dtype=torch.float32).to(dtype)

    eps = 1e-5

    out_tri = linear_attention(q, k, v, eps=eps)
    out_ref = linear_attention_reference(q.float(), k.float(), v.float(), eps=eps).to(dtype)

    torch.testing.assert_close(out_tri, out_ref, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")
def test_triton_linear_attention_benchmark_speed_and_mem() -> None:
    pytest.importorskip("triton")

    torch.manual_seed(0)
    device = torch.device("cuda")

    b, h, lq, lk, d = 4, 16, 1024, 1024, 64
    dtype = torch.bfloat16
    eps = 1e-5

    q = _make_positive_feature(torch.randn((b, h, lq, d), device=device, dtype=torch.float32)).to(dtype)
    k = _make_positive_feature(torch.randn((b, h, lk, d), device=device, dtype=torch.float32)).to(dtype)
    v = torch.randn((b, h, lk, d), device=device, dtype=torch.float32).to(dtype)

    q1 = q.detach().clone().requires_grad_(True)
    k1 = k.detach().clone().requires_grad_(True)
    v1 = v.detach().clone().requires_grad_(True)

    def tri_fwd_bwd() -> None:
        o = linear_attention(q1, k1, v1, eps=eps)
        (o.square().mean()).backward()
        q1.grad = None
        k1.grad = None
        v1.grad = None

    q2 = q.detach().clone().requires_grad_(True)
    k2 = k.detach().clone().requires_grad_(True)
    v2 = v.detach().clone().requires_grad_(True)

    def ref_fwd_bwd() -> None:
        o = linear_attention_reference(q2.float(), k2.float(), v2.float(), eps=eps).to(dtype)
        (o.square().mean()).backward()
        q2.grad = None
        k2.grad = None
        v2.grad = None

    tri_ms = _time_cuda(tri_fwd_bwd, warmup=5, iters=20)
    ref_ms = _time_cuda(ref_fwd_bwd, warmup=5, iters=20)

    torch.cuda.reset_peak_memory_stats()
    tri_fwd_bwd()
    torch.cuda.synchronize()
    tri_mem = torch.cuda.max_memory_allocated()

    torch.cuda.reset_peak_memory_stats()
    ref_fwd_bwd()
    torch.cuda.synchronize()
    ref_mem = torch.cuda.max_memory_allocated()

    print(f"[linear_attn] triton: {tri_ms:.3f} ms/iter, peak_mem={tri_mem / 1024 / 1024:.1f} MiB")
    print(f"[linear_attn] torch : {ref_ms:.3f} ms/iter, peak_mem={ref_mem / 1024 / 1024:.1f} MiB")

    out = linear_attention(q, k, v, eps=eps)
    assert torch.isfinite(out).all()
