import time
from collections.abc import Callable

import pytest
import torch
from timm.layers import create_norm_layer

import src.stage1.cosmos.modules.norm  # noqa: F401
from src.stage1.cosmos.modules.rmsnorm_triton import TritonRMSNorm2dFunc


def _run_case(shape: tuple[int, int, int, int], dtype: torch.dtype, make_noncontiguous: bool) -> None:
    device = torch.device("cuda")
    torch.manual_seed(0)
    base: torch.Tensor = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    x: torch.Tensor = base
    if make_noncontiguous:
        x = base[:, :, :, ::2]
        x.retain_grad()

    c = x.shape[1]
    ref_norm = create_norm_layer("rmsnorm2d", c, eps=1e-5).to(device=device)
    with torch.no_grad():
        ref_norm.weight.copy_(torch.randn_like(ref_norm.weight))
    b_ref_param = torch.nn.Parameter(torch.randn(c, device=device, dtype=torch.float32))

    w = ref_norm.weight.detach().clone().to(dtype=dtype).requires_grad_(True)
    b = b_ref_param.detach().clone().to(dtype=dtype).requires_grad_(True)
    eps = 1e-5

    y_triton = TritonRMSNorm2dFunc.apply(x, w, b, eps)
    loss_triton = y_triton.float().square().mean()
    loss_triton.backward()
    dx_triton = x.grad.detach().clone() if make_noncontiguous else base.grad.detach().clone()
    dw_triton = w.grad.detach().clone()
    db_triton = b.grad.detach().clone()

    base_ref = base.detach().clone().requires_grad_(True)
    x_ref = base_ref[:, :, :, ::2] if make_noncontiguous else base_ref
    if make_noncontiguous:
        x_ref.retain_grad()
    ref_norm.weight = torch.nn.Parameter(w.detach().clone().float())
    bias_for_ref = torch.nn.Parameter(b.detach().clone().float())
    y_ref = ref_norm(x_ref).to(dtype) + bias_for_ref.view(1, -1, 1, 1).to(dtype)
    loss_ref = y_ref.float().square().mean()
    loss_ref.backward()
    dx_ref = x_ref.grad.detach().clone() if make_noncontiguous else base_ref.grad.detach().clone()
    dw_ref = ref_norm.weight.grad.detach().clone().to(dtype)
    db_ref = bias_for_ref.grad.detach().clone().to(dtype)

    atol = 2e-2 if dtype == torch.bfloat16 else 4e-3
    rtol = 2e-2 if dtype == torch.bfloat16 else 4e-3
    assert torch.allclose(y_triton, y_ref, atol=atol, rtol=rtol)
    assert torch.allclose(dx_triton, dx_ref, atol=atol, rtol=rtol)
    assert torch.allclose(dw_triton, dw_ref, atol=atol, rtol=rtol)
    assert torch.allclose(db_triton, db_ref, atol=atol, rtol=rtol)


def _measure_cuda_op(op: Callable[[], None], warmup: int, iters: int) -> tuple[float, float]:
    for _ in range(warmup):
        op()

    torch.cuda.synchronize()
    start_alloc = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    for _ in range(iters):
        op()
    torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - t0

    peak_alloc = torch.cuda.max_memory_allocated()
    peak_delta_mb = max(0, peak_alloc - start_alloc) / (1024**2)
    avg_ms = elapsed_s * 1000.0 / iters
    return avg_ms, peak_delta_mb


def _build_ref_and_triton_ops(
    shape: tuple[int, int, int, int], dtype: torch.dtype, eps: float
) -> tuple[Callable[[], None], Callable[[], None]]:
    device = torch.device("cuda")
    torch.manual_seed(0)
    c = shape[1]
    x = torch.randn(shape, device=device, dtype=dtype)
    w = torch.randn(c, device=device, dtype=dtype)
    b = torch.randn(c, device=device, dtype=dtype)

    ref_norm = create_norm_layer("rmsnorm2d", c, eps=eps).to(device=device)
    with torch.no_grad():
        ref_norm.weight.copy_(w.float())

    def ref_op() -> None:
        x_ref = x.detach().clone().requires_grad_(True)
        w_ref = torch.nn.Parameter(w.detach().clone().float())
        b_ref = torch.nn.Parameter(b.detach().clone().float())
        ref_norm.weight = w_ref
        y_ref = ref_norm(x_ref).to(dtype) + b_ref.view(1, -1, 1, 1).to(dtype)
        loss = y_ref.float().square().mean()
        loss.backward()

    def triton_op() -> None:
        x_tri = x.detach().clone().requires_grad_(True)
        w_tri = w.detach().clone().requires_grad_(True)
        b_tri = b.detach().clone().requires_grad_(True)
        y_tri = TritonRMSNorm2dFunc.apply(x_tri, w_tri, b_tri, eps)
        loss = y_tri.float().square().mean()
        loss.backward()

    return ref_op, triton_op


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernel test")
@pytest.mark.parametrize(
    ("shape", "dtype", "make_noncontiguous"),
    [
        ((2, 16, 11, 13), torch.float16, False),
        ((2, 16, 11, 13), torch.bfloat16, False),
        ((2, 16, 11, 14), torch.float16, True),
    ],
)
def test_rmsnorm2d_triton_matches_reference(
    shape: tuple[int, int, int, int], dtype: torch.dtype, make_noncontiguous: bool
) -> None:
    _run_case(shape=shape, dtype=dtype, make_noncontiguous=make_noncontiguous)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernel test")
def test_rmsnorm2d_triton_benchmark_print() -> None:
    shape = (8, 128, 256, 256)
    dtype = torch.float16
    eps = 1e-5
    warmup = 10
    iters = 50

    ref_op, triton_op = _build_ref_and_triton_ops(shape=shape, dtype=dtype, eps=eps)
    ref_ms, ref_mem_mb = _measure_cuda_op(ref_op, warmup=warmup, iters=iters)
    tri_ms, tri_mem_mb = _measure_cuda_op(triton_op, warmup=warmup, iters=iters)

    speedup = ref_ms / tri_ms if tri_ms > 0 else float("inf")
    print(
        f"[rmsnorm2d bench] shape={shape} dtype={dtype} iters={iters} warmup={warmup}\n"
        f"  Py(timm+bias):   {ref_ms:.3f} ms/iter, peak_delta={ref_mem_mb:.2f} MB\n"
        f"  Triton(custom):  {tri_ms:.3f} ms/iter, peak_delta={tri_mem_mb:.2f} MB\n"
        f"  speedup(Py/Tri): {speedup:.3f}x"
    )

    assert ref_ms > 0
    assert tri_ms > 0
