from __future__ import annotations

import argparse

import torch

from src.stage1.cosmos.modules.swin_op.patch_merge.patch_merge_triton import (
    patch_merge_blc_pytorch,
    patch_merge_blc_triton,
)


def _bench_cuda(fn, warmup: int = 20, iters: int = 100) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    st = torch.cuda.Event(enable_timing=True)
    ed = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(iters):
        fn()
    ed.record()
    torch.cuda.synchronize()
    return st.elapsed_time(ed) / iters


def _one_case(b: int, h: int, w: int, c: int, dtype: torch.dtype) -> None:
    x = torch.randn((b, h * w, c), device="cuda", dtype=dtype)

    def fwd_py() -> None:
        with torch.no_grad():
            patch_merge_blc_pytorch(x, (h, w))

    def fwd_triton() -> None:
        with torch.no_grad():
            patch_merge_blc_triton(x, (h, w))

    def fwd_bwd_py() -> None:
        xin = x.detach().clone().requires_grad_(True)
        y, _ = patch_merge_blc_pytorch(xin, (h, w))
        y.float().square().mean().backward()

    def fwd_bwd_triton() -> None:
        xin = x.detach().clone().requires_grad_(True)
        y, _ = patch_merge_blc_triton(xin, (h, w))
        y.float().square().mean().backward()

    t_py_fwd = _bench_cuda(fwd_py)
    t_tri_fwd = _bench_cuda(fwd_triton)
    t_py_fb = _bench_cuda(fwd_bwd_py, warmup=10, iters=60)
    t_tri_fb = _bench_cuda(fwd_bwd_triton, warmup=10, iters=60)

    print(
        f"[patch-merge] dtype={str(dtype).replace('torch.', '')} B={b} H={h} W={w} C={c} "
        f"fwd_py={t_py_fwd:.3f}ms fwd_triton={t_tri_fwd:.3f}ms speedup={t_py_fwd / t_tri_fwd:.2f}x "
        f"fb_py={t_py_fb:.3f}ms fb_triton={t_tri_fb:.3f}ms speedup={t_py_fb / t_tri_fb:.2f}x"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark patch merge Triton vs PyTorch")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--height", type=int, default=56)
    parser.add_argument("--width", type=int, default=56)
    parser.add_argument("--channels", type=int, default=96)
    parser.add_argument("--dtype", type=str, choices=["float16", "float32", "bfloat16"], default="float16")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    _one_case(args.batch, args.height, args.width, args.channels, dtype_map[args.dtype])


if __name__ == "__main__":
    main()
