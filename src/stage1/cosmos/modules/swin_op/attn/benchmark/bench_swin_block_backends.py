from __future__ import annotations

import argparse

import torch

from src.stage1.cosmos.modules.swin_op.swin_transformer import Mlp, SwinTransformerBlock
from src.stage1.cosmos.modules.variants.mlp import SwiGLU


def _bench_cuda(fn, warmup: int, iters: int) -> float:
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
    return start.elapsed_time(end) / iters


def _build_block(
    dim: int,
    h: int,
    w: int,
    heads: int,
    window_size: int,
    shift_size: int,
    attn_backend: str,
    window_backend: str,
    dtype: torch.dtype,
    mlp_cls: type[torch.nn.Module],
    mlp_kwargs: dict[str, object] | None,
) -> SwinTransformerBlock:
    block = SwinTransformerBlock(
        dim=dim,
        input_resolution=(h, w),
        num_heads=heads,
        window_size=window_size,
        shift_size=shift_size,
        is_flash=(attn_backend != "py"),
        attn_backend=attn_backend,
        window_backend=window_backend,
        mlp_cls=mlp_cls,
        mlp_kwargs=mlp_kwargs,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        drop_path=0.0,
    ).cuda()
    return block.to(dtype=dtype)


def _one_step(block: SwinTransformerBlock, x: torch.Tensor, mode: str) -> None:
    if mode == "fwd":
        with torch.no_grad():
            _ = block(x)
        return
    if mode != "fwd_bwd":
        raise ValueError(f"Unsupported mode: {mode}")
    x_in = x.detach().clone().requires_grad_(True)
    out = block(x_in)
    loss = out.float().square().mean() + 0.1 * out.float().abs().mean()
    loss.backward()


def _measure_gpu_peak_mb(block: SwinTransformerBlock, x: torch.Tensor, mode: str) -> float:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _one_step(block, x, mode)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024**2)


def _run_case(
    batch: int,
    h: int,
    w: int,
    dim: int,
    heads: int,
    window_size: int,
    shift_size: int,
    dtype: torch.dtype,
    mode: str,
    warmup: int,
    iters: int,
    triton_attn_backend: str,
    mlp_kind: str,
    mlp_fused: str,
) -> None:
    x = torch.randn((batch, h * w, dim), device="cuda", dtype=dtype)
    mlp_cls, mlp_kwargs = _resolve_mlp(mlp_kind, mlp_fused)
    block_py = _build_block(dim, h, w, heads, window_size, shift_size, "py", "py", dtype, mlp_cls, mlp_kwargs)
    block_triton = _build_block(
        dim,
        h,
        w,
        heads,
        window_size,
        shift_size,
        triton_attn_backend,
        "triton",
        dtype,
        mlp_cls,
        mlp_kwargs,
    )
    block_triton.load_state_dict(block_py.state_dict(), strict=True)

    t_py = _bench_cuda(lambda: _one_step(block_py, x, mode), warmup=warmup, iters=iters)
    t_triton = _bench_cuda(lambda: _one_step(block_triton, x, mode), warmup=warmup, iters=iters)

    mem_py = _measure_gpu_peak_mb(block_py, x, mode)
    mem_triton = _measure_gpu_peak_mb(block_triton, x, mode)

    speedup = t_py / t_triton if t_triton > 0 else float("inf")
    mem_gain = mem_py / mem_triton if mem_triton > 0 else float("inf")
    print(
        f"[block-bench] mode={mode} dtype={str(dtype).replace('torch.', '')} "
        f"B={batch} H={h} W={w} C={dim} heads={heads} ws={window_size} shift={shift_size} "
        f"mlp={mlp_kind} fused={mlp_fused}"
    )
    print(f"[block-bench] py      time={t_py:.3f}ms gpu_peak={mem_py:.1f}MB")
    print(f"[block-bench] triton  time={t_triton:.3f}ms gpu_peak={mem_triton:.1f}MB backend={triton_attn_backend}")
    print(f"[block-bench] speedup={speedup:.2f}x mem_gain={mem_gain:.2f}x")


def _resolve_mlp(mlp_kind: str, mlp_fused: str) -> tuple[type[torch.nn.Module], dict[str, object] | None]:
    if mlp_kind == "mlp":
        return Mlp, None
    fused = None if mlp_fused == "none" else mlp_fused
    return SwiGLU, {"is_fused": fused, "use_conv": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SwinTransformerBlock: py vs triton kernels")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--height", type=int, default=56)
    parser.add_argument("--width", type=int, default=56)
    parser.add_argument("--dim", type=int, default=96)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--window-size", type=int, default=7)
    parser.add_argument("--shift-size", type=int, default=3)
    parser.add_argument("--mode", type=str, choices=["fwd", "fwd_bwd"], default="fwd_bwd")
    parser.add_argument("--dtype", type=str, choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=60)
    parser.add_argument(
        "--triton-attn-backend",
        type=str,
        choices=["triton_v1", "triton_v2", "triton_v3", "hybrid_v3", "sdpa"],
        default="triton_v3",
    )
    parser.add_argument("--mlp-kind", type=str, choices=["mlp", "swiglu"], default="mlp")
    parser.add_argument("--mlp-fused", type=str, choices=["none", "xformers", "fla"], default="none")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.dim % args.heads != 0:
        raise ValueError(f"dim must be divisible by heads, got dim={args.dim}, heads={args.heads}")
    if args.height % args.window_size != 0 or args.width % args.window_size != 0:
        raise ValueError("height and width must be divisible by window-size for this benchmark")
    if args.shift_size >= args.window_size:
        raise ValueError("shift-size must be smaller than window-size")

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    _run_case(
        batch=args.batch,
        h=args.height,
        w=args.width,
        dim=args.dim,
        heads=args.heads,
        window_size=args.window_size,
        shift_size=args.shift_size,
        dtype=dtype_map[args.dtype],
        mode=args.mode,
        warmup=args.warmup,
        iters=args.iters,
        triton_attn_backend=args.triton_attn_backend,
        mlp_kind=args.mlp_kind,
        mlp_fused=args.mlp_fused,
    )


if __name__ == "__main__":
    main()
