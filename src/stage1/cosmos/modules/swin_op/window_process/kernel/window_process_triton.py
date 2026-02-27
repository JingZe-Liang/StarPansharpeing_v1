from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _roll_and_window_partition_fwd_kernel(
    x_ptr,
    out_ptr,
    numel,
    H,
    W,
    C,
    window_size,
    shift_size,
    nH,
    nW,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel

    idx = offsets
    c = idx % C
    idx = idx // C
    wx = idx % window_size
    idx = idx // window_size
    wy = idx % window_size
    idx = idx // window_size
    win = idx
    ww = win % nW
    idx = idx // nW
    wh = idx % nH
    b = idx // nH

    in_y = (wh * window_size + wy - shift_size + H) % H
    in_x = (ww * window_size + wx - shift_size + W) % W
    in_offset = ((b * H + in_y) * W + in_x) * C + c

    x_vals = tl.load(x_ptr + in_offset, mask=mask)
    tl.store(out_ptr + offsets, x_vals, mask=mask)


@triton.jit
def _roll_and_window_partition_bwd_kernel(
    grad_in_ptr,
    grad_out_ptr,
    numel,
    H,
    W,
    C,
    window_size,
    shift_size,
    nW,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel

    idx = offsets
    c = idx % C
    idx = idx // C
    x = idx % W
    idx = idx // W
    y = idx % H
    b = idx // H

    y_shift = (y + shift_size + H) % H
    x_shift = (x + shift_size + W) % W
    win = b * (H // window_size) * nW + (y_shift // window_size) * nW + (x_shift // window_size)
    wy = y_shift % window_size
    wx = x_shift % window_size
    in_offset = ((win * window_size + wy) * window_size + wx) * C + c

    grad_vals = tl.load(grad_in_ptr + in_offset, mask=mask)
    tl.store(grad_out_ptr + offsets, grad_vals, mask=mask)


@triton.jit
def _window_merge_and_roll_fwd_kernel(
    x_ptr,
    out_ptr,
    numel,
    H,
    W,
    C,
    window_size,
    shift_size,
    nW,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel

    idx = offsets
    c = idx % C
    idx = idx // C
    x = idx % W
    idx = idx // W
    y = idx % H
    b = idx // H

    y_shift = (y - shift_size + H) % H
    x_shift = (x - shift_size + W) % W
    win = b * (H // window_size) * nW + (y_shift // window_size) * nW + (x_shift // window_size)
    wy = y_shift % window_size
    wx = x_shift % window_size
    in_offset = ((win * window_size + wy) * window_size + wx) * C + c

    x_vals = tl.load(x_ptr + in_offset, mask=mask)
    tl.store(out_ptr + offsets, x_vals, mask=mask)


@triton.jit
def _window_merge_and_roll_bwd_kernel(
    grad_in_ptr,
    grad_out_ptr,
    numel,
    H,
    W,
    C,
    window_size,
    shift_size,
    nH,
    nW,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel

    idx = offsets
    c = idx % C
    idx = idx // C
    wx = idx % window_size
    idx = idx // window_size
    wy = idx % window_size
    idx = idx // window_size
    win = idx
    ww = win % nW
    idx = idx // nW
    wh = idx % nH
    b = idx // nH

    in_y = (wh * window_size + wy + shift_size + H) % H
    in_x = (ww * window_size + wx + shift_size + W) % W
    in_offset = ((b * H + in_y) * W + in_x) * C + c

    grad_vals = tl.load(grad_in_ptr + in_offset, mask=mask)
    tl.store(grad_out_ptr + offsets, grad_vals, mask=mask)


def _check_cuda_contiguous(x: torch.Tensor, name: str) -> None:
    if not x.is_cuda:
        raise ValueError(f"{name} must be CUDA tensor")
    if not x.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _check_hw_window(H: int, W: int, window_size: int) -> tuple[int, int]:
    if H % window_size != 0 or W % window_size != 0:
        raise ValueError(f"H and W must be divisible by window_size, got H={H}, W={W}, ws={window_size}")
    return H // window_size, W // window_size


def _launch_grid(numel: int, block: int) -> tuple[int]:
    return (triton.cdiv(numel, block),)


def roll_and_window_partition_forward(
    x: torch.Tensor,
    B: int,
    H: int,
    W: int,
    C: int,
    shift_size: int,
    window_size: int,
) -> torch.Tensor:
    _check_cuda_contiguous(x, "x")
    nH, nW = _check_hw_window(H, W, window_size)
    out = torch.empty((B * nH * nW, window_size, window_size, C), device=x.device, dtype=x.dtype)
    numel = out.numel()
    block = 256
    _roll_and_window_partition_fwd_kernel[_launch_grid(numel, block)](
        x,
        out,
        numel,
        H,
        W,
        C,
        window_size,
        shift_size,
        nH,
        nW,
        BLOCK=block,  # type: ignore[invalid-argument-type]
    )
    return out


def roll_and_window_partition_backward(
    grad_in: torch.Tensor,
    B: int,
    H: int,
    W: int,
    C: int,
    shift_size: int,
    window_size: int,
) -> torch.Tensor:
    _check_cuda_contiguous(grad_in, "grad_in")
    _, nW = _check_hw_window(H, W, window_size)
    grad_out = torch.empty((B, H, W, C), device=grad_in.device, dtype=grad_in.dtype)
    numel = grad_out.numel()
    block = 256
    _roll_and_window_partition_bwd_kernel[_launch_grid(numel, block)](
        grad_in,
        grad_out,
        numel,
        H,
        W,
        C,
        window_size,
        shift_size,
        nW,
        BLOCK=block,  # type: ignore[invalid-argument-type]
    )
    return grad_out


def window_merge_and_roll_forward(
    x: torch.Tensor,
    B: int,
    H: int,
    W: int,
    C: int,
    shift_size: int,
    window_size: int,
) -> torch.Tensor:
    _check_cuda_contiguous(x, "x")
    _, nW = _check_hw_window(H, W, window_size)
    out = torch.empty((B, H, W, C), device=x.device, dtype=x.dtype)
    numel = out.numel()
    block = 256
    _window_merge_and_roll_fwd_kernel[_launch_grid(numel, block)](
        x,
        out,
        numel,
        H,
        W,
        C,
        window_size,
        shift_size,
        nW,
        BLOCK=block,  # type: ignore[invalid-argument-type]
    )
    return out


def window_merge_and_roll_backward(
    grad_in: torch.Tensor,
    B: int,
    H: int,
    W: int,
    C: int,
    shift_size: int,
    window_size: int,
) -> torch.Tensor:
    _check_cuda_contiguous(grad_in, "grad_in")
    nH, nW = _check_hw_window(H, W, window_size)
    grad_out = torch.empty((B * nH * nW, window_size, window_size, C), device=grad_in.device, dtype=grad_in.dtype)
    numel = grad_out.numel()
    block = 256
    _window_merge_and_roll_bwd_kernel[_launch_grid(numel, block)](
        grad_in,
        grad_out,
        numel,
        H,
        W,
        C,
        window_size,
        shift_size,
        nH,
        nW,
        BLOCK=block,  # type: ignore[invalid-argument-type]
    )
    return grad_out
