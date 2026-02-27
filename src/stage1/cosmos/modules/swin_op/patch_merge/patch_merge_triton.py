from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _patch_merge_fwd_kernel(
    x_ptr,
    out_ptr,
    numel,
    H,
    W,
    C,
    W2,
    L2,
    C4,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel

    idx = offsets
    c4 = idx % C4
    idx = idx // C4
    l2 = idx % L2
    b = idx // L2

    oy = l2 // W2
    ox = l2 - oy * W2

    quad = c4 // C
    ci = c4 - quad * C
    dy = (quad == 1) | (quad == 3)
    dx = quad >= 2

    in_y = oy * 2 + dy.to(tl.int32)
    in_x = ox * 2 + dx.to(tl.int32)
    valid = (in_y < H) & (in_x < W)

    in_l = in_y * W + in_x
    in_offset = (b * (H * W) + in_l) * C + ci
    vals = tl.load(x_ptr + in_offset, mask=mask & valid, other=0.0)
    tl.store(out_ptr + offsets, vals, mask=mask)


@triton.jit
def _patch_merge_bwd_kernel(
    grad_out_ptr,
    grad_in_ptr,
    numel,
    W,
    C,
    W2,
    C4,
    L,
    L2,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel

    idx = offsets
    ci = idx % C
    idx = idx // C
    l = idx % L
    b = idx // L

    y = l // W
    x = l - y * W
    l2 = (y // 2) * W2 + (x // 2)
    quad = (y % 2) + (x % 2) * 2
    c4 = quad * C + ci

    out_offset = (b * L2 + l2) * C4 + c4
    vals = tl.load(grad_out_ptr + out_offset, mask=mask)
    tl.store(grad_in_ptr + offsets, vals, mask=mask)


def patch_merge_blc_pytorch(x: torch.Tensor, hw_shape: tuple[int, int]) -> tuple[torch.Tensor, tuple[int, int]]:
    b, l, c = x.shape
    h, w = hw_shape
    if l != h * w:
        raise ValueError(f"input feature has wrong size, got L={l}, expected H*W={h * w}")

    x_hw = x.view(b, h, w, c)
    pad_h = h % 2
    pad_w = w % 2
    if pad_h > 0 or pad_w > 0:
        x_hw = torch.nn.functional.pad(x_hw, (0, 0, 0, pad_w, 0, pad_h))
        h, w = h + pad_h, w + pad_w

    x0 = x_hw[:, 0::2, 0::2, :]
    x1 = x_hw[:, 1::2, 0::2, :]
    x2 = x_hw[:, 0::2, 1::2, :]
    x3 = x_hw[:, 1::2, 1::2, :]
    out = torch.cat([x0, x1, x2, x3], dim=-1).reshape(b, -1, 4 * c).contiguous()
    return out, (h // 2, w // 2)


class _PatchMergeTritonFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        if not x.is_cuda:
            raise ValueError("PatchMerge Triton forward requires CUDA tensor")
        if x.ndim != 3:
            raise ValueError(f"Expected x in shape [B, L, C], got ndim={x.ndim}")
        if not x.is_contiguous():
            x = x.contiguous()

        b, l, c = x.shape
        if l != h * w:
            raise ValueError(f"input feature has wrong size, got L={l}, expected H*W={h * w}")

        h2 = (h + 1) // 2
        w2 = (w + 1) // 2
        l2 = h2 * w2
        c4 = 4 * c

        out = torch.empty((b, l2, c4), device=x.device, dtype=x.dtype)
        numel = out.numel()
        block = 256
        grid = (triton.cdiv(numel, block),)
        _patch_merge_fwd_kernel[grid](
            x,
            out,
            numel,
            h,
            w,
            c,
            w2,
            l2,
            c4,
            BLOCK=block,  # type: ignore[invalid-argument-type]
        )

        ctx.h = h
        ctx.w = w
        ctx.c = c
        ctx.l = l
        ctx.l2 = l2
        ctx.w2 = w2
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        if not grad_out.is_cuda:
            raise ValueError("PatchMerge Triton backward requires CUDA tensor")
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()

        b, _, _ = grad_out.shape
        h = int(ctx.h)
        w = int(ctx.w)
        c = int(ctx.c)
        l = int(ctx.l)
        l2 = int(ctx.l2)
        w2 = int(ctx.w2)
        c4 = 4 * c

        grad_in = torch.empty((b, l, c), device=grad_out.device, dtype=grad_out.dtype)
        numel = grad_in.numel()
        block = 256
        grid = (triton.cdiv(numel, block),)
        _patch_merge_bwd_kernel[grid](
            grad_out,
            grad_in,
            numel,
            w,
            c,
            w2,
            c4,
            l,
            l2,
            BLOCK=block,  # type: ignore[invalid-argument-type]
        )
        return grad_in, None, None


def patch_merge_blc_triton(x: torch.Tensor, hw_shape: tuple[int, int]) -> tuple[torch.Tensor, tuple[int, int]]:
    h, w = hw_shape
    out = _PatchMergeTritonFn.apply(x, h, w)
    return out, ((h + 1) // 2, (w + 1) // 2)


def patch_merge_blc(
    x: torch.Tensor,
    hw_shape: tuple[int, int],
    use_triton: bool = True,
) -> tuple[torch.Tensor, tuple[int, int]]:
    if use_triton and x.is_cuda:
        return patch_merge_blc_triton(x, hw_shape)
    return patch_merge_blc_pytorch(x, hw_shape)


__all__ = ["patch_merge_blc", "patch_merge_blc_pytorch", "patch_merge_blc_triton"]
