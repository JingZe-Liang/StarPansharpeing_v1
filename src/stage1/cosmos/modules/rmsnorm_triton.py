# Copyright 2024 MIT Han Lab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# -----------------------------------------------------------
# Modified by zihan-cao, UESTC
# fixed forward / backward stride issue
# - Performance comparison
# Shape: [8, 128, 256, 256]
# Py(timm+bias):   7.327 ms/iter, peak_delta=1665.00 MB
# Triton(custom):  3.529 ms/iter, peak_delta=1538.00 MB
# speedup(Py/Tri): 2.077x
# -----------------------------------------------------------

import torch
import triton
import triton.language as tl

__all__ = ["TritonRMSNorm2dFunc"]


@triton.jit
def _rms_norm_2d_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Rrms,  # pointer to the 1/rms
    M,
    C,
    N,
    stride_xm,
    stride_xc,
    stride_xn,
    stride_ym,
    stride_yc,
    stride_yn,
    stride_rm,
    stride_rn,
    num_blocks,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    m_n = tl.program_id(0)
    m, n = m_n // num_blocks, m_n % num_blocks

    cols = n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x_sum_square = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, C):
        x_ptr = X + m * stride_xm + off * stride_xc + cols * stride_xn
        x = tl.load(x_ptr, mask=mask, other=0.0).to(tl.float32)
        x_sum_square += x * x
    mean_square = x_sum_square / C
    rrms = 1 / tl.sqrt(mean_square + eps)
    tl.store(Rrms + m * stride_rm + cols * stride_rn, rrms, mask=mask)
    # Normalize and apply linear transformation
    for off in range(0, C):
        w = tl.load(W + off)
        b = tl.load(B + off) if HAS_BIAS else 0.0
        x_ptr = X + m * stride_xm + off * stride_xc + cols * stride_xn
        y_ptr = Y + m * stride_ym + off * stride_yc + cols * stride_yn
        x = tl.load(x_ptr, mask=mask, other=0.0).to(tl.float32)
        x_hat = x * rrms
        y = x_hat * w + b
        tl.store(y_ptr, y, mask=mask)


@triton.jit
def _rms_norm_2d_bwd_dx_fused(
    DX,  # pointer to the input gradient
    DY,  # pointer to the output gradient
    DW,  # pointer to the partial sum of weights gradient
    DB,  # pointer to the partial sum of biases gradient
    X,  # pointer to the input
    W,  # pointer to the weights
    Rrms,  # pointer to the 1/rms
    M,
    C,
    N,  # number of columns in X
    stride_dxm,
    stride_dxc,
    stride_dxn,
    stride_dym,
    stride_dyc,
    stride_dyn,
    stride_xm,
    stride_xc,
    stride_xn,
    stride_rm,
    stride_rn,
    num_blocks,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the elements of X, DX, and DY it should compute.
    m_n = tl.program_id(0)
    m, n = m_n // num_blocks, m_n % num_blocks

    cols = n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    # Offset locks and weights/biases gradient pointer for parallel reduction
    DW = DW + m_n * C
    DB = DB + m_n * C
    rrms = tl.load(Rrms + m * stride_rm + cols * stride_rn, mask=mask, other=1)
    # Load data to SRAM
    c1 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, C):
        x_ptr = X + m * stride_xm + off * stride_xc + cols * stride_xn
        dy_ptr = DY + m * stride_dym + off * stride_dyc + cols * stride_dyn
        x = tl.load(x_ptr, mask=mask, other=0).to(tl.float32)
        dy = tl.load(dy_ptr, mask=mask, other=0).to(tl.float32)
        w = tl.load(W + off).to(tl.float32)
        # Compute dx
        xhat = x * rrms
        wdy = w * dy
        xhat = tl.where(mask, xhat, 0.0)
        wdy = tl.where(mask, wdy, 0.0)
        c1 += xhat * wdy
        # Accumulate partial sums for dw/db
        tl.store(DW + off, tl.sum(dy * xhat, axis=0))
        if HAS_BIAS:
            tl.store(DB + off, tl.sum(dy, axis=0))

    c1 /= C
    for off in range(0, C):
        x_ptr = X + m * stride_xm + off * stride_xc + cols * stride_xn
        dy_ptr = DY + m * stride_dym + off * stride_dyc + cols * stride_dyn
        dx_ptr = DX + m * stride_dxm + off * stride_dxc + cols * stride_dxn
        x = tl.load(x_ptr, mask=mask, other=0).to(tl.float32)
        dy = tl.load(dy_ptr, mask=mask, other=0).to(tl.float32)
        w = tl.load(W + off).to(tl.float32)
        xhat = x * rrms
        wdy = w * dy
        dx = (wdy - (xhat * c1)) * rrms
        tl.store(dx_ptr, dx, mask=mask)


class TritonRMSNorm2dFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, eps: float) -> torch.Tensor:
        # allocate output
        y = torch.empty_like(x)

        x_arg = x.reshape(x.shape[0], x.shape[1], -1)
        y_arg = y.reshape_as(x_arg)
        bias_arg = bias if bias is not None else weight
        has_bias = bias is not None
        M, C, N = x_arg.shape
        rrms = torch.empty((M, N), dtype=torch.float32, device=x.device)
        BLOCK_SIZE = 256
        num_blocks = triton.cdiv(N, BLOCK_SIZE)
        num_warps = 8

        fwd_kernel = _rms_norm_2d_fwd_fused[(M * num_blocks,)]
        if has_bias:
            fwd_kernel(
                x_arg,
                y_arg,
                weight,
                bias_arg,
                rrms,
                M,
                C,
                N,
                x_arg.stride(0),
                x_arg.stride(1),
                x_arg.stride(2),
                y_arg.stride(0),
                y_arg.stride(1),
                y_arg.stride(2),
                rrms.stride(0),
                rrms.stride(1),
                num_blocks,
                eps,
                HAS_BIAS=True,  # type: ignore
                BLOCK_SIZE=BLOCK_SIZE,  # type: ignore
                num_warps=num_warps,  # type: ignore
                num_ctas=1,  # type: ignore
            )
        else:
            fwd_kernel(
                x_arg,
                y_arg,
                weight,
                bias_arg,
                rrms,
                M,
                C,
                N,
                x_arg.stride(0),
                x_arg.stride(1),
                x_arg.stride(2),
                y_arg.stride(0),
                y_arg.stride(1),
                y_arg.stride(2),
                rrms.stride(0),
                rrms.stride(1),
                num_blocks,
                eps,
                HAS_BIAS=False,  # type: ignore
                BLOCK_SIZE=BLOCK_SIZE,  # type: ignore
                num_warps=num_warps,  # type: ignore
                num_ctas=1,  # type: ignore
            )
        ctx.save_for_backward(x_arg, weight, rrms)
        ctx.has_bias = has_bias
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_blocks = num_blocks
        ctx.num_warps = num_warps
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, None]:
        x_arg, w, rrms = ctx.saved_tensors
        num_blocks = ctx.num_blocks
        has_bias: bool = ctx.has_bias

        dy_arg = dy.reshape_as(x_arg)
        M, C, N = x_arg.shape
        dx = torch.empty_like(dy)
        dx_arg = dx.reshape_as(x_arg)
        partial_count = M * num_blocks
        _dw = torch.empty((partial_count, C), dtype=torch.float32, device=w.device)
        _db = torch.empty((partial_count, C), dtype=torch.float32, device=w.device) if has_bias else _dw

        bwd_kernel = _rms_norm_2d_bwd_dx_fused[(M * num_blocks,)]
        if has_bias:
            bwd_kernel(
                dx_arg,
                dy_arg,
                _dw,
                _db,
                x_arg,
                w,
                rrms,
                M,
                C,
                N,
                dx_arg.stride(0),
                dx_arg.stride(1),
                dx_arg.stride(2),
                dy_arg.stride(0),
                dy_arg.stride(1),
                dy_arg.stride(2),
                x_arg.stride(0),
                x_arg.stride(1),
                x_arg.stride(2),
                rrms.stride(0),
                rrms.stride(1),
                num_blocks,
                HAS_BIAS=True,  # type: ignore
                BLOCK_SIZE=ctx.BLOCK_SIZE,
                num_warps=ctx.num_warps,  # type: ignore
            )
        else:
            bwd_kernel(
                dx_arg,
                dy_arg,
                _dw,
                _db,
                x_arg,
                w,
                rrms,
                M,
                C,
                N,
                dx_arg.stride(0),
                dx_arg.stride(1),
                dx_arg.stride(2),
                dy_arg.stride(0),
                dy_arg.stride(1),
                dy_arg.stride(2),
                x_arg.stride(0),
                x_arg.stride(1),
                x_arg.stride(2),
                rrms.stride(0),
                rrms.stride(1),
                num_blocks,
                HAS_BIAS=False,  # type: ignore
                BLOCK_SIZE=ctx.BLOCK_SIZE,
                num_warps=ctx.num_warps,  # type: ignore
            )
        dw = _dw.sum(dim=0).to(w.dtype)
        db = _db.sum(dim=0).to(w.dtype) if has_bias else None
        return dx, dw, db, None
