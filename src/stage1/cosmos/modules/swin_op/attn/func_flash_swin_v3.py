from __future__ import annotations

import torch

from .func_flash_swin_v2 import ceil_pow2
from .kernels.kernel_window_backward_v3 import _window_bwd_kernel_v3
from .kernels.kernel_window_forward_v2 import _window_fwd_kernel_v2


HEAD_CHUNK_DIM = 16
BIAS_REDUCE_CHUNK = 64


def _prepare_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor | None,
    window_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    if window_mask is not None:
        window_mask = window_mask.contiguous()
    return q, k, v, bias, window_mask


def flash_swin_attn_fwd_func_v3(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor | None,
    window_mask: torch.Tensor | None,
    scale_qk: float,
) -> torch.Tensor:
    batch, head, seq, head_dim = q.size()
    if head_dim % HEAD_CHUNK_DIM != 0:
        msg = f"head_dim must be divisible by {HEAD_CHUNK_DIM}, got {head_dim}"
        raise ValueError(msg)

    q, k, v, bias, window_mask = _prepare_inputs(q, k, v, bias, window_mask)

    seq_pad = ceil_pow2(seq)
    head_chunk = head_dim // HEAD_CHUNK_DIM
    out = torch.empty_like(q)

    grid = (batch, head, 1)
    _window_fwd_kernel_v2[grid](
        q,
        k,
        v,
        bias,
        window_mask,
        out,
        scale_qk,
        batch,
        head,
        head_dim,
        head_chunk,
        HEAD_CHUNK_DIM,
        seq,
        seq_pad,
    )
    return out


def flash_swin_attn_bwd_func_v3(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor | None,
    window_mask: torch.Tensor | None,
    d_out: torch.Tensor,
    scale_qk: float,
    bias_reduce_chunk: int = BIAS_REDUCE_CHUNK,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    batch_total, head, seq, head_dim = q.size()
    if head_dim % HEAD_CHUNK_DIM != 0:
        msg = f"head_dim must be divisible by {HEAD_CHUNK_DIM}, got {head_dim}"
        raise ValueError(msg)

    if bias_reduce_chunk <= 0:
        raise ValueError(f"bias_reduce_chunk must be positive, got {bias_reduce_chunk}")

    q, k, v, bias, window_mask = _prepare_inputs(q, k, v, bias, window_mask)
    d_out = d_out.contiguous()

    seq_pad = ceil_pow2(seq)
    head_chunk = head_dim // HEAD_CHUNK_DIM

    d_q = torch.empty_like(q)
    d_k = torch.empty_like(k)
    d_v = torch.empty_like(v)
    d_bias_accum = torch.zeros((head, seq, seq), device=q.device, dtype=torch.float32) if bias is not None else None

    for batch_start in range(0, batch_total, bias_reduce_chunk):
        batch_chunk = min(bias_reduce_chunk, batch_total - batch_start)
        d_bias_partial = (
            torch.empty((batch_chunk, head, seq, seq), device=q.device, dtype=torch.float32)
            if d_bias_accum is not None
            else None
        )

        grid = (batch_chunk, head, 1)
        _window_bwd_kernel_v3[grid](
            q,
            k,
            v,
            bias,
            window_mask,
            d_out,
            d_q,
            d_k,
            d_v,
            d_bias_partial,
            scale_qk,
            batch_start,
            batch_total,
            head,
            head_dim,
            head_chunk,
            HEAD_CHUNK_DIM,
            seq,
            seq_pad,
        )

        if d_bias_partial is not None and d_bias_accum is not None:
            d_bias_accum += d_bias_partial.sum(dim=0)

    d_bias = d_bias_accum.to(dtype=bias.dtype) if d_bias_accum is not None and bias is not None else None
    return d_q, d_k, d_v, d_bias


class FlashSwinFuncV3(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bias: torch.Tensor,
        window_mask: torch.Tensor | None,
        scale_qk: float,
    ) -> torch.Tensor:
        out = flash_swin_attn_fwd_func_v3(q, k, v, bias, window_mask, scale_qk)
        ctx.save_for_backward(q, k, v, bias)
        ctx.window_mask = window_mask
        ctx.scale_qk = scale_qk
        return out

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        d_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        q, k, v, bias = ctx.saved_tensors
        d_q, d_k, d_v, d_bias = flash_swin_attn_bwd_func_v3(
            q,
            k,
            v,
            bias,
            ctx.window_mask,
            d_out,
            ctx.scale_qk,
        )
        if d_bias is None:
            d_bias = torch.zeros_like(bias)
        return d_q, d_k, d_v, d_bias, None, None


flash_swin_attn_func_v3 = FlashSwinFuncV3.apply
