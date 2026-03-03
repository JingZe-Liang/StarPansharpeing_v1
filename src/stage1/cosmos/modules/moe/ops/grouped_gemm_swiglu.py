from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn.functional as F

triton: Any = None
tl: Any = None
try:
    import triton as _triton
    import triton.language as _tl

    triton = _triton
    tl = _tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False

Gemm1SwiGLUFusionBackend = Literal["off", "expert_loop", "triton", "auto"]


def _validate_grouped_gemm1_swiglu_inputs(
    x_sorted: torch.Tensor,
    w1_expert: torch.Tensor,
    counts: torch.Tensor,
) -> int:
    if x_sorted.dim() != 2:
        raise ValueError(f"Expected x_sorted to be 2D, got shape={tuple(x_sorted.shape)}")
    if w1_expert.dim() != 3:
        raise ValueError(f"Expected w1_expert to be 3D, got shape={tuple(w1_expert.shape)}")
    if w1_expert.shape[1] % 2 != 0:
        raise ValueError(f"Expected w1_expert.shape[1] to be even, got {w1_expert.shape[1]}")
    if counts.dim() != 1:
        raise ValueError(f"Expected counts to be 1D, got shape={tuple(counts.shape)}")
    if counts.shape[0] != w1_expert.shape[0]:
        raise ValueError(
            f"counts length must match num_experts, got counts={counts.shape[0]} num_experts={w1_expert.shape[0]}"
        )

    total_tokens = int(counts.to(dtype=torch.int64).sum().item())
    if total_tokens != x_sorted.shape[0]:
        raise ValueError(f"counts do not sum to x_sorted rows: sum(counts)={total_tokens}, rows={x_sorted.shape[0]}")
    return w1_expert.shape[1] // 2


def _can_use_triton_fused(x_sorted: torch.Tensor, w1_expert: torch.Tensor) -> bool:
    if not TRITON_AVAILABLE:
        return False
    if x_sorted.device.type != "cuda" or w1_expert.device.type != "cuda":
        return False
    if x_sorted.dtype != w1_expert.dtype:
        return False
    return x_sorted.dtype in {torch.float16, torch.bfloat16, torch.float32}


def _can_use_triton_tensor(x: torch.Tensor) -> bool:
    if not TRITON_AVAILABLE:
        return False
    if x.device.type != "cuda":
        return False
    return x.dtype in {torch.float16, torch.bfloat16, torch.float32}


def _make_offsets_from_counts(counts: torch.Tensor, device: torch.device) -> torch.Tensor:
    counts_i32 = counts.to(dtype=torch.int32, device=device).contiguous()
    return torch.cumsum(counts_i32, dim=0, dtype=torch.int32)


def _grouped_mm_reference(x_sorted: torch.Tensor, w_expert: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    out = torch.empty((x_sorted.shape[0], w_expert.shape[1]), dtype=w_expert.dtype, device=x_sorted.device)
    x_proj = x_sorted.to(dtype=w_expert.dtype)
    start = 0
    for expert_idx, size_i64 in enumerate(counts.to(dtype=torch.int64).tolist()):
        size = int(size_i64)
        if size <= 0:
            continue
        end = start + size
        out[start:end] = x_proj[start:end] @ w_expert[expert_idx].transpose(0, 1)
        start = end
    return out


def _grouped_mm_torch_functional(x_sorted: torch.Tensor, w_expert: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    if not hasattr(torch.nn.functional, "grouped_mm"):
        raise RuntimeError("torch.nn.functional.grouped_mm is not available.")
    offs = _make_offsets_from_counts(counts, device=x_sorted.device)
    w = w_expert.transpose(-2, -1).contiguous()
    return torch.nn.functional.grouped_mm(x_sorted.to(w.dtype), w, offs=offs)


def _grouped_mm_torch_private(x_sorted: torch.Tensor, w_expert: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    if not hasattr(torch, "_grouped_mm"):
        raise RuntimeError("torch._grouped_mm is not available.")
    offs = _make_offsets_from_counts(counts, device=x_sorted.device)
    w = w_expert.transpose(-2, -1).contiguous()
    return torch._grouped_mm(x_sorted.to(w.dtype), w, offs)


def _grouped_mm_dispatch(x_sorted: torch.Tensor, w_expert: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    errors: list[str] = []
    for impl in (_grouped_mm_torch_functional, _grouped_mm_torch_private):
        try:
            return impl(x_sorted, w_expert, counts)
        except Exception as exc:
            errors.append(f"{impl.__name__}: {exc}")
    try:
        return _grouped_mm_reference(x_sorted, w_expert, counts)
    except Exception as exc:
        errors.append(f"_grouped_mm_reference: {exc}")
        raise RuntimeError("No grouped mm backend is usable.\n" + "\n".join(errors)) from exc


def grouped_gemm1_swiglu_expert_loop(
    x_sorted: torch.Tensor,
    w1_expert: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    intermediate_size = _validate_grouped_gemm1_swiglu_inputs(x_sorted, w1_expert, counts)
    x_proj = x_sorted.to(dtype=w1_expert.dtype)
    y_hidden = torch.empty(
        (x_proj.shape[0], intermediate_size),
        dtype=x_proj.dtype,
        device=x_proj.device,
    )

    start = 0
    for expert_idx, size_i64 in enumerate(counts.to(dtype=torch.int64).tolist()):
        size = int(size_i64)
        if size <= 0:
            continue
        end = start + size
        x_chunk = x_proj[start:end]
        w_chunk = w1_expert[expert_idx]
        proj = x_chunk @ w_chunk.transpose(0, 1)
        gate, up = torch.split(proj, intermediate_size, dim=-1)
        y_hidden[start:end] = F.silu(gate) * up
        start = end
    return y_hidden


def _swiglu_backward_torch(
    gate: torch.Tensor, up: torch.Tensor, grad_out: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    sigmoid = torch.sigmoid(gate)
    silu = gate * sigmoid
    dsilu = sigmoid * (1.0 + gate * (1.0 - sigmoid))
    return grad_out * up * dsilu, grad_out * silu


if TRITON_AVAILABLE:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        ],
        key=["I", "H"],
    )
    @triton.jit
    def _grouped_gemm1_swiglu_fwd_kernel(
        x_ptr,
        w_ptr,
        counts_ptr,
        starts_ptr,
        y_ptr,
        stride_xm,
        stride_xk,
        stride_we,
        stride_wn,
        stride_wk,
        stride_ym,
        stride_yn,
        I: tl.constexpr,
        H: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_e = tl.program_id(2)

        local_count = tl.load(counts_ptr + pid_e)
        row_start = tl.load(starts_ptr + pid_e)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rows = row_start + offs_m

        mask_m = offs_m < local_count
        mask_n = offs_n < I

        acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, H, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            mask_k = offs_k < H

            x_ptrs = x_ptr + rows[:, None] * stride_xm + offs_k[None, :] * stride_xk
            x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

            w_gate_ptrs = w_ptr + pid_e * stride_we + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
            w_gate = tl.load(w_gate_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

            offs_n_up = offs_n + I
            w_up_ptrs = w_ptr + pid_e * stride_we + offs_n_up[None, :] * stride_wn + offs_k[:, None] * stride_wk
            w_up = tl.load(w_up_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

            acc_gate += tl.dot(x, w_gate)
            acc_up += tl.dot(x, w_up)

        sigmoid = 1.0 / (1.0 + tl.exp(-acc_gate))
        out = acc_gate * sigmoid * acc_up

        y_ptrs = y_ptr + rows[:, None] * stride_ym + offs_n[None, :] * stride_yn
        tl.store(y_ptrs, out, mask=mask_m[:, None] & mask_n[None, :])

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=3),
        ],
        key=["n_elements"],
    )
    @triton.jit
    def _swiglu_bwd_pointwise_kernel(
        gate_ptr,
        up_ptr,
        grad_out_ptr,
        grad_gate_ptr,
        grad_up_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements

        gate = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        grad_out = tl.load(grad_out_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        sigmoid = 1.0 / (1.0 + tl.exp(-gate))
        silu = gate * sigmoid
        dsilu = sigmoid * (1.0 + gate * (1.0 - sigmoid))
        grad_gate = grad_out * up * dsilu
        grad_up = grad_out * silu

        tl.store(grad_gate_ptr + offs, grad_gate, mask=mask)
        tl.store(grad_up_ptr + offs, grad_up, mask=mask)


def _swiglu_backward_dispatch(
    gate: torch.Tensor,
    up: torch.Tensor,
    grad_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        _can_use_triton_tensor(gate)
        and gate.shape == up.shape
        and gate.shape == grad_out.shape
        and gate.dtype == up.dtype
        and gate.dtype == grad_out.dtype
    ):
        gate_flat = gate.contiguous().view(-1)
        up_flat = up.contiguous().view(-1)
        grad_flat = grad_out.contiguous().view(-1)
        grad_gate_flat = torch.empty_like(gate_flat)
        grad_up_flat = torch.empty_like(up_flat)
        n_elements = gate_flat.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _swiglu_bwd_pointwise_kernel[grid](
            gate_flat,
            up_flat,
            grad_flat,
            grad_gate_flat,
            grad_up_flat,
            n_elements=n_elements,
        )
        return grad_gate_flat.view_as(gate), grad_up_flat.view_as(up)
    return _swiglu_backward_torch(gate, up, grad_out)


def _grouped_gemm1_swiglu_backward(
    grad_out: torch.Tensor,
    x_sorted: torch.Tensor,
    w1_expert: torch.Tensor,
    counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    intermediate_size = w1_expert.shape[1] // 2
    w_gate = w1_expert[:, :intermediate_size, :].contiguous()
    w_up = w1_expert[:, intermediate_size:, :].contiguous()
    grad_proj = grad_out.to(dtype=w1_expert.dtype)

    gate = _grouped_mm_dispatch(x_sorted, w_gate, counts)
    up = _grouped_mm_dispatch(x_sorted, w_up, counts)
    grad_gate, grad_up = _swiglu_backward_dispatch(gate, up, grad_proj)

    grad_x_gate = _grouped_mm_dispatch(grad_gate, w_gate.transpose(-2, -1).contiguous(), counts)
    grad_x_up = _grouped_mm_dispatch(grad_up, w_up.transpose(-2, -1).contiguous(), counts)
    grad_x = grad_x_gate + grad_x_up

    grad_w_gate = torch.zeros_like(w_gate)
    grad_w_up = torch.zeros_like(w_up)
    start = 0
    for expert_idx, size_i64 in enumerate(counts.to(dtype=torch.int64).tolist()):
        size = int(size_i64)
        if size <= 0:
            continue
        end = start + size
        x_chunk = x_sorted[start:end].to(dtype=w1_expert.dtype)
        grad_gate_chunk = grad_gate[start:end]
        grad_up_chunk = grad_up[start:end]
        grad_w_gate[expert_idx] = grad_gate_chunk.transpose(0, 1) @ x_chunk
        grad_w_up[expert_idx] = grad_up_chunk.transpose(0, 1) @ x_chunk
        start = end

    grad_w1 = torch.cat((grad_w_gate, grad_w_up), dim=1)
    return grad_x, grad_w1


def _grouped_gemm1_swiglu_triton_forward(
    x_sorted: torch.Tensor,
    w1_expert: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    intermediate_size = _validate_grouped_gemm1_swiglu_inputs(x_sorted, w1_expert, counts)
    if not _can_use_triton_fused(x_sorted, w1_expert):
        raise RuntimeError("Triton fused GEMM1+SwiGLU requires CUDA tensors with matching fp16/bf16/fp32 dtype.")

    if x_sorted.shape[0] == 0:
        return x_sorted.new_zeros((0, intermediate_size))

    x_proj = x_sorted.contiguous()
    w_proj = w1_expert.contiguous()
    counts_i32 = counts.to(dtype=torch.int32, device=x_proj.device).contiguous()
    starts_i32 = (torch.cumsum(counts_i32, dim=0, dtype=torch.int32) - counts_i32).contiguous()

    max_count = int(counts_i32.max().item())
    if max_count <= 0:
        return x_proj.new_zeros((x_proj.shape[0], intermediate_size))

    y_hidden = torch.empty((x_proj.shape[0], intermediate_size), dtype=x_proj.dtype, device=x_proj.device)
    grid = lambda meta: (
        triton.cdiv(max_count, meta["BLOCK_M"]),
        triton.cdiv(intermediate_size, meta["BLOCK_N"]),
        w_proj.shape[0],
    )

    _grouped_gemm1_swiglu_fwd_kernel[grid](
        x_proj,
        w_proj,
        counts_i32,
        starts_i32,
        y_hidden,
        x_proj.stride(0),
        x_proj.stride(1),
        w_proj.stride(0),
        w_proj.stride(1),
        w_proj.stride(2),
        y_hidden.stride(0),
        y_hidden.stride(1),
        I=intermediate_size,
        H=x_proj.shape[1],
    )
    return y_hidden


class _GroupedGemm1SwiGLUTritonFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_sorted: torch.Tensor, w1_expert: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        x_input_dtype = x_sorted.dtype
        x_proj = x_sorted.to(dtype=w1_expert.dtype)
        y_hidden = _grouped_gemm1_swiglu_triton_forward(x_proj, w1_expert, counts)
        ctx.save_for_backward(x_proj, w1_expert, counts)
        ctx.x_input_dtype = x_input_dtype
        return y_hidden

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        x_proj, w1_expert, counts = ctx.saved_tensors
        grad_x, grad_w1 = _grouped_gemm1_swiglu_backward(grad_out, x_proj, w1_expert, counts)
        return grad_x.to(dtype=ctx.x_input_dtype), grad_w1, None


def grouped_gemm1_swiglu_dispatch(
    x_sorted: torch.Tensor,
    w1_expert: torch.Tensor,
    counts: torch.Tensor,
    backend: Gemm1SwiGLUFusionBackend = "auto",
) -> torch.Tensor:
    if backend == "off":
        raise ValueError("backend='off' should be handled by caller, not grouped_gemm1_swiglu_dispatch.")
    if backend == "expert_loop":
        return grouped_gemm1_swiglu_expert_loop(x_sorted, w1_expert, counts)
    if backend == "triton":
        return _GroupedGemm1SwiGLUTritonFn.apply(x_sorted, w1_expert, counts)
    if _can_use_triton_fused(x_sorted, w1_expert):
        return _GroupedGemm1SwiGLUTritonFn.apply(x_sorted, w1_expert, counts)
    return grouped_gemm1_swiglu_expert_loop(x_sorted, w1_expert, counts)


__all__ = [
    "Gemm1SwiGLUFusionBackend",
    "TRITON_AVAILABLE",
    "grouped_gemm1_swiglu_dispatch",
    "grouped_gemm1_swiglu_expert_loop",
]

# TODO: add dw1 fused with dx

# triton kernels performances
# 1. fwd: ~0.99 x gemm1 + pytorch swiglu + gemm2
# 2. bwd: ~0.68 speed ...
