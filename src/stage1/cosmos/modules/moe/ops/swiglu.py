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

SwiGLUBackend = Literal["auto", "triton", "torch"]


def _can_use_triton(x: torch.Tensor) -> bool:
    if not TRITON_AVAILABLE:
        return False
    if x.device.type != "cuda":
        return False
    return x.dtype in {torch.float16, torch.bfloat16, torch.float32}


if TRITON_AVAILABLE:

    @triton.jit
    def _swiglu_fwd_kernel(
        gate_ptr,
        up_ptr,
        out_ptr,
        n_elements: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements

        gate = tl.load(gate_ptr + offs, mask=mask, other=0).to(tl.float32)
        up = tl.load(up_ptr + offs, mask=mask, other=0).to(tl.float32)
        sigmoid = 1.0 / (1.0 + tl.exp(-gate))
        out = gate * sigmoid * up
        tl.store(out_ptr + offs, out, mask=mask)


def _swiglu_forward_triton(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    if not _can_use_triton(gate):
        raise RuntimeError("Triton SwiGLU requires CUDA and fp16/bf16/fp32 dtype.")
    if gate.shape != up.shape:
        raise ValueError(f"gate/up shape mismatch: {tuple(gate.shape)} vs {tuple(up.shape)}")

    gate_flat = gate.contiguous().view(-1)
    up_flat = up.contiguous().view(-1)
    out_flat = torch.empty_like(gate_flat)
    n_elements = gate_flat.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _swiglu_fwd_kernel[grid](gate_flat, up_flat, out_flat, n_elements=n_elements, BLOCK_SIZE=1024)  # type: ignore[arg-type]
    return out_flat.view_as(gate)


class _SwiGLUTritonFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(gate, up)
        return _swiglu_forward_triton(gate, up)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate, up = ctx.saved_tensors

        gate_f32 = gate.to(torch.float32)
        up_f32 = up.to(torch.float32)
        grad_f32 = grad_out.to(torch.float32)

        sigmoid = torch.sigmoid(gate_f32)
        silu = gate_f32 * sigmoid
        dsilu = sigmoid * (1.0 + gate_f32 * (1.0 - sigmoid))

        grad_gate = (grad_f32 * up_f32 * dsilu).to(gate.dtype)
        grad_up = (grad_f32 * silu).to(up.dtype)
        return grad_gate, grad_up


def swiglu_dispatch(gate: torch.Tensor, up: torch.Tensor, backend: SwiGLUBackend = "torch") -> torch.Tensor:
    if gate.shape != up.shape:
        raise ValueError(f"gate/up shape mismatch: {tuple(gate.shape)} vs {tuple(up.shape)}")

    if backend == "torch":
        return F.silu(gate) * up

    if backend == "triton":
        if not _can_use_triton(gate):
            raise RuntimeError("Requested triton backend but it is unavailable in current environment.")
        return _SwiGLUTritonFn.apply(gate, up)

    if _can_use_triton(gate):
        return _SwiGLUTritonFn.apply(gate, up)
    return F.silu(gate) * up


def swiglu_from_packed(
    packed_x: torch.Tensor, intermediate_size: int, backend: SwiGLUBackend = "torch"
) -> torch.Tensor:
    if packed_x.shape[-1] != 2 * intermediate_size:
        raise ValueError(
            f"Expected packed_x.shape[-1] == 2*intermediate_size ({2 * intermediate_size}), got {packed_x.shape[-1]}"
        )
    gate, up = torch.split(packed_x, intermediate_size, dim=-1)
    return swiglu_dispatch(gate, up, backend=backend)


__all__ = ["SwiGLUBackend", "TRITON_AVAILABLE", "swiglu_dispatch", "swiglu_from_packed"]
