from typing import Final

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]

__all__ = ["apply_rope_bhsd", "apply_rope_qk_bhsd", "apply_rotary_ref"]

_MAX_HEAD_DIM: Final[int] = 256


def _normalize_cos_sin(cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor, bool]:
    if cos.ndim == 4:
        if cos.shape[1] != 1:
            raise ValueError(f"Expected cos.shape[1] == 1 for 4D input, got {cos.shape}")
        cos = cos.squeeze(1)
    if sin.ndim == 4:
        if sin.shape[1] != 1:
            raise ValueError(f"Expected sin.shape[1] == 1 for 4D input, got {sin.shape}")
        sin = sin.squeeze(1)

    if cos.shape != sin.shape:
        raise ValueError(f"cos and sin must have the same shape, got {cos.shape} and {sin.shape}")

    if cos.ndim == 2:
        return cos, sin, False
    if cos.ndim == 3:
        return cos, sin, True
    raise ValueError(f"Expected cos/sin ndim in (2, 3, 4), got {cos.ndim}")


def _resolve_rotary_dim(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    rotary_dim: int | None,
    batch_specific: bool,
) -> int:
    if x.ndim != 4:
        raise ValueError(f"Expected x to have shape [B, H, S, D], got {x.shape}")
    if cos.shape != sin.shape:
        raise ValueError(f"cos and sin must have the same shape, got {cos.shape} and {sin.shape}")

    expected_batch = x.shape[0]
    if batch_specific and cos.shape[0] != expected_batch:
        raise ValueError(f"Batch-specific cos/sin must match x batch size, got {cos.shape[0]} and {expected_batch}")

    inferred_rotary_dim = cos.shape[-1] * 2
    if rotary_dim is None:
        rotary_dim = inferred_rotary_dim
    if rotary_dim != inferred_rotary_dim:
        raise ValueError(
            f"rotary_dim must equal cos.shape[-1] * 2, got rotary_dim={rotary_dim}, inferred={inferred_rotary_dim}"
        )
    if rotary_dim % 2 != 0:
        raise ValueError(f"rotary_dim must be even, got {rotary_dim}")
    if rotary_dim > x.shape[-1]:
        raise ValueError(f"rotary_dim must be <= x.shape[-1], got {rotary_dim} and {x.shape[-1]}")
    if x.shape[-1] > _MAX_HEAD_DIM:
        raise ValueError(f"Only head_dim <= {_MAX_HEAD_DIM} is supported, got {x.shape[-1]}")
    return rotary_dim


def _resolve_positions(seqlen: int, offsets: int | Tensor, batch_size: int, device: torch.device) -> Tensor:
    positions = torch.arange(seqlen, device=device, dtype=torch.long)
    if isinstance(offsets, int):
        return positions.unsqueeze(0).expand(batch_size, -1) + offsets

    if offsets.ndim != 1 or offsets.shape[0] != batch_size:
        raise ValueError(f"Expected seqlen_offsets to have shape [{batch_size}], got {offsets.shape}")
    return positions.unsqueeze(0) + offsets.to(device=device, dtype=torch.long).unsqueeze(1)


def apply_rotary_ref(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    rotary_dim: int | None = None,
    interleaved: bool = False,
    seqlen_offsets: int | Tensor = 0,
    conjugate: bool = False,
) -> Tensor:
    cos, sin, batch_specific = _normalize_cos_sin(cos, sin)
    rotary_dim = _resolve_rotary_dim(x, cos, sin, rotary_dim, batch_specific)
    batch, _, seqlen, _ = x.shape
    half_dim = rotary_dim // 2

    if x.device != cos.device or x.device != sin.device:
        raise ValueError("x, cos, and sin must be on the same device")

    positions = _resolve_positions(seqlen, seqlen_offsets, batch, x.device)
    if batch_specific:
        if int(positions.max().item()) >= cos.shape[1]:
            raise ValueError(f"seqlen_offsets require cos/sin length >= {int(positions.max().item()) + 1}")
        gather_index = positions.unsqueeze(-1).expand(-1, -1, cos.shape[-1])
        cos_sel = torch.gather(cos, dim=1, index=gather_index)
        sin_sel = torch.gather(sin, dim=1, index=gather_index)
        cos_view = cos_sel[:, None, :, :]
        sin_view = sin_sel[:, None, :, :]
    else:
        if int(positions.max().item()) >= cos.shape[0]:
            raise ValueError(f"seqlen_offsets require cos/sin length >= {int(positions.max().item()) + 1}")
        if isinstance(seqlen_offsets, int):
            cos_sel = cos.index_select(0, positions[0])
            sin_sel = sin.index_select(0, positions[0])
            cos_view = cos_sel[None, None, :, :]
            sin_view = sin_sel[None, None, :, :]
        else:
            cos_expanded = cos.unsqueeze(0).expand(batch, -1, -1)
            sin_expanded = sin.unsqueeze(0).expand(batch, -1, -1)
            gather_index = positions.unsqueeze(-1).expand(-1, -1, cos.shape[-1])
            cos_sel = torch.gather(cos_expanded, dim=1, index=gather_index)
            sin_sel = torch.gather(sin_expanded, dim=1, index=gather_index)
            cos_view = cos_sel[:, None, :, :]
            sin_view = sin_sel[:, None, :, :]

    x_rope = x[..., :rotary_dim].float()
    sin_view = -sin_view if conjugate else sin_view

    if not interleaved:
        x0 = x_rope[..., :half_dim]
        x1 = x_rope[..., half_dim:rotary_dim]
        out_rope = torch.cat([x0 * cos_view - x1 * sin_view, x0 * sin_view + x1 * cos_view], dim=-1)
    else:
        x_pair = x_rope.reshape(*x_rope.shape[:-1], half_dim, 2)
        x0 = x_pair[..., 0]
        x1 = x_pair[..., 1]
        out_rope = torch.stack([x0 * cos_view - x1 * sin_view, x0 * sin_view + x1 * cos_view], dim=-1)
        out_rope = out_rope.reshape(*x_rope.shape)

    if rotary_dim == x.shape[-1]:
        return out_rope.to(dtype=x.dtype)
    return torch.cat([out_rope.to(dtype=x.dtype), x[..., rotary_dim:]], dim=-1)


if triton is not None:

    @triton.jit
    def _rotary_bhsd_shared_kernel(
        OUT,
        X,
        COS,
        SIN,
        SEQLEN_OFFSETS,
        seqlen,
        nheads,
        seqlen_ro,
        stride_out_batch,
        stride_out_head,
        stride_out_seq,
        stride_out_dim,
        stride_x_batch,
        stride_x_head,
        stride_x_seq,
        stride_x_dim,
        stride_cos_seq,
        stride_cos_dim,
        stride_sin_seq,
        stride_sin_dim,
        ROTARY_DIM: tl.constexpr,
        IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr,
        INTERLEAVED: tl.constexpr,
        CONJUGATE: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        block_k: tl.constexpr = triton.next_power_of_2(ROTARY_DIM)
        rotary_dim_half: tl.constexpr = ROTARY_DIM // 2

        pid_head = tl.program_id(axis=0)
        pid_seq = tl.program_id(axis=1)
        pid_batch = tl.program_id(axis=2)

        rh = pid_head * BLOCK_H + tl.arange(0, BLOCK_H)
        rs = pid_seq * BLOCK_M + tl.arange(0, BLOCK_M)
        if not IS_SEQLEN_OFFSETS_TENSOR:
            rs_cos = rs + SEQLEN_OFFSETS
        else:
            rs_cos = rs + tl.load(SEQLEN_OFFSETS + pid_batch)

        if pid_seq * BLOCK_M >= seqlen:
            return

        rk_half = tl.arange(0, block_k // 2)
        cos_ptrs = COS + rs_cos[:, None] * stride_cos_seq + rk_half[None, :] * stride_cos_dim
        sin_ptrs = SIN + rs_cos[:, None] * stride_sin_seq + rk_half[None, :] * stride_sin_dim
        cos_mask = (rs_cos[:, None] < seqlen_ro) & (rk_half[None, :] < rotary_dim_half)
        sin_mask = cos_mask
        cos = tl.load(cos_ptrs, mask=cos_mask, other=1.0).to(tl.float32)
        sin = tl.load(sin_ptrs, mask=sin_mask, other=0.0).to(tl.float32)
        if CONJUGATE:
            sin = -sin

        base_x = X + pid_batch * stride_x_batch
        base_out = OUT + pid_batch * stride_out_batch

        if not INTERLEAVED:
            x0_ptrs = (
                base_x
                + rh[:, None, None] * stride_x_head
                + rs[None, :, None] * stride_x_seq
                + rk_half[None, None, :] * stride_x_dim
            )
            x1_ptrs = x0_ptrs + rotary_dim_half * stride_x_dim
            out0_ptrs = (
                base_out
                + rh[:, None, None] * stride_out_head
                + rs[None, :, None] * stride_out_seq
                + rk_half[None, None, :] * stride_out_dim
            )
            out1_ptrs = out0_ptrs + rotary_dim_half * stride_out_dim
            x_mask = (
                (rh[:, None, None] < nheads) & (rs[None, :, None] < seqlen) & (rk_half[None, None, :] < rotary_dim_half)
            )
            x0 = tl.load(x0_ptrs, mask=x_mask, other=0.0).to(tl.float32)
            x1 = tl.load(x1_ptrs, mask=x_mask, other=0.0).to(tl.float32)
            out0 = x0 * cos[None, :, :] - x1 * sin[None, :, :]
            out1 = x0 * sin[None, :, :] + x1 * cos[None, :, :]
            tl.store(out0_ptrs, out0, mask=x_mask)
            tl.store(out1_ptrs, out1, mask=x_mask)
        else:
            rk = tl.arange(0, block_k)
            x_ptrs = (
                base_x
                + rh[:, None, None] * stride_x_head
                + rs[None, :, None] * stride_x_seq
                + rk[None, None, :] * stride_x_dim
            )
            out_ptrs = (
                base_out
                + rh[:, None, None] * stride_out_head
                + rs[None, :, None] * stride_out_seq
                + rk[None, None, :] * stride_out_dim
            )
            x_mask = (rh[:, None, None] < nheads) & (rs[None, :, None] < seqlen) & (rk[None, None, :] < ROTARY_DIM)
            x = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)
            x0, x1 = tl.split(tl.reshape(x, [BLOCK_H, BLOCK_M, block_k // 2, 2]))
            out0 = x0 * cos[None, :, :] - x1 * sin[None, :, :]
            out1 = x0 * sin[None, :, :] + x1 * cos[None, :, :]
            out = tl.reshape(tl.join(out0, out1), [BLOCK_H, BLOCK_M, block_k])
            tl.store(out_ptrs, out, mask=x_mask)

    @triton.jit
    def _rotary_bhsd_batched_kernel(
        OUT,
        X,
        COS,
        SIN,
        SEQLEN_OFFSETS,
        seqlen,
        nheads,
        seqlen_ro,
        stride_out_batch,
        stride_out_head,
        stride_out_seq,
        stride_out_dim,
        stride_x_batch,
        stride_x_head,
        stride_x_seq,
        stride_x_dim,
        stride_cos_batch,
        stride_cos_seq,
        stride_cos_dim,
        stride_sin_batch,
        stride_sin_seq,
        stride_sin_dim,
        ROTARY_DIM: tl.constexpr,
        IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr,
        INTERLEAVED: tl.constexpr,
        CONJUGATE: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        block_k: tl.constexpr = triton.next_power_of_2(ROTARY_DIM)
        rotary_dim_half: tl.constexpr = ROTARY_DIM // 2

        pid_head = tl.program_id(axis=0)
        pid_seq = tl.program_id(axis=1)
        pid_batch = tl.program_id(axis=2)

        rh = pid_head * BLOCK_H + tl.arange(0, BLOCK_H)
        rs = pid_seq * BLOCK_M + tl.arange(0, BLOCK_M)
        if not IS_SEQLEN_OFFSETS_TENSOR:
            rs_cos = rs + SEQLEN_OFFSETS
        else:
            rs_cos = rs + tl.load(SEQLEN_OFFSETS + pid_batch)

        if pid_seq * BLOCK_M >= seqlen:
            return

        rk_half = tl.arange(0, block_k // 2)
        cos_ptrs = (
            COS + pid_batch * stride_cos_batch + rs_cos[:, None] * stride_cos_seq + rk_half[None, :] * stride_cos_dim
        )
        sin_ptrs = (
            SIN + pid_batch * stride_sin_batch + rs_cos[:, None] * stride_sin_seq + rk_half[None, :] * stride_sin_dim
        )
        cos_mask = (rs_cos[:, None] < seqlen_ro) & (rk_half[None, :] < rotary_dim_half)
        sin_mask = cos_mask
        cos = tl.load(cos_ptrs, mask=cos_mask, other=1.0).to(tl.float32)
        sin = tl.load(sin_ptrs, mask=sin_mask, other=0.0).to(tl.float32)
        if CONJUGATE:
            sin = -sin

        base_x = X + pid_batch * stride_x_batch
        base_out = OUT + pid_batch * stride_out_batch

        if not INTERLEAVED:
            x0_ptrs = (
                base_x
                + rh[:, None, None] * stride_x_head
                + rs[None, :, None] * stride_x_seq
                + rk_half[None, None, :] * stride_x_dim
            )
            x1_ptrs = x0_ptrs + rotary_dim_half * stride_x_dim
            out0_ptrs = (
                base_out
                + rh[:, None, None] * stride_out_head
                + rs[None, :, None] * stride_out_seq
                + rk_half[None, None, :] * stride_out_dim
            )
            out1_ptrs = out0_ptrs + rotary_dim_half * stride_out_dim
            x_mask = (
                (rh[:, None, None] < nheads) & (rs[None, :, None] < seqlen) & (rk_half[None, None, :] < rotary_dim_half)
            )
            x0 = tl.load(x0_ptrs, mask=x_mask, other=0.0).to(tl.float32)
            x1 = tl.load(x1_ptrs, mask=x_mask, other=0.0).to(tl.float32)
            out0 = x0 * cos[None, :, :] - x1 * sin[None, :, :]
            out1 = x0 * sin[None, :, :] + x1 * cos[None, :, :]
            tl.store(out0_ptrs, out0, mask=x_mask)
            tl.store(out1_ptrs, out1, mask=x_mask)
        else:
            rk = tl.arange(0, block_k)
            x_ptrs = (
                base_x
                + rh[:, None, None] * stride_x_head
                + rs[None, :, None] * stride_x_seq
                + rk[None, None, :] * stride_x_dim
            )
            out_ptrs = (
                base_out
                + rh[:, None, None] * stride_out_head
                + rs[None, :, None] * stride_out_seq
                + rk[None, None, :] * stride_out_dim
            )
            x_mask = (rh[:, None, None] < nheads) & (rs[None, :, None] < seqlen) & (rk[None, None, :] < ROTARY_DIM)
            x = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)
            x0, x1 = tl.split(tl.reshape(x, [BLOCK_H, BLOCK_M, block_k // 2, 2]))
            out0 = x0 * cos[None, :, :] - x1 * sin[None, :, :]
            out1 = x0 * sin[None, :, :] + x1 * cos[None, :, :]
            out = tl.reshape(tl.join(out0, out1), [BLOCK_H, BLOCK_M, block_k])
            tl.store(out_ptrs, out, mask=x_mask)


def _launch_triton_apply_rope(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    rotary_dim: int,
    batch_specific: bool,
    interleaved: bool,
    inplace: bool,
    seqlen_offsets: int | Tensor,
    conjugate: bool,
) -> Tensor:
    if triton is None:
        raise RuntimeError("Triton is required for _launch_triton_apply_rope")

    batch, nheads, seqlen, headdim = x.shape
    seqlen_ro = cos.shape[1] if batch_specific else cos.shape[0]

    if isinstance(seqlen_offsets, Tensor):
        if seqlen_offsets.device != x.device:
            raise ValueError("seqlen_offsets must be on the same device as x")
        if seqlen_offsets.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"seqlen_offsets must be int32 or int64, got {seqlen_offsets.dtype}")
        offsets_arg: int | Tensor = seqlen_offsets.contiguous()
        max_required_len = int((torch.arange(seqlen, device=x.device) + seqlen_offsets.max()).max().item()) + 1
    else:
        offsets_arg = int(seqlen_offsets)
        max_required_len = seqlen + int(seqlen_offsets)

    if seqlen_ro < max_required_len:
        raise ValueError(f"cos/sin length {seqlen_ro} is insufficient for required positions {max_required_len}")

    output = x if inplace else torch.empty_like(x)
    if rotary_dim < headdim and not inplace:
        output[..., rotary_dim:].copy_(x[..., rotary_dim:])

    cos = cos.contiguous()
    sin = sin.contiguous()
    block_m = 8 if rotary_dim <= 128 else 4
    grid = lambda meta: (triton.cdiv(nheads, meta["BLOCK_H"]), triton.cdiv(seqlen, meta["BLOCK_M"]), batch)

    with torch.cuda.device(x.device.index):
        if batch_specific:
            _rotary_bhsd_batched_kernel[grid](
                output,
                x,
                cos,
                sin,
                offsets_arg,
                seqlen,
                nheads,
                seqlen_ro,
                output.stride(0),
                output.stride(1),
                output.stride(2),
                output.stride(3),
                x.stride(0),
                x.stride(1),
                x.stride(2),
                x.stride(3),
                cos.stride(0),
                cos.stride(1),
                cos.stride(2),
                sin.stride(0),
                sin.stride(1),
                sin.stride(2),
                ROTARY_DIM=rotary_dim,  # type: ignore
                IS_SEQLEN_OFFSETS_TENSOR=isinstance(offsets_arg, Tensor),  # type: ignore
                INTERLEAVED=interleaved,  # type: ignore
                CONJUGATE=conjugate,  # type: ignore
                BLOCK_M=block_m,  # type: ignore
                BLOCK_H=2,  # type: ignore
            )
        else:
            _rotary_bhsd_shared_kernel[grid](
                output,
                x,
                cos,
                sin,
                offsets_arg,
                seqlen,
                nheads,
                seqlen_ro,
                output.stride(0),
                output.stride(1),
                output.stride(2),
                output.stride(3),
                x.stride(0),
                x.stride(1),
                x.stride(2),
                x.stride(3),
                cos.stride(0),
                cos.stride(1),
                sin.stride(0),
                sin.stride(1),
                ROTARY_DIM=rotary_dim,  # type: ignore
                IS_SEQLEN_OFFSETS_TENSOR=isinstance(offsets_arg, Tensor),  # type: ignore
                INTERLEAVED=interleaved,  # type: ignore
                CONJUGATE=conjugate,  # type: ignore
                BLOCK_M=block_m,  # type: ignore
                BLOCK_H=2,  # type: ignore
            )
    return output


class _ApplyRopeBHSD(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        rotary_dim: int,
        interleaved: bool,
        inplace: bool,
        seqlen_offsets: int | Tensor,
        conjugate: bool,
    ) -> Tensor:
        cos, sin, batch_specific = _normalize_cos_sin(cos, sin)
        rotary_dim = _resolve_rotary_dim(x, cos, sin, rotary_dim, batch_specific)

        if x.device.type != "cuda" or triton is None:
            output = apply_rotary_ref(
                x=x,
                cos=cos,
                sin=sin,
                rotary_dim=rotary_dim,
                interleaved=interleaved,
                seqlen_offsets=seqlen_offsets,
                conjugate=conjugate,
            )
            if inplace:
                x.copy_(output)
                ctx.mark_dirty(x)
                output = x
        else:
            output = _launch_triton_apply_rope(
                x=x,
                cos=cos,
                sin=sin,
                rotary_dim=rotary_dim,
                batch_specific=batch_specific,
                interleaved=interleaved,
                inplace=inplace,
                seqlen_offsets=seqlen_offsets,
                conjugate=conjugate,
            )
            if inplace:
                ctx.mark_dirty(x)

        tensors_to_save = [cos, sin]
        if isinstance(seqlen_offsets, Tensor):
            tensors_to_save.append(seqlen_offsets)
            ctx.has_tensor_offsets = True
        else:
            ctx.has_tensor_offsets = False
            ctx.seqlen_offsets_int = int(seqlen_offsets)
        ctx.save_for_backward(*tensors_to_save)
        ctx.rotary_dim = rotary_dim
        ctx.interleaved = interleaved
        ctx.inplace = inplace
        ctx.conjugate = conjugate
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None, None, None, None, None, None, None]:
        saved_tensors = ctx.saved_tensors
        cos = saved_tensors[0]
        sin = saved_tensors[1]
        seqlen_offsets: int | Tensor
        if ctx.has_tensor_offsets:
            seqlen_offsets = saved_tensors[2]
        else:
            seqlen_offsets = ctx.seqlen_offsets_int

        _, _, batch_specific = _normalize_cos_sin(cos, sin)
        if grad_output.device.type != "cuda" or triton is None:
            dx = apply_rotary_ref(
                x=grad_output,
                cos=cos,
                sin=sin,
                rotary_dim=ctx.rotary_dim,
                interleaved=ctx.interleaved,
                seqlen_offsets=seqlen_offsets,
                conjugate=not ctx.conjugate,
            )
        else:
            dx = _launch_triton_apply_rope(
                x=grad_output,
                cos=cos,
                sin=sin,
                rotary_dim=ctx.rotary_dim,
                batch_specific=batch_specific,
                interleaved=ctx.interleaved,
                inplace=False,
                seqlen_offsets=seqlen_offsets,
                conjugate=not ctx.conjugate,
            )
        return dx, None, None, None, None, None, None, None


def apply_rope_bhsd(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    rotary_dim: int | None = None,
    interleaved: bool = False,
    inplace: bool = False,
    seqlen_offsets: int | Tensor = 0,
    conjugate: bool = False,
) -> Tensor:
    cos, sin, batch_specific = _normalize_cos_sin(cos, sin)
    rotary_dim = _resolve_rotary_dim(x, cos, sin, rotary_dim, batch_specific)
    return _ApplyRopeBHSD.apply(x, cos, sin, rotary_dim, interleaved, inplace, seqlen_offsets, conjugate)


def apply_rope_qk_bhsd(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    rotary_dim: int | None = None,
    interleaved: bool = False,
    num_prefix_tokens: int = 0,
    inplace: bool = False,
    seqlen_offsets: int | Tensor = 0,
) -> tuple[Tensor, Tensor]:
    if q.shape != k.shape:
        raise ValueError(f"q and k must have the same shape, got {q.shape} and {k.shape}")
    if num_prefix_tokens < 0:
        raise ValueError(f"num_prefix_tokens must be >= 0, got {num_prefix_tokens}")
    if num_prefix_tokens >= q.shape[2]:
        return q, k

    if num_prefix_tokens == 0:
        return (
            apply_rope_bhsd(
                q,
                cos=cos,
                sin=sin,
                rotary_dim=rotary_dim,
                interleaved=interleaved,
                inplace=inplace,
                seqlen_offsets=seqlen_offsets,
            ),
            apply_rope_bhsd(
                k,
                cos=cos,
                sin=sin,
                rotary_dim=rotary_dim,
                interleaved=interleaved,
                inplace=inplace,
                seqlen_offsets=seqlen_offsets,
            ),
        )

    q_prefix, q_patch = q[:, :, :num_prefix_tokens, :], q[:, :, num_prefix_tokens:, :]
    k_prefix, k_patch = k[:, :, :num_prefix_tokens, :], k[:, :, num_prefix_tokens:, :]

    q_patch = apply_rope_bhsd(
        q_patch,
        cos=cos,
        sin=sin,
        rotary_dim=rotary_dim,
        interleaved=interleaved,
        inplace=inplace,
        seqlen_offsets=seqlen_offsets,
    )
    k_patch = apply_rope_bhsd(
        k_patch,
        cos=cos,
        sin=sin,
        rotary_dim=rotary_dim,
        interleaved=interleaved,
        inplace=inplace,
        seqlen_offsets=seqlen_offsets,
    )

    if inplace:
        return q, k
    return torch.cat([q_prefix, q_patch], dim=2), torch.cat([k_prefix, k_patch], dim=2)
