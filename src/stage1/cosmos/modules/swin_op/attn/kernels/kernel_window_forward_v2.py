import triton
import triton.language as tl
import torch


_FWD_AUTOTUNE_CONFIGS = [
    triton.Config({}, num_warps=2, num_stages=2),
    triton.Config({}, num_warps=4, num_stages=2),
    triton.Config({}, num_warps=4, num_stages=3),
    triton.Config({}, num_warps=8, num_stages=2),
]


@triton.autotune(configs=_FWD_AUTOTUNE_CONFIGS, key=["seq_pad", "head_dim"])
@triton.jit
def _window_fwd_kernel_v2(
    Q,
    K,
    V,
    bias,
    window_mask,
    O,
    scale_qk,
    batch: tl.constexpr,
    head: tl.constexpr,
    head_dim: tl.constexpr,
    head_chunk: tl.constexpr,
    chunk_dim: tl.constexpr,
    seq: tl.constexpr,
    seq_pad: tl.constexpr,
):
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)

    stride_head = seq * head_dim
    stride_batch = stride_head * head
    offset = batch_id * stride_batch + head_id * stride_head

    if bias is not None:
        bias_ptr = tl.make_block_ptr(
            base=bias + head_id * seq * seq,
            shape=(seq, seq),
            strides=(seq, 1),
            offsets=(0, 0),
            block_shape=(seq_pad, seq_pad),
            order=(1, 0),
        )
        bias_data = tl.load(bias_ptr, boundary_check=(0, 1), padding_option="zero")

    if window_mask is not None:
        mask_ptr = tl.make_block_ptr(
            base=window_mask + batch_id * seq * seq,
            shape=(seq, seq),
            strides=(seq, 1),
            offsets=(0, 0),
            block_shape=(seq_pad, seq_pad),
            order=(1, 0),
        )
        mask_data = tl.load(mask_ptr, boundary_check=(0, 1), padding_option="zero")

    valid = tl.arange(0, seq_pad) < seq

    logits = tl.zeros((seq_pad, seq_pad), dtype=tl.float32)
    q_ptr = tl.make_block_ptr(
        base=Q + offset,
        shape=(seq, head_dim),
        strides=(head_dim, 1),
        offsets=(0, 0),
        block_shape=(seq_pad, chunk_dim),
        order=(1, 0),
    )
    k_ptr = tl.make_block_ptr(
        base=K + offset,
        shape=(seq, head_dim),
        strides=(head_dim, 1),
        offsets=(0, 0),
        block_shape=(seq_pad, chunk_dim),
        order=(1, 0),
    )

    for _ in range(head_chunk):
        q_data = tl.load(q_ptr, boundary_check=(0, 1), padding_option="zero")
        k_data = tl.load(k_ptr, boundary_check=(0, 1), padding_option="zero")
        logits = tl.dot(q_data, k_data.trans(1, 0), logits)
        q_ptr = tl.advance(q_ptr, (0, chunk_dim))
        k_ptr = tl.advance(k_ptr, (0, chunk_dim))

    logits *= scale_qk
    if bias is not None:
        logits += bias_data
    if window_mask is not None:
        logits += mask_data

    logits += tl.where(valid[None, :], 0, -float("inf"))
    logits -= tl.max(logits, axis=1, keep_dims=True)
    probs = tl.math.exp(logits)
    probs /= tl.sum(probs, axis=1, keep_dims=True)
    probs = probs.to(Q.dtype.element_ty)

    v_ptr = tl.make_block_ptr(
        base=V + offset,
        shape=(seq, head_dim),
        strides=(head_dim, 1),
        offsets=(0, 0),
        block_shape=(seq_pad, chunk_dim),
        order=(1, 0),
    )
    index = offset + tl.arange(0, seq_pad)[:, None] * head_dim + tl.arange(0, chunk_dim)[None, :]
    o_ptr = O + index

    for _ in range(head_chunk):
        v_data = tl.load(v_ptr, boundary_check=(0, 1), padding_option="zero")
        o_data = tl.dot(probs, v_data).cast(Q.dtype.element_ty)
        tl.store(o_ptr, o_data, mask=valid[:, None])
        v_ptr = tl.advance(v_ptr, (0, chunk_dim))
        o_ptr += chunk_dim
