import triton
import triton.language as tl
import torch


_BWD_AUTOTUNE_CONFIGS = [
    triton.Config({}, num_warps=2, num_stages=2),
    triton.Config({}, num_warps=4, num_stages=2),
    triton.Config({}, num_warps=4, num_stages=3),
    triton.Config({}, num_warps=8, num_stages=2),
]


@triton.autotune(configs=_BWD_AUTOTUNE_CONFIGS, key=["seq_pad", "head_dim"], reset_to_zero=["d_bias"])
@triton.jit
def _window_bwd_kernel_v2(
    Q,
    K,
    V,
    bias,
    window_mask,
    d_O,
    d_Q,
    d_K,
    d_V,
    d_bias,
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
    valid_2d = valid[:, None] & valid[None, :]

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

    d_logits = tl.zeros((seq_pad, seq_pad), dtype=tl.float32)
    d_o_ptr = tl.make_block_ptr(
        base=d_O + offset,
        shape=(seq, head_dim),
        strides=(head_dim, 1),
        offsets=(0, 0),
        block_shape=(seq_pad, chunk_dim),
        order=(1, 0),
    )
    v_ptr = tl.make_block_ptr(
        base=V + offset,
        shape=(seq, head_dim),
        strides=(head_dim, 1),
        offsets=(0, 0),
        block_shape=(seq_pad, chunk_dim),
        order=(1, 0),
    )

    index = offset + tl.arange(0, seq_pad)[:, None] * head_dim + tl.arange(0, chunk_dim)[None, :]
    d_v_ptr = d_V + index

    for _ in range(head_chunk):
        d_o_data = tl.load(d_o_ptr, boundary_check=(0, 1), padding_option="zero")
        v_data = tl.load(v_ptr, boundary_check=(0, 1), padding_option="zero")

        d_logits = tl.dot(d_o_data, v_data.trans(1, 0), d_logits)
        d_v_data = tl.dot(probs.trans(1, 0), d_o_data).cast(V.dtype.element_ty)
        tl.store(d_v_ptr, d_v_data, mask=valid[:, None])

        d_o_ptr = tl.advance(d_o_ptr, (0, chunk_dim))
        v_ptr = tl.advance(v_ptr, (0, chunk_dim))
        d_v_ptr += chunk_dim

    probs_sum = tl.sum(probs * d_logits, axis=1, keep_dims=True)
    d_logits = probs.to(tl.float32) * (d_logits - probs_sum)
    d_logits = tl.where(valid_2d, d_logits, 0.0)

    if bias is not None:
        bias_index = head_id * seq * seq + tl.arange(0, seq_pad)[:, None] * seq + tl.arange(0, seq_pad)[None, :]
        d_bias_ptr = d_bias + bias_index
        tl.atomic_add(d_bias_ptr, d_logits, mask=valid_2d)

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
    d_q_ptr = d_Q + index
    d_k_ptr = d_K + index

    for _ in range(head_chunk):
        q_data = tl.load(q_ptr, boundary_check=(0, 1), padding_option="zero")
        k_data = tl.load(k_ptr, boundary_check=(0, 1), padding_option="zero")
        d_logits_q = d_logits.to(Q.dtype.element_ty)

        d_q_data = (tl.dot(d_logits_q, k_data) * scale_qk).cast(Q.dtype.element_ty)
        tl.store(d_q_ptr, d_q_data, mask=valid[:, None])

        d_k_data = (tl.dot(d_logits_q.trans(1, 0), q_data) * scale_qk).cast(K.dtype.element_ty)
        tl.store(d_k_ptr, d_k_data, mask=valid[:, None])

        q_ptr = tl.advance(q_ptr, (0, chunk_dim))
        k_ptr = tl.advance(k_ptr, (0, chunk_dim))
        d_q_ptr += chunk_dim
        d_k_ptr += chunk_dim
