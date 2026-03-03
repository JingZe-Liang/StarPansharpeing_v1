import torch
import triton
import triton.language as tl


def _check_qkv_shapes(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[int, int, int, int, int]:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(f"Expected q/k/v to be 4D, got {q.ndim=}, {k.ndim=}, {v.ndim=}.")
    if q.shape[:2] != k.shape[:2] or k.shape[:2] != v.shape[:2]:
        raise ValueError(f"Expected matching (B, H), got {q.shape[:2]=}, {k.shape[:2]=}, {v.shape[:2]=}.")
    if k.shape[-2] != v.shape[-2]:
        raise ValueError(f"Expected matching Lk for k/v, got {k.shape[-2]=} and {v.shape[-2]=}.")
    if q.shape[-1] != k.shape[-1] or k.shape[-1] != v.shape[-1]:
        raise ValueError(f"Expected matching D for q/k/v, got {q.shape[-1]=}, {k.shape[-1]=}, {v.shape[-1]=}.")
    b, h, lq, d = q.shape
    lk = k.shape[-2]
    return b, h, lq, lk, d


def _flatten_bh(x: torch.Tensor) -> torch.Tensor:
    # (B, H, L, D) -> (BH, L, D)
    return x.reshape(-1, x.shape[-2], x.shape[-1])


@triton.jit
def _ksum_fwd_kernel(
    K,
    KSUM,
    Lk: tl.constexpr,
    D: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    ksum = tl.zeros((BLOCK_D,), dtype=tl.float32)

    k_base = pid_bh * Lk * D
    for l0 in range(0, Lk, BLOCK_L):
        offs_l = l0 + tl.arange(0, BLOCK_L)
        k = tl.load(
            K + k_base + offs_l[:, None] * D + offs_d[None, :],
            mask=(offs_l[:, None] < Lk) & (offs_d[None, :] < D),
            other=0.0,
        ).to(tl.float32)
        ksum += tl.sum(k, axis=0)

    tl.store(KSUM + pid_bh * D + offs_d, ksum, mask=offs_d < D)


@triton.jit
def _kv_fwd_kernel(
    K,
    V,
    KV,
    Lk: tl.constexpr,
    D: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_base = pid_bh * Lk * D
    v_base = pid_bh * Lk * D
    for l0 in range(0, Lk, BLOCK_L):
        offs_l = l0 + tl.arange(0, BLOCK_L)

        k = tl.load(
            K + k_base + offs_l[:, None] * D + offs_m[None, :],
            mask=(offs_l[:, None] < Lk) & (offs_m[None, :] < D),
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            V + v_base + offs_l[:, None] * D + offs_n[None, :],
            mask=(offs_l[:, None] < Lk) & (offs_n[None, :] < D),
            other=0.0,
        ).to(tl.float32)

        acc += tl.dot(tl.trans(k), v)

    kv_base = pid_bh * D * D
    tl.store(
        KV + kv_base + offs_m[:, None] * D + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < D) & (offs_n[None, :] < D),
    )


_KV_FWD_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_L": 64, "BLOCK_M": 16, "BLOCK_N": 16}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_L": 128, "BLOCK_M": 16, "BLOCK_N": 16}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_L": 128, "BLOCK_M": 32, "BLOCK_N": 16}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_L": 128, "BLOCK_M": 16, "BLOCK_N": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_L": 128, "BLOCK_M": 32, "BLOCK_N": 32}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=_KV_FWD_AUTOTUNE_CONFIGS, key=["Lk", "D"])
@triton.jit
def _kv_fwd_kernel_at(
    K,
    V,
    KV,
    Lk: tl.constexpr,
    D: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    _kv_fwd_kernel(K, V, KV, Lk=Lk, D=D, BLOCK_L=BLOCK_L, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)


@triton.jit
def _out_fwd_kernel(
    Q,
    KV,
    KSUM,
    O,
    Lq: tl.constexpr,
    D: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    q_base = pid_bh * Lq * D
    kv_base = pid_bh * D * D
    ksum_base = pid_bh * D

    denom = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k0 in range(0, D, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        qk = tl.load(
            Q + q_base + offs_m[:, None] * D + offs_k[None, :],
            mask=(offs_m[:, None] < Lq) & (offs_k[None, :] < D),
            other=0.0,
        ).to(tl.float32)
        ksum = tl.load(KSUM + ksum_base + offs_k, mask=offs_k < D, other=0.0).to(tl.float32)
        denom += tl.sum(qk * ksum[None, :], axis=1)
    denom = denom + eps
    inv_denom = 1.0 / denom

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, D, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        qk = tl.load(
            Q + q_base + offs_m[:, None] * D + offs_k[None, :],
            mask=(offs_m[:, None] < Lq) & (offs_k[None, :] < D),
            other=0.0,
        ).to(tl.float32)
        kv = tl.load(
            KV + kv_base + offs_k[:, None] * D + offs_n[None, :],
            mask=(offs_k[:, None] < D) & (offs_n[None, :] < D),
            other=0.0,
        ).to(tl.float32)
        acc += tl.dot(qk, kv)

    acc = acc * inv_denom[:, None]
    tl.store(
        O + q_base + offs_m[:, None] * D + offs_n[None, :],
        acc.to(O.dtype.element_ty),
        mask=(offs_m[:, None] < Lq) & (offs_n[None, :] < D),
    )


_OUT_FWD_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=_OUT_FWD_AUTOTUNE_CONFIGS, key=["Lq", "D"])
@triton.jit
def _out_fwd_kernel_at(
    Q,
    KV,
    KSUM,
    O,
    Lq: tl.constexpr,
    D: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    _out_fwd_kernel(
        Q,
        KV,
        KSUM,
        O,
        Lq=Lq,
        D=D,
        eps=eps,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )


@triton.jit
def _dkv_bwd_kernel(
    Q,
    GO,
    KSUM,
    DKV,
    Lq: tl.constexpr,
    D: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_KSUM: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    q_base = pid_bh * Lq * D
    ksum_base = pid_bh * D

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for l0 in range(0, Lq, BLOCK_L):
        offs_l = l0 + tl.arange(0, BLOCK_L)

        denom = tl.zeros((BLOCK_L,), dtype=tl.float32)
        for k0 in range(0, D, BLOCK_KSUM):
            offs_k = k0 + tl.arange(0, BLOCK_KSUM)
            qk = tl.load(
                Q + q_base + offs_l[:, None] * D + offs_k[None, :],
                mask=(offs_l[:, None] < Lq) & (offs_k[None, :] < D),
                other=0.0,
            ).to(tl.float32)
            ksum = tl.load(KSUM + ksum_base + offs_k, mask=offs_k < D, other=0.0).to(tl.float32)
            denom += tl.sum(qk * ksum[None, :], axis=1)
        inv_denom = 1.0 / (denom + eps)

        q = tl.load(
            Q + q_base + offs_l[:, None] * D + offs_m[None, :],
            mask=(offs_l[:, None] < Lq) & (offs_m[None, :] < D),
            other=0.0,
        ).to(tl.float32)
        go = tl.load(
            GO + q_base + offs_l[:, None] * D + offs_n[None, :],
            mask=(offs_l[:, None] < Lq) & (offs_n[None, :] < D),
            other=0.0,
        ).to(tl.float32)
        go = go * inv_denom[:, None]

        acc += tl.dot(tl.trans(q), go)

    dkv_base = pid_bh * D * D
    tl.store(
        DKV + dkv_base + offs_m[:, None] * D + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < D) & (offs_n[None, :] < D),
    )


_DKV_BWD_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_L": 64, "BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_KSUM": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_L": 128, "BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_KSUM": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_L": 128, "BLOCK_M": 32, "BLOCK_N": 16, "BLOCK_KSUM": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_L": 128, "BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_KSUM": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_L": 128, "BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_KSUM": 32}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=_DKV_BWD_AUTOTUNE_CONFIGS, key=["Lq", "D"])
@triton.jit
def _dkv_bwd_kernel_at(
    Q,
    GO,
    KSUM,
    DKV,
    Lq: tl.constexpr,
    D: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_KSUM: tl.constexpr,
):
    _dkv_bwd_kernel(
        Q,
        GO,
        KSUM,
        DKV,
        Lq=Lq,
        D=D,
        eps=eps,
        BLOCK_L=BLOCK_L,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_KSUM=BLOCK_KSUM,
    )


@triton.jit
def _dq_and_a_bwd_kernel(
    Q,
    GO,
    KV,
    KSUM,
    DQ,
    A,
    Lq: tl.constexpr,
    D: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q_base = pid_bh * Lq * D
    kv_base = pid_bh * D * D
    ksum_base = pid_bh * D

    q = tl.load(
        Q + q_base + offs_m[:, None] * D + offs_d[None, :],
        mask=(offs_m[:, None] < Lq) & (offs_d[None, :] < D),
        other=0.0,
    ).to(tl.float32)
    go = tl.load(
        GO + q_base + offs_m[:, None] * D + offs_d[None, :],
        mask=(offs_m[:, None] < Lq) & (offs_d[None, :] < D),
        other=0.0,
    ).to(tl.float32)
    ksum = tl.load(KSUM + ksum_base + offs_d, mask=offs_d < D, other=0.0).to(tl.float32)

    denom = tl.sum(q * ksum[None, :], axis=1) + eps
    inv_denom = 1.0 / denom
    grad_n = go * inv_denom[:, None]

    offs_k = tl.arange(0, BLOCK_D)
    kv = tl.load(
        KV + kv_base + offs_d[:, None] * D + offs_k[None, :],
        mask=(offs_d[:, None] < D) & (offs_k[None, :] < D),
        other=0.0,
    ).to(tl.float32)

    # (BLOCK_M, D) @ (D, D)
    acc = tl.dot(grad_n, tl.trans(kv))

    # a = - ( (grad_n @ KV^T) · q ) / denom
    s = tl.sum(acc * q, axis=1)
    a = -s * inv_denom

    dq = acc + a[:, None] * ksum[None, :]

    tl.store(
        DQ + q_base + offs_m[:, None] * D + offs_d[None, :],
        dq.to(DQ.dtype.element_ty),
        mask=(offs_m[:, None] < Lq) & (offs_d[None, :] < D),
    )
    tl.store(A + pid_bh * Lq + offs_m, a, mask=offs_m < Lq)


_DQ_BWD_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 256}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=_DQ_BWD_AUTOTUNE_CONFIGS, key=["Lq", "D"])
@triton.jit
def _dq_and_a_bwd_kernel_at(
    Q,
    GO,
    KV,
    KSUM,
    DQ,
    A,
    Lq: tl.constexpr,
    D: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    _dq_and_a_bwd_kernel(
        Q,
        GO,
        KV,
        KSUM,
        DQ,
        A,
        Lq=Lq,
        D=D,
        eps=eps,
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
        BLOCK_K=BLOCK_K,
    )


@triton.jit
def _dksum_bwd_kernel(
    Q,
    A,
    DK,
    Lq: tl.constexpr,
    D: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    q_base = pid_bh * Lq * D
    a_base = pid_bh * Lq

    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for l0 in range(0, Lq, BLOCK_L):
        offs_l = l0 + tl.arange(0, BLOCK_L)
        q = tl.load(
            Q + q_base + offs_l[:, None] * D + offs_d[None, :],
            mask=(offs_l[:, None] < Lq) & (offs_d[None, :] < D),
            other=0.0,
        ).to(tl.float32)
        a = tl.load(A + a_base + offs_l, mask=offs_l < Lq, other=0.0).to(tl.float32)
        acc += tl.sum(q * a[:, None], axis=0)

    tl.store(DK + pid_bh * D + offs_d, acc, mask=offs_d < D)


@triton.jit
def _dkdv_bwd_kernel(
    K,
    V,
    DKV,
    DKSUM,
    DK,
    DV,
    Lk: tl.constexpr,
    D: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_l = tl.program_id(1)

    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_d = tl.arange(0, BLOCK_D)

    k_base = pid_bh * Lk * D
    dkv_base = pid_bh * D * D
    dksum_base = pid_bh * D

    k = tl.load(
        K + k_base + offs_l[:, None] * D + offs_d[None, :],
        mask=(offs_l[:, None] < Lk) & (offs_d[None, :] < D),
        other=0.0,
    ).to(tl.float32)
    v = tl.load(
        V + k_base + offs_l[:, None] * D + offs_d[None, :],
        mask=(offs_l[:, None] < Lk) & (offs_d[None, :] < D),
        other=0.0,
    ).to(tl.float32)

    dksum = tl.load(DKSUM + dksum_base + offs_d, mask=offs_d < D, other=0.0).to(tl.float32)

    dv = tl.zeros((BLOCK_L, BLOCK_D), dtype=tl.float32)
    dk = tl.zeros((BLOCK_L, BLOCK_D), dtype=tl.float32)

    for k0 in range(0, D, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        dkv_col = tl.load(
            DKV + dkv_base + offs_k[:, None] * D + offs_d[None, :],
            mask=(offs_k[:, None] < D) & (offs_d[None, :] < D),
            other=0.0,
        ).to(tl.float32)
        dkv_row = tl.load(
            DKV + dkv_base + offs_d[:, None] * D + offs_k[None, :],
            mask=(offs_d[:, None] < D) & (offs_k[None, :] < D),
            other=0.0,
        ).to(tl.float32)

        kk = tl.load(
            K + k_base + offs_l[:, None] * D + offs_k[None, :],
            mask=(offs_l[:, None] < Lk) & (offs_k[None, :] < D),
            other=0.0,
        ).to(tl.float32)
        vv = tl.load(
            V + k_base + offs_l[:, None] * D + offs_k[None, :],
            mask=(offs_l[:, None] < Lk) & (offs_k[None, :] < D),
            other=0.0,
        ).to(tl.float32)

        dv += tl.dot(kk, dkv_col)
        dk += tl.dot(vv, tl.trans(dkv_row))

    dk += dksum[None, :]

    tl.store(
        DV + k_base + offs_l[:, None] * D + offs_d[None, :],
        dv.to(DV.dtype.element_ty),
        mask=(offs_l[:, None] < Lk) & (offs_d[None, :] < D),
    )
    tl.store(
        DK + k_base + offs_l[:, None] * D + offs_d[None, :],
        dk.to(DK.dtype.element_ty),
        mask=(offs_l[:, None] < Lk) & (offs_d[None, :] < D),
    )


_DKDV_BWD_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_L": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_L": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_L": 256}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=_DKDV_BWD_AUTOTUNE_CONFIGS, key=["Lk", "D"])
@triton.jit
def _dkdv_bwd_kernel_at(
    K,
    V,
    DKV,
    DKSUM,
    DK,
    DV,
    Lk: tl.constexpr,
    D: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    _dkdv_bwd_kernel(
        K,
        V,
        DKV,
        DKSUM,
        DK,
        DV,
        Lk=Lk,
        D=D,
        BLOCK_L=BLOCK_L,
        BLOCK_D=BLOCK_D,
        BLOCK_K=BLOCK_K,
    )


class _TritonLinearAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, eps: float) -> torch.Tensor:
        b, h, lq, lk, d = _check_qkv_shapes(q, k, v)

        if d > 128:
            raise ValueError(f"Triton path supports head_dim<=128, got {d}.")

        if not (q.is_cuda and k.is_cuda and v.is_cuda):
            raise ValueError("Triton linear attention expects CUDA tensors.")
        if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
            raise ValueError("Triton linear attention expects contiguous q/k/v.")

        q3 = _flatten_bh(q)
        k3 = _flatten_bh(k)
        v3 = _flatten_bh(v)
        bh = b * h

        ksum = torch.empty((bh, d), device=q.device, dtype=torch.float32)
        kv = torch.empty((bh, d, d), device=q.device, dtype=torch.float32)

        grid_ksum = (bh, triton.cdiv(d, 128))
        _ksum_fwd_kernel[grid_ksum](  # type: ignore[invalid-argument-type]
            k3,
            ksum,
            Lk=lk,
            D=d,
            BLOCK_L=256,
            BLOCK_D=128,
        )

        grid_kv = (bh, triton.cdiv(d, 32), triton.cdiv(d, 32))
        grid_kv = lambda meta: (bh, triton.cdiv(d, meta["BLOCK_M"]), triton.cdiv(d, meta["BLOCK_N"]))
        _kv_fwd_kernel_at[grid_kv](  # type: ignore[invalid-argument-type]
            k3,
            v3,
            kv,
            Lk=lk,
            D=d,
        )

        o3 = torch.empty((bh, lq, d), device=q.device, dtype=q.dtype)
        grid_o = lambda meta: (bh, triton.cdiv(lq, meta["BLOCK_M"]), triton.cdiv(d, meta["BLOCK_N"]))
        _out_fwd_kernel_at[grid_o](  # type: ignore[invalid-argument-type]
            q3,
            kv,
            ksum,
            o3,
            Lq=lq,
            D=d,
            eps=eps,
        )

        ctx.save_for_backward(q3, k3, v3, kv, ksum)
        ctx.meta = (b, h, lq, lk, d, eps)
        return o3.reshape(b, h, lq, d)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        q3, k3, v3, kv, ksum = ctx.saved_tensors
        b, h, lq, lk, d, eps = ctx.meta
        bh = b * h

        go3 = _flatten_bh(grad_out.contiguous())

        dkv = torch.empty((bh, d, d), device=go3.device, dtype=torch.float32)
        grid_dkv = lambda meta: (bh, triton.cdiv(d, meta["BLOCK_M"]), triton.cdiv(d, meta["BLOCK_N"]))
        _dkv_bwd_kernel_at[grid_dkv](  # type: ignore[invalid-argument-type]
            q3,
            go3,
            ksum,
            dkv,
            Lq=lq,
            D=d,
            eps=eps,
        )

        dq3 = torch.empty((bh, lq, d), device=go3.device, dtype=q3.dtype)
        a = torch.empty((bh, lq), device=go3.device, dtype=torch.float32)
        grid_dq = lambda meta: (bh, triton.cdiv(lq, meta["BLOCK_M"]))
        _dq_and_a_bwd_kernel_at[grid_dq](  # type: ignore[invalid-argument-type]
            q3,
            go3,
            kv,
            ksum,
            dq3,
            a,
            Lq=lq,
            D=d,
            eps=eps,
            BLOCK_D=d,
            BLOCK_K=32,
        )

        dksum = torch.empty((bh, d), device=go3.device, dtype=torch.float32)
        grid_dksum = (bh, triton.cdiv(d, 128))
        _dksum_bwd_kernel[grid_dksum](  # type: ignore[invalid-argument-type]
            q3,
            a,
            dksum,
            Lq=lq,
            D=d,
            BLOCK_L=256,
            BLOCK_D=128,
        )

        dk3 = torch.empty((bh, lk, d), device=go3.device, dtype=k3.dtype)
        dv3 = torch.empty((bh, lk, d), device=go3.device, dtype=v3.dtype)
        grid_dkdv = lambda meta: (bh, triton.cdiv(lk, meta["BLOCK_L"]))
        _dkdv_bwd_kernel_at[grid_dkdv](  # type: ignore[invalid-argument-type]
            k3,
            v3,
            dkv,
            dksum,
            dk3,
            dv3,
            Lk=lk,
            D=d,
            BLOCK_D=d,
            BLOCK_K=32,
        )

        dq = dq3.reshape(b, h, lq, d)
        dk = dk3.reshape(b, h, lk, d)
        dv = dv3.reshape(b, h, lk, d)
        return dq, dk, dv, None


def linear_attention_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, eps: float) -> torch.Tensor:
    """Reference implementation in PyTorch.

    Parameters
    ----------
    q : torch.Tensor
            (B, H, Lq, D)
    k : torch.Tensor
            (B, H, Lk, D)
    v : torch.Tensor
            (B, H, Lk, D)
    eps : float
            Stabilizer.
    """
    _check_qkv_shapes(q, k, v)
    kvsum = k.transpose(-1, -2) @ v
    ksum = k.sum(dim=-2, keepdim=True)
    denom = eps + (q * ksum).sum(dim=-1, keepdim=True)
    return (q @ kvsum) / denom


def linear_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, eps: float) -> torch.Tensor:
    """Linear attention with Triton CUDA path and PyTorch fallback."""
    _check_qkv_shapes(q, k, v)
    if q.is_cuda and q.shape[-1] <= 128:
        return _TritonLinearAttention.apply(q, k, v, float(eps))
    return linear_attention_reference(q.float(), k.float(), v.float(), eps=eps).to(q.dtype)
