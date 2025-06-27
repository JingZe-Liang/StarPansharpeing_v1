# see https://github.com/leloykun/spectral_clip/blob/main/spectral_clip.py
# https://kexue.fm/archives/11059

from typing import Callable

import torch
from torch import Tensor


def block_matmul(
    P1: Tensor,
    Q1: Tensor,
    R1: Tensor,
    P2: Tensor,
    Q2: Tensor,
    R2: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Performs block matrix multiplication elements of the (linear) sub-algebra
    of matrices of the form:
        [P   Q]
        [Q.T R]
    where Q is a MxN matrix, and P and R are symmetric matrices of size MxM and NxN respectively.
    """
    P = P1 @ P2 + Q1 @ Q2.T
    Q = P1 @ Q2 + Q1 @ R2
    R = Q1.T @ Q2 + R1 @ R2
    return P, Q, R


# block type NS step for once
def ns_once(P, Q, R, a, b, c):
    P2, Q2, R2 = block_matmul(P, Q, R, P, Q, R)
    P4, Q4, R4 = block_matmul(P2, Q2, R2, P2, Q2, R2)
    I_P = a * torch.eye(P.shape[0], device=P.device, dtype=P.dtype)
    I_R = a * torch.eye(R.shape[0], device=P.device, dtype=R.dtype)
    Ppoly = I_P + b * P2 + c * P4
    Qpoly = b * Q2 + c * Q4
    Rpoly = I_R + b * R2 + c * R4
    return block_matmul(P, Q, R, Ppoly, Qpoly, Rpoly)


def orthogonalize_blockwise(
    W: torch.Tensor,
    ortho_dtype=torch.float32,
    num_ns_steps: int = 5,
):
    """Orthogonalize a matrix via 5th order blockwise Newton-Schulz iteration.

    Tighter spectral norm bound:
    => Matrices of the form [I_m, W; W.T, I_n] have spectral norm 1 + ||W||_2
    => We can estimate ||W||_2 via power iteration or Gram iteration.
    => However, we can also use the fact that ||W||_2 <= ||W||_F and the latter is much cheaper to compute.
    """
    NS_COEFFS = [
        (3.5318, -4.7911, 1.9388),
        (3.3274, -4.0557, 1.5782),
        (3.0809, -3.5160, 1.3464),
        (2.7476, -2.8484, 1.0775),
        (2.2948, -2.0951, 0.7895),
        (2.1535, -1.8338, 0.6869),
    ]

    orig_dtype = W.dtype
    m, n = W.shape
    I_m, I_n = torch.eye(m).to(W), torch.eye(n).to(W)
    # norm = 1 + _power_iterate(W, jax.random.PRNGKey(0), num_iters=16)[1]
    norm = 1 + torch.linalg.norm(W)
    P = (I_m / (norm + 1e-12)).to(ortho_dtype)
    Q = (W / (norm + 1e-12)).to(ortho_dtype)
    R = (I_n / (norm + 1e-12)).to(ortho_dtype)
    for a, b, c in NS_COEFFS[:num_ns_steps]:
        P, Q, R = ns_once(P, Q, R, a=a, b=b, c=c)
    return P.to(orig_dtype), Q.to(orig_dtype), R.to(orig_dtype)


def _spectral_hardcap_blockwise(
    W: torch.Tensor, sigma_max=1.0, ortho_dtype=torch.float32, num_ns_steps=5
):
    def _spectral_hardcap_blockwise_util(W: torch.Tensor):
        if transpose := W.shape[0] > W.shape[1]:
            W = W.T
        orig_dtype = W.dtype
        W = W.to(ortho_dtype)
        # _, Q, R = orthogonalize_blockwise(W, ortho_dtype, num_ns_steps)
        # result = Q + W @ R
        P, Q, _ = orthogonalize_blockwise(W, ortho_dtype, num_ns_steps)
        result = Q + P @ W
        if transpose:
            result = result.T
        return result.to(orig_dtype)

    return sigma_max * _spectral_hardcap_blockwise_util(W / sigma_max)


def _spectral_hardcap_fully_materialized(
    W: torch.Tensor,
    ns_step_fn: Callable,
    sigma_max: float = 1.0,
    ortho_dtype=torch.float32,
):
    def _spectral_clip_fully_materialized_utl(W: torch.Tensor):
        if transpose := W.shape[0] > W.shape[1]:
            W = W.T
        orig_dtype = W.dtype
        W = W.to(ortho_dtype)
        m, n = W.shape
        I_m, I_n = torch.eye(m, dtype=W.dtype), torch.eye(n, dtype=W.dtype)
        # H = jnp.block([[I_m, W], [W.T, I_n]])
        H = torch.cat([torch.cat([I_m, W], dim=1), torch.cat([W.T, I_n], dim=1)], dim=0)
        # OH = orthogonalize(H, num_ns_steps)
        OH = ns_step_fn(H)
        # Q, R = OH[:m, m:], OH[m:, m:]
        # W_clipped = Q + W @ R
        P, Q = OH[:m, :m], OH[:m, m:]
        result = Q + P @ W
        if transpose:
            result = result.T
        return result.astype(orig_dtype)

    return sigma_max * _spectral_clip_fully_materialized_utl(W / sigma_max)


def _spectral_hardcap_nested(
    W: torch.Tensor,
    ns_step_fn: Callable,
    sigma_max: float = 1.0,
    ortho_dtype=torch.float32,
):
    def _spectral_hardcap_util(W: torch.Tensor):
        if transpose := W.shape[0] > W.shape[1]:
            W = W.T
        orig_dtype = W.dtype
        W = W.to(ortho_dtype)
        OW = ns_step_fn(W)
        aW = OW - W
        result = ((OW + W) - aW @ ns_step_fn(aW).T @ OW) / 2.0
        if transpose:
            result = result.T
        return result.astype(orig_dtype)

    return sigma_max * _spectral_hardcap_util(W / sigma_max)


def _spectral_clip(
    W: torch.Tensor,
    ns_step_fn: Callable,
    sigma_min: float = -1.0,
    sigma_max: float = 1.0,
    ortho_dtype=torch.float32,
):
    if flip := W.shape[0] > W.shape[1]:
        W = W.T
    orig_dtype = W.dtype
    W = W.to(ortho_dtype)
    OW = ns_step_fn(W)
    result = (
        (1 / 2)
        * (
            (sigma_min + sigma_max) * torch.eye(W.shape[0])
            + (sigma_min * OW - W) @ ns_step_fn(sigma_min * OW - W).T
            - (sigma_max * OW - W) @ ns_step_fn(sigma_max * OW - W).T
        )
        @ OW
    )
    if flip:
        result = result.T
    return result.astype(orig_dtype)


def _spectral_relu(
    W: torch.Tensor,
    ns_step_fn: Callable,
    sigma_min: float = 1.0,
    ortho_dtype=torch.float32,
):
    def _spectral_relu_util(W: torch.Tensor):
        if flip := W.shape[0] > W.shape[1]:
            W = W.T
        orig_dtype = W.dtype
        W = W.to(ortho_dtype)
        OW = ns_step_fn(W)
        aW = OW - W
        result = (1 / 2) * (OW + W + aW @ ns_step_fn(aW).T @ OW)
        if flip:
            result = result.T
        return result.astype(orig_dtype)

    return sigma_min * _spectral_relu_util(W / sigma_min)


def batch_project(M: torch.Tensor, project_fn: Callable) -> torch.Tensor:
    """Batch project tensors of shape [..., fanout, fanin]. Taken from Modula library."""
    matrix_shape = M.shape[-2:]
    M_flattened = M.reshape((-1,) + matrix_shape)
    M_projected = torch.vmap(project_fn)(M_flattened)
    return M_projected.reshape(M.shape) / len(M_flattened)


# *==============================================================
# * Spectral clipping interfaces
# *==============================================================


def spectral_hardcap(
    W: torch.Tensor, sigma_max=1.0, ortho_dtype=torch.float32, num_ns_steps=5
):
    # return batch_project(W, lambda x: _spectral_hardcap_fully_materialized(x, sigma_max=sigma_max, ortho_dtype=ortho_dtype, num_ns_steps=num_ns_steps))
    # return batch_project(W, lambda x: _spectral_hardcap_nested(x, sigma_max=sigma_max, ortho_dtype=ortho_dtype, num_ns_steps=num_ns_steps))
    # return batch_project(W, lambda x: _spectral_clip(x, sigma_min=0., sigma_max=sigma_max))
    return batch_project(
        W,
        lambda x: _spectral_hardcap_blockwise(
            x, sigma_max=sigma_max, ortho_dtype=ortho_dtype, num_ns_steps=num_ns_steps
        ),
    )


def spectral_clip(
    W: torch.Tensor,
    ns_fn_step: Callable,
    sigma_min: float = -1.0,
    sigma_max: float = 1.0,
    ortho_dtype=torch.float32,
):
    return batch_project(
        W,
        lambda x: _spectral_clip(
            x,
            ns_step_fn=ns_fn_step,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            ortho_dtype=ortho_dtype,
        ),
    )


def spectral_relu(
    W: torch.Tensor, sigma_min: float = 1.0, ortho_dtype=torch.float32, num_ns_steps=5
):
    return batch_project(
        W,
        lambda x: _spectral_relu(
            x, sigma_min=sigma_min, ortho_dtype=ortho_dtype, num_ns_steps=num_ns_steps
        ),
    )


def spectral_hardcap_v2(
    W: torch.Tensor,
    sigma_max: float = 1.0,
):
    def _spectral_hardcap_v2_util(W: torch.Tensor):
        if transpose := W.shape[0] > W.shape[1]:
            W = W.T

        OW = msign(W)
        result = 0.5 * (
            OW
            + W
            - msign(torch.eye(W.shape[0], device=W.device, dtype=W.dtype) - OW @ W.T)
            @ (OW - W)
        )
        if transpose:
            result = result.T
        return result

    return sigma_max * _spectral_hardcap_v2_util(W / sigma_max)


def spectral_clip_v2(
    W: torch.Tensor,
    sigma_min: float = -1.0,
    sigma_max: float = 1.0,
):
    if transpose := W.shape[0] > W.shape[1]:
        W = W.T
    OW = msign(W)
    result = (1 / 2) * (
        (sigma_min + sigma_max) * OW
        + msign(
            sigma_min * torch.eye(W.shape[0], device=W.device, dtype=W.dtype) - OW @ W.T
        )
        @ (sigma_min * OW - W)
        - msign(
            sigma_max * torch.eye(W.shape[0], device=W.device, dtype=W.dtype) - OW @ W.T
        )
        @ (sigma_max * OW - W)
    )
    if transpose:
        result = result.T
    return result


# * --- other not NS M clipping --- #


# Direct SVD clipping
def svd_clip(A, clip_val=1.0):
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    S_clipped = torch.clamp(S, max=clip_val)
    return (U * S_clipped) @ Vh


# Higham's cubic iteration (inverse-free)
def higham_sign(X, max_iter=100, tol=1e-9):
    norm_X = torch.linalg.norm(X, ord=2)
    X = X / norm_X
    for _ in range(max_iter):
        X_prev = X
        X = 1.5 * X - 0.5 * (X @ X @ X)
        if torch.norm(X - X_prev) / torch.norm(X) < tol:
            break
    return X


# Construct symmetric block matrix
def make_block_matrix(A):
    m, n = A.shape
    top = torch.cat([torch.zeros((m, m)).to(A), A], dim=1)
    bottom = torch.cat([A.T, torch.zeros((n, n)).to(A)], dim=1)
    return torch.cat([top, bottom], dim=0)


# Perform spectral clipping using f(x) = 0.5((1+x)sign(1+x) - (1-x)sign(1-x))
def higham_spectral_clip_via_sign(A):
    S = make_block_matrix(A)
    I = torch.eye(S.shape[0]).to(A)
    sign_plus = higham_sign(I + S)
    sign_minus = higham_sign(I - S)
    fS = 0.5 * ((I + S) @ sign_plus - (I - S) @ sign_minus)
    m, n = A.shape
    return fS[:m, m : m + n]


# * --- Mclip3 Non-nested --- #


# ns step that named using in Su's blog
def msign(x, steps=7, eps=1e-20):
    """The coefficients come from https://kexue.fm/archives/10996"""
    abc = [
        (8.287212018145622, -23.59588651909882, 17.300387312530923),
        (4.107059111542197, -2.9478499167379084, 0.54484310829266),
        (3.9486908534822938, -2.908902115962947, 0.5518191394370131),
        (3.3184196573706055, -2.488488024314878, 0.5100489401237208),
        (2.3006520199548186, -1.6689039845747518, 0.4188073119525678),
        (1.8913014077874002, -1.2679958271945908, 0.37680408948524996),
        (1.875, -1.25, 0.375),
    ]
    y = x.mT if x.shape[-2] > x.shape[-1] else x
    y = y * torch.rsqrt((y**2).sum(dim=[-2, -1], keepdims=True) + eps)
    for a, b, c in abc[:steps] + max(steps - 7, 0) * abc[-1:]:
        a, b, c = a / 1.01, b / 1.01**3, c / 1.01**5
        y = a * y + (b * (u := y @ y.mT) + c * u @ u) @ y
    return y.mT if x.shape[-2] > x.shape[-1] else y


def mclip3(m):
    """3rd version (3 non-nested msign)"""
    ms1 = msign(m)
    ms2 = msign(m.mT @ m + torch.eye(m.shape[-1]).to(m))
    ms3 = msign(m.mT @ m - torch.eye(m.shape[-1]).to(m))
    return ((ms1 + m) @ ms2 + (ms1 - m) @ ms3) / 2


# * --- MClipping Testing --- #


def __test():
    # init G to have a given spectral for this test
    m, n = 1024, 4 * 1024
    G = torch.rand((m, n)).cuda()

    U, S, Vt = torch.linalg.svd(G, full_matrices=False)
    S = torch.cat([torch.linspace(1, 1000, 128), torch.linspace(0, 1, 896)]).cuda()
    S = torch.sort(S, descending=True).values
    G = U @ torch.diag(S) @ Vt  # matrixes with large spectral norm

    print(f"G shape: {G.shape}, max sigular value: {S.max()}")

    # test spectral hardcap
    # 1. SVD
    G_svd = U @ torch.diag(S.clip(0, 1)) @ Vt
    S_svd = torch.linalg.svdvals(G_svd)
    print(
        f"SVD max sigular value: {S_svd.max()}, mean abs error: {torch.abs(S_svd - S.clip(0, 1)).mean()}"
    )

    # 2. spectral hardcap v1
    G_clip_hardcap = spectral_hardcap(G, ortho_dtype=torch.bfloat16, num_ns_steps=6)
    S_clip_hardcap = torch.linalg.svdvals(G_clip_hardcap.float())
    print(
        f"G_clip_hard_cap (nested version) max sigular value: {S_clip_hardcap.max()}, mean abs error: {torch.abs(S_clip_hardcap - S.clip(0, 1)).mean()}"
    )

    # 3. Spectral clipping non-nested version 3 steps
    G_clip_mclip3 = mclip3(G.to(torch.bfloat16))
    mclip3_S = torch.linalg.svdvals(G_clip_mclip3.float())
    print(
        f"G_clip_mclip3 max sigular value: {mclip3_S.max()}, mean abs error: {torch.abs(mclip3_S - S.clip(0, 1)).mean()}"
    )

    # # 4. Spectral hardcap v2
    G_clip_v2 = spectral_hardcap_v2(G, sigma_max=1.0)
    S_clip_v2 = torch.linalg.svdvals(G_clip_v2.float())
    print(
        f"G_clip_v2 max sigular value: {S_clip_v2.max()}, mean abs error: {torch.abs(S_clip_v2 - S.clip(0, 1)).mean()}"
    )

    # 5. Spectral clip v2
    # G_clip_svd = spectral_clip_v2(G, sigma_min=-1.0, sigma_max=1.0)
    # S_clip_svd = torch.linalg.svdvals(G_clip_svd.float())
    # print(
    #     f"G_clip_svd max sigular value: {S_clip_svd.max()}, mean abs error: {torch.abs(S_clip_svd - S.clip(0, 1)).mean()}"
    # )

    # 6. higham_sign
    # G_clip_higham = higham_spectral_clip_via_sign(G)
    # S_clip_higham = torch.linalg.svdvals(G_clip_higham.float())
    # print(
    #     f"G_clip_higham max sigular value: {S_clip_higham.max()}, mean abs error: {torch.abs(S_clip_higham - S.clip(0, 1)).mean()}"
    # )


if __name__ == "__main__":
    __test()
