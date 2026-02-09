from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import distributed as dist
from torch.distributed.nn.functional import all_gather as functional_all_gather
from timm.layers import get_norm_layer

from .lejepa_aug import lejepa_loss as _lejepa_loss_fn

_SigmaMode = Literal["sigma_GN", "sigma_RGN"]
_TargetDistribution = Literal["rectified_lp_distribution", "lp_distribution"]
_ProjectionVectorsType = Literal["random", "torch_svd_and_random", "torch_svd_bottom_half_eigen_and_random"]


class LeJEPALoss(torch.nn.Module):
    """
    A callable LeJEPA objective wrapper.

    This class is a thin wrapper around :func:`src.stage1.self_supervised.lejepa_aug.lejepa_loss`
    so you can plug it into a custom training loop without relying on the (script-like) minimal example.

    Parameters
    ----------
    lam:
        Interpolation weight between invariance loss and SIGReg loss.
        Same meaning as in ``lejepa_aug.lejepa_loss``.
    lejepa_loss_type:
        ``None`` (default) means invariance + SIGReg, or pick one side via
        ``"invariance"`` / ``"sigreg"``.
    infonce_weight, infonce_type, infonce_temp, infonce_anchor:
        Optional InfoNCE regularizer knobs. See ``lejepa_aug.lejepa_loss``.
    """

    def __init__(
        self,
        lam: float = 0.02,
        *,
        lejepa_loss_type: str | None = None,
        infonce_weight: float | None = None,
        infonce_type: str = "all@multi_positive_infonce",
        infonce_temp: float | None = None,
        infonce_anchor: Literal["all", "global"] = "global",
    ) -> None:
        super().__init__()
        self.lam = float(lam)
        self.lejepa_loss_type = lejepa_loss_type
        self.infonce_weight = infonce_weight
        self.infonce_type = infonce_type
        self.infonce_temp = infonce_temp
        self.infonce_anchor = infonce_anchor

    def forward(
        self,
        global_emb: Tensor,
        local_emb: Tensor | None = None,
        *,
        infonce_weight: float | None = None,
        infonce_type: str | None = None,
        infonce_temp: float | None = None,
        infonce_anchor: Literal["all", "global"] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        loss, breakdowns = _lejepa_loss_fn(
            global_emb,
            local_emb,
            lam=self.lam,
            lejepa_loss_type=self.lejepa_loss_type,
            infonce_weight=self.infonce_weight if infonce_weight is None else infonce_weight,
            infonce_type=self.infonce_type if infonce_type is None else infonce_type,
            infonce_temp=self.infonce_temp if infonce_temp is None else infonce_temp,
            infonce_anchor=self.infonce_anchor if infonce_anchor is None else infonce_anchor,
        )
        return loss, dict(breakdowns)


def create_rectified_lpjepa_projector(
    input_dim: int,
    *,
    hidden_dim: int,
    output_dim: int,
    rectified: bool = True,
    norm_type: str = "batchnorm1d",
) -> torch.nn.Module:
    """
    Create a 3-layer MLP projector used by Rectified LpJEPA.

    This mirrors the "rectified_mlp" option in the upstream rectified_lpjepa implementation:
    the final ReLU enforces non-negativity, which is the main mechanism for inducing L0 sparsity.
    """

    layers: list[torch.nn.Module] = [
        torch.nn.Linear(input_dim, hidden_dim),
        get_norm_layer(norm_type)(hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, hidden_dim),
        get_norm_layer(norm_type)(hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, output_dim),
    ]
    if rectified:
        layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)


def _ddp_all_gather_concat(x: Tensor) -> Tensor:
    if dist.is_available() and dist.is_initialized():
        gathered = functional_all_gather(x)
        return torch.cat(list(gathered), dim=0)
    return x


def determine_sigma_for_lp_dist(p: float) -> float:
    """
    Return sigma such that GN_p(0, sigma) has unit variance (before rectification).

    This matches the upstream rectified_lpjepa implementation.
    """

    if p <= 0:
        raise ValueError(f"`p` must be > 0, got {p}.")
    return (math.gamma(1 / p) ** 0.5) / ((p ** (1 / p)) * (math.gamma(3 / p) ** 0.5))


def _rectified_gengaus_mean_var_unified(p: float, mu: float, sigma: float) -> tuple[float, float]:
    """
    Mean/variance for Y = ReLU(X), X ~ GN_p(mu, sigma).

    Uses mpmath for incomplete gamma functions to be robust over (p, mu, sigma).
    """

    try:
        import mpmath as mp
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("`mpmath` is required for `sigma_RGN` mode but is not installed.") from exc

    p_mp = mp.mpf(p)
    mu_mp = mp.mpf(mu)
    sigma_mp = mp.mpf(sigma)
    if sigma_mp <= 0:
        raise ValueError("sigma must be > 0")
    if p_mp <= 0:
        raise ValueError("p must be > 0")

    sgn = mp.sign(mu_mp)
    s1 = mp.mpf(1) / p_mp
    s2 = mp.mpf(2) / p_mp
    s3 = mp.mpf(3) / p_mp

    t = (abs(mu_mp) ** p_mp) / (p_mp * (sigma_mp**p_mp))
    g1 = mp.gamma(s1)

    lower1 = mp.gammainc(s1, 0, t)
    lower3 = mp.gammainc(s3, 0, t)
    upper2 = mp.gammainc(s2, t, mp.inf)

    a = (g1 + sgn * lower1) / g1
    b = upper2 / g1
    c = (mp.gamma(s3) + sgn * lower3) / g1

    p1 = p_mp ** (mp.mpf(1) / p_mp)
    p2 = p_mp ** (mp.mpf(2) / p_mp)

    ey = mp.mpf("0.5") * (mu_mp * a + p1 * sigma_mp * b)
    ey2 = mp.mpf("0.5") * (mu_mp**2 * a + 2 * mu_mp * p1 * sigma_mp * b + p2 * sigma_mp**2 * c)
    vary = ey2 - ey**2
    return float(ey), float(vary)


def choose_sigma_for_unit_var(
    p: float,
    mu: float,
    *,
    target_var: float = 1.0,
    rtol: float = 1e-10,
    max_iter: int = 2000,
) -> float:
    """
    Solve sigma>0 such that Var(ReLU(GN_p(mu, sigma))) == target_var via bisection.
    """

    try:
        import mpmath as mp
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("`mpmath` is required for `sigma_RGN` mode but is not installed.") from exc

    p_mp = mp.mpf(p)
    mu_mp = mp.mpf(mu)
    target_var_mp = mp.mpf(target_var)

    def f(sig: Any) -> float:
        return _rectified_gengaus_mean_var_unified(float(p_mp), float(mu_mp), float(sig))[1] - float(target_var_mp)

    lo = mp.mpf("1e-8")
    hi = mp.mpf("1.0")
    flo = mp.mpf(f(lo))
    fhi = mp.mpf(f(hi))

    k = 0
    while flo * fhi > 0 and k < 200:
        hi *= 2
        fhi = mp.mpf(f(hi))
        k += 1

    if flo * fhi > 0:
        raise RuntimeError("Failed to bracket a root for sigma. Try different initial range.")

    for _ in range(max_iter):
        mid = (lo + hi) / 2
        fmid = mp.mpf(f(mid))
        if abs(fmid) <= mp.mpf(rtol) * (1 + abs(target_var_mp)):
            return float(mid)
        if flo * fmid <= 0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid

    return float((lo + hi) / 2)


def _sample_lp_distribution(
    shape: tuple[int, int],
    *,
    p: float,
    loc: float,
    scale: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if p == 1.0:
        loc_t = torch.tensor(loc, device=device, dtype=dtype)
        scale_t = torch.tensor(scale, device=device, dtype=dtype)
        return torch.distributions.Laplace(loc=loc_t, scale=scale_t).sample(shape)
    if p == 2.0:
        return torch.tensor(loc, device=device, dtype=dtype) + torch.tensor(
            scale, device=device, dtype=dtype
        ) * torch.randn(shape, device=device, dtype=dtype)

    sign = torch.empty(shape, device=device, dtype=dtype).bernoulli_(0.5)
    sign = 2 * sign - 1
    gamma_dist = torch.distributions.Gamma(concentration=1.0 / p, rate=torch.tensor(1.0, device=device, dtype=dtype))
    g = gamma_dist.sample(shape).to(device=device, dtype=dtype)
    x = sign * (p * g).pow(1.0 / p)
    return torch.tensor(loc, device=device, dtype=dtype) + torch.tensor(scale, device=device, dtype=dtype) * x


def _sliced_wasserstein_distance(
    features: Tensor,
    *,
    projection_vectors: Tensor,
    target_distribution: _TargetDistribution,
    mean_shift_value: float,
    lp_norm_parameter: float,
    chosen_sigma: float,
) -> Tensor:
    bsz, dim = features.shape
    projected_features = features @ projection_vectors.T  # [B, P]

    target_samples = _sample_lp_distribution(
        (bsz, dim),
        p=lp_norm_parameter,
        loc=mean_shift_value,
        scale=chosen_sigma,
        device=features.device,
        dtype=features.dtype,
    )
    if target_distribution == "rectified_lp_distribution":
        target_samples = torch.relu(target_samples)

    projected_targets = target_samples @ projection_vectors.T  # [B, P]

    sorted_features, _ = torch.sort(projected_features, dim=0)
    sorted_targets, _ = torch.sort(projected_targets, dim=0)
    wasserstein_1d = torch.mean((sorted_features - sorted_targets).square(), dim=0)
    return torch.mean(wasserstein_1d)


def _generate_random_projections(num_projections: int, dim: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    projections = torch.randn((num_projections, dim), device=device, dtype=dtype)
    return projections / (projections.norm(dim=1, keepdim=True) + 1e-12)


def _generate_svd_projections(z: Tensor) -> Tensor:
    z32 = z.detach().float()
    z_centered = z32 - z32.mean(dim=0, keepdim=False)
    try:
        _, _, vt = torch.linalg.svd(z_centered, full_matrices=False)
        return vt.to(device=z.device, dtype=z.dtype)
    except Exception:
        cov = z_centered.T @ z_centered
        _, v = torch.linalg.eigh(cov)
        vt = v.T.flip(0)
        return vt.to(device=z.device, dtype=z.dtype)


def _get_projection_vectors(
    z1: Tensor,
    z2: Tensor,
    *,
    num_projections: int,
    projection_vectors_type: _ProjectionVectorsType,
) -> Tensor | list[Tensor]:
    dim = z1.shape[1]
    if projection_vectors_type == "random":
        return _generate_random_projections(num_projections, dim, device=z1.device, dtype=z1.dtype)

    vt1 = _generate_svd_projections(z1)
    vt2 = _generate_svd_projections(z2)

    if projection_vectors_type == "torch_svd_bottom_half_eigen_and_random":
        vt1 = vt1[vt1.size(0) // 2 :]
        vt2 = vt2[vt2.size(0) // 2 :]
    elif projection_vectors_type != "torch_svd_and_random":
        raise ValueError(f"Unsupported projection_vectors_type: {projection_vectors_type}")

    if vt1.size(0) >= num_projections:
        return [vt1[:num_projections], vt2[:num_projections]]

    n_rand = num_projections - vt1.size(0)
    rand = _generate_random_projections(n_rand, dim, device=z1.device, dtype=z1.dtype)
    return [torch.vstack([vt1, rand]), torch.vstack([vt2, rand])]


@dataclass(frozen=True)
class RectifiedLpJEPAMetrics:
    invariance_loss: Tensor
    rdmreg_loss: Tensor
    total_loss: Tensor


class RectifiedLpJEPALoss(torch.nn.Module):
    """
    Rectified LpJEPA objective as a callable module (invariance + RDMReg).

    Compared to LeJEPA (invariance + SIGReg), Rectified LpJEPA replaces SIGReg with a
    distribution matching term (RDMReg) implemented as a sliced Wasserstein distance between
    projected features and samples from a (rectified) generalized Gaussian prior.
    """

    def __init__(
        self,
        *,
        invariance_loss_weight: float = 25.0,
        rdm_reg_loss_weight: float = 125.0,
        target_distribution: _TargetDistribution = "rectified_lp_distribution",
        projection_vectors_type: _ProjectionVectorsType = "random",
        num_projections: int = 256,
        mean_shift_value: float = 0.0,
        lp_norm_parameter: float = 1.0,
        sigma_mode: _SigmaMode = "sigma_GN",
        chosen_sigma: float | None = None,
        sync_ddp_gather: bool = True,
    ) -> None:
        super().__init__()
        self.invariance_loss_weight = float(invariance_loss_weight)
        self.rdm_reg_loss_weight = float(rdm_reg_loss_weight)
        self.target_distribution = target_distribution
        self.projection_vectors_type = projection_vectors_type
        self.num_projections = int(num_projections)
        self.mean_shift_value = float(mean_shift_value)
        self.lp_norm_parameter = float(lp_norm_parameter)
        self.sigma_mode = sigma_mode
        self.sync_ddp_gather = bool(sync_ddp_gather)

        if chosen_sigma is not None:
            self.chosen_sigma = float(chosen_sigma)
        else:
            if self.sigma_mode == "sigma_GN":
                self.chosen_sigma = determine_sigma_for_lp_dist(self.lp_norm_parameter)
            elif self.sigma_mode == "sigma_RGN":
                self.chosen_sigma = choose_sigma_for_unit_var(self.lp_norm_parameter, self.mean_shift_value)
            else:
                raise ValueError(f"Invalid sigma_mode: {self.sigma_mode}")

    def forward(
        self,
        z1: Tensor,
        z2: Tensor,
        *,
        projection_vectors: Tensor | list[Tensor] | None = None,
    ) -> tuple[Tensor, RectifiedLpJEPAMetrics]:
        if z1.ndim != 2 or z2.ndim != 2:
            raise ValueError(f"Expected z1/z2 to be [B, D], got {z1.shape=} {z2.shape=}.")
        if z1.shape != z2.shape:
            raise ValueError(f"Expected z1 and z2 to have the same shape, got {z1.shape=} {z2.shape=}.")

        inv_loss = F.mse_loss(z1, z2)

        z1_reg = _ddp_all_gather_concat(z1) if self.sync_ddp_gather else z1
        z2_reg = _ddp_all_gather_concat(z2) if self.sync_ddp_gather else z2

        if projection_vectors is None:
            projection_vectors = _get_projection_vectors(
                z1_reg,
                z2_reg,
                num_projections=self.num_projections,
                projection_vectors_type=self.projection_vectors_type,
            )

        if isinstance(projection_vectors, list):
            if len(projection_vectors) != 2:
                raise ValueError("When `projection_vectors` is a list, it must have length 2.")
            reg1 = _sliced_wasserstein_distance(
                z1_reg,
                projection_vectors=projection_vectors[0],
                target_distribution=self.target_distribution,
                mean_shift_value=self.mean_shift_value,
                lp_norm_parameter=self.lp_norm_parameter,
                chosen_sigma=self.chosen_sigma,
            )
            reg2 = _sliced_wasserstein_distance(
                z2_reg,
                projection_vectors=projection_vectors[1],
                target_distribution=self.target_distribution,
                mean_shift_value=self.mean_shift_value,
                lp_norm_parameter=self.lp_norm_parameter,
                chosen_sigma=self.chosen_sigma,
            )
            rdmreg = (reg1 + reg2) / 2
        else:
            reg1 = _sliced_wasserstein_distance(
                z1_reg,
                projection_vectors=projection_vectors,
                target_distribution=self.target_distribution,
                mean_shift_value=self.mean_shift_value,
                lp_norm_parameter=self.lp_norm_parameter,
                chosen_sigma=self.chosen_sigma,
            )
            reg2 = _sliced_wasserstein_distance(
                z2_reg,
                projection_vectors=projection_vectors,
                target_distribution=self.target_distribution,
                mean_shift_value=self.mean_shift_value,
                lp_norm_parameter=self.lp_norm_parameter,
                chosen_sigma=self.chosen_sigma,
            )
            rdmreg = (reg1 + reg2) / 2

        total = self.invariance_loss_weight * inv_loss + self.rdm_reg_loss_weight * rdmreg
        metrics = RectifiedLpJEPAMetrics(invariance_loss=inv_loss, rdmreg_loss=rdmreg, total_loss=total)
        return total, metrics
