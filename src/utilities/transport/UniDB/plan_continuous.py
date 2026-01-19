from __future__ import annotations

import math
from dataclasses import dataclass
import inspect
from typing import Callable, Literal

import torch

Tensor = torch.Tensor
PredictionType = Literal["epsilon", "x0", "v"]
SigmaSchedule = Literal["none", "quadratic"]
LossNorm = Literal["l1", "l2"]
LossWeighting = Literal["none", "inv"]
UniDBLossType = Literal["epsilon", "x0", "v", "mean"]


def _as_1d_time(t01: Tensor) -> Tensor:
    if t01.ndim == 0:
        return t01[None]
    return t01.reshape(-1)


def _broadcast_time_like(t01: Tensor, x: Tensor) -> Tensor:
    t01 = _as_1d_time(t01).to(device=x.device, dtype=x.dtype)
    return t01.view(-1, *([1] * (x.ndim - 1)))


def _interp_1d(values: Tensor, index: Tensor) -> Tensor:
    if values.ndim != 1:
        raise ValueError("values must be 1D")

    index = _as_1d_time(index).to(device=values.device, dtype=values.dtype)
    max_index = values.shape[0] - 1
    index = index.clamp(0, max_index)

    i0 = torch.floor(index).to(torch.int64)
    i1 = (i0 + 1).clamp(max=max_index)
    w = (index - i0.to(index.dtype)).to(values.dtype)

    v0 = values[i0]
    v1 = values[i1]
    return v0 + (v1 - v0) * w


def _maybe_add_model_kwargs(model: Callable[..., Tensor], model_kwargs: dict, **extra_kwargs: Tensor) -> dict:
    """Add extra kwargs only if the model accepts them."""
    signature = None
    try:
        signature = inspect.signature(model)  # __call__
    except (TypeError, ValueError):
        try:
            signature = inspect.signature(model.forward)  # type: ignore[attr-defined]
        except (TypeError, ValueError, AttributeError):
            signature = None

    if signature is None:
        return model_kwargs

    parameters = signature.parameters
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    if accepts_kwargs:
        return {**model_kwargs, **extra_kwargs}

    filtered = dict(model_kwargs)
    for k, v in extra_kwargs.items():
        if k in parameters:
            filtered[k] = v
    return filtered


class UniDBContinuous:
    """
    Continuous-time (t in [0, 1]) UniDB-GOU utilities.

    This module provides:
    - Forward marginal sampling x_t = m(t) x0 + (1-m(t)) mu + sigma(t) eps
    - Reparameterizations between epsilon / x0 / velocity v
    - Losses for the three parameterizations

    Parameters
    ----------
    lambda_square : float
        The lambda squared parameter for the diffusion process.
    gamma : float
        The gamma parameter for the drift adjustment.
    T : int, default=100
        Number of discrete timesteps.
    schedule : str, default="cosine"
        The schedule type for theta ("constant", "linear", or "cosine").
    eps : float, default=0.01
        The epsilon parameter for the minimum noise level.
    prediction : PredictionType, default="epsilon"
        The prediction type ("epsilon", "x0", or "v").
    loss_type : UniDBLossType, default="mean"
        The loss type ("epsilon", "x0", "v", or "mean").
    norm : LossNorm, default="l1"
        The loss norm ("l1" or "l2").
    weighting : LossWeighting, default="inv"
        The loss weighting ("none" or "inv").
    """

    def __init__(
        self,
        lambda_square: float,
        gamma: float,
        T: int = 100,
        schedule: str = "cosine",
        eps: float = 0.01,
        *,
        prediction: PredictionType = "epsilon",
        loss_type: UniDBLossType = "mean",
        norm: LossNorm = "l1",
        weighting: LossWeighting = "inv",
    ):
        if T <= 0:
            raise ValueError("T must be positive")
        if eps <= 0.0 or eps >= 1.0:
            raise ValueError("eps must be in (0, 1)")

        lambda_square = lambda_square / 255.0 if lambda_square >= 1.0 else lambda_square
        self.lambda_square = float(lambda_square)
        self.gamma = float(gamma)
        self.T = int(T)
        self.schedule = str(schedule)
        self.eps = float(eps)

        self.prediction = prediction
        self.loss_type = loss_type
        self.loss_norm = norm
        self.loss_weighting = weighting

        thetas = self._theta_schedule(self.T, self.schedule)
        thetas_cumsum = torch.cumsum(thetas, dim=0) - thetas[0]

        self._thetas = thetas
        self._thetas_cumsum = thetas_cumsum
        self._dt = (-1.0 / thetas_cumsum[-1] * math.log(self.eps)).to(torch.float32)

    def m(self, t01: Tensor, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Continuous counterpart of the discrete `m(t)` (coefficient of x0 in forward mean)."""
        return self._m(t01, device=device, dtype=dtype)

    def sigma(self, t01: Tensor, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Continuous counterpart of the discrete `f_sigma(t)` (forward marginal std)."""
        return self._sigma(t01, device=device, dtype=dtype)

    def sample_xt(
        self,
        x0: Tensor,
        mu: Tensor,
        t01: Tensor,
        *,
        eps: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Sample x_t from the forward marginal, returning (x_t, eps)."""
        return self._sample_xt(x0, mu, t01, eps=eps, generator=generator)

    def epsilon_from_xt_x0(self, xt: Tensor, x0: Tensor, mu: Tensor, t01: Tensor) -> Tensor:
        """Recover epsilon from (x_t, x0)."""
        return self._epsilon_from_xt_x0(xt, x0, mu, t01)

    def x0_from_xt_epsilon(self, xt: Tensor, mu: Tensor, t01: Tensor, eps: Tensor) -> Tensor:
        """Recover x0 from (x_t, epsilon)."""
        return self._x0_from_xt_epsilon(xt, mu, t01, eps)

    def velocity_target(self, x0: Tensor, mu: Tensor, t01: Tensor, eps: Tensor) -> Tensor:
        """Velocity target v corresponding to the UniDB marginal path."""
        return self._velocity_target(x0, mu, t01, eps)

    def velocity_from_epsilon(self, xt: Tensor, mu: Tensor, t01: Tensor, eps: Tensor) -> Tensor:
        """Compute velocity from epsilon and x_t."""
        return self._velocity_from_epsilon(xt, mu, t01, eps)

    def reverse_mean_theta_adjacent(self, xt: Tensor, mu: Tensor, t_index: int | Tensor, eps_hat: Tensor) -> Tensor:
        """Adjacent mean update (theta-based), matching the discrete UniDB implementation."""
        return self._reverse_mean_theta_adjacent(xt, mu, t_index, eps_hat)

    def reverse_mean_gamma_adjacent(self, xt: Tensor, x0: Tensor, mu: Tensor, t_index: int | Tensor) -> Tensor:
        """Adjacent mean update (gamma-based), matching the discrete UniDB implementation."""
        return self._reverse_mean_gamma_adjacent(xt, x0, mu, t_index)

    @staticmethod
    def _theta_schedule(timesteps: int, schedule: str) -> Tensor:
        if schedule == "constant":
            return torch.ones(timesteps + 1, dtype=torch.float32)

        if schedule == "linear":
            scale = 1000 / (timesteps + 1)
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            return torch.linspace(beta_start, beta_end, timesteps + 1, dtype=torch.float32)

        if schedule != "cosine":
            raise ValueError(f"Unknown schedule: {schedule}")

        s = 0.008
        t = timesteps + 2
        steps = t + 1
        x = torch.linspace(0, t, steps, dtype=torch.float32)
        alphas_cumprod = torch.cos(((x / t) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - alphas_cumprod[1:-1]
        if betas.shape[0] != timesteps + 1:
            raise RuntimeError("cosine schedule shape mismatch")
        return betas

    def _s_index(self, t01: Tensor, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        t01 = _as_1d_time(t01).to(device=device, dtype=dtype).clamp(0.0, 1.0)
        return t01 * float(self.T)

    def _theta(self, t01: Tensor, *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> Tensor:
        device = self._thetas.device if device is None else device
        dtype = self._thetas.dtype if dtype is None else dtype
        s = self._s_index(t01, device=device, dtype=dtype)
        return _interp_1d(self._thetas.to(device=device, dtype=dtype), s)

    def _theta_cumsum(
        self, t01: Tensor, *, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> Tensor:
        device = self._thetas_cumsum.device if device is None else device
        dtype = self._thetas_cumsum.dtype if dtype is None else dtype
        s = self._s_index(t01, device=device, dtype=dtype)
        return _interp_1d(self._thetas_cumsum.to(device=device, dtype=dtype), s)

    def _bar_theta(self, t01: Tensor, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        dt = self._dt.to(device=device, dtype=dtype)
        return self._theta_cumsum(t01, device=device, dtype=dtype) * dt

    def _bar_theta_total(self, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        dt = self._dt.to(device=device, dtype=dtype)
        return self._thetas_cumsum[-1].to(device=device, dtype=dtype) * dt

    def _bar_theta_prime(self, t01: Tensor, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        theta_t = self._theta(t01, device=device, dtype=dtype)
        dt = self._dt.to(device=device, dtype=dtype)
        return theta_t * float(self.T) * dt

    def _sigma_bar_sq(self, t01: Tensor, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        b = self._bar_theta(t01, device=device, dtype=dtype)
        lambda_sq = torch.tensor(self.lambda_square, device=device, dtype=dtype)
        return (lambda_sq**2) * (1.0 - torch.exp(-2.0 * b))

    def _sigma_bar_sq_t_T(self, t01: Tensor, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        b_total = self._bar_theta_total(device=device, dtype=dtype)
        b = self._bar_theta(t01, device=device, dtype=dtype)
        btT = b_total - b
        lambda_sq = torch.tensor(self.lambda_square, device=device, dtype=dtype)
        return (lambda_sq**2) * (1.0 - torch.exp(-2.0 * btT))

    def _sigma(self, t01: Tensor, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        sb_sq = self._sigma_bar_sq(t01, device=device, dtype=dtype).clamp_min(0.0)
        stT_sq = self._sigma_bar_sq_t_T(t01, device=device, dtype=dtype).clamp_min(0.0)
        sT_sq = self._sigma_bar_sq(torch.tensor(1.0, device=device, dtype=dtype), device=device, dtype=dtype).clamp_min(
            1e-20
        )
        return torch.sqrt(sb_sq) * torch.sqrt(stT_sq) / torch.sqrt(sT_sq)

    def _sigma_prime(self, t01: Tensor, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        b = self._bar_theta(t01, device=device, dtype=dtype)
        b_total = self._bar_theta_total(device=device, dtype=dtype)
        btT = b_total - b
        b_prime = self._bar_theta_prime(t01, device=device, dtype=dtype)

        lambda_sq = torch.tensor(self.lambda_square, device=device, dtype=dtype)
        sb_sq = (lambda_sq**2) * (1.0 - torch.exp(-2.0 * b))
        stT_sq = (lambda_sq**2) * (1.0 - torch.exp(-2.0 * btT))
        sT_sq = (lambda_sq**2) * (1.0 - torch.exp(-2.0 * b_total)).clamp_min(1e-20)

        sb_sq_prime = 2.0 * (lambda_sq**2) * torch.exp(-2.0 * b) * b_prime
        stT_sq_prime = -2.0 * (lambda_sq**2) * torch.exp(-2.0 * btT) * b_prime

        sb = torch.sqrt(sb_sq.clamp_min(1e-20))
        stT = torch.sqrt(stT_sq.clamp_min(1e-20))
        sT = torch.sqrt(sT_sq)

        sb_prime = 0.5 * sb_sq_prime / sb
        stT_prime = 0.5 * stT_sq_prime / stT
        return (sb_prime * stT + sb * stT_prime) / sT

    def _diffusion(self, t01: Tensor, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        theta_t = self._theta(t01, device=device, dtype=dtype)
        lambda_sq = torch.tensor(self.lambda_square, device=device, dtype=dtype)
        return torch.sqrt((lambda_sq**2) * 2.0 * theta_t)

    def _m(self, t01: Tensor, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        b = self._bar_theta(t01, device=device, dtype=dtype)
        s_t_T = self._sigma_bar_sq_t_T(t01, device=device, dtype=dtype)
        s_T = self._sigma_bar_sq(torch.tensor(1.0, device=device, dtype=dtype), device=device, dtype=dtype)
        gamma = torch.tensor(self.gamma, device=device, dtype=dtype)
        denom = (1.0 + gamma * s_T).clamp_min(1e-20)
        return torch.exp(-b) * (1.0 + gamma * s_t_T) / denom

    def _m_prime(self, t01: Tensor, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        b = self._bar_theta(t01, device=device, dtype=dtype)
        b_prime = self._bar_theta_prime(t01, device=device, dtype=dtype)

        b_total = self._bar_theta_total(device=device, dtype=dtype)
        btT = b_total - b

        lambda_sq = torch.tensor(self.lambda_square, device=device, dtype=dtype)
        s_t_T = (lambda_sq**2) * (1.0 - torch.exp(-2.0 * btT))
        s_t_T_prime = -2.0 * (lambda_sq**2) * torch.exp(-2.0 * btT) * b_prime

        s_T = self._sigma_bar_sq(torch.tensor(1.0, device=device, dtype=dtype), device=device, dtype=dtype)
        gamma = torch.tensor(self.gamma, device=device, dtype=dtype)
        denom = (1.0 + gamma * s_T).clamp_min(1e-20)

        m = torch.exp(-b) * (1.0 + gamma * s_t_T) / denom
        return m * (-b_prime) + torch.exp(-b) * (gamma * s_t_T_prime) / denom

    def _sample_xt(
        self,
        x0: Tensor,
        mu: Tensor,
        t01: Tensor,
        *,
        eps: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        if eps is None:
            if generator is None:
                eps = torch.randn_like(x0)
            else:
                eps = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=generator)

        m = _broadcast_time_like(self.m(t01, device=x0.device, dtype=x0.dtype), x0)
        sigma = _broadcast_time_like(self.sigma(t01, device=x0.device, dtype=x0.dtype), x0)
        xt = m * x0 + (1.0 - m) * mu + sigma * eps
        return xt, eps

    def _epsilon_from_xt_x0(self, xt: Tensor, x0: Tensor, mu: Tensor, t01: Tensor) -> Tensor:
        m = _broadcast_time_like(self._m(t01, device=xt.device, dtype=xt.dtype), xt)
        sigma = _broadcast_time_like(self._sigma(t01, device=xt.device, dtype=xt.dtype), xt).clamp_min(1e-20)
        return (xt - m * x0 - (1.0 - m) * mu) / sigma

    def _x0_from_xt_epsilon(self, xt: Tensor, mu: Tensor, t01: Tensor, eps: Tensor) -> Tensor:
        m = _broadcast_time_like(self._m(t01, device=xt.device, dtype=xt.dtype), xt).clamp_min(1e-20)
        sigma = _broadcast_time_like(self._sigma(t01, device=xt.device, dtype=xt.dtype), xt)
        return (xt - (1.0 - m) * mu - sigma * eps) / m

    def _velocity_target(self, x0: Tensor, mu: Tensor, t01: Tensor, eps: Tensor) -> Tensor:
        m_prime = _broadcast_time_like(self._m_prime(t01, device=x0.device, dtype=x0.dtype), x0)
        sigma_prime = _broadcast_time_like(self._sigma_prime(t01, device=x0.device, dtype=x0.dtype), x0)
        return m_prime * (x0 - mu) + sigma_prime * eps

    def _A_B(self, t01: Tensor, *, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        m = self._m(t01, device=device, dtype=dtype).clamp_min(1e-20)
        m_prime = self._m_prime(t01, device=device, dtype=dtype)
        sigma = self._sigma(t01, device=device, dtype=dtype)
        sigma_prime = self._sigma_prime(t01, device=device, dtype=dtype)
        A = m_prime / m
        B = sigma_prime - A * sigma
        return A, B

    def _velocity_from_epsilon(self, xt: Tensor, mu: Tensor, t01: Tensor, eps: Tensor) -> Tensor:
        A, B = self._A_B(t01, device=xt.device, dtype=xt.dtype)
        A = _broadcast_time_like(A, xt)
        B = _broadcast_time_like(B, xt)
        return A * (xt - mu) + B * eps

    def _epsilon_from_velocity(self, v: Tensor, xt: Tensor, mu: Tensor, t01: Tensor) -> Tensor:
        A, B = self._A_B(t01, device=xt.device, dtype=xt.dtype)
        A = _broadcast_time_like(A, xt)
        B = _broadcast_time_like(B, xt).clamp_min(1e-20)
        return (v - A * (xt - mu)) / B

    @staticmethod
    def sample_time(
        batch: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        t_min: float = 1e-3,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        if not (0.0 < t_min < 0.5):
            raise ValueError("t_min must be in (0, 0.5)")
        return torch.rand(batch, device=device, dtype=dtype, generator=generator) * (1.0 - 2.0 * t_min) + t_min

    def _compute_predictions(
        self,
        model_output: Tensor,
        *,
        prediction: PredictionType,
        xt: Tensor,
        mu: Tensor,
        t01: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if prediction == "epsilon":
            eps_hat = model_output
            x0_hat = self._x0_from_xt_epsilon(xt, mu, t01, eps_hat)
            v_hat = self._velocity_from_epsilon(xt, mu, t01, eps_hat)
            return eps_hat, x0_hat, v_hat

        if prediction == "x0":
            x0_hat = model_output
            eps_hat = self._epsilon_from_xt_x0(xt, x0_hat, mu, t01)
            v_hat = self._velocity_from_epsilon(xt, mu, t01, eps_hat)
            return eps_hat, x0_hat, v_hat

        if prediction == "v":
            v_hat = model_output
            eps_hat = self._epsilon_from_velocity(v_hat, xt, mu, t01)
            x0_hat = self._x0_from_xt_epsilon(xt, mu, t01, eps_hat)
            return eps_hat, x0_hat, v_hat

        raise ValueError(f"Unknown prediction type: {prediction}")

    def convert_pred(
        self,
        model_output: Tensor,
        *,
        xt: Tensor,
        mu: Tensor,
        t01: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return self._compute_predictions(model_output, prediction=self.prediction, xt=xt, mu=mu, t01=t01)

    def loss(
        self,
        *,
        model_output: Tensor,
        xt: Tensor,
        x0: Tensor,
        mu: Tensor,
        t01: Tensor,
        t_index: int | Tensor | None = None,
    ) -> Tensor:
        eps_hat, x0_hat, v_hat = self.convert_pred(model_output, xt=xt, mu=mu, t01=t01)

        if self.loss_type == "epsilon":
            eps = self._epsilon_target_from_xt(xt, x0, mu, t01)
            return self._pointwise_loss(eps_hat, eps)

        if self.loss_type == "x0":
            return self._pointwise_loss(x0_hat, x0)

        if self.loss_type == "v":
            eps = self._epsilon_target_from_xt(xt, x0, mu, t01)
            v = self._velocity_target(x0, mu, t01, eps)
            return self._pointwise_loss(v_hat, v)

        if self.loss_type == "mean":
            if t_index is None:
                raise ValueError("t_index is required for mean loss")
            return self._loss_mean_matching(
                xt=xt,
                x0=x0,
                mu=mu,
                t_index=t_index,
                eps_hat=eps_hat,
                norm=self.loss_norm,
                weighting=self.loss_weighting,
            )

        raise ValueError(f"Unknown loss_type: {self.loss_type}")

    def _pointwise_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        if self.loss_norm == "l1":
            return (pred - target).abs().mean()
        if self.loss_norm == "l2":
            return (pred - target).pow(2).mean()
        raise ValueError(f"Unknown norm: {self.loss_norm}")

    def _index_to_t01(self, t_index: int | Tensor, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        if isinstance(t_index, int):
            if not (1 <= t_index <= self.T):
                raise ValueError("t_index must be in [1, T]")
            return torch.tensor([t_index / self.T], device=device, dtype=dtype)

        t_index = t_index.to(device=device).to(torch.int64).reshape(-1)
        if torch.any((t_index < 1) | (t_index > self.T)):
            raise ValueError("t_index must be in [1, T]")
        return t_index.to(dtype=dtype) / float(self.T)

    def _m_at_index(self, t_index: int | Tensor, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        return self._m(self._index_to_t01(t_index, device=device, dtype=dtype), device=device, dtype=dtype)

    def _sigma_at_index(self, t_index: int | Tensor, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        return self._sigma(self._index_to_t01(t_index, device=device, dtype=dtype), device=device, dtype=dtype)

    def _diffusion_at_index(self, t_index: int | Tensor, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        return self._diffusion(self._index_to_t01(t_index, device=device, dtype=dtype), device=device, dtype=dtype)

    def _forward_mean(self, x0: Tensor, mu: Tensor, t01: Tensor) -> Tensor:
        m = _broadcast_time_like(self._m(t01, device=x0.device, dtype=x0.dtype), x0)
        return m * x0 + (1.0 - m) * mu

    def _forward_mean_at_index(self, x0: Tensor, mu: Tensor, t_index: int | Tensor) -> Tensor:
        m = _broadcast_time_like(self._m_at_index(t_index, device=x0.device, dtype=x0.dtype), x0)
        return m * x0 + (1.0 - m) * mu

    def _epsilon_target_from_xt(self, xt: Tensor, x0: Tensor, mu: Tensor, t01: Tensor) -> Tensor:
        mean = self._forward_mean(x0, mu, t01)
        sigma = _broadcast_time_like(self._sigma(t01, device=xt.device, dtype=xt.dtype), xt).clamp_min(1e-20)
        return (xt - mean) / sigma

    def _drift_h(self, x: Tensor, mu: Tensor, t01: Tensor) -> Tensor:
        device, dtype = x.device, x.dtype
        btT = self._bar_theta_total(device=device, dtype=dtype) - self._bar_theta(t01, device=device, dtype=dtype)
        tmp = torch.exp(-2.0 * btT)
        sigma_t_T_sq = self._sigma_bar_sq_t_T(t01, device=device, dtype=dtype)
        g = self._diffusion(t01, device=device, dtype=dtype)
        g_sq = g**2
        gamma = torch.tensor(self.gamma, device=device, dtype=dtype)
        denom = (1.0 + gamma * sigma_t_T_sq).clamp_min(1e-20)
        coeff = -(gamma * g_sq * tmp) / denom
        return _broadcast_time_like(coeff, x) * (x - mu)

    def _reverse_mean_theta_adjacent(self, xt: Tensor, mu: Tensor, t_index: int | Tensor, eps_hat: Tensor) -> Tensor:
        device, dtype = xt.device, xt.dtype
        t01 = self._index_to_t01(t_index, device=device, dtype=dtype)
        dt = self._dt.to(device=device, dtype=dtype)

        theta_t = self._theta(t01, device=device, dtype=dtype)
        g = self._diffusion(t01, device=device, dtype=dtype)
        g_sq = g**2
        drift_h = self._drift_h(xt, mu, t01)

        sigma_marg = _broadcast_time_like(self._sigma(t01, device=device, dtype=dtype), xt).clamp_min(1e-20)
        score = -eps_hat / sigma_marg

        drift = _broadcast_time_like(theta_t, xt) * (mu - xt) + drift_h - _broadcast_time_like(g_sq, xt) * score
        return xt - drift * dt

    def _reverse_mean_gamma_adjacent(self, xt: Tensor, x0: Tensor, mu: Tensor, t_index: int | Tensor) -> Tensor:
        device, dtype = xt.device, xt.dtype
        t_index_tensor = (
            t_index if isinstance(t_index, Tensor) else torch.tensor([t_index], device=device, dtype=torch.int64)
        )
        t_index_tensor = t_index_tensor.to(device=device).to(torch.int64).reshape(-1)
        if torch.any(t_index_tensor <= 1):
            raise ValueError("t_index must be >= 2 for adjacent posterior mean")

        m_t = self._m_at_index(t_index, device=device, dtype=dtype)
        m_tm1 = self._m_at_index(t_index_tensor - 1, device=device, dtype=dtype)

        sigma_t = self._sigma_at_index(t_index, device=device, dtype=dtype)
        sigma_tm1 = self._sigma_at_index(t_index_tensor - 1, device=device, dtype=dtype)

        f_m = m_t / m_tm1
        f_sigma_1 = torch.sqrt((sigma_t**2 - (sigma_tm1**2) * (f_m**2)).clamp_min(0.0))

        mean_tm1 = self._forward_mean_at_index(x0, mu, t_index_tensor - 1)
        f_n = (1.0 - m_t) - (1.0 - m_tm1) * (m_t / m_tm1)

        sigma_tm1_sq = _broadcast_time_like(sigma_tm1**2, xt)
        f_m_b = _broadcast_time_like(f_m, xt)
        f_n_b = _broadcast_time_like(f_n, xt)
        f_sigma_1_sq = _broadcast_time_like(f_sigma_1**2, xt)

        num = sigma_tm1_sq * f_m_b * (xt - f_n_b * mu) + f_sigma_1_sq * mean_tm1
        denom = _broadcast_time_like((sigma_t**2).clamp_min(1e-20), xt)
        return num / denom

    def _loss_mean_matching(
        self,
        *,
        xt: Tensor,
        x0: Tensor,
        mu: Tensor,
        t_index: int | Tensor,
        eps_hat: Tensor,
        norm: LossNorm = "l1",
        weighting: LossWeighting = "inv",
    ) -> Tensor:
        mu_theta = self._reverse_mean_theta_adjacent(xt, mu, t_index, eps_hat)
        mu_gamma = self._reverse_mean_gamma_adjacent(xt, x0, mu, t_index)

        if norm == "l1":
            per_elem = (mu_theta - mu_gamma).abs()
        elif norm == "l2":
            per_elem = (mu_theta - mu_gamma).pow(2)
        else:
            raise ValueError(f"Unknown norm: {norm}")

        loss_per_sample = per_elem.reshape(per_elem.shape[0], -1).mean(dim=1)

        if weighting == "none":
            return loss_per_sample.mean()
        if weighting == "inv":
            g = self._diffusion_at_index(t_index, device=xt.device, dtype=xt.dtype).clamp_min(1e-20)
            weight = 1.0 / (2.0 * (g**2))
        else:
            raise ValueError(f"Unknown weighting: {weighting}")
        return (loss_per_sample * weight).mean()


class UniDBEpsilonSampler:
    def __init__(
        self,
        sde: UniDBContinuous,
        predict_epsilon: Callable[[Tensor, Tensor, Tensor], Tensor],
        num_steps: int,
        method: Literal["sde", "mean_ode"] = "sde",
        t_min: float = 0.0,
        t_max: float = 1.0,
    ):
        self.sde = sde

        self.predict_epsilon = predict_epsilon
        self.num_steps = num_steps
        self.method = method
        self.t_min = t_min
        self.t_max = t_max

    def sample(
        self,
        x_init: Tensor,
        mu: Tensor,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        if self.num_steps < 2:
            raise ValueError("num_steps must be >= 2")

        device, dtype = x_init.device, x_init.dtype
        t_min_t = torch.tensor(self.t_min, device=device, dtype=dtype)
        t_max_t = torch.tensor(self.t_max, device=device, dtype=dtype)
        if not (0.0 <= float(t_min_t) < float(t_max_t) <= 1.0):
            raise ValueError("t_min/t_max must satisfy 0 <= t_min < t_max <= 1")

        s_max = t_max_t * float(self.sde.T)
        s_min = t_min_t * float(self.sde.T)
        s_grid = torch.linspace(s_max, s_min, self.num_steps, device=device, dtype=dtype)

        x = x_init
        dt = self.sde._dt.to(device=device, dtype=dtype)

        for i in range(self.num_steps - 1):
            s = s_grid[i]
            s_next = s_grid[i + 1]
            delta_s = (s - s_next).clamp_min(0.0)
            delta_tau = delta_s * dt
            if float(delta_tau) == 0.0:
                continue

            t01_scalar = (s / float(self.sde.T)).clamp(0.0, 1.0).reshape(1)
            t01 = t01_scalar.expand(x.shape[0])

            theta_t = self.sde._theta(t01, device=device, dtype=dtype)
            g = self.sde._diffusion(t01, device=device, dtype=dtype)
            g_sq = g**2
            drift_h = self.sde._drift_h(x, mu, t01)

            sigma_marg = _broadcast_time_like(self.sde._sigma(t01, device=device, dtype=dtype), x)
            if float(sigma_marg.max().item()) <= 0.0:
                eps_hat = torch.zeros_like(x)
                score = torch.zeros_like(x)
            else:
                eps_hat = self.predict_epsilon(x, mu, t01)
                score = -eps_hat / sigma_marg.clamp_min(1e-20)

            drift = _broadcast_time_like(theta_t, x) * (mu - x) + drift_h - _broadcast_time_like(g_sq, x) * score
            x = x - drift * delta_tau

            if self.method == "sde":
                if generator is None:
                    z = torch.randn_like(x)
                else:
                    z = torch.randn(x.shape, device=device, dtype=dtype, generator=generator)
                x = x - _broadcast_time_like(g, x) * torch.sqrt(delta_tau) * z
            elif self.method == "mean_ode":
                pass
            else:
                raise ValueError(f"Unknown method: {self.method}")

        return x


@dataclass(frozen=True)
class LinearInterpolantConfig:
    """
    A simple flow-matching-style probability path on t in [0, 1]:

        x_t = (1 - t) * x0 + t * x1 + sigma(t) * eps

    where x1 is the conditional endpoint (e.g. LQ/mu) and x0 is the data endpoint (e.g. HQ/GT).
    """

    sigma_max: float = 0.0
    sigma_schedule: SigmaSchedule = "quadratic"


class LinearInterpolant:
    def __init__(self, cfg: LinearInterpolantConfig):
        if cfg.sigma_max < 0.0:
            raise ValueError("sigma_max must be non-negative")
        self.cfg = LinearInterpolantConfig(sigma_max=float(cfg.sigma_max), sigma_schedule=cfg.sigma_schedule)

    @staticmethod
    def m(t01: Tensor, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        t01 = _as_1d_time(t01).to(device=device, dtype=dtype).clamp(0.0, 1.0)
        return 1.0 - t01

    @staticmethod
    def m_prime(t01: Tensor, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        t01 = _as_1d_time(t01).to(device=device, dtype=dtype)
        return torch.full_like(t01, -1.0)

    def sigma(self, t01: Tensor, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        t01 = _as_1d_time(t01).to(device=device, dtype=dtype).clamp(0.0, 1.0)
        sigma_max = torch.tensor(self.cfg.sigma_max, device=device, dtype=dtype)
        if self.cfg.sigma_schedule == "none" or float(self.cfg.sigma_max) == 0.0:
            return torch.zeros_like(t01)
        if self.cfg.sigma_schedule == "quadratic":
            return sigma_max * t01 * (1.0 - t01)
        raise ValueError(f"Unknown sigma_schedule: {self.cfg.sigma_schedule}")

    def sigma_prime(self, t01: Tensor, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        t01 = _as_1d_time(t01).to(device=device, dtype=dtype).clamp(0.0, 1.0)
        sigma_max = torch.tensor(self.cfg.sigma_max, device=device, dtype=dtype)
        if self.cfg.sigma_schedule == "none" or float(self.cfg.sigma_max) == 0.0:
            return torch.zeros_like(t01)
        if self.cfg.sigma_schedule == "quadratic":
            return sigma_max * (1.0 - 2.0 * t01)
        raise ValueError(f"Unknown sigma_schedule: {self.cfg.sigma_schedule}")

    def sample_xt(
        self,
        x0: Tensor,
        x1: Tensor,
        t01: Tensor,
        *,
        eps: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        if eps is None:
            if generator is None:
                eps = torch.randn_like(x0)
            else:
                eps = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=generator)

        m = _broadcast_time_like(self.m(t01, device=x0.device, dtype=x0.dtype), x0)
        sigma = _broadcast_time_like(self.sigma(t01, device=x0.device, dtype=x0.dtype), x0)
        xt = m * x0 + (1.0 - m) * x1 + sigma * eps
        return xt, eps

    def epsilon_from_xt_x0(self, xt: Tensor, x0: Tensor, x1: Tensor, t01: Tensor) -> Tensor:
        m = _broadcast_time_like(self.m(t01, device=xt.device, dtype=xt.dtype), xt)
        sigma = _broadcast_time_like(self.sigma(t01, device=xt.device, dtype=xt.dtype), xt).clamp_min(1e-20)
        return (xt - m * x0 - (1.0 - m) * x1) / sigma

    def x0_from_xt_epsilon(self, xt: Tensor, x1: Tensor, t01: Tensor, eps: Tensor) -> Tensor:
        m = _broadcast_time_like(self.m(t01, device=xt.device, dtype=xt.dtype), xt).clamp_min(1e-20)
        sigma = _broadcast_time_like(self.sigma(t01, device=xt.device, dtype=xt.dtype), xt)
        return (xt - (1.0 - m) * x1 - sigma * eps) / m

    def velocity_target(self, x0: Tensor, x1: Tensor, t01: Tensor, eps: Tensor) -> Tensor:
        m_prime = _broadcast_time_like(self.m_prime(t01, device=x0.device, dtype=x0.dtype), x0)
        sigma_prime = _broadcast_time_like(self.sigma_prime(t01, device=x0.device, dtype=x0.dtype), x0)
        return m_prime * (x0 - x1) + sigma_prime * eps

    def _A_B(self, t01: Tensor, *, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        m = self.m(t01, device=device, dtype=dtype).clamp_min(1e-20)
        m_prime = self.m_prime(t01, device=device, dtype=dtype)
        sigma = self.sigma(t01, device=device, dtype=dtype)
        sigma_prime = self.sigma_prime(t01, device=device, dtype=dtype)
        A = m_prime / m
        B = sigma_prime - A * sigma
        return A, B

    def velocity_from_epsilon(self, xt: Tensor, x1: Tensor, t01: Tensor, eps: Tensor) -> Tensor:
        A, B = self._A_B(t01, device=xt.device, dtype=xt.dtype)
        A = _broadcast_time_like(A, xt)
        B = _broadcast_time_like(B, xt)
        return A * (xt - x1) + B * eps

    def epsilon_from_velocity(self, v: Tensor, xt: Tensor, x1: Tensor, t01: Tensor) -> Tensor:
        A, B = self._A_B(t01, device=xt.device, dtype=xt.dtype)
        A = _broadcast_time_like(A, xt)
        B = _broadcast_time_like(B, xt).clamp_min(1e-20)
        return (v - A * (xt - x1)) / B


class UniDBLoss:
    """Training loss wrapper.

    Per your constraint, the public API is:
        `loss(model, x0, mu, model_kwargs)`

    This wrapper keeps all existing private method names in `UniDBContinuous` untouched,
    and only orchestrates: sample `t`, sample `x_t`, call `model`, then compute loss.
    """

    def __init__(
        self,
        sde: UniDBContinuous,
        *,
        prediction: PredictionType = "epsilon",
        loss_type: UniDBLossType = "mean",
        norm: LossNorm = "l1",
        weighting: LossWeighting = "inv",
    ):
        self.sde = sde
        self.prediction = prediction
        self.loss_type = loss_type
        self.norm = norm
        self.weighting = weighting

    def loss(
        self,
        model: Callable[..., Tensor],
        x0: Tensor,
        mu: Tensor,
        model_kwargs: dict | None = None,
        *,
        generator: torch.Generator | None = None,
        t_min: float = 1e-3,
    ) -> Tensor:
        if model_kwargs is None:
            model_kwargs = {}

        batch = x0.shape[0]
        t01 = self.sde.sample_time(batch, device=x0.device, dtype=x0.dtype, t_min=t_min, generator=generator)
        if self.loss_type == "mean":
            t_index = (t01 * float(self.sde.T)).to(torch.int64).clamp_min(2)
        else:
            t_index = (t01 * float(self.sde.T)).to(torch.int64).clamp_min(1)

        xt, eps = self.sde.sample_xt(x0, mu, t01, generator=generator)

        call_kwargs = _maybe_add_model_kwargs(model, model_kwargs, mu=mu)
        model_output = model(xt, t01, **call_kwargs)

        eps_hat, x0_hat, v_hat = self.sde._compute_predictions(  # noqa: SLF001
            model_output, prediction=self.prediction, xt=xt, mu=mu, t01=t01
        )

        if self.loss_type == "epsilon":
            return self._pointwise_loss(eps_hat, eps)
        if self.loss_type == "x0":
            return self._pointwise_loss(x0_hat, x0)
        if self.loss_type == "v":
            v = self.sde.velocity_target(x0, mu, t01, eps)
            return self._pointwise_loss(v_hat, v)
        if self.loss_type == "mean":
            return self.sde._loss_mean_matching(  # noqa: SLF001
                xt=xt,
                x0=x0,
                mu=mu,
                t_index=t_index,
                eps_hat=eps_hat,
                norm=self.norm,
                weighting=self.weighting,
            )
        raise ValueError(f"Unknown loss_type: {self.loss_type}")

    def _pointwise_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        if self.norm == "l1":
            return (pred - target).abs().mean()
        if self.norm == "l2":
            return (pred - target).pow(2).mean()
        raise ValueError(f"Unknown norm: {self.norm}")
