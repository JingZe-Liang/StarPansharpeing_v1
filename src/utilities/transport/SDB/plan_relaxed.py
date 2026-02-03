"""
Relaxed-endpoint (lambda_lq) + posterior/DDIM-style sampler for SDB.

Drop-in addition to plan.py:
- DOES NOT modify existing `SDBContinuousPlan(SDBPlan)` or its samplers.
- Adds a new Plan + Sampler pair that uses a relaxed terminal std gamma(t)
  and a posterior update (deterministic when eta=0, stochastic when eta>0).

Intuition:
    x_t = α(t) x0 + β(t) x1 + γ(t) z,  z ~ N(0, I)
We run a DDIM-like posterior step by carrying a predicted z forward and (optionally)
injecting fresh noise while keeping Var[x_s | ...] = γ(s)^2.
"""

from __future__ import annotations

import inspect
from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from .plan import (
    SDBPlan,
    SDBSampler,
    DiffusionTarget,
    expand_t_as,
    edm_t_sample,
    sigmoid_t_sample,
    _maybe_add_condition_kwargs,
)


class SDBRelaxedPosteriorPlan(SDBPlan):
    """A simple Gaussian bridge plan with a *relaxed* terminal noise level.

    Mean path (default): linear
        α(t) = 1 - t
        β(t) = t

    Noise std schedule (relaxed endpoint):
        γ(t) = sigma_base * (lambda_b * t(1-t) + lambda_lq * t^2)

    Notes
    -----
    - `lambda_lq` controls terminal relaxation: γ(1) = sigma_base * lambda_lq.
      Setting lambda_lq=0 collapses the endpoint to a Dirac (strict endpoint).
    - This plan is meant to be sampled with the posterior sampler below, not with
      SDE/ODE drift-diffusion samplers.
    """

    def __init__(
        self,
        plan_tgt: DiffusionTarget = DiffusionTarget.x_0,
        *,
        sigma_base: float = 1.0,
        lambda_lq: float = 1.0,
        lambda_b: float = 0.0,
        t_train_kwargs: dict | None = None,
        t_sample_type: str = "uniform",
    ):
        super().__init__(plan_tgt=plan_tgt)
        self.sigma_base = float(sigma_base)
        self.lambda_lq = float(lambda_lq)
        self.lambda_b = float(lambda_b)

        self.t_train_kwargs = t_train_kwargs or {"device": "cuda", "clip_t_min_max": (1e-4, 1 - 1e-4)}
        self.t_sample_type = t_sample_type

    # ---- basic helpers ----

    def expand_t_as(self, t: Tensor, x: Tensor) -> Tensor:
        return expand_t_as(t, x)

    # ---- schedules ----

    def alpha_t_with_derivative(self, t: Tensor) -> tuple[Tensor, Tensor]:
        # α(t) = 1 - t
        return (1.0 - t), (-torch.ones_like(t))

    def beta_t_with_derivative(self, t: Tensor) -> tuple[Tensor, Tensor]:
        # β(t) = t
        return t, torch.ones_like(t)

    def gamma_t_with_derivative(self, t: Tensor) -> tuple[Tensor, Tensor]:
        # γ(t) = sigma_base * (lambda_b * t(1-t) + lambda_lq * t^2)
        # γ'(t) = sigma_base * (lambda_b * (1 - 2t) + lambda_lq * 2t)
        t = t.to(torch.float32)
        gamma = self.sigma_base * (self.lambda_b * (t * (1.0 - t)) + self.lambda_lq * (t * t))
        gamma_p = self.sigma_base * (self.lambda_b * (1.0 - 2.0 * t) + self.lambda_lq * (2.0 * t))
        return gamma, gamma_p

    def epsilon_t(self, t: Tensor) -> Tensor:
        # Not used by the posterior sampler.
        # We keep it for interface completeness.
        return torch.zeros_like(t)

    # ---- time grids ----

    def sample_continous_t(self, **t_sample_kwargs) -> Tensor:
        t_min = max(self.t_train_kwargs["clip_t_min_max"][0], t_sample_kwargs.get("t_min", 1e-4))
        t_max = min(self.t_train_kwargs["clip_t_min_max"][1], t_sample_kwargs.get("t_max", 1 - 1e-4))
        t_sample_kwargs["t_min"] = t_min
        t_sample_kwargs["t_max"] = t_max
        device = self.t_train_kwargs.get("device", "cuda")

        if self.t_sample_type == "uniform":
            time_grid = torch.linspace(t_max, t_min, steps=t_sample_kwargs["n_timesteps"]).to(
                device, dtype=torch.float32
            )
        elif self.t_sample_type == "edm":
            time_grid = edm_t_sample(**t_sample_kwargs).to(device, dtype=torch.float32)
        elif self.t_sample_type == "sigmoid":
            time_grid = sigmoid_t_sample(**t_sample_kwargs).to(device, dtype=torch.float32)
        else:
            raise NotImplementedError(f"Unsupported t_sample_type: {self.t_sample_type}")

        return time_grid

    def train_continous_t(self, batch_size: int) -> Tensor:
        # Keep identical behavior to SDBContinuousPlan's default EDM sampler if present.
        # For simplicity, we use edm_t_train if available; otherwise uniform in [t_min, t_max].
        device = self.t_train_kwargs.get("device", "cuda")
        t_min, t_max = self.t_train_kwargs.get("clip_t_min_max", (1e-4, 1 - 1e-4))
        if "edm_t_train" in globals():
            return edm_t_train(batch_size=batch_size, device=torch.device(device), clip_t_min_max=(t_min, t_max))
        return torch.rand((batch_size,), device=device, dtype=torch.float32) * (t_max - t_min) + t_min

    # ---- target conversions (same Gaussian kernel identities as SDBContinuousPlan) ----

    def get_score_from_velocity(self, v: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        raise NotImplementedError("Velocity target not used in this relaxed posterior plan.")

    def get_x0_from_score(self, score: Tensor, t: Tensor, x_t: Tensor, x_1: Tensor) -> Tensor:
        alpha_t, _ = self.alpha_t_with_derivative(self.expand_t_as(t, x_t))
        beta_t, _ = self.beta_t_with_derivative(self.expand_t_as(t, x_t))
        gamma_t, _ = self.gamma_t_with_derivative(self.expand_t_as(t, x_t))
        return ((gamma_t**2) * score + x_t - beta_t * x_1) / alpha_t

    def get_score_from_x_0_x_t(self, x_0: Tensor, x_t: Tensor, t: Tensor, x_1: Tensor) -> Tensor:
        t = self.expand_t_as(t, x_t)
        alpha_t, _ = self.alpha_t_with_derivative(t)
        beta_t, _ = self.beta_t_with_derivative(t)
        gamma_t, _ = self.gamma_t_with_derivative(t)
        return (alpha_t * x_0 + beta_t * x_1 - x_t) / (gamma_t**2)

    def get_noise_from_score(self, score: Tensor, t: Tensor) -> Tensor:
        gamma_t, _ = self.gamma_t_with_derivative(self.expand_t_as(t, score))
        return -gamma_t * score

    def get_score_from_noise(self, noise: Tensor, t: Tensor) -> Tensor:
        gamma_t, _ = self.gamma_t_with_derivative(self.expand_t_as(t, noise))
        return -noise / gamma_t

    def get_noise_from_x_0_x_t(self, x_0: Tensor, x_t: Tensor, t: Tensor, x_1: Tensor) -> Tensor:
        t = self.expand_t_as(t, x_t)
        alpha_t, _ = self.alpha_t_with_derivative(t)
        beta_t, _ = self.beta_t_with_derivative(t)
        gamma_t, _ = self.gamma_t_with_derivative(t)
        return (x_t - alpha_t * x_0 - beta_t * x_1) / gamma_t

    # ---- sampling from the kernel ----

    def get_x_t(
        self, t: Tensor, x_0: Tensor, x_1: Tensor | None = None, z: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        if x_1 is None:
            x_1 = torch.randn_like(x_0)
        t = self.expand_t_as(t, x_0).to(torch.float32)

        alpha_t, _ = self.alpha_t_with_derivative(t)
        beta_t, _ = self.beta_t_with_derivative(t)
        gamma_t, _ = self.gamma_t_with_derivative(t)

        if z is None:
            z = torch.randn_like(x_0)

        x_t = alpha_t * x_0 + beta_t * x_1 + gamma_t * z
        return x_t, x_1

    def get_x_t_with_target(self, t: Tensor, x_0: Tensor, x_1: Tensor | None = None) -> tuple[Tensor, Tensor]:
        t = self.expand_t_as(t, x_0).to(torch.float32)
        alpha_t, alpha_p = self.alpha_t_with_derivative(t)
        beta_t, beta_p = self.beta_t_with_derivative(t)

        x_t, x_1 = self.get_x_t(t, x_0, x_1)

        if self.plan_tgt == DiffusionTarget.score:
            tgt = self.get_score_from_x_0_x_t(x_0, x_t, t, x_1)
        elif self.plan_tgt == DiffusionTarget.x_0:
            tgt = x_0
        elif self.plan_tgt == DiffusionTarget.noise:
            tgt = self.get_noise_from_x_0_x_t(x_0, x_t, t, x_1)
        elif self.plan_tgt == DiffusionTarget.velocity:
            tgt = x_0 * alpha_p + x_1 * beta_p
        else:
            raise ValueError(f"Unsupported diffusion target: {self.plan_tgt}")

        return x_t, tgt

    # ---- drift/diffusion interfaces: intentionally unsupported ----

    def get_sde_drift_diffusion(self, *args, **kwargs):
        raise NotImplementedError("Use the posterior sampler (DDIM-style), not SDE/ODE sampling, for this plan.")


class SDBRelaxedPosteriorSampler(SDBSampler):
    """DDIM-style posterior sampler for `SDBRelaxedPosteriorPlan`.

    Update rule (t -> s, with s < t):
        z_hat = (x_t - α_t x0_hat - β_t x1) / γ_t

        x_s = α_s x0_hat + β_s x1 + γ_s ( sqrt(1-eta^2) z_hat + eta * eps ), eps~N(0,I)

    - eta = 0: deterministic (closest to SDB's `step_mean`)
    - eta > 0: injects fresh noise while preserving marginal variance γ_s^2.
    """

    def __init__(self, plan: SDBRelaxedPosteriorPlan, *, eta: float = 0.0):
        super().__init__(plan)
        self.plan: SDBRelaxedPosteriorPlan = plan
        if not (0.0 <= eta <= 1.0):
            raise ValueError("eta must be in [0, 1].")
        self.eta = float(eta)

    def model_pred_to_x_0(
        self,
        model: nn.Module,
        x_t: Tensor,
        t: Tensor,
        x_cond: Tensor,
        x_1_kernel: Tensor,
        model_kwargs: dict | None = None,
    ) -> Tensor:
        """Convert model output to x0_hat, matching `DiffusionTarget` behavior."""
        model_kwargs = {} if model_kwargs is None else model_kwargs

        with torch.no_grad():
            cond_kwargs = _maybe_add_condition_kwargs(model, model_kwargs, x_1=x_cond)
            out = model(x_t, t, **cond_kwargs)
            if isinstance(out, Sequence):
                out = out[0]

        if self.plan_tgt == DiffusionTarget.score:
            return self.plan.get_x0_from_score(out, t, x_t, x_1_kernel)
        if self.plan_tgt == DiffusionTarget.x_0:
            return out
        if self.plan_tgt == DiffusionTarget.noise:
            score = self.plan.get_score_from_noise(out, t)
            return self.plan.get_x0_from_score(score, t, x_t, x_1_kernel)

        raise ValueError(f"Unsupported diffusion target: {self.plan_tgt}")

    @torch.no_grad()
    def step_posterior(
        self,
        model: nn.Module,
        x_t: Tensor,
        t: Tensor,
        t_prev: Tensor,
        *,
        x_cond: Tensor,
        x_1_kernel: Tensor,
        model_kwargs: dict | None = None,
        clip_value: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Single posterior step from t -> t_prev."""
        model_kwargs = {} if model_kwargs is None else model_kwargs

        x0_hat = self.model_pred_to_x_0(model, x_t, t, x_cond=x_cond, x_1_kernel=x_1_kernel, model_kwargs=model_kwargs)
        if clip_value:
            x0_hat = x0_hat.clamp(-1.0, 1.0)

        # current noise estimate z_hat at time t (kernel endpoint is x_1_kernel)
        t_exp = self.plan.expand_t_as(t, x_t)
        a_t, _ = self.plan.alpha_t_with_derivative(t_exp)
        b_t, _ = self.plan.beta_t_with_derivative(t_exp)
        g_t, _ = self.plan.gamma_t_with_derivative(t_exp)
        z_hat = (x_t - a_t * x0_hat - b_t * x_1_kernel) / g_t

        # build x_{t_prev}
        s_exp = self.plan.expand_t_as(t_prev, x_t)
        a_s, _ = self.plan.alpha_t_with_derivative(s_exp)
        b_s, _ = self.plan.beta_t_with_derivative(s_exp)
        g_s, _ = self.plan.gamma_t_with_derivative(s_exp)

        if self.eta == 0.0:
            z_s = z_hat
        else:
            eps = torch.randn_like(x_t)
            z_s = (1.0 - self.eta**2) ** 0.5 * z_hat + self.eta * eps

        x_prev = a_s * x0_hat + b_s * x_1_kernel + g_s * z_s
        return x_prev, x0_hat

    @torch.no_grad()
    def sample_posterior(
        self,
        model: nn.Module,
        x_1: Tensor,
        *,
        time_grid: Tensor | None = None,
        n_steps: int = 50,
        model_kwargs: dict | None = None,
        clip_value: bool = False,
        progress: bool = False,
    ) -> Tensor:
        """Run posterior sampling from t_max -> t_min.

        `x_1` is the *clean* conditional endpoint (LQ).
        The kernel endpoint equals `x_1` (relaxation is built into gamma(t)).
        Initialization: x_{t_max} = x_1 + gamma(t_max) * N(0,I).
        """
        model_kwargs = {} if model_kwargs is None else model_kwargs

        if time_grid is None:
            time_grid = self.plan.sample_continous_t(n_timesteps=n_steps)
        if time_grid.dim() != 1:
            raise ValueError("time_grid must be a 1D tensor.")

        # ensure descending: t_max -> t_min
        if time_grid[0] < time_grid[-1]:
            time_grid = torch.flip(time_grid, dims=[0])

        x_cond = x_1
        x_1_kernel = x_1

        # init at t0 = time_grid[0]
        t0 = time_grid[0:1].expand(x_1.shape[0])
        g0, _ = self.plan.gamma_t_with_derivative(self.plan.expand_t_as(t0, x_1))
        x_t = x_1 + g0 * torch.randn_like(x_1)

        # iterate
        it = range(len(time_grid) - 1)
        if progress:
            try:
                from tqdm import tqdm

                it = tqdm(it)
            except Exception:
                pass

        for i in it:
            t = time_grid[i : i + 1].expand(x_1.shape[0])
            t_prev = time_grid[i + 1 : i + 2].expand(x_1.shape[0])
            x_t, _ = self.step_posterior(
                model,
                x_t,
                t=t,
                t_prev=t_prev,
                x_cond=x_cond,
                x_1_kernel=x_1_kernel,
                model_kwargs=model_kwargs,
                clip_value=clip_value,
            )

        return x_t
