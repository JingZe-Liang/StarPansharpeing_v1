"""
Diffusion model implementation using Stochastic Control Transport

Author: Zihan Cao
Date: 2025-01-16

Copyright (c) 2025 Zihan Cao, Mathematical School, UESTC.
"""

import enum
import inspect
from typing import Sequence

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from einops import rearrange

# * Type hint ======================================================================

type X0 = Tensor
type Score = Tensor
type Noise = Tensor
type Xt = Tensor


def _maybe_add_condition_kwargs(model: nn.Module, model_kwargs: dict, *, x_1: Tensor) -> dict:
    """Best-effort inject `x_1` into `model_kwargs` if the model accepts it.

    This is used to align with arXiv:2410.21553v2 (Algorithm 1), where the denoiser
    is conditioned on the endpoint sample `x_T`. In this repo we pass the endpoint
    as `x_1`.
    """
    signature = None
    try:
        signature = inspect.signature(model.forward)
    except (TypeError, ValueError):
        pass

    if signature is None:
        return model_kwargs

    parameters = signature.parameters
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    if accepts_kwargs:
        return {**model_kwargs, "x_1": x_1}

    if "x_1" in parameters:
        return {**model_kwargs, "x_1": x_1}

    return model_kwargs


def _traj_save_indices(n_steps: int, traj_saved_n: int) -> set[int]:
    """Return step indices to save `traj_saved_n` snapshots (including both ends)."""
    traj_saved_n = max(1, min(traj_saved_n, n_steps))
    indices = torch.linspace(0, n_steps - 1, steps=traj_saved_n, dtype=torch.int64).tolist()
    return {int(i) for i in indices}


def _time_grid_from_plan(plan: "SDBContinuousPlan", *, device: torch.device, steps: int) -> Tensor:
    """Create a monotone decreasing time grid from `t_max` to `t_min`."""
    return torch.linspace(plan.t_max, plan.t_min, steps=steps, device=device)


# * utilities ======================================================================


class DiffusionTarget(str, enum.Enum):
    noise = "noise"
    velocity = "velocity"
    score = "score"
    x_0 = "x_0"


def expand_t_as(t: Tensor, x: Tensor, dim_not_match_raise: bool = True) -> Tensor:
    """Broadcast 1D time tensor `t` to match `x` shape (batch-first)."""
    if t.dim() != 1:
        if dim_not_match_raise:
            raise ValueError(f"t must be a 1D tensor, but got {t.dim()}D tensor to expand to {x.dim()}D tensor")
    else:
        shape = "1 " * (x.dim() - t.dim())
        t = rearrange(t, "b -> b " + shape)

    return t.to(device=x.device, dtype=torch.float32)


# * Stochastic Control Transport Planner ========================================


def edm_t_sample(
    rho: float, t_min: float, t_max: float, n_timesteps: int, device: str = "cuda", force_to_last_zero: bool = False
) -> Tensor:
    """EDM-style monotone decreasing time grid in (0, 1].

    This is a practical schedule choice (not required by the paper).
    """
    rho_inv = 1 / rho
    ts = torch.arange(n_timesteps, dtype=torch.float64, device=device)
    t_grid = (t_max**rho_inv + ts / (n_timesteps - 1) * (t_min**rho_inv - t_max**rho_inv)) ** rho
    if force_to_last_zero:
        t_grid = torch.cat([torch.as_tensor(t_grid), torch.zeros_like(t_grid[:1])])
    return t_grid.to(torch.float32)


def edm_t_train(batch_size: int, device: torch.device, clip_t_min_max: tuple | None = None) -> torch.Tensor:
    """EDM-style continuous time sampling for training (not required by the paper)."""
    P_mean = -1.2
    P_std = 1.2
    rnd_normal = torch.randn((batch_size,), device=device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    time = 1 / (1 + sigma)
    if clip_t_min_max:
        time = torch.clip(time, min=clip_t_min_max[0], max=clip_t_min_max[1])
    return time


def sigmoid_t_sample(
    n_timesteps: int,
    k: float = 7.0,
    t_min: float = 1e-5,
    t_max: float = 1 - 1e-5,
    force_to_last_zero: bool = False,
) -> Tensor:
    """Sigmoid-shaped monotone decreasing time grid (not required by the paper)."""
    step_indices = torch.linspace(0, 1, n_timesteps - 1, dtype=torch.float64)
    k_tensor = torch.tensor(k, dtype=step_indices.dtype, device=step_indices.device)
    numerator = torch.sigmoid(k_tensor * (step_indices - 0.5)) - torch.sigmoid(-k_tensor * 0.5)
    denominator = torch.sigmoid(k_tensor * 0.5) - torch.sigmoid(-k_tensor * 0.5)
    time_vec = numerator / denominator
    if force_to_last_zero:
        time_vec = torch.cat([time_vec.clip(min=t_min, max=t_max).flip(0), torch.zeros_like(time_vec[:1])])
    return time_vec.to(torch.float32)


# Define a custom probability density function
class ExponentialPDF(stats.rv_continuous):
    def _pdf(self, x, a):
        C = a / (np.exp(a) - 1)
        return C * np.exp(a * x)


exponential_pdf = ExponentialPDF(a=0, b=1, name="ExponentialPDF")


def cosh_t_train(
    batch_size: int,
    a: int = 4,
    device: torch.device = torch.device("cuda"),
    clip_t_min_max: tuple[float, float] = (1e-5, 1 - 1e-5),
):
    """A symmetric (t and 1-t) training time sampler (not required by the paper)."""
    global exponential_pdf

    t = exponential_pdf.rvs(size=batch_size, a=a)
    t = torch.from_numpy(t).float().to(device, torch.float32)
    t = torch.cat([t, 1 - t], dim=0)
    t = t[torch.randperm(t.shape[0])]
    t = t[:batch_size]

    t_min, t_max = clip_t_min_max

    # Scale t to [t_min, t_max]
    t = t * (t_max - t_min) + t_min
    return t


# * SDB Plan and Sampler base class ==============================================


class SDBPlan:
    def __init__(self, plan_tgt: DiffusionTarget):
        self.plan_tgt = plan_tgt

    def expand_t_as(self, t: Tensor, x: Tensor) -> Tensor:
        raise NotImplementedError

    def alpha_t_with_derivative(self, t: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def beta_t_with_derivative(self, t: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def gamma_t_with_derivative(self, t: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def epsilon_t(self, t: Tensor) -> Tensor:
        raise NotImplementedError

    def sample_continous_t(self, **t_sample_kwargs) -> Tensor:
        raise NotImplementedError

    def train_continous_t(self, batch_size: int) -> Tensor:
        raise NotImplementedError

    def get_score_from_velocity(self, v: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        raise NotImplementedError

    def get_x0_from_score(self, score: Tensor, t: Tensor, x_t: Tensor, x_1: Tensor) -> Tensor:
        raise NotImplementedError

    def get_score_from_x_0_x_t(self, x_0: Tensor, x_t: Tensor, t: Tensor, x_1: Tensor) -> Tensor:
        raise NotImplementedError

    def get_noise_from_score(self, score: Tensor, t: Tensor) -> Tensor:
        raise NotImplementedError

    def get_noise_from_x_0_x_t(self, x_0: Tensor, x_t: Tensor, t: Tensor, x_1: Tensor) -> Tensor:
        raise NotImplementedError

    def get_x_t(
        self, t: Tensor, x_0: Tensor, x_1: Tensor | None = None, z: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def get_x_t_with_target(self, t: Tensor, x_0: Tensor, x_1: Tensor | None = None) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def get_sde_drift_diffusion(
        self, t: Tensor, x_0: Tensor, x_t: Tensor, x_1: Tensor, delta_t: float
    ) -> tuple[Tensor, Tensor]:
        raise NotImplementedError


class SDBSampler:
    def __init__(self, plan: SDBPlan):
        self.plan: SDBPlan = plan
        self.plan_tgt = plan.plan_tgt

    def model_pred_to_x_0(self, model: nn.Module, x_t: Tensor, t: Tensor, x_1: Tensor, model_kwargs: dict) -> Tensor:
        raise NotImplementedError

    def step_mean(
        self,
        model: nn.Module,
        x_t: Tensor,
        t: Tensor,
        x_1: Tensor,
        delta_t: float,
        model_kwargs: dict,
        clip_value: bool,
        *,
        prev_x_0_pred: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def step_sde(
        self,
        model: nn.Module,
        x_t: Tensor,
        t: Tensor,
        x_1: Tensor,
        delta_t: float,
        model_kwargs: dict,
        clip_value: bool,
        *,
        prev_x_0_pred: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def sample_sde_euler(
        self,
        model: nn.Module,
        x_1: Tensor,
        time_grid: Tensor | None = None,
        sample_n_steps: int | None = None,
        last_n_step_only_mean: int = 1,
        traj_saved_n: int = 5,
        model_kwargs: dict = {},
        clip_value: bool = False,
        progress: bool = True,
    ) -> tuple[Tensor, list]:
        raise NotImplementedError

    def sample_ode_euler(
        self,
        model: nn.Module,
        x_1: Tensor,
        time_grid: Tensor | None = None,
        sample_n_steps: int | None = None,
        traj_saved_n: int = 5,
        model_kwargs: dict = {},
        clip_value: bool = False,
        progress: bool = True,
    ) -> tuple[Tensor, list]:
        raise NotImplementedError

    def sample_sde_midpoint(
        self,
        model: nn.Module,
        x_1: Tensor,
        time_grid: Tensor | None = None,
        sample_n_steps: int | None = None,
        last_n_step_only_mean: int = 1,
        traj_saved_n: int = 5,
        model_kwargs: dict = {},
        clip_value: bool = False,
        progress: bool = True,
        use_pred_x_0: bool = False,
    ) -> tuple[Tensor, list]:
        raise NotImplementedError

    def pre_noise_x_1(self, x_1: Tensor) -> Tensor:
        raise NotImplementedError


# * SDB Continuous Plan =========================================================


class SDBContinuousPlan(SDBPlan):
    """Continuous-time Endpoint-Conditioned Stochastic Interpolants (ECSI) plan.

    Paper: arXiv:2410.21553v2, Proposition 4.1.

    The bridge transition kernel is parameterized as:
        p_{t|0,T}(x_t | x_0, x_T) = N(x_t; α_t x_0 + β_t x_T, γ_t^2 I)

    Notes:
        - This implementation uses normalized time `t ∈ [0, 1]` (the paper sets `T=1` in experiments).
        - In this code, the endpoint inside the kernel is passed as `x_1` (paper notation: `x_T`).
    """

    def __init__(
        self,
        plan_tgt: DiffusionTarget = DiffusionTarget.x_0,
        gamma_max: float = 0.5,
        eps_eta: float = 1.0,
        alpha_beta_type: str = "linear",
        diffusion_type: str = "bridge",
        t_train_type: str = "edm",
        t_train_kwargs: dict = {"device": "cuda", "clip_t_min_max": (1e-4, 1 - 1e-4)},
        t_sample_type: str = "edm",
    ):
        super().__init__(plan_tgt)

        if plan_tgt == DiffusionTarget.velocity:
            raise NotImplementedError("Velocity target is not implemented yet")

        self.gamma_max = gamma_max
        self.eps_eta = eps_eta
        self.alpha_beta_type = alpha_beta_type
        assert self.alpha_beta_type in ["linear", "sin"], "Unsupported alpha beta type: {}".format(self.alpha_beta_type)
        self.diffusion_type = diffusion_type
        assert self.diffusion_type in ["bridge", "quad", "sin", "sin_one_side"], (
            "Unsupported diffusion type: {}".format(self.diffusion_type)
        )
        # train/sample group
        # assert (self.alpha_beta_type, self.diffusion_type) in (('linear', 'bridge'), ('sin', 'sin')), 'composition of alpha/beta and diffusion type in not support'

        self.t_train_type = t_train_type
        self.t_train_kwargs = t_train_kwargs
        assert self.t_train_type in ["uniform", "edm", "cosh"], "Unsupported train timestep type: {}".format(
            self.t_train_type
        )

        self.t_sample_type = t_sample_type
        assert self.t_sample_type in ["uniform", "edm", "sigmoid"], "Unsupported sample timestep type: {}".format(
            self.t_sample_type
        )

        t_min, t_max = self.t_train_kwargs.get("clip_t_min_max", (1e-4, 1 - 1e-4))
        self.t_min = float(t_min)
        self.t_max = float(t_max)

    def expand_t_as(self, t: Tensor, x: Tensor) -> Tensor:
        return expand_t_as(t, x, dim_not_match_raise=False).float()

    def alpha_t_with_derivative(self, t: Tensor) -> tuple[Tensor, Tensor]:
        """Return α_t and α̇_t used in the Gaussian bridge kernel."""
        t = t.float()
        if self.alpha_beta_type == "linear":
            return 1 - t, torch.full_like(t, -1.0)
        if self.alpha_beta_type == "sin":
            return torch.cos(torch.pi * t / 2), -torch.pi / 2 * torch.sin(torch.pi * t / 2)
        raise NotImplementedError(f"Unsupported alpha beta type: {self.alpha_beta_type}")

    def beta_t_with_derivative(self, t: Tensor) -> tuple[Tensor, Tensor]:
        """Return β_t and β̇_t used in the Gaussian bridge kernel."""
        t = t.float()
        if self.alpha_beta_type == "linear":
            return t, torch.ones_like(t)
        if self.alpha_beta_type == "sin":
            return torch.sin(torch.pi * t / 2), torch.pi / 2 * torch.cos(torch.pi * t / 2)
        raise NotImplementedError(f"Unsupported alpha beta type: {self.alpha_beta_type}")

    def gamma_t_with_derivative(self, t: Tensor) -> tuple[Tensor, Tensor]:
        """Return γ_t and γ̇_t used in the Gaussian bridge kernel.

        For `diffusion_type="bridge"`, this matches the paper's linear-kernel setup:
            γ_t^2 = 4 * γ_max^2 * t * (1 - t)
        """
        t = t.float()
        if self.diffusion_type == "bridge":
            # Clamp t to avoid singularities at t=0/1 (gamma_t' -> inf).
            t = torch.clamp(t, min=self.t_min, max=self.t_max)
            # Paper (arXiv:2410.21553v2)t. uses:
            #   gamma_t^2 = 4 * gamma_max^2 * t * (1 - t)
            # so that max(gamma_t) = gamma_max at t=0.5.
            return (
                2 * self.gamma_max * torch.sqrt((1 - t) * t),
                self.gamma_max * (1 - 2 * t) / torch.sqrt(t * (1 - t)),
            )
        elif self.diffusion_type == "quad":
            return 0.5 * t**2, t
        elif self.diffusion_type == "sin":
            return (torch.sin(t * np.pi) * self.gamma_max, torch.cos(t * np.pi) * np.pi * self.gamma_max)
        elif self.diffusion_type == "sin_one_side":
            return (torch.sin(t * np.pi / 2) * self.gamma_max, torch.cos(t * np.pi / 2) * np.pi / 2 * self.gamma_max)
        else:
            raise NotImplementedError(
                f"Unsupported diffusion type: {self.diffusion_type}, only support bridge, quad, sin"
            )

    def epsilon_t(self, t: Tensor) -> Tensor:
        """Return ϵ_t (paper notation) controlling stochasticity.

        Paper: arXiv:2410.21553v2, Section 4.3.
            ϵ_t = η * (γ_t γ̇_t - (α̇_t / α_t) * γ_t^2)
        Here `eps_eta` corresponds to the paper's interpolation parameter `η`.
        """
        gamma_t, gamma_t_prime = self.gamma_t_with_derivative(t)
        gamma_t2 = gamma_t**2
        alpha_t, alpha_t_prime = self.alpha_t_with_derivative(t)

        return self.eps_eta * (gamma_t * gamma_t_prime - alpha_t_prime / alpha_t * gamma_t2)

    def sample_continous_t(self, **t_sample_kwargs):
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
            raise NotImplementedError(f"Unsupported t sample type: {self.t_sample_type}")

        return time_grid

    def train_continous_t(self, batch_size: int):
        if self.t_train_type == "uniform":
            t_min, t_max = self.t_train_kwargs["clip_t_min_max"]
            return torch.rand(batch_size, device="cuda") * (t_max - t_min) + t_min
        elif self.t_train_type == "edm":
            return edm_t_train(batch_size, **self.t_train_kwargs)
        elif self.t_train_type == "cosh":
            return cosh_t_train(batch_size, **self.t_train_kwargs)
        else:
            raise NotImplementedError("Unsupported train timestep type: {}".format(self.t_train_type))

    def get_score_from_velocity(self, v: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        # velocity = x_0 - x_1 = (alpha_t * x_0 + beta_t * x_1 + gamma_t * z) - x_1
        # score = - (alpha_t * x_0 + beta_t * x_1 - x_t) / gamma_t ** 2

        raise NotImplementedError("Score from velocity is not implemented")

    def get_x0_from_score(self, score: Tensor, t: Tensor, x_t: Tensor, x_1: Tensor) -> Tensor:
        """Recover x̂0 from a score estimate under the Gaussian bridge kernel."""
        alpha_t, _ = self.alpha_t_with_derivative(t)
        beta_t, _ = self.beta_t_with_derivative(t)
        gamma_t, _ = self.gamma_t_with_derivative(t)
        gamma_t2 = gamma_t**2

        return (gamma_t2 * score + x_t - beta_t * x_1) / alpha_t

    def get_score_from_x_0_x_t(self, x_0: Tensor, x_t: Tensor, t: Tensor, x_1: Tensor) -> Tensor:
        """Return ∇_{x_t} log p_{t|0,T}(x_t | x_0, x_T) for the Gaussian kernel."""
        alpha_t, _ = self.alpha_t_with_derivative(t)
        beta_t, _ = self.beta_t_with_derivative(t)
        gamma_t, _ = self.gamma_t_with_derivative(t)

        return (alpha_t * x_0 + beta_t * x_1 - x_t) / (gamma_t**2)

    def get_noise_from_score(self, score: Tensor, t: Tensor) -> Tensor:
        """Convert score to ẑ_t = (x_t - α_t x̂0 - β_t x_T) / γ_t."""
        gamma_t, _ = self.gamma_t_with_derivative(t)

        return -gamma_t * score

    def get_score_from_noise(self, noise: Tensor, t: Tensor) -> Tensor:
        """Inverse of `get_noise_from_score`."""
        gamma_t, _ = self.gamma_t_with_derivative(t)
        return -noise / gamma_t

    def get_noise_from_x_0_x_t(self, x_0: Tensor, x_t: Tensor, t: Tensor, x_1: Tensor) -> Tensor:
        """Return ẑ_t = (x_t - α_t x_0 - β_t x_T) / γ_t."""
        alpha_t, _ = self.alpha_t_with_derivative(t)
        beta_t, _ = self.beta_t_with_derivative(t)
        gamma_t, _ = self.gamma_t_with_derivative(t)

        noise = (x_t - alpha_t * x_0 - beta_t * x_1) / gamma_t

        return noise

    def get_x_t(
        self, t: Tensor, x_0: Tensor, x_1: Tensor | None = None, z: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Sample x_t from p_{t|0,T}(x_t | x_0, x_T) = N(α_t x_0 + β_t x_T, γ_t^2 I)."""
        if x_1 is None:
            x_1 = torch.randn_like(x_0)

        t = self.expand_t_as(t, x_0).to(torch.float32)

        # x_t
        alpha_t, alpha_t_prime = self.alpha_t_with_derivative(t)
        beta_t, beta_t_prime = self.beta_t_with_derivative(t)
        gamma_t, gamma_t_prime = self.gamma_t_with_derivative(t)

        if z is None:
            z = torch.randn_like(x_0)

        x_t = alpha_t * x_0 + beta_t * x_1 + gamma_t * z

        return x_t, x_1

    def get_x_t_with_target(self, t: Tensor, x_0: Tensor, x_1: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Return a training pair (x_t, target) according to `plan_tgt`."""
        t = self.expand_t_as(t, x_0).to(torch.float32)

        alpha_t, alpha_t_prime = self.alpha_t_with_derivative(t)
        beta_t, beta_t_prime = self.beta_t_with_derivative(t)

        x_t, x_1 = self.get_x_t(t, x_0, x_1)

        # target
        if self.plan_tgt == DiffusionTarget.velocity:
            tgt = x_0 * alpha_t_prime + x_1 * beta_t_prime
        elif self.plan_tgt == DiffusionTarget.score:
            tgt = self.get_score_from_x_0_x_t(x_0, x_t, t, x_1)
        elif self.plan_tgt == DiffusionTarget.x_0:
            tgt = x_0
        elif self.plan_tgt == DiffusionTarget.noise:
            tgt = self.get_noise_from_x_0_x_t(x_0, x_t, t, x_1)
        else:
            raise ValueError(f"Unsupported diffusion target: {self.plan_tgt}")

        return x_t, tgt

    def get_sde_drift_diffusion(self, t: Tensor, x_0: Tensor, x_t: Tensor, x_1: Tensor, delta_t: float):
        """Compute drift and diffusion for the reverse sampling SDE.

        Paper: arXiv:2410.21553v2, Proposition 4.1 (Eq. (10)) and Euler discretization (Eq. (13)).
        The drift term corresponds to b(t, x_t, x_T) with x̂0 approximated by the network output `x_0`.
        """
        t = self.expand_t_as(t, x_0).to(torch.float32)

        # x_0 is the network prediction
        alpha_t, alpha_t_prime = self.alpha_t_with_derivative(t)
        beta_t, beta_t_prime = self.beta_t_with_derivative(t)
        gamma_t, gamma_t_prime = self.gamma_t_with_derivative(t)
        eps_t = self.epsilon_t(t)

        # noise = (x_t - alpha_t * x_0 - beta_t * x_1) / gamma_t
        noise = self.get_noise_from_x_0_x_t(x_0, x_t, t, x_1)

        # Matches Eq. (10) of Proposition 4.1 in arXiv:2410.21553v2
        # b(t, x_t, x_T) = alpha_dot * x_0 + beta_dot * x_T + (gamma_dot + eps_t / gamma_t) * z_hat
        drift = alpha_t_prime * x_0 + beta_t_prime * x_1 + (gamma_t_prime + eps_t / gamma_t) * noise

        diffusion = (2 * eps_t * delta_t).sqrt()

        return drift, diffusion


# * Stochastic Control Transport Sampler ========================================

# version 1: SDE sampler
# version 2: ODE sampler


class SDBContinuousSampler(SDBSampler):
    """Sampler for the ECSI reverse process.

    Paper: arXiv:2410.21553v2, Eq. (13) and Algorithm 1.

    This sampler distinguishes two endpoints when `sample_noisy_x1_b > 0`:
        - clean condition `x_cond` (Algorithm 1 line 2: x_T ~ π_cond), used to condition the model;
        - noisy base endpoint `x_T_base = x_cond + n0` (Algorithm 1 line 3: x_N), used inside the kernel.
    """

    def __init__(self, plan: SDBContinuousPlan, sample_noisy_x1_b: float = 0.0):
        super().__init__(plan)
        self.plan: SDBContinuousPlan = plan
        self.sample_noisy_x1_b = sample_noisy_x1_b  # first noise the x_1

    def model_pred_to_x_0(
        self,
        model: nn.Module,
        x_t: Tensor,
        t: Tensor,
        x_cond: Tensor,
        x_T_base: Tensor,
        model_kwargs: dict | None = None,
    ):
        """Convert the model output to x̂0 depending on `plan_tgt`.

        Paper uses a denoiser Dθ that predicts x̂0 by minimizing Eq. (11).
        """
        model_kwargs = {} if model_kwargs is None else model_kwargs

        # EDM precond model, the first output is the output tensor
        with torch.no_grad():
            cond_kwargs = _maybe_add_condition_kwargs(model, model_kwargs, x_1=x_cond)
            model_out = model(x_t, t, **cond_kwargs)
            if isinstance(model_out, Sequence):
                model_out = model_out[0]

        if self.plan_tgt == DiffusionTarget.score:
            x_0_pred = self.plan.get_x0_from_score(model_out, t, x_t, x_T_base)
        elif self.plan_tgt == DiffusionTarget.x_0:
            x_0_pred = model_out
        elif self.plan_tgt == DiffusionTarget.noise:
            x_0_pred = self.plan.get_x0_from_score(
                self.plan.get_score_from_noise(model_out, t),
                t,
                x_t,
                x_T_base,
            )
        else:
            raise ValueError(f"Unsupported diffusion target: {self.plan_tgt}")

        return x_0_pred

    def step_mean(
        self,
        model: nn.Module,
        x_t: Tensor,
        t: Tensor,
        x_cond: Tensor,
        x_T_base: Tensor,
        delta_t: float,
        model_kwargs: dict | None = None,
        clip_value: bool = True,
        *,
        prev_x_0_pred: Tensor | None = None,
    ):
        """Deterministic reverse step (paper Eq. (12) with ϵ_t = 0)."""
        model_kwargs = {} if model_kwargs is None else model_kwargs

        # get x_0_pred
        if prev_x_0_pred is None:
            x_0_pred = self.model_pred_to_x_0(model, x_t, t, x_cond, x_T_base, model_kwargs)
        else:
            x_0_pred = prev_x_0_pred

        # clip value
        if clip_value:
            x_0_pred = x_0_pred.clamp(-1, 1)

        # get noise
        z_t_pred = self.plan.get_noise_from_score(
            self.plan.get_score_from_x_0_x_t(x_0_pred, x_t, t, x_T_base),
            t,
        )

        # no diffusion term, directly to x mean
        x_t, _ = self.plan.get_x_t(t - delta_t, x_0_pred, x_T_base, z_t_pred)

        return x_t, x_0_pred

    def step_sde(
        self,
        model: nn.Module,
        x_t: Tensor,
        t: Tensor,
        x_cond: Tensor,
        x_T_base: Tensor,
        delta_t: float,
        model_kwargs: dict | None = None,
        clip_value: bool = False,
        *,
        prev_x_0_pred: Tensor | None = None,
    ):
        """Stochastic reverse step (paper Eq. (13))."""
        model_kwargs = {} if model_kwargs is None else model_kwargs
        t = t.float()

        if prev_x_0_pred is None:
            x_0_pred = self.model_pred_to_x_0(model, x_t, t, x_cond, x_T_base, model_kwargs)
        else:
            x_0_pred = prev_x_0_pred

        # clip value in pixel space
        if clip_value:
            x_0_pred = x_0_pred.clamp(-1, 1)

        # get drift and diffusion
        drift, diffusion = self.plan.get_sde_drift_diffusion(t, x_0_pred, x_t, x_T_base, delta_t)

        # update x_t
        x_t = x_t - drift * delta_t + diffusion * torch.randn_like(x_t)

        return x_t, x_0_pred

    def sample_sde_euler(
        self,
        model: nn.Module,
        x_1: Tensor,
        time_grid: Tensor | None = None,
        sample_n_steps: int | None = None,
        last_n_step_only_mean: int = 1,
        traj_saved_n: int = 5,
        model_kwargs: dict | None = None,
        clip_value: bool = False,
        progress: bool = True,
    ):
        """Euler sampler for the reverse SDE (paper Eq. (13), Algorithm 1)."""
        model_kwargs = {} if model_kwargs is None else model_kwargs
        assert time_grid is not None or sample_n_steps is not None, (
            "Either time_grid or sample_n_steps must be provided"
        )
        assert traj_saved_n > 0, "traj_saved_n must be greater than 0"

        x_cond = x_1
        x_T_base = self.pre_noise_x_1(x_cond)

        if time_grid is None:
            assert sample_n_steps is not None
            time_grid = _time_grid_from_plan(self.plan, device=x_cond.device, steps=sample_n_steps)
        else:
            time_grid = time_grid.to(device=x_cond.device)
            sample_n_steps = len(time_grid)

        # force time grid to be float32 dtype
        time_grid = time_grid.float()
        save_indices = _traj_save_indices(len(time_grid), traj_saved_n)

        # loops
        _remain_n_steps = len(time_grid) - last_n_step_only_mean - 1
        saved_x0_traj = []
        x_t = x_T_base
        saved_xt_traj = [x_t]
        if not progress:
            tbar = enumerate(zip(time_grid[:-1], time_grid[1:]), start=1)
        else:
            from tqdm.auto import tqdm

            tbar = tqdm(enumerate(zip(time_grid[:-1], time_grid[1:]), start=1), total=len(time_grid) - 1, leave=False)

        last_i = 0
        for i, (t_scalar, t_prev_scalar) in tbar:
            last_i = i
            delta_t = float(t_scalar - t_prev_scalar)
            t = self.plan.expand_t_as(t_scalar.unsqueeze(0).expand(x_t.shape[0]), x_t)

            if _remain_n_steps > 0:
                # step with SDE (drift + diffusion)
                x_t, x_0_pred = self.step_sde(model, x_t, t, x_cond, x_T_base, delta_t, model_kwargs, clip_value)
            else:
                # step by mean
                x_t, x_0_pred = self.step_mean(model, x_t, t, x_cond, x_T_base, delta_t, model_kwargs, clip_value)

            _remain_n_steps -= 1
            if i in save_indices:
                saved_x0_traj.append(x_0_pred)
                saved_xt_traj.append(x_t)

        if last_i not in save_indices:
            saved_x0_traj.append(x_t)
            saved_xt_traj.append(x_t)

        return x_t, saved_x0_traj, saved_xt_traj

    def sample_ode_euler(
        self,
        model: nn.Module,
        x_1: Tensor,
        time_grid: Tensor | None = None,
        sample_n_steps: int | None = None,
        last_n_step_only_mean: int = 1,
        traj_saved_n: int = 5,
        model_kwargs: dict | None = None,
        clip_value: bool = False,
        progress: bool = True,
    ):
        """Euler sampler for the probability-flow ODE (paper Eq. (13) with ϵ_t = 0).

        For the last `last_n_step_only_mean` steps, this uses the paper's deterministic
        re-parameterization (Eq. (12) reduced with ϵ_t = 0), matching Algorithm 1.
        """
        model_kwargs = {} if model_kwargs is None else model_kwargs
        assert time_grid is not None or sample_n_steps is not None, (
            "Either time_grid or sample_n_steps must be provided"
        )
        assert traj_saved_n > 0, "traj_saved_n must be greater than 0"

        x_cond = x_1
        x_T_base = self.pre_noise_x_1(x_cond)

        if time_grid is None:
            assert sample_n_steps is not None
            time_grid = _time_grid_from_plan(self.plan, device=x_cond.device, steps=sample_n_steps)
        else:
            time_grid = time_grid.to(device=x_cond.device)
            sample_n_steps = len(time_grid)

        # force time grid to be float32 dtype
        time_grid = time_grid.float()
        save_indices = _traj_save_indices(len(time_grid), traj_saved_n)

        # loops
        _remain_n_steps = len(time_grid) - last_n_step_only_mean - 1
        saved_x0_traj = []
        x_t = x_T_base
        saved_xt_traj = [x_t]
        if not progress:
            tbar = enumerate(zip(time_grid[:-1], time_grid[1:]), start=1)
        else:
            from tqdm.auto import tqdm

            tbar = tqdm(enumerate(zip(time_grid[:-1], time_grid[1:]), start=1), total=len(time_grid) - 1, leave=False)

        last_i = 0
        for i, (t_scalar, t_prev_scalar) in tbar:
            last_i = i
            delta_t = float(t_scalar - t_prev_scalar)
            t = self.plan.expand_t_as(t_scalar.unsqueeze(0).expand(x_t.shape[0]), x_t)

            if _remain_n_steps > 0:
                x_0_pred = self.model_pred_to_x_0(model, x_t, t, x_cond, x_T_base, model_kwargs)
                if clip_value:
                    x_0_pred = x_0_pred.clamp(-1, 1)

                z_hat = self.plan.get_noise_from_x_0_x_t(x_0_pred, x_t, t, x_T_base)
                _, alpha_prime = self.plan.alpha_t_with_derivative(t)
                _, beta_prime = self.plan.beta_t_with_derivative(t)
                _, gamma_prime = self.plan.gamma_t_with_derivative(t)
                drift_ode = alpha_prime * x_0_pred + beta_prime * x_T_base + gamma_prime * z_hat
                x_t = x_t - drift_ode * delta_t
            else:
                # last steps: use the deterministic re-parameterization in the paper
                x_t, x_0_pred = self.step_mean(model, x_t, t, x_cond, x_T_base, delta_t, model_kwargs, clip_value)

            _remain_n_steps -= 1

            if i in save_indices:
                saved_x0_traj.append(x_0_pred)
                saved_xt_traj.append(x_t)

        if last_i not in save_indices:
            saved_x0_traj.append(x_t)
            saved_xt_traj.append(x_t)

        return x_t, saved_x0_traj, saved_xt_traj

    def sample_sde_midpoint(
        self,
        model: nn.Module,
        x_1: Tensor,
        time_grid: Tensor | None = None,
        sample_n_steps: int | None = None,
        last_n_step_only_mean: int = 1,
        traj_saved_n: int = 5,
        model_kwargs: dict | None = None,
        clip_value: bool = False,
        progress: bool = True,
        use_pred_x_0: bool = False,
    ):
        """Midpoint sampler (two half Euler SDE steps)."""
        model_kwargs = {} if model_kwargs is None else model_kwargs
        assert time_grid is not None or sample_n_steps is not None, (
            "Either time_grid or sample_n_steps must be provided"
        )
        assert traj_saved_n > 0, "traj_saved_n must be greater than 0"

        x_cond = x_1
        x_T_base = self.pre_noise_x_1(x_cond)

        if time_grid is None:
            assert sample_n_steps is not None
            time_grid = _time_grid_from_plan(self.plan, device=x_cond.device, steps=sample_n_steps)
        else:
            time_grid = time_grid.to(device=x_cond.device)
            sample_n_steps = len(time_grid)

        time_grid = time_grid.float()
        save_indices = _traj_save_indices(len(time_grid), traj_saved_n)

        _remain_n_steps = len(time_grid) - last_n_step_only_mean - 1

        # loops
        saved_x0_traj = []
        saved_xt_traj = []
        x_t = x_T_base
        if not progress:
            tbar = enumerate(zip(time_grid[:-1], time_grid[1:]), start=1)
        else:
            from tqdm.auto import tqdm

            tbar = tqdm(enumerate(zip(time_grid[:-1], time_grid[1:]), start=1), total=len(time_grid) - 1, leave=False)

        last_i = 0
        for i, (t_scalar, t_prev_scalar) in tbar:
            last_i = i
            delta_t = float(t_scalar - t_prev_scalar)
            t = self.plan.expand_t_as(t_scalar.unsqueeze(0).expand(x_t.shape[0]), x_t)

            if _remain_n_steps > 0:
                # first step x_t -> x_mid
                x_t_mid, x_0_pred_mid = self.step_sde(
                    model, x_t, t, x_cond, x_T_base, delta_t / 2, model_kwargs, clip_value
                )

                # prepare t_mid for second step
                t_mid_scalar = t_scalar - delta_t / 2
                t_mid = self.plan.expand_t_as(t_mid_scalar.unsqueeze(0).expand(x_t.shape[0]), x_t)

                # second step x_mid -> x_t
                x_t, x_0_pred = self.step_sde(
                    model,
                    x_t_mid,
                    t_mid,
                    x_cond,
                    x_T_base,
                    delta_t / 2,
                    model_kwargs,
                    clip_value,
                    prev_x_0_pred=x_0_pred_mid if use_pred_x_0 else None,
                )
            else:
                # step by mean
                x_t, x_0_pred = self.step_mean(model, x_t, t, x_cond, x_T_base, delta_t, model_kwargs, clip_value)

            _remain_n_steps -= 1
            if i in save_indices:
                saved_x0_traj.append(x_0_pred)
                saved_xt_traj.append(x_t)

        if last_i not in save_indices:
            saved_x0_traj.append(x_t)
            saved_xt_traj.append(x_t)

        return x_t, saved_x0_traj, saved_xt_traj

    def pre_noise_x_1(self, x_1: Tensor) -> Tensor:
        """Sample from the modified base distribution π_T = π_cond * N(0, b^2 I).

        Paper: arXiv:2410.21553v2, Algorithm 1 line 2-3.
        """
        if self.sample_noisy_x1_b <= 0:
            return x_1
        return x_1 + torch.randn_like(x_1) * self.sample_noisy_x1_b
