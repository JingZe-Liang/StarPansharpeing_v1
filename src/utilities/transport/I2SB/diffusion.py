import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from functools import partial
from loguru import logger
from typing import cast

logger = logger.bind(_name_="I2SB")


def make_beta_schedule(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    betas = torch.linspace(linear_start**0.5, linear_end**0.5, n_timestep, dtype=torch.float64) ** 2
    return betas.numpy()


def unsqueeze_xdim(z, xdim):
    bc_dim = (...,) + (None,) * len(xdim)
    return z[bc_dim]


def compute_gaussian_product_coef(sigma1, sigma2):
    """Given p1 = N(x_t|x_0, sigma_1**2) and p2 = N(x_t|x_1, sigma_2**2)
    return p1 * p2 = N(x_t| coef1 * x0 + coef2 * x1, var)"""

    denom = sigma1**2 + sigma2**2
    coef1 = sigma2**2 / denom
    coef2 = sigma1**2 / denom
    var = (sigma1**2 * sigma2**2) / denom
    return coef1, coef2, var


def space_indices(num_steps, count):
    assert count <= num_steps

    if count <= 1:
        frac_stride = 1
    else:
        frac_stride = (num_steps - 1) / (count - 1)

    cur_idx = 0.0
    taken_steps = []
    for _ in range(count):
        taken_steps.append(round(cur_idx))
        cur_idx += frac_stride

    return taken_steps


class Diffusion(nn.Module):
    def __init__(
        self,
        n_timestep=1000,
        linear_start=1e-4,
        beta_max: float = 0.3,
        linear_end: float | None = None,
        model_pred="mu",
        ot_ode=False,
    ):
        super().__init__()
        self.model_pred = model_pred
        assert model_pred in ("mu", "x0")
        self.n_timestep = n_timestep
        self.ot_ode = ot_ode

        linear_end = linear_end if linear_end is not None else beta_max / n_timestep
        betas = make_beta_schedule(n_timestep=n_timestep, linear_start=linear_start, linear_end=linear_end)
        betas = np.concatenate([betas[: n_timestep // 2], np.flip(betas[: n_timestep // 2])])

        # compute analytic std: eq 11
        std_fwd = np.sqrt(np.cumsum(betas))
        std_bwd = np.sqrt(np.flip(np.cumsum(np.flip(betas))))
        mu_x0, mu_x1, var = compute_gaussian_product_coef(std_fwd, std_bwd)
        std_sb = np.sqrt(var)

        # tensorize everything
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.betas = nn.Buffer(to_torch(betas), persistent=False)
        self.std_fwd = nn.Buffer(to_torch(std_fwd), persistent=False)
        self.std_bwd = nn.Buffer(to_torch(std_bwd), persistent=False)
        self.std_sb = nn.Buffer(to_torch(std_sb), persistent=False)
        self.mu_x0 = nn.Buffer(to_torch(mu_x0), persistent=False)
        self.mu_x1 = nn.Buffer(to_torch(mu_x1), persistent=False)

    @property
    def device(self) -> torch.device:
        return self.betas.device

    def get_std_fwd(self, step, xdim=None):
        std_fwd = self.std_fwd[step]
        return std_fwd if xdim is None else unsqueeze_xdim(std_fwd, xdim)

    def q_sample(self, step, x0, x1, ot_ode=False):
        """Sample q(x_t | x_0, x_1), i.e. eq 11"""

        assert x0.shape == x1.shape
        batch, *xdim = x0.shape

        mu_x0 = unsqueeze_xdim(self.mu_x0[step], xdim)
        mu_x1 = unsqueeze_xdim(self.mu_x1[step], xdim)
        std_sb = unsqueeze_xdim(self.std_sb[step], xdim)

        xt = mu_x0 * x0 + mu_x1 * x1
        if not ot_ode:
            xt = xt + std_sb * torch.randn_like(xt)
        return xt.detach()

    def get_target(self, x0, xt, t):
        std_fwd = self.get_std_fwd(t, xdim=x0.shape[1:])
        label = (xt - x0) / std_fwd
        return label.detach()

    def p_posterior(self, nprev, n, x_n, x0, ot_ode=False):
        """Sample p(x_{nprev} | x_n, x_0), i.e. eq 4"""

        assert nprev < n
        std_n = self.std_fwd[n]
        std_nprev = self.std_fwd[nprev]
        std_delta = (std_n**2 - std_nprev**2).sqrt()

        mu_x0, mu_xn, var = compute_gaussian_product_coef(std_nprev, std_delta)

        xt_prev = mu_x0 * x0 + mu_xn * x_n
        if not ot_ode and nprev > 0:
            xt_prev = xt_prev + var.sqrt() * torch.randn_like(xt_prev)

        return xt_prev

    def ddpm_sampling(self, steps, pred_x0_fn, x1, mask=None, ot_ode=False, log_steps=None, verbose=True):
        xt = x1.detach().to(self.device)

        xs = []
        pred_x0s = []

        log_steps = log_steps or steps
        assert steps[0] == log_steps[0] == 0

        steps = steps[::-1]

        pair_steps = zip(steps[1:], steps[:-1])
        pair_steps = tqdm(pair_steps, desc="DDPM sampling", total=len(steps) - 1) if verbose else pair_steps
        for prev_step, step in pair_steps:
            assert prev_step < step, f"{prev_step=}, {step=}"

            pred_x0 = pred_x0_fn(xt, step)
            xt = self.p_posterior(prev_step, step, xt, pred_x0, ot_ode=ot_ode)

            if mask is not None:
                xt_true = x1
                if not ot_ode:
                    _prev_step = torch.full((xt.shape[0],), prev_step, device=self.device, dtype=torch.long)  # type: ignore
                    std_sb = unsqueeze_xdim(self.std_sb[_prev_step], xdim=x1.shape[1:])
                    xt_true = xt_true + std_sb * torch.randn_like(xt_true)
                xt = (1.0 - mask) * xt_true + mask * xt

            if prev_step in log_steps:
                pred_x0s.append(pred_x0.detach().cpu())
                xs.append(xt.detach().cpu())

        stack_bwd_traj = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
        return stack_bwd_traj(xs), stack_bwd_traj(pred_x0s)

    def compute_pred_x0(self, step, xt, net_out, clip_denoise=False):
        """Given network output, recover x0. This should be the inverse of Eq 12"""
        if self.model_pred == "x0":
            return net_out
        else:
            std_fwd = self.get_std_fwd(step, xdim=xt.shape[1:])
            pred_x0 = xt - std_fwd * net_out
            if clip_denoise:
                pred_x0.clamp_(-1.0, 1.0)
            return pred_x0

    def training_loss(self, model, x0, x1, model_kwargs: dict = {}):
        bs = x0.shape[0]
        t = torch.randint(0, self.n_timestep, (bs,), device=x0.device, dtype=torch.long)
        xt = self.q_sample(t, x0, x1, ot_ode=self.ot_ode)

        # model forward
        net_out = model(xt, t, **model_kwargs)
        target = self.get_target(x0, xt, t)
        if self.model_pred == "x0":
            loss = F.mse_loss(net_out, x0)
        else:
            loss = F.mse_loss(net_out, target)
        x0_hat = self.compute_pred_x0(t, xt, net_out)

        return {"loss": loss, "pred_x0": x0_hat}

    def sample(
        self,
        model,
        x1,
        clip_denoise=False,
        nfe=None,
        model_kwargs: dict = {},
        log_steps: list[int] | None = None,
        ot_ode: bool | None = None,
        log_count=0,
        verbose=False,
    ):
        nfe = nfe or self.n_timestep - 1
        assert 0 < nfe < self.n_timestep == len(self.betas)
        time_grid = space_indices(self.n_timestep, nfe + 1)

        log_count = min(len(time_grid) - 1, log_count)
        log_steps = [time_grid[i] for i in space_indices(len(time_grid) - 1, log_count)]
        assert log_steps[0] == 0
        logger.trace(f"[DDPM Sampling] steps={self.n_timestep}, {nfe=}, {log_steps=}!")

        def pred_x0_fn(xt, t):
            step = torch.full((xt.shape[0],), t, device=x1.device, dtype=torch.long)
            out = model(xt, step, **model_kwargs)
            return self.compute_pred_x0(step, xt, out, clip_denoise=clip_denoise)

        xs, pred_x0 = self.ddpm_sampling(
            time_grid,
            pred_x0_fn,
            x1,
            mask=None,
            ot_ode=ot_ode or self.ot_ode,
            log_steps=log_steps,
            verbose=verbose,
        )

        b, *xdim = x1.shape
        assert xs.shape == pred_x0.shape == (b, log_count, *xdim)

        return {"sampled": pred_x0, "traj": xs}
