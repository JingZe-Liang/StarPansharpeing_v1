import math
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict
from loguru import logger
from torch import Tensor, nn
from torch.autograd.functional import jvp

from .transport import ModelType, PathType, Transport, WeightType


def expand_as(t, x):
    return t.view(t.shape + (1,) * (x.ndim - t.ndim))


def expand_to(t, bs, total=4):
    return t.view(bs, *([1] * (total - 1)))


def sample_logit_norm_time(
    time_sampling_cfg, shape: torch.Size | tuple, device: torch.device | None = None
) -> torch.Tensor:
    """
    Time Samples following the Logit Normal distribution of Stable Diffusion 3
    Produces times in [0, 1-eps] following "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow"
    """
    randn = (
        torch.randn(shape, device=device) * time_sampling_cfg.scale
        + time_sampling_cfg.location
    )  # [b, ...]
    logit_normal = randn.sigmoid()  # [b, ...]
    logit_normal_rescaled = logit_normal * (
        1 - time_sampling_cfg.eps
    )  # [b, ...]. Rescales between [0, 1-eps]

    return logit_normal_rescaled


def sample_truncated_logit_norm_time(
    time_sampling_cfg,
    batch_size: int,
    device: torch.device | None = None,
    upper_truncated=None,
) -> torch.Tensor:
    """
    Time Samples following the Logit Normal distribution of Stable Diffusion 3
    Produces times in [0, 1-eps] following "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow"
    """
    from torchrl.modules.distributions import TruncatedNormal

    assert upper_truncated is not None, "upper_truncated must be specified"
    loc = torch.ones(batch_size, device=device) * time_sampling_cfg.location  # [b]
    scale = torch.ones(batch_size, device=device) * time_sampling_cfg.scale  # [b]
    trunc_randn = TruncatedNormal(
        loc=loc, scale=scale, low=-float("Inf"), high=torch.logit(upper_truncated)
    ).sample()  # [b]
    logit_normal = torch.sigmoid(trunc_randn)  # [b]

    return logit_normal


class AlphaFlow(nn.Module):
    def __init__(
        self,
        # alpha-flow specific configs
        fm_ratio_cfg,
        alpha_cfg,
        time_sampling_cfg,
        distrib_t_t_next_mf_cfg,
        CFG_cfg,
    ):
        super().__init__(self)
        self.ratio_fm = fm_ratio_cfg
        self.alpha = alpha_cfg
        self.time_sampling = time_sampling_cfg
        self.distrib_t_t_next_mf = distrib_t_t_next_mf_cfg
        self.cfg_params = CFG_cfg

        self.clamp_utgt = 1e4
        self.adaptive_loss_weight_eps = 1e-4

    def _sample_timestep(
        self,
        sampling_cfg,
        cur_step: int,
        batch_size: int,
        device: torch.device,
        upper_truncated: Optional[float] = None,
    ):
        ### logit norm
        if sampling_cfg.timestep_distrib_type == "logit_norm":
            return sample_logit_norm_time(sampling_cfg, batch_size, device=device)

        ### truncated logit norm
        elif sampling_cfg.timestep_distrib_type == "truncated_logit_norm":
            assert upper_truncated is not None
            return sample_truncated_logit_norm_time(
                sampling_cfg, batch_size, device=device, upper_truncated=upper_truncated
            )

        ### Beta distribution
        elif sampling_cfg.timestep_distrib_type == "adaptive_beta":
            # Linearly interpolate alpha and beta from initial to end values between init_steps and end_steps
            if cur_step < sampling_cfg.init_steps:
                alpha, beta = sampling_cfg.initial_alpha, sampling_cfg.initial_beta
            elif cur_step > sampling_cfg.end_steps:
                alpha, beta = sampling_cfg.end_alpha, sampling_cfg.end_beta
            else:
                progress = (cur_step - sampling_cfg.init_steps) / max(
                    1, (sampling_cfg.end_steps - sampling_cfg.init_steps)
                )
                alpha = (
                    sampling_cfg.initial_alpha
                    + (sampling_cfg.end_alpha - sampling_cfg.initial_alpha) * progress
                )
                beta = (
                    sampling_cfg.initial_beta
                    + (sampling_cfg.end_beta - sampling_cfg.initial_beta) * progress
                )
            beta_distrib = torch.distributions.Beta(
                torch.tensor([alpha]), torch.tensor([beta])
            )
            return beta_distrib.sample((batch_size,), device=device)

        ### Uniform distribution
        elif sampling_cfg.timestep_distrib_type == "uniform":
            return (
                torch.rand((batch_size,), device=device)
                * (sampling_cfg.max - sampling_cfg.min)
                + sampling_cfg.min
            )

        ### Constant distribution
        elif sampling_cfg.timestep_distrib_type == "constant":
            return torch.ones((batch_size,), device=device) * sampling_cfg.scale

        ## Arctan timesteps
        elif sampling_cfg.timestep_distrib_type == "arctan":
            sigma = torch.exp(
                torch.randn((batch_size,), device=device) * sampling_cfg.scale
                + sampling_cfg.location
            )
            return 2 / math.pi * torch.atan(sigma)

    def _sample_timestep_mf(
        self, cfg, cur_step, batch_size, device
    ) -> tuple[Tensor, Tensor]:
        if cfg.type == "truncated":
            t = self._sample_timestep(
                cfg.time_sampling_mf_t, cur_step, batch_size, device=device
            )
            t_next = self._sample_timestep(
                cfg.time_sampling_mf_t_next,
                cur_step,
                batch_size,
                device=device,
                upper_truncated=t,
            )
        elif cfg.type in ["minmax", "min", "r_in_t_range"]:
            t_1 = self._sample_timestep(
                cfg.time_sampling_mf_t, cur_step, batch_size, device=device
            )
            t_2 = self._sample_timestep(
                cfg.time_sampling_mf_t_next, cur_step, batch_size, device=device
            )

            # make sure t > r
            if cfg.type == "minmax":
                t = torch.maximum(t_1, t_2)
                t_next = torch.minimum(t_1, t_2)
            elif cfg.type == "min":
                t = t_1
                t_next = torch.minimum(t_1, t_2)
            elif cfg.type == "r_in_t_range":
                t = t_1
                t_next = t_2 * t_1
            else:
                raise NotImplementedError(
                    f"Unknown meanflow distribution type: {cfg.type}"
                )

        return t, t_next

    def _get_time_ratio(self, cfg, cur_step: int):
        if cfg.scheduler == "constant":
            current_ratio = cfg.initial_value
        elif cfg.scheduler == "step":
            assert cfg.change_init_steps == cfg.change_end_steps, (
                "For step scheduler, change_init_steps and change_end_steps must be equal."
            )
            current_ratio = (
                cfg.initial_value if cur_step < cfg.change_init_steps else cfg.end_value
            )
        elif cfg.scheduler in ["linear", "exponential", "log", "sigmoid"]:
            if cur_step < cfg.change_init_steps:
                current_ratio = cfg.initial_value
            elif cur_step > cfg.change_end_steps:
                current_ratio = cfg.end_value
            else:
                if cfg.scheduler in ["linear", "exponential", "log"]:
                    progress = (cur_step - cfg.change_init_steps) / (
                        cfg.change_end_steps - cfg.change_init_steps
                    )
                elif cfg.scheduler == "sigmoid":
                    middle_step = (
                        cfg.change_init_steps
                        + (cfg.change_end_steps - cfg.change_init_steps) / 2
                    )
                    progress = (cur_step - middle_step) / (
                        cfg.change_end_steps - cfg.change_init_steps
                    )

                if cfg.scheduler == "linear":
                    current_ratio = (
                        cfg.initial_value
                        + (cfg.end_value - cfg.initial_value) * progress
                    )
                elif cfg.scheduler == "exponential":
                    progress = progress**cfg.gamma
                    current_ratio = cfg.initial_value * (
                        (cfg.end_value / cfg.initial_value) ** progress
                    )
                elif cfg.scheduler == "log":
                    log_progress = math.log(1 + progress * 9) / math.log(10)
                    current_ratio = (
                        cfg.initial_value
                        + (cfg.end_value - cfg.initial_value) * log_progress
                    )
                elif cfg.scheduler == "sigmoid":
                    current_ratio = cfg.initial_value + (
                        cfg.end_value - cfg.initial_value
                    ) * (1 / (1 + math.exp(-progress * cfg.gamma)))
        else:
            raise NotImplementedError(f"Unknown scheduler type: {cfg.scheduler}")

        if current_ratio < cfg.clamp_value:
            current_ratio = 0.0
            if "discrete_training" in cfg and cfg.discrete_training:
                current_ratio = cfg.clamp_value
        elif current_ratio > 1 - cfg.clamp_value or (
            cfg.up_clamp_value is not None and current_ratio > cfg.up_clamp_value
        ):
            current_ratio = 1.0
        return current_ratio

    def sample(self, x1: Tensor, cur_step):
        # FM and MF batch size ratios
        ratio_fm = self._get_time_ratio(self.ratio_fm, cur_step)
        alpha = self._get_time_ratio(self.alpha, cur_step)
        batch_size = x1.shape[0]
        device = x1.device

        # N batchsize for Flow matching, not MF
        bs_fm = int(batch_size * ratio_fm)
        bs_mf = batch_size - bs_fm

        # Timesteps for Flow Matching
        t_fm: Tensor = self._sample_timestep(
            self.time_sampling, cur_step, bs_fm, device=device
        )
        t_next_fm: Tensor = t_fm
        dt_fm = torch.zeros_like(t_next_fm)

        # Timesteps for Mean Flow
        t_mf, t_next_mf = self._sample_timestep_mf(
            self.distrib_t_t_next_mf, cur_step, bs_mf, device=device
        )
        dt_mf = alpha * (t_mf - t_next_mf)

        # t and t_next
        t = torch.cat([t_fm, t_mf], dim=0)
        t_next = torch.cat([t_next_fm, t_next_mf], dim=0)
        dt = torch.cat([dt_fm, dt_mf], dim=0)

        return expand_as(t, x1), expand_as(t_next, x1), expand_as(dt, x1), alpha

    def _apply_noise(self, x, noise_scaled, t, model):
        xt = (1 - t) * x + self.time_sampling.sigma_noise * noise_scaled
        return xt

    @torch.no_grad()
    def _compute_velocity_cfg(
        self,
        velocity: Tensor,
        x_t,
        t,
        cond,
        model: nn.Module | Callable,
        batch_size: int,
        drop_label: bool = False,
        model_kwargs: dict = {},
    ):
        cfg_params = self.cfg_params
        # Create classifier free guidance mask for t: True
        # where self.cfg.cfg_params.t_min < t < self.cfg.cfg_params.t_max
        t_flat = t.view(batch_size, -1)
        omega = cfg_params.omega
        kappa = cfg_params.kappa
        mask = (t_flat > cfg_params.t_min) & (t_flat < cfg_params.t_max)
        cfg_mask_idx = mask.view(batch_size).bool()  # [b]
        velocity_cfg = velocity.clone()

        # Drop labels, with probability self.cfg.label_dropout, set cond.label[...] to zero vector
        # if self.cfg.model.label_dropout > 0.0:

        if drop_label:
            ## !! not used
            raise
            label_drop_mask_idx = (
                torch.rand(cond.label.shape[0], device=cond.label.device)
                < self.cfg.model.label_dropout
            )
            drop_mask = cfg_mask_idx & label_drop_mask_idx
            cond.label[drop_mask] = torch.zeros_like(cond.label[drop_mask])
        else:
            drop_mask = cfg_mask_idx

        if 1 - omega - kappa != 0.0:
            videos_u_t_t_uncond = model(
                x_t,
                t,
                t,
                cond=None,
                **model_kwargs,
                # augment_labels=augment_labels,
                # return_extra_output=False,
            )
        else:
            videos_u_t_t_uncond = torch.zeros_like(x_t)

        if kappa != 0:
            videos_u_t_t_cond = model(
                x_t,
                t,
                t,
                cond=None,
                **model_kwargs,
                # augment_labels=augment_labels,
                # return_extra_output=False,
            )
        else:
            videos_u_t_t_cond = torch.zeros_like(x_t)

        guided = (
            omega * velocity
            + kappa * videos_u_t_t_cond
            + (1 - omega - kappa) * videos_u_t_t_uncond
        )
        velocity_cfg[cfg_mask_idx] = guided[cfg_mask_idx]
        return velocity_cfg

    @torch.no_grad()
    def _compute_mean_velocity_c(
        self,
        x_t,
        t_next,
        t,
        velocity_cfg,
        cond,
        model_kwargs: dict,
        net: nn.Module | Callable,
    ):
        # Continues is empty
        if x_t.shape[0] == 0:
            return torch.empty((0, *velocity_cfg.shape[1:]), device=velocity_cfg.device)

        t = t.flatten()  # [b]
        t_next = t_next.flatten()  # [b]

        mask_mf = ~torch.isclose(t_next, t)  # [b]
        batch_size_mf = mask_mf.sum().item()
        mean_velocity = velocity_cfg.clone()

        def wrap_net(x_t, t_next, t):
            return net(
                x_t,
                t_next,
                t,
                cond=cond[mask_mf],
                # return_extra_output=False,
                **model_kwargs,
            )

        x_t_mf, t_mf, t_next_mf, velocity_cfg_mf = (
            x_t[mask_mf],
            t[mask_mf],
            t_next[mask_mf],
            velocity_cfg[mask_mf],
        )

        if batch_size_mf == 0:
            return mean_velocity

        _, dudt_mf = jvp(
            wrap_net,
            (x_t_mf, t_next_mf, t_mf),
            (velocity_cfg_mf, torch.zeros_like(t_next_mf), torch.ones_like(t_mf)),
        )
        mean_velocity_mf = (
            velocity_cfg_mf - expand_to((t_mf - t_next_mf), batch_size_mf, 4) * dudt_mf
        )
        mean_velocity[mask_mf] = mean_velocity_mf
        return mean_velocity

    @torch.no_grad()
    def _compute_mean_velocity_d(
        self,
        x_t,
        velocity_cfg,
        t_next,
        t,
        dt,
        cond,
        model_kwargs: dict,
        net: nn.Module | Callable,
    ):
        if x_t.shape[0] == 0:
            return torch.empty((0, *velocity_cfg.shape[1:]), device=velocity_cfg.device)

        x_t_minus_dt = x_t - dt * velocity_cfg

        if torch.isclose(1 - dt / (t - t_next), torch.zeros_like(t)).all():
            mean_velocity_next = torch.zeros_like(x_t_minus_dt)
        else:
            mean_velocity_next, _ = net(
                x_t_minus_dt,
                t_next,
                t - dt,
                cond=cond,
                **model_kwargs,
                # augment_labels=augment_labels,
                # return_extra_output=True,
            )

        mean_velocity = (dt * velocity_cfg + (t - dt - t_next) * mean_velocity_next) / (
            t - t_next
        )
        mean_velocity = torch.clip(
            mean_velocity, min=-self.clamp_utgt, max=self.clamp_utgt
        )
        return mean_velocity

    def training_losses(
        self,
        model,
        x1,
        cond,
        model_kwargs: dict = {},
        phase="gen",
        cur_step=None,
        force_t_val: float | None = None,
        force_t_r_dt_val: float | None = None,
        align_ctx=None,
        compute_decomposed_loss=False,
    ):
        assert x1.ndim == 4, f"Dim must be 4, but got {x1.ndim}"
        b = len(x1)

        # Sample time steps and noise x
        t, t_next, dt, alpha = self.sample(x1.shape[0], cur_step)
        noise_unscaled = torch.randn_like(x1)
        noise_scaled = noise_unscaled * t
        x_t = self._apply_noise(x1, noise_scaled, t, model)

        # Get the GT velocity
        velocity = noise_unscaled - x1

        # CFG training
        velocity_cfg = self._compute_velocity_cfg(
            velocity, x_t, t, cond, model, b, drop_label=False
        )

        # Split batch for continuous (alpha == 1 or r == t) and discrete training (0 < alpha <= 1)
        mask_c = (dt == 0).flatten()  # [b]  # Continuous
        mask_d = ~mask_c  # [b]  # Discrete
        bc, bd = mask_c.sum().item(), mask_d.sum().item()

        # Continuous variables
        v_cfg_c, xt_c, t_c, t_next_c, cond_c = (
            velocity_cfg[mask_c],
            x_t[mask_c],
            t[mask_c],
            t_next[mask_c],
            cond[mask_c],
        )

        # Discrete variables
        v_cfg_d, xt_d, t_d, t_next_d, dt_d, cond_d = (
            velocity_cfg[mask_d],
            x_t[mask_d],
            t[mask_d],
            t_next[mask_d],
            dt[mask_d],
            cond[mask_d],
        )

        # Calculate u_tgt when alpha == 1 or r == t
        mean_velocity_c = self._compute_mean_velocity_c(
            xt_c,
            t_next_c,
            t_c,
            velocity_cfg,
            cond,
            model_kwargs=model_kwargs,
            net=model,
        )

        # Calculate u_tgt when 0 < alpha <= 1
        mean_velocity_d = self._compute_mean_velocity_d(
            xt_d,
            v_cfg_d,
            t_next_d,
            t_d,
            dt_d,
            cond_d,
            model_kwargs=model_kwargs,
            net=model,
        )

        # Combine mean velocities
        mean_velocity = torch.cat([mean_velocity_c, mean_velocity_d], dim=0)

        ###### Model predictions
        pred_mean_velocity, _ = model(
            x_t,
            t_next,
            t,
            cond=cond,
            **model_kwargs,
            # augment_labels=augment_labels,
            # return_extra_output=True,
        )

        ###### Adaptive loss
        loss_unscaled = (
            ((pred_mean_velocity - mean_velocity) ** 2).flatten(1).mean(1)
        )  # [b]
        weight_c = torch.ones(bc, device=velocity.device)  # [b_c]
        weight_d = torch.ones(bd, device=velocity.device) * alpha  # [b_d]
        weight = torch.cat([weight_c, weight_d], dim=0) / (
            loss_unscaled.detach() + self.adaptive_loss_weight_eps
        )  # [b]
        loss = weight * loss_unscaled  # [b]

        ## Compute trajectory flow matching loss
        loss_tfm = ((pred_mean_velocity - velocity_cfg) ** 2).flatten(1).mean(1)  # [b]

        ## Compute consistency flow matching loss
        loss_tcc = (
            (2 * (velocity_cfg - mean_velocity) * pred_mean_velocity).flatten(1).mean(1)
        )  # [b]
        loss_tfm_plus_tcc = loss_tfm + loss_tcc  # [b]

        loss_dict = {
            "total_loss": loss,  # [b,]
            "loss_FM": loss_tfm,  # [b,]
            "loss_consistency": loss_tcc,
            "loss_sum": loss_tfm_plus_tcc,
        }

        return loss_dict
