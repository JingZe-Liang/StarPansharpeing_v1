from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch
from torch import Tensor


def mean_flat(x: Tensor) -> Tensor:
    """Take the mean over all non-batch dimensions."""

    return torch.mean(x, dim=tuple(range(1, x.ndim)))


def sum_flat(x: Tensor) -> Tensor:
    """Take the sum over all non-batch dimensions."""

    return torch.sum(x, dim=tuple(range(1, x.ndim)))


def interpolant(
    t: Tensor,
    *,
    path_type: Literal["linear", "cosine"],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if path_type == "linear":
        alpha_t = 1.0 - t
        sigma_t = t
        d_alpha_t = torch.full_like(t, -1.0)
        d_sigma_t = torch.ones_like(t)
        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    if path_type == "cosine":
        alpha_t = torch.cos(t * np.pi / 2.0)
        sigma_t = torch.sin(t * np.pi / 2.0)
        d_alpha_t = -np.pi / 2.0 * torch.sin(t * np.pi / 2.0)
        d_sigma_t = np.pi / 2.0 * torch.cos(t * np.pi / 2.0)
        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    raise ValueError(f"Unknown {path_type=}")


def apply_time_shift(t: Tensor, *, shift_dim: int, shift_base: int = 4096) -> Tensor:
    shift = math.sqrt(shift_dim / shift_base)
    t_shifted = (shift * t) / (1.0 + (shift - 1.0) * t)
    return torch.clamp(t_shifted, 0.0, 1.0)


def estimate_x0_from_v(
    x_t: Tensor,
    v_t: Tensor,
    t: Tensor,
    *,
    path_type: Literal["linear", "cosine"],
) -> Tensor:
    if t.ndim == 1:
        t = t.view(-1, 1, 1, 1)
    alpha_t, sigma_t, d_alpha_t, d_sigma_t = interpolant(t, path_type=path_type)
    det = alpha_t * d_sigma_t - sigma_t * d_alpha_t
    return (d_sigma_t * x_t - sigma_t * v_t) / det


def pack_conditions(conditions: Tensor | list[Tensor] | tuple[Tensor, ...] | None) -> Tensor | None:
    if conditions is None:
        return None
    if isinstance(conditions, Tensor):
        return conditions
    if isinstance(conditions, (list, tuple)):
        if not conditions:
            return None
        if not all(isinstance(item, Tensor) for item in conditions):
            raise TypeError(f"conditions list/tuple must be all tensors, got {[type(item) for item in conditions]}")
        return torch.cat(list(conditions), dim=1)
    raise TypeError(f"Unsupported conditions type: {type(conditions)}")


class SpeedrunSILoss:
    """SpeedrunDiT-style SILoss (requires cls_token + zs), kept for backward compatibility."""

    def __init__(
        self,
        prediction: str = "v",
        path_type: str = "linear",
        weighting: str = "uniform",
        cfm_weighting: str = "uniform",
        encoders: list | None = None,
        accelerator=None,
        apply_time_shift: bool = False,
        shift_base: int = 4096,
    ) -> None:
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.encoders = encoders or []
        self.accelerator = accelerator
        self.cfm_weighting = cfm_weighting
        self.apply_time_shift = apply_time_shift
        self.shift_base = shift_base

    def __call__(
        self,
        model,
        images: Tensor,
        model_kwargs: dict | None = None,
        zs=None,
        cls_token: Tensor | None = None,
        time_input: Tensor | None = None,
        noises: Tensor | None = None,
    ):
        if model_kwargs is None:
            model_kwargs = {}

        if time_input is None:
            if self.weighting == "uniform":
                time_input = torch.rand((images.shape[0], 1, 1, 1), device=images.device, dtype=images.dtype)
            elif self.weighting == "lognormal":
                rnd_normal = torch.randn((images.shape[0], 1, 1, 1), device=images.device, dtype=images.dtype)
                sigma = rnd_normal.exp()
                if self.path_type == "linear":
                    time_input = sigma / (1 + sigma)
                elif self.path_type == "cosine":
                    time_input = 2 / np.pi * torch.atan(sigma)

        if time_input is None:
            raise ValueError("SpeedrunSILoss: time_input is None and sampling did not set it.")

        if self.apply_time_shift:
            shift_dim = images.shape[1] * images.shape[2] * images.shape[3]
            shift = math.sqrt(shift_dim / self.shift_base)
            time_input = (shift * time_input) / (1 + (shift - 1) * time_input)
            time_input = torch.clamp(time_input, 0.0, 1.0)

        time_input = time_input.to(device=images.device, dtype=images.dtype)

        if cls_token is None:
            raise ValueError("SpeedrunSILoss requires cls_token.")
        cls_token = cls_token

        if noises is None:
            noises = torch.randn_like(images)
        noises_cls = torch.randn_like(cls_token)

        alpha_t, sigma_t, d_alpha_t, d_sigma_t = interpolant(time_input, path_type=self.path_type)  # type: ignore[arg-type]

        model_input = alpha_t * images + sigma_t * noises
        cls_input = alpha_t.squeeze(-1).squeeze(-1) * cls_token + sigma_t.squeeze(-1).squeeze(-1) * noises_cls
        if self.prediction == "v":
            model_target = d_alpha_t * images + d_sigma_t * noises
            cls_target = d_alpha_t * cls_token + d_sigma_t * noises_cls
        else:
            raise NotImplementedError()

        model_output, zs_tilde, cls_output = model(
            model_input, time_input.flatten(), **model_kwargs, cls_token=cls_input
        )

        denoising_loss = mean_flat((model_output - model_target) ** 2)
        denoising_loss_cls = mean_flat((cls_output - cls_target) ** 2)

        if zs is None:
            raise ValueError("SpeedrunSILoss requires zs.")

        proj_loss = 0.0
        bsz = zs[0].shape[0]
        for z, z_tilde in zip(zs, zs_tilde):
            for z_j, z_tilde_j in zip(z, z_tilde):
                z_tilde_j = torch.nn.functional.normalize(z_tilde_j, dim=-1)
                z_j = torch.nn.functional.normalize(z_j, dim=-1)
                proj_loss += mean_flat(-(z_j * z_tilde_j).sum(dim=-1))
        proj_loss /= len(zs) * bsz

        cfm_target = torch.roll(model_target, shifts=1, dims=0)
        cfm_target_cls = torch.roll(cls_target, shifts=1, dims=0)
        if self.cfm_weighting == "uniform":
            cfm_loss = -((model_output - cfm_target) ** 2).mean()
            cfm_loss_cls = -((cls_output - cfm_target_cls) ** 2).mean()
        elif self.cfm_weighting == "linear":
            cfm_loss = -(((model_output - cfm_target) ** 2) * time_input).mean()
            cfm_loss_cls = -(((cls_output - cfm_target_cls) ** 2) * time_input).mean()
        else:
            raise ValueError(f"Unknown {self.cfm_weighting=}")

        return denoising_loss, proj_loss, time_input, noises, denoising_loss_cls, cfm_loss, cfm_loss_cls


# Backward compatible alias (old name was ambiguous).
SILoss = SpeedrunSILoss


class CloudRemovalSILoss:
    """Cloud-removal v-pred SILoss for SiT(x,t,conditions=...)."""

    def __init__(
        self,
        *,
        prediction: Literal["v"] = "v",
        path_type: Literal["linear", "cosine"] = "linear",
        weighting: Literal["uniform", "lognormal"] = "uniform",
        apply_time_shift: bool = False,
        shift_base: int = 4096,
    ) -> None:
        if prediction != "v":
            raise ValueError("CloudRemovalSILoss currently supports v-pred only.")
        self.prediction = prediction
        self.path_type = path_type
        self.weighting = weighting
        self.apply_time_shift = apply_time_shift
        self.shift_base = shift_base

    def sample_time(self, images: Tensor) -> Tensor:
        if self.weighting == "uniform":
            return torch.rand((images.shape[0], 1, 1, 1), device=images.device, dtype=images.dtype)
        if self.weighting == "lognormal":
            rnd_normal = torch.randn((images.shape[0], 1, 1, 1), device=images.device, dtype=images.dtype)
            sigma = rnd_normal.exp()
            if self.path_type == "linear":
                return sigma / (1.0 + sigma)
            if self.path_type == "cosine":
                return 2.0 / np.pi * torch.atan(sigma)
        raise ValueError(f"Unknown {self.weighting=}")

    def __call__(
        self,
        model,
        images: Tensor,
        *,
        conditions: Tensor | list[Tensor] | tuple[Tensor, ...] | None = None,
        time_input: Tensor | None = None,
        noises: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if time_input is None:
            time_input = self.sample_time(images)

        if self.apply_time_shift:
            shift_dim = images.shape[1] * images.shape[2] * images.shape[3]
            time_input = apply_time_shift(time_input, shift_dim=shift_dim, shift_base=self.shift_base)

        if noises is None:
            noises = torch.randn_like(images)

        alpha_t, sigma_t, d_alpha_t, d_sigma_t = interpolant(time_input, path_type=self.path_type)
        model_input = alpha_t * images + sigma_t * noises
        model_target = d_alpha_t * images + d_sigma_t * noises

        cond_tensor = pack_conditions(conditions)
        model_output = model(model_input, time_input.flatten(), conditions=cond_tensor)[0]
        denoising_loss = mean_flat((model_output - model_target) ** 2)
        return denoising_loss, time_input, model_input, model_output
