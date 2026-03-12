from __future__ import annotations

import ast
import warnings
from typing import Any, Callable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionLoss(nn.Module):
    def __init__(
        self,
        model: nn.Module | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        *,
        lambda0: float | None = None,
        min_logsnr: float | None = None,
        end_logsnr: float = -15.0,
        schedule: str = "linear_logsnr_vp",
        time_sample_type: str = "uniform",
        include_terminal_kl: bool = False,
        model_type: str = "velocity",
        path_type: str = "linear",
        loss_type: str = "none",
        train_eps: float | None = None,
        sample_eps: float | None = None,
        cfm_factor: float = 0.0,
        get_pred_x_clean: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.max_logsnr = self._resolve_max_logsnr(lambda0=lambda0, min_logsnr=min_logsnr)
        self.end_logsnr = float(end_logsnr)
        self.schedule = schedule
        self.time_sample_type = time_sample_type
        self.include_terminal_kl = include_terminal_kl

        self._warn_if_legacy_args_are_used(
            model_type=model_type,
            path_type=path_type,
            loss_type=loss_type,
            train_eps=train_eps,
            sample_eps=sample_eps,
            cfm_factor=cfm_factor,
            get_pred_x_clean=get_pred_x_clean,
        )

    @staticmethod
    def _resolve_max_logsnr(*, lambda0: float | None, min_logsnr: float | None) -> float:
        if lambda0 is not None and min_logsnr is not None and float(lambda0) != float(min_logsnr):
            raise ValueError(
                f"`lambda0` ({lambda0}) and `min_logsnr` ({min_logsnr}) must match when both are provided."
            )
        if lambda0 is not None:
            return float(lambda0)
        if min_logsnr is not None:
            return float(min_logsnr)
        return 5.0

    @staticmethod
    def _warn_if_legacy_args_are_used(**legacy_kwargs: Any) -> None:
        used_legacy_keys: list[str] = []
        defaults = {
            "model_type": "velocity",
            "path_type": "linear",
            "loss_type": "none",
            "train_eps": None,
            "sample_eps": None,
            "cfm_factor": 0.0,
            "get_pred_x_clean": False,
        }
        for key, value in legacy_kwargs.items():
            if value != defaults[key]:
                used_legacy_keys.append(key)
        if used_legacy_keys:
            warnings.warn(
                "DiffusionLoss now implements a UL-style prior and ignores legacy Transport arguments: "
                + ", ".join(used_legacy_keys),
                stacklevel=2,
            )

    @staticmethod
    def _reduce_loss(
        loss: torch.Tensor,
        reduction: Literal["mean", "sum", "none"],
    ) -> torch.Tensor:
        if reduction == "mean":
            return loss.mean()
        if reduction == "sum":
            return loss.sum()
        if reduction == "none":
            return loss
        raise ValueError(f"Unsupported reduction: {reduction}")

    @staticmethod
    def _mean_flat(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], -1).mean(dim=1)

    def _sample_timesteps(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        t_forced: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if t_forced is not None:
            if t_forced.ndim == 0:
                t_forced = t_forced.expand(batch_size)
            if t_forced.shape != (batch_size,):
                raise ValueError(f"`t_forced` must have shape {(batch_size,)}, got {tuple(t_forced.shape)}.")
            return t_forced.to(device=device, dtype=dtype)

        if self.time_sample_type == "uniform":
            return torch.rand(batch_size, device=device, dtype=dtype)
        if self.time_sample_type == "sigmoid":
            t = torch.randn(batch_size, device=device, dtype=dtype)
            return torch.sigmoid(t)
        if isinstance(self.time_sample_type, str):
            try:
                time_list = ast.literal_eval(self.time_sample_type)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(f"Unsupported time_sample_type: {self.time_sample_type}") from exc
            time_choices = torch.tensor(time_list, device=device, dtype=dtype)
            if time_choices.ndim != 1 or torch.any(time_choices < 0) or torch.any(time_choices > 1):
                raise ValueError("Discrete `time_sample_type` values must be a 1D list in [0, 1].")
            indices = torch.randint(0, time_choices.shape[0], (batch_size,), device=device)
            return time_choices[indices]
        raise ValueError(f"Unsupported time_sample_type: {self.time_sample_type}")

    def _logsnr(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.schedule != "linear_logsnr_vp":
            raise ValueError(f"Unsupported UL schedule: {self.schedule}")
        lambda_t = self.max_logsnr + (self.end_logsnr - self.max_logsnr) * t
        d_lambda_dt = torch.full_like(lambda_t, self.end_logsnr - self.max_logsnr)
        return lambda_t, d_lambda_dt

    @staticmethod
    def _logsnr_to_alpha_sigma(lambda_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        alpha_t = torch.sqrt(torch.sigmoid(lambda_t))
        sigma_t = torch.sqrt(torch.sigmoid(-lambda_t))
        return alpha_t, sigma_t

    @staticmethod
    def _expand_to_latent(t: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        return t.view(t.shape[0], *([1] * (latent.ndim - 1)))

    def _terminal_kl(self, latent_clean: torch.Tensor) -> torch.Tensor:
        lambda_1 = torch.full(
            (latent_clean.shape[0],),
            self.end_logsnr,
            device=latent_clean.device,
            dtype=latent_clean.dtype,
        )
        alpha_1, sigma_1 = self._logsnr_to_alpha_sigma(lambda_1)
        alpha_1 = self._expand_to_latent(alpha_1, latent_clean)
        sigma_1 = self._expand_to_latent(sigma_1, latent_clean)
        variance = sigma_1.square()
        mu_sq = (alpha_1 * latent_clean).square()
        kl = 0.5 * (mu_sq + variance - 1.0 - variance.log())
        return self._mean_flat(kl)

    def forward(
        self,
        latent: torch.Tensor,
        model_kwargs: dict[str, Any] | None = None,
        *,
        model: nn.Module | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        reduction: Literal["mean", "sum", "none"] = "mean",
        t_forced: torch.Tensor | None = None,
        x0_forced: torch.Tensor | None = None,
        return_terms: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        run_model = model if model is not None else self.model
        if run_model is None:
            raise ValueError("DiffusionLoss requires a model. Pass `model=` in forward or set it in __init__.")
        if x0_forced is not None:
            warnings.warn("`x0_forced` is ignored by the UL prior implementation.", stacklevel=2)
        if model_kwargs is None:
            model_kwargs = {}

        latent_clean = latent
        t = self._sample_timesteps(
            batch_size=latent_clean.shape[0],
            device=latent_clean.device,
            dtype=latent_clean.dtype,
            t_forced=t_forced,
        )
        lambda_t, d_lambda_dt = self._logsnr(t)
        alpha_t, sigma_t = self._logsnr_to_alpha_sigma(lambda_t)
        alpha_t_expanded = self._expand_to_latent(alpha_t, latent_clean)
        sigma_t_expanded = self._expand_to_latent(sigma_t, latent_clean)
        eps = torch.randn_like(latent_clean)
        z_t = alpha_t_expanded * latent_clean + sigma_t_expanded * eps

        pred = run_model(z_t, t, **model_kwargs)
        if pred.shape != latent_clean.shape:
            raise ValueError(
                f"UL prior model output shape must match latent shape, got {tuple(pred.shape)} vs {tuple(latent_clean.shape)}."
            )

        mse = self._mean_flat(F.mse_loss(pred, latent_clean, reduction="none"))
        elbo_coef = -0.5 * torch.exp(lambda_t) * d_lambda_dt
        per_sample_loss = elbo_coef * mse

        terminal_kl = torch.zeros_like(per_sample_loss)
        if self.include_terminal_kl:
            terminal_kl = self._terminal_kl(latent_clean)
            per_sample_loss = per_sample_loss + terminal_kl

        loss = self._reduce_loss(per_sample_loss, reduction=reduction)
        if return_terms:
            return {
                "loss": loss,
                "pred": pred,
                "target": latent_clean,
                "t": t,
                "elbo_coef": elbo_coef,
                "z_t": z_t,
                "terminal_kl": terminal_kl,
            }
        return loss
