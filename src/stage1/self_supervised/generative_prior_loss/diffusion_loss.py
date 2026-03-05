from __future__ import annotations

from typing import Any, Callable, Literal

import torch
import torch.nn as nn

from src.utilities.transport.flow_matching.transport import Transport


class DiffusionLoss(nn.Module):
    def __init__(
        self,
        model: nn.Module | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        *,
        model_type: str = "velocity",
        path_type: str = "linear",
        loss_type: str = "none",
        train_eps: float | None = None,
        sample_eps: float | None = None,
        time_sample_type: str = "uniform",
        cfm_factor: float = 0.0,
        get_pred_x_clean: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.get_pred_x_clean = get_pred_x_clean
        self.transport = Transport(
            model_type=model_type,
            path_type=path_type,
            loss_type=loss_type,
            train_eps=train_eps,
            sample_eps=sample_eps,
            time_sample_type=time_sample_type,
            cfm_factor=cfm_factor,
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

        terms = self.transport.training_losses(
            model=run_model,
            x1=latent,
            model_kwargs=model_kwargs,
            get_pred_x_clean=self.get_pred_x_clean,
            t_forced=t_forced,
            x0_forced=x0_forced,
        )
        loss = self._reduce_loss(terms["loss"], reduction=reduction)
        if return_terms:
            terms["loss"] = loss
            return terms
        return loss
