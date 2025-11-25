from typing import Callable

import torch
import torch.nn as nn
from loguru import logger
from torch import Tensor

from src.utilities.config_utils import function_config_to_basic_types

from .basic_loss import get_loss


class AmotizedPixelLoss(nn.Module):
    @function_config_to_basic_types
    def __init__(
        self,
        pixel_loss_type: str,
        amotized_loss: Callable[[Tensor, Tensor], Tensor] | None = None,
        pixel_loss_kwargs: dict = {},
        factors: tuple = (0.1, 1.0),
        is_neg_1_1: bool = True,
    ):
        super().__init__()
        self.amotized_loss = amotized_loss
        self.pixel_loss = get_loss(pixel_loss_type, **pixel_loss_kwargs)
        # tuple of latent loss factor and pixel loss factor
        self.factors = factors
        # ensure the pixel loss in computed at range of (0, 1)
        self.is_neg_1_1 = is_neg_1_1

    def _map_to_0_1(self, x: Tensor | None):
        """Map tensor from [-1, 1] to [0, 1]"""
        if not self.is_neg_1_1 or x is None:
            return x
        return (x + 1.0) / 2.0

    def forward(
        self,
        pred_latent: Tensor,
        sr_latent: Tensor,
        pred_sr: Tensor | None = None,
        sr: Tensor | None = None,
        pred_sr_from_latent: Tensor | None = None,
        sr2: Tensor | None = None,
    ):
        pred_sr, sr, pred_sr_from_latent, sr2 = map(self._map_to_0_1, (pred_sr, sr, pred_sr_from_latent, sr2))

        # 1. loss on latent of sr and gt
        latent_loss = 0.0
        if self.factors[0] > 0 and pred_latent is not None and sr_latent is not None:
            assert self.amotized_loss is not None, "amotized_loss function must be provided."
            latent_loss = self.amotized_loss(pred_latent, sr_latent) * self.factors[0]

        # 2. pixel loss on sr (e.g., dircectly predicted sr pixels) and gt
        sr_pixel_loss = 0.0
        if pred_sr is not None and sr is not None and self.factors[1] > 0:
            sr_pixel_loss = self.pixel_loss(pred_sr, sr)
            if isinstance(sr_pixel_loss, tuple):
                sr_pixel_loss, _ = sr_pixel_loss
            sr_pixel_loss = sr_pixel_loss * self.factors[1]

        # 3. pixel loss on tokenizer decoded sr and gt, may backward from the de-tokenizer
        sr_pixel_loss2 = 0.0
        if pred_sr_from_latent is not None and sr2 is not None:
            sr_pixel_loss2 = self.pixel_loss(pred_sr_from_latent, sr2)
            if isinstance(sr_pixel_loss2, tuple):
                sr_pixel_loss2, _ = sr_pixel_loss2
            sr_pixel_loss2 = sr_pixel_loss2 * self.factors[1]

        # 4. Sum all losses
        loss = latent_loss + sr_pixel_loss + sr_pixel_loss2

        # Detach the loss for logs
        _to_out_tensor_detached = lambda x: x.detach() if torch.is_tensor(x) else torch.tensor(0.0).to(pred_latent)
        latent_loss, pixel_loss, pixel_from_latent = map(
            _to_out_tensor_detached, [latent_loss, sr_pixel_loss, sr_pixel_loss2]
        )

        loss_dict = {
            "latent_loss": latent_loss,
            "pixel_loss": pixel_loss,
            "pixel_from_latent_loss": pixel_from_latent,
            "total_loss": loss,
        }

        return loss, loss_dict
