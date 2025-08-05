from typing import Callable

import torch
import torch.nn as nn

from .basic_loss import get_loss


class AmotizedPixelLoss(nn.Module):
    def __init__(
        self,
        pixel_loss_type: str,
        amotized_loss: Callable,
        pixel_loss_kwargs: dict = {},
        factors: tuple = (1.0, 1.0),
    ):
        super().__init__()
        self.amotized_loss = amotized_loss
        self.pixel_loss = get_loss(pixel_loss_type, **pixel_loss_kwargs)
        self.factors = factors

    def forward(
        self,
        pred_latent,
        sr_latent,
        pred_sr=None,
        sr=None,
        pred_sr_from_latent=None,
        sr2=None,
    ):
        latent_loss = self.amotized_loss(pred_latent, sr_latent) * self.factors[0]

        sr_pixel_loss = 0.0
        if pred_sr is not None and sr is not None:
            sr_pixel_loss = self.pixel_loss(pred_sr, sr) * self.factors[1]

        sr_pixel_loss2 = 0.0
        if pred_sr_from_latent is not None and sr2 is not None:
            sr_pixel_loss2 += (
                self.pixel_loss(pred_sr_from_latent, sr2) * self.factors[-1]
            )

        loss = latent_loss + sr_pixel_loss + sr_pixel_loss2

        _to_out_tensor_detached = lambda x: x.detach() if torch.is_tensor(x) else 0.0

        loss_dict = {
            "latent_loss": _to_out_tensor_detached(latent_loss),
            "sr_pixel_loss": _to_out_tensor_detached(sr_pixel_loss),
            "sr_pixel_from_latent": _to_out_tensor_detached(sr_pixel_loss2),
            "total_loss": loss,
        }

        return loss, loss_dict
