from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .ijepa.src.models.vision_transformer import (
    VisionTransformerPredictor,
    vit_base,
    vit_giant,
    vit_large,
    vit_predictor,
    vit_small,
    vit_tiny,
)
from .jepa_blockutils import MaskCollator, apply_masks, repeat_interleave_batch


class IJEPALoss(nn.Module):
    def __init__(
        self,
        predictor: VisionTransformerPredictor,
        patch_size=16,
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.15, 0.2),
        aspect_ratio=(0.3, 3.0),
        nenc=1,
        npred=4,
        min_keep=10,
        allow_overlap=False,
    ):
        super().__init__()
        self.predictor = predictor

        # Mask options
        self.mask_options = dict(
            patch_size=patch_size,
            enc_mask_scale=enc_mask_scale,
            pred_mask_scale=pred_mask_scale,
            aspect_ratio=aspect_ratio,
            nenc=nenc,
            npred=npred,
            min_keep=min_keep,
            allow_overlap=allow_overlap,
        )

    def create_masks(self, x: Tensor):
        assert x.ndim == 4, f"x should be an batched image, {x.shape=}"
        h, w = x.shape[2:]
        mask_collator = MaskCollator(input_size=(h, w), **self.mask_options)
        x, masks_enc, masks_pred = mask_collator(x)
        return x, masks_enc, masks_pred

    def forward(
        self,
        x: Tensor,
        model: nn.Module,
        ema_model: nn.Module,
        model_kwargs={},
        masks_enc=None,
        masks_pred=None,
    ):
        bs = x.size(0)

        # Create masks
        if masks_enc is None or masks_pred is None:
            x, masks_enc, masks_pred = self.create_masks(x)

        # Model context target
        with torch.no_grad():
            h = ema_model(x, jepa_masks=masks_enc, **model_kwargs)
            h = torch.layernorm(h, (h.shape[-1],))
            h = apply_masks(h, masks_pred)
            h_tgt = repeat_interleave_batch(h, bs, repeat=len(masks_enc))

        # Context and predict
        h_ctx = model(x, jepa_masks=masks_enc, **model_kwargs)
        h_pred = self.predictor(h_ctx, masks_enc, masks_pred)

        # Loss
        loss = F.smooth_l1_loss(h_pred, h_tgt)

        return loss


def create_ijepa_loss(
    # Predictor options
    patched_size: tuple[int, int],  # will be interpolated
    embed_dim: int,
    predictor_embed_dim: int,
    depth: int,
    num_heads: int,
    predictor_kwargs: dict = {},
    # Mask options
    patch_size=16,
    enc_mask_scale=(0.85, 1.0),
    pred_mask_scale=(0.15, 0.2),
    aspect_ratio=(0.3, 3.0),
    nenc=1,
    npred=4,
    min_keep=10,
    allow_overlap=False,
):
    predictor = vit_predictor(
        patched_size=patched_size,
        embed_dim=embed_dim,
        predictor_embed_dim=predictor_embed_dim,
        depth=depth,
        num_heads=num_heads,
        **predictor_kwargs,
    )

    loss_fn = IJEPALoss(
        predictor=predictor,
        patch_size=patch_size,
        enc_mask_scale=enc_mask_scale,
        pred_mask_scale=pred_mask_scale,
        aspect_ratio=aspect_ratio,
        nenc=nenc,
        npred=npred,
        min_keep=min_keep,
        allow_overlap=allow_overlap,
    )

    return loss_fn
