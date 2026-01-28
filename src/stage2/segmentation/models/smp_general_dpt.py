"""
SMP DPT wrapper for configurable backbones.
"""

from typing import Any, Literal

import segmentation_models_pytorch as smp
from torch import Tensor, nn


class SMPDPT(nn.Module):
    """
    Segmentation model with DPT head from segmentation_models.pytorch.

    Common backbone examples (encoder_name):
        - tu-vit_base_patch16_224.augreg_in21k_ft_in1k
        - tu-vit_base_patch16_224.augreg_in1k
        - tu-vit_small_patch16_224.augreg_in1k
        - tu-swin_base_patch4_window7_224.ms_in1k
        - tu-swin_tiny_patch4_window7_224.ms_in1k
        - tu-deit_base_patch16_224.fb_in1k
        - tu-mvitv2_base.fb_in1k
    """

    def __init__(
        self,
        encoder_name: str = "tu-vit_base_patch16_224.augreg_in21k",
        encoder_depth: int = 4,
        encoder_weights: str | None = "imagenet",
        encoder_output_indices: tuple[int, ...] | None = None,
        decoder_readout: Literal["ignore", "add", "cat"] = "cat",
        decoder_intermediate_channels: tuple[int, int, int, int] = (256, 512, 1024, 1024),
        decoder_fusion_channels: int = 256,
        in_channels: int = 3,
        classes: int = 1,
        activation: str | None = None,
        aux_params: dict[str, Any] | None = None,
        dynamic_img_size: bool | None = None,
        **encoder_kwargs: Any,
    ) -> None:
        """
        Args:
            encoder_name: Timm encoder name with "tu-" prefix (required by SMP DPT).
            encoder_depth: Number of encoder stages, range [1, 4].
            encoder_weights: Pretrained weights name, or None for random init.
            encoder_output_indices: Indices of encoder blocks to extract features from.
            decoder_readout: Strategy to use prefix tokens: "ignore", "add", or "cat".
            decoder_intermediate_channels: Intermediate decoder channels.
            decoder_fusion_channels: Fusion channels for decoder output.
            in_channels: Input image channels.
            classes: Number of output classes.
            activation: Optional activation after logits.
            aux_params: Optional classification head params.
            dynamic_img_size: Enable dynamic input size if encoder supports it.
            **encoder_kwargs: Extra timm encoder kwargs.
        """
        super().__init__()

        if dynamic_img_size is not None:
            encoder_kwargs["dynamic_img_size"] = dynamic_img_size

        self.model = smp.DPT(
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights=encoder_weights,
            encoder_output_indices=list(encoder_output_indices) if encoder_output_indices is not None else None,
            decoder_readout=decoder_readout,
            decoder_intermediate_channels=decoder_intermediate_channels,
            decoder_fusion_channels=decoder_fusion_channels,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
            aux_params=aux_params,
            **encoder_kwargs,
        )

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        return self.model(x)
