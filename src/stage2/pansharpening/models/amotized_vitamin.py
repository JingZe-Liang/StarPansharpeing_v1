from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor

from src.stage2.pansharpening.models.transformer import Transformer, TransformerConfig
from src.stage2.pansharpening.models.vitamin_conv import (
    ConvCfg,
    VitaminCfg,
    VitaminModel,
)
from src.stage2.utilities.amotized.amotized_model_wrapper import AmotizedModelMixin


class AmotizedPansharpeningVitamin(AmotizedModelMixin):
    @classmethod
    def create_specific_model(
        cls,
        ms_chan: int,
        pan_chan: int,
        latent_chan: int,
        decoder_fn: Callable[[Tensor], Tensor],
        version: str = "small",
    ):
        model_version_getter = getattr(cls, f"_create_{version}_model")
        return model_version_getter(ms_chan, pan_chan, latent_chan, decoder_fn)

    # specifical for pansharpening task
    @classmethod
    def _create_small_model(
        cls,
        ms_chan: int,
        pan_chan: int,
        latent_chan: int,
        decoder_fn: Callable | nn.Module,
    ):
        latent_cfg = TransformerConfig(
            in_dim=latent_chan,
            dim=256,
            depth=8,
            num_heads=8,
            mlp_ratio=4,
            drop=0.0,
            patch_size=2,
            out_channels=256,
            norm_layer="layernorm",
            mlp_norm_layer="layernorm",
        )
        conv_cfg = ConvCfg(
            expand_ratio=2.0, kernel_size=3, act_layer="gelu", norm_layer="layernorm2d"
        )
        vitamin_cfg = VitaminCfg(
            stem_width=32,
            embed_dim=[64, 192, 192],
            depths=[2, 2, 2],
            ms_channel=ms_chan,
            pan_channel=pan_chan,
            condition_channel=256,
            use_residual=True,
            conv_cfg=conv_cfg,
        )

        latent_model = Transformer(latent_cfg)
        pixel_model = VitaminModel(vitamin_cfg)

        return cls(
            pixel_model=pixel_model,
            amotized_model=latent_model,
            decoder_fn=decoder_fn,
            amotize_type="latent_to_pixel_fusion",
            backward_decoder=False,
            learn_decoder=False,
        )


# * --- Test --- #


def test_amotized_pansharpening_model():
    """Test the AmotizedPansharpeningVitamin model with synthetic data."""

    # Model parameters
    ms_chan = 8  # Multi-spectral channels
    pan_chan = 1  # Panchromatic channel
    latent_chan = 16  # Latent channel dimension
    batch_size = 2
    height = 64
    width = 64

    # Create a simple decoder function
    def simple_decoder(latent: torch.Tensor) -> torch.Tensor:
        """Simple decoder that converts latent back to image space."""
        return nn.Conv2d(latent_chan, ms_chan, kernel_size=1)(latent)

    # Create the model
    model = AmotizedPansharpeningVitamin._create_small_model(
        ms_chan=ms_chan,
        pan_chan=pan_chan,
        latent_chan=latent_chan,
        decoder_fn=simple_decoder,
    )

    # Set model to eval mode
    model.eval()

    # Create synthetic input data
    lrms_latent = torch.randn(batch_size, latent_chan, height, width)
    pan_latent = torch.randn(batch_size, latent_chan, height, width)
    lrms = torch.randn(batch_size, ms_chan, height, width)
    pan = torch.randn(batch_size, pan_chan, height, width)

    # Test forward pass
    with torch.no_grad():
        try:
            # AmotizedModelMixin expects (pixel_in, latent_in) where each can be a tuple
            pixel_in = (lrms, pan)  # Multi-spectral and panchromatic images
            latent_in = (lrms_latent, pan_latent)  # Latent representations
            output = model(pixel_in, latent_in)

            if isinstance(output, dict):
                print("Model output keys:", output.keys())
                print("latent_out shape:", output["latent_out"].shape)
                if output["pixel_from_latent"] is not None:
                    print("pixel_from_latent shape:", output["pixel_from_latent"].shape)
                print("pixel_out shape:", output["pixel_out"].shape)
            else:
                print("Output type:", type(output))
                print(
                    "Output shape(s):",
                    [out.shape for out in output]
                    if isinstance(output, (tuple, list))
                    else output.shape,
                )

            print("✓ Model forward pass successful!")

        except Exception as e:
            print(f"✗ Model forward pass failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    # Test model parameters count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return True


if __name__ == "__main__":
    print("Testing AmotizedPansharpeningVitamin model...")
    test_amotized_pansharpening_model()
