from __future__ import annotations

import torch
import torch.nn as nn

from src.stage2.change_detection.models.adapter import UNetDecoder


class _FakeEncoder:
    def __init__(self) -> None:
        self.output_channels = [16, 32, 64, 128]
        self.strides = [[2, 2], [2, 2], [2, 2], [2, 2]]
        self.conv_op = nn.Conv2d
        self.conv_bias = True
        self.norm_op = "layernorm2d"
        self.norm_op_kwargs = {}
        self.dropout_op = None
        self.dropout_op_kwargs = None
        self.nonlin = "gelu"


def _make_skips(batch_size: int = 2) -> list[torch.Tensor]:
    return [
        torch.randn(batch_size, 16, 64, 64),
        torch.randn(batch_size, 32, 32, 32),
        torch.randn(batch_size, 64, 16, 16),
        torch.randn(batch_size, 128, 8, 8),
    ]


def test_unet_decoder_forward_with_raw_pixel_cat() -> None:
    decoder = UNetDecoder(
        encoder=_FakeEncoder(),  # type: ignore[arg-type]
        num_classes=2,
        latent_width=None,
        raw_px_width=9,
        n_conv_per_stage=1,
        depths_per_stage=1,
        block_types=["ghost", "ghost", "ghost"],
        deep_supervision=False,
    )

    skips = _make_skips()
    raw_x = torch.randn(2, 9, 64, 64)
    out = decoder(skips=skips, raw_x=raw_x)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 2, 64, 64)


def test_unet_decoder_forward_backward_compatible_call() -> None:
    decoder = UNetDecoder(
        encoder=_FakeEncoder(),  # type: ignore[arg-type]
        num_classes=2,
        latent_width=None,
        raw_px_width=9,
        n_conv_per_stage=1,
        depths_per_stage=1,
        block_types=["ghost", "ghost", "ghost"],
        deep_supervision=False,
    )

    skips = _make_skips()
    dummy_cond = torch.randn(2, 4, 1, 1)
    raw_x = torch.randn(2, 9, 64, 64)

    out = decoder(skips, dummy_cond, raw_x=raw_x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 2, 64, 64)


def test_unet_decoder_forward_without_raw_injection() -> None:
    decoder = UNetDecoder(
        encoder=_FakeEncoder(),  # type: ignore[arg-type]
        num_classes=2,
        latent_width=None,
        raw_px_width=None,
        n_conv_per_stage=1,
        depths_per_stage=1,
        block_types=["ghost", "ghost", "ghost"],
        deep_supervision=False,
    )

    skips = _make_skips()
    out = decoder(skips=skips, raw_x=torch.randn(2, 9, 64, 64))
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 2, 64, 64)
