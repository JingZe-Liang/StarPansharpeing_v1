from __future__ import annotations

import numpy as np
import pytest
import torch

from src.stage2.generative.tools.metrics import HSIGenerationMetrics, interp_to_nbands_if_needed, interp_wavelength


def test_interp_to_nbands_matches_numpy_interp() -> None:
    wavelength = torch.linspace(380, 2500, 224)
    wavelength_n = torch.linspace(400, 1000, 48)
    interp_bands = 48
    x = (wavelength - wavelength.min()) / (wavelength.max() - wavelength.min())
    spectral_curve = torch.cos(x)
    img = spectral_curve.reshape(1, 224, 1, 1).repeat(1, 1, 8, 8)

    img_interp = interp_to_nbands_if_needed(
        img=img,
        wavelength=wavelength,
        wavelength_n=wavelength_n,
        interp_bands=interp_bands,
    )
    expected = np.interp(
        wavelength_n.numpy(),
        wavelength.numpy(),
        spectral_curve.numpy(),
    )

    assert img_interp.shape == (1, interp_bands, 8, 8)
    assert torch.allclose(img_interp[0, :, 0, 0], torch.from_numpy(expected).float(), atol=1e-5, rtol=1e-5)


def test_prepare_rgb_with_wavelength_interpolation() -> None:
    wavelength = torch.linspace(380, 2500, 224)
    rgb_wavelength = torch.tensor([650.0, 550.0, 450.0])
    x = (wavelength - wavelength.min()) / (wavelength.max() - wavelength.min())
    spectral_curve = torch.cos(x)
    img = spectral_curve.reshape(1, 224, 1, 1).repeat(1, 1, 4, 4)

    metric = HSIGenerationMetrics(
        cal_metrics=["spr"],
        wavelength=wavelength,
        rgb_wavelength=rgb_wavelength,
    )
    rgb = metric._prepare_rgb(img)
    expected = np.interp(
        rgb_wavelength.numpy(),
        wavelength.numpy(),
        spectral_curve.numpy(),
    )
    expected_rgb = torch.from_numpy(expected).float().reshape(1, 3, 1, 1).repeat(1, 1, 4, 4).clamp(0.0, 1.0)

    assert rgb.shape == (1, 3, 4, 4)
    assert torch.allclose(rgb, expected_rgb, atol=1e-5, rtol=1e-5)


def test_update_optional_interp_to_nbands_in_update() -> None:
    wavelength = torch.linspace(380, 2500, 224)
    wavelength_n = torch.linspace(400, 1000, 48)
    interp_bands = 48
    metric = HSIGenerationMetrics(
        cal_metrics=["spr", "srec"],
        wavelength=wavelength,
        wavelength_n=wavelength_n,
        interp_bands=interp_bands,
        spectral_sample_size=4096,
        spectral_k=5,
        spectral_groups=4,
        spectral_chunk_size=256,
    )

    fake = torch.rand(1, 224, 8, 8)
    real = torch.rand(1, 224, 8, 8)
    metric.update(fake, real)  # type: ignore[invalid-argument-type]
    assert metric._fake_profiles is not None
    assert metric._real_profiles is not None
    assert metric._fake_profiles.shape[1] == interp_bands
    assert metric._real_profiles.shape[1] == interp_bands

    out = metric.compute()  # type: ignore[missing-argument]
    assert "sPr" in out
    assert "sRec" in out
    assert 0.0 <= float(out["sPr"].item()) <= 1.0
    assert 0.0 <= float(out["sRec"].item()) <= 1.0


def test_interp_wavelength_raises_when_out_of_range() -> None:
    img = torch.rand(1, 10, 4, 4)
    wavelength = torch.linspace(400, 700, 10)
    with pytest.raises(ValueError):
        interp_wavelength(img, wavelength, torch.tensor([350.0, 500.0, 650.0]))
