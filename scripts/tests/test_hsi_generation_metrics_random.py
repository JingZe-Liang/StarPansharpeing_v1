from __future__ import annotations

import torch

from src.stage2.generative.tools.metrics import HSIGenerationMetrics


def test_hsi_generation_metrics_spr_srec_on_200_random_samples() -> None:
    sample_count = 200
    channel_num = 48
    height = 8
    width = 8

    generator = torch.Generator().manual_seed(42)
    metric = HSIGenerationMetrics(
        cal_metrics=["spr", "srec"],
        spectral_sample_size=20_000,
        spectral_k=10,
        spectral_groups=10,
        spectral_chunk_size=256,
    )

    for _ in range(sample_count):
        fake = torch.rand(1, channel_num, height, width, generator=generator)
        real = torch.rand(1, channel_num, height, width, generator=generator)
        metric.update(fake, real)  # type: ignore[invalid-argument-type]

    result = metric.compute()  # type: ignore[missing-argument]
    spr = float(result["sPr"].item())
    srec = float(result["sRec"].item())
    print(f"[200 random samples] sPr={spr:.6f}, sRec={srec:.6f}")

    assert 0.0 <= spr <= 1.0
    assert 0.0 <= srec <= 1.0
