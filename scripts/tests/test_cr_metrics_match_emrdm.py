from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from src.stage2.cloud_removal.metrics.basic import CRMetrics

_EMRDM_ROOT = Path(__file__).resolve().parents[2] / "src/stage2/cloud_removal/third_party/EMRDM"
sys.path.insert(0, _EMRDM_ROOT.as_posix())

from sgm.modules.learning.evaluator import img_metrics


def _avg_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    keys = ["PSNR", "SSIM", "LPIPS", "RMSE"]
    return {k: sum(m[k] for m in metrics) / len(metrics) for k in keys}


def test_crmetrics_matches_emrdm_img_metrics() -> None:
    torch.manual_seed(0)
    preds = torch.rand(1, 4, 256, 256)
    targets = torch.rand(1, 4, 256, 256)

    ref_metrics = [img_metrics(target=targets[0].unsqueeze(0), pred=preds[0].unsqueeze(0))]
    ref_avg = _avg_metrics(ref_metrics)

    metric: CRMetrics = CRMetrics()
    metric.update(preds, targets)  # type: ignore[call-arg]
    out = metric.compute()  # type: ignore[call-arg]

    assert out["PSNR"].item() == pytest.approx(ref_avg["PSNR"], rel=1e-5, abs=1e-5)
    assert out["SSIM"].item() == pytest.approx(ref_avg["SSIM"], rel=1e-5, abs=1e-5)
    assert out["LPIPS"].item() == pytest.approx(ref_avg["LPIPS"], rel=1e-5, abs=1e-5)
    assert out["RMSE"].item() == pytest.approx(ref_avg["RMSE"], rel=1e-5, abs=1e-5)
