from __future__ import annotations

import torch
import pytest

from src.stage2.change_detection.metrics.basic import ChangeDetectionScore


def test_change_detection_score_ignores_255_targets_for_jaccard() -> None:
    metric = ChangeDetectionScore(
        n_classes=2,
        ignore_index=255,
        include_bg=False,
        input_format="index",
    )

    pred = torch.tensor(
        [
            [
                [0, 1, 1, 0],
                [1, 0, 1, 0],
            ]
        ],
        dtype=torch.long,
    )
    gt = torch.tensor(
        [
            [
                [0, 1, 255, 0],
                [1, 255, 1, 0],
            ]
        ],
        dtype=torch.long,
    )

    metric.update(pred, gt)
    out = metric.compute()

    assert "jaccard" in out
    assert not torch.isnan(out["jaccard"]).any()


def test_change_detection_f1_cd_matches_binary_formula_ignoring_255() -> None:
    metric = ChangeDetectionScore(
        n_classes=2,
        ignore_index=255,
        include_bg=False,
        input_format="index",
        f1_cd=True,
    )

    pred = torch.tensor(
        [
            [
                [1, 0, 1, 1],
                [0, 1, 0, 1],
            ]
        ],
        dtype=torch.long,
    )
    gt = torch.tensor(
        [
            [
                [1, 0, 255, 1],
                [0, 1, 255, 0],
            ]
        ],
        dtype=torch.long,
    )

    # valid pixels exclude gt==255:
    # TP=3, FP=1, FN=0 -> F1=2*TP/(2*TP+FP+FN)=6/7
    expected_f1 = torch.tensor(6.0 / 7.0)
    metric.update(pred, gt)
    out = metric.compute()
    assert "f1_cd" in out
    torch.testing.assert_close(out["f1_cd"], expected_f1, atol=1e-6, rtol=1e-6)


def test_change_detection_f1_cd_requires_binary_classes() -> None:
    with pytest.raises(ValueError, match="f1_cd=True only supports binary change detection"):
        ChangeDetectionScore(
            n_classes=3,
            ignore_index=255,
            include_bg=False,
            input_format="index",
            f1_cd=True,
        )
