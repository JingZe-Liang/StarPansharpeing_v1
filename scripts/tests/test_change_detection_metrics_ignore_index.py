from __future__ import annotations

import torch

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
