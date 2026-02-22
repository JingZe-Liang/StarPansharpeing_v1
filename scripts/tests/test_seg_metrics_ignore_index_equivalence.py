from __future__ import annotations

import torch

from src.stage2.segmentation.metrics.basic import HyperSegmentationScore


def test_ignore_index_none_and_shifted_ignore_case_are_equivalent() -> None:
    # Build a 2-class prediction/label map where both classes are present.
    pred = torch.tensor(
        [
            [
                [0, 1, 1, 0],
                [0, 1, 0, 1],
                [1, 0, 1, 0],
                [1, 0, 0, 1],
            ]
        ],
        dtype=torch.long,
    )
    gt = torch.tensor(
        [
            [
                [0, 1, 0, 0],
                [0, 1, 1, 1],
                [1, 0, 1, 0],
                [1, 0, 0, 1],
            ]
        ],
        dtype=torch.long,
    )

    # Case A: direct 0..N-1 metrics
    metrics_direct = HyperSegmentationScore(
        n_classes=2,
        ignore_index=None,
        include_bg=True,
        reduction="macro",
        per_class=False,
    )
    out_direct = metrics_direct(pred, gt)

    # Case B: shift mapping path (ignore_index=N, include_bg=False)
    metrics_shifted = HyperSegmentationScore(
        n_classes=2,
        ignore_index=2,
        include_bg=False,
        reduction="macro",
        per_class=False,
    )
    out_shifted = metrics_shifted(pred, gt)

    for key in out_direct.keys():
        torch.testing.assert_close(out_direct[key], out_shifted[key], atol=1e-6, rtol=1e-6)
