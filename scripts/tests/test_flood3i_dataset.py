from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

from src.stage2.segmentation.data.flood3i import get_flood3i_dataloader

Split = Literal["train", "val"]


@pytest.mark.parametrize("split", ["val"])
def test_flood3i_dataloader_basic(split: Split) -> None:
    root = Path("data/Downstreams/Flood-3i")
    if not root.exists():
        pytest.skip("Flood-3i dataset not found in data/Downstreams/Flood-3i")
    dataset, dataloader = get_flood3i_dataloader(
        batch_size=2,
        num_workers=0,
        split=split,
        root=str(root),
        normalize=True,
    )
    assert len(dataset) > 0
    batch = next(iter(dataloader))
    assert "image" in batch
    assert "mask" in batch
