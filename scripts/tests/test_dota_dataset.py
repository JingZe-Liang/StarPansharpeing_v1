from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

from src.stage2.object_detection.data.DOTA import get_dataloader

Split = Literal["train", "val"]
SPLITS: tuple[Split, Split] = ("train", "val")


@pytest.mark.parametrize("split", SPLITS)
def test_dota_dataloader_basic(split: Split) -> None:
    root = Path("data/Downstreams/DOTA")
    # if not root.exists():
    #     pytest.skip("DOTA dataset not found in data/Downstreams/DOTA")
    dataset, dataloader = get_dataloader(
        batch_size=2,
        num_workers=0,
        split=split,
        root=str(root),
        download=True,
        checksum=True,
    )
    assert len(dataset) > 0
    batch = next(iter(dataloader))
    assert isinstance(batch, list)
    assert len(batch) > 0
    sample = batch[0]
    assert "image" in sample
    assert "labels" in sample
