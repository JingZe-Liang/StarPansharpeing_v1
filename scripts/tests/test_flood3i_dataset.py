from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from PIL import Image

from src.stage2.segmentation.data.flood3i import Flood3IDataset, get_flood3i_dataloader

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


def test_flood3i_binary_target_class_mapping(tmp_path: Path) -> None:
    root = tmp_path / "Flood-3i"
    img_dir = root / "Images"
    mask_dir = root / "Semantic_mask"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    image = np.zeros((2, 2, 3), dtype=np.uint8)
    mask = np.array([[0, 2], [255, 3]], dtype=np.uint8)
    Image.fromarray(image).save(img_dir / "10000_0_0.jpg")
    Image.fromarray(mask).save(mask_dir / "10000_lab_0_0.png")

    list_file = root / "train.txt"
    list_file.write_text("Images/10000_0_0.jpg Semantic_mask/10000_lab_0_0.png\n", encoding="utf-8")

    dataset = Flood3IDataset(
        root=root,
        split="train",
        list_file=list_file,
        normalize=False,
        binary_target_class=2,
        ignore_index=255,
    )
    sample = dataset[0]
    got_mask = sample["mask"].squeeze(0).numpy()

    expected = np.array([[0, 1], [255, 0]], dtype=np.int64)
    assert np.array_equal(got_mask, expected)
