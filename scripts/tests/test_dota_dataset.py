from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

from src.stage2.object_detection.data.DOTA import get_dataloader
from src.stage2.object_detection.utils.vis import draw_detections

Split = Literal["train", "val"]
SPLITS: tuple[Split, Split] = ("train", "val")


@pytest.mark.parametrize("split", SPLITS)
def test_dota_dataloader_basic(split: Split) -> None:
    root = Path("data/Downstreams/DOTA")
    if not root.exists():
        raise ValueError("DOTA dataset not found in data/Downstreams/DOTA")
    dataset, dataloader = get_dataloader(
        batch_size=4,
        num_workers=0,
        split=split,
        root=str(root),
        target_size=512,
        download=False,
        checksum=False,
    )
    assert len(dataset) > 0
    batch = next(iter(dataloader))
    assert isinstance(batch, list)
    assert len(batch) > 0
    sample = batch[0]
    assert "image" in sample
    assert "labels" in sample

    labels = sample["labels"]
    if hasattr(labels, "tolist"):
        labels = labels.tolist()
    labels_list = [int(x) for x in labels]

    bbox_xyxy = sample.get("bbox_xyxy")
    bbox_obb = sample.get("bbox_obb_le90")
    print(f"Image shape: {sample['image'].shape}")
    print(bbox_obb)

    img = draw_detections(
        sample["image"] * 255.0,
        boxes_xyxy=bbox_xyxy,
        labels_xyxy=labels_list,
        boxes_obb=bbox_obb,
        labels_obb=labels_list,
    )

    assert sample["image"].shape[-2:] == (512, 512)

    out_path = Path("/tmp") / f"dota_vis_{split}.png"
    img.save(out_path)
    print(out_path)


if __name__ == "__main__":
    test_dota_dataloader_basic("train")
