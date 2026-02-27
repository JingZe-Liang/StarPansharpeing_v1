from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
from unittest import SkipTest

import pytest

from src.stage2.object_detection.data.dior_mmdet import DIOR_CLASSES, DiorMMDetDataset, build_dataloader
from src.stage2.object_detection.utils.vis import draw_detections

Split = Literal["train", "val"]
SPLITS: tuple[Split, Split] = ("train", "val")


def _to_box_list(boxes: Any) -> list[list[float]]:
    if hasattr(boxes, "tensor"):
        boxes = boxes.tensor
    if hasattr(boxes, "tolist"):
        boxes = boxes.tolist()
    out: list[list[float]] = []
    for box in boxes or []:
        out.append([float(v) for v in box])
    return out


@pytest.mark.parametrize("split", SPLITS)
def test_dior_horizontal_mmdet_dataset_and_visualization(split: Split) -> None:
    root = Path("data/Downstreams/ObjectDetection/DIOR")
    if not root.exists():
        raise SkipTest("DIOR dataset not found in data/Downstreams/ObjectDetection/DIOR")

    dataset = DiorMMDetDataset(
        root=str(root),
        split=split,
        target_size=512,
        bbox_orientation="horizontal",
        use_obb=False,
    )
    dataloader = build_dataloader(
        dataset=dataset,
        batch_size=2,
        num_workers=0,
        shuffle=(split == "train"),
    )

    assert len(dataset) > 0
    batch = next(iter(dataloader))
    assert isinstance(batch, dict)
    assert len(batch["inputs"]) > 0
    assert len(batch["data_samples"]) > 0

    sample_input = batch["inputs"][0]
    data_sample = batch["data_samples"][0]
    assert sample_input.shape[-2:] == (512, 512)

    gt_instances = data_sample.gt_instances
    boxes_xyxy = _to_box_list(gt_instances.bboxes)
    labels = gt_instances.labels.tolist()

    img = draw_detections(
        sample_input,
        boxes_xyxy=boxes_xyxy,
        labels_xyxy=[int(v) for v in labels],
        class_names=list(DIOR_CLASSES),
    )

    out_path = Path("/tmp") / f"dior_horizontal_vis_{split}.png"
    img.save(out_path)
    assert out_path.exists()
    print(out_path)


@pytest.mark.parametrize("split", SPLITS)
def test_dior_oriented_mmdet_dataset_and_visualization(split: Split) -> None:
    root = Path("data/Downstreams/ObjectDetection/DIOR")
    if not root.exists():
        raise SkipTest("DIOR dataset not found in data/Downstreams/ObjectDetection/DIOR")

    dataset = DiorMMDetDataset(
        root=str(root),
        split=split,
        target_size=512,
        bbox_orientation="oriented",
        use_obb=True,
    )
    dataloader = build_dataloader(
        dataset=dataset,
        batch_size=2,
        num_workers=0,
        shuffle=(split == "train"),
    )

    assert len(dataset) > 0
    batch = next(iter(dataloader))
    assert isinstance(batch, dict)
    assert len(batch["inputs"]) > 0
    assert len(batch["data_samples"]) > 0

    sample_input = batch["inputs"][0]
    data_sample = batch["data_samples"][0]
    assert sample_input.shape[-2:] == (512, 512)

    gt_instances = data_sample.gt_instances
    boxes_obb = _to_box_list(gt_instances.bboxes)
    labels = gt_instances.labels.tolist()

    img = draw_detections(
        sample_input,
        boxes_obb=boxes_obb,
        labels_obb=[int(v) for v in labels],
        class_names=list(DIOR_CLASSES),
    )

    out_path = Path("/tmp") / f"dior_oriented_vis_{split}.png"
    img.save(out_path)
    assert out_path.exists()
    print(out_path)
