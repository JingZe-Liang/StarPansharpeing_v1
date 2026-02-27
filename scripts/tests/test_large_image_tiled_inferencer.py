from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from mmengine.structures import InstanceData

from src.stage2.object_detection.utils.large_image_inferencer import LargeImageTiledInferencer


def _fake_predictor(images: list[np.ndarray]) -> list[Any]:
    pytest.importorskip("mmdet")
    from mmdet.structures import DetDataSample  # type: ignore[import-not-found]

    outputs: list[Any] = []
    for _image in images:
        boxes = torch.tensor([[20.0, 20.0, 60.0, 60.0]], dtype=torch.float32)
        labels = torch.tensor([1], dtype=torch.long)
        scores = torch.tensor([0.95], dtype=torch.float32)

        instances = InstanceData()
        instances.bboxes = boxes
        instances.labels = labels
        instances.scores = scores

        sample = DetDataSample()
        sample.pred_instances = instances
        outputs.append(sample)
    return outputs


def test_large_image_tiled_inferencer_end2end() -> None:
    pytest.importorskip("sahi")
    pytest.importorskip("mmdet")

    inferencer = LargeImageTiledInferencer(
        predictor=_fake_predictor,
        patch_size=128,
        patch_overlap_ratio=0.25,
        batch_size=2,
        merge_iou_thr=0.5,
        merge_nms_type="nms",
    )

    image = np.zeros((256, 256, 3), dtype=np.uint8)
    merged = inferencer.infer(image)
    assert hasattr(merged, "pred_instances")
    assert len(merged.pred_instances) > 0

    out_path = Path("/tmp/large_image_tiled_inferencer.png")
    inferencer.visualize(
        image=image,
        result=merged,
        class_names=["background", "target"],
        score_thr=0.1,
        out_file=out_path,
    )
    assert out_path.exists()
