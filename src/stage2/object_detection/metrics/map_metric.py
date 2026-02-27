from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from mmengine.evaluator import BaseMetric
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def _as_list(data: Any) -> list[Any]:
    if data is None:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, tuple):
        return list(data)
    return [data]


def _collect_batch_samples(data_batch: Any) -> list[Any]:
    if isinstance(data_batch, dict):
        return _as_list(data_batch.get("data_samples"))
    if isinstance(data_batch, Sequence) and not isinstance(data_batch, str | bytes):
        samples: list[Any] = []
        for item in data_batch:
            if isinstance(item, dict):
                samples.extend(_as_list(item.get("data_samples")))
        return samples
    return []


def _get_field(obj: Any, name: str) -> Any:
    """Access a field from either a dict or an object (e.g. DetDataSample).

    mmdet's Runner may return predictions as plain dicts instead of
    DetDataSample objects depending on version / pipeline. This helper
    handles both transparently.
    """
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _extract_boxes(boxes: Any) -> torch.Tensor:
    if boxes is None:
        return torch.zeros((0, 4), dtype=torch.float32)
    if hasattr(boxes, "tensor"):
        boxes = boxes.tensor
    if hasattr(boxes, "detach"):
        boxes_t = boxes.detach()
    else:
        boxes_t = torch.as_tensor(boxes)
    if boxes_t.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float32)
    if boxes_t.ndim == 1:
        boxes_t = boxes_t.unsqueeze(0)
    return boxes_t[:, :4].float().cpu()


def _extract_labels(labels: Any, num_boxes: int) -> torch.Tensor:
    if labels is None:
        return torch.zeros((num_boxes,), dtype=torch.long)
    if hasattr(labels, "detach"):
        labels_t = labels.detach()
    else:
        labels_t = torch.as_tensor(labels)
    if labels_t.numel() == 0:
        return torch.zeros((num_boxes,), dtype=torch.long)
    if labels_t.ndim == 0:
        labels_t = labels_t.unsqueeze(0)
    labels_t = labels_t.long().cpu()
    if labels_t.numel() == num_boxes:
        return labels_t
    if labels_t.numel() > num_boxes:
        return labels_t[:num_boxes]
    pad = torch.zeros((num_boxes - labels_t.numel(),), dtype=torch.long)
    return torch.cat([labels_t, pad], dim=0)


def _extract_scores(scores: Any, num_boxes: int) -> torch.Tensor:
    if scores is None:
        return torch.zeros((num_boxes,), dtype=torch.float32)
    if hasattr(scores, "detach"):
        scores_t = scores.detach()
    else:
        scores_t = torch.as_tensor(scores)
    if scores_t.numel() == 0:
        return torch.zeros((num_boxes,), dtype=torch.float32)
    if scores_t.ndim == 0:
        scores_t = scores_t.unsqueeze(0)
    scores_t = scores_t.float().cpu()
    if scores_t.numel() == num_boxes:
        return scores_t
    if scores_t.numel() > num_boxes:
        return scores_t[:num_boxes]
    pad = torch.zeros((num_boxes - scores_t.numel(),), dtype=torch.float32)
    return torch.cat([scores_t, pad], dim=0)


def _to_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0.0
        return float(value.item())
    return float(value)


class DetectionMAPMetric(BaseMetric):
    default_prefix: str = "det"

    def __init__(self, iou_type: str = "bbox", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if iou_type not in {"bbox", "segm"}:
            raise ValueError(f"Unsupported iou_type: {iou_type}")
        self.iou_type = iou_type

    def process(self, data_batch: dict[str, Any], data_samples: list[Any]) -> None:
        batch_samples = _collect_batch_samples(data_batch)

        for idx, sample in enumerate(data_samples):
            pred_instances = _get_field(sample, "pred_instances")
            pred_boxes = _extract_boxes(_get_field(pred_instances, "bboxes"))
            pred_labels = _extract_labels(_get_field(pred_instances, "labels"), int(pred_boxes.shape[0]))
            pred_scores = _extract_scores(_get_field(pred_instances, "scores"), int(pred_boxes.shape[0]))

            gt_sample = batch_samples[idx] if idx < len(batch_samples) else sample
            gt_instances = _get_field(gt_sample, "gt_instances")
            gt_boxes = _extract_boxes(_get_field(gt_instances, "bboxes"))
            gt_labels = _extract_labels(_get_field(gt_instances, "labels"), int(gt_boxes.shape[0]))

            self.results.append(
                {
                    "pred": {"boxes": pred_boxes, "scores": pred_scores, "labels": pred_labels},
                    "target": {"boxes": gt_boxes, "labels": gt_labels},
                }
            )

    def compute_metrics(self, results: list[dict[str, dict[str, torch.Tensor]]]) -> dict[str, float]:
        if not results:
            return {"map": 0.0, "map50": 0.0}

        metric = MeanAveragePrecision(box_format="xyxy", iou_type=self.iou_type)  # type: ignore[arg-type]
        preds = [item["pred"] for item in results]
        targets = [item["target"] for item in results]
        metric.update(preds, targets)  # type: ignore[arg-type]
        summary = metric.compute()  # type: ignore[misc]

        return {
            "map": _to_float(summary.get("map", 0.0)),
            "map50": _to_float(summary.get("map_50", 0.0)),
        }
