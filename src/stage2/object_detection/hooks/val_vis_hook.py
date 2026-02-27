from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from mmengine.hooks import Hook
from mmengine.runner import Runner

from src.stage2.object_detection.utils.vis import draw_detections


def _as_list(data: Any) -> list[Any]:
    if data is None:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, tuple):
        return list(data)
    return [data]


def _extract_xyxy_boxes(boxes: Any) -> torch.Tensor:
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


def _extract_labels(labels: Any, default_len: int) -> list[int]:
    if labels is None:
        return [0] * default_len
    if hasattr(labels, "detach"):
        labels_t = labels.detach().long().cpu()
    else:
        labels_t = torch.as_tensor(labels, dtype=torch.long)
    if labels_t.ndim == 0:
        labels_t = labels_t.unsqueeze(0)
    labels_list = labels_t.tolist()
    if len(labels_list) < default_len:
        labels_list = labels_list + [0] * (default_len - len(labels_list))
    return [int(v) for v in labels_list[:default_len]]


def _extract_scores(scores: Any, default_len: int) -> torch.Tensor:
    if scores is None:
        return torch.ones((default_len,), dtype=torch.float32)
    if hasattr(scores, "detach"):
        scores_t = scores.detach().float().cpu()
    else:
        scores_t = torch.as_tensor(scores, dtype=torch.float32)
    if scores_t.ndim == 0:
        scores_t = scores_t.unsqueeze(0)
    if scores_t.numel() < default_len:
        pad = torch.ones((default_len - scores_t.numel(),), dtype=torch.float32)
        scores_t = torch.cat([scores_t, pad], dim=0)
    return scores_t[:default_len]


class DetectionValVisHook(Hook):
    def __init__(
        self,
        save_dir: str = "vis",
        max_images: int = 16,
        score_thr: float = 0.3,
        draw_gt: bool = True,
    ) -> None:
        self.save_dir = save_dir
        self.max_images = max_images
        self.score_thr = score_thr
        self.draw_gt = draw_gt
        self._saved = 0

    def before_val_epoch(self, runner: Runner) -> None:
        self._saved = 0

    def after_val_iter(
        self,
        runner: Runner,
        batch_idx: int,
        data_batch: dict[str, Any] | None = None,
        outputs: list[Any] | Any | None = None,
    ) -> None:
        del batch_idx
        if self._saved >= self.max_images or data_batch is None or outputs is None:
            return

        inputs = _as_list(data_batch.get("inputs"))
        gt_samples = _as_list(data_batch.get("data_samples"))
        pred_samples = _as_list(outputs)
        batch_size = min(len(inputs), len(pred_samples))
        if batch_size == 0:
            return

        epoch_id = int(runner.epoch) + 1
        out_dir = Path(runner.work_dir) / self.save_dir / f"epoch_{epoch_id:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        for item_idx in range(batch_size):
            if self._saved >= self.max_images:
                break

            image = inputs[item_idx]
            pred = pred_samples[item_idx]
            pred_instances = getattr(pred, "pred_instances", None)
            pred_boxes = _extract_xyxy_boxes(getattr(pred_instances, "bboxes", None))
            pred_labels = _extract_labels(getattr(pred_instances, "labels", None), int(pred_boxes.shape[0]))
            pred_scores = _extract_scores(getattr(pred_instances, "scores", None), int(pred_boxes.shape[0]))

            keep = pred_scores >= self.score_thr
            vis_pred_boxes = pred_boxes[keep]
            vis_pred_labels = [pred_labels[idx] for idx, k in enumerate(keep.tolist()) if k]

            pred_img = draw_detections(image=image, boxes_xyxy=vis_pred_boxes, labels_xyxy=vis_pred_labels)
            base_name = f"val_{self._saved:04d}"
            pred_img.save(out_dir / f"{base_name}_pred.jpg")

            if self.draw_gt and item_idx < len(gt_samples):
                gt = gt_samples[item_idx]
                gt_instances = getattr(gt, "gt_instances", None)
                gt_boxes = _extract_xyxy_boxes(getattr(gt_instances, "bboxes", None))
                gt_labels = _extract_labels(getattr(gt_instances, "labels", None), int(gt_boxes.shape[0]))
                gt_img = draw_detections(image=image, boxes_xyxy=gt_boxes, labels_xyxy=gt_labels)
                gt_img.save(out_dir / f"{base_name}_gt.jpg")

            self._saved += 1
