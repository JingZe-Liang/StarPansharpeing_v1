from __future__ import annotations

from typing import Any

from mmengine.hooks import Hook
from mmengine.runner import Runner


class DetectionBatchHook(Hook):
    def __init__(
        self,
        dataset_tag: str = "DET",
        log_once: bool = True,
        log_input_shape: bool = True,
        log_num_boxes: bool = True,
    ) -> None:
        self.dataset_tag = dataset_tag
        self.log_once = log_once
        self.log_input_shape = log_input_shape
        self.log_num_boxes = log_num_boxes
        self._logged = False

    def _get_bbox_count(self, bboxes: Any) -> int | None:
        if hasattr(bboxes, "tensor"):
            tensor = bboxes.tensor
            if hasattr(tensor, "shape") and len(tensor.shape) > 0:
                return int(tensor.shape[0])
            return None
        if hasattr(bboxes, "shape") and len(bboxes.shape) > 0:
            return int(bboxes.shape[0])
        return None

    def before_train_iter(self, runner: Runner, batch_idx: int, data_batch: dict | None = None) -> None:
        del batch_idx
        if data_batch is None:
            return
        if self.log_once and self._logged:
            return

        tag = f"[{self.dataset_tag}]"
        inputs = data_batch.get("inputs", [])
        data_samples = data_batch.get("data_samples", [])

        if self.log_input_shape and inputs:
            first_input = inputs[0]
            shape = getattr(first_input, "shape", None)
            if shape is not None:
                runner.logger.info(f"{tag} First batch inputs: {shape}")

        if self.log_num_boxes and data_samples:
            first_sample = data_samples[0]
            gt_instances = getattr(first_sample, "gt_instances", None)
            if gt_instances is not None:
                bboxes = getattr(gt_instances, "bboxes", None)
                if bboxes is not None:
                    box_count = self._get_bbox_count(bboxes)
                    if box_count is not None:
                        runner.logger.info(f"{tag} First batch boxes: {box_count}")

        self._logged = True
