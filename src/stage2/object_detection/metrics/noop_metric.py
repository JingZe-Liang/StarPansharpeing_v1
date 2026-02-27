from __future__ import annotations

from typing import Any

from mmengine.evaluator import BaseMetric


class DetectionNoopMetric(BaseMetric):
    def process(self, data_batch: dict[str, Any], data_samples: list[Any]) -> None:
        del data_batch
        for sample in data_samples:
            pred_instances = getattr(sample, "pred_instances", None)
            box_count = 0
            if pred_instances is not None:
                bboxes = getattr(pred_instances, "bboxes", None)
                if bboxes is not None and hasattr(bboxes, "shape") and len(bboxes.shape) > 0:
                    box_count = int(bboxes.shape[0])
            self.results.append({"pred_boxes": box_count})

    def compute_metrics(self, results: list[dict[str, int]]) -> dict[str, float]:
        total_samples = float(len(results))
        total_boxes = float(sum(item["pred_boxes"] for item in results))
        mean_boxes = total_boxes / total_samples if total_samples > 0 else 0.0
        return {
            "val_num_samples": total_samples,
            "val_mean_pred_boxes": mean_boxes,
        }
