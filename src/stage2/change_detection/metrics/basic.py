from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor
from torchmetrics.classification import ConfusionMatrix, JaccardIndex

from ...segmentation.metrics import HyperSegmentationScore


class ChangeDetectionScore(HyperSegmentationScore):
    def __init__(
        self,
        n_classes: int = 2,
        ignore_index: int | None = None,
        top_k: int = 1,
        reduction: Literal["micro", "macro", "weighted", "none"] | None = "micro",
        per_class: bool = False,
        include_bg: bool = False,
        input_format: Literal["one-hot", "index", "mixed"] = "index",
        f1_cd: bool = True,
    ) -> None:
        if f1_cd and n_classes != 2:
            raise ValueError(f"f1_cd=True only supports binary change detection (n_classes=2), got {n_classes}.")

        super().__init__(
            task="multiclass",
            n_classes=n_classes,
            ignore_index=ignore_index,
            top_k=top_k,
            reduction=reduction,
            per_class=per_class,
            include_bg=include_bg,
            input_format=input_format,
        )
        self.f1_cd = f1_cd
        self._cm_metric = ConfusionMatrix(
            task="multiclass" if n_classes > 2 else "binary",
            num_classes=n_classes,
            ignore_index=ignore_index,
        )
        if hasattr(self, "jaccard_score"):
            self.jaccard_score = JaccardIndex(
                task="multiclass",
                num_classes=n_classes,
                ignore_index=ignore_index,
                average="none",
            )
            self._all_metric_fns["jaccard"] = self.jaccard_score

    def _compute_custom_from_cm(self, cm: Tensor) -> dict[str, Tensor]:
        if cm.ndim != 2:
            default_metrics = {
                "mean_accuracy": torch.tensor(0.0, device=cm.device),
                "precision": torch.tensor(0.0, device=cm.device),
                "iou": torch.tensor(0.0, device=cm.device),
                "fwiou": torch.tensor(0.0, device=cm.device),
            }
            if self.f1_cd:
                default_metrics["f1_cd"] = torch.tensor(0.0, device=cm.device)
            return default_metrics

        class_sums = cm.sum(dim=1)
        acc_per_class = torch.diag(cm) / (class_sums + 1e-7)
        valid_acc = acc_per_class[~torch.isnan(acc_per_class)]
        mean_accuracy = valid_acc.mean() if valid_acc.numel() > 0 else torch.tensor(0.0, device=cm.device)

        if cm.shape[0] == 2:
            precision = cm[1, 1] / (cm[0, 1] + cm[1, 1] + 1e-7)
            iou = cm[1, 1] / (cm[0, 1] + cm[1, 0] + cm[1, 1] + 1e-7)
        else:
            precision = torch.tensor(0.0, device=cm.device)
            iou = torch.tensor(0.0, device=cm.device)

        freq = cm.sum(dim=1) / (cm.sum() + 1e-7)
        intersection = torch.diag(cm)
        union = cm.sum(dim=1) + cm.sum(dim=0) - intersection
        iou_per_class = intersection / (union + 1e-7)
        fwiou = (freq * iou_per_class).sum()

        out = {
            "mean_accuracy": mean_accuracy,
            "precision": precision,
            "iou": iou,
            "fwiou": fwiou,
        }
        if self.f1_cd:
            if cm.shape[0] != 2:
                raise ValueError(f"f1_cd=True requires binary confusion matrix shape [2,2], got {tuple(cm.shape)}.")
            recall = cm[1, 1] / (cm[1, 0] + cm[1, 1] + 1e-7)
            out["f1_cd"] = 2.0 * precision * recall / (precision + recall + 1e-7)

        return out

    def update(self, pred: Tensor, gt: Tensor, mask: Tensor | None = None) -> None:
        super().update(pred, gt, mask=mask)
        pred_m = self._mask_out(pred, mask)
        gt_m = self._mask_out(gt, mask)
        self._cm_metric.update(pred_m, gt_m)  # type: ignore[arg-type]

    def compute(self) -> dict[str, Tensor]:
        metrics = super().compute()
        cm = self._cm_metric.compute()  # type: ignore[misc]
        if cm is None:
            return metrics
        metrics.update(self._compute_custom_from_cm(cm))
        return metrics

    def reset(self) -> None:
        super().reset()
        self._cm_metric.reset()
