import torch
import torch.nn as nn
from torchmetrics.classification import ConfusionMatrix

from ...segmentation.metrics import HyperSegmentationScore


class ChangeDetectionMetric(nn.Module):
    """Custom metric module for confusion matrix-based calculations"""

    def __init__(self, n_classes: int, ignore_index: int | None = None):
        super().__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = ConfusionMatrix(
            task="multiclass", num_classes=n_classes, ignore_index=ignore_index
        )
        self._last_pred = None
        self._last_gt = None

    def update(self, pred: torch.Tensor, gt: torch.Tensor):
        """Update the metric with new predictions and ground truth"""
        self._last_pred = pred
        self._last_gt = gt
        self.confusion_matrix.update(pred, gt)

    def compute(self) -> torch.Tensor:
        """Compute the metric value"""
        if self._last_pred is None or self._last_gt is None:
            return torch.tensor(0.0)

        cm = self.confusion_matrix.compute()
        return cm

    def reset(self):
        """Reset the metric state"""
        self.confusion_matrix.reset()
        self._last_pred = None
        self._last_gt = None


class ChangeDetectionScore(HyperSegmentationScore):
    def __init__(
        self,
        n_classes: int = 2,
        ignore_index: int | None = None,
        top_k: int = 1,
        reduction: str = "micro",
        per_class: bool = False,
        include_bg: bool = False,
        use_aggregation: bool = False,
    ):
        super().__init__(
            task="binary",
            n_classes=n_classes,
            ignore_index=ignore_index,
            top_k=top_k,
            reduction=reduction,
            per_class=per_class,
            include_bg=include_bg,
            use_aggregation=use_aggregation,
        )

        # Add custom metrics to _all_metric_fns for forward compatibility
        self.mean_accuracy_metric = ChangeDetectionMetric(n_classes, ignore_index)
        self.precision_metric = ChangeDetectionMetric(n_classes, ignore_index)
        self.iou_metric = ChangeDetectionMetric(n_classes, ignore_index)
        self.fwiou_metric = ChangeDetectionMetric(n_classes, ignore_index)

        self._all_metric_fns.update(
            dict(
                mean_accuracy=self.mean_accuracy_metric,
                precision=self.precision_metric,
                iou=self.iou_metric,
                fwiou=self.fwiou_metric,
            )
        )

        # Keep original confusion matrix for direct access
        self.confusion_matrix = ConfusionMatrix(
            task="multiclass", num_classes=n_classes, ignore_index=ignore_index
        )
        self._all_metric_fns.update(dict(confusion_matrix=self.confusion_matrix))

    def _compute_custom_metric(self, metric_func, pred, gt):
        """Helper to compute custom metrics"""
        cm = self._get_confusion_matrix(pred, gt)
        return metric_func(cm)

    def _get_confusion_matrix(
        self, pred: torch.Tensor, gt: torch.Tensor
    ) -> torch.Tensor:
        """Get confusion matrix tensor"""
        # Update confusion matrix
        self.confusion_matrix.update(pred, gt)
        cm = self.confusion_matrix.compute()
        self.confusion_matrix.reset()
        return cm

    def compute_mean_accuracy(
        self, pred: torch.Tensor, gt: torch.Tensor
    ) -> torch.Tensor:
        """Compute mean accuracy (average of per-class accuracy)"""

        def _compute_mean_accuracy(cm):
            # Calculate accuracy for each class
            class_sums = cm.sum(dim=1)
            acc_per_class = torch.diag(cm) / (class_sums + 1e-7)

            # Calculate mean accuracy, ignoring NaN values
            valid_acc = acc_per_class[~torch.isnan(acc_per_class)]
            if len(valid_acc) > 0:
                mean_acc = valid_acc.mean()
            else:
                mean_acc = torch.tensor(0.0, device=cm.device)

            return mean_acc

        return self._compute_custom_metric(_compute_mean_accuracy, pred, gt)

    def compute_precision(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Compute precision (for positive class in binary classification)"""

        def _compute_precision(cm):
            assert cm.shape[0] == 2, "Precision only supports binary classification"
            # Precision = TP / (TP + FP) = cm[1, 1] / (cm[0, 1] + cm[1, 1])
            precision = cm[1, 1] / (cm[0, 1] + cm[1, 1] + 1e-7)
            return precision

        return self._compute_custom_metric(_compute_precision, pred, gt)

    def compute_iou(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Compute IoU (Intersection over Union for positive class in binary classification)"""

        def _compute_iou(cm):
            assert cm.shape[0] == 2, "IoU only supports binary classification"
            # IoU = TP / (FP + FN + TP) = cm[1, 1] / (cm[0, 1] + cm[1, 0] + cm[1, 1])
            iou = cm[1, 1] / (cm[0, 1] + cm[1, 0] + cm[1, 1] + 1e-7)
            return iou

        return self._compute_custom_metric(_compute_iou, pred, gt)

    def compute_frequency_weighted_iou(
        self, pred: torch.Tensor, gt: torch.Tensor
    ) -> torch.Tensor:
        """Compute frequency weighted IoU"""

        def _compute_fwiou(cm):
            # Calculate frequency
            freq = cm.sum(dim=1) / cm.sum()

            # Calculate IoU for each class
            intersection = torch.diag(cm)
            union = cm.sum(dim=1) + cm.sum(dim=0) - intersection
            iou_per_class = intersection / (union + 1e-7)

            # Calculate weighted IoU
            fwiou = (freq * iou_per_class).sum()
            return fwiou

        return self._compute_custom_metric(_compute_fwiou, pred, gt)

    def forward(self, pred: torch.Tensor, gt: torch.Tensor):
        """Override forward to handle custom metrics properly"""
        # Update all metrics including custom ones
        self._update_all_metrics(pred, gt)

        # Get basic metrics
        metrics = self._compute_all_metrics()

        # Compute and add custom metrics
        metrics["mean_accuracy"] = self.compute_mean_accuracy(pred, gt)
        metrics["precision"] = self.compute_precision(pred, gt)
        metrics["iou"] = self.compute_iou(pred, gt)
        metrics["fwiou"] = self.compute_frequency_weighted_iou(pred, gt)

        return metrics
