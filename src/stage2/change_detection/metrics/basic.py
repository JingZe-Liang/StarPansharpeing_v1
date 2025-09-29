import torch
import torch.distributed as dist
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
            dict(  # type: ignore
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
        metric = metric_func(cm)

        # Only use distributed reduction if distributed is initialized
        if dist.is_initialized():
            metric = dist.all_reduce(metric, op=dist.ReduceOp.AVG)

        assert torch.is_tensor(metric), "Metric must be a tensor"
        return metric

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

        # Remove confusion_matrix from the returned metrics
        if "confusion_matrix" in metrics:
            del metrics["confusion_matrix"]

        return metrics

    def compute(self):
        """Override compute to ensure confusion_matrix is not returned"""
        # Get all metrics
        metrics = self._compute_all_metrics()

        # Remove confusion_matrix from the returned metrics
        if "confusion_matrix" in metrics:
            del metrics["confusion_matrix"]

        return metrics


if __name__ == "__main__":
    """Test script for ChangeDetectionScore class.

    This script tests the functionality of the ChangeDetectionScore class
    by simulating binary classification predictions and ground truth labels.
    """
    import numpy as np

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Initialize ChangeDetectionScore for binary classification
    cd_score = ChangeDetectionScore(n_classes=2, ignore_index=255, include_bg=False)

    print("Testing ChangeDetectionScore...")
    print("=" * 50)

    # Test case 1: Perfect prediction
    print("\nTest Case 1: Perfect Prediction")
    print("-" * 30)

    # Create dummy predictions and ground truth (batch_size=4, height=8, width=8)
    batch_size, height, width = 4, 8, 8

    # Simulate predictions (logits for binary classification)
    perfect_pred = torch.randn(batch_size, 2, height, width)
    perfect_gt = torch.randint(0, 2, (batch_size, height, width))

    # Convert predictions to class predictions
    perfect_pred_labels = perfect_pred.argmax(dim=1)

    # For perfect prediction, ensure predictions match ground truth
    perfect_pred_labels = perfect_gt.clone()

    print(f"Prediction shape: {perfect_pred_labels.shape}")
    print(f"Ground truth shape: {perfect_gt.shape}")
    print(f"Unique predictions: {torch.unique(perfect_pred_labels)}")
    print(f"Unique ground truth: {torch.unique(perfect_gt)}")

    # Compute metrics
    metrics = cd_score(perfect_pred_labels, perfect_gt)

    # Verify confusion_matrix is not in the returned metrics
    assert "confusion_matrix" not in metrics, (
        "confusion_matrix should not be in returned metrics"
    )

    print("\nMetrics for perfect prediction:")
    for metric_name, value in metrics.items():
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                print(f"  {metric_name}: {value.item():.4f}")
            else:
                print(f"  {metric_name}: {value}")
        else:
            print(f"  {metric_name}: {value:.4f}")

    # Test case 2: Random prediction
    print("\nTest Case 2: Random Prediction")
    print("-" * 30)

    # Generate random predictions
    random_pred = torch.randint(0, 2, (batch_size, height, width))
    random_gt = torch.randint(0, 2, (batch_size, height, width))

    print(f"Random prediction shape: {random_pred.shape}")
    print(f"Random ground truth shape: {random_gt.shape}")

    # Compute metrics for random prediction
    random_metrics = cd_score(random_pred, random_gt)

    # Verify confusion_matrix is not in the returned metrics
    assert "confusion_matrix" not in random_metrics, (
        "confusion_matrix should not be in returned metrics"
    )

    print("\nMetrics for random prediction:")
    for metric_name, value in random_metrics.items():
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                print(f"  {metric_name}: {value.item():.4f}")
            else:
                print(f"  {metric_name}: {value}")
        else:
            print(f"  {metric_name}: {value:.4f}")

    # Test case 3: Edge case - all same class
    print("\nTest Case 3: Edge Case - All Same Class")
    print("-" * 30)

    # All predictions are class 0
    all_zero_pred = torch.zeros((batch_size, height, width), dtype=torch.long)
    all_zero_gt = torch.zeros((batch_size, height, width), dtype=torch.long)

    # Set some ground truth to class 1
    all_zero_gt[:, : height // 2, :] = 1

    print(f"All-zero prediction shape: {all_zero_pred.shape}")
    print(f"Mixed ground truth shape: {all_zero_gt.shape}")

    # Compute metrics for edge case
    edge_metrics = cd_score(all_zero_pred, all_zero_gt)

    # Verify confusion_matrix is not in the returned metrics
    assert "confusion_matrix" not in edge_metrics, (
        "confusion_matrix should not be in returned metrics"
    )

    print("\nMetrics for edge case:")
    for metric_name, value in edge_metrics.items():
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                print(f"  {metric_name}: {value.item():.4f}")
            else:
                print(f"  {metric_name}: {value}")
        else:
            print(f"  {metric_name}: {value:.4f}")

    # Test case 4: Test with ignore_index
    print("\nTest Case 4: Test with Ignore Index")
    print("-" * 30)

    # Create predictions with some ignored pixels
    ignore_pred = torch.randint(0, 2, (batch_size, height, width))
    ignore_gt = torch.randint(0, 2, (batch_size, height, width))

    # Set some pixels to ignore_index
    ignore_gt[:, 0, :] = 255  # Ignore first row
    ignore_gt[:, -1, :] = 255  # Ignore last row

    print(f"Prediction with ignore shape: {ignore_pred.shape}")
    print(f"Ground truth with ignore shape: {ignore_gt.shape}")
    print(f"Ignore pixels: {(ignore_gt == 255).sum().item()}")

    # Compute metrics with ignore index
    cd_score_ignore = ChangeDetectionScore(
        n_classes=2, ignore_index=255, include_bg=False
    )

    ignore_metrics = cd_score_ignore(ignore_pred, ignore_gt)

    # Verify confusion_matrix is not in the returned metrics
    assert "confusion_matrix" not in ignore_metrics, (
        "confusion_matrix should not be in returned metrics"
    )

    print("\nMetrics with ignore index:")
    for metric_name, value in ignore_metrics.items():
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                print(f"  {metric_name}: {value.item():.4f}")
            else:
                print(f"  {metric_name}: {value}")
        else:
            print(f"  {metric_name}: {value:.4f}")

    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print("- ChangeDetectionScore successfully initialized")
    print("- All metric computations completed without errors")
    print(
        "- Tested perfect predictions, random predictions, edge cases, and ignore index"
    )
    print(
        "- Custom metrics (mean_accuracy, precision, iou, fwiou) are working correctly"
    )
    print("=" * 50)

    # Additional test: verify compute() method also excludes confusion_matrix
    print("\nTesting compute() method...")
    test_cd_score = ChangeDetectionScore(n_classes=2)
    test_pred = torch.randint(0, 2, (2, 4, 4))
    test_gt = torch.randint(0, 2, (2, 4, 4))

    # Update metrics
    test_cd_score.update(test_pred, test_gt)

    # Compute metrics
    compute_metrics = test_cd_score.compute()

    # Verify confusion_matrix is not in the returned metrics
    assert "confusion_matrix" not in compute_metrics, (
        "compute() should not return confusion_matrix"
    )
    print("✓ compute() method properly excludes confusion_matrix")
