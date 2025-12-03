import warnings
from typing import Any, Callable, Literal

import torch
import torch.nn as nn
from jaxtyping import Int
from loguru import logger
from torch import Tensor
from torchmetrics.classification import Accuracy, CohenKappa, F1Score, Precision, Recall
from torchmetrics.segmentation import GeneralizedDiceScore, MeanIoU


class HyperSegmentationScore(nn.Module):
    def __init__(
        self,
        n_classes: int,
        ignore_index: int | None = None,
        task: Literal["binary", "multiclass"] = "multiclass",
        top_k: int = 1,
        reduction: Literal["micro", "macro", "weighted", "none"] | None = "micro",
        per_class: bool | None = None,
        include_bg: bool = False,
        input_format: Literal["one-hot", "index", "mixed"] = "index",
        use_aggregation: bool = False,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.task = task
        self.top_k = top_k
        self.reduction = reduction

        # macro means per class; micro means over all classes
        self.per_class = per_class if isinstance(per_class, bool) else reduction == "macro"

        if ignore_index is not None:
            if include_bg:
                logger.warning(
                    f"include_bg=True is automatically set to False when ignore_index={ignore_index} "
                    f"is used. This prevents mixing ignored pixels (mapped to class 0) with "
                    f"real background class 0 in segmentation metrics.",
                )
            self.include_bg = False
        else:
            self.include_bg = include_bg

        self.input_format = input_format

        # Classification metrics support ignore_index directly
        self.accuracy = Accuracy(
            task=task,
            num_classes=n_classes,
            ignore_index=ignore_index,
            top_k=top_k,
            average=reduction,
        )
        self.precision = Precision(
            task=task,
            num_classes=n_classes,
            ignore_index=ignore_index,
            top_k=top_k,
            average=reduction,
        )
        self.recall = Recall(
            task=task,
            num_classes=n_classes,
            ignore_index=ignore_index,
            top_k=top_k,
            average=reduction,
        )
        self.cohen_kappa = CohenKappa(task=task, num_classes=n_classes, ignore_index=ignore_index)
        self.f1_score = F1Score(
            task=task,
            num_classes=n_classes,
            average=reduction,
            top_k=top_k,
            ignore_index=ignore_index,
        )

        # Segmentation metrics need special handling for ignore_index
        # We map ignore_index to class 0 and shift valid classes by +1,
        # so total number of classes becomes n_classes + 1.
        if ignore_index is not None:
            if ignore_index < n_classes:
                raise ValueError(f"ignore_index {ignore_index} should exceed number of classes {n_classes}")
            seg_n_classes = n_classes + 1
        else:
            seg_n_classes = n_classes

        self.dice_score = GeneralizedDiceScore(
            include_background=self.include_bg,
            num_classes=seg_n_classes,
            per_class=self.per_class,
            input_format=input_format,
        )
        self.mean_iou = MeanIoU(
            num_classes=seg_n_classes,
            per_class=self.per_class,
            include_background=self.include_bg,
            input_format=input_format,
        )

        self._all_metric_fns = dict(
            accuracy=self.accuracy,
            precision=self.precision,
            recall=self.recall,
            kappa=self.cohen_kappa,
            dice=self.dice_score,
            mean_iou=self.mean_iou,
            f1_score=self.f1_score,
        )

        # self.use_aggregation = use_aggregation

    def _update_all_metrics(self, pred, gt):
        """
        Update all metrics with prediction and ground truth.

        Strategy for handling ignore_index:
        - Classification metrics: Use ignore_index parameter directly
        - Segmentation metrics: Map ignore_index to 0, with include_bg=False
          to avoid mixing ignored pixels with real background class
        """
        # Classification metrics support ignore_index directly
        classification_metrics = {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "kappa": self.cohen_kappa,
            "f1_score": self.f1_score,
        }

        # Segmentation metrics need special handling
        segmentation_metrics = {
            "dice": self.dice_score,
            "mean_iou": self.mean_iou,
        }

        # Update classification metrics directly
        for metric in classification_metrics.values():
            metric.update(pred, gt)

        # Update segmentation metrics with preprocessing for ignore_index
        if self.ignore_index is not None:
            # This ensures ignored pixels don't interfere with real background evaluation
            pred_processed = pred.clone()
            gt_processed = gt.clone()

            # Replace ignore_index with 0 (background class)
            # ignore_mask = gt_processed == self.ignore_index
            # gt_processed[ignore_mask] = 0
            gt_processed = torch.where(gt_processed == self.ignore_index, 0, gt_processed + 1)
            pred_processed = pred_processed + 1

            # Update segmentation metrics with processed data
            for metric in segmentation_metrics.values():
                metric.update(pred_processed, gt_processed)
        else:
            # No ignore_index, use original data
            for metric in segmentation_metrics.values():
                metric.update(pred, gt)

    def _compute_all_metrics(self):
        return {name: metric.compute() for name, metric in self._all_metric_fns.items()}

    def _reset_all_metrics(self):
        for metric in self._all_metric_fns.values():
            metric.reset()

    def update(self, pred, gt, mask=None):
        pred = self._mask_out(pred, mask)
        gt = self._mask_out(gt, mask)
        self._update_all_metrics(pred, gt)

    def compute(self):
        return self._compute_all_metrics()

    def reset(self):
        self._reset_all_metrics()

    def _mask_out(self, x: Tensor, mask: Tensor | None) -> Tensor:
        if mask is not None:
            assert mask.ndim == 3, "Mask must be 3D tensor, shaped as [bs, h, w]"
            x = torch.masked_select(x, mask)[None, None, :]
        return x

    def forward(
        self,
        pred: Int[Tensor, "... h w"],
        gt: Int[Tensor, "... h w"],
        mask: Int[Tensor, "... h w"] | None = None,
    ):
        """
        Forward metrics
            pred: Predicted labels (batch_size, height, width)
            gt: Ground truth labels (batch_size, height, width)
            mask: Optional mask to ignore certain pixels (batch_size, height, width)
        """
        assert pred.ndim == gt.ndim == 3, "Prediction and ground truth must be 3D tensors, shaped as [bs, h, w]"
        pred = self._mask_out(pred, mask)
        gt = self._mask_out(gt, mask)
        self._update_all_metrics(pred, gt)
        metrics = self._compute_all_metrics()
        return metrics


# * --- Test --- #


def test_metrics():
    """Test basic functionality of HyperSegmentationScore with random data"""
    # Create test data
    generator = torch.Generator()
    pred = torch.randint(0, 5, (2, 32, 32), generator=generator.manual_seed(42))  # 2 images, 32x32, 5 classes
    gt = torch.randint(0, 5, (2, 32, 32), generator=generator.manual_seed(43))  # 2 images, 32x32, 5 classes

    # Initialize metrics
    metrics = HyperSegmentationScore(
        n_classes=5,
        ignore_index=None,
        top_k=1,
        reduction="macro",
        per_class=True,
        include_bg=True,
        use_aggregation=True,
    )

    # Calculate metrics
    results = metrics(pred, gt)

    # Check that all metrics return valid values
    for metric_name, value in results.items():
        assert isinstance(value, torch.Tensor), f"{metric_name} should return a tensor"
        assert not torch.isnan(value).any(), f"{metric_name} should not be NaN"
        # Cohen's kappa can be negative, so we don't check >= 0 for it
        if metric_name != "kappa":
            assert (value >= 0).all(), f"{metric_name} should be non-negative"
        if metric_name in ["accuracy", "recall", "dice", "mean_iou", "f1_score"]:
            assert (value <= 1).all(), f"{metric_name} should be <= 1"
        # Cohen's kappa should be between -1 and 1
        if metric_name == "kappa":
            assert ((value >= -1) & (value <= 1)).all(), f"{metric_name} should be between -1 and 1"

    print("✓ Basic test passed")
    return results


def test_perfect_prediction():
    """Test with perfect predictions (should give maximum scores)"""
    # Create perfect predictions
    pred = torch.tensor([[[0, 1], [2, 3]], [[1, 2], [3, 0]]])  # 2 images, 2x2, 4 classes
    gt = pred.clone()

    metrics = HyperSegmentationScore(
        n_classes=4,
        ignore_index=None,
        top_k=1,
        reduction="macro",
        per_class=False,
        include_bg=True,
        use_aggregation=False,
    )

    results = metrics(pred, gt)

    # Perfect predictions should give maximum scores
    assert results["accuracy"] == 1.0, "Perfect prediction should give accuracy=1"
    assert results["f1_score"] == 1.0, "Perfect prediction should give F1=1"
    assert results["mean_iou"] == 1.0, "Perfect prediction should give IoU=1"
    assert results["dice"] == 1.0, "Perfect prediction should give Dice=1"
    assert results["recall"] == 1.0, "Perfect prediction should give recall=1"
    assert results["kappa"] == 1.0, "Perfect prediction should give kappa=1"

    print("✓ Perfect prediction test passed")


def test_wrong_prediction():
    """Test with completely wrong predictions"""
    # Create completely wrong predictions
    pred = torch.tensor([[[0, 1], [2, 3]], [[1, 2], [3, 0]]])  # 2 images, 2x2, 4 classes
    gt = torch.tensor([[[1, 2], [3, 0]], [[2, 3], [0, 1]]])  # All different

    metrics = HyperSegmentationScore(
        n_classes=4,
        ignore_index=None,
        top_k=1,
        reduction="macro",
        per_class=False,
        include_bg=True,
        use_aggregation=False,
    )

    results = metrics(pred, gt)

    # All wrong predictions should give minimum scores
    assert results["accuracy"] == 0.0, "All wrong prediction should give accuracy=0"
    assert results["f1_score"] == 0.0, "All wrong prediction should give F1=0"
    assert results["mean_iou"] == 0.0, "All wrong prediction should give IoU=0"
    assert results["dice"] == 0.0, "All wrong prediction should give Dice=0"
    assert results["recall"] == 0.0, "All wrong prediction should give recall=0"

    print("✓ Wrong prediction test passed")


def test_ignore_index():
    """Test with ignore_index functionality"""
    pred = torch.tensor([[[0, 1, 2], [1, 2, 0]]])  # 1 image, 2x3, 3 classes
    gt = torch.tensor([[[0, 1, 255], [1, 255, 0]]])  # 255 = ignore index

    metrics = HyperSegmentationScore(
        n_classes=256,  # Need to include the ignore index
        ignore_index=255,
        top_k=1,
        reduction="macro",
        per_class=False,
        include_bg=True,
        use_aggregation=False,
    )

    results = metrics(pred, gt)

    # Should ignore the 255 positions
    assert isinstance(results["accuracy"], torch.Tensor), "Should return a tensor"
    assert not torch.isnan(results["accuracy"]).any(), "Should not be NaN"
    assert (results["accuracy"] >= 0).all() and (results["accuracy"] <= 1).all(), "Accuracy should be between 0 and 1"

    print("✓ Ignore index test passed")


def run_all_tests():
    """Run all test cases"""
    print("Running HyperSegmentationScore tests...")

    test_metrics()
    test_perfect_prediction()
    test_wrong_prediction()
    test_ignore_index()

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    run_all_tests()
