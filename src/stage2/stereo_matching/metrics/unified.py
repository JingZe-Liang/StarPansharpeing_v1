"""
Unified Metrics for Stereo Matching and Semantic Segmentation

This module provides a comprehensive metrics suite that combines:
1. Stereo Matching Metrics: EPE, D1, Threshold-based errors
2. Semantic Segmentation Metrics: IoU, F1, Dice, Pixel Accuracy

Compatible with both SemStereo and custom implementations.
Supports DDP training with torchmetrics backend.
"""

from typing import Any

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassJaccardIndex,
)


# =============================================================================
# Stereo Matching Metrics (Compatible with SemStereo)
# =============================================================================


def compute_epe(
    pred: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
    min_disp: float | None = None,
    max_disp: float | None = None,
) -> Tensor:
    """
    Compute End-Point Error (EPE) - Mean Absolute Error for disparity.

    Compatible with SemStereo's EPE_metric.

    Args:
        pred: Predicted disparity [B, H, W] or [B, 1, H, W]
        target: Ground truth disparity [B, H, W] or [B, 1, H, W]
        mask: Optional valid pixel mask [B, H, W]
        min_disp: Minimum valid disparity (for auto mask generation)
        max_disp: Maximum valid disparity (for auto mask generation)

    Returns:
        EPE value (scalar tensor)
    """
    # Unify shape
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)

    # Generate or use mask
    if mask is None:
        if min_disp is not None and max_disp is not None:
            mask = (target >= min_disp) & (target < max_disp)
        else:
            mask = torch.ones_like(target, dtype=torch.bool)

    # Compute MAE
    error = torch.abs(pred - target)
    masked_error = error[mask]

    if masked_error.numel() > 0:
        return masked_error.mean()
    return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)


def compute_d1(
    pred: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
    min_disp: float | None = None,
    max_disp: float | None = None,
    abs_threshold: float = 3.0,
    rel_threshold: float = 0.05,
) -> Tensor:
    """
    Compute D1 Error - Bad pixels with both absolute and relative thresholds.

    Compatible with SemStereo's D1_metric.
    A pixel is "bad" if: (error > 3px) AND (error / |gt| > 5%)

    Args:
        pred: Predicted disparity
        target: Ground truth disparity
        mask: Valid pixel mask
        min_disp, max_disp: Range for auto mask
        abs_threshold: Absolute error threshold (default 3.0 pixels)
        rel_threshold: Relative error threshold (default 0.05 = 5%)

    Returns:
        D1 error (ratio of bad pixels)
    """
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)

    if mask is None:
        if min_disp is not None and max_disp is not None:
            mask = (target >= min_disp) & (target < max_disp)
        else:
            mask = torch.ones_like(target, dtype=torch.bool)

    # Apply mask
    pred_valid = pred[mask]
    target_valid = target[mask]

    if target_valid.numel() == 0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    # Compute error
    error = torch.abs(target_valid - pred_valid)

    # Bad pixel criteria: (error > abs_threshold) AND (error/|target| > rel_threshold)
    bad_pixels = (error > abs_threshold) & (error / target_valid.abs() > rel_threshold)

    return bad_pixels.float().mean()


def compute_threshold_error(
    pred: Tensor,
    target: Tensor,
    threshold: float,
    mask: Tensor | None = None,
    min_disp: float | None = None,
    max_disp: float | None = None,
) -> Tensor:
    """
    Compute threshold-based error (Thres_metric in SemStereo).

    Percentage of pixels where error > threshold.

    Args:
        pred: Predicted disparity
        target: Ground truth disparity
        threshold: Error threshold in pixels
        mask: Valid pixel mask
        min_disp, max_disp: Range for auto mask

    Returns:
        Ratio of pixels with error > threshold
    """
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)

    if mask is None:
        if min_disp is not None and max_disp is not None:
            mask = (target >= min_disp) & (target < max_disp)
        else:
            mask = torch.ones_like(target, dtype=torch.bool)

    pred_valid = pred[mask]
    target_valid = target[mask]

    if target_valid.numel() == 0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    error = torch.abs(target_valid - pred_valid)
    bad_pixels = error > threshold

    return bad_pixels.float().mean()


# =============================================================================
# TorchMetrics-based Classes (for DDP training)
# =============================================================================


class StereoMatchingMetrics(Metric):
    """
    Comprehensive stereo matching metrics compatible with SemStereo.

    Computes: EPE, D1, Thres1px, Thres2px
    """

    def __init__(
        self,
        min_disp: float = -128,
        max_disp: float = 128,
        compute_d1: bool = True,
        compute_thresholds: list[float] = [1.0, 2.0],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.min_disp = min_disp
        self.max_disp = max_disp
        self.compute_d1_flag = compute_d1
        self.thresholds = compute_thresholds

        # EPE states
        self.add_state("epe_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("epe_count", default=torch.tensor(0), dist_reduce_fx="sum")

        # D1 states
        if compute_d1:
            self.add_state("d1_bad", default=torch.tensor(0), dist_reduce_fx="sum")
            self.add_state("d1_total", default=torch.tensor(0), dist_reduce_fx="sum")

        # Threshold states
        for thresh in compute_thresholds:
            self.add_state(
                f"thres{int(thresh)}_bad",
                default=torch.tensor(0),
                dist_reduce_fx="sum",
            )
            self.add_state(
                f"thres{int(thresh)}_total",
                default=torch.tensor(0),
                dist_reduce_fx="sum",
            )

    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        Update metric states.

        Args:
            preds: Predicted disparity [B, H, W] or [B, 1, H, W]
            target: Ground truth disparity [B, H, W] or [B, 1, H, W]
        """
        # Unify shape
        if preds.dim() == 4:
            preds = preds.squeeze(1)
        if target.dim() == 4:
            target = target.squeeze(1)

        # Generate mask
        mask = (target >= self.min_disp) & (target < self.max_disp)

        # Extract valid pixels once
        preds_valid = preds[mask]
        target_valid = target[mask]

        if target_valid.numel() == 0:
            return

        # Compute error once for all metrics
        error = torch.abs(preds_valid - target_valid)

        # EPE
        self.epe_sum += error.sum()
        self.epe_count += error.numel()

        # D1
        if self.compute_d1_flag:
            bad_d1 = (error > 3) & (error / target_valid.abs() > 0.05)
            self.d1_bad += bad_d1.sum()
            self.d1_total += target_valid.numel()

        # Threshold errors
        for thresh in self.thresholds:
            bad_thresh = error > thresh
            setattr(
                self,
                f"thres{int(thresh)}_bad",
                getattr(self, f"thres{int(thresh)}_bad") + bad_thresh.sum(),
            )
            setattr(
                self,
                f"thres{int(thresh)}_total",
                getattr(self, f"thres{int(thresh)}_total") + target_valid.numel(),
            )

    def compute(self) -> dict[str, Tensor]:
        """
        Compute final metric values.

        Returns:
            Dictionary with all metric values
        """
        results = {}

        # EPE
        if self.epe_count > 0:
            results["EPE"] = self.epe_sum / self.epe_count
        else:
            results["EPE"] = torch.tensor(0.0, device=self.epe_sum.device)

        # D1
        if self.compute_d1_flag and self.d1_total > 0:
            results["D1"] = self.d1_bad.float() / self.d1_total
        elif self.compute_d1_flag:
            results["D1"] = torch.tensor(0.0, device=self.d1_bad.device)

        # Thresholds
        for thresh in self.thresholds:
            total = getattr(self, f"thres{int(thresh)}_total")
            if total > 0:
                bad = getattr(self, f"thres{int(thresh)}_bad")
                results[f"Thres{int(thresh)}px"] = bad.float() / total
            else:
                bad = getattr(self, f"thres{int(thresh)}_bad")
                results[f"Thres{int(thresh)}px"] = torch.tensor(0.0, device=bad.device)

        return results


class SemanticSegmentationMetrics(Metric):
    """
    Semantic segmentation metrics using TorchMetrics builtin functions.

    Computes: PA, MPA, mIoU, F1, Dice, per-class IoU and Accuracy
    Compatible with SemStereo's SegmentationMetric.

    Uses torchmetrics builtin metrics for reliability and efficiency.
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        # Use torchmetrics builtin metrics
        # Note: These are child modules, will be auto-registered
        self.pixel_accuracy = MulticlassAccuracy(
            num_classes=num_classes,
            average="micro",  # Overall accuracy
            ignore_index=ignore_index,
        )

        self.mean_pixel_accuracy = MulticlassAccuracy(
            num_classes=num_classes,
            average="macro",  # Mean per-class accuracy
            ignore_index=ignore_index,
        )

        self.per_class_accuracy = MulticlassAccuracy(
            num_classes=num_classes,
            average="none",  # Per-class accuracy
            ignore_index=ignore_index,
        )

        self.miou = MulticlassJaccardIndex(
            num_classes=num_classes,
            average="macro",  # Mean IoU
            ignore_index=ignore_index,
        )

        self.per_class_iou = MulticlassJaccardIndex(
            num_classes=num_classes,
            average="none",  # Per-class IoU
            ignore_index=ignore_index,
        )

        self.f1_score = MulticlassF1Score(
            num_classes=num_classes,
            average="macro",
            ignore_index=ignore_index,
        )

    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        Update all metrics.

        Args:
            preds: Prediction logits [B, C, H, W] or labels [B, H, W]
            target: Ground truth labels [B, H, W]
        """
        # Convert logits to labels if needed
        if preds.dim() == 4:
            preds = preds.argmax(dim=1)  # [B, H, W]

        # Update all metrics (they handle flattening internally)
        self.pixel_accuracy.update(preds, target)
        self.mean_pixel_accuracy.update(preds, target)
        self.per_class_accuracy.update(preds, target)
        self.miou.update(preds, target)
        self.per_class_iou.update(preds, target)
        self.f1_score.update(preds, target)

    def compute(self) -> dict[str, Tensor]:
        """
        Compute all segmentation metrics.

        Returns:
            Dictionary containing:
            - PA: Pixel Accuracy (overall accuracy)
            - MPA: Mean Pixel Accuracy (macro-averaged per-class accuracy)
            - mIoU: Mean Intersection over Union
            - F1: Mean F1 Score
            - Dice: Mean Dice Coefficient (same as F1 for multi-class)
            - IoU_per_class: IoU for each class
            - Acc_per_class: Accuracy for each class
        """
        pa = self.pixel_accuracy.compute()
        mpa = self.mean_pixel_accuracy.compute()
        miou = self.miou.compute()
        f1 = self.f1_score.compute()
        iou_per_class = self.per_class_iou.compute()
        acc_per_class = self.per_class_accuracy.compute()

        return {
            "PA": pa,
            "MPA": mpa,
            "mIoU": miou,
            "F1": f1,
            "Dice": f1,  # Dice = F1 for multi-class segmentation
            "IoU_per_class": iou_per_class,
            "Acc_per_class": acc_per_class,
        }


# =============================================================================
# Unified Metrics Class
# =============================================================================


class UnifiedSemStereoMetrics(Metric):
    """
    Unified metrics for SemStereo model combining stereo and segmentation metrics.

    This class provides a single interface to compute all metrics needed for
    evaluating the SemStereo model (or similar multi-task models).

    Example:
        >>> metrics = UnifiedSemStereoMetrics(
        ...     num_classes=6,
        ...     min_disp=-64,
        ...     max_disp=64,
        ... )
        >>> # During training/validation
        >>> metrics.update(
        ...     disp_pred=model_output['d_final'],
        ...     disp_target=batch['disparity'],
        ...     seg_pred=model_output['P_l'],
        ...     seg_target=batch['label'],
        ... )
        >>> # Get results
        >>> results = metrics.compute()
        >>> print(f"EPE: {results['stereo']['EPE']:.3f}")
        >>> print(f"mIoU: {results['seg']['mIoU']:.3f}")
    """

    def __init__(
        self,
        num_classes: int,
        min_disp: float = -128,
        max_disp: float = 128,
        compute_seg: bool = True,
        compute_stereo: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.compute_seg_flag = compute_seg
        self.compute_stereo_flag = compute_stereo

        # Initialize sub-metrics
        if compute_stereo:
            self.stereo_metrics = StereoMatchingMetrics(
                min_disp=min_disp,
                max_disp=max_disp,
                compute_d1=True,
                compute_thresholds=[1.0, 2.0],
            )

        if compute_seg:
            self.seg_metrics = SemanticSegmentationMetrics(
                num_classes=num_classes,
                ignore_index=None,
            )

    def update(
        self,
        disp_pred: Tensor | None = None,
        disp_target: Tensor | None = None,
        seg_pred: Tensor | None = None,
        seg_target: Tensor | None = None,
    ) -> None:
        """
        Update both stereo and segmentation metrics.

        Args:
            disp_pred: Predicted disparity [B, H, W] or [B, 1, H, W]
            disp_target: Ground truth disparity [B, H, W]
            seg_pred: Predicted segmentation logits [B, C, H, W]
            seg_target: Ground truth labels [B, H, W]
        """
        if self.compute_stereo_flag and disp_pred is not None and disp_target is not None:
            self.stereo_metrics.update(disp_pred, disp_target)

        if self.compute_seg_flag and seg_pred is not None and seg_target is not None:
            self.seg_metrics.update(seg_pred, seg_target)

    def compute(self) -> dict[str, dict[str, Tensor]]:
        """
        Compute all metrics.

        Returns:
            Nested dictionary:
            {
                'stereo': {'EPE': ..., 'D1': ..., 'Thres1px': ..., 'Thres2px': ...},
                'seg': {'PA': ..., 'MPA': ..., 'mIoU': ..., 'F1': ..., 'Dice': ...},
            }
        """
        results = {}

        if self.compute_stereo_flag:
            results["stereo"] = self.stereo_metrics.compute()

        if self.compute_seg_flag:
            results["seg"] = self.seg_metrics.compute()

        return results

    def reset(self) -> None:
        """Reset all metric states."""
        if self.compute_stereo_flag:
            self.stereo_metrics.reset()
        if self.compute_seg_flag:
            self.seg_metrics.reset()


# =============================================================================
# Utility functions for easy integration
# =============================================================================


def create_semstereo_metrics(
    num_classes: int = 6,
    min_disp: float = -64,
    max_disp: float = 64,
) -> UnifiedSemStereoMetrics:
    """
    Create metrics for US3D/WHU dataset (SemStereo default settings).

    Args:
        num_classes: Number of semantic classes (default: 6 for US3D)
        min_disp: Minimum disparity (default: -64 for US3D)
        max_disp: Maximum disparity (default: 64 for US3D)

    Returns:
        Configured UnifiedSemStereoMetrics instance
    """
    return UnifiedSemStereoMetrics(
        num_classes=num_classes,
        min_disp=min_disp,
        max_disp=max_disp,
        compute_seg=True,
        compute_stereo=True,
    )


if __name__ == "__main__":
    # Test the unified metrics
    print("Testing Unified SemStereo Metrics")
    print("=" * 60)

    # Create metrics
    metrics = create_semstereo_metrics(num_classes=6, min_disp=-64, max_disp=64)

    # Simulate predictions
    B, H, W = 2, 320, 320
    C = 6

    disp_pred = torch.randn(B, H, W) * 20  # Disparity in [-64, 64] range
    disp_target = torch.randn(B, H, W) * 20 + torch.randn(B, H, W) * 5  # GT with noise

    seg_pred = torch.randn(B, C, H, W)  # Logits
    seg_target = torch.randint(0, C, (B, H, W))  # Labels

    # Update metrics
    metrics.update(
        disp_pred=disp_pred,
        disp_target=disp_target,
        seg_pred=seg_pred,
        seg_target=seg_target,
    )

    # Compute results
    results = metrics.compute()

    print("\n📊 Stereo Matching Metrics:")
    print("-" * 60)
    for key, value in results["stereo"].items():
        if value.dim() == 0:  # Scalar
            print(f"  {key:<15}: {value.item():.4f}")

    print("\n🎨 Semantic Segmentation Metrics:")
    print("-" * 60)
    for key, value in results["seg"].items():
        if value.dim() == 0:  # Scalar
            print(f"  {key:<15}: {value.item():.4f}")
        elif key == "IoU_per_class":
            print(f"  {key:<15}: {[f'{v:.3f}' for v in value.tolist()]}")

    print("\n✅ All metrics computed successfully!")
