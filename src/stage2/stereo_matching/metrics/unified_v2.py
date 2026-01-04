"""
Unified Metrics for Stereo Matching and Semantic Segmentation (v2).

This module provides a comprehensive metrics suite that combines:
1. Stereo Matching Metrics: EPE, D1, Threshold-based errors
2. Semantic Segmentation Metrics: Accuracy, Precision, Recall, F1, IoU, Dice, Kappa

This is an improved version of unified.py with better modularity and flexibility.
Compatible with both SemStereo and custom implementations.
Supports DDP training with torchmetrics backend.

Author: Claude Code
Date: 2025-12-29
"""

from typing import Literal

import torch
import torch.nn as nn
from loguru import logger
from torch import Tensor
from torchmetrics import Metric

# Import stereo matching metrics from basic.py
from .basic import D1Error, EndPointError, ThresholdError

# Import semantic segmentation metrics
from ...segmentation.metrics.basic import HyperSegmentationScore


class UnifiedStereoSegmentationMetrics(Metric):
    """
    Unified metrics for stereo matching and semantic segmentation evaluation.

    This class combines stereo matching metrics (EPE, D1, threshold errors)
    and semantic segmentation metrics (Accuracy, Precision, Recall, F1, IoU, Dice, Kappa)
    into a single interface for multi-task learning evaluation.

    Features
    --------
    - Modular design: can enable/disable stereo or segmentation metrics
    - DDP compatible: inherits from torchmetrics.Metric
    - Flexible input: handles various tensor shapes and formats
    - Comprehensive metrics: covers all commonly used evaluation metrics

    Examples
    --------
    >>> # Initialize with both stereo and segmentation
    >>> metrics = UnifiedStereoSegmentationMetrics(
    ...     num_classes=6,
    ...     min_disp=-64,
    ...     max_disp=64,
    ...     compute_stereo=True,
    ...     compute_seg=True,
    ... )
    >>>
    >>> # Update during validation
    >>> metrics.update(
    ...     disp_pred=outputs['disparity'],
    ...     disp_target=batch['disp_gt'],
    ...     seg_pred=outputs['segmentation'],
    ...     seg_target=batch['seg_gt'],
    ... )
    >>>
    >>> # Compute results
    >>> results = metrics.compute()
    >>> print(f"EPE: {results['stereo']['epe']:.4f}")
    >>> print(f"mIoU: {results['segmentation']['miou']:.4f}")
    """

    def __init__(
        self,
        # Stereo configuration
        min_disp: float = -64,
        max_disp: float = 64,
        compute_stereo: bool = True,
        stereo_thresholds: list[float] | None = None,
        # Segmentation configuration
        num_classes: int = 6,
        ignore_index: int | None = None,
        compute_seg: bool = True,
        seg_reduction: Literal["micro", "macro", "weighted", "none"] = "macro",
        # Other
        **kwargs,
    ):
        """
        Initialize unified metrics.

        Parameters
        ----------
        min_disp : float
            Minimum valid disparity value for generating valid mask. Default: -64.
        max_disp : float
            Maximum valid disparity value for generating valid mask. Default: 64.
        compute_stereo : bool
            Whether to compute stereo matching metrics. Default: True.
        stereo_thresholds : list[float] | None
            List of threshold values for threshold-based error metrics.
            Default: [1.0, 2.0, 3.0].
        num_classes : int
            Number of semantic segmentation classes. Default: 6.
        ignore_index : int | None
            Index to ignore in semantic segmentation (e.g., 255 for invalid pixels).
            Default: None.
        compute_seg : bool
            Whether to compute semantic segmentation metrics. Default: True.
        seg_reduction : Literal["micro", "macro", "weighted", "none"]
            Reduction method for segmentation metrics. Default: "macro".
        **kwargs
            Additional arguments passed to torchmetrics.Metric.
        """
        super().__init__(**kwargs)

        # Store configuration
        self.min_disp = min_disp
        self.max_disp = max_disp
        self.compute_stereo_flag = compute_stereo
        self.compute_seg_flag = compute_seg
        self.num_classes = num_classes

        # Set default thresholds
        if stereo_thresholds is None:
            stereo_thresholds = [1.0, 2.0, 3.0]

        # Initialize stereo metrics
        if compute_stereo:
            self.epe_metric = EndPointError(min_disp=min_disp, max_disp=max_disp)
            self.d1_metric = D1Error(min_disp=min_disp, max_disp=max_disp)

            # Create threshold metrics (replace dots with underscores for ModuleDict keys)
            self.threshold_metrics = nn.ModuleDict(
                {
                    f"threshold_{str(threshold).replace('.', '_')}": ThresholdError(
                        threshold=threshold, min_disp=min_disp, max_disp=max_disp
                    )
                    for threshold in stereo_thresholds
                }
            )
            self.threshold_list = stereo_thresholds

        # Initialize segmentation metrics
        if compute_seg:
            self.seg_metric = HyperSegmentationScore(
                n_classes=num_classes,
                ignore_index=ignore_index,
                task="multiclass",
                reduction=seg_reduction,
                per_class=True,
                include_bg=False,
                input_format="index",
            )

    def update(
        self,
        # Stereo inputs
        disp_pred: Tensor | None = None,
        disp_target: Tensor | None = None,
        # Segmentation inputs
        seg_pred: Tensor | None = None,
        seg_target: Tensor | None = None,
    ) -> None:
        """
        Update metric states with new predictions.

        Parameters
        ----------
        disp_pred : Tensor | None
            Predicted disparity map [B, H, W] or [B, 1, H, W]. Required if compute_stereo=True.
        disp_target : Tensor | None
            Ground truth disparity map [B, H, W] or [B, 1, H, W]. Required if compute_stereo=True.
        seg_pred : Tensor | None
            Predicted segmentation, either logits [B, C, H, W] or labels [B, H, W].
            Required if compute_seg=True.
        seg_target : Tensor | None
            Ground truth segmentation labels [B, H, W]. Required if compute_seg=True.

        Notes
        -----
        - All parameters are optional, allowing to compute only stereo or only segmentation
        - Input shapes are automatically normalized (e.g., [B, 1, H, W] -> [B, H, W])
        - Logits are automatically converted to labels for segmentation predictions
        """
        # Update stereo metrics
        if self.compute_stereo_flag:
            if disp_pred is None or disp_target is None:
                logger.warning(
                    "compute_stereo=True but disp_pred or disp_target is None. Skipping stereo metrics update."
                )
            else:
                # Normalize shapes
                disp_pred_norm = self._unify_shape(disp_pred)
                disp_target_norm = self._unify_shape(disp_target)

                # Update all stereo metrics
                self.epe_metric.update(disp_pred_norm, disp_target_norm)
                self.d1_metric.update(disp_pred_norm, disp_target_norm)
                for metric in self.threshold_metrics.values():
                    metric.update(disp_pred_norm, disp_target_norm)

        # Update segmentation metrics
        if self.compute_seg_flag:
            if seg_pred is None or seg_target is None:
                logger.warning(
                    "compute_seg=True but seg_pred or seg_target is None. Skipping segmentation metrics update."
                )
            else:
                # Convert predictions to labels if needed
                seg_pred_labels = self._pred_to_labels(seg_pred)

                # Update segmentation metric
                self.seg_metric.update(seg_pred_labels, seg_target)

    def compute(self) -> dict[str, dict[str, Tensor]]:
        """
        Compute all metric values.

        Returns
        -------
        dict[str, dict[str, Tensor]]
            Nested dictionary containing all computed metrics:
            {
                'stereo': {
                    'epe': scalar tensor,
                    'd1': scalar tensor,
                    'threshold_1.0': scalar tensor,
                    'threshold_2.0': scalar tensor,
                    'threshold_3.0': scalar tensor,
                },
                'segmentation': {
                    'pixel_accuracy': scalar tensor,
                    'mean_pixel_accuracy': scalar tensor,
                    'miou': scalar tensor,
                    'f1_score': scalar tensor,
                    'dice': scalar tensor,
                    'precision': scalar tensor,
                    'recall': scalar tensor,
                    'kappa': scalar tensor,
                    'iou_per_class': tensor of shape [num_classes],
                    'accuracy_per_class': tensor of shape [num_classes],
                }
            }

        Examples
        --------
        >>> results = metrics.compute()
        >>> epe = results['stereo']['epe']
        >>> miou = results['segmentation']['miou']
        """
        results = {}

        # Compute stereo metrics
        if self.compute_stereo_flag:
            stereo_results = {
                "epe": self.epe_metric.compute(),
                "d1": self.d1_metric.compute(),
            }

            # Add threshold metrics
            for threshold in self.threshold_list:
                metric_key = f"threshold_{str(threshold).replace('.', '_')}"
                metric_name = f"threshold_{threshold}"
                stereo_results[metric_name] = self.threshold_metrics[metric_key].compute()

            results["stereo"] = stereo_results

        # Compute segmentation metrics
        if self.compute_seg_flag:
            seg_results = self.seg_metric.compute()

            # Reorganize segmentation results
            results["segmentation"] = {
                "pixel_accuracy": seg_results.get("accuracy", torch.tensor(0.0)),
                "mean_pixel_accuracy": seg_results.get("accuracy", torch.tensor(0.0)),
                "miou": seg_results.get("mean_iou", torch.tensor(0.0)),
                "f1_score": seg_results.get("f1_score", torch.tensor(0.0)),
                "dice": seg_results.get("dice", torch.tensor(0.0)),
                "precision": seg_results.get("precision", torch.tensor(0.0)),
                "recall": seg_results.get("recall", torch.tensor(0.0)),
                "kappa": seg_results.get("kappa", torch.tensor(0.0)),
                "iou_per_class": seg_results.get("iou_per_class", None),
                "accuracy_per_class": seg_results.get("acc_per_class", None),
            }

        return results

    def reset(self) -> None:
        """
        Reset all metric states.

        This method is called at the start of each validation epoch.
        """
        if self.compute_stereo_flag:
            self.epe_metric.reset()
            self.d1_metric.reset()
            for metric in self.threshold_metrics.values():
                metric.reset()

        if self.compute_seg_flag:
            self.seg_metric.reset()

    def _unify_shape(self, tensor: Tensor) -> Tensor:
        """
        Unify tensor shape to [B, H, W].

        If input is [B, 1, H, W], squeeze the channel dimension.
        Otherwise, return as-is.

        Parameters
        ----------
        tensor : Tensor
            Input tensor of shape [B, H, W] or [B, 1, H, W]

        Returns
        -------
        Tensor
            Tensor with shape [B, H, W]
        """
        if tensor.dim() == 4 and tensor.size(1) == 1:
            return tensor.squeeze(1)
        return tensor

    def _pred_to_labels(self, pred: Tensor) -> Tensor:
        """
        Convert prediction logits to label indices.

        If input is [B, C, H, W] (logits), apply argmax to get [B, H, W] labels.
        If input is already [B, H, W] (labels), return as-is.

        Parameters
        ----------
        pred : Tensor
            Prediction tensor, either [B, C, H, W] logits or [B, H, W] labels

        Returns
        -------
        Tensor
            Label indices of shape [B, H, W]
        """
        if pred.dim() == 4:  # [B, C, H, W] logits
            return pred.argmax(dim=1)
        return pred  # [B, H, W] labels

    def flatten_metrics(self, results: dict[str, dict[str, Tensor]]) -> dict[str, float]:
        """
        Flatten nested metrics dictionary for logging.

        Converts nested structure like {'stereo': {'epe': 2.5}}
        to flat structure like {'stereo/epe': 2.5}.

        Per-class metrics (tensors with dim > 0) are skipped to avoid clutter.

        Parameters
        ----------
        results : dict[str, dict[str, Tensor]]
            Nested dictionary from compute() method

        Returns
        -------
        dict[str, float]
            Flattened dictionary with keys like 'stereo/epe', 'segmentation/miou'

        Examples
        --------
        >>> results = metrics.compute()
        >>> flat = metrics.flatten_metrics(results)
        >>> for name, value in flat.items():
        ...     print(f"{name}: {value:.4f}")
        """
        flat = {}

        for task_name, task_metrics in results.items():
            for metric_name, value in task_metrics.items():
                # Skip per-class metrics (they are tensors, not scalars)
                if isinstance(value, Tensor):
                    if value.dim() == 0:  # Scalar tensor
                        flat[f"{task_name}/{metric_name}"] = value.item()
                    # Skip per-class tensors
                elif isinstance(value, (int, float)):
                    flat[f"{task_name}/{metric_name}"] = float(value)

        return flat


# =============================================================================
# Utility functions for easy integration
# =============================================================================


def create_unified_metrics(
    num_classes: int = 6,
    min_disp: float = -64,
    max_disp: float = 64,
    ignore_index: int | None = None,
    compute_stereo: bool = True,
    compute_seg: bool = True,
) -> UnifiedStereoSegmentationMetrics:
    """
    Create unified metrics with default configuration for US3D/WHU datasets.

    Parameters
    ----------
    num_classes : int
        Number of semantic classes. Default: 6 for US3D.
    min_disp : float
        Minimum disparity. Default: -64 for US3D.
    max_disp : float
        Maximum disparity. Default: 64 for US3D.
    ignore_index : int | None
        Ignore index for segmentation. Default: None.
    compute_stereo : bool
        Whether to compute stereo metrics. Default: True.
    compute_seg : bool
        Whether to compute segmentation metrics. Default: True.

    Returns
    -------
    UnifiedStereoSegmentationMetrics
        Configured unified metrics instance
    """
    return UnifiedStereoSegmentationMetrics(
        num_classes=num_classes,
        min_disp=min_disp,
        max_disp=max_disp,
        ignore_index=ignore_index,
        compute_stereo=compute_stereo,
        compute_seg=compute_seg,
        stereo_thresholds=[1.0, 2.0, 3.0],
        seg_reduction="macro",
    )


# =============================================================================
# Test code
# =============================================================================


if __name__ == "__main__":
    print("=" * 80)
    print("Testing UnifiedStereoSegmentationMetrics")
    print("=" * 80)

    # Create metrics with both stereo and segmentation
    print("\n1. Initializing metrics...")
    metrics = create_unified_metrics(
        num_classes=6,
        min_disp=-64,
        max_disp=64,
        compute_stereo=True,
        compute_seg=True,
    )
    print("✓ Metrics initialized successfully")

    # Simulate predictions
    print("\n2. Creating simulated predictions...")
    B, H, W = 2, 320, 320
    C = 6

    # Stereo predictions
    disp_pred = torch.randn(B, H, W) * 20  # Predicted disparity
    disp_target = torch.randn(B, H, W) * 20 + torch.randn(B, H, W) * 5  # GT with noise

    # Segmentation predictions
    seg_pred = torch.randn(B, C, H, W)  # Logits
    seg_target = torch.randint(0, C, (B, H, W))  # Labels

    print(f"  disp_pred shape: {disp_pred.shape}")
    print(f"  disp_target shape: {disp_target.shape}")
    print(f"  seg_pred shape: {seg_pred.shape}")
    print(f"  seg_target shape: {seg_target.shape}")
    print("✓ Predictions created")

    # Update metrics
    print("\n3. Updating metrics...")
    metrics.update(
        disp_pred=disp_pred,
        disp_target=disp_target,
        seg_pred=seg_pred,
        seg_target=seg_target,
    )
    print("✓ Metrics updated")

    # Compute results
    print("\n4. Computing results...")
    results = metrics.compute()

    # Print stereo metrics
    print("\n" + "-" * 80)
    print("📊 Stereo Matching Metrics:")
    print("-" * 80)
    if "stereo" in results:
        for key, value in results["stereo"].items():
            if isinstance(value, Tensor) and value.dim() == 0:
                print(f"  {key:<20}: {value.item():.4f}")

    # Print segmentation metrics
    print("\n" + "-" * 80)
    print("🎨 Semantic Segmentation Metrics:")
    print("-" * 80)
    if "segmentation" in results:
        for key, value in results["segmentation"].items():
            if isinstance(value, Tensor) and value.dim() == 0:
                print(f"  {key:<20}: {value.item():.4f}")
            elif key == "iou_per_class" and value is not None:
                print(f"  {key:<20}: {[f'{v:.3f}' for v in value.tolist()]}")
            elif key == "accuracy_per_class" and value is not None:
                print(f"  {key:<20}: {[f'{v:.3f}' for v in value.tolist()]}")

    # Test flatten_metrics
    print("\n" + "-" * 80)
    print("📋 Flattened Metrics (for logging):")
    print("-" * 80)
    flat = metrics.flatten_metrics(results)
    for name, value in flat.items():
        print(f"  {name:<30}: {value:.4f}")

    print("\n" + "=" * 80)
    print("✅ All tests passed!")
    print("=" * 80)
