"""
Anomaly Detection Metrics Module

This module provides comprehensive metrics for hyperspectral anomaly detection evaluation,
including various AUC-based metrics and other evaluation metrics commonly used in HAD.

The implementation follows the protocol defined in src.stage2.utilities.metrics.__init__.py
and leverages torchmetrics for efficient computation.
"""

from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from sklearn.metrics import roc_auc_score, roc_curve


class AnomalyDetectionMetrics(nn.Module):
    """
    Comprehensive metrics for hyperspectral anomaly detection.

    This class implements various evaluation metrics commonly used in hyperspectral anomaly detection,
    including multiple AUC variants and other performance indicators.

    Key features:
    - Multiple AUC variants (AUC1-AUC5 as defined in HyperSIGMA)
    - ROC curve computation
    - Batch processing support
    - Efficient accumulation using torchmetrics
    - Compatible with PyTorch training loops

    Reference:
        Based on the metrics used in HyperSIGMA:
        https://github.com/Z-Z-H/HyperSIGMA

    Args:
        compute_kwargs: Additional kwargs for metric computation
        prefix: Prefix for metric names (useful for multi-task scenarios)

    Example:
        >>> metrics = AnomalyDetectionMetrics()
        >>> for batch in dataloader:
        ...     pred, target = batch
        ...     metrics.update(pred, target)
        >>> results = metrics.compute()
        >>> print(results['auc1'])  # Main AUC score
    """

    def __init__(
        self, compute_kwargs: Optional[Dict[str, Any]] = None, prefix: str = ""
    ):
        super().__init__()
        self.compute_kwargs = compute_kwargs or {}
        self.prefix = prefix

        # Core metrics using torchmetrics
        self.auc = torchmetrics.AUROC(task="binary")

        # For accumulation of predictions and targets
        self.reset()

    def reset(self):
        """Reset all accumulated predictions and targets."""
        self.preds_list = []
        self.targets_list = []
        self.auc.reset()

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        anomaly_scores: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Update metrics with new predictions and targets.

        Args:
            preds: Model predictions [B, ...] or anomaly scores [B, H, W]
            target: Ground truth labels [B, ...] or [B, H, W]
            anomaly_scores: Optional explicit anomaly scores if preds are class probabilities
        """
        # Ensure tensors are on the same device
        device = (
            next(self.parameters()).device
            if len(list(self.parameters())) > 0
            else preds.device
        )
        preds = preds.to(device)
        target = target.to(device)

        # If preds are class probabilities (B, 2, H, W), extract anomaly scores
        if preds.dim() == 4 and preds.shape[1] == 2:
            # Assuming preds are [B, 2, H, W] with [background, anomaly] probabilities
            anomaly_scores = preds[:, 1, :, :]  # Take anomaly probability
        elif anomaly_scores is not None:
            anomaly_scores = anomaly_scores.to(device)
        else:
            # Assume preds are already anomaly scores
            anomaly_scores = preds

        # Flatten spatial dimensions for computation
        if anomaly_scores.dim() > 2:
            anomaly_scores_flat = anomaly_scores.flatten()
            target_flat = target.flatten()
        else:
            anomaly_scores_flat = anomaly_scores
            target_flat = target

        # Store for detailed AUC computation
        self.preds_list.append(anomaly_scores_flat.detach().cpu())
        self.targets_list.append(target_flat.detach().cpu())

        # Update torchmetrics AUC
        self.auc.update(anomaly_scores_flat, target_flat.long())

    def _compute_auc_variants(
        self, fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute various AUC variants as defined in HyperSIGMA.

        Args:
            fpr: False positive rates
            tpr: True positive rates
            thresholds: Decision thresholds

        Returns:
            Dictionary containing auc1-auc5 values
        """
        # Remove the first point (threshold = inf)
        fpr = fpr[1:]
        tpr = tpr[1:]
        thresholds = thresholds[1:]

        # AUC1: Standard ROC AUC
        auc1 = float(np.trapezoid(tpr, fpr))

        # AUC2: AUC of threshold vs FPR
        auc2 = float(np.trapezoid(fpr, thresholds))

        # AUC3: AUC of threshold vs TPR
        auc3 = float(np.trapezoid(tpr, thresholds))

        # AUC4: AUC1 + AUC3 - AUC2
        auc4 = auc1 + auc3 - auc2

        # AUC5: AUC3 / AUC2
        auc5 = auc3 / auc2 if auc2 != 0 else 0.0

        return {
            "auc1": round(auc1, 4),
            "auc2": round(auc2, 4),
            "auc3": round(auc3, 4),
            "auc4": round(auc4, 4),
            "auc5": round(auc5, 4),
        }

    def compute_additional_metrics(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute additional anomaly detection metrics.

        Args:
            preds: Anomaly scores
            target: Ground truth labels

        Returns:
            Dictionary of additional metrics
        """
        preds_np = preds.cpu().numpy()
        target_np = target.cpu().numpy()

        metrics_dict = {}

        # Precision-Recall AUC
        try:
            from sklearn.metrics import auc, precision_recall_curve

            precision, recall, _ = precision_recall_curve(target_np, preds_np)
            pr_auc = auc(recall, precision)
            metrics_dict["pr_auc"] = round(float(pr_auc), 4)
        except Exception:
            metrics_dict["pr_auc"] = 0.0

        # F1 score at optimal threshold
        try:
            fpr, tpr, thresholds = roc_curve(target_np, preds_np)
            # Find optimal threshold (closest to (0,1) on ROC curve)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]

            # Compute binary predictions at optimal threshold
            binary_pred = (preds_np >= optimal_threshold).astype(int)

            # F1 score
            from sklearn.metrics import f1_score

            f1 = f1_score(target_np, binary_pred, zero_division=0)
            metrics_dict["f1_score"] = round(float(f1), 4)
            metrics_dict["optimal_threshold"] = round(float(optimal_threshold), 4)
        except Exception:
            metrics_dict["f1_score"] = 0.0
            metrics_dict["optimal_threshold"] = 0.5

        # EER (Equal Error Rate)
        try:
            fpr, tpr, thresholds = roc_curve(target_np, preds_np)
            # Find threshold where FPR ≈ FNR (1 - TPR)
            fnr = 1 - tpr
            eer_idx = np.argmin(np.abs(fpr - fnr))
            eer = fpr[eer_idx]
            metrics_dict["eer"] = round(float(eer), 4)
        except Exception:
            metrics_dict["eer"] = 0.0

        return metrics_dict

    def compute(self) -> Dict[str, Any]:
        """
        Compute all accumulated metrics.

        Returns:
            Dictionary containing all computed metrics
        """
        if not self.preds_list or not self.targets_list:
            return {}

        # Concatenate all accumulated predictions and targets
        all_preds = torch.cat(self.preds_list, dim=0)
        all_targets = torch.cat(self.targets_list, dim=0)

        # Convert to numpy for detailed computations
        preds_np = all_preds.numpy()
        targets_np = all_targets.numpy()

        results = {}

        # Compute torchmetrics AUC
        torchmetrics_auc = self.auc.compute()
        results[f"{self.prefix}torchmetrics_auc"] = round(float(torchmetrics_auc), 4)

        # Compute ROC curve and AUC variants
        try:
            fpr, tpr, thresholds = roc_curve(targets_np, preds_np)

            # Compute AUC variants
            auc_variants = self._compute_auc_variants(fpr, tpr, thresholds)
            for key, value in auc_variants.items():
                results[f"{self.prefix}{key}"] = value

            # Store ROC curve for analysis
            results[f"{self.prefix}roc_curve"] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thresholds.tolist(),
            }

        except Exception as e:
            print(f"Error computing AUC variants: {e}")
            # Fallback to basic AUC
            try:
                basic_auc = roc_auc_score(targets_np, preds_np)
                results[f"{self.prefix}auc1"] = round(float(basic_auc), 4)
            except Exception:
                results[f"{self.prefix}auc1"] = 0.0

        # Compute additional metrics
        additional_metrics = self.compute_additional_metrics(all_preds, all_targets)
        for key, value in additional_metrics.items():
            results[f"{self.prefix}{key}"] = value

        # Add statistics
        results[f"{self.prefix}num_samples"] = len(all_preds)
        results[f"{self.prefix}num_anomalies"] = int(all_targets.sum().item())
        results[f"{self.prefix}anomaly_ratio"] = round(
            float(all_targets.mean().item()), 4
        )

        return results

    def __repr__(self) -> str:
        return f"AnomalyDetectionMetrics(prefix='{self.prefix}')"


class HADDetectionMetrics(nn.Module):
    """
    Specialized metrics for Hyperspectral Anomaly Detection following HyperSIGMA conventions.

    This is a convenience class that provides the exact metrics used in HyperSIGMA
    experiments, with the same naming conventions and computation methods.
    """

    def __init__(self):
        super().__init__()
        self.metrics = AnomalyDetectionMetrics(prefix="had_")

    def reset(self):
        """Reset accumulated metrics."""
        self.metrics.reset()

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update metrics with new predictions."""
        self.metrics.update(preds, target)

    def compute(self) -> Dict[str, Any]:
        """Compute all metrics, returning HyperSIGMA-compatible results."""
        results = self.metrics.compute()

        # Extract main results in HyperSIGMA format
        had_results = {
            "auc1": results.get("had_auc1", 0.0),
            "auc2": results.get("had_auc2", 0.0),
            "auc3": results.get("had_auc3", 0.0),
            "auc4": results.get("had_auc4", 0.0),
            "auc5": results.get("had_auc5", 0.0),
        }

        return had_results


def test_anomaly_detection_metrics():
    """Test function to demonstrate usage."""
    print("Testing AnomalyDetectionMetrics...")

    # Create sample data
    batch_size = 4
    height, width = 32, 32

    # Initialize metrics
    metrics = AnomalyDetectionMetrics(prefix="test_")

    # Simulate validation loop
    for i in range(3):
        # Create sample predictions and targets
        preds = torch.rand(batch_size, height, width)

        # Create some anomalies (binary targets)
        target = torch.zeros(batch_size, height, width)
        # Add random anomalies
        for b in range(batch_size):
            num_anomalies = torch.randint(1, 10, (1,)).item()
            for _ in range(num_anomalies):
                y, x = (
                    torch.randint(0, height, (1,)).item(),
                    torch.randint(0, width, (1,)).item(),
                )
                target[b, y, x] = 1

        print(f"\nBatch {i + 1}:")
        print(f"  Pred range: [{preds.min():.4f}, {preds.max():.4f}]")
        print(f"  Target sum: {target.sum().item()}")

        # Update metrics
        metrics.update(preds, target)

    # Compute final metrics
    results = metrics.compute()

    print("\nFinal Results:")
    for key, value in results.items():
        if not isinstance(value, dict):  # Skip nested dicts like roc_curve
            print(f"  {key}: {value}")

    # Test HAD-specific metrics
    print("\nTesting HADDetectionMetrics...")
    had_metrics = HADDetectionMetrics()

    # Use same data
    preds = torch.rand(batch_size, height, width)
    target = torch.zeros(batch_size, height, width)
    target[:, : height // 4, : width // 4] = 1  # Add some anomalies

    had_metrics.update(preds, target)
    had_results = had_metrics.compute()

    print("HAD Results:")
    for key, value in had_results.items():
        print(f"  {key}: {value}")

    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    test_anomaly_detection_metrics()
