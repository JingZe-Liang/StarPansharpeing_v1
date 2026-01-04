"""
Anomaly Detection Metrics Module

This module provides comprehensive metrics for hyperspectral anomaly detection evaluation,
including various AUC-based metrics and other evaluation metrics commonly used in HAD.

The implementation follows the protocol defined in src.stage2.utilities.metrics.__init__.py
and leverages torchmetrics for efficient computation.
"""

from collections import namedtuple
from typing import Any, TypedDict, no_type_check

import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.utilities.metrics.aggregation import StackMeanMetrics

from ..utils.RX import RX

Pr_Auc_F1_EER_TypedDict = TypedDict(
    "Pr_Auc_F1_EER_TypedDict",
    {
        "pr_auc": float,
        "f1_score": float,
        "optimal_threshold": float,
        "eer": float,
    },
)
FPR_TPE_Threshold_TypedDict = TypedDict(
    "FPR_TPE_Threshold_TypedDict",
    {
        "fpr": np.ndarray,
        "tpr": np.ndarray,
        "thresholds": np.ndarray,
    },
)
AUC5_TypedDict = TypedDict(
    "AUC5_TypedDict",
    {
        "auc1": float,
        "auc2": float,
        "auc3": float,
        "auc4": float,
        "auc5": float,
    },
)


class AnomalyDetectionMetricsBase(nn.Module):
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
        self,
        compute_kwargs: dict[str, Any] | None = None,
        prefix: str = "",
        device: str | torch.device = "cuda",
    ):
        super().__init__()
        self.compute_kwargs = compute_kwargs or {}
        self.prefix = prefix
        self.device = torch.device(device)

        # Core metrics using torchmetrics
        self.auc = torchmetrics.AUROC(task="binary")
        self.pr_auc_f1_metrics = nn.ModuleDict(
            {
                "pr_auc": torchmetrics.MeanMetric(),
                "f1_score": torchmetrics.MeanMetric(),
                "optimal_threshold": torchmetrics.MeanMetric(),
                "eer": torchmetrics.MeanMetric(),
            }
        )
        # We don't need to store fpr, tpr, thresholds arrays since they vary in length
        # Instead, we'll compute AUC5 metrics directly in each batch
        self.auc5 = nn.ModuleDict({f"auc{i}": torchmetrics.MeanMetric() for i in range(1, 6)})

        # For accumulation of predictions and targets
        self.reset()

    def _update_any(self, metrics_module, **updates):
        for k, v in updates.items():
            metrics_module[k].update(v)

    def reset(self):
        """Reset all accumulated predictions and targets."""
        for metric in self.pr_auc_f1_metrics.values():
            metric.reset()
        for metric in self.auc5.values():
            metric.reset()
        self.auc.reset()

    def _to_tensor(self, x):
        return torch.as_tensor(x)

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        anomaly_scores: torch.Tensor | None = None,
    ) -> None:
        """
        Update metrics with new predictions and targets.

        Args:
            preds: Reconstructed images from model [B, C, H, W]
            target: Ground truth anomaly labels [B, H, W] (binary: 0=normal, 1=anomaly)
            anomaly_scores: Optional explicit anomaly scores [B, H, W], if None use RX on preds
        """
        # Ensure tensors are on the same device
        device = preds.device
        target = target.to(device)

        # If no explicit anomaly scores provided, compute using RX on reconstructed images
        if anomaly_scores is None:
            anomaly_scores = RX(preds)
        else:
            anomaly_scores = anomaly_scores.to(device)

        # Flatten spatial dimensions for computation
        anomaly_scores_flat = anomaly_scores.flatten()
        target_flat = target.flatten().to(torch.int32)

        # Check shapes consistency
        if anomaly_scores_flat.shape != target_flat.shape:
            raise ValueError(f"Shape mismatch after flattening: {anomaly_scores_flat.shape} vs {target_flat.shape}")

        # Metrics
        self.auc.update(anomaly_scores_flat, target_flat)
        target_flat = target_flat.cpu().numpy()
        anomaly_scores_flat = anomaly_scores_flat.cpu().numpy()
        f1_eer_metrics = self._compute_pr_auc_f1_score_eer_metrics(anomaly_scores_flat, target_flat)
        fpr, tpr, thresholds = roc_curve(target_flat, anomaly_scores_flat)
        auc5 = self._compute_auc5(fpr, tpr, thresholds)

        # Update
        self._update_any(self.pr_auc_f1_metrics, **f1_eer_metrics)
        self._update_any(self.auc5, **auc5)

    def _format_result(self):
        results = {}
        # Compute nested metrics and flatten them
        pr_metrics = self._compute_any(self.pr_auc_f1_metrics)
        for k, v in pr_metrics.items():
            results[f"{self.prefix}{k}"] = v.item() if isinstance(v, torch.Tensor) else v

        auc5_metrics = self._compute_any(self.auc5)
        for k, v in auc5_metrics.items():
            results[f"{self.prefix}{k}"] = v.item() if isinstance(v, torch.Tensor) else v

        results[f"{self.prefix}torchmetrics_auc"] = self.auc.compute().item()
        return results

    def _add_prefix(self, d: dict):
        d = {f"{self.prefix}_{k}": v for k, v in d.items()}
        return d

    @no_type_check
    def _update_torchmetrics_from_numpy_metrics(
        self,
        numpy_metrics: dict[str, np.ndarray | float] | Any,
        metrics: nn.ModuleDict,
    ):
        if hasattr(numpy_metrics, "_asdict"):
            numpy_metrics = numpy_metrics._asdict()

        for key, value in numpy_metrics.items():
            metrics[key].update(torch.as_tensor(value, device=self.device))

    def _compute_auc5(self, fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray) -> AUC5_TypedDict:
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

        return {"auc1": auc1, "auc2": auc2, "auc3": auc3, "auc4": auc4, "auc5": auc5}

    def _compute_pr_auc_f1_score_eer_metrics(
        self, preds_np: np.ndarray, target_np: np.ndarray
    ) -> Pr_Auc_F1_EER_TypedDict:
        """
        Compute additional anomaly detection metrics.

        Args:
            preds: Anomaly scores
            target: Ground truth labels

        Returns:
            Dictionary of additional metrics
        """

        # Precision-Recall AUC
        precision, recall, _ = precision_recall_curve(target_np, preds_np)
        pr_auc = auc(recall, precision)

        # F1 score at optimal threshold
        fpr, tpr, thresholds = roc_curve(target_np, preds_np)
        # Find optimal threshold (closest to (0,1) on ROC curve)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        # Compute binary predictions at optimal threshold
        binary_pred = (preds_np >= optimal_threshold).astype(int)

        # F1 score
        f1 = f1_score(target_np, binary_pred, zero_division=0)
        f1_score_val = round(float(f1), 4)
        optimal_threshold = round(float(optimal_threshold), 4)

        # EER (Equal Error Rate)
        fpr, tpr, thresholds = roc_curve(target_np, preds_np)
        # Find threshold where FPR ≈ FNR (1 - TPR)
        fnr = 1 - tpr
        eer_idx = np.argmin(np.abs(fpr - fnr))
        eer = fpr[eer_idx]

        # pr_auc, f1_score, optimal_threshold, eer
        # return metrics_dict
        return {
            "pr_auc": pr_auc,
            "f1_score": f1_score_val,
            "optimal_threshold": optimal_threshold,
            "eer": eer,
        }

    def _compute_any(self, metrics):
        if isinstance(metrics, torchmetrics.Metric):
            return metrics.compute()
        elif isinstance(metrics, nn.ModuleDict):
            return {k: v.compute() for k, v in metrics.items()}
        else:
            raise ValueError(f"Metrics type {type(metrics)} are not supported to compute")

    def compute(self) -> dict[str, Any]:
        """
        Compute all accumulated metrics.

        Returns:
            Dictionary containing all computed metrics
        """

        return self._format_result()

    def forward(self, all_preds, all_targets):
        self.update(all_preds, all_targets)
        return self._format_result()

    def __old_compute(self, all_preds, all_targets) -> dict[str, Any]:
        # Convert to numpy for detailed computations
        preds_np = all_preds.detach().cpu().numpy()
        targets_np = all_targets.detach().cpu().numpy()

        # Compute all metrics

        results = {}

        # Compute torchmetrics AUC
        torchmetrics_auc = self.auc.compute()
        results[f"{self.prefix}torchmetrics_auc"] = round(float(torchmetrics_auc), 4)

        # Compute ROC curve and AUC variants
        fpr, tpr, thresholds = roc_curve(targets_np, preds_np)

        # Compute AUC variants
        auc_variants = self._compute_auc5(fpr, tpr, thresholds)
        for key, value in auc_variants.items():
            results[f"{self.prefix}{key}"] = value

        # Store ROC curve for analysis
        results[f"{self.prefix}roc_curve"] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
        }

        # Compute additional metrics
        additional_metrics = self._compute_pr_auc_f1_score_eer_metrics(all_preds, all_targets)
        for key, value in additional_metrics.items():
            results[f"{self.prefix}{key}"] = value

        # Add statistics
        results[f"{self.prefix}num_samples"] = len(all_preds)
        results[f"{self.prefix}num_anomalies"] = int(all_targets.sum().item())
        results[f"{self.prefix}anomaly_ratio"] = round(float(all_targets.mean().item()), 4)

        return results

    def to(self, device, **kwargs):
        self.auc.to(device)

    def __repr__(self) -> str:
        return f"AnomalyDetectionMetrics(prefix='{self.prefix}')"


class HADDetectionMetrics(nn.Module):
    """
    Specialized metrics for Hyperspectral Anomaly Detection following HyperSIGMA conventions.

    This is a convenience class that provides the exact metrics used in HyperSIGMA
    experiments, with the same naming conventions and computation methods.
    """

    def __init__(self, device: str | torch.device = "cuda"):
        """
        Initialize HADDetectionMetrics.

        Args:
            device: Device to run computations on
        """
        super().__init__()
        self.metrics = AnomalyDetectionMetricsBase(prefix="had_", device=device)

    def reset(self) -> None:
        """Reset accumulated metrics."""
        self.metrics.reset()

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update metrics with new predictions.

        Args:
            preds: Reconstructed images from model
            target: Ground truth anomaly labels
        """
        self.metrics.update(preds, target)

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
        """
        Forward method for PyTorch compatibility.

        Args:
            preds: Predictions
            target: Ground truth labels

        Returns:
            Computed metrics
        """
        self.update(preds, target)
        return self.compute()

    def compute(self) -> dict[str, float]:
        """
        Compute all metrics, returning HyperSIGMA-compatible results.

        Returns:
            Dictionary containing AUC1-AUC5 metrics
        """
        results = self.metrics.compute()

        # Extract main results in HyperSIGMA format
        # Extract values from results, handling both tensor and float types
        def _extract_value(key, default=0.0):
            val = results.get(key, default)
            return val.item() if isinstance(val, torch.Tensor) else val

        had_results = {
            "auc1": _extract_value("had_auc1"),
            "auc2": _extract_value("had_auc2"),
            "auc3": _extract_value("had_auc3"),
            "auc4": _extract_value("had_auc4"),
            "auc5": _extract_value("had_auc5"),
        }

        return had_results

    def __repr__(self) -> str:
        """Return string representation."""
        return f"HADDetectionMetrics(device='{self.metrics.device}')"


def test_anomaly_detection_metrics() -> None:
    """
    Test function to demonstrate usage of anomaly detection metrics.

    This function tests both AnomalyDetectionMetricsbase and HADDetectionMetrics
    with synthetic data to verify functionality.
    """
    print("=" * 60)
    print("Testing Anomaly Detection Metrics")
    print("=" * 60)

    # Create sample data
    batch_size = 4
    channels = 224
    height, width = 32, 32

    # Initialize metrics with device specification
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    metrics = AnomalyDetectionMetricsBase(prefix="test_", device=device)

    print("\n" + "-" * 40)
    print("Testing AnomalyDetectionMetricsbase")
    print("-" * 40)

    # Simulate validation loop
    total_samples = 0
    total_anomalies = 0

    for i in range(3):
        # Create sample predictions and targets
        preds = torch.rand(batch_size, channels, height, width, device=device)

        # Create some anomalies (binary targets)
        target = torch.zeros(batch_size, height, width, device=device)
        batch_anomalies = 0

        # Add random anomalies
        for b in range(batch_size):
            num_anomalies: int = torch.randint(1, 10, (1,)).item()  # type: ignore
            batch_anomalies += num_anomalies
            for _ in range(num_anomalies):
                y, x = (
                    torch.randint(0, height, (1,)).item(),
                    torch.randint(0, width, (1,)).item(),
                )
                target[b, y, x] = 1

        total_samples += batch_size * height * width
        total_anomalies += batch_anomalies

        print(f"\nBatch {i + 1}:")
        print(f"  Pred range: [{preds.min():.4f}, {preds.max():.4f}]")
        print(f"  Target anomalies: {batch_anomalies}")
        print(f"  Anomaly ratio: {batch_anomalies / (batch_size * height * width):.4f}")

        # Update metrics
        metrics.update(preds, target)

    # Compute final metrics
    results = metrics.compute()

    print("\n" + "-" * 40)
    print("AnomalyDetectionMetricsbase Results")
    print("-" * 40)

    for key, value in results.items():
        if not isinstance(value, dict):  # Skip nested dicts like roc_curve
            print(f"  {key}: {value}")

    print(f"\nDataset statistics:")
    print(f"  Total samples: {total_samples}")
    print(f"  Total anomalies: {total_anomalies}")
    print(f"  Overall anomaly ratio: {total_anomalies / total_samples:.4f}")

    # Test HAD-specific metrics
    print("\n" + "-" * 40)
    print("Testing HADDetectionMetrics")
    print("-" * 40)

    had_metrics = HADDetectionMetrics(device=device)

    # Create test data with known anomaly pattern
    preds = torch.rand(batch_size, channels, height, width, device=device)
    target = torch.zeros(batch_size, height, width, device=device)

    # Add structured anomalies (top-left quadrant)
    target[:, : height // 4, : width // 4] = 1
    num_anomalies = batch_size * (height // 4) * (width // 4)

    print(f"\nHAD Test Data:")
    print(f"  Pred range: [{preds.min():.4f}, {preds.max():.4f}]")
    print(f"  Target anomalies: {num_anomalies}")
    print(f"  Anomaly ratio: {num_anomalies / (batch_size * height * width):.4f}")

    had_metrics.update(preds, target)
    had_results = had_metrics.compute()

    print("\nHAD Results:")
    for key, value in had_results.items():
        print(f"  {key}: {value}")

    # Test multiple updates for HAD metrics
    print("\n" + "-" * 40)
    print("Testing HADDetectionMetrics with multiple updates")
    print("-" * 40)

    had_metrics.reset()

    for i in range(2):
        # Create different anomaly patterns
        preds = torch.rand(batch_size, channels, height, width, device=device)
        target = torch.zeros(batch_size, height, width, device=device)

        if i == 0:
            # Center anomalies
            h_start, h_end = height // 3, 2 * height // 3
            w_start, w_end = width // 3, 2 * width // 3
            target[:, h_start:h_end, w_start:w_end] = 1
        else:
            # Corner anomalies
            target[:, : height // 6, : width // 6] = 1
            target[:, -height // 6 :, -width // 6 :] = 1

        print(f"\nUpdate {i + 1}:")
        print(f"  Anomaly pattern: {'center' if i == 0 else 'corners'}")
        print(f"  Anomaly count: {target.sum().item()}")

        had_metrics.update(preds, target)

    final_had_results = had_metrics.compute()
    print("\nFinal HAD Results (after multiple updates):")
    for key, value in final_had_results.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_anomaly_detection_metrics()
