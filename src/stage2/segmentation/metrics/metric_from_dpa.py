"""
DPA-style segmentation metrics implemented using torchmetrics.

This module ports the metrics from DPA (Deep Parsing Network) to a torchmetrics-based
class for efficient and distributed-friendly computation.

Reference: src/stage2/segmentation/third_party/DPA/evaluate.py
"""

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.classification import MulticlassConfusionMatrix


class DPASegmentationScore(Metric):
    """
    DPA-style segmentation metrics based on confusion matrix.

    Metrics computed:
    - Overall Accuracy (OA): diag(CM).sum() / CM.sum()
    - Mean F1 (mF1): mean of per-class F1 scores
    - Mean IoU (mIoU): mean of per-class IoU scores
    - Users Accuracy (UA/Precision): TP / (TP + FP) per class
    - Producers Accuracy (PA/Recall): TP / (TP + FN) per class
    - Kappa: Cohen's Kappa coefficient

    Args:
        n_classes: Number of classes (excluding ignore_index if specified).
        ignore_index: Label to ignore in metric computation. If specified, pixels
            with this label will be excluded from the confusion matrix.
        per_class: If True, return per-class metrics for UA, PA, F1, IoU.
            Otherwise, return mean values.
    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(
        self,
        n_classes: int,
        ignore_index: int | None = None,
        per_class: bool = False,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.per_class = per_class

        # Use MulticlassConfusionMatrix as the underlying state
        self.confmat = MulticlassConfusionMatrix(
            num_classes=n_classes,
            ignore_index=ignore_index,
            normalize=None,  # We need raw counts
        )

    def _infer_device(self) -> torch.device:
        for buffer in self.confmat.buffers(recurse=True):
            return buffer.device
        for parameter in self.confmat.parameters(recurse=True):
            return parameter.device
        return torch.device("cpu")

    def _ensure_on_device(self, device: torch.device) -> None:
        if self._infer_device() != device:
            self.to(device)

    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        Update the confusion matrix with predictions and targets.

        Args:
            preds: Predicted labels, shape (N, H, W) or (N,)
            target: Ground truth labels, shape (N, H, W) or (N,)
        """
        if preds.device != target.device:
            target = target.to(device=preds.device)
        self._ensure_on_device(preds.device)

        preds = preds.long()
        target = target.long()
        self.confmat.update(preds.flatten(), target.flatten())

    def compute(self) -> dict[str, Tensor]:
        """
        Compute all metrics from the accumulated confusion matrix.

        Returns:
            Dictionary containing:
            - oa: Overall Accuracy
            - mf1: Mean F1 Score
            - miou: Mean IoU
            - ua: Users Accuracy (Precision), per-class or mean
            - pa: Producers Accuracy (Recall), per-class or mean
            - kappa: Cohen's Kappa
        """
        cm = self.confmat.compute().float()  # (n_classes, n_classes)

        # TP, FP, FN for each class
        tp = torch.diag(cm)
        fp = cm.sum(dim=0) - tp  # column sum - diag
        fn = cm.sum(dim=1) - tp  # row sum - diag

        # Overall Accuracy
        oa = tp.sum() / cm.sum().clamp(min=1e-8)

        # Users Accuracy (Precision): TP / (TP + FP)
        ua = tp / (tp + fp).clamp(min=1e-8)

        # Producers Accuracy (Recall): TP / (TP + FN)
        pa = tp / (tp + fn).clamp(min=1e-8)

        # F1 Score: 2*TP / (2*TP + FP + FN)
        f1 = 2 * tp / (2 * tp + fp + fn).clamp(min=1e-8)

        # IoU: TP / (TP + FP + FN)
        iou = tp / (tp + fp + fn).clamp(min=1e-8)

        # Kappa coefficient
        po = tp.sum() / cm.sum().clamp(min=1e-8)  # observed agreement
        pe = (cm.sum(dim=1) * cm.sum(dim=0)).sum() / (cm.sum() ** 2).clamp(min=1e-8)  # expected agreement
        kappa = (po - pe) / (1 - pe).clamp(min=1e-8)

        valid = (tp + fp + fn) > 0
        if valid.any():
            ua_mean = ua[valid].mean()
            pa_mean = pa[valid].mean()
            f1_mean = f1[valid].mean()
            iou_mean = iou[valid].mean()
        else:
            ua_mean = torch.zeros((), device=cm.device)
            pa_mean = torch.zeros((), device=cm.device)
            f1_mean = torch.zeros((), device=cm.device)
            iou_mean = torch.zeros((), device=cm.device)

        # Mean or per-class
        if self.per_class:
            return {
                "oa": oa,
                "mf1": f1_mean,
                "miou": iou_mean,
                "ua": ua,
                "pa": pa,
                "f1": f1,
                "iou": iou,
                "kappa": kappa,
            }
        else:
            return {
                "oa": oa,
                "mf1": f1_mean,
                "miou": iou_mean,
                "ua": ua_mean,
                "pa": pa_mean,
                "kappa": kappa,
            }

    def reset(self) -> None:
        """Reset the confusion matrix."""
        self.confmat.reset()


# * --- Test --- #


def test_dpa_metrics():
    """Test basic functionality of DPASegmentationScore with random data."""
    generator = torch.Generator().manual_seed(42)
    pred = torch.randint(0, 5, (2, 32, 32), generator=generator)
    gt = torch.randint(0, 5, (2, 32, 32), generator=torch.Generator().manual_seed(43))

    metrics = DPASegmentationScore(n_classes=5, ignore_index=None, per_class=False)
    metrics.update(pred, gt)
    results = metrics.compute()

    for name, value in results.items():
        assert isinstance(value, torch.Tensor), f"{name} should return a tensor"
        assert not torch.isnan(value).any(), f"{name} should not be NaN"
        if name != "kappa":
            assert (value >= 0).all(), f"{name} should be non-negative"
        if name in ["oa", "mf1", "miou", "ua", "pa"]:
            assert (value <= 1).all(), f"{name} should be <= 1"
        if name == "kappa":
            assert ((value >= -1) & (value <= 1)).all(), f"{name} should be between -1 and 1"

    print("✓ Basic test passed")
    print(f"  Results: {results}")


def test_perfect_prediction():
    """Test with perfect predictions (should give maximum scores)."""
    pred = torch.tensor([[[0, 1], [2, 3]], [[1, 2], [3, 0]]])
    gt = pred.clone()

    metrics = DPASegmentationScore(n_classes=4, ignore_index=None, per_class=False)
    metrics.update(pred, gt)
    results = metrics.compute()

    assert torch.isclose(results["oa"], torch.tensor(1.0)), "Perfect prediction should give OA=1"
    assert torch.isclose(results["mf1"], torch.tensor(1.0)), "Perfect prediction should give mF1=1"
    assert torch.isclose(results["miou"], torch.tensor(1.0)), "Perfect prediction should give mIoU=1"
    assert torch.isclose(results["kappa"], torch.tensor(1.0)), "Perfect prediction should give Kappa=1"

    print("✓ Perfect prediction test passed")


def test_ignore_index():
    """Test with ignore_index functionality."""
    pred = torch.tensor([[[0, 1, 2], [1, 2, 0]]])
    gt = torch.tensor([[[0, 1, 255], [1, 255, 0]]])  # 255 = ignore index

    metrics = DPASegmentationScore(n_classes=3, ignore_index=255, per_class=False)
    metrics.update(pred, gt)
    results = metrics.compute()

    assert isinstance(results["oa"], torch.Tensor), "Should return a tensor"
    assert not torch.isnan(results["oa"]).any(), "Should not be NaN"
    assert (results["oa"] >= 0).all() and (results["oa"] <= 1).all(), "OA should be between 0 and 1"

    print("✓ Ignore index test passed")


def test_per_class():
    """Test per_class=True returns per-class metrics."""
    pred = torch.randint(0, 3, (2, 8, 8))
    gt = torch.randint(0, 3, (2, 8, 8))

    metrics = DPASegmentationScore(n_classes=3, ignore_index=None, per_class=True)
    metrics.update(pred, gt)
    results = metrics.compute()

    assert results["ua"].shape == (3,), "UA should have shape (n_classes,)"
    assert results["pa"].shape == (3,), "PA should have shape (n_classes,)"
    assert results["f1"].shape == (3,), "F1 should have shape (n_classes,)"
    assert results["iou"].shape == (3,), "IoU should have shape (n_classes,)"

    print("✓ Per-class test passed")


def run_all_tests():
    """Run all test cases."""
    print("Running DPASegmentationScore tests...")

    test_dpa_metrics()
    test_perfect_prediction()
    test_ignore_index()
    test_per_class()

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    run_all_tests()
