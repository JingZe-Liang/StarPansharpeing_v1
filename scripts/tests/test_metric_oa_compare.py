"""
Compare OA (Overall Accuracy) between HyperSegmentationScore and sklearn accuracy_score,
including ignore_index=255 handling. Simulate gt and pred with noise.
"""

import numpy as np
import torch
from sklearn.metrics import accuracy_score

from src.stage2.segmentation.metrics.basic import HyperSegmentationScore


def simulate_data(n_classes: int, shape: tuple[int, int], ignore_index: int = 255, ignore_ratio: float = 0.1):
    gt = np.random.randint(0, n_classes, size=shape, dtype=np.int64)
    mask = np.random.rand(*shape) < ignore_ratio
    gt[mask] = ignore_index
    # pred: 先复制 gt，非 ignore_index 处加噪声
    pred = gt.copy()
    noise_mask = (np.random.rand(*shape) < 0.2) & (gt != ignore_index)
    pred[noise_mask] = np.random.randint(0, n_classes, size=noise_mask.sum())
    # pred 里所有 ignore_index 位置赋值为随机合法类别
    pred[gt == ignore_index] = np.random.randint(0, n_classes, size=(gt == ignore_index).sum())
    return torch.from_numpy(pred), torch.from_numpy(gt)


def compute_sklearn_oa(pred: torch.Tensor, gt: torch.Tensor, ignore_index: int = 255):
    pred_np = pred.numpy().flatten()
    gt_np = gt.numpy().flatten()
    valid = gt_np != ignore_index
    return accuracy_score(gt_np[valid], pred_np[valid])


def main():
    n_classes = 5
    ignore_index = 255
    shape = (128, 128)
    pred, gt = simulate_data(n_classes, shape, ignore_index, ignore_ratio=0.1)

    # micro OA 对比
    metric = HyperSegmentationScore(
        n_classes=n_classes,
        ignore_index=ignore_index,
        reduction="micro",
        per_class=False,
        include_bg=True,
    )
    print(f"pred max: {pred.max().item()}, gt max: {gt.max().item()}")
    metric.update(pred, gt)
    result = metric.compute()
    oa_metric = result["accuracy"].item() if hasattr(result["accuracy"], "item") else float(result["accuracy"])

    oa_sklearn = compute_sklearn_oa(pred, gt, ignore_index)

    print(f"HyperSegmentationScore OA: {oa_metric:.6f}")
    print(f"sklearn accuracy_score OA: {oa_sklearn:.6f}")
    print(f"是否相等: {np.isclose(oa_metric, oa_sklearn)}")

    # macro + per_class 测试
    print("\n===== Macro & Per-class metrics =====")
    metric_macro = HyperSegmentationScore(
        n_classes=n_classes,
        ignore_index=ignore_index,
        reduction="macro",
        per_class=True,
        include_bg=True,
    )
    metric_macro.update(pred, gt)
    result_macro = metric_macro.compute()
    for k, v in result_macro.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
