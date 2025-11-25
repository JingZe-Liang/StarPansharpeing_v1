"""
Patch utilities for hyperspectral images.

Provides functions to cut an HSI cube into sliding-window patches (stride=1 with zero padding)
and to reconstruct a full (H, W) label map from flat patch-wise predictions.

This is a small, self-contained reimplementation adapted from the project's
`model/split_data.py:create_patches` and `create_patches_inference` logic.

Functions
- create_patches(X, y, window_size, remove_zero_labels)
- create_patches_inference(X, window_size)
- reconstruct_from_flat_predictions(preds_flat, height, width)

All functions use numpy arrays.

NOTE:
This following patching methods are really memory-hungry.
This can be mitigated by using Python generator, but I choose to use semantic method.
"""

from dataclasses import asdict, dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy.io import loadmat


@dataclass
class HyperClassificationConfig:
    window_size: int = 64
    marginal: int = 2  # (win-1)//2
    remove_zero_labels: bool = True
    stride: int = 1


def pad_with_zeros(X: np.ndarray, margin: int) -> np.ndarray:
    """Pad spatial dimensions with zeros.

    Args:
        X: array with shape (H, W, C)
        margin: padding width on each side

    Returns:
        padded array with shape (H + 2*margin, W + 2*margin, C)
    """
    new_shape = (X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2])
    newX = np.zeros(new_shape, dtype=X.dtype)
    x_offset = margin
    y_offset = margin
    newX[x_offset : x_offset + X.shape[0], y_offset : y_offset + X.shape[1], :] = X
    return newX


def create_patches(
    X: np.ndarray,
    y: np.ndarray,
    window_size: int = 64,
    remove_zero_labels: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding-window patches for every pixel in X and corresponding labels.

    Behavior matches the repository's original utility: stride=1, zero-padding at
    borders so every pixel has a centered patch.

    Args:
        X: input cube, shape (H, W, bands)
        y: label map, shape (H, W) (integer labels, 0 can indicate unlabeled)
        window_size: spatial size of patch (odd recommended)
        remove_zero_labels: if True, remove patches whose center label == 0

    Returns:
        patches: numpy array shape (N, window_size, window_size, bands)
        labels: numpy array shape (N,) with integer labels
    """
    assert X.ndim == 3, "X must be HxWxC"
    assert y.ndim == 2, "y must be HxW"
    assert X.shape[0] == y.shape[0] and X.shape[1] == y.shape[1]

    margin = int((window_size - 1) / 2)
    padded = pad_with_zeros(X, margin=margin)

    H, W = X.shape[0], X.shape[1]
    patches = np.zeros((H * W, window_size, window_size, X.shape[2]), dtype=X.dtype)
    labels = np.zeros((H * W,), dtype=y.dtype)

    patch_index = 0
    for r in range(margin, padded.shape[0] - margin):
        for c in range(margin, padded.shape[1] - margin):
            patch = padded[r - margin : r + margin + 1, c - margin : c + margin + 1]
            patches[patch_index] = patch
            labels[patch_index] = y[r - margin, c - margin]
            patch_index += 1

    if remove_zero_labels:
        mask = labels > 0
        patches = patches[mask]
        labels = labels[mask]

    return patches, labels


def create_patches_inference(X: np.ndarray, window_size: int = 5) -> Tuple[np.ndarray, int, int]:
    """Create patches for whole-image inference.

    Returns patches in row-major order and original height/width so caller can
    reconstruct the image.

    Args:
        X: input cube (H, W, bands)
        window_size: patch size

    Returns:
        patches: (H*W, window_size, window_size, bands)
        H: original height
        W: original width
    """
    assert X.ndim == 3, "X must be HxWxC"
    margin = int((window_size - 1) / 2)
    padded = pad_with_zeros(X, margin=margin)
    H, W = X.shape[0], X.shape[1]

    patches = np.zeros((H * W, window_size, window_size, X.shape[2]), dtype=X.dtype)
    patch_index = 0
    for r in range(margin, padded.shape[0] - margin):
        for c in range(margin, padded.shape[1] - margin):
            patch = padded[r - margin : r + margin + 1, c - margin : c + margin + 1]
            patches[patch_index] = patch
            patch_index += 1

    return patches, H, W


def reconstruct_from_flat_predictions(preds_flat: np.ndarray, height: int, width: int) -> np.ndarray:
    """Reconstruct a 2D label map from flat predictions in row-major order.

    Args:
        preds_flat: 1D array with length == height*width
        height: original image height
        width: original image width

    Returns:
        label_map: (height, width)
    """
    assert preds_flat.size == height * width, "preds_flat length must equal H*W"
    return preds_flat.reshape((height, width))


# * --- Test --- #


def test_patching_recon(cfg: HyperClassificationConfig):
    from src.utilities.io import read_image

    mat_file = "data/Downstreams/ClassificationCollection/cls/mat/PaviaU.mat"
    gt_file = "data/Downstreams/ClassificationCollection/cls/cls_GT/PaviaU_gt.mat"

    def _load_mat_image_and_gt(mat_path_img: str, mat_path_gt: str):
        try:
            m_img = read_image(mat_path_img)
            m_gt = read_image(mat_path_gt)
        except Exception as e:
            print(f"Failed to load .mat files: {e}")
            return None, None

        # find first 3D array in image mat and first 2D array in gt mat
        img = m_img
        gt = m_gt
        # for k, v in m_img.items():
        #     if isinstance(v, np.ndarray) and v.ndim == 3:
        #         img = v
        #         break
        # for k, v in m_gt.items():
        #     if isinstance(v, np.ndarray) and v.ndim == 2:
        #         gt = v
        #         break

        return img, gt

    X, y = _load_mat_image_and_gt(mat_file, gt_file)

    # fallback to small synthetic example if loading failed or shapes invalid
    if X is None or y is None:
        print("Could not find valid image/gt in provided .mat files; using synthetic example.")
        H, W, C = 3, 4, 2
        X = np.arange(H * W * C, dtype=np.int32).reshape(H, W, C)
        y = np.array(
            [
                [1, 2, 0, 1],
                [2, 1, 1, 0],
                [1, 0, 2, 2],
            ],
            dtype=np.int32,
        )
    else:
        # ensure image is HxWxC and gt is HxW
        if X.ndim == 3 and y.ndim == 2 and X.shape[0] == y.shape[0] and X.shape[1] == y.shape[1]:
            print(f"Loaded image shape: {X.shape}, gt shape: {y.shape}")
        else:
            print("Loaded mats but shapes are incompatible; using synthetic example.")
            H, W, C = 3, 4, 2
            X = np.arange(H * W * C, dtype=np.int32).reshape(H, W, C)
            y = np.array(
                [
                    [1, 2, 0, 1],
                    [2, 1, 1, 0],
                    [1, 0, 2, 2],
                ],
                dtype=np.int32,
            )

    window_size = cfg.window_size
    print("Input X shape:", X.shape)
    print("Input y shape:", y.shape)

    patches_all, labels_all = create_patches(X, y, window_size, remove_zero_labels=False)
    print("patches_all.shape:", patches_all.shape)
    print("labels_all.shape:", labels_all.shape)
    # print("labels_all (row-major):", labels_all.tolist())

    patches_nonzero, labels_nonzero = create_patches(X, y, window_size, remove_zero_labels=True)
    print("patches_nonzero.shape:", patches_nonzero.shape)
    print("labels_nonzero.shape:", labels_nonzero.shape)
    # print("labels_nonzero (filtered):", labels_nonzero.tolist())

    patches_inf, H0, W0 = create_patches_inference(X, window_size)
    print("patches_inf.shape:", patches_inf.shape, "H0,W0:", H0, W0)

    preds_flat = np.arange(H0 * W0, dtype=np.int32) % 3
    # print("preds_flat (example):", preds_flat.tolist())
    label_map = reconstruct_from_flat_predictions(preds_flat, H0, W0)
    print("label_map shape:", label_map.shape)
    print("label_map:\n", label_map)


# Inference model patching test


def test_patching_model_inference():
    import torch

    from src.data.window_slider import WindowSlider

    x = torch.rand(1, 3, 256, 256)
    window_size = 64
    ws = WindowSlider(slide_keys=["img"], window_size=window_size, stride=window_size)
    model = lambda x: (x > 0.5).type(torch.float32)  # dummy model

    outs = []
    for patch in ws.slide_windows({"img": x}):
        print("Patch img shape:", patch["img"].shape)
        pred_patch = model(patch["img"])
        win_info = patch["window_info"]
        out = dict(img=pred_patch, window_info=win_info)
        outs.append(out)
        print("Pred patch shape:", pred_patch.shape)

    outs_merged = ws.merge_windows(outs, merge_method="average")
    img_model_out = outs_merged["img"]

    assert img_model_out.shape == x.shape, "Output shape must match input shape"
    assert (img_model_out == (x > 0.5).type(torch.float32)).all(), "Output values must match model output"


if __name__ == "__main__":
    # test_patching_recon(cfg=HyperClassificationConfig())
    test_patching_model_inference()
