"""
Unified stereo matching visualization tool.

Supports three input types:
1. str: File path, automatically reads the image
2. Tensor: PyTorch tensor, default format (B, C, H, W)
3. np.ndarray: NumPy array, default format (H, W, C)

Output: list[Image.Image]
"""

import io
from typing import Union, Optional, Literal
from pathlib import Path

import matplotlib
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from PIL import Image
import cv2
import tifffile


# =============================================================================
# Helper Functions
# =============================================================================


def load_image_from_path(path: str) -> np.ndarray:
    """
    Load image from path as a NumPy array (H, W, C) or (H, W).

    Args:
        path: Image file path

    Returns:
        np.ndarray: (H, W, C) for RGB or (H, W) for grayscale/disparity
    """
    if path.endswith((".tiff", ".tif")):
        img = tifffile.imread(path)
    else:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise FileNotFoundError(f"Could not read image at {path}")

    # BGR -> RGB for OpenCV images
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img.astype(np.float32)


def tensor_to_numpy(tensor: Tensor) -> np.ndarray:
    """
    Convert Tensor (B, C, H, W) or (C, H, W) to NumPy (H, W, C) or (H, W).

    Args:
        tensor: PyTorch Tensor

    Returns:
        np.ndarray: (H, W, C) or (H, W)
    """
    # Remove batch dimension (if only one sample)
    if tensor.dim() == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)

    img = tensor.detach().cpu().numpy()

    if img.ndim == 3:
        # (C, H, W) -> (H, W, C)
        img = img.transpose(1, 2, 0)

        # Handle channels
        if img.shape[2] > 3:
            img = img[:, :, :3]  # Keep only first 3 channels
        elif img.shape[2] == 1:
            img = img[:, :, 0]  # Remove single channel dimension

    return img


def normalize_input(data: Union[str, Tensor, np.ndarray]) -> np.ndarray:
    """
    Unify input format to NumPy array (H, W, C) or (H, W).

    Args:
        data: Input data (path, Tensor, or NumPy array)

    Returns:
        np.ndarray: (H, W, C) or (H, W)
    """
    if isinstance(data, str):
        # Path: read image
        return load_image_from_path(data)
    elif isinstance(data, Tensor):
        # Tensor (B, C, H, W) -> NumPy (H, W, C)
        return tensor_to_numpy(data)
    elif isinstance(data, np.ndarray):
        # Already NumPy array, assume (H, W, C) or (H, W)
        return data
    else:
        raise TypeError(f"Unsupported input type: {type(data)}. Expected str, Tensor, or np.ndarray")


# =============================================================================
# Main Visualization Function
# =============================================================================


def visualize_stereo(
    left_rgb: Union[str, Tensor, np.ndarray],
    right_rgb: Union[str, Tensor, np.ndarray],
    dsp_gt: Union[str, Tensor, np.ndarray],
    dsp_pred: Optional[Union[str, Tensor, np.ndarray]] = None,
    agl: Optional[Union[str, Tensor, np.ndarray]] = None,
    left_seg: Optional[Union[str, Tensor, np.ndarray]] = None,
    right_seg: Optional[Union[str, Tensor, np.ndarray]] = None,
    pred_seg_left: Optional[Union[str, Tensor, np.ndarray]] = None,
    pred_seg_right: Optional[Union[str, Tensor, np.ndarray]] = None,
    title: Optional[str] = None,
    dsp_vmin: Optional[float] = None,
    dsp_vmax: Optional[float] = None,
    invalid_thres: Optional[float | int] = None,
    img_normalization: Optional[Literal["auto", "neg1_1", "0_1"]] = None,
) -> list[Image.Image]:
    """
    Unified stereo matching visualization function.

    Supports three input types:
    - str: File path
    - Tensor: PyTorch Tensor (B, C, H, W) or (C, H, W)
    - np.ndarray: NumPy array (H, W, C) or (H, W)

    Args:
        left_rgb: Left RGB/grayscale image
        right_rgb: Right RGB/grayscale image
        dsp_gt: Ground truth disparity
        dsp_pred: Predicted disparity (optional)
        agl: AGL (Height) map (optional)
        left_seg: Left semantic segmentation ground truth (optional)
        right_seg: Right semantic segmentation ground truth (optional)
        pred_seg_left: Left semantic segmentation prediction (optional)
        pred_seg_right: Right semantic segmentation prediction (optional)
        title: Title (optional)
        dsp_vmin: Minimum value for disparity color range (optional, default uses data range)
        dsp_vmax: Maximum value for disparity color range (optional, default uses data range)
        invalid_thres: Threshold for invalid disparity values (e.g., -500 for US3D)
        img_normalization: Image normalization mode:
            - "auto": Automatically detect from value range (default)
            - "neg1_1": Input is in [-1, 1] range, will convert to [0, 1]
            - "0_1": Input is already in [0, 1] range, no conversion needed

    Returns:
        list[Image.Image]: List of PIL Images (multiple if batch is present)
    """
    # Normalize all inputs
    left_np = normalize_input(left_rgb)
    right_np = normalize_input(right_rgb)
    dsp_gt_np = normalize_input(dsp_gt)

    dsp_pred_np = normalize_input(dsp_pred) if dsp_pred is not None else None
    agl_np = normalize_input(agl) if agl is not None else None
    left_seg_np = normalize_input(left_seg) if left_seg is not None else None
    right_seg_np = normalize_input(right_seg) if right_seg is not None else None
    pred_seg_left_np = normalize_input(pred_seg_left) if pred_seg_left is not None else None
    pred_seg_right_np = normalize_input(pred_seg_right) if pred_seg_right is not None else None

    # Handle batch dimension (if Tensor input has batch)
    # Detect batch: if Tensor and first input is 4D
    has_batch = False
    batch_size = 1

    if isinstance(left_rgb, Tensor) and left_rgb.dim() == 4:
        has_batch = True
        batch_size = left_rgb.shape[0]

    # If batch exists, process each sample
    outputs = []

    for b in range(batch_size):
        # Extract single sample
        if has_batch and isinstance(left_rgb, Tensor):
            # Use tensor indexing to extract sample, then convert
            left_b = tensor_to_numpy(left_rgb[b])
            right_b = tensor_to_numpy(right_rgb[b])
            dsp_gt_b = tensor_to_numpy(dsp_gt[b])

            dsp_pred_b = tensor_to_numpy(dsp_pred[b]) if dsp_pred is not None else None
            agl_b = tensor_to_numpy(agl[b]) if agl is not None else None
            left_seg_b = tensor_to_numpy(left_seg[b]) if left_seg is not None else None
            right_seg_b = tensor_to_numpy(right_seg[b]) if right_seg is not None else None
            pred_seg_left_b = tensor_to_numpy(pred_seg_left[b]) if pred_seg_left is not None else None
            pred_seg_right_b = tensor_to_numpy(pred_seg_right[b]) if pred_seg_right is not None else None
        else:
            # Non-batch mode, use already converted arrays
            left_b = left_np
            right_b = right_np
            dsp_gt_b = dsp_gt_np
            dsp_pred_b = dsp_pred_np
            agl_b = agl_np
            left_seg_b = left_seg_np
            right_seg_b = right_seg_np
            pred_seg_left_b = pred_seg_left_np
            pred_seg_right_b = pred_seg_right_np

        # Generate single visualization
        sample_title = title
        if batch_size > 1 and title:
            sample_title = f"{title} - Sample {b + 1}/{batch_size}"

        img = _visualize_single_sample(
            left_rgb=left_b,
            right_rgb=right_b,
            dsp_gt=dsp_gt_b,
            dsp_pred=dsp_pred_b,
            agl=agl_b,
            left_seg=left_seg_b,
            right_seg=right_seg_b,
            pred_seg_left=pred_seg_left_b,
            pred_seg_right=pred_seg_right_b,
            title=sample_title,
            dsp_vmin=dsp_vmin,
            dsp_vmax=dsp_vmax,
            invalid_thres=invalid_thres,
            img_normalization=img_normalization,
        )
        outputs.append(img)

    return outputs


def _visualize_single_sample(
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    dsp_gt: np.ndarray,
    dsp_pred: Optional[np.ndarray] = None,
    agl: Optional[np.ndarray] = None,
    left_seg: Optional[np.ndarray] = None,
    right_seg: Optional[np.ndarray] = None,
    pred_seg_left: Optional[np.ndarray] = None,
    pred_seg_right: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    invalid_thres: Optional[float] = None,
    dsp_vmin: Optional[float] = None,
    dsp_vmax: Optional[float] = None,
    img_normalization: Optional[Literal["auto", "neg1_1", "0_1"]] = None,
) -> Image.Image:
    """
    Visualize a single sample (internal function).

    All inputs are NumPy arrays:
    - RGB: (H, W, 3)
    - Disparity/AGL: (H, W)
    - Segmentation: (H, W)
    """
    # Verify shape - support both RGB (H, W, 3) and grayscale (H, W) or (H, W, 1)
    if left_rgb.ndim == 3:
        if left_rgb.shape[2] == 3:
            h, w = left_rgb.shape[:2]
        elif left_rgb.shape[2] == 1:
            # Grayscale with channel dimension - squeeze it
            left_rgb = left_rgb[:, :, 0]
            right_rgb = right_rgb[:, :, 0]
            h, w = left_rgb.shape
        else:
            raise ValueError(f"left_rgb must be (H, W, 3) for RGB or (H, W, 1) for grayscale, got {left_rgb.shape}")
    elif left_rgb.ndim == 2:
        # Already grayscale (H, W)
        h, w = left_rgb.shape
    else:
        raise ValueError(f"left_rgb must be 2D (H, W) or 3D (H, W, C), got {left_rgb.shape}")

    # Collect items to plot
    items = []

    # Determine if images are grayscale
    is_grayscale = left_rgb.ndim == 2
    img_title = "Left Grayscale" if is_grayscale else "Left RGB"

    # Images (RGB or Grayscale)
    items.append((left_rgb, img_title, "gray" if is_grayscale else None, False))
    items.append((right_rgb, img_title.replace("Left", "Right"), "gray" if is_grayscale else None, False))

    # AGL (Optional)
    if agl is not None:
        items.append((agl, "AGL (Height)", "jet", False))

    # Disparity
    items.append((dsp_gt, "DSP (GT)", "RdBu_r", False))
    if dsp_pred is not None:
        items.append((dsp_pred, "DSP (Pred)", "RdBu_r", False))

    # Semantic Segmentation
    if left_seg is not None:
        items.append((left_seg, "Left Seg (GT)", "tab20", True))
    if pred_seg_left is not None:
        items.append((pred_seg_left, "Left Seg (Pred)", "tab20", True))
    if right_seg is not None:
        items.append((right_seg, "Right Seg (GT)", "tab20", True))
    if pred_seg_right is not None:
        items.append((pred_seg_right, "Right Seg (Pred)", "tab20", True))

    # Create subplots
    num_items = len(items)
    cols = 3
    rows = (num_items + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if title:
        fig.suptitle(title, fontsize=16)

    # Handle axes
    if isinstance(axes, np.ndarray):
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]

    # Plot each item
    for i, ax in enumerate(axes_flat):
        if i < num_items:
            img_data, name, cmap, is_discrete = items[i]

            plot_kwargs = {}
            if cmap:
                plot_kwargs["cmap"] = cmap

            processed_data = img_data

            # Normalize RGB/Grayscale to [0, 1]
            if "RGB" in name or "Grayscale" in name:
                if img_normalization == "neg1_1":
                    # User specified input is [-1, 1], convert to [0, 1]
                    processed_data = (img_data + 1.0) / 2.0
                elif img_normalization == "0_1":
                    # User specified input is already [0, 1], no conversion
                    processed_data = img_data
                elif img_normalization == "auto" or img_normalization is None:
                    # Auto-detect from value range
                    if img_data.min() < -0.1:  # Likely [-1, 1] range
                        processed_data = (img_data + 1.0) / 2.0
                    elif img_data.max() > 1.1:  # Likely [0, 255] range
                        processed_data = img_data / 255.0
                    else:
                        processed_data = img_data  # Assume [0, 1]
                else:
                    # Invalid normalization mode, default to auto
                    if img_data.min() < -0.1:
                        processed_data = (img_data + 1.0) / 2.0
                    elif img_data.max() > 1.1:
                        processed_data = img_data / 255.0
                    else:
                        processed_data = img_data
                processed_data = np.clip(processed_data, 0.0, 1.0)

            # Special handling
            if "AGL" in name:
                masked_data = np.ma.masked_invalid(img_data)
                valid_data = masked_data.compressed()
                # if valid_data.size > 0:
                #     vmin, vmax = np.percentile(valid_data, [1, 99])
                #     plot_kwargs["vmin"] = vmin
                #     plot_kwargs["vmax"] = vmax
                processed_data = masked_data

            elif "DSP" in name:
                # Mask invalid values (< -500 typically means invalid)
                if invalid_thres is not None:
                    masked_data = np.ma.masked_where(img_data < invalid_thres, img_data)
                else:
                    masked_data = np.ma.masked_invalid(img_data)

                # Robust scaling using percentiles to avoid outliers and fix "all white" issue
                valid_data = masked_data.compressed()
                # if valid_data.size > 0:
                #     # Use 2nd and 98th percentiles
                #     # p_min, p_max = np.percentile(valid_data, [2, 98])

                #     p_min, p_max = valid_data.min(), valid_data.max()
                #     # Symmetric range centered at 0
                #     limit = max(abs(p_min), abs(p_max))
                #     # if limit < 1e-4:
                #     #     limit = 1.0  # default range if data is all 0

                #     plot_kwargs["vmin"] = -limit
                #     plot_kwargs["vmax"] = limit

                # Handle mask visualization - use separate variable to avoid overwriting cmap
                cmap_name = plot_kwargs.get("cmap", "RdBu_r")
                if isinstance(cmap_name, str):
                    try:
                        # Modern matplotlib (3.5+)
                        # The linter might not know about colormaps or __getitem__ on it
                        dsp_cmap = matplotlib.colormaps[cmap_name].copy()  # type: ignore
                    except (AttributeError, KeyError, TypeError):
                        # Fallback for older versions or if colormaps attr is missing
                        if hasattr(matplotlib.cm, "get_cmap"):
                            dsp_cmap = matplotlib.cm.get_cmap(cmap_name).copy()
                        else:
                            # Should not happen in typical matplotlib versions, but raise clear error
                            raise AttributeError(f"Could not find colormap {cmap_name} and get_cmap is missing.")
                else:
                    dsp_cmap = cmap_name.copy()

                # Set masked values to a distinct color (e.g., black)
                # to separate from 0-value (white) data
                dsp_cmap.set_bad(color="black")
                plot_kwargs["cmap"] = dsp_cmap

                plot_kwargs["interpolation"] = "nearest"
                processed_data = masked_data

            elif is_discrete:
                plot_kwargs["interpolation"] = "nearest"

            im = ax.imshow(processed_data, **plot_kwargs)
            ax.set_title(name)
            ax.axis("off")

            # Use original cmap from items to decide whether to add colorbar
            if cmap is not None:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.axis("off")

    plt.tight_layout()

    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return Image.open(buf)


# =============================================================================
# Backward Compatibility (Legacy Functions)
# =============================================================================


def visualize_stereo_sample(
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    dsp_gt: np.ndarray,
    dsp_pred: Optional[np.ndarray] = None,
    agl: Optional[np.ndarray] = None,
    left_seg: Optional[np.ndarray] = None,
    right_seg: Optional[np.ndarray] = None,
    pred_seg_left: Optional[np.ndarray] = None,
    pred_seg_right: Optional[np.ndarray] = None,
    title: Optional[str] = None,
) -> Image.Image:
    """Backward compatibility interface, accepts only NumPy arrays."""
    return visualize_stereo(
        left_rgb=left_rgb,
        right_rgb=right_rgb,
        dsp_gt=dsp_gt,
        dsp_pred=dsp_pred,
        agl=agl,
        left_seg=left_seg,
        right_seg=right_seg,
        pred_seg_left=pred_seg_left,
        pred_seg_right=pred_seg_right,
        title=title,
    )[0]  # Return the first image (single sample)
