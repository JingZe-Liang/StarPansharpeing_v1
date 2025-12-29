import io
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def visualize_stereo_sample(
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    dsp_gt: np.ndarray,
    dsp_pred: Optional[np.ndarray] = None,
    agl: Optional[np.ndarray] = None,
    left_seg: Optional[np.ndarray] = None,
    right_seg: Optional[np.ndarray] = None,
    title: Optional[str] = None,
) -> Image.Image:
    """
    Visualizes a stereo sample including RGB images, disparity (GT and predicted),
    AGL, and segmentation labels. Returns a PIL Image.
    """

    # Assertions for shapes
    # Check Left RGB
    if left_rgb.ndim == 3 and left_rgb.shape[2] == 3:
        pass  # Expected (H, W, 3)
    else:
        raise ValueError(f"Left RGB must be (H, W, 3), got {left_rgb.shape}")

    h, w = left_rgb.shape[:2]

    # Check Right RGB
    if right_rgb.ndim == 3 and right_rgb.shape[2] == 3:
        assert right_rgb.shape[:2] == (h, w), f"Right RGB shape mismatch: {right_rgb.shape[:2]} vs {h, w}"
    else:
        raise ValueError(f"Right RGB must be (H, W, 3), got {right_rgb.shape}")

    # Check DSP GT
    if dsp_gt.ndim == 2:
        assert dsp_gt.shape == (h, w), f"DSP GT shape mismatch: {dsp_gt.shape} vs {h, w}"
    else:
        raise ValueError(f"DSP GT must be (H, W), got {dsp_gt.shape}")

    # Check Optional Inputs
    if dsp_pred is not None:
        assert dsp_pred.shape == (h, w), f"DSP Pred shape mismatch: {dsp_pred.shape} vs {h, w}"

    if agl is not None:
        assert agl.shape == (h, w), f"AGL shape mismatch: {agl.shape} vs {h, w}"

    if left_seg is not None:
        assert left_seg.shape == (h, w), f"Left Seg shape mismatch: {left_seg.shape} vs {h, w}"

    if right_seg is not None:
        assert right_seg.shape == (h, w), f"Right Seg shape mismatch: {right_seg.shape} vs {h, w}"

    # Collect items to plot to determine grid size
    # Structure: (Image, Title, Colormap, IsDiscrete)
    items = []

    items.append((left_rgb, "Left RGB", None, False))
    items.append((right_rgb, "Right RGB", None, False))

    if agl is not None:
        items.append((agl, "AGL (Height)", "jet", False))

    items.append((dsp_gt, "DSP (GT)", "RdBu_r", False))

    if dsp_pred is not None:
        items.append((dsp_pred, "DSP (Pred)", "RdBu_r", False))

    if left_seg is not None:
        items.append((left_seg, "Left Seg", "tab20", True))

    if right_seg is not None:
        items.append((right_seg, "Right Seg", "tab20", True))

    num_items = len(items)
    cols = 3
    rows = (num_items + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if title:
        fig.suptitle(title, fontsize=16)

    # Flatten axes for easy iteration, handle case of single subplot
    if isinstance(axes, np.ndarray):
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]

    for i, ax in enumerate(axes_flat):
        if i < num_items:
            img_data, name, cmap, is_discrete = items[i]

            # Handle specific data types
            plot_kwargs = {}
            if cmap:
                plot_kwargs["cmap"] = cmap

            processed_data = img_data

            # Special handling based on name/type
            if "AGL" in name:
                # Mask NaNs
                masked_data = np.ma.masked_invalid(img_data)
                valid_data = masked_data.compressed()
                if valid_data.size > 0:
                    vmin, vmax = np.percentile(valid_data, [1, 99])
                    plot_kwargs["vmin"] = vmin
                    plot_kwargs["vmax"] = vmax
                processed_data = masked_data

            elif "DSP" in name:
                # Mask Invalid values (<-500)
                masked_data = np.ma.masked_where(img_data < -500, img_data)
                valid_data = masked_data.compressed()
                if valid_data.size > 0:
                    vmin, vmax = np.percentile(valid_data, [1, 99])
                    plot_kwargs["vmin"] = vmin
                    plot_kwargs["vmax"] = vmax
                # Use nearest interpolation for sharp disparity boundaries
                plot_kwargs["interpolation"] = "nearest"
                processed_data = masked_data

            elif is_discrete:
                plot_kwargs["interpolation"] = "nearest"

            im = ax.imshow(processed_data, **plot_kwargs)
            ax.set_title(name)
            ax.axis("off")

            # Add colorbar for non-RGB images
            if cmap is not None:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.axis("off")

    plt.tight_layout()

    # Save to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return Image.open(buf)
