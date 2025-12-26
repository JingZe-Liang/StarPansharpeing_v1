import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from typing import Optional
import os
import cv2
import glob
import tifffile


def load_as_tensor(path: str) -> Tensor:
    """Load image from path as a torch tensor [C, H, W]."""
    if path.endswith(("tiff", "tif")):
        img = tifffile.imread(path)
    else:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {path}")

    # If it's a 2D image (disparity or grayscale), add channel dim
    if img.ndim == 2:
        img = img[np.newaxis, ...]
    else:
        # HWC -> CHW
        img = img.transpose(2, 0, 1)
        # BGR -> RGB if it has 3 channels
        if img.shape[0] == 3:
            img = img[::-1, ...]

    return torch.from_numpy(img.astype(np.float32))


def tensor_to_numpy_img(tensor: Tensor) -> np.ndarray:
    """
    Convert a [C, H, W] or [1, C, H, W] tensor to a [H, W, C] numpy image.
    If C > 3, only the first 3 channels are taken.
    If C == 1, it returns [H, W].
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    img = tensor.detach().cpu().numpy()

    if img.ndim == 3:
        # [C, H, W] -> [H, W, C]
        img = img.transpose(1, 2, 0)

        # Handle channels
        if img.shape[2] > 3:
            img = img[:, :, :3]
        elif img.shape[2] == 1:
            img = img[:, :, 0]

    return img


def visualize_stereo_pair(
    left: Tensor,
    right: Tensor,
    gt: Tensor,
    pred: Optional[Tensor] = None,
    save_path: Optional[str] = None,
    title: str = "Stereo Matching Visualization",
    disp_min: Optional[float] = None,
    disp_max: Optional[float] = None,
    cmap: str = "jet",
) -> plt.Figure:
    """
    Visualize a stereo pair including left image, right image,
    ground truth disparity, and optional predicted disparity.

    Args:
        left: Left image tensor [C, H, W] or [1, C, H, W]
        right: Right image tensor [C, H, W] or [1, C, H, W]
        gt: Ground truth disparity tensor [1, H, W] or [H, W]
        pred: Predicted disparity tensor [1, H, W] or [H, W]
        save_path: Path to save the visualization image.
        title: Title of the plot.
        disp_min: Minimum value for disparity color scaling.
        disp_max: Maximum value for disparity color scaling.
        cmap: Colormap for disparity maps.

    Returns:
        matplotlib.figure.Figure: The generated figure object.
    """
    # Convert tensors to numpy
    left_np = tensor_to_numpy_img(left)
    right_np = tensor_to_numpy_img(right)
    gt_np = tensor_to_numpy_img(gt)

    # Normalize images for visualization if they are not in [0, 1]
    def normalize_img(img):
        if img.dtype != np.uint8:
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                img = (img - img_min) / (img_max - img_min)
        return img

    left_np = normalize_img(left_np)
    right_np = normalize_img(right_np)

    has_pred = pred is not None
    num_cols = 2
    num_rows = 2 if has_pred else 2  # Default 2x2: Left, Right, GT, (Pred or Empty)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=16)

    # 1. Left Image
    axes[0, 0].imshow(left_np)
    axes[0, 0].set_title("Left Image")
    axes[0, 0].axis("off")

    # 2. Right Image
    axes[0, 1].imshow(right_np)
    axes[0, 1].set_title("Right Image")
    axes[0, 1].axis("off")

    # Determine disparity limits
    if disp_min is None:
        disp_min = gt_np.min()
    if disp_max is None:
        disp_max = gt_np.max()

    # 3. Ground Truth Disparity
    im_gt = axes[1, 0].imshow(gt_np, cmap=cmap, vmin=disp_min, vmax=disp_max)
    axes[1, 0].set_title("Ground Truth Disparity")
    axes[1, 0].axis("off")
    fig.colorbar(im_gt, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # 4. Predicted Disparity (or placeholder)
    if has_pred:
        pred_np = tensor_to_numpy_img(pred)
        im_pred = axes[1, 1].imshow(pred_np, cmap=cmap, vmin=disp_min, vmax=disp_max)
        axes[1, 1].set_title("Predicted Disparity")
        axes[1, 1].axis("off")
        fig.colorbar(im_pred, ax=axes[1, 1], fraction=0.046, pad=0.04)
    else:
        axes[1, 1].text(0.5, 0.5, "No Prediction Provided", ha="center", va="center")
        axes[1, 1].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved visualization to {save_path}")

    return fig


def visualize_from_paths(
    left_path: str,
    right_path: str,
    gt_path: str,
    pred_path: Optional[str] = None,
    save_path: Optional[str] = None,
    **kwargs,
) -> plt.Figure:
    """
    Convenience function to visualize stereo results directly from file paths.
    """
    left = load_as_tensor(left_path)
    right = load_as_tensor(right_path)
    gt = load_as_tensor(gt_path)
    pred = load_as_tensor(pred_path) if pred_path else None

    return visualize_stereo_pair(left=left, right=right, gt=gt, pred=pred, save_path=save_path, **kwargs)


def __test_vis():
    root_dir = "/Data2/ZihanCao/dataset/WHU_Stereo_Matching/experimental data/with ground truth/train"
    left_dir = os.path.join(root_dir, "left")
    right_dir = os.path.join(root_dir, "right")
    disp_dir = os.path.join(root_dir, "disp")

    if not os.path.exists(root_dir):
        print(f"Dataset root not found: {root_dir}")
        return

    # Find the first image in left dir
    left_files = sorted(glob.glob(os.path.join(left_dir, "*")))
    if not left_files:
        print("No images found in left directory")
        return

    left_path = left_files[0]
    filename = os.path.basename(left_path)

    # Naming convention:
    # Left:  KM_left_X.tiff
    # Right: KM_right_X.tiff
    # Disp:  KM_disparity_X.tiff

    right_filename = filename.replace("left", "right")
    disp_filename = filename.replace("left", "disparity")

    # Construct paths for right and disp
    right_path = os.path.join(right_dir, right_filename)
    disp_path = os.path.join(disp_dir, disp_filename)

    # Check if they exist
    if not os.path.exists(right_path):
        print(f"Right image not found: {right_path}")
        return
    if not os.path.exists(disp_path):
        print(f"Disparity image not found: {disp_path}")
        return

    print(f"Visualizing pair: {filename}")
    save_path = "./test_vis_stereo.png"

    visualize_from_paths(
        left_path=left_path,
        right_path=right_path,
        gt_path=disp_path,
        save_path=save_path,
        title=f"Test Visualization: {filename}",
    )


if __name__ == "__main__":
    __test_vis()
