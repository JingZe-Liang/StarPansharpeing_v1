from collections.abc import Generator
from typing import Literal

import torch
from beartype import beartype
from jaxtyping import Float, Int
from torch import Tensor


def neighbor(
    img: Tensor,
    center: Tensor | list,
    patch_size: int = 32,
    padding_mode: str = "reflect",
):
    """
    Extract a patch from image centered at the given coordinates.

    Args:
        img: Input image tensor (..., H, W)
        center: Center coordinates (x, y)
        patch_size: Size of the square patch to extract
        padding_mode: How to handle boundaries - 'reflect', 'constant', 'replicate', 'circular'

    Returns:
        Patch of shape (..., patch_size, patch_size)
    """
    if torch.is_tensor(center):
        center = center.squeeze()
        assert center.ndim == 1 and center.numel() == 2, (
            f"Center must be a 1D tensor with 2 elements, got {center}"
        )
        x, y = center.tolist()
    else:
        x, y = center

    height, width = img.shape[-2:]
    half_patch = patch_size // 2

    # Calculate padding needed
    pad_left = max(0, half_patch - y)
    pad_right = max(0, y + half_patch - width)
    pad_top = max(0, half_patch - x)
    pad_bottom = max(0, x + half_patch - height)

    # Add padding if needed
    if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
        # Calculate padding for each dimension (only for H and W)
        # For torch.nn.functional.pad: (left, right, top, bottom) for last two dimensions
        padding = (pad_left, pad_right, pad_top, pad_bottom)

        # Handle 2D tensors (labels) by adding a channel dimension
        if img.dim() == 2:
            # For 2D tensors, use constant padding mode since non-constant modes aren't supported
            img = torch.nn.functional.pad(
                img.unsqueeze(0), padding, mode="constant"
            ).squeeze(0)
        else:
            # For 3D+ tensors, use the specified padding mode
            img = torch.nn.functional.pad(img, padding, mode=padding_mode)

        # Update coordinates after padding
        x += pad_top
        y += pad_left

    # Extract the patch
    return img[
        ...,
        x - half_patch : x + half_patch,
        y - half_patch : y + half_patch,
    ]


def img_changed_by_label(
    label: Int[Tensor, "H W"],
    # CD only cares two types of changes ?
    unchanged_label: int = 0,
    changed_label: int | list[int] = 1,
):
    if isinstance(changed_label, int):
        changed = (label == changed_label).nonzero()
    else:
        # list of changed labels, merge them all
        changes_ = []
        for i in changed_label:
            changes_.append((label == i).nonzero())
        changed = torch.cat(changes_)
    unchanged = (label == unchanged_label).nonzero()

    return changed, unchanged


def permute_batch(data: Tensor, perm: Tensor | None = None):
    if perm is None:
        bs = data.shape[0]
        perm = torch.randperm(bs, device=data.device)
    return data[perm], perm


@beartype
def label_centrical_patcher(
    img1: Float[Tensor, "B C H W"],
    img2: Float[Tensor, "B C H W"],
    label: Int[Tensor, "B H W"],
    # CD only cares two types of changes ?
    unchanged_label: int = 2,
    changed_label: int | list[int] = 1,
    micro_batch_size: int = 32,
    changed_ratio: float = 0.5,
    patch_size: int = 32,
    label_mode: Literal["seg", "cls"] = "seg",
) -> Generator[tuple[Tensor, Tensor, Tensor], None, None]:
    """
    Generate balanced patches for change detection tasks by sampling from changed and unchanged regions.

    This function extracts patches from paired images (img1, img2) and their corresponding labels,
    ensuring a balanced sampling of changed and unchanged regions based on the specified ratio.
    The function yields micro-batches of patches to facilitate efficient training.

    Args:
        img1 (Tensor): First image tensor with shape (B, C, H, W)
        img2 (Tensor): Second image tensor with shape (B, C, H, W)
        label (Tensor): Label tensor with shape (B, H, W) containing change detection labels
        unchanged_label (int): Label value representing unchanged regions. Defaults to 2.
        changed_label (int | list[int]): Label value(s) representing changed regions. Defaults to 1.
        micro_batch_size (int): Number of patches to yield in each micro-batch. Defaults to 32.
        changed_ratio (float): Ratio of changed patches in each micro-batch (0.0-1.0). Defaults to 0.5.
        patch_size (int): Size of square patches to extract (patch_size x patch_size). Defaults to 32.
        label_mode (str): Mode for label processing ('seg' or 'cls'). Defaults to "seg".
            - 'seg': Return segmentation patches with shape (B, patch_size, patch_size)
            - 'cls': Return classification labels with shape (B,)

    Yields:
        tuple[Tensor, Tensor, Tensor]: A tuple containing:
            - img1_patches: Patches from first image with shape (micro_batch_size, C, patch_size, patch_size)
            - img2_patches: Patches from second image with shape (micro_batch_size, C, patch_size, patch_size)
            - label_patches: Label patches or classifications based on label_mode

    Raises:
        ValueError: If no changes are found in the label image
        AssertionError: If image shapes don't match or image/label spatial dimensions don't align

    Note:
        Dataset label conventions:
        - OSCD: changed: 1, unchanged: 0
        - Barbara: changed: 1, unchanged: 2, unknown: 0
        - BayArea: changed: 1, unchanged: 2, unknown: 0
        - Hermiston: changed: 1, unchanged: 2, unknown: 0
        - FarmLand: changed: 2, unchanged: 1, unknown: 0

    Example:
        >>> for img1_patches, img2_patches, label_patches in label_centrical_patcher(img1, img2, labels):
        ...     # Train model with balanced patches
        ...     loss = model(img1_patches, img2_patches, label_patches)
    """
    collect_fn_ = lambda img_or_gt, center: neighbor(img_or_gt, center, patch_size)
    selected_collect_fn = lambda data, selected_indices: [
        collect_fn_(data, center) for center in selected_indices
    ]

    # Assertions
    assert img1.shape == img2.shape, (
        f"Image shapes must match, got {img1.shape} and {img2.shape}"
    )
    assert img1.shape[-2:] == label.shape[-2:], (
        f"Image and label spatial shapes must match, got {img1.shape[-2:]} and {label.shape[-2:]}"
    )

    bs = img1.shape[0]

    # Must for-loop across batch dim
    # For simplicity, we just store them all and then yield

    changed_indices = {}
    unchanged_indices = {}
    for i in range(bs):
        label_i = label[i]

        changed_idx, unchanged_idx = img_changed_by_label(
            label_i, unchanged_label, changed_label
        )
        if len(changed_idx) == 0 or len(unchanged_idx) == 0:
            # If no changes, return empty tensors
            raise ValueError("No changes found in the label image")

        # Randomly sample from changed and unchanged pixels
        changed_indices[i] = changed_idx
        unchanged_indices[i] = unchanged_idx

    num_changed_per_batch = int(micro_batch_size * changed_ratio)
    num_unchanged_per_batch = micro_batch_size - num_changed_per_batch

    # Collect patches
    img1_patches = []
    img2_patches = []
    label_patches = []
    for i in range(bs):
        changed_idx = changed_indices[i]
        unchanged_idx = unchanged_indices[i]

        # batch indices
        changed_perm = torch.randperm(len(changed_idx))[:num_changed_per_batch]
        unchanged_perm = torch.randperm(len(unchanged_idx))[:num_unchanged_per_batch]

        selected_changed = changed_idx[changed_perm]
        selected_unchanged = unchanged_idx[unchanged_perm]

        # Collect patches
        for select_idx in [selected_changed, selected_unchanged]:
            img1_patches.extend(selected_collect_fn(img1[i], select_idx))
            img2_patches.extend(selected_collect_fn(img2[i], select_idx))
            if label_mode == "seg":
                label_patches.extend(selected_collect_fn(label[i], select_idx))
            elif label_mode == "cls":
                # For cls, just use the center pixel's label
                center_labels = label[i][..., select_idx[:, 0], select_idx[:, 1]]
                label_patches.extend(center_labels.tolist())

        # Yield micro-batches
        assert len(img1_patches) == len(img2_patches) == len(label_patches)
        if len(img1_patches) >= micro_batch_size:
            img1_y = torch.stack(img1_patches)[:micro_batch_size]
            img2_y = torch.stack(img2_patches)[:micro_batch_size]
            if label_mode == "seg":
                label_y = torch.stack(label_patches)[:micro_batch_size]
            else:
                # for cls
                label_y = torch.as_tensor(
                    label_patches[:micro_batch_size], dtype=torch.int64
                )

            # permute
            img1_y, perm = permute_batch(img1_y)
            img2_y, _ = permute_batch(img2_y, perm)
            label_y, _ = permute_batch(label_y, perm)

            yield img1_y, img2_y, label_y

            img1_patches = img1_patches[micro_batch_size:]
            img2_patches = img2_patches[micro_batch_size:]
            label_patches = label_patches[micro_batch_size:]

    # Yield remaining patches if any
    if len(img1_patches) > 0:
        img1_y = torch.stack(img1_patches)
        img2_y = torch.stack(img2_patches)
        if label_mode == "seg":
            label_y = torch.stack(label_patches)
        else:
            # for cls
            label_y = torch.as_tensor(label_patches, dtype=torch.int64)

        img1_y, perm = permute_batch(img1_y)
        img2_y, _ = permute_batch(img2_y, perm)
        label_y, _ = permute_batch(label_y, perm)

        yield img1_y, img2_y, label_y
