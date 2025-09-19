"""
PyTorch implementation of RX (Reed-Xiaoli) anomaly detector for hyperspectral images.

This module provides both NumPy and PyTorch implementations of the RX algorithm,
with support for batched processing in PyTorch version.

The RX algorithm computes the Mahalanobis distance to detect anomalies in
hyperspectral images based on statistical properties of the background.

Reference:
    Reed, I.S. and Xiaoli Yu, "Adaptive multiple-band CFAR detection of an optical
    pattern with unknown spectral distribution", IEEE Transactions on Acoustics,
    Speech and Signal Processing, 1990.
"""

import numpy as np
import torch


def RX_numpy(hsi_img):
    """
    Standard NumPy implementation of RX detector (kept for compatibility).

    Args:
        hsi_img (np.ndarray): Hyperspectral image of shape (n_row, n_col, n_band)

    Returns:
        np.ndarray: Anomaly score map of shape (n_row, n_col)
    """
    # Handle constant images (avoid division by zero)
    img_min = np.min(hsi_img)
    img_max = np.max(hsi_img)
    if img_max == img_min:
        # For constant images, return zero scores
        return np.zeros((hsi_img.shape[0], hsi_img.shape[1]))

    hsi_img = (hsi_img - img_min) / (img_max - img_min)
    n_row, n_col, n_band = hsi_img.shape
    n_pixels = n_row * n_col
    hsi_data = np.reshape(hsi_img, (n_pixels, n_band), order="F").T

    mu = np.mean(hsi_data, 1)
    sigma = np.cov(hsi_data.T, rowvar=False)

    z = hsi_data - mu[:, np.newaxis]
    # Use pseudo-inverse for better numerical stability with singular matrices
    sig_inv = np.linalg.pinv(sigma)

    dist_data = np.zeros(n_pixels)
    for i in range(n_pixels):
        dist_data[i] = z[:, i].T @ sig_inv @ z[:, i]

    return dist_data.reshape([n_col, n_row]).T


def RX_torch(hsi_img_batch):
    """
    PyTorch implementation of RX detector with batch support.

    Args:
        hsi_img_batch (torch.Tensor): Batched hyperspectral images of shape (B, C, H, W)
            where B is batch size, C is number of channels/bands, H is height, W is width

    Returns:
        torch.Tensor: Batched anomaly score maps of shape (B, H, W)
    """
    # Get dimensions
    B, C, H, W = hsi_img_batch.shape
    n_pixels = H * W

    # Normalize each image in the batch
    img_min = hsi_img_batch.amin(dim=(-3, -2, -1), keepdim=True)
    img_max = hsi_img_batch.amax(dim=(-3, -2, -1), keepdim=True)

    # Handle constant images (avoid division by zero)
    constant_mask = (img_max == img_min).squeeze()
    if constant_mask.any():
        # For constant images, return zero scores
        result = torch.zeros(
            B, H, W, device=hsi_img_batch.device, dtype=hsi_img_batch.dtype
        )
        return result
    hsi_normalized = (hsi_img_batch - img_min) / (img_max - img_min)

    # Reshape to (B, C, n_pixels) and then transpose to (B, n_pixels, C)
    hsi_data = hsi_normalized.reshape(B, C, n_pixels).permute(0, 2, 1)

    # Compute mean for each image in batch: (B, C)
    mu = hsi_data.mean(dim=1)

    # Compute covariance matrix for each image in batch: (B, C, C)
    # Center the data
    centered_data = hsi_data - mu.unsqueeze(1)  # (B, n_pixels, C)

    # Compute covariance matrix: (B, C, C)
    sigma = torch.bmm(centered_data.permute(0, 2, 1), centered_data) / (n_pixels - 1)

    # Compute pseudo-inverse of covariance matrix for each image in batch
    # Add small regularization to ensure invertibility
    reg = 1e-6
    eye = torch.eye(C, device=sigma.device).unsqueeze(0).expand(B, -1, -1)  # (B, C, C)
    sigma_reg = sigma + reg * eye

    # Use pseudo-inverse for better numerical stability with singular matrices
    sig_inv = torch.linalg.pinv(sigma_reg)

    # Compute RX score for each pixel
    z = centered_data  # (B, n_pixels, C)

    # Compute Mahalanobis distance: z @ sig_inv @ z.T for each pixel
    # This is equivalent to sum((z @ sig_inv) * z, dim=-1)
    z_sig_inv = torch.bmm(z, sig_inv)  # (B, n_pixels, C)
    dist_data = torch.sum(z_sig_inv * z, dim=-1)  # (B, n_pixels)

    # Reshape to (B, H, W)
    return dist_data.reshape(B, H, W)


def RX(hsi_img: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """
    Wrapper function that handles both NumPy and PyTorch inputs.

    Args:
        hsi_img: Either numpy array of shape (n_row, n_col, n_band) or
                 torch tensor of shape (B, C, H, W) or (C, H, W)

    Returns:
        numpy array or torch tensor: Anomaly score map
    """
    if isinstance(hsi_img, torch.Tensor):
        # Handle different input shapes for PyTorch tensors
        if hsi_img.dim() == 3:  # (C, H, W) - add batch dimension
            hsi_img = hsi_img.unsqueeze(0)
        if hsi_img.dim() == 4:  # (B, C, H, W) - batched format
            return RX_torch(hsi_img)
        else:
            raise ValueError(f"Unsupported PyTorch tensor shape: {hsi_img.shape}")
    elif isinstance(hsi_img, np.ndarray):
        # Assume numpy array is (n_row, n_col, n_band)
        return RX_numpy(hsi_img)
    else:
        raise TypeError(f"Unsupported input type: {type(hsi_img)}")
