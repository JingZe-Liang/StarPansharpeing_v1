from collections.abc import Sequence
from typing import Any, cast

import torch as th
import torch


def choose_lightest_bands(img: th.Tensor | Any):
    mean_cs = img.mean((-3, -2, -1))  # (c,)
    mean_cs = th.as_tensor(mean_cs)
    _, indices = th.topk(mean_cs, k=3, largest=True)
    indices = indices.tolist()
    indices = sorted(indices, reverse=True)
    assert indices[-1] < img.shape[1], f"Invalid channel index {indices[-1]} for image with {img.shape[1]} channels."
    return indices


def get_rgb_channels_for_model(
    rgb_channels: Sequence | torch.Tensor | str | None,
    img: th.Tensor,
    use_linstretch: bool,
    pca_fn=None,
):
    """Extract RGB channels from multi/hyperspectral images using various strategies.

    This function provides a unified interface for selecting RGB channels from
    multispectral or hyperspectral images using different strategies like random
    selection, mean splitting, brightest bands selection, or PCA.

    Parameters
    ----------
    rgb_channels : Sequence | torch.Tensor | str | None
        RGB channel selection strategy. Can be:
        - None: keep original channels
        - "random": randomly select 3 channels
        - "random_X_Y": randomly select from channels X to Y
        - "mean": take mean of three equal channel splits
        - "largest": select 3 channels with highest mean values
        - "pca": use PCA to extract 3 principal components
        - list/tuple: specific channel indices
    img : th.Tensor
        Input image tensor of shape (b, c, h, w) or (c, h, w)
    use_linstretch : bool
        Whether to apply linear stretching for contrast enhancement
    pca_fn : callable, optional
        Function to perform PCA transformation when rgb_channels="pca"

    Returns
    -------
    th.Tensor
        RGB image tensor of shape (b, 3, h, w) or (3, h, w)
    """
    if rgb_channels is not None and img.shape[1] > 3:
        assert img.shape[1] >= 3, "img must be hyperspectral images"
        if rgb_channels == "random":
            _rgb_chan_select = th.randperm(img.shape[1])[:3]
            rgb_channels = _rgb_chan_select.tolist()
            img = img[:, rgb_channels]
        elif (
            not isinstance(rgb_channels, (list, tuple))
            and isinstance(rgb_channels, str)
            and rgb_channels.startswith("random")
            and rgb_channels != "random"
        ):
            # e.g., random_5_12, means select 3 of channels from 5 to 12 channel index
            _lft_idx = rgb_channels.split("_")[1]
            _rgt_idx = rgb_channels.split("_")[2]

            _lft_idx = int(_lft_idx)
            _rgt_idx = int(_rgt_idx)

            assert _lft_idx < _rgt_idx, "rgb_channels must be in the range of [lft, rgt)"
            assert _rgt_idx < img.shape[1], "rgb_channels must be in the range of [lft, rgt)"
            _rgb_chan_select = th.randperm(_rgt_idx - _lft_idx)[:3] + _lft_idx
            rgb_channels = th.tensor([_rgb_chan_select[0], _rgb_chan_select[1], _rgb_chan_select[2]])
            img = img[:, rgb_channels]
        elif rgb_channels == "mean":
            # mean three splitted bands
            c = img.shape[1]
            c_3 = c // 3
            bands = [img[:, i * c_3 : (i + 1) * c_3, :, :].mean(dim=1) for i in range(3)]
            img = th.stack(bands, dim=1)
        elif rgb_channels == "largest":
            bands = choose_lightest_bands(img)
            img = img[:, bands]
        elif rgb_channels == "pca":
            if pca_fn is not None:
                img = pca_fn(img, 3)
            else:
                raise ValueError("pca_fn must be provided when rgb_channels='pca'")
        else:
            rgb_channels = cast(list, rgb_channels)
            img = img[:, rgb_channels]

    assert img.shape[1] == 3, "img must be rgb images"
    if use_linstretch:
        img = linstretch_torch(img)

    return img


def linstretch_torch(images: torch.Tensor, tol: list[float] | None = None, bins: int = 256) -> torch.Tensor:
    """Linear stretching for image contrast enhancement using PyTorch.

    This function provides a PyTorch-native implementation of linear stretching,
    which avoids CPU-GPU transfers and is more efficient for batch processing.

    Parameters
    ----------
    images : torch.Tensor
        Input image tensor of shape (b, c, h, w) or (c, h, w) or (h, w).
    tol : list[float], optional
        Tolerance values [min_percentile, max_percentile]. Defaults to [0.01, 0.995].
    bins : int, optional
        Number of histogram bins. Defaults to 256.

    Returns
    -------
    torch.Tensor
        Linear stretched image tensor with same shape as input.

    Raises
    ------
    ValueError
        If input tensor has invalid number of dimensions.
    """
    if tol is None:
        tol = [0.01, 0.995]

    # Handle different input shapes
    original_shape = images.shape
    if images.ndim == 2:
        # (h, w) -> (1, 1, h, w)
        images = images.unsqueeze(0).unsqueeze(0)
    elif images.ndim == 3:
        # (c, h, w) -> (1, c, h, w)
        images = images.unsqueeze(0)
    elif images.ndim != 4:
        raise ValueError(f"Input tensor must have 2, 3, or 4 dimensions, got {images.ndim}")

    batch_size, channels, height, width = images.shape
    total_pixels = height * width

    # Flatten spatial dimensions for each channel and batch
    flattened = images.view(batch_size, channels, -1)  # (b, c, h*w)

    # Convert to float32 for processing
    flattened = flattened.to(torch.float32)

    # Calculate min and max values for each channel and batch
    min_vals = flattened.amin(dim=-1, keepdim=True)  # (b, c, 1)
    max_vals = flattened.amax(dim=-1, keepdim=True)  # (b, c, 1)

    # Handle case where all values are the same
    range_vals = max_vals - min_vals
    range_vals = torch.where(range_vals == 0, torch.ones_like(range_vals), range_vals)

    # Calculate histogram for each channel and batch
    hist = torch.zeros(batch_size, channels, bins, device=images.device)

    for b in range(batch_size):
        for c in range(channels):
            channel_data = flattened[b, c]  # (h*w,)
            channel_min = min_vals[b, c].item()
            channel_max = max_vals[b, c].item()

            # Create histogram bins for this channel
            hist_edges = torch.linspace(channel_min, channel_max, bins + 1, device=images.device)  # (bins+1,)

            # Calculate histogram using torch.histc
            hist[b, c] = torch.histc(channel_data, bins=bins, min=channel_min, max=channel_max)

    # Calculate cumulative histogram
    cumsum_hist = torch.cumsum(hist, dim=-1)  # (b, c, bins)

    # Calculate bin centers for each channel
    bin_centers = torch.zeros(batch_size, channels, bins, device=images.device)
    for b in range(batch_size):
        for c in range(channels):
            channel_min = min_vals[b, c].item()
            channel_max = max_vals[b, c].item()
            hist_edges = torch.linspace(channel_min, channel_max, bins + 1, device=images.device)  # (bins+1,)
            bin_centers[b, c] = (hist_edges[:-1] + hist_edges[1:]) / 2  # (bins,)

    # Find percentile thresholds
    lower_threshold_idx = torch.argmax((cumsum_hist > total_pixels * tol[0]).float(), dim=-1)  # (b, c)
    upper_threshold_idx = (
        bins - 1 - torch.argmax((cumsum_hist.flip(-1) < total_pixels * tol[1]).float(), dim=-1)
    )  # (b, c)

    # Get threshold values
    lower_thresholds = torch.gather(bin_centers, -1, lower_threshold_idx.unsqueeze(-1)).squeeze(-1)  # (b, c)
    upper_thresholds = torch.gather(bin_centers, -1, upper_threshold_idx.unsqueeze(-1)).squeeze(-1)  # (b, c)

    # Reshape thresholds for broadcasting
    lower_thresholds = lower_thresholds.unsqueeze(-1)  # (b, c, 1)
    upper_thresholds = upper_thresholds.unsqueeze(-1)  # (b, c, 1)

    # Apply linear stretching
    stretched = torch.clamp(flattened, lower_thresholds, upper_thresholds)
    stretched = (stretched - lower_thresholds) / (upper_thresholds - lower_thresholds + 1e-8)

    # Reshape back to original dimensions
    result = stretched.view(original_shape)

    return result
