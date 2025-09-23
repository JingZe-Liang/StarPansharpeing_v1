import math
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from beartype import beartype
from beartype.door import is_bearable
from jaxtyping import Float, Float32, Int, UInt
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from numpy.typing import NDArray
from PIL import Image
from scipy.stats import entropy
from torch import Tensor
from torchvision.utils import make_grid

from ..config_utils import function_config_to_basic_types

# Typing aliases

type HyperImageType = Float[Tensor, "b c h w"] | Float[NDArray, "h w c"]

type GTMapType = (
    Int[NDArray, "h w"]
    | Int[Tensor, "b c h w"]
    | Int[Tensor, "b h w"]
    | UInt[NDArray, "h w"]
    | UInt[Tensor, "b c h w"]
    | UInt[Tensor, "b h w"]
)
type VisGTMapType = (
    Image.Image | list[Image.Image] | Float[NDArray, "b h w"] | Float[NDArray, "h w"]
)


RGB_CHANNELS_BY_BANDS = {
    4: [2, 1, 0],
    8: [4, 2, 0],
    10: [6, 5, 4],
    12: [3, 2, 1],
    13: [4, 3, 2],
    32: [12, 9, 3],
    50: "mean",
    150: "mean",  # [37, 28, 13],
    175: "mean",  # [42, 32, 13],
    191: [19, 12, 8],  # WDC mall
    202: "mean",  # [39, 32, 16],
    224: "mean",  # [39, 32, 16],
    242: "mean",  # [66, 40, 13],
    256: "mean",  # Xiongan
    270: "mean",
    368: "mean",  # [74, 42, 10],
    369: "mean",
    438: "mean",  # [62, 33, 19],
    439: "mean",
}


def get_coco_colors():
    COCO_CATEGORIES = np.array(
        [
            [220, 20, 60],
            [119, 11, 32],
            [0, 0, 142],
            [0, 0, 230],
            [106, 0, 228],
            [0, 60, 100],
            [0, 80, 100],
            [0, 0, 70],
            [0, 0, 192],
            [250, 170, 30],
            [100, 170, 30],
            [220, 220, 0],
            [175, 116, 175],
            [250, 0, 30],
            [165, 42, 42],
            [255, 77, 255],
            [0, 226, 252],
            [182, 182, 255],
            [0, 82, 0],
            [120, 166, 157],
            [110, 76, 0],
            [174, 57, 255],
            [199, 100, 0],
            [72, 0, 118],
            [255, 179, 240],
            [0, 125, 92],
            [209, 0, 151],
            [188, 208, 182],
            [0, 220, 176],
            [255, 99, 164],
            [92, 0, 73],
            [133, 129, 255],
            [78, 180, 255],
            [0, 228, 0],
            [174, 255, 243],
            [45, 89, 255],
            [134, 134, 103],
            [145, 148, 174],
            [255, 208, 186],
            [197, 226, 255],
            [171, 134, 1],
            [109, 63, 54],
            [207, 138, 255],
            [151, 0, 95],
            [9, 80, 61],
            [84, 105, 51],
            [74, 65, 105],
            [166, 196, 102],
            [208, 195, 210],
            [255, 109, 65],
            [0, 143, 149],
            [179, 0, 194],
            [209, 99, 106],
            [5, 121, 0],
            [227, 255, 205],
            [147, 186, 208],
            [153, 69, 1],
            [3, 95, 161],
            [163, 255, 0],
            [119, 0, 170],
            [0, 182, 199],
            [0, 165, 120],
            [183, 130, 88],
            [95, 32, 0],
            [130, 114, 135],
            [110, 129, 133],
            [166, 74, 118],
            [219, 142, 185],
            [79, 210, 114],
            [178, 90, 62],
            [65, 70, 15],
            [127, 167, 115],
            [59, 105, 106],
            [142, 108, 45],
            [196, 172, 0],
            [95, 54, 80],
            [128, 76, 255],
            [201, 57, 1],
            [246, 0, 122],
            [191, 162, 208],
        ]
    )
    COCO_CATEGORIES = np.concatenate(
        (COCO_CATEGORIES, np.ones((COCO_CATEGORIES.shape[0], 1))), axis=1
    )
    return COCO_CATEGORIES / 255.0


@beartype
def _choose_largest_bands(
    img: Float[Tensor, "... c h w"] | Float[NDArray, "h w c"],
) -> list[int]:
    if torch.is_tensor(img):
        mean_cs = img.view(-1, *img.shape[-3:]).mean((0, -2, -1)).detach().cpu()
    else:
        mean_cs = img.mean((0, 1))
    mean_cs = np.asarray(mean_cs)
    indices = np.argsort(mean_cs)[::-1][:3].tolist()
    assert indices[-1] < img.shape[1], (
        f"Invalid channel index {indices[-1]} for image with {img.shape[1]} channels."
    )
    return indices


def _calculate_band_entropy(band: torch.Tensor) -> float:
    """Calculate entropy of a single band for band selection."""
    # Convert to numpy and normalize to [0, 1] for entropy calculation
    band_np = band.cpu().numpy()
    band_np = (band_np - band_np.min()) / (band_np.max() - band_np.min() + 1e-10)

    # Calculate histogram
    hist, _ = np.histogram(band_np, bins=256, range=(0, 1))
    prob = hist / hist.sum()

    # Calculate entropy
    return entropy(prob)


def _select_bands_by_entropy(img: torch.Tensor, n_bands: int = 3) -> list[int]:
    """Select bands with highest entropy."""
    c = img.shape[1]
    entropies = torch.tensor(
        [_calculate_band_entropy(img[:, i, :, :]) for i in range(c)]
    )
    return torch.argsort(entropies, descending=True)[:n_bands].tolist()


def _select_bands_by_low_correlation(img: torch.Tensor, n_bands: int = 3) -> list[int]:
    """Select bands with lowest correlation to other bands."""
    c = img.shape[1]

    # Reshape to 2D for correlation calculation
    img_2d = img.permute(0, 2, 3, 1).reshape(-1, c)

    # Calculate correlation matrix
    correlation_matrix = torch.corrcoef(img_2d.T)

    # Handle NaN values (can occur with constant bands)
    correlation_matrix = torch.nan_to_num(correlation_matrix, nan=0.0)

    # Calculate correlation scores (sum of absolute correlations)
    correlation_scores = torch.sum(torch.abs(correlation_matrix), dim=1)

    # Select bands with lowest correlation scores
    return torch.argsort(correlation_scores)[:n_bands].tolist()


@function_config_to_basic_types
def get_rgb_image(
    img: torch.Tensor,
    rgb_channels: list[int] | str | None = None,
    use_linstretch: bool = False,
    linstretch_tol: list[float] | None = None,
):
    """
    Convert hyperspectral image to RGB format with optional linear stretching.

    Parameters
    ----------
    img : torch.Tensor
        Input hyperspectral image tensor of shape (b, c, h, w).
    rgb_channels : list[int] | str | None, optional
        Indices of channels to use for RGB visualization. If None, uses default RGB channels.
    use_linstretch : bool, optional
        Whether to apply linear stretching to enhance contrast. Defaults to False.
    linstretch_tol : list[float] | None, optional
        Tolerance values for linear stretching [min_percentile, max_percentile].
        If None, uses [0.01, 0.995]. Defaults to None.

    Returns
    -------
    torch.Tensor
        RGB image tensor of shape (b, 3, h, w).

    Raises
    ------
    ValueError
        If invalid RGB channels mapping is provided.
    """
    if (img.ndim == 4 and img.shape[1] == 1) or img.ndim == 3:
        # [b, 1, h, w] or [b, h, w]
        img = img.unsqueeze(1) if img.ndim == 3 else img
        img = img.repeat(1, 3, 1, 1)
        return img

    global RGB_CHANNELS_BY_BANDS

    c = img.shape[1]
    if c not in RGB_CHANNELS_BY_BANDS and rgb_channels is None:
        raise ValueError(
            f"Invalid number of channels: {c}. Expected one of {list(RGB_CHANNELS_BY_BANDS.keys())}"
        )

    rgb_channels = rgb_channels or RGB_CHANNELS_BY_BANDS[c]
    if isinstance(rgb_channels, (list, tuple)):
        rgb_img = img[:, rgb_channels, :, :]
    elif rgb_channels == "mean":
        # split three parts
        c_3 = c // 3
        bands = [img[:, i * c_3 : (i + 1) * c_3, :, :].mean(dim=1) for i in range(3)]
        rgb_img = torch.stack(bands, dim=1)
    elif rgb_channels == "largest":
        indices = _choose_largest_bands(img)
        rgb_img = img[:, indices, :, :]
    elif rgb_channels == "std":
        stds = img.var(dim=(1, 2, 3))
        index = torch.argsort(stds, descending=True)[:3]
        rgb_img = img[:, index, :, :]
    elif rgb_channels == "entropy":
        indices = _select_bands_by_entropy(img)
        rgb_img = img[:, indices, :, :]
    elif rgb_channels == "low_correlation":
        indices = _select_bands_by_low_correlation(img)
        rgb_img = img[:, indices, :, :]
    else:
        raise ValueError(
            f"Invalid RGB channels mapping: {rgb_channels}. Expected list, tuple, 'mean', 'largest', 'std', 'entropy', or 'low_correlation'."
        )

    # Apply linear stretching if requested
    if use_linstretch:
        rgb_img = linstretch_torch(rgb_img, tol=linstretch_tol)

    return rgb_img


@beartype
@function_config_to_basic_types
def visualize_hyperspectral_image(
    img: HyperImageType,
    to_pil=False,
    norm=True,
    rgb_channels: list[int] | str | None = None,
    nrows: int = 1,
    to_grid=False,
    to_uint8=True,
    use_linstretch: bool = False,
    linstretch_tol: list[float] | None = None,
) -> UInt[NDArray, "h w c"] | Float[NDArray, "h w c"] | list[Image.Image] | Image.Image:
    """Visualize a hyperspectral image by converting it to RGB format.

    Args:
        img: Hyperspectral image tensor or array of shape (b, c, h, w) or (h, w, c).
        to_pil: Whether to convert the output to PIL Image format. Defaults to False.
        norm: Whether to normalize the image to [0, 1] range. Defaults to True.
        rgb_channels: Indices of channels to use for RGB visualization. If None, uses default RGB channels.
        nrows: Number of rows when creating a grid of images. Defaults to 1.
        to_grid: Whether to arrange images in a grid. Defaults to False.
        to_uint8: Whether to convert output to uint8 format. Defaults to True.
        use_linstretch: Whether to apply linear stretching to enhance contrast. Defaults to False.
        linstretch_tol: Tolerance values for linear stretching [min_percentile, max_percentile].
            If None, uses [0.01, 0.995]. Defaults to None.

    Returns:
        If to_pil is True, returns PIL Image or list of PIL Images.
        If to_pil is False, returns numpy array of shape (h, w, 3).
        When returning a grid, returns a single image with multiple rows.
    """
    if isinstance(img, np.ndarray):
        assert img.ndim == 3, f"Invalid image shape: {img.shape}. Expected (h, w, c)."
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1, c, h, w)
    else:
        assert img.ndim == 4, (
            f"Invalid image shape: {img.shape}. Expected (b, c, h, w)."
        )
        img = img.detach()

    rgb_img = get_rgb_image(
        img, rgb_channels, use_linstretch=use_linstretch, linstretch_tol=linstretch_tol
    )  # (b, 3, h, w)
    if norm:
        rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
        rgb_img = torch.clamp(rgb_img, 0, 1)

    if to_grid:
        rgb_img = make_grid(rgb_img, nrow=nrows)  # (c, h, w)

    if to_pil:
        rgb_img = rgb_img.mul(255).to(torch.uint8).cpu()
        pil_imgs = [
            Image.fromarray(im.permute(1, 2, 0).numpy(), mode="RGB") for im in rgb_img
        ]
        return pil_imgs[0] if len(pil_imgs) == 1 else pil_imgs
    else:
        rgb_img = rgb_img.permute(1, 2, 0)  # (h, w, c)
        if to_uint8:
            rgb_img = rgb_img.mul(255).to(torch.uint8)
        rgb_img = rgb_img.numpy()
        return rgb_img


@beartype
@function_config_to_basic_types
def visualize_segmentation_map(
    gt_map: GTMapType,
    cmap: str = "tab20",
    n_class: int = 20,
    bg_black=True,
    colors: Float32[NDArray, "n_class 4"] | list[list[int | float]] | None = None,
    use_coco_colors=False,
    alpha: float = 1.0,
    to_rgba=False,
    to_pil=False,
    add_channel_dim_1=False,
) -> VisGTMapType:
    if colors is None:
        if use_coco_colors:
            colors = get_coco_colors()
        else:
            colors = np.array(plt.get_cmap(cmap)(np.linspace(0, 1, n_class)))

    # Convert 0-class to be black
    if bg_black:
        alpha_bg = 1.0
        colors[0] = [0, 0, 0, alpha_bg]

    custom_cmap = ListedColormap(colors)
    custom_cmap.colors = custom_cmap.colors * np.array([1, 1, 1, alpha])
    assert n_class <= custom_cmap.N, f"n_class {n_class} > cmap.N {custom_cmap.N}"

    norm = BoundaryNorm(boundaries=np.arange(0, n_class), ncolors=n_class)
    if torch.is_tensor(gt_map):
        # (b, c, h, w)
        gt_map.squeeze_(1)
        assert gt_map.ndim == 3, f"Invalid gt_map shape: {gt_map.shape}"

        gt_map = gt_map.unbind(0)
        gt_map = [g.cpu().numpy() for g in gt_map]
    else:
        gt_map = [gt_map]

    # Convert to RGB/RGBA color maps
    ms = []
    gt_map = cast(list[np.ndarray], gt_map)  # type: ignore
    for m in gt_map:
        m = norm(m)
        m = custom_cmap(m)
        if not to_rgba:
            m = m[..., :3]
        if to_pil:
            mode = "RGBA" if to_rgba else "RGB"
            m = Image.fromarray((m * 255).astype(np.uint8), mode=mode)
        ms.append(m)

    if to_pil:
        return ms[0] if len(ms) == 1 else ms
    else:
        ms = np.stack(ms).squeeze(0)
        if add_channel_dim_1:
            if ms.ndim == 2:
                ms = ms[None, ...]
            elif ms.ndim == 3:
                ms = ms[:, None]
        return ms


@beartype
@function_config_to_basic_types
def visualize_data_range_bins(
    data: Tensor | np.ndarray,
    nbins: int = 100,
    title: str = "Data Distribution",
    figsize: tuple[int, int] = (10, 6),
    return_fig: bool = False,
    save_path: str | None = None,
    log_scale: bool = False,
    show_stats: bool = True,
):
    """Visualize data distribution using histogram bins.

    Args:
        data: Input data tensor or numpy array
        nbins: Number of histogram bins (default: 100)
        title: Plot title (default: "Data Distribution")
        figsize: Figure size (default: (10, 6))
        return_fig: Whether to return the figure object (default: False)
        save_path: Path to save the plot (default: None)
        log_scale: Whether to use log scale for y-axis (default: False)
        show_stats: Whether to show statistics on the plot (default: True)

    Returns:
        If return_fig is True, returns matplotlib Figure object
        Otherwise, displays the plot
    """
    # Convert to tensor and flatten
    data = torch.asarray(data).flatten()

    # Remove NaN and Inf values
    data = data[torch.isfinite(data)]

    if data.numel() == 0:
        raise ValueError("No valid data points after removing NaN/Inf values")

    # Convert to numpy for plotting
    data_np = data.cpu().numpy()

    # Calculate statistics
    data_min = float(data.min().item())
    data_max = float(data.max().item())
    data_mean = float(data.mean().item())
    data_std = float(data.std().item())
    data_median = float(data.median().item())

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create histogram
    if log_scale:
        hist, bins, patches = ax.hist(
            data_np, bins=nbins, alpha=0.7, edgecolor="black", log=True
        )
    else:
        hist, bins, patches = ax.hist(data_np, bins=nbins, alpha=0.7, edgecolor="black")

    # Set labels and title
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency" + (" (log scale)" if log_scale else ""))
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Add statistics text
    if show_stats:
        stats_text = (
            f"Min: {data_min:.4f}\n"
            f"Max: {data_max:.4f}\n"
            f"Mean: {data_mean:.4f}\n"
            f"Std: {data_std:.4f}\n"
            f"Median: {data_median:.4f}\n"
            f"Count: {len(data_np)}"
        )
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    # Add vertical lines for mean and median
    ax.axvline(
        data_mean,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Mean: {data_mean:.4f}",
    )
    ax.axvline(
        data_median,
        color="green",
        linestyle="--",
        alpha=0.7,
        label=f"Median: {data_median:.4f}",
    )
    ax.legend()

    plt.tight_layout()

    # Save plot if path provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    # Return or show
    if return_fig:
        return fig
    else:
        plt.show()
        plt.close()


@beartype
@function_config_to_basic_types
def visualize_batch_comparisons_imgs(
    *x: HyperImageType,
    rgb_channels: list[int] | str | None = None,
    norm: bool = False,
    spacing: int = 5,
    to_uint8: bool = True,
    to_pil: bool = False,
    to_grid: bool = True,
    use_linstretch: bool = False,
    linstretch_tol: list[float] | None = None,
) -> UInt[NDArray, "h w c"] | Float[NDArray, "h w c"] | Image.Image | list[Image.Image]:
    """Visualize multiple hyperspectral images by concatenating them horizontally.

    This function takes multiple hyperspectral images with the same batch size,
    converts them to RGB, and concatenates them horizontally along the width axis.

    Parameters
    ----------
    *x : HyperImageType
        Variable number of hyperspectral image tensors or arrays.
        All images should have the same batch size.
    rgb_channels : list[int] | str | None, optional
        Indices of channels to use for RGB visualization.
        If None, uses default RGB channels for each image.
    norm : bool, optional
        Whether to normalize each image to [0, 1] range. Defaults to False.
    spacing : int, optional
        Number of pixels to insert between images as spacing. Defaults to 5.
    to_uint8 : bool, optional
        Whether to convert output to uint8 format. Defaults to True.
    to_pil : bool, optional
        Whether to convert output to PIL Image format. Defaults to False.
    to_grid : bool, optional
        Whether to arrange batch images in a grid layout. If False, returns
        individual images. Only applies when batch_size > 1. Defaults to True.
    use_linstretch : bool, optional
        Whether to apply linear stretching to enhance contrast. Defaults to False.
    linstretch_tol : list[float] | None, optional
        Tolerance values for linear stretching [min_percentile, max_percentile].
        If None, uses [0.01, 0.995]. Defaults to None.

    Returns
    -------
    UInt[NDArray, "h w c"] | Float[NDArray, "h w c"] | Image.Image | list[Image.Image]
        Concatenated RGB image as numpy array, PIL Image, or list of images.

    Raises
    ------
    ValueError
        If input images have different batch sizes or heights.
    """
    if not x:
        raise ValueError("At least one image must be provided")

    # Convert all inputs to tensors and validate shapes
    tensors = []
    batch_size = None
    height = None

    for img in x:
        if isinstance(img, np.ndarray):
            assert img.ndim == 3, (
                f"Invalid image shape: {img.shape}. Expected (h, w, c)."
            )
            img_tensor = (
                torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            )  # (1, c, h, w)
        else:
            assert img.ndim == 4, (
                f"Invalid image shape: {img.shape}. Expected (b, c, h, w)."
            )
            img_tensor = img.detach()

        # Validate consistent batch size and height
        current_batch = img_tensor.shape[0]
        current_height = img_tensor.shape[2]

        if batch_size is None:
            batch_size = current_batch
            height = current_height
        else:
            if current_batch != batch_size:
                raise ValueError(
                    f"All images must have the same batch size. "
                    f"Found {batch_size} and {current_batch}."
                )
            if current_height != height:
                raise ValueError(
                    f"All images must have the same height. "
                    f"Found {height} and {current_height}."
                )

        tensors.append(img_tensor)

    # Ensure batch_size and height are not None
    assert batch_size is not None, "batch_size cannot be None after processing"
    assert height is not None, "height cannot be None after processing"

    # Convert each tensor to RGB
    rgb_images = []
    for tensor in tensors:
        rgb_img = get_rgb_image(
            tensor,
            rgb_channels,
            use_linstretch=use_linstretch,
            linstretch_tol=linstretch_tol,
        )  # (b, 3, h, w)
        if norm:
            rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
            rgb_img = torch.clamp(rgb_img, 0, 1)
        rgb_images.append(rgb_img)

    # Concatenate images horizontally for each batch item
    concatenated_batch = []
    for batch_idx in range(batch_size):
        batch_images = [
            rgb_img[batch_idx] for rgb_img in rgb_images
        ]  # List of (3, h, w)

        # Calculate total width
        total_width = sum(img.shape[2] for img in batch_images) + spacing * (
            len(batch_images) - 1
        )

        # Create concatenated image
        concatenated_img = torch.ones(3, height, total_width)

        current_width = 0
        for i, img in enumerate(batch_images):
            img_width = img.shape[2]
            concatenated_img[:, :, current_width : current_width + img_width] = img
            current_width += img_width

            # Add spacing (except after last image)
            if i < len(batch_images) - 1:
                concatenated_img[:, :, current_width : current_width + spacing] = (
                    0.5  # Gray spacing
                )
                current_width += spacing

        concatenated_batch.append(concatenated_img)

    # Stack batch items
    final_tensor = torch.stack(concatenated_batch, dim=0)  # (b, 3, h, total_w)

    # Convert to output format
    if to_uint8:
        final_tensor = final_tensor.mul(255).to(torch.uint8)

    # Handle output based on batch size and parameters
    if batch_size == 1:
        # Single image case
        result = final_tensor[0].permute(1, 2, 0).numpy()  # (h, total_w, 3)

        if to_pil:
            return Image.fromarray(result)
        else:
            return result
    else:
        # Batch case
        if to_grid:
            # Arrange images in a grid
            grid_tensor = make_grid(
                final_tensor,
                nrow=int(np.ceil(np.sqrt(batch_size))),
                padding=spacing,
                normalize=False,
            )

            if to_pil:
                grid_array = grid_tensor.permute(1, 2, 0).numpy()
                return Image.fromarray(grid_array)
            else:
                return grid_tensor.permute(1, 2, 0).numpy()
        else:
            # Return individual images
            if to_pil:
                return [
                    Image.fromarray(img.permute(1, 2, 0).numpy())
                    for img in final_tensor
                ]
            else:
                return final_tensor.permute(0, 2, 3, 1).numpy()  # (b, h, total_w, 3)


def linstretch(images, tol=None):
    """Linear stretching for image contrast enhancement using NumPy.

    Parameters
    ----------
    images : np.ndarray
        Input image or images of shape (h, w) or (h, w, c).
    tol : list[float], optional
        Tolerance values [min_percentile, max_percentile]. Defaults to [0.01, 0.995].

    Returns
    -------
    np.ndarray
        Linear stretched image(s).
    """
    if tol is None:
        tol = [0.01, 0.995]
    if images.ndim == 3:
        h, w, channels = images.shape
    else:
        images = np.expand_dims(images, axis=-1)
        h, w, channels = images.shape
    N = h * w
    for c in range(channels):
        image = np.float32(np.round(images[:, :, c])).reshape(N, 1)
        # Handle case where all values are the same
        if image.max() == image.min():
            # If all values are the same, return zeros
            images[..., c] = np.zeros((h, w))
            continue

        hb, levelb = np.histogram(
            image, bins=max(1, math.ceil(image.max() - image.min()))
        )
        chb = np.cumsum(hb, 0)
        levelb_center = levelb[:-1] + (levelb[1] - levelb[0]) / 2

        # Handle edge cases for threshold selection
        lower_mask = chb > N * tol[0]
        upper_mask = chb < N * tol[1]

        if not np.any(lower_mask):
            lbc_min = levelb_center[0]
        else:
            lbc_min = levelb_center[lower_mask][0]

        if not np.any(upper_mask):
            lbc_max = levelb_center[-1]
        else:
            lbc_max = levelb_center[upper_mask][-1]
        image = np.clip(image, a_min=lbc_min, a_max=lbc_max)
        # Handle division by zero
        if lbc_max == lbc_min:
            image = np.zeros_like(image)
        else:
            image = (image - lbc_min) / (lbc_max - lbc_min)
        images[..., c] = np.reshape(image, (h, w))

    images = np.squeeze(images)

    return images


def linstretch_torch(
    images: torch.Tensor, tol: list[float] | None = None, bins: int = 256
) -> torch.Tensor:
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
        raise ValueError(
            f"Input tensor must have 2, 3, or 4 dimensions, got {images.ndim}"
        )

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
            hist_edges = torch.linspace(
                channel_min, channel_max, bins + 1, device=images.device
            )  # (bins+1,)

            # Calculate histogram using torch.histc
            hist[b, c] = torch.histc(
                channel_data, bins=bins, min=channel_min, max=channel_max
            )

    # Calculate cumulative histogram
    cumsum_hist = torch.cumsum(hist, dim=-1)  # (b, c, bins)

    # Calculate bin centers for each channel
    bin_centers = torch.zeros(batch_size, channels, bins, device=images.device)
    for b in range(batch_size):
        for c in range(channels):
            channel_min = min_vals[b, c].item()
            channel_max = max_vals[b, c].item()
            hist_edges = torch.linspace(
                channel_min, channel_max, bins + 1, device=images.device
            )  # (bins+1,)
            bin_centers[b, c] = (hist_edges[:-1] + hist_edges[1:]) / 2  # (bins,)

    # Find percentile thresholds
    lower_threshold_idx = torch.argmax(
        (cumsum_hist > total_pixels * tol[0]).float(), dim=-1
    )  # (b, c)
    upper_threshold_idx = (
        bins
        - 1
        - torch.argmax((cumsum_hist.flip(-1) < total_pixels * tol[1]).float(), dim=-1)
    )  # (b, c)

    # Get threshold values
    lower_thresholds = torch.gather(
        bin_centers, -1, lower_threshold_idx.unsqueeze(-1)
    ).squeeze(-1)  # (b, c)
    upper_thresholds = torch.gather(
        bin_centers, -1, upper_threshold_idx.unsqueeze(-1)
    ).squeeze(-1)  # (b, c)

    # Reshape thresholds for broadcasting
    lower_thresholds = lower_thresholds.unsqueeze(-1)  # (b, c, 1)
    upper_thresholds = upper_thresholds.unsqueeze(-1)  # (b, c, 1)

    # Apply linear stretching
    stretched = torch.clamp(flattened, lower_thresholds, upper_thresholds)
    stretched = (stretched - lower_thresholds) / (
        upper_thresholds - lower_thresholds + 1e-8
    )

    # Reshape back to original dimensions
    result = stretched.view(original_shape)

    return result
