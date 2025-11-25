import glob
import io
import math
import sys
from pathlib import Path
from typing import Iterable, Literal, cast

import einops
import matplotlib.pyplot as plt
import natsort
import numpy as np
import safetensors
import safetensors.torch
import tifffile
import torch
import webdataset as wds
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from PIL import Image
from scipy.io import loadmat
from torchvision.io import read_video
from tqdm import tqdm

from src.utilities.logging.print import configure_logger, log, logger


def read_image(
    img_path: str | Path,
    *,
    mat_load_key="I",
    verbose=True,
    tiff_read_mode="array",
    tiff_bands_seperated: bool = False,
    force_to_dtype: np.dtype | None = None,
):
    if isinstance(img_path, str):
        img_path = Path(img_path)

    if verbose:
        logger.info("reading image from: {}", img_path.as_posix())

    if img_path.suffix == ".mat":
        try:
            d = loadmat(img_path)
            key_ = list(d.keys())[-1]
            img = d[key_]
        except NotImplementedError as e:
            logger.warning(f"Mat file is not supported by scipy.io.loadmat reading: {e}. Try to read using h5py.")
            import h5py

            with h5py.File(img_path, "r") as f:
                key_ = list(f.keys())[-1]
                img = f[key_][:]
    elif img_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
        try:
            img = np.array(Image.open(img_path))
            # may be mask
            if img.ndim == 2 and img_path.suffix.lower() == ".png":
                img = img[..., None]
        except Exception as e:
            logger.warning(f"Failed to load image from: {img_path.as_posix()}. {e}")
            return None
    elif img_path.suffix == ".npy":
        img = np.load(img_path)
    elif img_path.suffix.lower() in [".tif", ".tiff"] and tiff_bands_seperated:
        # assume img_path endswith *B01.tif
        basic_name = img_path.name  # data/HLS/HLS.L30.T58UEF.2025152T235555.v2.0.B01.tif
        uni_tif_paths = []

        for band_name in BANDS_NAME:
            parts = basic_name.split(".")  # replace B01 with band_name
            name = ".".join(parts[:-2]) + "." + band_name + ".tif"
            band_path = img_path.parent / name
            if not band_path.exists():
                logger.warning(f"band {band_name} not found in: {img_path}")
                return None
            uni_tif_paths.append(band_path)
        # read all bands
        bands_imgs = []
        for p in uni_tif_paths:
            try:
                img = tifffile.imread(p)
            except Exception as e:
                logger.warning(f"failed to load image from: {p}. {e}")
                return None
            img = np.clip(img, 0, None)
            bands_imgs.append(img)
        img = np.stack(bands_imgs, axis=-1)  # [h, w, c]
    elif img_path.suffix.lower() in [".tif", ".tiff"]:
        # memmap
        tif = tifffile.TiffFile(img_path)
        try:
            img = tif.asarray(out="memmap") if tiff_read_mode == "memmap" else tif.asarray()
        except Exception as e:
            logger.warning(f"failed to load image from: {img_path.as_posix()}. {e}")
            return None
    elif img_path.suffix.lower() in [".mp4"]:
        from torchvision.io import video_reader

        # read video frame
        reader = video_reader.VideoReader(img_path.as_posix(), num_threads=2)
        frames = []
        for frame in reader:
            pts = frame["pts"]
            if pts % 0.5 <= 0.001:
                data = frame["data"].numpy().transpose(1, 2, 0)  # [c, h, w] -> [h, w, c]
                frames.append(data)

            if pts > 60:
                logger.warning("the frames are too many, stop reading")
        return frames
    else:
        raise ValueError(f"Unsupported image format: {img_path.suffix}")

    if force_to_dtype is not None:
        try:
            assert isinstance(img, np.ndarray), "img must be a numpy array"
            img = img.astype(force_to_dtype)
        except Exception as e:
            logger.warning(
                f"Failed to convert image to {force_to_dtype} dtype: {e}. "
                "Please check the image data type and the target dtype."
            )

    return img


def to_color(gt_map: np.ndarray, cmap="tab10"):
    assert gt_map.ndim == 2
    assert gt_map.dtype == np.int32

    base_cmap = plt.get_cmap(cmap)
    colors = base_cmap(np.linspace(0, 1, 10))
    colors[0] = [0, 0, 0, 1]
    custom_cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries=np.arange(0, 8), ncolors=10)
    gt_rgb = custom_cmap(norm(gt_map))
    Image.fromarray((gt_rgb[:, :, :3] * 255).astype(np.uint8))


def hyper_to_rgb(img: np.ndarray | torch.Tensor, rgb_bands: list[int]):
    """Convert hyperspectral image to RGB image using specified bands.

    Args:
        img (np.ndarray or torch.Tensor): Input hyperspectral image of shape (H, W, C) or (C, H, W).
        rgb_bands (list[int]): List of three band indices to use for R, G, B channels.

    Returns:
        np.ndarray: RGB image of shape (H, W, 3) with values in the range [0, 255].
    """
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()

    if img.ndim == 3 and img.shape[0] < img.shape[2]:
        # Assume shape is (C, H, W)
        img = einops.rearrange(img, "c h w -> h w c")

    assert img.ndim == 3 and img.shape[2] > 3, "Input image must be HWC with more than 3 channels"
    assert len(rgb_bands) == 3, "rgb_bands must contain exactly three band indices"

    # Clip and normalize each channel
    rgb_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.float32)
    for i, band in enumerate(rgb_bands):
        band_data = img[:, :, band]
        band_data = np.clip(band_data, 0, None)  # Clip negative values
        band_data = (band_data - band_data.min()) / (band_data.max() - band_data.min() + 1e-8)  # Normalize to [0, 1]
        rgb_img[:, :, i] = band_data

    rgb_img = (rgb_img * 255).astype(np.uint8)  # Scale to [0, 255]
    return rgb_img
