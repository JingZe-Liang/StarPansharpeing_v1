from pathlib import Path

import h5py
import numpy as np
import rasterio
import tifffile
import torch
from PIL import Image
from scipy.io import loadmat
from torchvision.io import video_reader

from ..logging import log


def read_image(
    img_path: str | Path,
    *,
    mat_load_key="I",
    verbose=False,
    tiff_read_mode="array",
    tiff_bands_seperated: bool = False,
    force_to_dtype: np.dtype | None = None,
    bands_name: list[str] | None = None,
    mat_ret_all=False,
) -> np.ndarray | dict[str, np.ndarray] | np.memmap | list[np.ndarray] | None:
    if isinstance(img_path, str):
        img_path = Path(img_path)

    if verbose:
        log("reading image from: {}", img_path.as_posix())

    if img_path.suffix == ".mat":
        try:
            d = loadmat(img_path)
            if not mat_ret_all:
                key_ = list(d.keys())[-1]
                img = d[key_]
                return img
            else:
                d_ret = {k: v for k, v in d.items() if not k.startswith("__")}
                return d_ret
        except Exception as e:
            log(
                f"Mat file is not supported by scipy.io.loadmat reading: {e}. Try to "
                "read using h5py.",
                level="warning",
            )

            with h5py.File(img_path, "r") as f:
                if not mat_ret_all:
                    key_ = list(f.keys())[-1]
                    img = f[key_][:]
                    return img
                else:
                    d_ret = {k: v[:] for k, v in f.items()}
                    return d_ret
    elif img_path.suffix.lower() == ".img":
        with rasterio.open(img_path) as dataset:
            bands = dataset.count
            img = np.zeros(
                (dataset.height, dataset.width, bands), dtype=dataset.dtypes[0]
            )
            for i in range(bands):
                img[..., i] = dataset.read(i + 1)
            return img

    elif img_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
        try:
            img = np.array(Image.open(img_path))
            # may be mask
            if img.ndim == 2 and img_path.suffix.lower() == ".png":
                img = img[..., None]
        except Exception as e:
            log(f"Failed to load image from: {img_path.as_posix()}. {e}", "warning")
            return None
    elif img_path.suffix == ".npy":
        img = np.load(img_path)
    elif img_path.suffix.lower() in [".tif", ".tiff"] and tiff_bands_seperated:
        # assume img_path endswith *B01.tif
        basic_name = (
            img_path.name
        )  # data/HLS/HLS.L30.T58UEF.2025152T235555.v2.0.B01.tif
        uni_tif_paths = []

        assert bands_name is not None, (
            "bands_name should be provided when tiff_bands_seperated is True"
        )
        for band_name in bands_name:
            parts = basic_name.split(".")  # replace B01 with band_name
            name = ".".join(parts[:-2]) + "." + band_name + ".tif"
            band_path = img_path.parent / name
            if not band_path.exists():
                log(f"band {band_name} not found in: {img_path}", "warning")
                return None
            uni_tif_paths.append(band_path)
        # read all bands
        bands_imgs = []
        for p in uni_tif_paths:
            try:
                img = tifffile.imread(p)
            except Exception as e:
                log(f"failed to load image from: {p}. {e}", level="warning")
                return None
            img = np.clip(img, 0, None)
            bands_imgs.append(img)
        img = np.stack(bands_imgs, axis=-1)  # [h, w, c]
    elif img_path.suffix.lower() in [".tif", ".tiff"]:
        # memmap
        tif = tifffile.TiffFile(img_path)
        try:
            img = (
                tif.asarray(out="memmap")
                if tiff_read_mode == "memmap"
                else tif.asarray()
            )
        except Exception as e:
            log(
                f"failed to load image from: {img_path.as_posix()}. {e}",
                level="warning",
            )
            return None
    elif img_path.suffix.lower() in [".mp4"]:
        # read video frame
        reader = video_reader.VideoReader(img_path.as_posix(), num_threads=2)
        frames = []
        for frame in reader:
            pts = frame["pts"]
            if pts % 0.5 <= 0.001:
                data = (
                    frame["data"].numpy().transpose(1, 2, 0)
                )  # [c, h, w] -> [h, w, c]
                frames.append(data)

            if pts > 60:
                log("the frames are too many, stop reading", level="warning")
        return frames
    else:
        raise ValueError(f"Unsupported image format: {img_path.suffix}")

    if force_to_dtype is not None:
        try:
            assert isinstance(img, np.ndarray), "img must be a numpy array"
            img = img.astype(force_to_dtype)
        except Exception as e:
            log(
                f"Failed to convert image to {force_to_dtype} dtype: {e}. "
                "Please check the image data type and the target dtype.",
                level="warning",
            )

    return img
