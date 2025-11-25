from enum import Enum
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import rasterio
import spectral
import tifffile
import torch
from PIL import Image
from scipy.io import loadmat
from torchvision.io import video_reader

from ..logging import log


class ExtractMode(Enum):
    LAST = "last"
    ALL = "all"
    NOT_META = "not_meta"
    SPECIFIC = "specific"


def dict_extract(
    d: dict | h5py.File,
    keys: list[str] | None = None,
    extract_type: ExtractMode = ExtractMode.LAST,
    force_load: bool = True,
) -> dict | Any:
    etype = extract_type.value if isinstance(extract_type, ExtractMode) else extract_type
    force_load_fn = lambda x: x[:] if force_load else x

    if keys is None:
        if etype == "not_meta":
            ret = {k: force_load_fn(v) for k, v in d.items() if not k.startswith("__")}
        elif etype == "all":
            ret = d
        elif etype == "last":
            k = list(d.keys())[-1]
            ret = force_load_fn(d[k])
        else:
            raise Exception(f"Invalid extract_type: {etype} when keys is None")
    else:
        assert etype == "specific", f"extract_type must be 'specific' when keys is provided, got {etype}"
        ret = {}
        d_keys = list(d.keys())
        for k in keys:
            if k in d_keys:
                ret[k] = force_load_fn(d[k])

    if isinstance(ret, dict):
        if len(ret) == 0:
            log(f"No valid keys found in the dict, available keys: {list(d.keys())}")
            return None
        if len(ret) == 1:
            return list(ret.values())[0]

    return ret


def read_image(
    img_path: str | Path,
    *,
    key_if_dict: list[str] | str | None = None,
    dict_extract_type: ExtractMode = ExtractMode.LAST,
    verbose=False,
    tiff_read_mode="array",
    tiff_bands_seperated: bool = False,
    force_to_dtype: np.dtype | None = None,
    bands_name: list[str] | None = None,
    mat_ret_all=False,
) -> np.ndarray | dict[str, np.ndarray] | np.memmap | list[np.ndarray] | None:
    if isinstance(img_path, str):
        img_path = Path(img_path)
    if isinstance(key_if_dict, str):
        key_if_dict = [key_if_dict]
    if key_if_dict is not None:
        assert not mat_ret_all, "mat_ret_all is not supported when key_if_dict is not None"
        dict_extract_type = ExtractMode.SPECIFIC

    if verbose:
        log(f"reading image from: {img_path.as_posix()}")

    if img_path.suffix == ".mat":
        try:
            d = loadmat(img_path)
            return dict_extract(d, key_if_dict, dict_extract_type)
        except Exception as e:
            log(
                f"Mat file is not supported by scipy.io.loadmat reading: {e}. Try to read using h5py.",
                level="warning",
            )

            with h5py.File(img_path, "r") as f:
                d = dict_extract(f, key_if_dict, dict_extract_type)
                return d
    elif img_path.suffix.lower() == ".img":
        with rasterio.open(img_path) as dataset:
            bands = dataset.count
            img = np.zeros((dataset.height, dataset.width, bands), dtype=dataset.dtypes[0])
            for i in range(bands):
                img[..., i] = dataset.read(i + 1)
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
        basic_name = img_path.name  # data/HLS/HLS.L30.T58UEF.2025152T235555.v2.0.B01.tif
        uni_tif_paths = []

        assert bands_name is not None, "bands_name should be provided when tiff_bands_seperated is True"
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
    elif img_path.suffix in [".dat", ".hdr"]:
        # is envi file
        if img_path.suffix == ".hdr":
            hdr_file = str(img_path)
            img_file = str(img_path.with_suffix(".dat"))
        else:
            hdr_file = str(img_path.with_suffix(".hdr"))
            img_file = img_path
        if not Path(hdr_file).exists() or not Path(img_file).exists():
            log(f"envi file pair not found: {hdr_file}, {img_file}", "warning")
            return None
        img = spectral.envi.open(hdr_file, img_file).load()
    elif img_path.suffix.lower() in [".tif", ".tiff"]:
        # memmap
        tif = tifffile.TiffFile(img_path)
        try:
            img = tif.asarray(out="memmap") if tiff_read_mode == "memmap" else tif.asarray()
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
                data = frame["data"].numpy().transpose(1, 2, 0)  # [c, h, w] -> [h, w, c]
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


if __name__ == "__main__":
    path = "data/Downstreams/HAD/HAD100/data/aviris_normal/f080709t01p00r15_1.dat"
    img = read_image(path, verbose=True)
    print(img.shape, type(img))
