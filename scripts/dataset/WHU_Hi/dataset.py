from pathlib import Path

import numpy as np
import scipy.io as sio
import tifffile
import torch
from loguru import logger
from PIL import Image
from scipy.io import loadmat

from src.stage1.utilities.losses.repa.feature_pca import (
    feature_pca_sk,
    feature_pca_torch,
)
from src.utilities.train_utils.visualization import visualize_segmentation_map


def read_image(
    img_path: str | Path,
    *,
    mat_load_key="I",
    verbose=False,
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
            logger.warning(
                f"Mat file is not supported by scipy.io.loadmat reading: {e}. Try to "
                "read using h5py."
            )
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
        basic_name = (
            img_path.name
        )  # data/HLS/HLS.L30.T58UEF.2025152T235555.v2.0.B01.tif
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
            img = (
                tif.asarray(out="memmap")
                if tiff_read_mode == "memmap"
                else tif.asarray()
            )
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
                data = (
                    frame["data"].numpy().transpose(1, 2, 0)
                )  # [c, h, w] -> [h, w, c]
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


def list_files(dir, pattern, recursive=False):
    dir = Path(dir)
    if recursive:
        paths = dir.rglob(pattern)
    else:
        paths = dir.glob(pattern)
    return list(paths)


def visualize_dataset_maps(dir, pattern="*gt.mat"):
    files = list_files(dir, pattern)
    for f in files:
        save_map_path = f.with_name(f.stem + "_map.png")
        img = read_image(f)
        img = img.astype("int16")
        print(Path(f).name, img.shape)
        m = visualize_segmentation_map(img, use_coco_colors=False, to_pil=True)
        m.save(save_map_path)
        print(f"save map at {save_map_path}")


def pca_dataset(dir, pattern="*.mat"):
    files = list_files(dir, pattern)
    for f in files:
        save_dir = f.parent / "pca_images"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_map_path = save_dir / (f.stem + "_map.png")
        save_raw_img_path = (f.parent / "images" / f.name).with_suffix(".png")
        save_raw_img_path.parent.mkdir(parents=True, exist_ok=True)

        # image
        img = read_image(f).astype("float32")
        print(f"Image shaped as {img.shape}")
        img -= img.min()
        img /= img.max()
        img = torch.as_tensor(img).permute(-1, 0, 1)[None]

        # pca
        # pca_img = feature_pca_torch(img, 3)
        # pca_img = pca_img[0].permute(1, 2, 0).cpu().numpy()
        # Image.fromarray((pca_img * 255.0).astype("uint8"), mode="RGB").save(
        #     save_map_path
        # )
        # print(f"save PCA image at {save_map_path}")

        # save image
        img = img[0].permute(1, 2, 0).cpu().numpy()
        img = img[..., [29, 19, 9]]
        img = (img * 255.0).astype("uint8")
        Image.fromarray(img, mode="RGB").save(save_raw_img_path)
        print(f"save image at {save_raw_img_path}")


if __name__ == "__main__":
    # visualize_dataset_maps(
    #     "/Data4/cao/ZiHanCao/exps/HyperspectralTokenizer/data/HyperSpectral-Collections/cls/cls_GT"
    # )

    pca_dataset(
        "/Data4/cao/ZiHanCao/exps/HyperspectralTokenizer/data/HyperSpectral-Collections/cls/mat"
    )
