import os
from pathlib import Path

import numpy as np
import tifffile

from src.utilities.logging import log_print


def background_is_all_zero(
    img,
    norm=False,
    ratio: float = 0.7,
    thresh: int = 5,
):
    img = img.astype(np.float32)
    if norm:
        img = img.clip(0, None)
        img = img / img.max() * 255.0

    # [h, w, c]
    img_gray = img  # .mean(-1)
    pix_total = img.shape[0] * img.shape[1]
    pix_zero = np.sum(img_gray < thresh)  # 0 .. 255

    ratio = pix_zero / pix_total
    return ratio > ratio


BANDS_NAME = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B10",
    "B11",
    "B12",
]


def extract_all_band_paths(dir: str):
    lst = Path(dir).glob("*B01.tif")
    broken_uni_count = 0
    for p in lst:
        name = p.name  # data/HLS/HLS.L30.T58UEF.2025152T235555.v2.0.B01.tif
        uni_tif_paths = []
        for band_name in BANDS_NAME:
            name_rep = name.replace("B01", band_name)
            band_path = p.parent / name_rep
            no_band_name = ".".join(p.name.rsplit(".")[:-2])
            if not band_path.exists():
                log_print(f"Band path does not exist: {band_path}", "warning")
                related_paths = list(p.parent.glob(no_band_name + "*"))
                log_print(f"related paths:", "warning")
                for related_p in related_paths:
                    log_print(related_p.as_posix(), "warning")
                broken_uni_count += 1
                break
            uni_tif_paths.append(band_path)
        yield (uni_tif_paths, broken_uni_count)


def bands_reading_and_filter(path: list[str]):
    bands_imgs = []
    for p in path:
        try:
            img = tifffile.imread(p)
        except Exception as e:
            log_print(f"failed to load image from: {p}. {e}", "warning")
            return None
        img = np.clip(img, 0, None)  # clip negative values
        if background_is_all_zero(img, norm=True):
            log_print(f"Background is all zero: {p}", "warning")
            return None
        bands_imgs.append(img.astype(np.uint16))
        # print(img.shape)
    img = np.stack(bands_imgs, axis=-1)
    return img


def img_to_tif(img: np.ndarray, path: str) -> None:
    """
    Save the image to a TIFF file.
    :param img: The image to save.
    :param path: The path to save the image to.
    """
    tifffile.imwrite(
        path,
        img,
        compression="jpeg2000",
        compressionargs={"level": 90, "reversible": False},
        maxworkers=8,
    )
    log_print(f"Image saved to {path}", "info")
    sz = os.path.getsize(path)
    log_print(f"Image size: {sz / 1024 / 1024:.2f} MB", "info")


if __name__ == "__main__":
    total = 0
    saved = 0
    for uni_tif_paths, b in extract_all_band_paths("data/HLS"):
        # print(uni_tif_paths)
        # print(f"Broken uni tif count: {b}")
        # total += 1
        # print(f"Total: {total}, Broken uni tif count: {b}")

        img = bands_reading_and_filter(uni_tif_paths)
        # save
        path_to_save = Path(
            uni_tif_paths[0].as_posix().replace(".B01.tif", ".tif")
        ).name
        path_to_save = Path("data/HLS/tiff") / path_to_save
        path_to_save.parent.mkdir(parents=True, exist_ok=True)

        if img is not None:
            img_to_tif(img, path_to_save.as_posix())
            saved += 1
        if saved % 10 == 0:
            log_print(f"Saved {saved} images so far.", "info")
