import os
from pathlib import Path

import numpy as np
import tifffile
import torch
import webdataset as wds
from skimage.transform import rescale, resize
from tqdm import tqdm

from src.data.codecs import tiff_codec_io
from src.utilities.io import read_image
from src.utilities.logging import catch_any, print_info_if_raise

bands = [
    "B01.tif",
    "B02.tif",
    "B03.tif",
    "B04.tif",
    "B05.tif",
    "B06.tif",
    "B07.tif",
    "B08.tif",
    "B8A.tif",
    "B09.tif",
    "B10.tif",
    "B11.tif",
    "B12.tif",
]
base_dir = "data/Downstreams/ChangeDetection/OSCD/Onera Satellite Change Detection dataset - Images"
gt_name = "data/Downstreams/ChangeDetection/OSCD/Onera Satellite Change Detection dataset - Train Labels"


def read_bands(img_dir, bands=bands):
    return np.stack(
        [read_image(os.path.join(img_dir, band)) for band in bands], axis=-1
    )  # (h, w, c)  # type: ignore


# @print_info_if_raise()
def read_dir_img(img_dir, use_rgb=False):
    if not use_rgb:
        img_names = ["imgs_1_rect", "imgs_2_rect"]
        img1 = os.path.join(img_dir, img_names[0])
        img2 = os.path.join(img_dir, img_names[1])
        img1 = read_bands(img1)
        img2 = read_bands(img2)
    else:
        img1 = read_image(os.path.join(img_dir, "pair", "img1.png"))
        img2 = read_image(os.path.join(img_dir, "pair", "img2.png"))

    scene_name = Path(img_dir).name
    gt_path = Path(gt_name, scene_name, "cm") / "cm.png"
    gt = read_image(gt_path).astype(np.long)
    return img1, img2, gt


# img1, img2, gt = read_dir_img(
#     "data/Downstreams/ChangeDetection/OSCD/Onera Satellite Change Detection dataset - Images/abudhabi"
# )
# print(img1.shape, img2.shape, gt.shape)


@catch_any()
def main(use_rgb=False):
    writter = wds.TarWriter(
        "data/Downstreams/ChangeDetection/OSCD/OSCD_3bands_train.tar"
    )

    tbar = tqdm(os.listdir(base_dir))
    for sub_dir in tbar:
        if os.path.isdir(os.path.join(base_dir, sub_dir)):
            # print(sub_dir)
            # print(img1.shape, img2.shape, gt.shape)
            try:
                img1, img2, gt = read_dir_img(
                    os.path.join(base_dir, sub_dir), use_rgb=use_rgb
                )
            except Exception as e:
                print(f"Error in {sub_dir}: {e}")
                continue

            tbar.set_postfix(dict(dir=sub_dir, img_size=tuple(img1.shape)))
            if not use_rgb:
                imgs_dict = {
                    "img1.tif": tiff_codec_io(img1, compression="lzw"),
                    "img2.tif": tiff_codec_io(img2, compression="lzw"),
                }
            else:
                imgs_dict = {
                    "img1.png": img1,
                    "img2.png": img2,
                }
            writter.write(
                {
                    "__key__": sub_dir,
                    **imgs_dict,
                    "gt.npy": gt,
                }
            )
    writter.close()

    print("Done!")


if __name__ == "__main__":
    main(use_rgb=True)
