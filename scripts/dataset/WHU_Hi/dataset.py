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
from src.utilities.io import read_image
from src.utilities.train_utils.visualization import visualize_segmentation_map


def list_files(dir, pattern, recursive=False):
    dir = Path(dir)
    if recursive:
        paths = dir.rglob(pattern)
    else:
        paths = dir.glob(pattern)
    return list(paths)


def visualize_dataset_maps(dir, pattern="*gt.mat"):
    files = list_files(dir, pattern)
    print(f"Found {len(files)} GT maps")
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
    visualize_dataset_maps("data/Downstreams/ClassificationCollection/cls/cls_GT")

    # pca_dataset("data/Downstreams/ClassificationCollection/cls/mat")
