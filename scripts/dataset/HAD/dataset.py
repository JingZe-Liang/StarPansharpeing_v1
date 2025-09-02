from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.utilities.io import read_image
from src.utilities.train_utils.visualization import (
    visualize_hyperspectral_image,
    visualize_segmentation_map,
)

HAD_dataset_paths = {
    "bay_area": [
        "/HardDisk/ZiHanCao/datasets/Hyperspectral-ChangeDetection-Collection/ChangeDetectionDataset/bayArea/mat/Bay_Area_2013.mat",
        "/HardDisk/ZiHanCao/datasets/Hyperspectral-ChangeDetection-Collection/ChangeDetectionDataset/bayArea/mat/Bay_Area_2015.mat",
        "/HardDisk/ZiHanCao/datasets/Hyperspectral-ChangeDetection-Collection/ChangeDetectionDataset/bayArea/mat/bayArea_gtChanges2.mat.mat",
    ],
    "hermiston": [
        "/HardDisk/ZiHanCao/datasets/Hyperspectral-ChangeDetection-Collection/ChangeDetectionDataset/Hermiston/hermiston2004.mat",
        "/HardDisk/ZiHanCao/datasets/Hyperspectral-ChangeDetection-Collection/ChangeDetectionDataset/Hermiston/hermiston2007.mat",
        "/HardDisk/ZiHanCao/datasets/Hyperspectral-ChangeDetection-Collection/ChangeDetectionDataset/Hermiston/rdChangesHermiston_5classes.mat",
    ],
    "santa_barbara": [
        "/HardDisk/ZiHanCao/datasets/Hyperspectral-ChangeDetection-Collection/ChangeDetectionDataset/santaBarbara/mat/barbara_2013.mat",
        "/HardDisk/ZiHanCao/datasets/Hyperspectral-ChangeDetection-Collection/ChangeDetectionDataset/santaBarbara/mat/barbara_2014.mat",
        "/HardDisk/ZiHanCao/datasets/Hyperspectral-ChangeDetection-Collection/ChangeDetectionDataset/santaBarbara/mat/barbara_gtChanges.mat",
    ],
}


def visualize_had_image(path1, path2, gt_path):
    d1 = read_image(path1)
    d2 = read_image(path2)
    gt = read_image(gt_path)

    print(d1.shape, d1.min(), d1.max())
    print(d2.shape, d2.min(), d2.max())
    print(gt.shape)

    d1_vis = visualize_hyperspectral_image(d1, to_pil=True, rgb_channels=[39, 29, 19])
    d2_vis = visualize_hyperspectral_image(d2, to_pil=True, rgb_channels=[39, 29, 19])
    gt_vis = visualize_segmentation_map(gt, use_coco_colors=False, to_pil=True)

    # Save all PIL Images
    d1_vis.save(f"tmp/{Path(path1).stem}.png")
    d2_vis.save(f"tmp/{Path(path2).stem}.png")
    gt_vis.save(f"tmp/{Path(gt_path).stem}.png")


if __name__ == "__main__":
    visualize_had_image(*HAD_dataset_paths["santa_barbara"])
