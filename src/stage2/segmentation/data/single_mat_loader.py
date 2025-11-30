from pathlib import Path
from typing import Dict, List, Tuple, cast

import numpy as np
import torch as th
from easydict import EasyDict as edict
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from src.utilities.io import read_image

from .robust_sampler import RobustHyperspectralSampler, sample_img_with_gt_indices


class SingleMatDataset(Dataset):
    task: str = "classification"
    paths: edict = edict(
        {
            "Houston13": {"img": "mat/Houston13_hwc", "gt": "cls_GT/Houston13_7gt.mat"},
            "Houston18": {"img": "mat/Houston18_hwc", "gt": "cls_GT/Houston18_7gt.mat"},
            "Indian_pines": {"img": "mat/Indian_pines_corrected.mat", "gt": "cls_GT/Indian_pines_gt.mat"},
            "Pavia": {"img": "mat/Pavia.mat", "gt": "cls_GT/Pavia_gt.mat"},
            "PaviaU": {"img": "mat/PaviaU.mat", "gt": "cls_GT/PaviaU_gt.mat"},
            "WHU_Hi_HanChuan": {"img": "mat/WHU_Hi_HanChuan.mat", "gt": "cls_GT/WHU_Hi_HanChuan_gt.mat"},
            "WHU_Hi_HongHu": {"img": "mat/WHU_Hi_HongHu.mat", "gt": "cls_GT/WHU_Hi_HongHu_gt.mat"},
            "WHU_Hi_LongKou": {"img": "mat/WHU_Hi_LongKou.mat", "gt": "cls_GT/WHU_Hi_LongKou_gt.mat"},
        }
    )

    def __init__(
        self,
        data_dir: str,
        dataset_name: str,
        train_ratio: float = 0.2,
        samples_per_cls: int = 5,
        patch_size: int = 32,
    ):
        self.train_ratio = train_ratio
        self.dataset_name = dataset_name
        path = self.paths[dataset_name]

        img_path = Path(data_dir, path.img)
        gt_path = Path(data_dir, path.gt)

        self.img: np.ndarray = read_image(img_path)
        self.gt: np.ndarray = read_image(gt_path)
        self.sampler = RobustHyperspectralSampler(
            self.gt, balance_strategy="equal_size", random_seed=2025, verbose=True
        )

        # Samples
        self.sample_indices_train = self.sampler.sample(samples_per_class=samples_per_cls)
        patches_indices, label_indices, self.unsampled_area = sample_img_with_gt_indices(
            self.img, self.gt, self.sample_indices_train, patch_size=patch_size
        )

        # Patches flatten
        self.patches_indices, self.label_indices = [], []
        for img_p in patches_indices.values():
            self.patches_indices += img_p
        for gt_p in label_indices.values():
            self.label_indices += gt_p
        logger.info(f"{self.dataset_name} train patches: {len(self.patches_indices)}")
        assert len(self.patches_indices) == len(self.label_indices), (
            f"{self.dataset_name} train patches and labels are not equal"
        )

    def __len__(self) -> int:
        return len(self.patches_indices)

    def __getitem__(self, idx: int):
        patch = self.patches_indices[idx].transpose(-1, 0, 1)
        label = self.label_indices[idx]
        return {"img": patch, "gt": label}


def _test_single_mat_loader():
    img_dir = "data/Downstreams/ClassificationCollection/cls"

    ds = SingleMatDataset(
        img_dir,
        "WHU_Hi_LongKou",
        train_ratio=0.2,
        samples_per_cls=5,
        patch_size=64,
    )

    train_loader = DataLoader(ds, batch_size=1, shuffle=True)
    for batch in train_loader:
        print(batch["img"].shape)
        print(batch["gt"].shape)


if __name__ == "__main__":
    """
    python -m src.stage2.segmentation.data.single_mat_loader
    """
    _test_single_mat_loader()
