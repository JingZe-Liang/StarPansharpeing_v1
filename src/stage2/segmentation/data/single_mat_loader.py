from pathlib import Path
from typing import Callable, Dict, List, Literal, Tuple, cast

import numpy as np
import torch as th
from easydict import EasyDict as edict
from kornia.augmentation import AugmentationSequential, RandomHorizontalFlip, RandomRotation, RandomVerticalFlip, Resize
from loguru import logger
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.data.utils import normalize_image_ as norm
from src.utilities.config_utils import function_config_to_easy_dict
from src.utilities.io import read_image

from .robust_sampler import RobustHyperspectralSampler, sample_img_with_gt_indices


def label_background_convert(label: Tensor | np.ndarray, convert_bg: bool = True):
    where_fn = th.where if th.is_tensor(label) else np.where
    result = where_fn(label == 0, 255, label - 1) if convert_bg else label
    return result


def label_background_recover(label: Tensor | np.ndarray, convert_bg: bool = True):
    where_fn = th.where if th.is_tensor(label) else np.where
    result = where_fn(label == 255, 0, label + 1) if convert_bg else label
    return result


def get_default_transform(p=0.5):
    """Get default augmentation transform for hyperspectral images."""
    transform = AugmentationSequential(
        RandomHorizontalFlip(p=p),
        RandomVerticalFlip(p=p),
        RandomRotation(degrees=90, p=p),
        data_keys=["input", "mask"],
        same_on_batch=False,
        keepdim=True,
    )
    return transform


class SingleMatDataset(Dataset):
    """
    Hyperspectral Collection Dataset,
    loading one single mat file to train an classification model.
    """

    task: str = "classification"
    paths: edict = edict(
        {
            "Houston13": {"img": "mat/Houston13_hwc.mat", "gt": "cls_GT/Houston13_7gt.mat"},
            "Houston18": {"img": "mat/Houston18_hwc.mat", "gt": "cls_GT/Houston18_7gt.mat"},
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
        full_img: bool = False,
        skip_bg: bool = True,
        to_neg_1_1: bool = True,
        transform: Callable | None | Literal["default"] = None,
    ):
        self.train_ratio = train_ratio
        self.dataset_name = dataset_name
        assert dataset_name in list(self.paths.keys()), (
            f"Dataset {dataset_name} not in supported datasets: {list(self.paths.keys())}"
        )
        path = self.paths[dataset_name]

        img_path = Path(data_dir, path.img)
        gt_path = Path(data_dir, path.gt)

        self.img: np.ndarray = read_image(img_path)
        self.gt: np.ndarray = read_image(gt_path)

        self.full_img = full_img
        self.skip_bg = skip_bg

        logger.info(f"Load image name: {dataset_name}, shaped {self.img.shape}")
        logger.info(f"Load gt, labels class has number: {np.unique(self.gt)}")

        # Normalize to [0, 1]
        self.to_neg_1_1 = to_neg_1_1
        self.img = (self.img - self.img.min()) / (self.img.max() - self.img.min())
        if self.to_neg_1_1:
            self.img = self.img * 2.0 - 1.0

        # Transform
        if transform == "default":
            transform = get_default_transform()
        self.transform = transform
        self.resize_fn = AugmentationSequential(
            Resize(patch_size, keepdim=True), data_keys=["input", "mask"], keepdim=True
        )

        if not full_img:
            self.sampler = RobustHyperspectralSampler(
                self.gt,
                balance_strategy="equal_size",
                skip_bg=0 if skip_bg else None,  # skip_bg is int type, 0 is background class
                random_seed=2025,
                verbose=True,
            )

            # Samples
            self.sample_indices_train = self.sampler.sample(samples_per_class=samples_per_cls)
            patches_indices, label_indices, self.unsampled_area = sample_img_with_gt_indices(
                self.img, self.gt, self.sample_indices_train, patch_size=patch_size
            )
            self.unsampled_area = th.as_tensor(self.unsampled_area)

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

    def _label_post_process(self, label: np.ndarray) -> np.ndarray:
        """Convert 0 background to 255"""
        return label_background_convert(label, self.skip_bg)

    def _get_full_img_tensor(self):
        img = th.as_tensor(self.img).permute(-1, 0, 1)
        label = th.as_tensor(self._label_post_process(self.gt)).type(th.long)
        return {"img": img, "gt": label}

    def _get_patch_img_tensor(self, idx: int):
        patch = th.as_tensor(self.patches_indices[idx].transpose(-1, 0, 1))
        label = th.as_tensor(self.label_indices[idx]).type(th.long)

        # Resize
        # patch, label = self.resize_fn(patch, label)
        # print(patch.shape, label.shape)

        # Transform
        if self.transform is not None:
            patch, label = self.transform(patch, label)

        return {"img": patch, "gt": label}

    def __getitem__(self, idx: int):
        if self.full_img:
            return self._get_full_img_tensor()
        else:
            return self._get_patch_img_tensor(idx)

    def get_unsampled_area(self):
        return self.unsampled_area.unsqueeze(0) if not self.full_img else None

    @classmethod
    @function_config_to_easy_dict
    def from_config(cls, cfg):
        ds = cls(**cfg.dataset_kwargs)
        dl_cfg = cfg.dataloader_kwargs
        dl = DataLoader(
            ds,
            batch_size=dl_cfg.batch_size,
            num_workers=dl_cfg.num_workers,
            shuffle=dl_cfg.get("shuffle", False),
            pin_memory=dl_cfg.get("pin_memory", True),
            drop_last=dl_cfg.get("drop_last", False),
        )
        return ds, dl


def _test_single_mat_loader():
    img_dir = "data/Downstreams/ClassificationCollection/cls"

    ds = SingleMatDataset(
        img_dir,
        "WHU_Hi_HongHu",
        train_ratio=0.2,
        samples_per_cls=5,
        patch_size=64,
    )

    train_loader = DataLoader(ds, batch_size=4, shuffle=True)
    for batch in train_loader:
        print(batch["img"].shape, "min: {}, max: {}".format(batch["img"].min(), batch["img"].max()))
        print(batch["gt"].shape)


if __name__ == "__main__":
    """
    python -m src.stage2.segmentation.data.single_mat_loader
    """
    _test_single_mat_loader()
