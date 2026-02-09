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
    if convert_bg:
        # Convert background (0) to 255, but preserve existing 255 (padding)
        # This ensures padding values remain 255 and don't become 254
        fixed_padded_label = where_fn(label == 255, 255, label - 1)  # type: ignore[no-matching-overload]
        result = where_fn(label == 0, 255, fixed_padded_label)  # type: ignore[no-matching-overload]
    else:
        result = label
    return result


def label_background_recover(label: Tensor | np.ndarray, convert_bg: bool = True):
    where_fn = th.where if th.is_tensor(label) else np.where
    result = where_fn(label == 255, 0, label + 1) if convert_bg else label  # type: ignore[no-matching-overload]
    return result


def get_default_transform(p=0.5):
    """Get default augmentation transform for hyperspectral images."""
    transform = AugmentationSequential(
        RandomHorizontalFlip(p=p),
        RandomVerticalFlip(p=p),
        RandomRotation(degrees=90, p=p, align_corners=False),
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
            "XiongAn": {"img": "mat/XiongAn.img", "gt": "cls_GT/farm_roi.img"},
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
        online_slice: bool = False,
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
        if self.gt.ndim == 3 and self.gt.shape[-1] == 1:
            self.gt = self.gt[..., 0]
        assert self.gt.ndim == 2, f"GT must be 2D, got shape {self.gt.shape}"

        self.full_img = full_img
        self.skip_bg = skip_bg
        self.n_classes = np.unique(self.gt).shape[0] - (1 if skip_bg else 0)

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
        self.online_slice = online_slice
        self.patch_size = patch_size

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
            # If online slice is requested, we don't pre-extract patches into memory;
            # instead, we keep the center coordinates and will extract patches in __getitem__.
            if self.online_slice:
                # Build list of valid coordinates and unsampled area mask
                margin = patch_size // 2
                self.patches_coords = []
                unsampled_area = np.zeros_like(self.gt)
                height, width = self.gt.shape

                # Cache padded image and gt ONCE to avoid repeated allocation
                self.padded_img = np.pad(
                    self.img, ((margin, margin), (margin, margin), (0, 0)), mode="constant", constant_values=0
                )
                self.padded_gt = np.pad(
                    self.gt, ((margin, margin), (margin, margin)), mode="constant", constant_values=255
                )

                for class_id, coords in self.sample_indices_train.items():
                    for coord in coords:
                        c = np.asarray(coord)
                        cx, cy = int(c[0]), int(c[1])
                        # ensure valid center
                        if cx < margin or cx >= height - margin or cy < margin or cy >= width - margin:
                            continue
                        self.patches_coords.append((int(cx), int(cy)))
                        # mark unsampled area
                        x0, x1 = cx - margin, cx + margin + 1
                        y0, y1 = cy - margin, cy + margin + 1
                        unsampled_area[x0:x1, y0:y1] = 1
                # unsampled_area means True is unsampled
                self.unsampled_area = np.logical_not(unsampled_area.astype(np.bool_))
                # No pre-extracted patches
                self.patches_indices, self.label_indices = [], []
            else:
                patches_indices, label_indices, self.unsampled_area, patches_coords = sample_img_with_gt_indices(
                    self.img, self.gt, cast(Dict[int, np.ndarray], self.sample_indices_train), patch_size=patch_size
                )
                # Flatten as before
                self.patches_indices, self.label_indices = [], []
                self.patches_coords = []
                for img_p in patches_indices.values():
                    self.patches_indices += img_p
                for gt_p in label_indices.values():
                    self.label_indices += gt_p
                for coords_p in patches_coords.values():
                    self.patches_coords += coords_p
            logger.info(f"Dataset sampled done.")
            self.unsampled_area = th.as_tensor(self.unsampled_area)

            # We already created / flattened patches or coords above
            if self.online_slice:
                logger.info(f"{self.dataset_name} train patches (online coords): {len(self.patches_coords)}")
            else:
                logger.info(f"{self.dataset_name} train patches: {len(self.patches_indices)}")
                assert len(self.patches_indices) == len(self.label_indices), (
                    f"{self.dataset_name} train patches and labels are not equal"
                )

    def __len__(self) -> int:
        return len(self.patches_coords) if self.online_slice else len(self.patches_indices)

    def _extract_patch_at_coord(self, coord: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """Extract a single image patch and label patch centered at `coord`, using cached padding."""
        cx, cy = coord
        patch_size = self.patch_size
        padded_x = int(cx)
        padded_y = int(cy)
        slices = slice(padded_x, padded_x + patch_size), slice(padded_y, padded_y + patch_size)
        patch = self.padded_img[slices[0], slices[1], :]
        label = self.padded_gt[slices[0], slices[1]]
        return patch, label

    def _label_post_process(self, label: np.ndarray | Tensor) -> np.ndarray | Tensor:
        """Convert 0 background to 255"""
        return label_background_convert(label, self.skip_bg)

    def _get_full_img_tensor(self):
        img = th.as_tensor(self.img).permute(-1, 0, 1)
        label = th.as_tensor(self._label_post_process(self.gt)).type(th.long)
        return edict({"img": img, "gt": label})

    def _get_patch_img_tensor(self, idx: int):
        if self.online_slice:
            center = self.patches_coords[idx]
            patch_arr, label_arr = self._extract_patch_at_coord(center)
            patch = th.as_tensor(patch_arr.transpose(-1, 0, 1))
            label = th.as_tensor(label_arr).type(th.long)
        else:
            patch = th.as_tensor(self.patches_indices[idx].transpose(-1, 0, 1))
            label = th.as_tensor(self.label_indices[idx]).type(th.long)
        label = self._label_post_process(label)

        # Resize
        # patch, label = self.resize_fn(patch, label)
        # print(patch.shape, label.shape)

        # Transform
        if self.transform is not None:
            patch, label = self.transform(patch, label)

        # Optionally include the center coordinate when using online slicing
        if self.online_slice:
            return edict({"img": patch, "gt": label, "coord": self.patches_coords[idx]})
        return edict({"img": patch, "gt": label})

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
    from src.utilities.train_utils.visualization import get_rgb_image, visualize_segmentation_map

    img_dir = "data/Downstreams/ClassificationCollection/cls"

    ds = SingleMatDataset(img_dir, "XiongAn", train_ratio=0.4, samples_per_cls=10, patch_size=128, online_slice=True)
    print("Dataset length:", len(ds))

    # full_img = th.as_tensor((ds.img + 1) / 2)[None].permute(0, -1, 1, 2)
    # full_gt = ds.gt
    # full_img_vis = get_rgb_image(full_img, "mean", True)
    # full_gt_vis = visualize_segmentation_map(full_gt, n_class=ds.n_classes, to_pil=False)

    train_loader = DataLoader(ds, batch_size=4, shuffle=True)
    for batch in train_loader:
        img_vis = get_rgb_image((batch.img + 1) / 2, "mean", True)
        gt = label_background_recover(batch.gt)
        gt_vis = th.as_tensor(visualize_segmentation_map(gt, n_class=ds.n_classes, to_pil=False)).permute(0, -1, 1, 2)

        print(batch["img"].shape, "min: {}, max: {}".format(batch["img"].min(), batch["img"].max()))
        print(batch["gt"].shape, batch["gt"].unique())


if __name__ == "__main__":
    """
    python -m src.stage2.segmentation.data.single_mat_loader
    """
    _test_single_mat_loader()
