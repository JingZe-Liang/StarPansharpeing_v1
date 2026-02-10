from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from kornia.augmentation import AugmentationSequential, RandomHorizontalFlip, RandomVerticalFlip, RandomResizedCrop
from kornia.constants import DataKey
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

SUPPORTED_IMAGE_EXTS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def get_default_transform(prob: float = 0.5) -> AugmentationSequential:
    return AugmentationSequential(
        RandomHorizontalFlip(p=prob),
        RandomVerticalFlip(p=prob),
        data_keys=[DataKey.INPUT, DataKey.INPUT, DataKey.MASK],
        same_on_batch=True,
        keepdim=True,
    )


def _list_images(folder: Path) -> list[Path]:
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS]
    return sorted(files)


def _to_base_id(stem: str) -> str:
    return stem.replace("_pre_disaster", "").replace("_post_disaster", "")


@dataclass(frozen=True)
class XView2SamplePaths:
    img1: Path
    img2: Path
    gt: Path
    name: str


def _find_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _resolve_pair_and_mask(
    pre_path: Path,
    images_dir: Path,
    targets_dir: Path | None,
    labels_dir: Path | None,
) -> XView2SamplePaths | None:
    stem = pre_path.stem
    base_id = _to_base_id(stem)
    post_stem = stem.replace("_pre_disaster", "_post_disaster")

    img2 = _find_existing([images_dir / f"{post_stem}{ext}" for ext in SUPPORTED_IMAGE_EXTS])
    if img2 is None:
        return None

    mask_candidates: list[Path] = []
    for root in (targets_dir, labels_dir):
        if root is None:
            continue
        mask_candidates.extend(
            [root / f"{post_stem}_target{ext}" for ext in SUPPORTED_IMAGE_EXTS]
            + [root / f"{base_id}_target{ext}" for ext in SUPPORTED_IMAGE_EXTS]
            + [root / f"{post_stem}{ext}" for ext in SUPPORTED_IMAGE_EXTS]
            + [root / f"{base_id}{ext}" for ext in SUPPORTED_IMAGE_EXTS]
        )
    gt = _find_existing(mask_candidates)
    if gt is None:
        return None

    return XView2SamplePaths(
        img1=pre_path,
        img2=img2,
        gt=gt,
        name=base_id,
    )


def _collect_split_samples(split_root: Path) -> list[XView2SamplePaths]:
    images_dir = split_root / "images"
    targets_dir = split_root / "targets"
    labels_dir = split_root / "labels"

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images dir: {images_dir}")
    targets_dir = targets_dir if targets_dir.is_dir() else None
    labels_dir = labels_dir if labels_dir.is_dir() else None
    if targets_dir is None and labels_dir is None:
        raise FileNotFoundError(f"Neither targets nor labels dir exists in: {split_root}")

    pre_images = [p for p in _list_images(images_dir) if "_pre_disaster" in p.stem]
    if not pre_images:
        raise RuntimeError(f"No '*_pre_disaster' image found in: {images_dir}")

    samples: list[XView2SamplePaths] = []
    missed = 0
    for pre_path in pre_images:
        sample = _resolve_pair_and_mask(
            pre_path=pre_path,
            images_dir=images_dir,
            targets_dir=targets_dir,
            labels_dir=labels_dir,
        )
        if sample is None:
            missed += 1
            continue
        samples.append(sample)

    if not samples:
        raise RuntimeError(f"No valid xView2 pairs with masks found in: {split_root}")
    if missed > 0:
        print(f"[xView2 Dataset] skipped {missed} unmatched samples in split '{split_root.name}'.")
    return samples


def _load_rgb(path: Path) -> Tensor:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(arr).float().permute(2, 0, 1)


def _load_mask(path: Path, binarize_gt: bool) -> Tensor:
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.uint8).copy()
    if binarize_gt:
        arr = (arr > 0).astype(np.uint8)
    return torch.from_numpy(arr).long().unsqueeze(0)


class XView2ChangeDetectionDataset(Dataset[dict[str, Tensor | str]]):
    def __init__(
        self,
        data_root: str | Path = "data/Downstreams/xView2_raw",
        split: Literal["train", "test"] = "train",
        include_hold_in_train: bool = True,
        transform: AugmentationSequential | None | str = "default",
        normalize: bool = True,
        img_to_neg1_1: bool = False,
        resize_to_mask: bool = True,
        resize_crop_to: int | None = 512,
        resize_crop_scale: tuple[float, float] = (0.8, 1.0),
        resize_crop_ratio: tuple[float, float] = (0.75, 1.3333333333333333),
        binarize_gt: bool = True,
        return_name: bool = False,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.normalize = normalize
        self.img_to_neg1_1 = img_to_neg1_1
        self.resize_to_mask = resize_to_mask
        self.resize_crop_to = resize_crop_to
        self.resize_crop_scale = resize_crop_scale
        self.resize_crop_ratio = resize_crop_ratio
        self.binarize_gt = binarize_gt
        self.return_name = return_name
        self.resize_crop_transform: AugmentationSequential | None = None

        if transform == "default":
            transform = get_default_transform()
        self.transform = transform
        if self.resize_crop_to is not None:
            out_size = int(self.resize_crop_to)
            self.resize_crop_transform = AugmentationSequential(
                RandomResizedCrop(
                    size=(out_size, out_size),
                    scale=self.resize_crop_scale,
                    ratio=self.resize_crop_ratio,
                    resample="BILINEAR",
                    same_on_batch=True,
                    align_corners=False,
                    p=1.0,
                    keepdim=True,
                ),
                data_keys=[DataKey.INPUT, DataKey.INPUT, DataKey.MASK],
                same_on_batch=True,
                keepdim=True,
            )

        split_dirs: list[Path]
        if split == "train":
            split_dirs = [self.data_root / "train"]
            if include_hold_in_train:
                split_dirs.append(self.data_root / "hold")
        else:
            split_dirs = [self.data_root / "test"]

        samples: list[XView2SamplePaths] = []
        for split_dir in split_dirs:
            if not split_dir.is_dir():
                raise FileNotFoundError(f"Missing split dir: {split_dir}")
            samples.extend(_collect_split_samples(split_dir))

        if not samples:
            raise RuntimeError(f"No xView2 samples found for split={split} under {self.data_root}")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        sample = self.samples[index]
        img1 = _load_rgb(sample.img1)
        img2 = _load_rgb(sample.img2)
        gt = _load_mask(sample.gt, binarize_gt=self.binarize_gt)

        if self.resize_to_mask and img1.shape[-2:] != gt.shape[-2:]:
            target_h, target_w = gt.shape[-2:]
            img1 = torch.nn.functional.interpolate(
                img1.unsqueeze(0), size=(target_h, target_w), mode="bilinear", align_corners=False
            ).squeeze(0)
            img2 = torch.nn.functional.interpolate(
                img2.unsqueeze(0), size=(target_h, target_w), mode="bilinear", align_corners=False
            ).squeeze(0)

        if self.resize_crop_transform is not None:
            img1, img2, gt = self.resize_crop_transform(
                img1.unsqueeze(0),
                img2.unsqueeze(0),
                gt.float().unsqueeze(0),
            )
            img1 = img1.squeeze(0)
            img2 = img2.squeeze(0)
            gt = gt.squeeze(0).round().long()

        if self.normalize:
            img1 = img1 / 255.0
            img2 = img2 / 255.0
        if self.img_to_neg1_1:
            img1 = img1 * 2.0 - 1.0
            img2 = img2 * 2.0 - 1.0

        if self.transform is not None:
            img1, img2, gt = self.transform(
                img1.unsqueeze(0),
                img2.unsqueeze(0),
                gt.float().unsqueeze(0),
            )
            img1 = img1.squeeze(0)
            img2 = img2.squeeze(0)
            gt = gt.squeeze(0).long()

        out: dict[str, Tensor | str] = {"img1": img1, "img2": img2, "gt": gt}
        if self.return_name:
            out["name"] = sample.name
        return out


def create_xview2_change_detection_dataloader(
    data_root: str | Path = "data/Downstreams/xView2_raw",
    split: Literal["train", "test"] = "train",
    include_hold_in_train: bool = True,
    batch_size: int = 4,
    shuffle: bool | None = None,
    num_workers: int = 4,
    transform: AugmentationSequential | None | str = "default",
    normalize: bool = True,
    img_to_neg1_1: bool = False,
    resize_to_mask: bool = True,
    resize_crop_to: int | None = 512,
    resize_crop_scale: tuple[float, float] = (0.8, 1.0),
    resize_crop_ratio: tuple[float, float] = (0.75, 1.3333333333333333),
    binarize_gt: bool = True,
    return_name: bool = False,
    pin_memory: bool = True,
    drop_last: bool = False,
    **loader_kwargs,
) -> tuple[XView2ChangeDetectionDataset, DataLoader]:
    dataset = XView2ChangeDetectionDataset(
        data_root=data_root,
        split=split,
        include_hold_in_train=include_hold_in_train,
        transform=transform,
        normalize=normalize,
        img_to_neg1_1=img_to_neg1_1,
        resize_to_mask=resize_to_mask,
        resize_crop_to=resize_crop_to,
        resize_crop_scale=resize_crop_scale,
        resize_crop_ratio=resize_crop_ratio,
        binarize_gt=binarize_gt,
        return_name=return_name,
    )

    if shuffle is None:
        shuffle = split == "train"

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        **loader_kwargs,
    )
    return dataset, dataloader
