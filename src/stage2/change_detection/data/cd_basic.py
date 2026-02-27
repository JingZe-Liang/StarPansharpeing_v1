from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from kornia.augmentation import (
    AugmentationSequential,
    ColorJitter,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomRotation,
    RandomVerticalFlip,
)
from kornia.constants import DataKey
from PIL import Image
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, Dataset


SplitName = Literal["train", "val", "test", "trainval", "train_val", "train-val", "all"]

SUPPORTED_IMAGE_SUFFIXES: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


@dataclass(frozen=True)
class CDSamplePaths:
    img1: Path
    img2: Path
    label: Path
    name: str


LEVIRSamplePaths = CDSamplePaths

warnings.filterwarnings(
    "ignore",
    message="Default grid_sample and affine_grid behavior has changed",
    category=UserWarning,
)


def get_default_transform(prob: float = 0.5) -> AugmentationSequential:
    return AugmentationSequential(
        RandomHorizontalFlip(p=prob),
        RandomVerticalFlip(p=prob),
        ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1, p=prob),
        RandomRotation(degrees=[-30, 30], p=prob),
        data_keys=[DataKey.INPUT, DataKey.INPUT, DataKey.MASK],
        same_on_batch=True,
        keepdim=True,
        random_apply=2,
    )


def _normalize_split_name(split: str) -> str:
    normalized = split.strip().lower()
    aliases = {
        "train_val": "trainval",
        "train-val": "trainval",
        "trainval": "trainval",
    }
    return aliases.get(normalized, normalized)


def read_list(list_path: Path) -> list[str]:
    if not list_path.exists():
        raise FileNotFoundError(f"List file not found: {list_path}")
    names = [line.strip() for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise ValueError(f"Empty list file: {list_path}")
    return names


def _list_image_names(image_dir: Path) -> list[str]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    names = sorted(
        path.name for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    if not names:
        raise ValueError(f"No image files found in: {image_dir}")
    return names


def _build_samples_from_names(
    names: list[str],
    img1_root: Path,
    img2_root: Path,
    label_root: Path,
    name_prefix: str = "",
) -> list[CDSamplePaths]:
    for required_root in [img1_root, img2_root, label_root]:
        if not required_root.exists():
            raise FileNotFoundError(f"Required directory not found: {required_root}")

    samples: list[CDSamplePaths] = []
    for name in names:
        img1 = img1_root / name
        img2 = img2_root / name
        label = label_root / name
        if not img1.exists() or not img2.exists() or not label.exists():
            raise FileNotFoundError(f"Missing file for sample '{name}': {img1}, {img2}, {label}")

        sample_name = f"{name_prefix}{name}" if name_prefix else name
        samples.append(CDSamplePaths(img1=img1, img2=img2, label=label, name=sample_name))
    return samples


def build_standard_cd_sample_paths(
    data_root: Path,
    list_dir: str,
    split: str,
    img1_dir: str = "A",
    img2_dir: str = "B",
    label_dir: str = "label",
    use_split_subdir: bool = False,
) -> list[CDSamplePaths]:
    normalized_split = _normalize_split_name(split)

    if not use_split_subdir:
        list_path = data_root / list_dir / f"{normalized_split}.txt"
        names = read_list(list_path)
        return _build_samples_from_names(
            names=names,
            img1_root=data_root / img1_dir,
            img2_root=data_root / img2_dir,
            label_root=data_root / label_dir,
        )

    split_names = {
        "trainval": ["train", "val"],
        "train": ["train"],
        "val": ["val"],
        "all": ["train", "val", "test"],
        "test": ["test"],
    }[normalized_split]
    samples: list[CDSamplePaths] = []
    for split_name in split_names:
        split_root = data_root / split_name
        names = _list_image_names(split_root / img1_dir)
        split_samples = _build_samples_from_names(
            names=names,
            img1_root=split_root / img1_dir,
            img2_root=split_root / img2_dir,
            label_root=split_root / label_dir,
            name_prefix=f"{split_name}/",
        )
        samples.extend(split_samples)

    if not samples:
        raise ValueError(f"No samples found for split '{split}' in {data_root}")
    return samples


def build_sample_paths(data_root: Path, list_dir: str, split: str) -> list[LEVIRSamplePaths]:
    return build_standard_cd_sample_paths(data_root=data_root, list_dir=list_dir, split=split)


def load_rgb(path: Path) -> Tensor:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(arr).float().permute(2, 0, 1)


def load_binary_label(path: Path, threshold: int = 0) -> Tensor:
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.uint8).copy()
    label = (arr > threshold).astype(np.uint8)
    return torch.from_numpy(label).long().unsqueeze(0)


def _resolve_resize_shape(resize_to: int | tuple[int, int] | list[int] | None) -> tuple[int, int] | None:
    if resize_to is None:
        return None
    if isinstance(resize_to, int):
        if resize_to <= 0:
            raise ValueError(f"resize_to must be positive, got {resize_to}")
        return (resize_to, resize_to)
    if len(resize_to) != 2:
        raise ValueError(f"resize_to must have 2 items when sequence is used, got {resize_to}")

    h = int(resize_to[0])
    w = int(resize_to[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"resize_to values must be positive, got {(h, w)}")
    return (h, w)


def _resolve_shuffle(split: str, shuffle: bool | None) -> bool:
    if shuffle is not None:
        return shuffle
    return _normalize_split_name(split) in {"train", "trainval"}


def _build_random_crop_resize_transform(
    crop_resize_to: tuple[int, int] | None,
    crop_scale: tuple[float, float],
    crop_ratio: tuple[float, float],
    crop_prob: float,
) -> AugmentationSequential | None:
    if crop_resize_to is None:
        return None

    return AugmentationSequential(
        RandomResizedCrop(size=crop_resize_to, scale=crop_scale, ratio=crop_ratio, p=crop_prob),
        data_keys=[DataKey.INPUT, DataKey.INPUT, DataKey.MASK],
        same_on_batch=True,
        keepdim=True,
    )


class StandardBinaryChangeDetectionDataset(Dataset[dict[str, Tensor | str]]):
    def __init__(
        self,
        data_root: str | Path,
        split: str,
        list_dir: str = "list",
        img1_dir: str = "A",
        img2_dir: str = "B",
        label_dir: str = "label",
        use_split_subdir: bool = False,
        transform: AugmentationSequential | None | str = "default",
        normalize: bool = True,
        img_to_neg1_1: bool = False,
        label_threshold: int = 0,
        resize_to: int | tuple[int, int] | list[int] | None = None,
        random_crop_resize_to: int | tuple[int, int] | list[int] | None = None,
        random_crop_resize_scale: tuple[float, float] = (0.5, 1.0),
        random_crop_resize_ratio: tuple[float, float] = (0.75, 1.3333333333333333),
        random_crop_resize_prob: float = 1.0,
        return_name: bool = False,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = _normalize_split_name(split)
        self.list_dir = list_dir
        self.img1_dir = img1_dir
        self.img2_dir = img2_dir
        self.label_dir = label_dir
        self.use_split_subdir = use_split_subdir
        self.normalize = normalize
        self.img_to_neg1_1 = img_to_neg1_1
        self.label_threshold = label_threshold
        self.resize_to = _resolve_resize_shape(resize_to)
        crop_resize_shape = _resolve_resize_shape(random_crop_resize_to)
        self.return_name = return_name
        self.samples = build_standard_cd_sample_paths(
            data_root=self.data_root,
            list_dir=self.list_dir,
            split=self.split,
            img1_dir=self.img1_dir,
            img2_dir=self.img2_dir,
            label_dir=self.label_dir,
            use_split_subdir=self.use_split_subdir,
        )

        self.random_crop_resize_transform = _build_random_crop_resize_transform(
            crop_resize_to=crop_resize_shape,
            crop_scale=random_crop_resize_scale,
            crop_ratio=random_crop_resize_ratio,
            crop_prob=random_crop_resize_prob,
        )

        if transform == "default":
            self.transform: AugmentationSequential | None = get_default_transform()
        elif transform is None:
            self.transform = None
        elif isinstance(transform, AugmentationSequential):
            self.transform = transform
        else:
            raise ValueError("transform only supports 'default', None, or AugmentationSequential.")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_sample(self, sample: CDSamplePaths) -> tuple[Tensor, Tensor, Tensor]:
        img1 = load_rgb(sample.img1)
        img2 = load_rgb(sample.img2)
        gt = load_binary_label(sample.label, threshold=self.label_threshold)
        return img1, img2, gt

    def _random_crop_resize_sample(self, img1: Tensor, img2: Tensor, gt: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if self.random_crop_resize_transform is None:
            return img1, img2, gt

        img1, img2, gt = self.random_crop_resize_transform(
            img1.unsqueeze(0),
            img2.unsqueeze(0),
            gt.float().unsqueeze(0),
        )
        return img1.squeeze(0), img2.squeeze(0), gt.squeeze(0).long()

    def _resize_sample(self, img1: Tensor, img2: Tensor, gt: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if self.resize_to is None:
            return img1, img2, gt

        img1 = F.interpolate(img1.unsqueeze(0), size=self.resize_to, mode="bilinear", align_corners=False).squeeze(0)
        img2 = F.interpolate(img2.unsqueeze(0), size=self.resize_to, mode="bilinear", align_corners=False).squeeze(0)
        gt = F.interpolate(gt.float().unsqueeze(0), size=self.resize_to, mode="nearest").squeeze(0).long()
        return img1, img2, gt

    def _normalize_images(self, img1: Tensor, img2: Tensor) -> tuple[Tensor, Tensor]:
        if self.normalize:
            img1 = img1 / 255.0
            img2 = img2 / 255.0
        if self.img_to_neg1_1:
            img1 = img1 * 2.0 - 1.0
            img2 = img2 * 2.0 - 1.0
        return img1, img2

    def _transform_sample(self, img1: Tensor, img2: Tensor, gt: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if self.transform is None:
            return img1, img2, gt

        img1, img2, gt = self.transform(
            img1.unsqueeze(0),
            img2.unsqueeze(0),
            gt.float().unsqueeze(0),
        )
        return img1.squeeze(0), img2.squeeze(0), gt.squeeze(0).long()

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        sample = self.samples[index]
        img1, img2, gt = self._load_sample(sample)
        img1, img2, gt = self._random_crop_resize_sample(img1, img2, gt)
        img1, img2, gt = self._resize_sample(img1, img2, gt)
        img1, img2 = self._normalize_images(img1, img2)
        img1, img2, gt = self._transform_sample(img1, img2, gt)

        out: dict[str, Tensor | str] = {"img1": img1, "img2": img2, "gt": gt}
        if self.return_name:
            out["name"] = sample.name
        return out


class LEVIRCDDataset(StandardBinaryChangeDetectionDataset):
    def __init__(
        self,
        data_root: str | Path = "data/Downstreams/ChangeDetection/LEVIR-CD/LEVIR-CD256",
        split: SplitName = "train",
        list_dir: str = "list",
        transform: AugmentationSequential | None | str = "default",
        normalize: bool = True,
        img_to_neg1_1: bool = False,
        label_threshold: int = 0,
        resize_to: int | tuple[int, int] | list[int] | None = None,
        random_crop_resize_to: int | tuple[int, int] | list[int] | None = None,
        random_crop_resize_scale: tuple[float, float] = (0.5, 1.0),
        random_crop_resize_ratio: tuple[float, float] = (0.75, 1.3333333333333333),
        random_crop_resize_prob: float = 1.0,
        return_name: bool = False,
    ) -> None:
        super().__init__(
            data_root=data_root,
            split=split,
            list_dir=list_dir,
            transform=transform,
            normalize=normalize,
            img_to_neg1_1=img_to_neg1_1,
            label_threshold=label_threshold,
            resize_to=resize_to,
            random_crop_resize_to=random_crop_resize_to,
            random_crop_resize_scale=random_crop_resize_scale,
            random_crop_resize_ratio=random_crop_resize_ratio,
            random_crop_resize_prob=random_crop_resize_prob,
            return_name=return_name,
        )


class LEVIRRaw1024Dataset(StandardBinaryChangeDetectionDataset):
    def __init__(
        self,
        data_root: str | Path = "data/Downstreams/ChangeDetection/LEVIR-CD/raw",
        split: SplitName = "test",
        transform: AugmentationSequential | None | str = None,
        normalize: bool = True,
        img_to_neg1_1: bool = False,
        label_threshold: int = 0,
        resize_to: int | tuple[int, int] | list[int] | None = None,
        random_crop_resize_to: int | tuple[int, int] | list[int] | None = None,
        random_crop_resize_scale: tuple[float, float] = (1.0, 1.0),
        random_crop_resize_ratio: tuple[float, float] = (1.0, 1.0),
        random_crop_resize_prob: float = 1.0,
        return_name: bool = True,
    ) -> None:
        super().__init__(
            data_root=data_root,
            split=split,
            img1_dir="A",
            img2_dir="B",
            label_dir="label",
            use_split_subdir=True,
            transform=transform,
            normalize=normalize,
            img_to_neg1_1=img_to_neg1_1,
            label_threshold=label_threshold,
            resize_to=resize_to,
            random_crop_resize_to=random_crop_resize_to,
            random_crop_resize_scale=random_crop_resize_scale,
            random_crop_resize_ratio=random_crop_resize_ratio,
            random_crop_resize_prob=random_crop_resize_prob,
            return_name=return_name,
        )


class S2LookingCDDataset(StandardBinaryChangeDetectionDataset):
    def __init__(
        self,
        data_root: str | Path = "data/Downstreams/ChangeDetection/S2Looking",
        split: SplitName = "train",
        transform: AugmentationSequential | None | str = "default",
        normalize: bool = True,
        img_to_neg1_1: bool = False,
        label_threshold: int = 0,
        resize_to: int | tuple[int, int] | list[int] | None = None,
        random_crop_resize_to: int | tuple[int, int] | list[int] | None = 256,
        random_crop_resize_scale: tuple[float, float] = (0.5, 1.0),
        random_crop_resize_ratio: tuple[float, float] = (0.75, 1.3333333333333333),
        random_crop_resize_prob: float = 1.0,
        return_name: bool = False,
    ) -> None:
        super().__init__(
            data_root=data_root,
            split=split,
            img1_dir="Image1",
            img2_dir="Image2",
            label_dir="label",
            use_split_subdir=True,
            transform=transform,
            normalize=normalize,
            img_to_neg1_1=img_to_neg1_1,
            label_threshold=label_threshold,
            resize_to=resize_to,
            random_crop_resize_to=random_crop_resize_to,
            random_crop_resize_scale=random_crop_resize_scale,
            random_crop_resize_ratio=random_crop_resize_ratio,
            random_crop_resize_prob=random_crop_resize_prob,
            return_name=return_name,
        )


class SVCDDataset(StandardBinaryChangeDetectionDataset):
    def __init__(
        self,
        data_root: str | Path = "data/Downstreams/ChangeDetection/SVCD_subset",
        split: SplitName = "train",
        transform: AugmentationSequential | None | str = "default",
        normalize: bool = True,
        img_to_neg1_1: bool = False,
        label_threshold: int = 0,
        resize_to: int | tuple[int, int] | list[int] | None = 256,
        random_crop_resize_to: int | tuple[int, int] | list[int] | None = None,
        random_crop_resize_scale: tuple[float, float] = (0.5, 1.0),
        random_crop_resize_ratio: tuple[float, float] = (0.75, 1.3333333333333333),
        random_crop_resize_prob: float = 1.0,
        return_name: bool = False,
    ) -> None:
        super().__init__(
            data_root=data_root,
            split=split,
            img1_dir="A",
            img2_dir="B",
            label_dir="OUT",
            use_split_subdir=True,
            transform=transform,
            normalize=normalize,
            img_to_neg1_1=img_to_neg1_1,
            label_threshold=label_threshold,
            resize_to=resize_to,
            random_crop_resize_to=random_crop_resize_to,
            random_crop_resize_scale=random_crop_resize_scale,
            random_crop_resize_ratio=random_crop_resize_ratio,
            random_crop_resize_prob=random_crop_resize_prob,
            return_name=return_name,
        )


class JL1CDDataset(StandardBinaryChangeDetectionDataset):
    def __init__(
        self,
        data_root: str | Path = "data/Downstreams/ChangeDetection/JL1-CD",
        split: SplitName = "train",
        transform: AugmentationSequential | None | str = "default",
        normalize: bool = True,
        img_to_neg1_1: bool = False,
        label_threshold: int = 0,
        resize_to: int | tuple[int, int] | list[int] | None = 256,
        random_crop_resize_to: int | tuple[int, int] | list[int] | None = None,
        random_crop_resize_scale: tuple[float, float] = (0.5, 1.0),
        random_crop_resize_ratio: tuple[float, float] = (0.75, 1.3333333333333333),
        random_crop_resize_prob: float = 1.0,
        return_name: bool = False,
    ) -> None:
        super().__init__(
            data_root=data_root,
            split=split,
            img1_dir="A",
            img2_dir="B",
            label_dir="label",
            use_split_subdir=True,
            transform=transform,
            normalize=normalize,
            img_to_neg1_1=img_to_neg1_1,
            label_threshold=label_threshold,
            resize_to=resize_to,
            random_crop_resize_to=random_crop_resize_to,
            random_crop_resize_scale=random_crop_resize_scale,
            random_crop_resize_ratio=random_crop_resize_ratio,
            random_crop_resize_prob=random_crop_resize_prob,
            return_name=return_name,
        )


def _create_standard_cd_dataloader(
    dataset: Dataset[dict[str, Tensor | str]],
    split: str,
    batch_size: int,
    shuffle: bool | None,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool,
) -> tuple[Dataset[dict[str, Tensor | str]], DataLoader]:
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=_resolve_shuffle(split=split, shuffle=shuffle),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return dataset, dataloader


def create_levir_cd_dataloader(
    data_root: str | Path = "data/Downstreams/ChangeDetection/LEVIR-CD/LEVIR-CD256",
    split: SplitName = "train",
    list_dir: str = "list",
    batch_size: int = 4,
    shuffle: bool | None = None,
    num_workers: int = 4,
    transform: AugmentationSequential | None | str = "default",
    normalize: bool = True,
    img_to_neg1_1: bool = False,
    label_threshold: int = 0,
    resize_to: int | tuple[int, int] | list[int] | None = None,
    random_crop_resize_to: int | tuple[int, int] | list[int] | None = None,
    random_crop_resize_scale: tuple[float, float] = (0.5, 1.0),
    random_crop_resize_ratio: tuple[float, float] = (0.75, 1.3333333333333333),
    random_crop_resize_prob: float = 1.0,
    return_name: bool = False,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> tuple[LEVIRCDDataset, DataLoader]:
    dataset = LEVIRCDDataset(
        data_root=data_root,
        split=split,
        list_dir=list_dir,
        transform=transform,
        normalize=normalize,
        img_to_neg1_1=img_to_neg1_1,
        label_threshold=label_threshold,
        resize_to=resize_to,
        random_crop_resize_to=random_crop_resize_to,
        random_crop_resize_scale=random_crop_resize_scale,
        random_crop_resize_ratio=random_crop_resize_ratio,
        random_crop_resize_prob=random_crop_resize_prob,
        return_name=return_name,
    )
    _, dataloader = _create_standard_cd_dataloader(
        dataset=dataset,
        split=split,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return dataset, dataloader


def create_levir_raw1024_test_dataloader(
    data_root: str | Path = "data/Downstreams/ChangeDetection/LEVIR-CD/raw",
    split: SplitName = "test",
    batch_size: int = 1,
    shuffle: bool | None = False,
    num_workers: int = 2,
    transform: AugmentationSequential | None | str = None,
    normalize: bool = True,
    img_to_neg1_1: bool = False,
    label_threshold: int = 0,
    resize_to: int | tuple[int, int] | list[int] | None = None,
    random_crop_resize_to: int | tuple[int, int] | list[int] | None = None,
    random_crop_resize_scale: tuple[float, float] = (1.0, 1.0),
    random_crop_resize_ratio: tuple[float, float] = (1.0, 1.0),
    random_crop_resize_prob: float = 1.0,
    return_name: bool = True,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> tuple[LEVIRRaw1024Dataset, DataLoader]:
    dataset = LEVIRRaw1024Dataset(
        data_root=data_root,
        split=split,
        transform=transform,
        normalize=normalize,
        img_to_neg1_1=img_to_neg1_1,
        label_threshold=label_threshold,
        resize_to=resize_to,
        random_crop_resize_to=random_crop_resize_to,
        random_crop_resize_scale=random_crop_resize_scale,
        random_crop_resize_ratio=random_crop_resize_ratio,
        random_crop_resize_prob=random_crop_resize_prob,
        return_name=return_name,
    )
    _, dataloader = _create_standard_cd_dataloader(
        dataset=dataset,
        split=split,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return dataset, dataloader


def create_s2looking_cd_dataloader(
    data_root: str | Path = "data/Downstreams/ChangeDetection/S2Looking",
    split: SplitName = "train",
    batch_size: int = 4,
    shuffle: bool | None = None,
    num_workers: int = 4,
    transform: AugmentationSequential | None | str = "default",
    normalize: bool = True,
    img_to_neg1_1: bool = False,
    label_threshold: int = 0,
    resize_to: int | tuple[int, int] | list[int] | None = None,
    random_crop_resize_to: int | tuple[int, int] | list[int] | None = 256,
    random_crop_resize_scale: tuple[float, float] = (0.8, 1.0),
    random_crop_resize_ratio: tuple[float, float] = (0.75, 1.3333333333333333),
    random_crop_resize_prob: float = 1.0,
    return_name: bool = False,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> tuple[S2LookingCDDataset, DataLoader]:
    dataset = S2LookingCDDataset(
        data_root=data_root,
        split=split,
        transform=transform,
        normalize=normalize,
        img_to_neg1_1=img_to_neg1_1,
        label_threshold=label_threshold,
        resize_to=resize_to,
        random_crop_resize_to=random_crop_resize_to,
        random_crop_resize_scale=random_crop_resize_scale,
        random_crop_resize_ratio=random_crop_resize_ratio,
        random_crop_resize_prob=random_crop_resize_prob,
        return_name=return_name,
    )
    _, dataloader = _create_standard_cd_dataloader(
        dataset=dataset,
        split=split,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return dataset, dataloader


def create_svcd_dataloader(
    data_root: str | Path = "data/Downstreams/ChangeDetection/SVCD_subset",
    split: SplitName = "train",
    batch_size: int = 4,
    shuffle: bool | None = None,
    num_workers: int = 4,
    transform: AugmentationSequential | None | str = "default",
    normalize: bool = True,
    img_to_neg1_1: bool = False,
    label_threshold: int = 0,
    resize_to: int | tuple[int, int] | list[int] | None = 256,
    random_crop_resize_to: int | tuple[int, int] | list[int] | None = None,
    random_crop_resize_scale: tuple[float, float] = (1.0, 1.0),
    random_crop_resize_ratio: tuple[float, float] = (0.75, 1.3333333333333333),
    random_crop_resize_prob: float = 1.0,
    return_name: bool = False,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> tuple[SVCDDataset, DataLoader]:
    dataset = SVCDDataset(
        data_root=data_root,
        split=split,
        transform=transform,
        normalize=normalize,
        img_to_neg1_1=img_to_neg1_1,
        label_threshold=label_threshold,
        resize_to=resize_to,
        random_crop_resize_to=random_crop_resize_to,
        random_crop_resize_scale=random_crop_resize_scale,
        random_crop_resize_ratio=random_crop_resize_ratio,
        random_crop_resize_prob=random_crop_resize_prob,
        return_name=return_name,
    )
    _, dataloader = _create_standard_cd_dataloader(
        dataset=dataset,
        split=split,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return dataset, dataloader


def create_jl1_cd_dataloader(
    data_root: str | Path = "data/Downstreams/ChangeDetection/JL1-CD",
    split: SplitName = "train",
    batch_size: int = 4,
    shuffle: bool | None = None,
    num_workers: int = 4,
    transform: AugmentationSequential | None | str = "default",
    normalize: bool = True,
    img_to_neg1_1: bool = False,
    label_threshold: int = 0,
    resize_to: int | tuple[int, int] | list[int] | None = 256,
    random_crop_resize_to: int | tuple[int, int] | list[int] | None = None,
    random_crop_resize_scale: tuple[float, float] = (0.5, 1.0),
    random_crop_resize_ratio: tuple[float, float] = (0.75, 1.3333333333333333),
    random_crop_resize_prob: float = 1.0,
    return_name: bool = False,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> tuple[JL1CDDataset, DataLoader]:
    dataset = JL1CDDataset(
        data_root=data_root,
        split=split,
        transform=transform,
        normalize=normalize,
        img_to_neg1_1=img_to_neg1_1,
        label_threshold=label_threshold,
        resize_to=resize_to,
        random_crop_resize_to=random_crop_resize_to,
        random_crop_resize_scale=random_crop_resize_scale,
        random_crop_resize_ratio=random_crop_resize_ratio,
        random_crop_resize_prob=random_crop_resize_prob,
        return_name=return_name,
    )
    _, dataloader = _create_standard_cd_dataloader(
        dataset=dataset,
        split=split,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return dataset, dataloader


def _normalize_dataset_alias(dataset_name: str) -> str:
    return dataset_name.strip().lower().replace("-", "_")


def create_concat_cd_train_dataloader(
    batch_size: int = 4,
    num_workers: int = 4,
    dataset_names: tuple[str, ...] = ("levir", "s2looking", "svcd", "jl1"),
    output_size: int | tuple[int, int] | list[int] = 256,
    shuffle: bool = True,
    transform: AugmentationSequential | None | str = "default",
    normalize: bool = True,
    img_to_neg1_1: bool = True,
    label_threshold: int = 0,
    return_name: bool = False,
    pin_memory: bool = True,
    drop_last: bool = False,
    levir_data_root: str | Path = "data/Downstreams/ChangeDetection/LEVIR-CD/LEVIR-CD256",
    s2looking_data_root: str | Path = "data/Downstreams/ChangeDetection/S2Looking",
    svcd_data_root: str | Path = "data/Downstreams/ChangeDetection/SVCD_subset",
    jl1_data_root: str | Path = "data/Downstreams/ChangeDetection/JL1-CD",
    # split
    levir_split: SplitName = "trainval",
    s2looking_split: SplitName = "train",
    svcd_split: SplitName = "train",
    jl1_split: SplitName = "train",
    # cropping
    s2looking_use_random_crop_resize: bool = True,
    jl1_use_random_crop_resize: bool = False,
    random_crop_resize_scale: tuple[float, float] = (0.5, 1.0),
    random_crop_resize_ratio: tuple[float, float] = (0.75, 1.3333333333333333),
    random_crop_resize_prob: float = 1.0,
) -> tuple[ConcatDataset[dict[str, Tensor | str]], DataLoader]:
    resolved_output_size = _resolve_resize_shape(output_size)
    if resolved_output_size is None:
        raise ValueError("output_size must not be None for concat training dataloader")

    datasets: list[Dataset[dict[str, Tensor | str]]] = []
    for dataset_name in dataset_names:
        alias = _normalize_dataset_alias(dataset_name)

        if alias in {"levir", "levir_cd", "levir_cd256"}:
            datasets.append(
                LEVIRCDDataset(
                    data_root=levir_data_root,
                    split=levir_split,
                    transform=transform,
                    normalize=normalize,
                    img_to_neg1_1=img_to_neg1_1,
                    label_threshold=label_threshold,
                    resize_to=resolved_output_size,
                    return_name=return_name,
                )
            )

        elif alias in {"s2looking", "s2_looking"}:
            datasets.append(
                S2LookingCDDataset(
                    data_root=s2looking_data_root,
                    split=s2looking_split,
                    transform=transform,
                    normalize=normalize,
                    img_to_neg1_1=img_to_neg1_1,
                    label_threshold=label_threshold,
                    resize_to=None if s2looking_use_random_crop_resize else resolved_output_size,
                    random_crop_resize_to=resolved_output_size if s2looking_use_random_crop_resize else None,
                    random_crop_resize_scale=(0.8, 1.0),
                    random_crop_resize_ratio=(0.75, 1.3333333333333333),
                    return_name=return_name,
                )
            )

        elif alias in {"svcd", "svcd_subset"}:
            datasets.append(
                SVCDDataset(
                    data_root=svcd_data_root,
                    split=svcd_split,
                    transform=transform,
                    normalize=normalize,
                    img_to_neg1_1=img_to_neg1_1,
                    label_threshold=label_threshold,
                    resize_to=resolved_output_size,
                    return_name=return_name,
                )
            )

        elif alias in {"jl1", "jl1_cd", "jl1cd"}:
            datasets.append(
                JL1CDDataset(
                    data_root=jl1_data_root,
                    split=jl1_split,
                    transform=transform,
                    normalize=normalize,
                    img_to_neg1_1=img_to_neg1_1,
                    label_threshold=label_threshold,
                    resize_to=None if jl1_use_random_crop_resize else resolved_output_size,
                    random_crop_resize_to=resolved_output_size if jl1_use_random_crop_resize else None,
                    random_crop_resize_scale=(0.5, 1.0),
                    return_name=return_name,
                )
            )

        else:
            raise ValueError(f"Unsupported dataset alias '{dataset_name}' in dataset_names={dataset_names}")

    if not datasets:
        raise ValueError("No datasets were selected for concat training dataloader")

    concat_dataset: ConcatDataset[dict[str, Tensor | str]] = ConcatDataset(datasets)
    dataloader = DataLoader(
        concat_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return concat_dataset, dataloader


if __name__ == "__main__":
    import math

    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    OUT_PATH = "/tmp/cd_basic_batch.png"
    BATCH_SIZE = 8

    print("Building concat dataloader ...")
    _, loader = create_concat_cd_train_dataloader(
        batch_size=BATCH_SIZE,
        num_workers=2,
        dataset_names=("levir", "s2looking", "svcd", "jl1"),
        output_size=256,
        shuffle=True,
        img_to_neg1_1=False,  # keep [0,1] for easy display
        pin_memory=False,
    )
    print(f"Total samples: {len(loader.dataset)}")  # type: ignore[arg-type]

    batch = next(iter(loader))
    img1: Tensor = batch["img1"]  # (B, 3, H, W)
    img2: Tensor = batch["img2"]
    gt: Tensor = batch["gt"]  # (B, 1, H, W)

    B = img1.shape[0]
    cols = B
    rows = 3  # img1 / img2 / gt

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    for i in range(B):

        def _show(ax: plt.Axes, t: Tensor, title: str) -> None:
            arr = t.permute(1, 2, 0).clamp(0.0, 1.0).numpy()
            ax.imshow(arr)
            ax.set_title(title, fontsize=7)
            ax.axis("off")

        _show(axes[0, i], img1[i], f"img1 [{i}]")
        _show(axes[1, i], img2[i], f"img2 [{i}]")
        # gt is 1-channel long; display as grayscale
        gt_show = gt[i].float()  # (1, H, W)
        axes[2, i].imshow(gt_show.squeeze(0).numpy(), cmap="gray", vmin=0, vmax=1)
        axes[2, i].set_title(f"gt [{i}]", fontsize=7)
        axes[2, i].axis("off")

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=120)
    print(f"Saved batch visualization to {OUT_PATH}")
