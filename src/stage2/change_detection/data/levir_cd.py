from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from kornia.augmentation import AugmentationSequential, RandomHorizontalFlip, RandomVerticalFlip
from kornia.constants import DataKey
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class LEVIRSamplePaths:
    img1: Path
    img2: Path
    label: Path
    name: str


def get_default_transform(prob: float = 0.5) -> AugmentationSequential:
    return AugmentationSequential(
        RandomHorizontalFlip(p=prob),
        RandomVerticalFlip(p=prob),
        data_keys=[DataKey.INPUT, DataKey.INPUT, DataKey.MASK],
        same_on_batch=True,
        keepdim=True,
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


def build_sample_paths(data_root: Path, list_dir: str, split: str) -> list[LEVIRSamplePaths]:
    normalized_split = _normalize_split_name(split)
    list_path = data_root / list_dir / f"{normalized_split}.txt"
    names = read_list(list_path)

    a_dir = data_root / "A"
    b_dir = data_root / "B"
    label_dir = data_root / "label"

    samples: list[LEVIRSamplePaths] = []
    for name in names:
        img1 = a_dir / name
        img2 = b_dir / name
        label = label_dir / name
        if not img1.exists() or not img2.exists() or not label.exists():
            raise FileNotFoundError(f"Missing file for sample '{name}': {img1}, {img2}, {label}")
        samples.append(LEVIRSamplePaths(img1=img1, img2=img2, label=label, name=name))
    return samples


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


class LEVIRCDDataset(Dataset[dict[str, Tensor | str]]):
    def __init__(
        self,
        data_root: str | Path = "data/Downstreams/ChangeDetection/LEVIR-CD/LEVIR-CD256",
        split: Literal["train", "val", "test", "trainval", "train_val"] = "train",
        list_dir: str = "list",
        transform: AugmentationSequential | None | str = "default",
        normalize: bool = True,
        img_to_neg1_1: bool = False,
        label_threshold: int = 0,
        resize_to: int | tuple[int, int] | list[int] | None = None,
        return_name: bool = False,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = _normalize_split_name(split)
        self.list_dir = list_dir
        self.normalize = normalize
        self.img_to_neg1_1 = img_to_neg1_1
        self.label_threshold = label_threshold
        self.resize_to = _resolve_resize_shape(resize_to)
        self.return_name = return_name
        self.samples = build_sample_paths(self.data_root, self.list_dir, self.split)

        if transform == "default":
            transform = get_default_transform()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        sample = self.samples[index]
        img1 = load_rgb(sample.img1)
        img2 = load_rgb(sample.img2)
        gt = load_binary_label(sample.label, threshold=self.label_threshold)

        if self.resize_to is not None:
            img1 = F.interpolate(img1.unsqueeze(0), size=self.resize_to, mode="bilinear", align_corners=False).squeeze(
                0
            )
            img2 = F.interpolate(img2.unsqueeze(0), size=self.resize_to, mode="bilinear", align_corners=False).squeeze(
                0
            )
            gt = F.interpolate(gt.float().unsqueeze(0), size=self.resize_to, mode="nearest").squeeze(0).long()

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


def create_levir_cd_dataloader(
    data_root: str | Path = "data/Downstreams/ChangeDetection/LEVIR-CD/LEVIR-CD256",
    split: Literal["train", "val", "test", "trainval", "train_val"] = "train",
    list_dir: str = "list",
    batch_size: int = 4,
    shuffle: bool | None = None,
    num_workers: int = 4,
    transform: AugmentationSequential | None | str = "default",
    normalize: bool = True,
    img_to_neg1_1: bool = False,
    label_threshold: int = 0,
    resize_to: int | tuple[int, int] | list[int] | None = None,
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
        return_name=return_name,
    )

    normalized_split = _normalize_split_name(split)
    if shuffle is None:
        shuffle = normalized_split in {"train", "trainval"}

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return dataset, dataloader
