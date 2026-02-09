from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import tifffile
from torch.utils.data import DataLoader, Dataset

Split = Literal["train", "val", "test"]


@dataclass(frozen=True)
class AtlanticForestSample:
    image: Path
    mask: Path
    name: str


@dataclass(frozen=True)
class AtlanticForestKeyTransform:
    image_key: str = "image"
    mask_key: str = "mask"
    img_key: str = "img"
    gt_key: str = "gt"
    squeeze_mask: bool = True
    keep_extra: bool = True

    def __call__(self, sample: dict[str, torch.Tensor | str]) -> dict[str, torch.Tensor | str]:
        if self.image_key not in sample or self.mask_key not in sample:
            raise KeyError(f"Missing keys: {self.image_key} or {self.mask_key}")
        image = sample[self.image_key]
        mask = sample[self.mask_key]
        if self.squeeze_mask and isinstance(mask, torch.Tensor) and mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)
        out: dict[str, torch.Tensor | str] = {self.img_key: image, self.gt_key: mask}
        if self.keep_extra:
            for key, value in sample.items():
                if key in (self.image_key, self.mask_key):
                    continue
                out[key] = value
        return out


def _resolve_split_dirs(root: Path, split: Split) -> tuple[Path, Path]:
    if split == "train":
        return root / "Training" / "image", root / "Training" / "label"
    if split == "val":
        return root / "Validation" / "images", root / "Validation" / "masks"
    if split == "test":
        return root / "Test" / "image", root / "Test" / "mask"
    raise ValueError(f"Unsupported split: {split}")


def _read_pairs_from_folders(root: Path, split: Split) -> list[AtlanticForestSample]:
    image_dir, mask_dir = _resolve_split_dirs(root, split)
    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(f"Missing image/mask folders for split={split}: {image_dir}, {mask_dir}")

    image_paths = sorted(image_dir.glob("*.tif"))
    if not image_paths:
        raise ValueError(f"No .tif images found in {image_dir}")

    samples: list[AtlanticForestSample] = []
    for image_path in image_paths:
        mask_path = mask_dir / image_path.name
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for image {image_path.name}: {mask_path}")
        samples.append(AtlanticForestSample(image=image_path, mask=mask_path, name=image_path.stem))
    return samples


def _read_pairs_from_list(root: Path, list_path: Path) -> list[AtlanticForestSample]:
    if not list_path.exists():
        raise FileNotFoundError(f"List file not found: {list_path}")
    lines = [line.strip() for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Empty list file: {list_path}")

    samples: list[AtlanticForestSample] = []
    for line in lines:
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"Invalid list entry: {line}")
        img_rel, mask_rel = parts
        img_path = root / img_rel
        mask_path = root / mask_rel
        if not img_path.exists() or not mask_path.exists():
            raise FileNotFoundError(f"Missing pair: {img_path}, {mask_path}")
        samples.append(AtlanticForestSample(image=img_path, mask=mask_path, name=img_path.stem))
    return samples


def _load_tif_image(path: Path) -> np.ndarray:
    arr = tifffile.imread(path)
    image = np.asarray(arr)
    if image.ndim != 3:
        raise ValueError(f"Expected 3D image (H, W, C), got {image.shape} from {path}")
    return image.copy()


def _load_tif_mask(path: Path) -> np.ndarray:
    arr = tifffile.imread(path)
    mask = np.asarray(arr)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask (H, W), got {mask.shape} from {path}")
    return mask.copy()


def _apply_simple_augmentation(
    image_t: torch.Tensor,
    mask_t: torch.Tensor,
    *,
    p: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if p <= 0:
        return image_t, mask_t
    if torch.rand(1).item() < p:
        image_t = torch.flip(image_t, dims=[2])
        mask_t = torch.flip(mask_t, dims=[2])
    if torch.rand(1).item() < p:
        image_t = torch.flip(image_t, dims=[1])
        mask_t = torch.flip(mask_t, dims=[1])
    if torch.rand(1).item() < p:
        k = int(torch.randint(1, 4, (1,)).item())
        image_t = torch.rot90(image_t, k=k, dims=[1, 2])
        mask_t = torch.rot90(mask_t, k=k, dims=[1, 2])
    return image_t, mask_t


class AtlanticForestSegmentationDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: Split = "train",
        list_file: str | Path | None = None,
        transform: Callable[[dict[str, torch.Tensor | str]], dict[str, torch.Tensor | str]] | None = None,
        is_neg_1_1: bool = True,
        norm_const: float | None = 10000.0,
        augmentation: bool = False,
        augmentation_prob: float = 0.5,
        return_name: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.list_file = Path(list_file) if list_file is not None else None
        self.transform = transform
        self.is_neg_1_1 = is_neg_1_1
        self.norm_const = norm_const
        self.augmentation = augmentation
        self.augmentation_prob = augmentation_prob
        self.return_name = return_name
        self.samples = self._build_samples()

    def _build_samples(self) -> list[AtlanticForestSample]:
        if self.list_file is not None:
            return _read_pairs_from_list(self.root, self.list_file)
        return _read_pairs_from_folders(self.root, self.split)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor] | dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        image = _load_tif_image(sample.image)
        mask = _load_tif_mask(sample.mask)

        image_t = torch.from_numpy(image).float().permute(2, 0, 1)
        image_t = self._normalize_image(image_t)
        if self.is_neg_1_1:
            image_t = image_t * 2.0 - 1.0

        mask_t = torch.from_numpy(mask).long().unsqueeze(0)
        if self.augmentation and self.split == "train":
            image_t, mask_t = _apply_simple_augmentation(image_t, mask_t, p=self.augmentation_prob)

        out_t: dict[str, torch.Tensor | str] = {"image": image_t, "mask": mask_t}
        if self.transform is not None:
            out_t = self.transform(out_t)
        if self.return_name:
            return {**out_t, "name": sample.name}
        return out_t

    def _normalize_image(self, image_t: torch.Tensor) -> torch.Tensor:
        if self.norm_const is not None:
            return (image_t / float(self.norm_const)).clamp(0.0, 1.0)

        img_min = image_t.amin()
        img_max = image_t.amax()
        if torch.isclose(img_max, img_min):
            return torch.zeros_like(image_t)
        return (image_t - img_min) / (img_max - img_min)


def get_atlantic_forest_dataloader(
    batch_size: int,
    num_workers: int,
    split: Split = "train",
    root: str = "data/Downstreams/AtlanticForeastSegmentation",
    list_file: str | Path | None = None,
    transform: Callable[[dict[str, torch.Tensor | str]], dict[str, torch.Tensor | str]] | None = None,
    is_neg_1_1: bool = True,
    norm_const: float | None = 10000.0,
    augmentation: bool = False,
    augmentation_prob: float = 0.5,
    return_name: bool = False,
    shuffle: bool | None = None,
    **loader_kwargs: Any,
) -> tuple[AtlanticForestSegmentationDataset, DataLoader]:
    dataset = AtlanticForestSegmentationDataset(
        root=root,
        split=split,
        list_file=list_file,
        transform=transform,
        is_neg_1_1=is_neg_1_1,
        norm_const=norm_const,
        augmentation=augmentation,
        augmentation_prob=augmentation_prob,
        return_name=return_name,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=(split == "train") if shuffle is None else shuffle,
        **loader_kwargs,
    )
    return dataset, dataloader
