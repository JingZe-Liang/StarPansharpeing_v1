from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

Split = Literal["train", "val"]


@dataclass(frozen=True)
class Flood3ISample:
    image: Path
    mask: Path
    name: str


@dataclass(frozen=True)
class Flood3IKeyTransform:
    image_key: str = "image"
    mask_key: str = "mask"
    img_key: str = "img"
    gt_key: str = "gt"
    squeeze_mask: bool = True
    keep_extra: bool = True

    def __call__(self, sample: dict[str, torch.Tensor | str]) -> dict[str, torch.Tensor | str]:
        if self.image_key not in sample or self.mask_key not in sample:
            raise KeyError(f"Missing keys: {self.image_key} or {self.mask_key}")
        img = sample[self.image_key]
        mask = sample[self.mask_key]
        if self.squeeze_mask and isinstance(mask, torch.Tensor) and mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)
        out: dict[str, torch.Tensor | str] = {self.img_key: img, self.gt_key: mask}
        if self.keep_extra:
            for key, value in sample.items():
                if key in (self.image_key, self.mask_key):
                    continue
                out[key] = value
        return out


def _read_pairs_from_list(root: Path, list_path: Path) -> list[Flood3ISample]:
    if not list_path.exists():
        raise FileNotFoundError(f"List file not found: {list_path}")
    lines = [line.strip() for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Empty list file: {list_path}")
    samples: list[Flood3ISample] = []
    for line in lines:
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"Invalid list entry: {line}")
        img_rel, mask_rel = parts
        img_path = root / img_rel
        mask_path = root / mask_rel
        if not img_path.exists() or not mask_path.exists():
            raise FileNotFoundError(f"Missing pair: {img_path}, {mask_path}")
        samples.append(Flood3ISample(image=img_path, mask=mask_path, name=img_path.stem))
    return samples


def _mask_name_from_image(image_name: str) -> str:
    stem = Path(image_name).stem
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Unexpected image name: {image_name}")
    return f"{parts[0]}_lab_{'_'.join(parts[1:])}.png"


def _read_pairs_from_folders(root: Path) -> list[Flood3ISample]:
    images_dir = root / "Images"
    masks_dir = root / "Semantic_mask"
    if not images_dir.exists() or not masks_dir.exists():
        raise FileNotFoundError(f"Missing Images or Semantic_mask directory under {root}")
    images = sorted(images_dir.glob("*.jpg"))
    if not images:
        raise ValueError(f"No images found in {images_dir}")
    samples: list[Flood3ISample] = []
    for img_path in images:
        mask_name = _mask_name_from_image(img_path.name)
        mask_path = masks_dir / mask_name
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {img_path.name}: {mask_path}")
        samples.append(Flood3ISample(image=img_path, mask=mask_path, name=img_path.stem))
    return samples


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB")).copy()


def _load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")).copy()


def _to_binary_mask(mask: np.ndarray, *, target_class: int, ignore_index: int) -> np.ndarray:
    binary = np.zeros(mask.shape, dtype=np.uint8)
    binary[mask == target_class] = 1
    if 0 <= ignore_index <= 255:
        binary[mask == ignore_index] = np.uint8(ignore_index)
    return binary


class Flood3IDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: Split = "train",
        list_file: str | Path | None = None,
        transform: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None = None,
        normalize: bool = True,
        return_name: bool = False,
        binary_target_class: int | None = None,
        ignore_index: int = 255,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.list_file = Path(list_file) if list_file is not None else None
        self.transform = transform
        self.normalize = normalize
        self.return_name = return_name
        self.binary_target_class = binary_target_class
        self.ignore_index = ignore_index
        if self.binary_target_class is not None and self.binary_target_class < 0:
            raise ValueError(f"binary_target_class must be >= 0, got {self.binary_target_class}")
        self.samples = self._build_samples()

    def _build_samples(self) -> list[Flood3ISample]:
        if self.list_file is not None:
            return _read_pairs_from_list(self.root, self.list_file)
        default_list = self.root / f"{self.split}.txt"
        if default_list.exists():
            return _read_pairs_from_list(self.root, default_list)
        return _read_pairs_from_folders(self.root)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor] | dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        image = _load_rgb(sample.image)
        mask = _load_mask(sample.mask)
        if self.binary_target_class is not None:
            mask = _to_binary_mask(mask, target_class=self.binary_target_class, ignore_index=self.ignore_index)

        image_t = torch.from_numpy(image).float()
        if self.normalize:
            image_t = image_t / 255.0
        image_t = image_t.permute(2, 0, 1)
        mask_t = torch.from_numpy(mask).long().unsqueeze(0)

        out_t: dict[str, torch.Tensor] = {"image": image_t, "mask": mask_t}
        if self.transform is not None:
            out_t = self.transform(out_t)
        if self.return_name:
            return {**out_t, "name": sample.name}
        return out_t


def get_flood3i_dataloader(
    batch_size: int,
    num_workers: int,
    split: Split = "train",
    root: str = "data/Downstreams/Flood-3i",
    list_file: str | Path | None = None,
    transform: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None = None,
    normalize: bool = True,
    return_name: bool = False,
    binary_target_class: int | None = None,
    ignore_index: int = 255,
    shuffle: bool | None = None,
    **loader_kwargs: Any,
) -> tuple[Flood3IDataset, DataLoader]:
    dataset = Flood3IDataset(
        root=root,
        split=split,
        list_file=list_file,
        transform=transform,
        normalize=normalize,
        return_name=return_name,
        binary_target_class=binary_target_class,
        ignore_index=ignore_index,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=(split == "train") if shuffle is None else shuffle,
        **loader_kwargs,
    )
    return dataset, dataloader
