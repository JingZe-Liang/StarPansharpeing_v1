from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from loguru import logger

logger = logger.bind(_name_="SOSOilLeakage")

Split = Literal["train", "val", "test"]
Modality = Literal["sentinel", "palsar", "both"]
Layout = Literal["flat", "legacy"]


@dataclass(frozen=True)
class SOSOilLeakageSample:
    image: Path
    mask: Path
    name: str
    modality: Literal["sentinel", "palsar"]


@dataclass(frozen=True)
class SOSOilLeakageKeyTransform:
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


def _iter_modalities(modality: Modality) -> tuple[Literal["sentinel", "palsar"], ...]:
    if modality == "both":
        return ("sentinel", "palsar")
    return (modality,)


def _normalize_split(split: Split) -> Literal["train", "val"]:
    return "train" if split == "train" else "val"


def _detect_layout(root: Path) -> Layout:
    flat_image_root = root / "images"
    flat_mask_root = root / "masks"
    if flat_image_root.exists() and flat_mask_root.exists():
        return "flat"
    return "legacy"


def _resolve_pair_dirs(
    root: Path, split: Split, modality: Literal["sentinel", "palsar"], layout: Layout
) -> tuple[Path, Path]:
    if layout == "flat":
        split_name = _normalize_split(split)
        return root / "images" / split_name, root / "masks" / split_name

    base = root / split / modality
    if split == "train":
        return base, base
    return base / "sat", base / "gt"


def _collect_image_paths(image_dir: Path, modality: Literal["sentinel", "palsar"], layout: Layout) -> list[Path]:
    image_paths = (
        sorted(image_dir.glob(f"{modality}_*.png")) if layout == "flat" else sorted(image_dir.glob("*_sat.jpg"))
    )
    if not image_paths:
        raise ValueError(f"No images found for modality={modality} in {image_dir}")
    return image_paths


def _resolve_mask_path(image_path: Path, mask_dir: Path, layout: Layout) -> tuple[Path, str]:
    if layout == "flat":
        sample_id = image_path.stem
        mask_path = mask_dir / image_path.name
        return mask_path, sample_id

    stem = image_path.stem
    if not stem.endswith("_sat"):
        raise ValueError(f"Unexpected image file name: {image_path.name}")
    sample_id = stem.removesuffix("_sat")
    mask_path = mask_dir / f"{sample_id}_mask.png"
    return mask_path, sample_id


def _read_pairs_from_folders(root: Path, split: Split, modality: Modality) -> list[SOSOilLeakageSample]:
    layout = _detect_layout(root)
    samples: list[SOSOilLeakageSample] = []
    for mod in _iter_modalities(modality):
        image_dir, mask_dir = _resolve_pair_dirs(root, split, mod, layout)
        if not image_dir.exists() or not mask_dir.exists():
            raise FileNotFoundError(
                f"Missing directories for split={split}, modality={mod}, layout={layout}: {image_dir}, {mask_dir}"
            )

        image_paths = _collect_image_paths(image_dir, mod, layout)

        for image_path in image_paths:
            mask_path, sample_id = _resolve_mask_path(image_path, mask_dir, layout)
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing mask for image {image_path.name}: {mask_path}")
            samples.append(SOSOilLeakageSample(image=image_path, mask=mask_path, name=sample_id, modality=mod))
    logger.info(f"Found {len(samples)} samples for split={split}, modality={modality}, layout={layout}")
    return samples


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB")).copy()


def _load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")).copy()


class SOSOilLeakageDataset(Dataset):
    def __init__(
        self,
        root: str | Path = "data/Downstreams/SOS_OilLeakage",
        split: Split = "train",
        modality: Modality = "sentinel",
        transform: Callable[[dict[str, torch.Tensor | str]], dict[str, torch.Tensor | str]] | None = None,
        normalize: bool = True,
        to_neg_1_1: bool = True,
        binarize_mask: bool = True,
        return_name: bool = False,
        return_modality: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.modality = modality
        self.transform = transform
        self.normalize = normalize
        self.to_neg_1_1 = to_neg_1_1
        self.binarize_mask = binarize_mask
        self.return_name = return_name
        self.return_modality = return_modality

        if self.split not in ("train", "val", "test"):
            raise ValueError(f"Unsupported split={self.split}, expected 'train', 'val' or 'test'")
        if self.modality not in ("sentinel", "palsar", "both"):
            raise ValueError(f"Unsupported modality={self.modality}, expected sentinel/palsar/both")
        if self.to_neg_1_1 and not self.normalize:
            raise ValueError("to_neg_1_1=True requires normalize=True")

        self.samples = _read_pairs_from_folders(self.root, self.split, self.modality)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor] | dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        image = _load_rgb(sample.image)
        mask = _load_mask(sample.mask)

        if self.binarize_mask:
            mask = (mask > 0).astype(np.uint8)

        image_t = torch.from_numpy(image).float()
        if self.normalize:
            image_t = image_t / 255.0
        if self.to_neg_1_1:
            image_t = image_t * 2.0 - 1.0
        image_t = image_t.permute(2, 0, 1)
        mask_t = torch.from_numpy(mask).long().unsqueeze(0)

        out_t: dict[str, torch.Tensor | str] = {"image": image_t, "mask": mask_t}
        if self.transform is not None:
            out_t = self.transform(out_t)
        if self.return_name:
            out_t = {**out_t, "name": sample.name}
        if self.return_modality:
            out_t = {**out_t, "modality": sample.modality}
        return out_t


def get_sos_oil_leakage_dataloader(
    batch_size: int,
    num_workers: int,
    split: Split = "train",
    root: str | Path = "data/Downstreams/SOS_OilLeakage",
    modality: Modality = "sentinel",
    transform: Callable[[dict[str, torch.Tensor | str]], dict[str, torch.Tensor | str]] | None = None,
    normalize: bool = True,
    to_neg_1_1: bool = True,
    binarize_mask: bool = True,
    return_name: bool = False,
    return_modality: bool = False,
    shuffle: bool | None = None,
    **loader_kwargs: Any,
) -> tuple[SOSOilLeakageDataset, DataLoader]:
    dataset = SOSOilLeakageDataset(
        root=root,
        split=split,
        modality=modality,
        transform=transform,
        normalize=normalize,
        to_neg_1_1=to_neg_1_1,
        binarize_mask=binarize_mask,
        return_name=return_name,
        return_modality=return_modality,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=(split == "train") if shuffle is None else shuffle,
        **loader_kwargs,
    )
    return dataset, dataloader
