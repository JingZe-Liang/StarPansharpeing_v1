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
class DSIFNSamplePaths:
    img1: Path
    img2: Path
    mask: Path
    name: str


SUPPORTED_IMG_EXTS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def get_default_transform(prob: float = 0.5) -> AugmentationSequential:
    return AugmentationSequential(
        RandomHorizontalFlip(p=prob),
        RandomVerticalFlip(p=prob),
        data_keys=[DataKey.INPUT, DataKey.INPUT, DataKey.MASK],
        same_on_batch=True,
        keepdim=True,
    )


def _resolve_mask_dir(root: Path, mask_dir: str, mask_subdir: str | None, allow_mask_fallback: bool = True) -> Path:
    candidate = root / mask_dir
    if not candidate.exists():
        if mask_dir == "mask256":
            candidate = root / "mask_256"
        elif mask_dir == "mask_256":
            candidate = root / "mask256"
    if not candidate.exists() and allow_mask_fallback:
        fallback = root / "mask"
        if fallback.exists():
            candidate = fallback
    if not candidate.exists():
        raise FileNotFoundError(f"Mask directory not found: {root / mask_dir}")

    if mask_subdir is None:
        sub_candidate = candidate / "m"
        if sub_candidate.exists():
            return sub_candidate
        return candidate

    selected = candidate / mask_subdir
    if not selected.exists():
        raise FileNotFoundError(f"Mask sub-directory not found: {selected}")
    return selected


def _collect_files_by_stem(folder: Path, exts: tuple[str, ...]) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for ext in exts:
        for path in folder.glob(f"*{ext}"):
            files[path.stem] = path
    return files


def _collect_common_ids(
    root: Path, mask_path: Path
) -> tuple[list[str], dict[str, Path], dict[str, Path], dict[str, Path]]:
    t1_map = _collect_files_by_stem(root / "t1", SUPPORTED_IMG_EXTS)
    t2_map = _collect_files_by_stem(root / "t2", SUPPORTED_IMG_EXTS)
    mask_map = _collect_files_by_stem(mask_path, (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))

    common = set(t1_map) & set(t2_map) & set(mask_map)
    if not common:
        raise RuntimeError(
            f"No matched DSIFN samples found under {root}, expected ids shared by t1/t2/{mask_path.relative_to(root)}"
        )

    common_ids = sorted(common, key=lambda x: int(x) if x.isdigit() else x)
    return common_ids, t1_map, t2_map, mask_map


def _has_explicit_split_layout(root: Path) -> bool:
    return any((root / split).is_dir() for split in ("train", "val", "test"))


def _iter_split_roots(data_root: Path, split: Literal["train", "val", "test", "all"]) -> list[Path]:
    if not _has_explicit_split_layout(data_root):
        return [data_root]

    if split == "all":
        return [p for p in (data_root / "train", data_root / "val", data_root / "test") if p.is_dir()]

    split_root = data_root / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_root}")
    return [split_root]


def _split_ids(
    ids: list[str],
    split: Literal["train", "val", "test", "all"],
    train_ratio: float,
    val_ratio: float,
) -> list[str]:
    if split == "all":
        return ids

    total = len(ids)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)

    train_ids = ids[:n_train]
    val_ids = ids[n_train : n_train + n_val]
    test_ids = ids[n_train + n_val :]

    if split == "train":
        selected = train_ids
    elif split == "val":
        selected = val_ids
    else:
        selected = test_ids

    if not selected:
        raise ValueError(
            f"Split '{split}' is empty with total={total}, train_ratio={train_ratio}, val_ratio={val_ratio}."
        )
    return selected


def _build_samples(
    ids: list[str],
    t1_map: dict[str, Path],
    t2_map: dict[str, Path],
    mask_map: dict[str, Path],
) -> list[DSIFNSamplePaths]:
    samples: list[DSIFNSamplePaths] = []
    for sid in ids:
        img1 = t1_map.get(sid)
        img2 = t2_map.get(sid)
        mask = mask_map.get(sid)
        if img1 is None or img2 is None or mask is None:
            raise FileNotFoundError(f"Missing DSIFN sample files for id={sid}.")
        samples.append(DSIFNSamplePaths(img1=img1, img2=img2, mask=mask, name=sid))
    return samples


def _load_rgb(path: Path) -> Tensor:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(arr).float().permute(2, 0, 1)


def _load_binary_mask(path: Path) -> Tensor:
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.uint8).copy()
    mask = (arr > 0).astype(np.uint8)
    return torch.from_numpy(mask).long().unsqueeze(0)


class DSIFNChangeDetectionDataset(Dataset[dict[str, Tensor | str]]):
    def __init__(
        self,
        data_root: str | Path = "data/Downstreams/ChangeDetection-DSIFN",
        split: Literal["train", "val", "test", "all"] = "train",
        mask_dir: str = "mask_256",
        mask_subdir: str | None = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        transform: AugmentationSequential | None | str = "default",
        normalize: bool = True,
        img_to_neg1_1: bool = False,
        resize_to_mask: bool = True,
        return_name: bool = False,
    ) -> None:
        if not (0.0 < train_ratio < 1.0):
            raise ValueError(f"train_ratio must be in (0,1), got {train_ratio}")
        if not (0.0 <= val_ratio < 1.0):
            raise ValueError(f"val_ratio must be in [0,1), got {val_ratio}")
        if train_ratio + val_ratio >= 1.0:
            raise ValueError(f"train_ratio + val_ratio must be < 1, got {train_ratio + val_ratio}")

        self.data_root = Path(data_root)
        self.split = split
        self.normalize = normalize
        self.img_to_neg1_1 = img_to_neg1_1
        self.resize_to_mask = resize_to_mask
        self.return_name = return_name

        samples: list[DSIFNSamplePaths] = []
        split_roots = _iter_split_roots(self.data_root, split=split)
        explicit_split_layout = split_roots != [self.data_root]

        for split_root in split_roots:
            mask_path = _resolve_mask_dir(split_root, mask_dir=mask_dir, mask_subdir=mask_subdir)
            all_ids, t1_map, t2_map, mask_map = _collect_common_ids(split_root, mask_path)

            if explicit_split_layout:
                selected_ids = all_ids
            else:
                selected_ids = _split_ids(all_ids, split=split, train_ratio=train_ratio, val_ratio=val_ratio)

            samples.extend(_build_samples(selected_ids, t1_map=t1_map, t2_map=t2_map, mask_map=mask_map))

        if not samples:
            raise RuntimeError(f"No DSIFN samples found under {self.data_root} for split={split}.")
        self.samples = samples

        if transform == "default":
            transform = get_default_transform()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        sample = self.samples[index]

        img1 = _load_rgb(sample.img1)
        img2 = _load_rgb(sample.img2)
        gt = _load_binary_mask(sample.mask)

        if self.resize_to_mask and img1.shape[-2:] != gt.shape[-2:]:
            target_hw = gt.shape[-2:]
            img1 = F.interpolate(img1.unsqueeze(0), size=target_hw, mode="bilinear", align_corners=False).squeeze(0)
            img2 = F.interpolate(img2.unsqueeze(0), size=target_hw, mode="bilinear", align_corners=False).squeeze(0)

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


def create_dsifn_change_detection_dataloader(
    data_root: str | Path = "data/Downstreams/ChangeDetection-DSIFN",
    split: Literal["train", "val", "test", "all"] = "train",
    mask_dir: str = "mask_256",
    mask_subdir: str | None = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    batch_size: int = 4,
    shuffle: bool | None = None,
    num_workers: int = 4,
    transform: AugmentationSequential | None | str = "default",
    normalize: bool = True,
    img_to_neg1_1: bool = False,
    resize_to_mask: bool = True,
    return_name: bool = False,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> tuple[DSIFNChangeDetectionDataset, DataLoader]:
    dataset = DSIFNChangeDetectionDataset(
        data_root=data_root,
        split=split,
        mask_dir=mask_dir,
        mask_subdir=mask_subdir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        transform=transform,
        normalize=normalize,
        img_to_neg1_1=img_to_neg1_1,
        resize_to_mask=resize_to_mask,
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
    )
    return dataset, dataloader
