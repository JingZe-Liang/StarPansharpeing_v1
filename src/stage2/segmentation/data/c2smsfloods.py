from __future__ import annotations

import json
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

SensorMode = Literal["s1", "s2_rgb_swir", "s2_all", "all"]
SplitUnit = Literal["scene", "chip"]


def get_default_transform(prob: float = 0.5) -> AugmentationSequential:
    return AugmentationSequential(
        RandomHorizontalFlip(p=prob),
        RandomVerticalFlip(p=prob),
        data_keys=[DataKey.INPUT, DataKey.MASK],
        same_on_batch=True,
        keepdim=True,
    )


@dataclass(frozen=True)
class C2SMSFloodSamplePaths:
    name: str
    scene_id: str
    s1_dir: Path | None = None
    s2_dir: Path | None = None


def _list_scene_dirs(chips_root: Path) -> list[Path]:
    scenes = sorted((p for p in chips_root.iterdir() if p.is_dir()), key=lambda p: p.name)
    if not scenes:
        raise RuntimeError(f"No scene directories found under {chips_root}")
    return scenes


def _split_scene_ids(
    scene_ids: list[str],
    split: Literal["train", "val", "test", "all"],
    train_ratio: float,
    val_ratio: float,
) -> set[str]:
    if split == "all":
        return set(scene_ids)

    n_total = len(scene_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_ids = scene_ids[:n_train]
    val_ids = scene_ids[n_train : n_train + n_val]
    test_ids = scene_ids[n_train + n_val :]

    selected = train_ids if split == "train" else val_ids if split == "val" else test_ids
    if not selected:
        raise ValueError(
            f"Split '{split}' is empty: total={n_total}, train_ratio={train_ratio}, val_ratio={val_ratio}."
        )
    return set(selected)


def _resolve_split_file_path(
    data_root: Path,
    split_file: str | Path | None,
    train_ratio: float,
    val_ratio: float,
    split_seed: int,
    split_unit: SplitUnit,
) -> Path:
    if split_file is not None:
        return Path(split_file)

    split_name = f"{split_unit}_split_tr{train_ratio:.3f}_val{val_ratio:.3f}_seed{split_seed}.json"
    return data_root / "splits" / split_name


def _split_scene_ids_persistent(
    scene_ids: list[str],
    split: Literal["train", "val", "test", "all"],
    train_ratio: float,
    val_ratio: float,
    split_file: Path,
    split_seed: int,
) -> set[str]:
    if split == "all":
        return set(scene_ids)

    scene_ids_sorted = sorted(scene_ids)
    split_file.parent.mkdir(parents=True, exist_ok=True)

    loaded: dict | None = None
    if split_file.exists():
        with split_file.open("r", encoding="utf-8") as f:
            loaded = json.load(f)

    should_rebuild = True
    if loaded is not None:
        loaded_ids = sorted(loaded.get("scene_ids", []))
        loaded_train_ratio = float(loaded.get("meta", {}).get("train_ratio", -1.0))
        loaded_val_ratio = float(loaded.get("meta", {}).get("val_ratio", -1.0))
        if (
            loaded_ids == scene_ids_sorted
            and abs(loaded_train_ratio - train_ratio) < 1e-9
            and abs(loaded_val_ratio - val_ratio) < 1e-9
        ):
            should_rebuild = False

    if should_rebuild:
        rng = np.random.default_rng(split_seed)
        shuffled = scene_ids_sorted.copy()
        rng.shuffle(shuffled)

        n_total = len(shuffled)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        train_ids = shuffled[:n_train]
        val_ids = shuffled[n_train : n_train + n_val]
        test_ids = shuffled[n_train + n_val :]

        payload = {
            "scene_ids": scene_ids_sorted,
            "train": sorted(train_ids),
            "val": sorted(val_ids),
            "test": sorted(test_ids),
            "meta": {
                "train_ratio": train_ratio,
                "val_ratio": val_ratio,
                "split_seed": split_seed,
            },
        }
        with split_file.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)
        loaded = payload

    assert loaded is not None
    selected = loaded.get(split, [])
    if not selected:
        raise ValueError(
            f"Split '{split}' is empty in split file {split_file}. "
            f"scene_count={len(scene_ids_sorted)}, train_ratio={train_ratio}, val_ratio={val_ratio}"
        )
    return set(selected)


def _split_samples_persistent(
    samples: list[C2SMSFloodSamplePaths],
    split: Literal["train", "val", "test", "all"],
    train_ratio: float,
    val_ratio: float,
    split_file: Path,
    split_seed: int,
) -> list[C2SMSFloodSamplePaths]:
    if split == "all":
        return samples

    split_file.parent.mkdir(parents=True, exist_ok=True)
    sample_names = [sample.name for sample in samples]

    loaded: dict | None = None
    if split_file.exists():
        with split_file.open("r", encoding="utf-8") as f:
            loaded = json.load(f)

    should_rebuild = True
    if loaded is not None:
        loaded_names = loaded.get("sample_names", [])
        loaded_train_ratio = float(loaded.get("meta", {}).get("train_ratio", -1.0))
        loaded_val_ratio = float(loaded.get("meta", {}).get("val_ratio", -1.0))
        loaded_split_unit = str(loaded.get("meta", {}).get("split_unit", ""))
        if (
            loaded_names == sample_names
            and abs(loaded_train_ratio - train_ratio) < 1e-9
            and abs(loaded_val_ratio - val_ratio) < 1e-9
            and loaded_split_unit == "chip"
        ):
            should_rebuild = False

    if should_rebuild:
        rng = np.random.default_rng(split_seed)
        indices = np.arange(len(samples))
        rng.shuffle(indices)
        shuffled_names = [sample_names[i] for i in indices]

        n_total = len(shuffled_names)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        train_names = shuffled_names[:n_train]
        val_names = shuffled_names[n_train : n_train + n_val]
        test_names = shuffled_names[n_train + n_val :]

        payload = {
            "sample_names": sample_names,
            "train": train_names,
            "val": val_names,
            "test": test_names,
            "meta": {
                "train_ratio": train_ratio,
                "val_ratio": val_ratio,
                "split_seed": split_seed,
                "split_unit": "chip",
            },
        }
        with split_file.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)
        loaded = payload

    assert loaded is not None
    selected = set(loaded.get(split, []))
    if not selected:
        raise ValueError(
            f"Split '{split}' is empty in split file {split_file}. "
            f"sample_count={len(samples)}, train_ratio={train_ratio}, val_ratio={val_ratio}"
        )
    return [sample for sample in samples if sample.name in selected]


def _extract_chip_position(chip_name: str) -> str:
    return chip_name.split("_")[-1]


def _collect_s1_samples(scene_dir: Path, scene_id: str) -> list[C2SMSFloodSamplePaths]:
    s1_dir = scene_dir / "s1"
    if not s1_dir.is_dir():
        return []

    out: list[C2SMSFloodSamplePaths] = []
    for chip_dir in sorted((p for p in s1_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
        vv = chip_dir / "VV.tif"
        vh = chip_dir / "VH.tif"
        label = chip_dir / "LabelWater.tif"
        if vv.exists() and vh.exists() and label.exists():
            out.append(C2SMSFloodSamplePaths(name=f"{scene_id}/{chip_dir.name}", scene_id=scene_id, s1_dir=chip_dir))
    return out


def _collect_s2_samples(scene_dir: Path, scene_id: str) -> list[C2SMSFloodSamplePaths]:
    s2_dir = scene_dir / "s2"
    if not s2_dir.is_dir():
        return []

    out: list[C2SMSFloodSamplePaths] = []
    for chip_dir in sorted((p for p in s2_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
        label = chip_dir / "LabelWater.tif"
        rgb = chip_dir / "RGB.png"
        swir = chip_dir / "SWIR.png"
        band_files = list(chip_dir.glob("B*.tif"))
        if label.exists() and rgb.exists() and swir.exists() and len(band_files) > 0:
            out.append(C2SMSFloodSamplePaths(name=f"{scene_id}/{chip_dir.name}", scene_id=scene_id, s2_dir=chip_dir))
    return out


def _collect_all_samples(scene_dir: Path, scene_id: str) -> list[C2SMSFloodSamplePaths]:
    s1_root = scene_dir / "s1"
    s2_root = scene_dir / "s2"
    if not (s1_root.is_dir() and s2_root.is_dir()):
        return []

    s1_by_pos: dict[str, Path] = {}
    for chip_dir in s1_root.iterdir():
        if not chip_dir.is_dir():
            continue
        if not (
            (chip_dir / "VV.tif").exists() and (chip_dir / "VH.tif").exists() and (chip_dir / "LabelWater.tif").exists()
        ):
            continue
        s1_by_pos[_extract_chip_position(chip_dir.name)] = chip_dir

    s2_by_pos: dict[str, Path] = {}
    for chip_dir in s2_root.iterdir():
        if not chip_dir.is_dir():
            continue
        if not (
            (chip_dir / "RGB.png").exists()
            and (chip_dir / "SWIR.png").exists()
            and (chip_dir / "LabelWater.tif").exists()
        ):
            continue
        if len(list(chip_dir.glob("B*.tif"))) == 0:
            continue
        s2_by_pos[_extract_chip_position(chip_dir.name)] = chip_dir

    out: list[C2SMSFloodSamplePaths] = []
    for pos in sorted(set(s1_by_pos) & set(s2_by_pos)):
        out.append(
            C2SMSFloodSamplePaths(
                name=f"{scene_id}/{pos}",
                scene_id=scene_id,
                s1_dir=s1_by_pos[pos],
                s2_dir=s2_by_pos[pos],
            )
        )
    return out


def _collect_samples(
    data_root: Path,
    chips_root: Path,
    sensor: SensorMode,
    split_unit: SplitUnit,
    split: Literal["train", "val", "test", "all"],
    train_ratio: float,
    val_ratio: float,
    split_file: str | Path | None,
    split_seed: int,
) -> list[C2SMSFloodSamplePaths]:
    scene_dirs = _list_scene_dirs(chips_root)
    split_file_path = _resolve_split_file_path(
        data_root=data_root,
        split_file=split_file,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        split_seed=split_seed,
        split_unit=split_unit,
    )

    samples: list[C2SMSFloodSamplePaths] = []
    if split_unit == "scene":
        selected_scene_ids = _split_scene_ids_persistent(
            scene_ids=[p.name for p in scene_dirs],
            split=split,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            split_file=split_file_path,
            split_seed=split_seed,
        )

        for scene_dir in scene_dirs:
            scene_id = scene_dir.name
            if scene_id not in selected_scene_ids:
                continue

            if sensor == "s1":
                samples.extend(_collect_s1_samples(scene_dir, scene_id=scene_id))
            elif sensor in {"s2_rgb_swir", "s2_all"}:
                samples.extend(_collect_s2_samples(scene_dir, scene_id=scene_id))
            else:
                samples.extend(_collect_all_samples(scene_dir, scene_id=scene_id))
    else:
        for scene_dir in scene_dirs:
            scene_id = scene_dir.name
            if sensor == "s1":
                samples.extend(_collect_s1_samples(scene_dir, scene_id=scene_id))
            elif sensor in {"s2_rgb_swir", "s2_all"}:
                samples.extend(_collect_s2_samples(scene_dir, scene_id=scene_id))
            else:
                samples.extend(_collect_all_samples(scene_dir, scene_id=scene_id))
        samples = _split_samples_persistent(
            samples=samples,
            split=split,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            split_file=split_file_path,
            split_seed=split_seed,
        )

    if not samples:
        raise RuntimeError(f"No valid c2smsfloods samples found under {chips_root} for sensor={sensor}, split={split}.")
    return samples


def _load_tif(path: Path) -> Tensor:
    arr = np.asarray(Image.open(path), dtype=np.float32).copy()
    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim == 3:
        arr = np.transpose(arr, (2, 0, 1))
    else:
        raise ValueError(f"Unsupported tif shape: {arr.shape} for {path}")
    return torch.from_numpy(arr)


def _load_rgb(path: Path) -> Tensor:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32).copy()
    return torch.from_numpy(arr).permute(2, 0, 1)


def _load_binary_mask(path: Path) -> Tensor:
    arr = np.asarray(Image.open(path), dtype=np.uint8).copy()
    if arr.ndim == 3:
        arr = arr[..., 0]
    mask = (arr > 0).astype(np.uint8)
    return torch.from_numpy(mask).long().unsqueeze(0)


def _resize_tensor_to(tensor: Tensor, target_hw: tuple[int, int]) -> Tensor:
    if tuple(tensor.shape[-2:]) == tuple(target_hw):
        return tensor
    return F.interpolate(tensor.unsqueeze(0), size=target_hw, mode="bilinear", align_corners=False).squeeze(0)


def _to_hw(size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(size, int):
        return (size, size)
    return size


def _sample_crop_top_left(
    h: int,
    w: int,
    crop_h: int,
    crop_w: int,
    random_crop: bool,
) -> tuple[int, int]:
    max_top = max(h - crop_h, 0)
    max_left = max(w - crop_w, 0)
    if random_crop and max_top > 0:
        top = int(torch.randint(0, max_top + 1, (1,)).item())
    else:
        top = max_top // 2
    if random_crop and max_left > 0:
        left = int(torch.randint(0, max_left + 1, (1,)).item())
    else:
        left = max_left // 2
    return top, left


def _crop_tensor(tensor: Tensor, top: int, left: int, crop_h: int, crop_w: int) -> Tensor:
    return tensor[..., top : top + crop_h, left : left + crop_w]


def _resize_image_tensor(tensor: Tensor, target_hw: tuple[int, int]) -> Tensor:
    if tuple(tensor.shape[-2:]) == tuple(target_hw):
        return tensor
    return F.interpolate(tensor.unsqueeze(0), size=target_hw, mode="bilinear", align_corners=False).squeeze(0)


def _resize_mask_tensor(tensor: Tensor, target_hw: tuple[int, int]) -> Tensor:
    if tuple(tensor.shape[-2:]) == tuple(target_hw):
        return tensor
    return F.interpolate(tensor.float().unsqueeze(0), size=target_hw, mode="nearest").squeeze(0).long()


def _apply_crop_resize(
    modalities: dict[str, Tensor],
    gt: Tensor,
    crop_size: int | tuple[int, int] | None,
    resize_to: int | tuple[int, int] | None,
    random_crop: bool,
) -> tuple[dict[str, Tensor], Tensor]:
    h, w = gt.shape[-2:]
    if crop_size is not None:
        crop_h, crop_w = _to_hw(crop_size)
        crop_h = min(crop_h, h)
        crop_w = min(crop_w, w)
        top, left = _sample_crop_top_left(h, w, crop_h=crop_h, crop_w=crop_w, random_crop=random_crop)
        modalities = {k: _crop_tensor(v, top, left, crop_h, crop_w) for k, v in modalities.items()}
        gt = _crop_tensor(gt, top, left, crop_h, crop_w)

    if resize_to is not None:
        target_hw = _to_hw(resize_to)
        modalities = {k: _resize_image_tensor(v, target_hw=target_hw) for k, v in modalities.items()}
        gt = _resize_mask_tensor(gt, target_hw=target_hw)

    return modalities, gt


def _normalize_per_channel_minmax(image: Tensor, to_neg_1_1: bool) -> Tensor:
    ch_min = image.amin(dim=(1, 2), keepdim=True)
    ch_max = image.amax(dim=(1, 2), keepdim=True)
    ch_range = (ch_max - ch_min).clamp(min=1e-6)
    image = ((image - ch_min) / ch_range).clamp(0.0, 1.0)
    if to_neg_1_1:
        image = image * 2.0 - 1.0
    return image


def _normalize_s1_baseline(image: Tensor, min_normalize: float, max_normalize: float, to_neg_1_1: bool) -> Tensor:
    scale = max(max_normalize - min_normalize, 1e-6)
    image = torch.clamp(image, min=min_normalize, max=max_normalize)
    image = (image - min_normalize) / scale
    if to_neg_1_1:
        image = image * 2.0 - 1.0
    return image


def _normalize_s2_rgb_swir_baseline(image: Tensor, to_neg_1_1: bool) -> Tensor:
    image = torch.clamp(image / 255.0, min=0.0, max=1.0)
    if to_neg_1_1:
        image = image * 2.0 - 1.0
    return image


def _normalize_s2_all_baseline(image: Tensor, min_value: float, max_value: float, to_neg_1_1: bool) -> Tensor:
    scale = max(max_value - min_value, 1e-6)
    image = torch.clamp(image, min=min_value, max=max_value)
    image = (image - min_value) / scale
    if to_neg_1_1:
        image = image * 2.0 - 1.0
    return image


def _sort_s2_band_paths(paths: list[Path]) -> list[Path]:
    order = {
        "B1": 1,
        "B2": 2,
        "B3": 3,
        "B4": 4,
        "B5": 5,
        "B6": 6,
        "B7": 7,
        "B8": 8,
        "B8A": 9,
        "B9": 10,
        "B10": 11,
        "B11": 12,
        "B12": 13,
    }

    def _key(path: Path) -> tuple[int, str]:
        stem = path.stem.upper()
        return order.get(stem, 999), stem

    return sorted(paths, key=_key)


def _load_s1_img_gt(s1_dir: Path) -> tuple[Tensor, Tensor]:
    vv = _load_tif(s1_dir / "VV.tif")
    vh = _load_tif(s1_dir / "VH.tif")
    gt = _load_binary_mask(s1_dir / "LabelWater.tif")
    target_hw = gt.shape[-2:]
    vv = _resize_tensor_to(vv, target_hw=target_hw)
    vh = _resize_tensor_to(vh, target_hw=target_hw)
    img = torch.cat([vv, vh], dim=0)
    return img, gt


def _load_s2_rgb_swir_img_gt(s2_dir: Path) -> tuple[Tensor, Tensor]:
    rgb = _load_rgb(s2_dir / "RGB.png")
    swir = _load_rgb(s2_dir / "SWIR.png")
    gt = _load_binary_mask(s2_dir / "LabelWater.tif")
    target_hw = gt.shape[-2:]
    rgb = _resize_tensor_to(rgb, target_hw=target_hw)
    swir = _resize_tensor_to(swir, target_hw=target_hw)
    img = torch.cat([rgb, swir], dim=0)
    return img, gt


def _load_s2_all_img_gt(s2_dir: Path) -> tuple[Tensor, Tensor]:
    band_paths = _sort_s2_band_paths(list(s2_dir.glob("B*.tif")))
    if not band_paths:
        raise RuntimeError(f"No S2 band file found under {s2_dir}")

    gt = _load_binary_mask(s2_dir / "LabelWater.tif")
    target_hw = gt.shape[-2:]

    bands: list[Tensor] = []
    for path in band_paths:
        band = _load_tif(path)
        band = _resize_tensor_to(band, target_hw=target_hw)
        bands.append(band)
    img = torch.cat(bands, dim=0)
    return img, gt


class C2SMSFloodChangeDetectionDataset(Dataset[dict[str, Tensor | str]]):
    def __init__(
        self,
        data_root: str | Path = "data/Downstreams/c2smsfloods",
        split: Literal["train", "val", "test", "all"] = "train",
        sensor: SensorMode = "s1",
        split_unit: SplitUnit = "scene",
        chips_dir_name: str = "ms-dataset-chips",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        split_file: str | Path | None = None,
        split_seed: int = 1778,
        crop_size: int | tuple[int, int] | None = None,
        resize_to: int | tuple[int, int] | None = None,
        random_crop: bool = False,
        transform: AugmentationSequential | None | str = "default",
        normalize: bool = True,
        normalization_strategy: Literal["baseline", "per_sample"] = "baseline",
        img_to_neg1_1: bool = False,
        s1_min_normalize: float = -77.0,
        s1_max_normalize: float = 26.0,
        s2_all_min_normalize: float = 0.0,
        s2_all_max_normalize: float = 10000.0,
        return_name: bool = False,
    ) -> None:
        if not (0.0 < train_ratio < 1.0):
            raise ValueError(f"train_ratio must be in (0,1), got {train_ratio}")
        if not (0.0 <= val_ratio < 1.0):
            raise ValueError(f"val_ratio must be in [0,1), got {val_ratio}")
        if train_ratio + val_ratio >= 1.0:
            raise ValueError(f"train_ratio + val_ratio must be < 1, got {train_ratio + val_ratio}")

        self.data_root = Path(data_root)
        self.chips_root = self.data_root / chips_dir_name
        if not self.chips_root.is_dir():
            chips_fallback = self.data_root / "chips"
            ms_chips_fallback = self.data_root / "ms-dataset-chips"
            raise FileNotFoundError(
                f"Missing directory: {self.chips_root}. "
                f"Available common candidates may include: {chips_fallback}, {ms_chips_fallback}"
            )

        self.split = split
        self.sensor = sensor
        self.split_unit = split_unit
        self.crop_size = crop_size
        self.resize_to = resize_to
        self.random_crop = random_crop
        self.normalize = normalize
        self.normalization_strategy = normalization_strategy
        self.img_to_neg1_1 = img_to_neg1_1
        self.s1_min_normalize = s1_min_normalize
        self.s1_max_normalize = s1_max_normalize
        self.s2_all_min_normalize = s2_all_min_normalize
        self.s2_all_max_normalize = s2_all_max_normalize
        self.return_name = return_name

        self.samples = _collect_samples(
            data_root=self.data_root,
            chips_root=self.chips_root,
            sensor=sensor,
            split_unit=split_unit,
            split=split,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            split_file=split_file,
            split_seed=split_seed,
        )

        if transform == "default":
            transform = get_default_transform()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def _apply_normalize(self, image: Tensor, modality: Literal["s1", "s2_rgb_swir", "s2_all"]) -> Tensor:
        image = torch.nan_to_num(image, nan=0.0)
        if self.normalization_strategy == "per_sample":
            return _normalize_per_channel_minmax(image, to_neg_1_1=self.img_to_neg1_1)

        if modality == "s1":
            return _normalize_s1_baseline(
                image,
                min_normalize=self.s1_min_normalize,
                max_normalize=self.s1_max_normalize,
                to_neg_1_1=self.img_to_neg1_1,
            )
        if modality == "s2_rgb_swir":
            return _normalize_s2_rgb_swir_baseline(image, to_neg_1_1=self.img_to_neg1_1)
        return _normalize_s2_all_baseline(
            image,
            min_value=self.s2_all_min_normalize,
            max_value=self.s2_all_max_normalize,
            to_neg_1_1=self.img_to_neg1_1,
        )

    def _sensor_to_modality(self, sensor: SensorMode) -> Literal["s1", "s2_rgb_swir", "s2_all"]:
        modality_map: dict[SensorMode, Literal["s1", "s2_rgb_swir", "s2_all"]] = {
            "s1": "s1",
            "s2_rgb_swir": "s2_rgb_swir",
            "s2_all": "s2_all",
            "all": "s1",
        }
        return modality_map[sensor]

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        sample = self.samples[index]

        if self.sensor == "s1":
            if sample.s1_dir is None:
                raise RuntimeError("s1_dir is missing for sensor='s1'")
            img, gt = _load_s1_img_gt(sample.s1_dir)
            modalities, gt = _apply_crop_resize(
                modalities={"img": img},
                gt=gt,
                crop_size=self.crop_size,
                resize_to=self.resize_to,
                random_crop=self.random_crop,
            )
            img = modalities["img"]

            if self.normalize:
                img = self._apply_normalize(img, modality=self._sensor_to_modality(self.sensor))

            if self.transform is not None:
                img, gt = self.transform(img.unsqueeze(0), gt.float().unsqueeze(0))
                img = img.squeeze(0)
                gt = gt.squeeze(0).long()

            out: dict[str, Tensor | str] = {"img": img, "gt": gt}
            if self.return_name:
                out["name"] = sample.name
            return out

        if self.sensor == "s2_rgb_swir":
            if sample.s2_dir is None:
                raise RuntimeError("s2_dir is missing for sensor='s2_rgb_swir'")
            img, gt = _load_s2_rgb_swir_img_gt(sample.s2_dir)
            modalities, gt = _apply_crop_resize(
                modalities={"img": img},
                gt=gt,
                crop_size=self.crop_size,
                resize_to=self.resize_to,
                random_crop=self.random_crop,
            )
            img = modalities["img"]

            if self.normalize:
                img = self._apply_normalize(img, modality=self._sensor_to_modality(self.sensor))

            if self.transform is not None:
                img, gt = self.transform(img.unsqueeze(0), gt.float().unsqueeze(0))
                img = img.squeeze(0)
                gt = gt.squeeze(0).long()

            out = {"img": img, "gt": gt}
            if self.return_name:
                out["name"] = sample.name
            return out

        if self.sensor == "s2_all":
            if sample.s2_dir is None:
                raise RuntimeError("s2_dir is missing for sensor='s2_all'")
            img, gt = _load_s2_all_img_gt(sample.s2_dir)
            modalities, gt = _apply_crop_resize(
                modalities={"img": img},
                gt=gt,
                crop_size=self.crop_size,
                resize_to=self.resize_to,
                random_crop=self.random_crop,
            )
            img = modalities["img"]

            if self.normalize:
                img = self._apply_normalize(img, modality=self._sensor_to_modality(self.sensor))

            if self.transform is not None:
                img, gt = self.transform(img.unsqueeze(0), gt.float().unsqueeze(0))
                img = img.squeeze(0)
                gt = gt.squeeze(0).long()

            out = {"img": img, "gt": gt}
            if self.return_name:
                out["name"] = sample.name
            return out

        if sample.s1_dir is None or sample.s2_dir is None:
            raise RuntimeError("s1_dir or s2_dir is missing for sensor='all'")

        s1, gt_s1 = _load_s1_img_gt(sample.s1_dir)
        s2_rgb, gt_s2_rgb = _load_s2_rgb_swir_img_gt(sample.s2_dir)
        s2_all, gt_s2_all = _load_s2_all_img_gt(sample.s2_dir)

        gt = gt_s1
        modalities, gt = _apply_crop_resize(
            modalities={"s1": s1, "s2_rgb": s2_rgb, "s2_all": s2_all},
            gt=gt,
            crop_size=self.crop_size,
            resize_to=self.resize_to,
            random_crop=self.random_crop,
        )
        s1 = modalities["s1"]
        s2_rgb = modalities["s2_rgb"]
        s2_all = modalities["s2_all"]

        if self.normalize:
            s1 = self._apply_normalize(s1, modality="s1")
            s2_rgb = self._apply_normalize(s2_rgb, modality="s2_rgb_swir")
            s2_all = self._apply_normalize(s2_all, modality="s2_all")

        if self.transform is not None:
            c1 = s1.shape[0]
            c2 = s2_rgb.shape[0]
            merged = torch.cat([s1, s2_rgb, s2_all], dim=0)
            merged, gt = self.transform(merged.unsqueeze(0), gt.float().unsqueeze(0))
            merged = merged.squeeze(0)
            gt = gt.squeeze(0).long()
            s1 = merged[:c1]
            s2_rgb = merged[c1 : c1 + c2]
            s2_all = merged[c1 + c2 :]

        out = {"s1": s1, "s2_rgb": s2_rgb, "s2_all": s2_all, "gt": gt}
        if self.return_name:
            out["name"] = sample.name
        return out


def create_c2smsfloods_segmentation_dataloader(
    data_root: str | Path = "data/Downstreams/c2smsfloods",
    split: Literal["train", "val", "test", "all"] = "train",
    sensor: SensorMode = "s1",
    split_unit: SplitUnit = "scene",
    chips_dir_name: str = "chips",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    split_file: str | Path | None = None,
    split_seed: int = 1778,
    crop_size: int | tuple[int, int] | None = None,
    resize_to: int | tuple[int, int] | None = None,
    random_crop: bool = False,
    batch_size: int = 4,
    shuffle: bool | None = None,
    num_workers: int = 4,
    transform: AugmentationSequential | None | str = "default",
    normalize: bool = True,
    normalization_strategy: Literal["baseline", "per_sample"] = "baseline",
    img_to_neg1_1: bool = False,
    s1_min_normalize: float = -77.0,
    s1_max_normalize: float = 26.0,
    s2_all_min_normalize: float = 0.0,
    s2_all_max_normalize: float = 10000.0,
    return_name: bool = False,
    pin_memory: bool = True,
    drop_last: bool = False,
    **loader_kwargs,
) -> tuple[C2SMSFloodChangeDetectionDataset, DataLoader]:
    dataset = C2SMSFloodChangeDetectionDataset(
        data_root=data_root,
        split=split,
        sensor=sensor,
        split_unit=split_unit,
        chips_dir_name=chips_dir_name,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        split_file=split_file,
        split_seed=split_seed,
        crop_size=crop_size,
        resize_to=resize_to,
        random_crop=random_crop,
        transform=transform,
        normalize=normalize,
        normalization_strategy=normalization_strategy,
        img_to_neg1_1=img_to_neg1_1,
        s1_min_normalize=s1_min_normalize,
        s1_max_normalize=s1_max_normalize,
        s2_all_min_normalize=s2_all_min_normalize,
        s2_all_max_normalize=s2_all_max_normalize,
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
