from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from src.data import _BaseStreamingDataset

type SplitName = Literal["train", "val", "test"]
type RescaleMethod = Literal["default", "resnet"]


@dataclass(frozen=True)
class MetaRow:
    index: int
    key: str
    split: str
    split_id: int | None
    season: str | None
    scene_id: int | None
    patch_id: int | None
    s1_path: str | None
    s2_path: str | None
    s2_cloudy_path: str | None


def _rescale(img: torch.Tensor, old_min: float, old_max: float) -> torch.Tensor:
    old_range = old_max - old_min
    return (img - old_min) / old_range


def _process_ms(img: torch.Tensor, method: RescaleMethod) -> torch.Tensor:
    if method == "default":
        intensity_min, intensity_max = 0.0, 10000.0
        img = torch.clamp(img, intensity_min, intensity_max)
        img = _rescale(img, intensity_min, intensity_max)
    elif method == "resnet":
        intensity_min, intensity_max = 0.0, 10000.0
        img = torch.clamp(img, intensity_min, intensity_max)
        img = img / 2000.0
    else:
        raise ValueError(f"Unsupported rescale method: {method}")
    return torch.nan_to_num(img)


def _process_sar(img: torch.Tensor, method: RescaleMethod) -> torch.Tensor:
    if method == "default":
        db_min, db_max = -25.0, 0.0
        img = torch.clamp(img, db_min, db_max)
        img = _rescale(img, db_min, db_max)
    elif method == "resnet":
        db_min = torch.tensor([-25.0, -32.5], dtype=img.dtype, device=img.device)
        db_max = torch.tensor([0.0, 0.0], dtype=img.dtype, device=img.device)
        if img.shape[0] != 2:
            raise ValueError(f"Expected SAR with 2 channels for resnet mode, got {img.shape}")
        part0 = 2 * (torch.clamp(img[0], db_min[0], db_max[0]) - db_min[0]) / (db_max[0] - db_min[0])
        part1 = 2 * (torch.clamp(img[1], db_min[1], db_max[1]) - db_min[1]) / (db_max[1] - db_min[1])
        img = torch.stack([part0, part1], dim=0)
    else:
        raise ValueError(f"Unsupported rescale method: {method}")
    return torch.nan_to_num(img)


def _load_meta_rows(meta_path: Path) -> list[MetaRow]:
    rows: list[MetaRow] = []
    with meta_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for line_no, row in enumerate(reader, start=2):
            if not row:
                continue
            if "index" not in row or "__key__" not in row or "split" not in row:
                raise ValueError(f"Missing required columns in meta csv at line {line_no}: {row}")
            rows.append(
                MetaRow(
                    index=int(row["index"]),
                    key=row["__key__"],
                    split=row["split"],
                    split_id=int(row["split_id"]) if row.get("split_id") else None,
                    season=row.get("season"),
                    scene_id=int(row["scene_id"]) if row.get("scene_id") else None,
                    patch_id=int(row["patch_id"]) if row.get("patch_id") else None,
                    s1_path=row.get("s1_path"),
                    s2_path=row.get("s2_path"),
                    s2_cloudy_path=row.get("s2_cloudy_path"),
                )
            )
    return rows


def _ensure_tensor(img: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(img, np.ndarray):
        return torch.from_numpy(img)
    return img


class SEN12_CR_StreamingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        input_dir: str,
        split: SplitName,
        *,
        to_neg_1_1: bool = False,
        rescale_method: RescaleMethod = "default",
        return_meta: bool = True,
        validate_meta_index: bool = True,
        s1_dir_name: str = "litdata_s1",
        s2_dir_name: str = "litdata_s2_clean",
        s2_cloudy_dir_name: str = "litdata_s2_cloudy",
        meta_dir_name: str = "litdata_meta",
        **stream_kwargs,
    ):
        base_dir = Path(input_dir)
        s1_dir = base_dir / s1_dir_name / split
        s2_dir = base_dir / s2_dir_name / split
        s2_cloudy_dir = base_dir / s2_cloudy_dir_name / split

        if not s1_dir.exists():
            raise FileNotFoundError(s1_dir)
        if not s2_dir.exists():
            raise FileNotFoundError(s2_dir)
        if not s2_cloudy_dir.exists():
            raise FileNotFoundError(s2_cloudy_dir)

        self.s1_ds = _BaseStreamingDataset.create_dataset(s1_dir.as_posix(), **stream_kwargs)
        self.s2_ds = _BaseStreamingDataset.create_dataset(s2_dir.as_posix(), **stream_kwargs)
        self.s2_cloudy_ds = _BaseStreamingDataset.create_dataset(s2_cloudy_dir.as_posix(), **stream_kwargs)

        lengths = (len(self.s1_ds), len(self.s2_ds), len(self.s2_cloudy_ds))
        if len(set(lengths)) != 1:
            raise ValueError(f"Dataset lengths mismatch: s1={lengths[0]}, s2={lengths[1]}, s2_cloudy={lengths[2]}")

        self.split = split
        self.rescale_method = rescale_method
        self.to_neg_1_1 = to_neg_1_1

        self._meta_rows: list[MetaRow] | None = None
        if return_meta:
            meta_path = base_dir / meta_dir_name / f"{split}.csv"
            if not meta_path.exists():
                raise FileNotFoundError(meta_path)
            self._meta_rows = _load_meta_rows(meta_path)
            if validate_meta_index:
                if len(self._meta_rows) != lengths[0]:
                    raise ValueError(
                        f"Meta rows mismatch: meta={len(self._meta_rows)}, data={lengths[0]}",
                    )
                for idx, row in enumerate(self._meta_rows):
                    if row.index != idx:
                        raise ValueError(f"Meta index mismatch at {idx}: {row.index}")

    def __len__(self) -> int:
        return len(self.s1_ds)

    def _normalize_ms(self, img: torch.Tensor | np.ndarray) -> torch.Tensor:
        img = _ensure_tensor(img).float()
        img = _process_ms(img, self.rescale_method)
        if self.to_neg_1_1 and self.rescale_method == "default":
            img = img * 2.0 - 1.0
        return img

    def _normalize_sar(self, img: torch.Tensor | np.ndarray) -> torch.Tensor:
        img = _ensure_tensor(img).float()
        img = _process_sar(img, self.rescale_method)
        if self.to_neg_1_1 and self.rescale_method == "default":
            img = img * 2.0 - 1.0
        return img

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | MetaRow]:
        s1 = self._normalize_sar(self.s1_ds[idx]["img"])
        s2 = self._normalize_ms(self.s2_ds[idx]["img"])
        s2_cloudy = self._normalize_ms(self.s2_cloudy_ds[idx]["img"])

        sample: dict[str, torch.Tensor | MetaRow] = {
            "s1": s1,
            "s2": s2,
            "s2_cloudy": s2_cloudy,
            "img": s2_cloudy,
            "gt": s2,
            "target": s2,
            "conditions": s2_cloudy,
        }
        if self._meta_rows is not None:
            sample["meta"] = self._meta_rows[idx]
        return sample

    @classmethod
    def create_dataset(
        cls,
        input_dir: str,
        split: SplitName,
        **kwargs,
    ) -> "SEN12_CR_StreamingDataset":
        return cls(input_dir=input_dir, split=split, **kwargs)
