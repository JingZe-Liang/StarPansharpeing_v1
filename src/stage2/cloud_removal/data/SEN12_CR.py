from __future__ import annotations

import os
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import litdata as ld
from loguru import logger
from torch.utils.data import Dataset
from litdata import ParallelStreamingDataset

from src.data import _BaseStreamingDataset


logger = logger.bind(_name_="SEN12_CR")

type SplitName = Literal["train", "val", "test"]
type RescaleMethod = Literal["default", "resnet", "default_per_channel", "quantile"]


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


@dataclass(frozen=True)
class MetaRowBatched:
    index: torch.Tensor
    key: list[str]
    split: list[str]
    split_id: list[int | None]
    season: list[str | None]
    scene_id: list[int | None]
    patch_id: list[int | None]
    s1_path: list[str | None]
    s2_path: list[str | None]
    s2_cloudy_path: list[str | None]


def _rescale(
    img: torch.Tensor,
    old_min: float | torch.Tensor | None = None,
    old_max: float | torch.Tensor | None = None,
    per_channel: bool = False,
) -> torch.Tensor:
    if old_min is None or old_max is None:
        if per_channel:
            old_min, old_max = img.min(dim=-1, keepdim=True).values, img.max(dim=-1, keepdim=True).values
        else:
            old_min, old_max = img.min(), img.max()
    old_range = old_max - old_min
    return (img - old_min) / old_range


def _process_ms(
    img: torch.Tensor,
    method: RescaleMethod,
    quantile_percents: tuple[float, float] | None = None,
    intensity_min_max: tuple = (0, 10000),
) -> torch.Tensor:
    if method in ("default", "default_per_channel"):
        intensity_min, intensity_max = intensity_min_max
        img = torch.clamp(img, intensity_min, intensity_max)
        per_channel = "per_channel" in method
        img = _rescale(img, per_channel=per_channel)  # , intensity_min, intensity_max)
    elif method == "quantile":
        assert quantile_percents is not None
        q1, q2 = quantile_percents
        intensity_min, intensity_max = intensity_min_max
        img = torch.clamp(img, intensity_min, intensity_max)
        c = img.shape[0]
        flat = img.reshape(c, -1)
        q1v = torch.quantile(flat, q1, dim=1, keepdim=True)
        q2v = torch.quantile(flat, q2, dim=1, keepdim=True)
        img = torch.clamp(img, q1v[:, None, None], q2v[:, None, None])
        img = _rescale(img, per_channel=True)
    elif method == "resnet":
        intensity_min, intensity_max = intensity_min_max
        img = torch.clamp(img, intensity_min, intensity_max)
        img = img / 2000.0
    else:
        raise ValueError(f"Unsupported rescale method: {method}")
    return torch.nan_to_num(img)


def _process_sar(
    img: torch.Tensor,
    method: RescaleMethod,
    quantile_percents: tuple[float, float] | None = None,
    *,
    apply_log1p: bool = False,
) -> torch.Tensor:
    if method in ("default", "default_per_channel"):
        db_min, db_max = -25.0, 0.0
        img = torch.clamp(img, db_min, db_max)
        per_channel = "per_channel" in method
        img = _rescale(img, per_channel=per_channel)  # , db_min, db_max)
    elif method == "quantile":
        assert quantile_percents is not None
        q1, q2 = quantile_percents
        db_min, db_max = -25.0, 0.0
        img = torch.clamp(img, db_min, db_max)
        c = img.shape[0]
        flat = img.reshape(c, -1)
        q1v = torch.quantile(flat, q1, dim=1, keepdim=True)
        q2v = torch.quantile(flat, q2, dim=1, keepdim=True)
        img = torch.clamp(img, q1v[:, None, None], q2v[:, None, None])
        img = _rescale(img, per_channel=True)
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
    img = torch.nan_to_num(img)

    if apply_log1p and method != "resnet":
        img = torch.log1p(torch.clamp_min(img, 0.0))
        img = img / float(np.log(2.0))  # keep in [0, 1] when img in [0, 1]

    return img


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


def _to_chw(img: torch.Tensor) -> torch.Tensor:
    if img.ndim != 3:
        return img
    if img.shape[0] in (2, 13):
        return img  # CHW
    if img.shape[-1] in (2, 13):
        return img.permute(2, 0, 1)  # HWC -> CHW
    if img.shape[1] in (2, 13):
        return img.permute(1, 2, 0)  # WCH -> CHW (litdata tiff transpose artifact)
    return img


_inherited_from = os.getenv("SEN12_DATA_CLS", "litdata")
_inherited_cls = ParallelStreamingDataset if _inherited_from == "litdata" else Dataset


class SEN12_CR_StreamingDataset(_inherited_cls):
    def __init__(
        self,
        input_dir: str,
        split: SplitName,
        *,
        to_neg_1_1: bool = False,
        rescale_method: RescaleMethod = "default",
        sar_log1p: bool = False,
        return_meta: bool = True,
        validate_meta_index: bool = True,
        s1_dir_name: str = "litdata_s1",
        s2_dir_name: str = "litdata_s2_clean",
        s2_cloudy_dir_name: str = "litdata_s2_cloudy",
        meta_dir_name: str = "litdata_meta",
        intensity_min_max: tuple[int, int] = (0, 10000),
        use_quantile: bool = True,
        quantile_kwargs: dict[str, tuple[float, float]] | None = None,
        no_s1: bool = False,
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

        self.no_s1 = no_s1
        self.no_metainfo = not return_meta
        self.intensity_min_max = intensity_min_max

        # mro init
        global _inherited_from
        if _inherited_from == "litdata":
            super().__init__([self.s1_ds, self.s2_ds, self.s2_cloudy_ds], transform=self.transform)  # type: ignore
            self.no_metainfo = True
        else:
            super().__init__()  # type: ignore

        lengths = (len(self.s1_ds), len(self.s2_ds), len(self.s2_cloudy_ds))
        if len(set(lengths)) != 1:
            raise ValueError(f"Dataset lengths mismatch: s1={lengths[0]}, s2={lengths[1]}, s2_cloudy={lengths[2]}")

        self.split = split
        self.rescale_method = rescale_method
        self.to_neg_1_1 = to_neg_1_1
        self.sar_log1p = sar_log1p
        self.use_quantile = use_quantile
        self._quantile_percents = {
            "s1": (0.01, 0.99),
            "s2": (0.01, 0.99),
            "s2_cloudy": (0.01, 0.99),
        }
        self._quantile_percents.update(quantile_kwargs or {})
        if not use_quantile:
            self._quantile_percents = {"s1": None, "s2": None, "s2_cloudy": None}
        else:
            logger.info(f"Using quantile normalization with percents: {self._quantile_percents}")

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

    def _normalize_ms(self, img: torch.Tensor | np.ndarray, quantile_percents=None) -> torch.Tensor:
        img = _to_chw(_ensure_tensor(img)).float()
        img = _process_ms(
            img, self.rescale_method, quantile_percents=quantile_percents, intensity_min_max=self.intensity_min_max
        )
        if self.to_neg_1_1 and self.rescale_method == "default":
            img = img * 2.0 - 1.0
        return img

    def _normalize_sar(self, img: torch.Tensor | np.ndarray, quantile_percents=None) -> torch.Tensor:
        img = _to_chw(_ensure_tensor(img)).float()
        img = _process_sar(
            img,
            self.rescale_method,
            quantile_percents=quantile_percents,
            apply_log1p=self.sar_log1p,
        )
        if self.to_neg_1_1 and self.rescale_method == "default":
            img = img * 2.0 - 1.0
        return img

    def __getitem__(self, idx: int):
        sample: dict[str, torch.Tensor | str | MetaRow | list[torch.Tensor]] = {}
        if not self.no_s1:
            s1 = self._normalize_sar(self.s1_ds[idx]["img"], quantile_percents=self._quantile_percents["s1"])
            sample["s1"] = s1
        s2 = self._normalize_ms(self.s2_ds[idx]["img"], quantile_percents=self._quantile_percents["s2"])
        s2_cloudy = self._normalize_ms(
            self.s2_cloudy_ds[idx]["img"], quantile_percents=self._quantile_percents["s2_cloudy"]
        )

        sample.update(
            {
                "dataset_name": "sen12cr",
                "img": s2_cloudy,
                "gt": s2,
                "conditions": s2_cloudy if self.no_s1 else [s2_cloudy, s1],
            }
        )
        if self._meta_rows is not None and not self.no_metainfo:
            sample["meta"] = self._meta_rows[idx]
        return sample

    def transform(self, samples: tuple[dict, ...], rng):
        if self.no_s1:
            s2, s2_cloudy = samples
            s2 = self._normalize_ms(s2["img"], quantile_percents=self._quantile_percents["s2"])
            s2_cloudy = self._normalize_ms(s2_cloudy["img"], quantile_percents=self._quantile_percents["s2_cloudy"])
            sample = {
                "dataset_name": "sen12cr",
                "img": s2_cloudy,
                "gt": s2,
                "conditions": s2_cloudy,
            }
        else:
            s1, s2, s2_cloudy = samples
            s1 = self._normalize_sar(s1["img"], quantile_percents=self._quantile_percents["s1"])
            s2 = self._normalize_ms(s2["img"], quantile_percents=self._quantile_percents["s2"])
            s2_cloudy = self._normalize_ms(s2_cloudy["img"], quantile_percents=self._quantile_percents["s2_cloudy"])
            sample = {
                "dataset_name": "sen12cr",
                "img": s2_cloudy,
                "gt": s2,
                "conditions": [s2_cloudy, s1],
            }
        # No index in, no meta infomation returned
        return sample

    @classmethod
    def create_dataset(
        cls,
        input_dir: str,
        split: SplitName,
        **kwargs,
    ) -> "SEN12_CR_StreamingDataset":
        return cls(input_dir=input_dir, split=split, **kwargs)

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        from torch.utils.data._utils.collate import default_collate

        meta_list = []
        batch_without_meta = []

        for sample in batch:
            if "meta" in sample:
                meta_list.append(sample["meta"])
                sample_copy = {k: v for k, v in sample.items() if k != "meta"}
                batch_without_meta.append(sample_copy)
            else:
                batch_without_meta.append(sample)

        collated = default_collate(batch_without_meta)

        if meta_list:
            meta_batched = MetaRowBatched(
                index=torch.tensor([m.index for m in meta_list], dtype=torch.long),
                key=[m.key for m in meta_list],
                split=[m.split for m in meta_list],
                split_id=[m.split_id for m in meta_list],
                season=[m.season for m in meta_list],
                scene_id=[m.scene_id for m in meta_list],
                patch_id=[m.patch_id for m in meta_list],
                s1_path=[m.s1_path for m in meta_list],
                s2_path=[m.s2_path for m in meta_list],
                s2_cloudy_path=[m.s2_cloudy_path for m in meta_list],
            )
            collated["meta"] = meta_batched

        return collated

    @classmethod
    def create_dataloader(
        cls,
        input_dir: str,
        split: SplitName,
        batch_size: int = 8,
        num_workers: int = 0,
        shuffle: bool = False,
        **dataset_kwargs,
    ):
        ds = cls.create_dataset(input_dir=input_dir, split=split, **dataset_kwargs)
        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            collate_fn=cls.collate_fn,
        )
        return ds, dl


#  ---------------- Tests ------------------- #

import pytest


def _has_sen12_litdata(input_dir: str | Path, *, split: SplitName = "train") -> bool:
    input_dir = Path(input_dir)
    required = [
        input_dir / "litdata_s1" / split,
        input_dir / "litdata_s2_clean" / split,
        input_dir / "litdata_s2_cloudy" / split,
        input_dir / "litdata_meta" / f"{split}.csv",
    ]
    return all(path.exists() for path in required)


def test_sen12_cr_vis_preview(tmp_path: Path) -> None:
    data_root = Path("data/SEN12MS-CR")
    if not _has_sen12_litdata(data_root, split="train"):
        pytest.skip("SEN12MS-CR litdata not found. Run SEN12MS-CR conversion first.")

    out_path = Path("tmp/sen12_cr_preview_from_pytest.png")
    demo_vis_preview(input_dir=data_root, split="train", index=10, save_path=out_path)
    assert out_path.exists()


def demo_vis_preview(
    *,
    input_dir: str | Path = "data/SEN12MS-CR",
    split: SplitName = "train",
    index: int = 0,
    save_path: str | Path = "tmp/sen12_cr_preview.png",
) -> Path:
    """
    python -c "from src.stage2.cloud_removal.data.SEN12_CR import demo_vis_preview; demo_vis_preview(index=1)"
    """
    from src.stage2.cloud_removal.data.sen12_vis import plot_sen12_triplet

    input_dir = Path(input_dir)
    if not _has_sen12_litdata(input_dir, split=split):
        raise FileNotFoundError(f"SEN12MS-CR litdata not found under {input_dir} for split={split}")

    ds = SEN12_CR_StreamingDataset.create_dataset(
        input_dir=input_dir.as_posix(),
        split=split,
        return_meta=True,
        to_neg_1_1=False,
        rescale_method="default",
    )
    sample = ds[index]
    s1 = sample["s1"]
    s2 = sample["s2"]
    s2_cloudy = sample["s2_cloudy"]
    meta = sample.get("meta")

    if not isinstance(s1, torch.Tensor) or not isinstance(s2, torch.Tensor) or not isinstance(s2_cloudy, torch.Tensor):
        raise TypeError("Expected tensors in sample.")

    plot_sen12_triplet(
        s1=s1,
        s2=s2,
        s2_cloudy=s2_cloudy,
        title=str(meta) if meta is not None else None,
        save_path=save_path,
    )
    return Path(save_path)


def test_sen12_cr_loader():
    data_root = Path("data/SEN12MS-CR")

    ds, dl = SEN12_CR_StreamingDataset.create_dataloader(
        input_dir=data_root.as_posix(),
        split="train",
        batch_size=8,
        return_meta=True,
        to_neg_1_1=False,
        rescale_method="default",
        sar_log1p=True,
        num_workers=4,
    )

    sample = next(iter(dl))
    print(sample.keys())
    print(f"s1 shape: {sample['s1'].shape}")
    print(f"s2 shape: {sample['s2'].shape}")
    print(f"s2_cloudy shape: {sample['s2_cloudy'].shape}")

    if "meta" in sample:
        meta = sample["meta"]
        print(f"\nmeta type: {type(meta)}")
        print(f"meta.index shape: {meta.index.shape}")
        print(f"meta.index: {meta.index}")
        print(f"meta.key length: {len(meta.key)}")
        print(f"first key: {meta.key[0]}")
        print(f"first season: {meta.season[0]}")

    from tqdm import tqdm

    tbar = tqdm(dl)
    for sample in tbar:
        ...


if __name__ == "__main__":
    test_sen12_cr_loader()
