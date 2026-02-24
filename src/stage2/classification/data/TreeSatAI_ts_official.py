from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import json
import numpy as np
import rasterio
from skmultilearn.model_selection import iterative_train_test_split
import torch
from torch.utils.data import Dataset

from src.stage2.classification.data.builder import create_dataloader as build_dataloader


DEFAULT_CLASSES: list[str] = [
    "Abies",
    "Acer",
    "Alnus",
    "Betula",
    "Cleared",
    "Fagus",
    "Fraxinus",
    "Larix",
    "Picea",
    "Pinus",
    "Populus",
    "Prunus",
    "Pseudotsuga",
    "Quercus",
    "Tilia",
]


def subset_dict_by_filename(files_to_subset: list[str], dictionary: dict[str, Any]) -> dict[str, Any]:
    return {file: dictionary[file] for file in files_to_subset if file in dictionary}


def filter_labels_by_threshold(
    labels_dict: dict[str, list[list[Any]]], area_threshold: float = 0.07
) -> dict[str, list[str]]:
    filtered: dict[str, list[str]] = {}
    for img_name, labels in labels_dict.items():
        for lbl, area in labels:
            if float(area) > area_threshold:
                filtered.setdefault(img_name, []).append(str(lbl))
    return filtered


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    keys = list(batch[0].keys())
    output: dict[str, Any] = {}
    for key in ["s2", "s1-asc", "s1-des", "s1"]:
        if key in keys:
            tensors = [sample[key] for sample in batch]
            max_t = max(tensor.size(0) for tensor in tensors)
            stacked_tensor = torch.stack(
                [torch.nn.functional.pad(tensor, (0, 0, 0, 0, 0, 0, 0, max_t - tensor.size(0))) for tensor in tensors],
                dim=0,
            )
            output[key] = stacked_tensor
            keys.remove(key)

            date_key = f"{key}_dates"
            date_tensors = [sample[date_key] for sample in batch]
            max_t_dates = max(tensor.size(0) for tensor in date_tensors)
            output[date_key] = torch.stack(
                [torch.nn.functional.pad(tensor, (0, max_t_dates - tensor.size(0))) for tensor in date_tensors],
                dim=0,
            )
            keys.remove(date_key)

    if "name" in keys:
        output["name"] = [sample["name"] for sample in batch]
        keys.remove("name")

    for key in keys:
        output[key] = torch.stack([sample[key] for sample in batch])
    return output


def day_number_in_year(date_arr: Any, place: int = 4) -> torch.Tensor:
    day_number: list[int] = []
    for date_string in date_arr:
        if isinstance(date_string, bytes):
            value = date_string.decode("utf-8")
        else:
            value = str(date_string)
        date_object = datetime.strptime(value.split("_")[place][:8], "%Y%m%d")
        day_number.append(date_object.timetuple().tm_yday)
    return torch.tensor(day_number, dtype=torch.long)


def replace_nans_with_mean(batch_of_images: torch.Tensor) -> torch.Tensor:
    output = batch_of_images.clone()
    image_means = torch.nanmean(output, dim=(2, 3), keepdim=True)
    image_means[torch.isnan(image_means)] = 0.0
    nan_mask = torch.isnan(output)
    output[nan_mask] = image_means.expand_as(output)[nan_mask]
    return output


def _sample_temporal_tensor_and_dates(
    images: torch.Tensor,
    dates: torch.Tensor,
    sample_cap: int = 50,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid_len = min(int(images.shape[0]), int(dates.shape[0]))
    images = images[:valid_len]
    dates = dates[:valid_len]
    if valid_len > sample_cap:
        random_indices = torch.randperm(valid_len)[:sample_cap]
        images = images[random_indices]
        dates = dates[random_indices]
    return images, dates


class TreeSAT(Dataset):
    def __init__(
        self,
        path: str | Path,
        modalities: list[str],
        transform=lambda x: x,
        split: str = "train",
        classes: list[str] | None = None,
        partition: float = 1.0,
        mono_strict: bool = False,
    ):
        self.path = Path(path)
        self.transform = transform
        self.partition = float(partition)
        self.modalities = modalities
        self.mono_strict = mono_strict
        self.classes = classes if classes is not None else DEFAULT_CLASSES

        split_path = self._resolve_split_path(split)
        with split_path.open("r") as file:
            self.data_list = [line.strip() for line in file.readlines() if line.strip()]

        self.load_labels(self.classes)
        self.collate_fn = collate_fn

    def _resolve_split_path(self, split: str) -> Path:
        split_file = f"{split}_filenames.lst"
        candidates = [self.path / split_file, self.path / "split" / split_file]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Split file not found. Tried: {[str(x) for x in candidates]}")

    def _resolve_label_path(self) -> Path:
        candidates = [
            self.path / "labels" / "TreeSatBA_v9_60m_multi_labels.json",
            self.path / "TreeSatBA_v9_60m_multi_labels.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Label file not found. Tried: {[str(x) for x in candidates]}")

    def _resolve_h5_path(self, tif_name: str) -> Path:
        base = Path(tif_name).stem
        for folder in ("sentinel-ts", "sentinel"):
            root = self.path / folder
            if not root.exists():
                continue
            direct = root / f"{base}.h5"
            if direct.exists():
                return direct
            matched = sorted(root.glob(f"{base}_*.h5"))
            if len(matched) == 1:
                return matched[0]
        raise FileNotFoundError(f"No matching h5 found for sample: {tif_name}")

    def load_labels(self, classes: list[str]) -> None:
        with self._resolve_label_path().open() as file:
            jfile = json.load(file)
            subsetted_dict = subset_dict_by_filename(self.data_list, jfile)
            labels = filter_labels_by_threshold(subsetted_dict, 0.07)
            lines = list(labels.keys())

        y = [[0 for _ in range(len(classes))] for _ in lines]
        for i, line in enumerate(lines):
            for genus in labels[line]:
                if genus in classes:
                    y[i][classes.index(genus)] = 1

        if self.partition >= 1.0:
            self.data_list = lines
            self.labels = np.array(y)
            return

        self.data_list, self.labels, _, _ = iterative_train_test_split(
            np.expand_dims(np.array(lines), axis=1),
            np.array(y),
            test_size=1.0 - self.partition,
        )
        self.data_list = list(np.concatenate(self.data_list).flat)

    @staticmethod
    def _load_s1_branch(file: h5py.File, key_data: str, key_products: str) -> tuple[torch.Tensor, torch.Tensor]:
        images = torch.tensor(file[key_data][:], dtype=torch.float32)[:, :2, :, :]
        images = replace_nans_with_mean(images)
        dates = day_number_in_year(file[key_products][:])
        return _sample_temporal_tensor_and_dates(images, dates, sample_cap=50)

    def __getitem__(self, i: int) -> dict[str, Any]:
        name = self.data_list[i]
        output: dict[str, Any] = {"label": torch.tensor(self.labels[i]), "name": name}

        if "aerial" in self.modalities:
            with rasterio.open(self.path / "aerial" / name) as f:
                output["aerial"] = torch.tensor(f.read(), dtype=torch.float32)

        h5_path = self._resolve_h5_path(name)
        with h5py.File(h5_path, "r") as file:
            if "s2" in self.modalities:
                s2 = torch.tensor(file["sen-2-data"][:], dtype=torch.float32)
                s2_dates = day_number_in_year(file["sen-2-products"][:], place=2)
                s2, s2_dates = _sample_temporal_tensor_and_dates(s2, s2_dates, sample_cap=50)
                output["s2"] = s2
                output["s2_dates"] = s2_dates

            if "s1-asc" in self.modalities:
                s1_asc, s1_asc_dates = self._load_s1_branch(file, "sen-1-asc-data", "sen-1-asc-products")
                output["s1-asc"] = s1_asc
                output["s1-asc_dates"] = s1_asc_dates

            if "s1-des" in self.modalities:
                s1_des, s1_des_dates = self._load_s1_branch(file, "sen-1-des-data", "sen-1-des-products")
                output["s1-des"] = s1_des
                output["s1-des_dates"] = s1_des_dates

            if "s1" in self.modalities:
                s1_asc, s1_asc_dates = self._load_s1_branch(file, "sen-1-asc-data", "sen-1-asc-products")
                s1_des, s1_des_dates = self._load_s1_branch(file, "sen-1-des-data", "sen-1-des-products")
                s1 = torch.cat([s1_asc, s1_des], dim=0)
                s1_dates = torch.cat([s1_asc_dates, s1_des_dates], dim=0)
                s1, s1_dates = _sample_temporal_tensor_and_dates(s1, s1_dates, sample_cap=50)
                output["s1"] = s1
                output["s1_dates"] = s1_dates

            if "s2-4season-median" in self.modalities:
                output_inter = torch.tensor(file["sen-2-data"][:], dtype=torch.float32)
                dates = day_number_in_year(file["sen-2-products"][:], place=2)
                medians: list[torch.Tensor] = []
                for j in range(4):
                    mask = (dates >= 92 * j) & (dates < 92 * (j + 1))
                    if bool(mask.any()):
                        median_j, _ = torch.median(output_inter[mask], dim=0)
                        medians.append(median_j)
                    else:
                        medians.append(
                            torch.zeros((output_inter.shape[1], output_inter.shape[-2], output_inter.shape[-1]))
                        )
                output["s2-4season-median"] = torch.cat(medians)

            if "s2-median" in self.modalities:
                output["s2-median"], _ = torch.median(torch.tensor(file["sen-2-data"][:], dtype=torch.float32), dim=0)

            if "s1-4season-median" in self.modalities:
                output_inter = replace_nans_with_mean(
                    torch.tensor(file["sen-1-asc-data"][:], dtype=torch.float32)[:, :2, :, :]
                )
                dates = day_number_in_year(file["sen-1-asc-products"][:])
                medians = []
                for j in range(4):
                    mask = (dates >= 92 * j) & (dates < 92 * (j + 1))
                    if bool(mask.any()):
                        median_j, _ = torch.median(output_inter[mask], dim=0)
                        medians.append(median_j)
                    else:
                        medians.append(
                            torch.zeros((output_inter.shape[1], output_inter.shape[-2], output_inter.shape[-1]))
                        )
                output["s1-4season-median"] = torch.cat(medians)

            if "s1-median" in self.modalities:
                s1_asc = replace_nans_with_mean(
                    torch.tensor(file["sen-1-asc-data"][:], dtype=torch.float32)[:, :2, :, :]
                )
                output["s1-median"], _ = torch.median(s1_asc, dim=0)

        if "s1-mono" in self.modalities:
            with rasterio.open(self.path / "s1/60m" / name) as f:
                s1_mono = torch.tensor(f.read().astype(np.float32), dtype=torch.float32)
            if self.mono_strict:
                s1_mono = s1_mono[:2, :, :]
            output["s1-mono"] = s1_mono

        if "s2-mono" in self.modalities:
            with rasterio.open(self.path / "s2/60m" / name) as f:
                s2_mono = torch.tensor(f.read().astype(np.float32), dtype=torch.float32)
            if self.mono_strict:
                s2_mono = s2_mono[:10, :, :]
            output["s2-mono"] = s2_mono

        return self.transform(output)

    def __len__(self) -> int:
        return len(self.data_list)

    @classmethod
    def create_dataloader(
        cls,
        dataset_kwargs: dict[str, Any] | None = None,
        loader_kwargs: dict[str, Any] | None = None,
    ):
        dataset_kwargs = dataset_kwargs or {}
        loader_kwargs = loader_kwargs or {}
        dataset = cls(**dataset_kwargs)
        _, dataloader = build_dataloader(
            dataset=dataset,
            loader_kwargs=loader_kwargs,
            collate_fn=collate_fn,
        )
        return dataset, dataloader
