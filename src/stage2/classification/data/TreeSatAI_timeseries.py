from __future__ import annotations

import datetime as dt
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import h5py
import tifffile

from src.stage2.classification.data.builder import create_dataloader as build_dataloader


TREE_SAT_GENUS_CLASSES: tuple[str, ...] = (
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
)


def _decode_string_array(values: Any) -> list[str]:
    output: list[str] = []
    for item in values:
        if isinstance(item, bytes):
            output.append(item.decode("utf-8", errors="ignore"))
        else:
            output.append(str(item))
    return output


def _extract_day_of_year(product_name: str) -> int:
    match = re.search(r"(20\d{6})", product_name)
    if match is None:
        return -1
    try:
        parsed = dt.datetime.strptime(match.group(1), "%Y%m%d")
    except ValueError:
        return -1
    return parsed.timetuple().tm_yday - 1


def _scale_tensor(image: Tensor, mode: Literal["none", "minmax_0_1", "minmax_neg1_1"]) -> Tensor:
    if mode == "none":
        return image
    min_value = image.amin()
    max_value = image.amax()
    scaled = (image - min_value) / (max_value - min_value).clamp_min(1e-6)
    if mode == "minmax_0_1":
        return scaled
    return scaled * 2.0 - 1.0


def _scale_to_neg_1_1_strict(image: Tensor) -> Tensor:
    scaled = _scale_tensor(image, mode="minmax_neg1_1")
    return scaled.clamp(-1.0, 1.0)


def _normalize_hw(hw: tuple[int, int] | Sequence[int] | None) -> tuple[int, int] | None:
    if hw is None:
        return None
    if len(hw) != 2:
        raise ValueError(f"Expected hw length=2, got {hw}")
    height, width = int(hw[0]), int(hw[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid target hw={hw}")
    return (height, width)


def _normalize_max_t(max_t: int | None, name: str) -> int | None:
    if max_t is None:
        return None
    max_t = int(max_t)
    if max_t <= 0:
        raise ValueError(f"{name} must be > 0 when set, got {max_t}")
    return max_t


def _truncate_temporal_tensor(tensor: Tensor, max_t: int | None) -> Tensor:
    if max_t is None or int(tensor.shape[0]) <= max_t:
        return tensor
    return tensor[:max_t]


def _truncate_list(values: list[str], max_t: int | None) -> list[str]:
    if max_t is None or len(values) <= max_t:
        return values
    return values[:max_t]


def _replace_nan_with_zero(tensor: Tensor) -> Tensor:
    if torch.isnan(tensor).any():
        return torch.nan_to_num(tensor, nan=0.0)
    return tensor


def _resize_timeseries(
    image: Tensor,
    target_hw: tuple[int, int] | None,
    mode: Literal["nearest", "bilinear"] = "bilinear",
) -> Tensor:
    if target_hw is None:
        return image
    if image.ndim != 4:
        raise ValueError(f"Expected time-series tensor shape (T,C,H,W), got {tuple(image.shape)}")
    if tuple(image.shape[-2:]) == target_hw:
        return image
    kwargs: dict[str, Any] = {"size": target_hw, "mode": mode}
    if mode != "nearest":
        kwargs["align_corners"] = False
    return F.interpolate(image, **kwargs)


def _resize_image_chw(
    image: Tensor,
    target_hw: tuple[int, int] | None,
    mode: Literal["nearest", "bilinear"] = "bilinear",
) -> Tensor:
    if target_hw is None:
        return image
    if image.ndim != 3:
        raise ValueError(f"Expected image tensor shape (C,H,W), got {tuple(image.shape)}")
    if tuple(image.shape[-2:]) == target_hw:
        return image
    image_4d = image.unsqueeze(0)
    kwargs: dict[str, Any] = {"size": target_hw, "mode": mode}
    if mode != "nearest":
        kwargs["align_corners"] = False
    resized = F.interpolate(image_4d, **kwargs)
    return resized.squeeze(0)


def _pad_sequence_tensors(
    items: list[Tensor],
    pad_value: float = 0.0,
) -> tuple[Tensor, Tensor]:
    if not items:
        raise ValueError("items must not be empty.")
    max_len = max(int(item.shape[0]) for item in items)
    output_shape = (len(items), max_len, *items[0].shape[1:])
    output = torch.full(output_shape, pad_value, dtype=items[0].dtype)
    valid = torch.zeros((len(items), max_len), dtype=torch.bool)
    for index, item in enumerate(items):
        length = int(item.shape[0])
        output[index, :length] = item
        valid[index, :length] = True
    return output, valid


def treesatai_timeseries_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    collated: dict[str, Any] = {}
    if not batch:
        return collated

    tensor_keys = [key for key, value in batch[0].items() if isinstance(value, Tensor)]
    for key in tensor_keys:
        tensors = [sample[key] for sample in batch]
        if any(tensor.ndim >= 1 and tensor.shape[0] != tensors[0].shape[0] for tensor in tensors):
            padded, valid = _pad_sequence_tensors(tensors, pad_value=0.0)
            collated[key] = padded
            collated[f"{key}_valid"] = valid
        else:
            collated[key] = torch.stack(tensors, dim=0)

    list_keys = [key for key, value in batch[0].items() if isinstance(value, list)]
    for key in list_keys:
        collated[key] = [sample[key] for sample in batch]

    scalar_keys = [
        key for key in batch[0].keys() if key not in tensor_keys and key not in list_keys and key != "sample_id"
    ]
    for key in scalar_keys:
        collated[key] = [sample[key] for sample in batch]

    collated["sample_id"] = [sample["sample_id"] for sample in batch]
    return collated


class TreeSatAITimeSeriesDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        root: str = "data/Downstreams/TreeSatAI_TimeSeries",
        split: Literal["train", "val", "test"] = "train",
        sensors: Sequence[Literal["aerial", "s1_asc", "s1_des", "s2"]] = ("aerial", "s1_asc", "s1_des", "s2"),
        label_mode: Literal["multi_hot", "proportion", "both"] = "both",
        scale_mode: Literal["none", "minmax_0_1", "minmax_neg1_1"] = "none",
        aerial_hw: tuple[int, int] | Sequence[int] | None = None,
        s1_hw: tuple[int, int] | Sequence[int] | None = None,
        s2_hw: tuple[int, int] | Sequence[int] | None = None,
        max_t_s1: int | None = None,
        max_t_s2: int | None = None,
        image_interp_mode: Literal["nearest", "bilinear"] = "bilinear",
        s2_mask_interp_mode: Literal["nearest", "bilinear"] = "nearest",
        s2_to_neg_1_1: bool = False,
        strict_files: bool = True,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.sensors = tuple(sensors)
        self.label_mode = label_mode
        self.scale_mode = scale_mode
        self.aerial_hw = _normalize_hw(aerial_hw)
        self.s1_hw = _normalize_hw(s1_hw)
        self.s2_hw = _normalize_hw(s2_hw)
        self.max_t_s1 = _normalize_max_t(max_t_s1, "max_t_s1")
        self.max_t_s2 = _normalize_max_t(max_t_s2, "max_t_s2")
        self.image_interp_mode = image_interp_mode
        self.s2_mask_interp_mode = s2_mask_interp_mode
        self.s2_to_neg_1_1 = s2_to_neg_1_1
        self.strict_files = strict_files

        if not self.sensors:
            raise ValueError("sensors must not be empty.")

        valid_sensors = {"aerial", "s1_asc", "s1_des", "s2"}
        invalid_sensors = set(self.sensors) - valid_sensors
        if invalid_sensors:
            raise ValueError(f"Unsupported sensors: {sorted(invalid_sensors)}")

        self.split_files = self._read_split_list()
        self.h5_lookup = self._build_h5_lookup()
        self.labels = self._read_labels()
        self.samples = self._build_samples()
        self.classes = TREE_SAT_GENUS_CLASSES

    def _read_split_list(self) -> list[str]:
        split_path = self.root / "split" / f"{self.split}_filenames.lst"
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")
        lines = split_path.read_text().splitlines()
        return [line.strip() for line in lines if line.strip()]

    def _build_h5_lookup(self) -> dict[str, Path]:
        ts_root = self.root / "sentinel-ts"
        if not ts_root.exists():
            raise FileNotFoundError(f"sentinel-ts directory not found: {ts_root}")

        lookup: dict[str, Path] = {}
        for file_path in sorted(ts_root.glob("*.h5")):
            base_name = file_path.stem.rsplit("_", 1)[0]
            if base_name in lookup:
                raise ValueError(f"Duplicate base sample detected: {base_name}")
            lookup[base_name] = file_path
        return lookup

    def _read_labels(self) -> dict[str, list[list[Any]]]:
        label_path = self.root / "TreeSatBA_v9_60m_multi_labels.json"
        if not label_path.exists():
            raise FileNotFoundError(f"Label file not found: {label_path}")
        with label_path.open() as file:
            return json.load(file)

    def _build_samples(self) -> list[tuple[str, Path]]:
        samples: list[tuple[str, Path]] = []
        missing: list[str] = []
        for tif_name in self.split_files:
            base_name = Path(tif_name).stem
            h5_path = self.h5_lookup.get(base_name)
            if h5_path is None:
                missing.append(tif_name)
                continue
            samples.append((tif_name, h5_path))

        if missing and self.strict_files:
            raise FileNotFoundError(f"Missing .h5 files for {len(missing)} samples, first={missing[0]}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _build_label_tensors(self, tif_name: str) -> dict[str, Tensor]:
        entries = self.labels.get(tif_name, [])
        proportion = torch.zeros(len(self.classes), dtype=torch.float32)
        for genus, value in entries:
            if genus in self.classes:
                proportion[self.classes.index(genus)] = float(value)

        output: dict[str, Tensor] = {}
        if self.label_mode in {"multi_hot", "both"}:
            output["label"] = (proportion > 0).to(torch.float32)
        if self.label_mode in {"proportion", "both"}:
            output["label_proportion"] = proportion
        return output

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Read one TreeSatAI-TS sample.

        Returns a dictionary with metadata, labels, and time-series tensors.
        Keys depend on dataset options:

        Always returned:
            - ``sample_id``: ``str``. H5 stem, e.g. ``Abies_alba_1_1005_WEFL_NLF_2020``.
            - ``split_file``: ``str``. Split list filename, e.g. ``Abies_alba_1_1005_WEFL_NLF.tif``.
            - ``h5_path``: ``str``. Absolute/relative path to the source H5 file.

        Labels (controlled by ``label_mode``):
            - ``label``: ``Tensor[15]`` float32 multi-hot genus labels when ``label_mode`` is
              ``"multi_hot"`` or ``"both"``.
            - ``label_proportion``: ``Tensor[15]`` float32 genus proportions when
              ``label_mode`` is ``"proportion"`` or ``"both"``.

        SAR ascending branch (when ``"s1_asc" in sensors``):
            - ``image_s1_asc``: ``Tensor[T_asc, 2, H1, W1]`` float32.
              ``H1,W1`` are ``s1_hw`` if provided, otherwise original ``6,6``.
              If ``max_t_s1`` is set, ``T_asc <= max_t_s1``.
              Raw NaN values in S1 are replaced by ``0.0`` before interpolation.
            - ``products_s1_asc``: ``list[str]`` of length ``T_asc``.
            - ``doy_s1_asc``: ``Tensor[T_asc]`` int64 day-of-year index in ``[0, 364]`` or ``-1`` if parsing fails.

        SAR descending branch (when ``"s1_des" in sensors``):
            - ``image_s1_des``: ``Tensor[T_des, 2, H1, W1]`` float32.
              If ``max_t_s1`` is set, ``T_des <= max_t_s1``.
              Raw NaN values in S1 are replaced by ``0.0`` before interpolation.
            - ``products_s1_des``: ``list[str]`` of length ``T_des``.
            - ``doy_s1_des``: ``Tensor[T_des]`` int64.

        Sentinel-2 branch (when ``"s2" in sensors``):
            - ``image_s2``: ``Tensor[T_s2, 10, H2, W2]`` float32.
              ``H2,W2`` are ``s2_hw`` if provided, otherwise original ``6,6``.
              If ``max_t_s2`` is set, ``T_s2 <= max_t_s2``.
              If ``s2_to_neg_1_1=True``, values are strictly clipped to ``[-1, 1]``.
              Otherwise scaling follows ``scale_mode``.
            - ``mask_s2``: ``Tensor[T_s2, 2, H2, W2]`` float32 (snow/cloud probability masks).
            - ``products_s2``: ``list[str]`` of length ``T_s2``.
            - ``doy_s2``: ``Tensor[T_s2]`` int64.

        Aerial branch (when ``"aerial" in sensors``):
            - ``image_aerial``: ``Tensor[4, Ha, Wa]`` float32.
              ``Ha,Wa`` are ``aerial_hw`` if provided, otherwise original ``304,304``.
        """
        tif_name, h5_path = self.samples[index]
        output: dict[str, Any] = {
            "sample_id": h5_path.stem,
            "split_file": tif_name,
            "h5_path": str(h5_path),
        }
        output.update(self._build_label_tensors(tif_name))

        if "aerial" in self.sensors:
            aerial = tifffile.imread(self.root / "aerial" / tif_name)
            aerial_chw = torch.from_numpy(aerial).permute(2, 0, 1).to(torch.float32)
            aerial_chw = _resize_image_chw(aerial_chw, target_hw=self.aerial_hw, mode=self.image_interp_mode)
            output["image_aerial"] = _scale_tensor(aerial_chw, mode=self.scale_mode)

        with h5py.File(h5_path, "r") as file:
            if "s1_asc" in self.sensors:
                s1_asc = torch.from_numpy(file["sen-1-asc-data"][:]).to(torch.float32)
                s1_asc = _replace_nan_with_zero(s1_asc)
                s1_asc = _truncate_temporal_tensor(s1_asc, self.max_t_s1)
                s1_asc = _resize_timeseries(s1_asc, target_hw=self.s1_hw, mode=self.image_interp_mode)
                products_asc = _truncate_list(_decode_string_array(file["sen-1-asc-products"][:]), self.max_t_s1)
                output["image_s1_asc"] = _scale_tensor(s1_asc, mode=self.scale_mode)
                output["products_s1_asc"] = products_asc
                output["doy_s1_asc"] = torch.tensor([_extract_day_of_year(x) for x in products_asc], dtype=torch.long)

            if "s1_des" in self.sensors:
                s1_des = torch.from_numpy(file["sen-1-des-data"][:]).to(torch.float32)
                s1_des = _replace_nan_with_zero(s1_des)
                s1_des = _truncate_temporal_tensor(s1_des, self.max_t_s1)
                s1_des = _resize_timeseries(s1_des, target_hw=self.s1_hw, mode=self.image_interp_mode)
                products_des = _truncate_list(_decode_string_array(file["sen-1-des-products"][:]), self.max_t_s1)
                output["image_s1_des"] = _scale_tensor(s1_des, mode=self.scale_mode)
                output["products_s1_des"] = products_des
                output["doy_s1_des"] = torch.tensor([_extract_day_of_year(x) for x in products_des], dtype=torch.long)

            if "s2" in self.sensors:
                s2 = torch.from_numpy(file["sen-2-data"][:]).to(torch.float32)
                s2_masks = torch.from_numpy(file["sen-2-masks"][:]).to(torch.float32)
                s2 = _truncate_temporal_tensor(s2, self.max_t_s2)
                s2_masks = _truncate_temporal_tensor(s2_masks, self.max_t_s2)
                s2 = _resize_timeseries(s2, target_hw=self.s2_hw, mode=self.image_interp_mode)
                s2_masks = _resize_timeseries(s2_masks, target_hw=self.s2_hw, mode=self.s2_mask_interp_mode)
                products_s2 = _truncate_list(_decode_string_array(file["sen-2-products"][:]), self.max_t_s2)
                if self.s2_to_neg_1_1:
                    output["image_s2"] = _scale_to_neg_1_1_strict(s2)
                else:
                    output["image_s2"] = _scale_tensor(s2, mode=self.scale_mode)
                output["mask_s2"] = s2_masks
                output["products_s2"] = products_s2
                output["doy_s2"] = torch.tensor([_extract_day_of_year(x) for x in products_s2], dtype=torch.long)

        return output

    @classmethod
    def create_dataloader(
        cls,
        dataset_kwargs: dict[str, Any] | None = None,
        loader_kwargs: dict[str, Any] | None = None,
        pad_time_series: bool = True,
    ) -> tuple["TreeSatAITimeSeriesDataset", DataLoader[Any]]:
        dataset_kwargs = dataset_kwargs or {}
        loader_kwargs = loader_kwargs or {}
        dataset = cls(**dataset_kwargs)
        collate_fn = treesatai_timeseries_collate_fn if pad_time_series else None
        _, dataloader = build_dataloader(
            dataset=dataset,
            loader_kwargs=loader_kwargs,
            collate_fn=collate_fn,
        )
        return dataset, dataloader
