from __future__ import annotations

import datetime as dt
import json
import re
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import h5py
import tifffile
from omegaconf import OmegaConf

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


def _normalize_positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def _normalize_label_area_threshold(value: float) -> float:
    value = float(value)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"label_area_threshold must be in [0, 1], got {value}")
    return value


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


def _replace_nans_with_spatial_mean(tensor: Tensor) -> Tensor:
    if tensor.ndim != 4:
        raise ValueError(f"Expected S1 tensor shape (T,C,H,W), got {tuple(tensor.shape)}")
    image_means = torch.nanmean(tensor, dim=(2, 3), keepdim=True)
    image_means = torch.nan_to_num(image_means, nan=0.0)
    nan_mask = torch.isnan(tensor)
    if nan_mask.any():
        tensor = tensor.clone()
        tensor[nan_mask] = image_means.expand_as(tensor)[nan_mask]
    return tensor


def _select_temporal_indices(
    length: int,
    sample_cap: int,
    max_t: int | None,
    mode: Literal["official_random_cap_then_max_t", "truncate_only"],
) -> Tensor:
    if length <= 0:
        return torch.zeros((0,), dtype=torch.long)
    if mode == "official_random_cap_then_max_t":
        if length > sample_cap:
            indices = torch.randperm(length)[:sample_cap]
        else:
            indices = torch.arange(length, dtype=torch.long)
    elif mode == "truncate_only":
        indices = torch.arange(length, dtype=torch.long)
    else:
        raise ValueError(f"Unsupported temporal_sampling_mode={mode}")
    if max_t is not None:
        indices = indices[:max_t]
    return indices


def _select_tensor_by_indices(tensor: Tensor, indices: Tensor) -> Tensor:
    if indices.numel() == 0:
        return tensor[:0]
    return tensor.index_select(0, indices.to(device=tensor.device))


def _select_list_by_indices(values: list[str], indices: Tensor) -> list[str]:
    output: list[str] = []
    for index in indices.tolist():
        output.append(values[int(index)])
    return output


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


def treesatai_timeseries_collate_fn(
    batch: list[dict[str, Any]],
    emit_valid_mask: bool = True,
) -> dict[str, Any]:
    collated: dict[str, Any] = {}
    if not batch:
        return collated

    tensor_keys = [key for key, value in batch[0].items() if isinstance(value, Tensor)]
    for key in tensor_keys:
        tensors = [sample[key] for sample in batch]
        if any(tensor.ndim >= 1 and tensor.shape[0] != tensors[0].shape[0] for tensor in tensors):
            padded, valid = _pad_sequence_tensors(tensors, pad_value=0.0)
            collated[key] = padded
            if emit_valid_mask:
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
        label_area_threshold: float = 0.07,
        temporal_sample_cap: int = 50,
        temporal_sampling_mode: Literal["official_random_cap_then_max_t", "truncate_only"] = (
            "official_random_cap_then_max_t"
        ),
        emit_valid_mask: bool = False,
        s1_nan_fill_mode: Literal["spatial_mean", "zero"] = "spatial_mean",
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
        self.label_area_threshold = _normalize_label_area_threshold(label_area_threshold)
        self.temporal_sample_cap = _normalize_positive_int(temporal_sample_cap, "temporal_sample_cap")
        self.temporal_sampling_mode = temporal_sampling_mode
        self.emit_valid_mask = bool(emit_valid_mask)
        self.s1_nan_fill_mode = s1_nan_fill_mode

        if not self.sensors:
            raise ValueError("sensors must not be empty.")

        valid_sensors = {"aerial", "s1_asc", "s1_des", "s2"}
        invalid_sensors = set(self.sensors) - valid_sensors
        if invalid_sensors:
            raise ValueError(f"Unsupported sensors: {sorted(invalid_sensors)}")
        if self.temporal_sampling_mode not in {"official_random_cap_then_max_t", "truncate_only"}:
            raise ValueError(f"Unsupported temporal_sampling_mode={self.temporal_sampling_mode}")
        if self.s1_nan_fill_mode not in {"spatial_mean", "zero"}:
            raise ValueError(f"Unsupported s1_nan_fill_mode={self.s1_nan_fill_mode}")

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
            output["label"] = (proportion > self.label_area_threshold).to(torch.float32)
        if self.label_mode in {"proportion", "both"}:
            output["label_proportion"] = proportion
        return output

    def _maybe_fill_s1_nan(self, tensor: Tensor) -> Tensor:
        if self.s1_nan_fill_mode == "spatial_mean":
            return _replace_nans_with_spatial_mean(tensor)
        if self.s1_nan_fill_mode == "zero":
            return _replace_nan_with_zero(tensor)
        raise ValueError(f"Unsupported s1_nan_fill_mode={self.s1_nan_fill_mode}")

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
                s1_asc = self._maybe_fill_s1_nan(s1_asc)
                products_asc = _decode_string_array(file["sen-1-asc-products"][:])
                s1_asc_indices = _select_temporal_indices(
                    length=int(s1_asc.shape[0]),
                    sample_cap=self.temporal_sample_cap,
                    max_t=self.max_t_s1,
                    mode=self.temporal_sampling_mode,
                )
                s1_asc = _select_tensor_by_indices(s1_asc, s1_asc_indices)
                products_asc = _select_list_by_indices(products_asc, s1_asc_indices)
                s1_asc = _resize_timeseries(s1_asc, target_hw=self.s1_hw, mode=self.image_interp_mode)
                output["image_s1_asc"] = _scale_tensor(s1_asc, mode=self.scale_mode)
                output["products_s1_asc"] = products_asc
                output["doy_s1_asc"] = torch.tensor([_extract_day_of_year(x) for x in products_asc], dtype=torch.long)

            if "s1_des" in self.sensors:
                s1_des = torch.from_numpy(file["sen-1-des-data"][:]).to(torch.float32)
                s1_des = self._maybe_fill_s1_nan(s1_des)
                products_des = _decode_string_array(file["sen-1-des-products"][:])
                s1_des_indices = _select_temporal_indices(
                    length=int(s1_des.shape[0]),
                    sample_cap=self.temporal_sample_cap,
                    max_t=self.max_t_s1,
                    mode=self.temporal_sampling_mode,
                )
                s1_des = _select_tensor_by_indices(s1_des, s1_des_indices)
                products_des = _select_list_by_indices(products_des, s1_des_indices)
                s1_des = _resize_timeseries(s1_des, target_hw=self.s1_hw, mode=self.image_interp_mode)
                output["image_s1_des"] = _scale_tensor(s1_des, mode=self.scale_mode)
                output["products_s1_des"] = products_des
                output["doy_s1_des"] = torch.tensor([_extract_day_of_year(x) for x in products_des], dtype=torch.long)

            if "s2" in self.sensors:
                s2 = torch.from_numpy(file["sen-2-data"][:]).to(torch.float32)
                s2_masks = torch.from_numpy(file["sen-2-masks"][:]).to(torch.float32)
                products_s2 = _decode_string_array(file["sen-2-products"][:])
                s2_indices = _select_temporal_indices(
                    length=int(s2.shape[0]),
                    sample_cap=self.temporal_sample_cap,
                    max_t=self.max_t_s2,
                    mode=self.temporal_sampling_mode,
                )
                s2 = _select_tensor_by_indices(s2, s2_indices)
                s2_masks = _select_tensor_by_indices(s2_masks, s2_indices)
                products_s2 = _select_list_by_indices(products_s2, s2_indices)
                s2 = _resize_timeseries(s2, target_hw=self.s2_hw, mode=self.image_interp_mode)
                s2_masks = _resize_timeseries(s2_masks, target_hw=self.s2_hw, mode=self.s2_mask_interp_mode)
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
        collate_fn = (
            partial(treesatai_timeseries_collate_fn, emit_valid_mask=dataset.emit_valid_mask)
            if pad_time_series
            else None
        )
        _, dataloader = build_dataloader(
            dataset=dataset,
            loader_kwargs=loader_kwargs,
            collate_fn=collate_fn,
        )
        return dataset, dataloader


def _print_batch_debug_info(batch: dict[str, Any]) -> None:
    print(f"batch keys: {list(batch.keys())}")
    for key, value in batch.items():
        if isinstance(value, Tensor):
            print(f"{key}: shape={tuple(value.shape)}, dtype={value.dtype}")
        elif isinstance(value, list):
            print(f"{key}: list(len)={len(value)}")
        else:
            print(f"{key}: type={type(value)}")
    label = batch.get("label", None)
    if isinstance(label, Tensor):
        print(f"label max: {float(label.max().item())}")
    else:
        print("label not found in batch.")


def _to_display_rgb(image_chw: Tensor, rgb_indices: tuple[int, int, int] | None = None) -> Tensor:
    if image_chw.ndim != 3:
        raise ValueError(f"Expected CHW image, got {tuple(image_chw.shape)}")
    channels = int(image_chw.shape[0])
    if channels == 1:
        rgb = image_chw.repeat(3, 1, 1)
    elif channels >= 3:
        if rgb_indices is not None and max(rgb_indices) < channels:
            rgb = image_chw[list(rgb_indices)]
        else:
            rgb = image_chw[:3]
    else:
        pad = image_chw.new_zeros((3 - channels, *image_chw.shape[1:]))
        rgb = torch.cat([image_chw, pad], dim=0)

    rgb = rgb.to(torch.float32)
    min_v = rgb.amin()
    max_v = rgb.amax()
    rgb = (rgb - min_v) / (max_v - min_v).clamp_min(1e-6)
    return rgb.permute(1, 2, 0).cpu()


def _get_valid_frame_indices(valid_mask: Tensor | None, sample_index: int, t: int) -> list[int]:
    if valid_mask is None:
        return list(range(t))
    if valid_mask.ndim != 2:
        raise ValueError(f"Expected valid mask shape [B,T], got {tuple(valid_mask.shape)}")
    sample_valid = valid_mask[sample_index]
    indices = torch.where(sample_valid)[0].tolist()
    if not indices:
        return [0]
    return [int(idx) for idx in indices]


def _label_indices_as_int_list(labels: Tensor, sample_index: int) -> list[int]:
    sample_label = labels[sample_index]
    if sample_label.ndim != 1:
        raise ValueError(f"Expected label shape [C], got {tuple(sample_label.shape)}")
    positive = torch.where(sample_label > 0.5)[0].tolist()
    return [int(idx) for idx in positive]


def _label_text(labels: Tensor, sample_index: int) -> str:
    label_ints = _label_indices_as_int_list(labels, sample_index)
    if not label_ints:
        return "[]"
    return "[" + ",".join(str(idx) for idx in label_ints) + "]"


def plot_batch_modalities(batch: dict[str, Any], save_path: str | None = None) -> None:
    labels = batch.get("label", None)
    if not isinstance(labels, Tensor):
        raise ValueError("batch must contain tensor key 'label'.")
    bsz = int(labels.shape[0])
    if bsz <= 0:
        raise ValueError("Empty batch cannot be plotted.")

    modality_keys = [key for key in ("image_aerial", "image_s1_asc", "image_s1_des", "image_s2") if key in batch]
    if not modality_keys:
        raise ValueError("No supported image modality key found in batch.")

    frame_indices_map: dict[tuple[int, str], list[int]] = {}
    max_cols = 1
    for row in range(bsz):
        for image_key in modality_keys:
            value = batch[image_key]
            if not isinstance(value, Tensor):
                continue
            if value.ndim == 4:
                indices = [0]
            elif value.ndim == 5:
                valid_key = f"{image_key}_valid"
                valid_value = batch.get(valid_key, None)
                valid_mask = valid_value if isinstance(valid_value, Tensor) else None
                indices = _get_valid_frame_indices(valid_mask, row, int(value.shape[1]))
            else:
                continue
            frame_indices_map[(row, image_key)] = indices
            max_cols = max(max_cols, len(indices))

    rows = bsz * len(modality_keys)
    cols = max_cols
    fig_w = max(2.2 * cols, 6)
    fig_h = max(2.2 * rows, 4)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False)

    for sample_index in range(bsz):
        cur_label_text = _label_text(labels, sample_index)
        for modality_index, image_key in enumerate(modality_keys):
            row_index = sample_index * len(modality_keys) + modality_index
            image_tensor = batch.get(image_key, None)
            if not isinstance(image_tensor, Tensor):
                for col in range(cols):
                    axes[row_index][col].axis("off")
                continue

            frame_indices = frame_indices_map.get((sample_index, image_key), [])
            if not frame_indices:
                for col in range(cols):
                    axes[row_index][col].axis("off")
                continue

            for col in range(cols):
                ax = axes[row_index][col]
                if col >= len(frame_indices):
                    ax.axis("off")
                    continue
                t_idx = frame_indices[col]
                if image_tensor.ndim == 4:
                    image_chw = image_tensor[sample_index]
                elif image_tensor.ndim == 5:
                    image_chw = image_tensor[sample_index, t_idx]
                else:
                    ax.axis("off")
                    continue

                if image_key == "image_s2":
                    rgb = _to_display_rgb(image_chw, rgb_indices=(3, 2, 1))
                else:
                    rgb = _to_display_rgb(image_chw, rgb_indices=None)
                ax.imshow(rgb.numpy())
                ax.set_xticks([])
                ax.set_yticks([])
                if col == 0:
                    ax.set_title(f"{image_key} | label={cur_label_text}", fontsize=8)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"Saved batch plot to: {save_path}")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[4]
    cfg_path = repo_root / "scripts/configs/classification/dataset/treesatai_ts.yaml"
    cfg = OmegaConf.load(cfg_path)
    train_cfg = cfg.train
    raw_dataset_kwargs = OmegaConf.to_container(train_cfg.dataset_kwargs, resolve=True)
    raw_loader_kwargs = OmegaConf.to_container(train_cfg.loader_kwargs, resolve=True)
    pad_time_series = bool(train_cfg.pad_time_series)

    if not isinstance(raw_dataset_kwargs, dict) or not isinstance(raw_loader_kwargs, dict):
        raise ValueError("dataset_kwargs and loader_kwargs must be mapping objects.")
    dataset_kwargs: dict[str, Any] = {str(key): value for key, value in raw_dataset_kwargs.items()}
    loader_kwargs: dict[str, Any] = {str(key): value for key, value in raw_loader_kwargs.items()}

    dataset, dataloader = TreeSatAITimeSeriesDataset.create_dataloader(
        dataset_kwargs=dataset_kwargs,
        loader_kwargs=loader_kwargs,
        pad_time_series=pad_time_series,
    )
    print(f"dataset size: {len(dataset)}")
    first_batch = next(iter(dataloader))
    if not isinstance(first_batch, dict):
        raise ValueError(f"Expected dict batch, got {type(first_batch)}")
    _print_batch_debug_info(first_batch)
    plot_batch_modalities(first_batch, save_path="treesatai_batch_debug.jpg")
