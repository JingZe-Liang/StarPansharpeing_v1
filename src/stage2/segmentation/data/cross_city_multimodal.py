from __future__ import annotations

import io
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import h5py
import litdata as ld
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
import tifffile
from PIL import Image, ImageDraw
from safetensors import safe_open
from kornia.augmentation import AugmentationSequential, RandomHorizontalFlip, RandomRotation, RandomVerticalFlip
from litdata import StreamingDataLoader
from loguru import logger
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.data import _BaseStreamingDataset
from src.utilities.config_utils import function_config_to_easy_dict


def get_default_transform(p: float = 0.5) -> AugmentationSequential:
    return AugmentationSequential(
        RandomHorizontalFlip(p=p),
        RandomVerticalFlip(p=p),
        RandomRotation(degrees=90, p=p, align_corners=False),
        data_keys=["input", "mask"],
        same_on_batch=False,
        keepdim=True,
    )


def generate_sliding_positions(size: int, patch_size: int, stride: int) -> list[int]:
    if patch_size <= 0 or stride <= 0:
        raise ValueError(f"patch_size and stride must be > 0, got {patch_size=}, {stride=}")
    if size < patch_size:
        raise ValueError(f"Image size {size} is smaller than patch size {patch_size}.")

    positions = list(range(0, size - patch_size + 1, stride))
    last_pos = size - patch_size
    if positions[-1] != last_pos:
        positions.append(last_pos)
    return positions


def generate_patch_coords(height: int, width: int, patch_size: int, stride: int) -> list[tuple[int, int]]:
    y_positions = generate_sliding_positions(height, patch_size, stride)
    x_positions = generate_sliding_positions(width, patch_size, stride)
    return [(y, x) for y in y_positions for x in x_positions]


def remap_background_to_ignore(label: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    mapped = label.astype(np.int64, copy=True)
    is_bg = mapped == 0
    mapped[~is_bg] -= 1
    mapped[is_bg] = ignore_index
    return mapped


def normalize_minmax(img: np.ndarray, per_band: bool = True, eps: float = 1e-6) -> np.ndarray:
    data = img.astype(np.float32, copy=False)
    if per_band:
        flat = data.reshape(-1, data.shape[-1])
        min_v = flat.min(axis=0, keepdims=True)
        max_v = flat.max(axis=0, keepdims=True)
        scale = np.maximum(max_v - min_v, eps)
        out = (flat - min_v) / scale
        return out.reshape(data.shape).astype(np.float32, copy=False)

    min_v = float(data.min())
    max_v = float(data.max())
    scale = max(max_v - min_v, eps)
    return ((data - min_v) / scale).astype(np.float32, copy=False)


def to_neg_one_to_one(img: np.ndarray) -> np.ndarray:
    return (img * 2.0 - 1.0).astype(np.float32, copy=False)


@dataclass
class _SplitFiles:
    image_file: Path
    label_file: Path | None
    use_h5: bool


@dataclass
class CrossCityLitDataStreamKwargs:
    transform: Callable | Literal["default"] | None = "default"
    patch_resize_to: int | None = None
    to_neg_1_1: bool = True
    convert_bg_to_ignore: bool = True
    ignore_index: int = 255
    upsample_hsi_to_msi: bool = True
    shuffle: bool = False
    is_cycled: bool = False
    index_file_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CrossCityLitDataCombinedKwargs:
    batching_method: str = "per_stream"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CrossCityLitDataLoaderKwargs:
    batch_size: int = 4
    num_workers: int = 2
    shuffle: bool = True
    pin_memory: bool = True
    drop_last: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CrossCityTrainLitDataConfig:
    input_dir: str | list[str]
    stream_ds_kwargs: CrossCityLitDataStreamKwargs = field(default_factory=CrossCityLitDataStreamKwargs)
    combined_kwargs: CrossCityLitDataCombinedKwargs = field(default_factory=CrossCityLitDataCombinedKwargs)
    loader_kwargs: CrossCityLitDataLoaderKwargs = field(default_factory=CrossCityLitDataLoaderKwargs)


class CrossCityMultimodalSegmentationDataset(Dataset[dict[str, Tensor]]):
    DATASET_NAMES = ("augsburg", "beijing")

    def __init__(
        self,
        data_root: str,
        dataset_name: Literal["augsburg", "beijing"] = "augsburg",
        split: Literal["train", "test"] = "train",
        patch_size: int = 256,
        patch_resize_to: int | None = None,
        full_image_upsample_scale: float | None = None,
        stride: int = 128,
        apply_normalization: bool = True,
        normalize_per_band: bool = True,
        to_neg_1_1: bool = True,
        convert_bg_to_ignore: bool = True,
        ignore_index: int = 255,
        upsample_hsi_to_msi: bool = True,
        transform: Callable | Literal["default"] | None = None,
        return_modalities: bool = False,
    ) -> None:
        self.data_root = Path(data_root)
        self.dataset_name = dataset_name
        self.split = split
        self.patch_size = patch_size
        self.patch_resize_to = patch_resize_to
        self.full_image_upsample_scale = full_image_upsample_scale
        self.stride = stride
        self.apply_normalization = apply_normalization
        self.normalize_per_band = normalize_per_band
        self.to_neg_1_1 = to_neg_1_1
        self.convert_bg_to_ignore = convert_bg_to_ignore
        self.ignore_index = int(ignore_index)
        self.upsample_hsi_to_msi = upsample_hsi_to_msi
        self.return_modalities = return_modalities

        if dataset_name not in self.DATASET_NAMES:
            raise ValueError(f"Unknown dataset_name={dataset_name}, expected one of {self.DATASET_NAMES}")
        if split not in ("train", "test"):
            raise ValueError(f"Unknown split={split}, expected train/test")
        if split == "train" and (patch_size <= 0 or stride <= 0):
            raise ValueError(f"Train split requires valid patching params, got {patch_size=}, {stride=}")
        if patch_resize_to is not None and patch_resize_to <= 0:
            raise ValueError(f"patch_resize_to must be > 0, got {patch_resize_to}")
        if full_image_upsample_scale is not None and full_image_upsample_scale <= 0:
            raise ValueError(f"full_image_upsample_scale must be > 0, got {full_image_upsample_scale}")

        if transform == "default":
            transform = get_default_transform()
        self.transform = transform

        self.hsi, self.msi, self.sar, self.gt = self._load_split_data()

        if self.convert_bg_to_ignore:
            self.gt = remap_background_to_ignore(self.gt, ignore_index=self.ignore_index)
        else:
            self.gt = self.gt.astype(np.int64, copy=False)

        self.n_classes = int(len(np.unique(self.gt[self.gt != self.ignore_index])))
        self.modal_channels = [int(self.hsi.shape[-1]), int(self.msi.shape[-1]), int(self.sar.shape[-1])]
        self.total_channels = int(sum(self.modal_channels))

        self.patch_coords: list[tuple[int, int]] = []
        if self.split == "train":
            h, w = self.gt.shape
            self.patch_coords = generate_patch_coords(h, w, self.patch_size, self.stride)

        logger.info(
            f"[CrossCity Dataset] split={self.split}, dataset={self.dataset_name}, "
            f"shape(h,w)=({self.gt.shape[0]},{self.gt.shape[1]}), channels={self.modal_channels}, "
            f"num_classes={self.n_classes}, patches={len(self.patch_coords)}"
        )

    def _resolve_split_files(self) -> _SplitFiles:
        logger.info(f"[CrossCity Dataset] Loading {self.split} {self.dataset_name} data")
        if self.dataset_name == "augsburg":
            image_file = (
                self.data_root
                / "data1"
                / ("augsburg_multimodal.mat" if self.split == "train" else "berlin_multimodal.mat")
            )
            return _SplitFiles(image_file=image_file, label_file=None, use_h5=False)

        base_img_dir = self.data_root / "data2"
        name = "beijing" if self.split == "train" else "wuhan"
        safetensors_file = base_img_dir / f"{name}.safetensors"
        mat_file = base_img_dir / f"{name}.mat"
        image_file = safetensors_file if safetensors_file.exists() else mat_file
        label_file = self.data_root / "data2" / ("beijing_label.mat" if self.split == "train" else "wuhan_label.mat")
        return _SplitFiles(image_file=image_file, label_file=label_file, use_h5=True)

    @staticmethod
    def _read_h5_cube(h5_file: h5py.File, key: str) -> np.ndarray:
        arr = np.array(h5_file[key])
        if arr.ndim == 3 and arr.shape[0] < arr.shape[-1]:
            arr = np.transpose(arr, (1, 2, 0))
        return arr

    @staticmethod
    def _read_safetensors_cube(file_path: Path, key: str) -> np.ndarray:
        with safe_open(str(file_path), framework="np", device="cpu") as f:
            keys = list(f.keys())
            if key not in keys:
                raise KeyError(f"Key '{key}' not found in {file_path}. Available keys: {keys}")
            arr = f.get_tensor(key)
        if arr.ndim == 3 and arr.shape[0] < arr.shape[-1]:
            arr = np.transpose(arr, (1, 2, 0))
        return np.asarray(arr)

    @staticmethod
    def _upsample_hsi(hsi: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
        if hsi.shape[:2] == target_hw:
            return hsi
        t = torch.from_numpy(hsi).permute(2, 0, 1).unsqueeze(0).float()
        up = F.interpolate(t, size=target_hw, mode="nearest")
        return up.squeeze(0).permute(1, 2, 0).numpy().astype(np.float32, copy=False)

    def _load_split_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        files = self._resolve_split_files()
        if not files.image_file.exists():
            raise FileNotFoundError(f"Missing image file: {files.image_file}")

        if not files.use_h5:
            data = sio.loadmat(files.image_file)
            hsi = data["HSI"]
            msi = data["MSI"]
            sar = data["SAR"]
            gt = data["label"]
            if self.dataset_name == "augsburg" and self.split == "test" and hsi.shape[-1] > 242:
                hsi = hsi[:, :, :242]
        else:
            if files.image_file.suffix == ".safetensors":
                hsi = self._read_safetensors_cube(files.image_file, "HSI")
                msi = self._read_safetensors_cube(files.image_file, "MSI")
                sar = self._read_safetensors_cube(files.image_file, "SAR")
                gt = None
            else:
                with h5py.File(files.image_file, "r") as f:
                    hsi = self._read_h5_cube(f, "HSI")
                    msi = self._read_h5_cube(f, "MSI")
                    sar = self._read_h5_cube(f, "SAR")
                gt = None

            if files.label_file is None or not files.label_file.exists():
                raise FileNotFoundError(f"Missing label file: {files.label_file}")
            with h5py.File(files.label_file, "r") as f_label:
                gt = self._read_h5_cube(f_label, "label")

            if gt is None:
                raise ValueError(f"No label found for {files.image_file}")

            if self.upsample_hsi_to_msi:
                hsi = self._upsample_hsi(hsi.astype(np.float32, copy=False), msi.shape[:2])

        if gt.ndim == 3:
            gt = np.squeeze(gt)
        if gt.ndim != 2:
            raise ValueError(f"GT must be 2D, got shape {gt.shape}")

        hsi = hsi.astype(np.float32, copy=False)
        msi = msi.astype(np.float32, copy=False)
        sar = sar.astype(np.float32, copy=False)

        logger.debug(f"Processing hsi/msi/sar images ...")
        if self.apply_normalization:
            hsi = normalize_minmax(hsi, per_band=self.normalize_per_band)
            msi = normalize_minmax(msi, per_band=self.normalize_per_band)
            sar = normalize_minmax(sar, per_band=self.normalize_per_band)

        if self.apply_normalization and self.to_neg_1_1:
            hsi = to_neg_one_to_one(hsi)
            msi = to_neg_one_to_one(msi)
            sar = to_neg_one_to_one(sar)

        return hsi, msi, sar, gt.astype(np.int64, copy=False)

    def __len__(self) -> int:
        if self.split == "train":
            return len(self.patch_coords)
        return 1

    def _slice_patch(self, arr: np.ndarray, y: int, x: int) -> np.ndarray:
        return arr[y : y + self.patch_size, x : x + self.patch_size, ...]

    def _build_output(self, hsi: np.ndarray, msi: np.ndarray, sar: np.ndarray, gt: np.ndarray) -> dict[str, Tensor]:
        if not (hsi.shape[:2] == msi.shape[:2] == sar.shape[:2] == gt.shape[:2]):
            raise ValueError(
                "Spatial shape mismatch among modalities/gt. "
                f"Got hsi={hsi.shape[:2]}, msi={msi.shape[:2]}, sar={sar.shape[:2]}, gt={gt.shape[:2]}. "
                "Set upsample_hsi_to_msi=true for datasets with different native resolutions."
            )
        hsi_t = torch.from_numpy(hsi).permute(2, 0, 1).float()
        msi_t = torch.from_numpy(msi).permute(2, 0, 1).float()
        sar_t = torch.from_numpy(sar).permute(2, 0, 1).float()
        gt_t = torch.from_numpy(gt).long()

        img = torch.cat([hsi_t, msi_t, sar_t], dim=0)

        if self.split == "train" and self.patch_resize_to is not None:
            img = (
                F.interpolate(
                    img.unsqueeze(0),
                    size=(self.patch_resize_to, self.patch_resize_to),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .float()
            )
            gt_t = (
                F.interpolate(
                    gt_t.unsqueeze(0).unsqueeze(0).float(),
                    size=(self.patch_resize_to, self.patch_resize_to),
                    mode="nearest",
                )
                .squeeze(0)
                .squeeze(0)
                .long()
            )

        if self.split == "train" and self.transform is not None:
            img_b = img.unsqueeze(0)
            gt_b = gt_t.unsqueeze(0).unsqueeze(0).float()
            img_b, gt_b = self.transform(img_b, gt_b)
            img = img_b.squeeze(0)
            gt_t = gt_b.squeeze(0).squeeze(0).long()

        out: dict[str, Tensor] = {"img": img, "gt": gt_t}
        if self.return_modalities:
            c_hsi, c_msi, _ = self.modal_channels
            out["img_hsi"] = img[:c_hsi]
            out["img_msi"] = img[c_hsi : c_hsi + c_msi]
            out["img_sar"] = img[c_hsi + c_msi :]
        return out

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        if self.split == "train":
            y, x = self.patch_coords[index]
            hsi = self._slice_patch(self.hsi, y, x)
            msi = self._slice_patch(self.msi, y, x)
            sar = self._slice_patch(self.sar, y, x)
            gt = self._slice_patch(self.gt, y, x)
            return self._build_output(hsi, msi, sar, gt)

        out = self._build_output(self.hsi, self.msi, self.sar, self.gt)
        scale = self.full_image_upsample_scale
        if scale is None or np.isclose(scale, 1.0):
            return out

        img = out["img"].unsqueeze(0)
        gt = out["gt"].unsqueeze(0).unsqueeze(0).float()
        img = F.interpolate(img, scale_factor=scale, mode="bilinear", align_corners=False).squeeze(0)
        gt = F.interpolate(gt, scale_factor=scale, mode="nearest").squeeze(0).squeeze(0).long()

        out["img"] = img
        out["gt"] = gt
        if self.return_modalities:
            c_hsi, c_msi, _ = self.modal_channels
            out["img_hsi"] = img[:c_hsi]
            out["img_msi"] = img[c_hsi : c_hsi + c_msi]
            out["img_sar"] = img[c_hsi + c_msi :]
        return out

    @classmethod
    @function_config_to_easy_dict
    def from_config(cls, cfg):
        ds = cls(**cfg.dataset_kwargs)
        dl_cfg = cfg.dataloader_kwargs
        dl = DataLoader(
            ds,
            batch_size=dl_cfg.batch_size,
            num_workers=dl_cfg.num_workers,
            shuffle=dl_cfg.get("shuffle", False),
            pin_memory=dl_cfg.get("pin_memory", True),
            drop_last=dl_cfg.get("drop_last", False),
        )
        return ds, dl


class CrossCityMultimodalPatchStreamingDataset(_BaseStreamingDataset):
    """
    Training-only streaming dataset for pre-cut CrossCity patches stored in LitData.
    Each sample must contain:
    - img_hsi/img_msi/img_sar: TIFF(zlib) bytes or CHW arrays/tensors
    - gt: int tensor/array in HW
    """

    def __init__(
        self,
        input_dir: str,
        transform: Callable | Literal["default"] | None = None,
        patch_resize_to: int | None = None,
        to_neg_1_1: bool = True,
        convert_bg_to_ignore: bool = True,
        ignore_index: int = 255,
        upsample_hsi_to_msi: bool = True,
        **kwargs: Any,
    ) -> None:
        if transform == "default":
            transform = get_default_transform()
        self.aug_transform = transform
        self.patch_resize_to = patch_resize_to
        self.to_neg_1_1 = to_neg_1_1
        self.convert_bg_to_ignore = convert_bg_to_ignore
        self.ignore_index = int(ignore_index)
        self.upsample_hsi_to_msi = upsample_hsi_to_msi

        if self.patch_resize_to is not None and self.patch_resize_to <= 0:
            raise ValueError(f"patch_resize_to must be > 0, got {self.patch_resize_to}")
        super().__init__(input_dir=input_dir, **kwargs)  # type: ignore[unknown-argument]

    @staticmethod
    def _to_tensor_img(img: np.ndarray | Tensor) -> Tensor:
        if isinstance(img, Tensor):
            img_t = img
        else:
            img_t = torch.from_numpy(np.array(img, copy=True))
        img_t = img_t.float()
        if img_t.ndim == 2:
            img_t = img_t.unsqueeze(0)
        if img_t.ndim != 3:
            raise ValueError(f"Expected img ndim=3 (CHW), got shape={tuple(img_t.shape)}")
        return img_t

    @staticmethod
    def _to_tensor_gt(gt: np.ndarray | Tensor) -> Tensor:
        if isinstance(gt, Tensor):
            gt_t = gt
        else:
            gt_t = torch.from_numpy(np.array(gt, copy=True))
        gt_t = gt_t.long()
        if gt_t.ndim == 3 and gt_t.shape[0] == 1:
            gt_t = gt_t.squeeze(0)
        if gt_t.ndim != 2:
            raise ValueError(f"Expected gt ndim=2 (HW), got shape={tuple(gt_t.shape)}")
        return gt_t

    @staticmethod
    def _resize_img_gt(img: Tensor, gt: Tensor, patch_resize_to: int) -> tuple[Tensor, Tensor]:
        img = (
            F.interpolate(
                img.unsqueeze(0),
                size=(patch_resize_to, patch_resize_to),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .float()
        )
        gt = (
            F.interpolate(
                gt.unsqueeze(0).unsqueeze(0).float(),
                size=(patch_resize_to, patch_resize_to),
                mode="nearest",
            )
            .squeeze(0)
            .squeeze(0)
            .long()
        )
        return img, gt

    @staticmethod
    def _maybe_upsample_hsi_to_match_msi(hsi: Tensor, msi: Tensor, enabled: bool) -> Tensor:
        if hsi.shape[-2:] == msi.shape[-2:]:
            return hsi
        if not enabled:
            raise ValueError(
                f"HSI/MSI spatial mismatch: hsi={tuple(hsi.shape[-2:])}, msi={tuple(msi.shape[-2:])}. "
                "Set upsample_hsi_to_msi=True to align HSI."
            )
        return (
            F.interpolate(
                hsi.unsqueeze(0),
                size=msi.shape[-2:],
                mode="nearest",
            )
            .squeeze(0)
            .float()
        )

    def _maybe_postprocess_img_gt(self, img: Tensor, gt: Tensor) -> tuple[Tensor, Tensor]:
        if self.patch_resize_to is not None:
            img, gt = self._resize_img_gt(img, gt, self.patch_resize_to)
        if self.to_neg_1_1:
            img = img * 2.0 - 1.0
        if self.convert_bg_to_ignore:
            bg = gt == 0
            gt = gt.clone()
            gt[~bg] = gt[~bg] - 1
            gt[bg] = self.ignore_index
        return img, gt

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        sample = super().__getitem__(idx)
        if "gt" not in sample:
            raise KeyError(f"LitData sample must contain 'gt', got keys={list(sample.keys())}")

        if all(k in sample for k in ("img_hsi", "img_msi", "img_sar")):
            hsi_raw = sample["img_hsi"]
            msi_raw = sample["img_msi"]
            sar_raw = sample["img_sar"]
            if isinstance(hsi_raw, bytes) or isinstance(msi_raw, bytes) or isinstance(sar_raw, bytes):
                raise ValueError(
                    "Expected decoded numpy arrays/tensors from LitData, but got bytes. "
                    "Please ensure index/serializer auto-decodes TIFF bytes."
                )
            hsi = self._to_tensor_img(hsi_raw)
            msi = self._to_tensor_img(msi_raw)
            sar = self._to_tensor_img(sar_raw)
            hsi = self._maybe_upsample_hsi_to_match_msi(hsi, msi, enabled=self.upsample_hsi_to_msi)
            if sar.shape[-2:] != msi.shape[-2:]:
                raise ValueError(f"SAR/MSI spatial mismatch: sar={tuple(sar.shape[-2:])}, msi={tuple(msi.shape[-2:])}.")
            img = torch.cat([hsi, msi, sar], dim=0)
        elif "img" in sample:
            # backward-compatible reading for previously exported litdata
            img_raw = sample["img"]
            if isinstance(img_raw, bytes):
                raise ValueError(
                    "Expected decoded numpy array/tensor for key 'img', but got bytes. "
                    "Please ensure index/serializer auto-decodes TIFF bytes."
                )
            img = self._to_tensor_img(img_raw)
        else:
            raise KeyError(
                "LitData sample must contain either ('img_hsi','img_msi','img_sar') or 'img'. "
                f"Got keys={list(sample.keys())}"
            )

        gt = self._to_tensor_gt(sample["gt"])
        img, gt = self._maybe_postprocess_img_gt(img, gt)

        if self.aug_transform is not None:
            img_b = img.unsqueeze(0)
            gt_b = gt.unsqueeze(0).unsqueeze(0).float()
            img_b, gt_b = self.aug_transform(img_b, gt_b)
            img = img_b.squeeze(0).float()
            gt = gt_b.squeeze(0).squeeze(0).long()

        return {"img": img, "gt": gt}

    @classmethod
    def create_dataloader(
        cls,
        input_dir: str | list[str],
        stream_ds_kwargs: dict | None = None,
        combined_kwargs: dict | None = None,
        loader_kwargs: dict | None = None,
    ):
        stream_ds_kwargs = stream_ds_kwargs or {}
        combined_kwargs = combined_kwargs or {"batching_method": "per_stream"}
        loader_kwargs = loader_kwargs or {}
        ds = cls.create_dataset(input_dir=input_dir, combined_kwargs=combined_kwargs, **stream_ds_kwargs)
        dl = StreamingDataLoader(ds, **loader_kwargs)
        return ds, dl

    @classmethod
    @function_config_to_easy_dict
    def from_config(cls, cfg):
        stream_cfg = dict(cfg.get("stream_ds_kwargs", {}))
        combined_cfg = dict(cfg.get("combined_kwargs", {}))
        loader_cfg = dict(cfg.get("loader_kwargs", {}))

        config = CrossCityTrainLitDataConfig(
            input_dir=cfg.input_dir,
            stream_ds_kwargs=CrossCityLitDataStreamKwargs(**stream_cfg),
            combined_kwargs=CrossCityLitDataCombinedKwargs(**combined_cfg),
            loader_kwargs=CrossCityLitDataLoaderKwargs(**loader_cfg),
        )
        return cls.create_dataloader(
            input_dir=config.input_dir,
            stream_ds_kwargs=config.stream_ds_kwargs.to_dict(),
            combined_kwargs=config.combined_kwargs.to_dict(),
            loader_kwargs=config.loader_kwargs.to_dict(),
        )


def test_dataset_plot():
    input_dir = Path("data/Downstreams/CrossCitySegmentation/litdata_train/beijing/train")
    if not input_dir.exists():
        raise FileNotFoundError(f"LitData train dir not found: {input_dir}")

    ds = CrossCityMultimodalPatchStreamingDataset(
        input_dir=str(input_dir),
        transform=None,
        patch_resize_to=None,
        to_neg_1_1=False,
        convert_bg_to_ignore=False,
        ignore_index=255,
        upsample_hsi_to_msi=True,
    )

    sample = ds[200]
    img = sample["img"].cpu().numpy()
    gt = sample["gt"].cpu().numpy()

    if img.shape[0] == 248:
        c_hsi, c_msi, c_sar = 242, 4, 2
    elif img.shape[0] == 122:
        c_hsi, c_msi, c_sar = 116, 4, 2
    else:
        raise ValueError(f"Unexpected channel size: {img.shape[0]}")

    hsi = img[:c_hsi]
    msi = img[c_hsi : c_hsi + c_msi]
    sar = img[c_hsi + c_msi : c_hsi + c_msi + c_sar]

    def _to_u8(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        min_v = float(np.nanmin(x))
        max_v = float(np.nanmax(x))
        if max_v - min_v < 1e-8:
            return np.zeros(x.shape, dtype=np.uint8)
        x = (x - min_v) / (max_v - min_v)
        x = np.clip(x * 255.0, 0.0, 255.0)
        return x.astype(np.uint8)

    hsi_rgb = np.stack([_to_u8(hsi[0]), _to_u8(hsi[hsi.shape[0] // 2]), _to_u8(hsi[-1])], axis=-1)
    msi_rgb = np.stack([_to_u8(msi[0]), _to_u8(msi[1]), _to_u8(msi[2])], axis=-1)
    sar_g = _to_u8(sar[0])
    sar_rgb = np.stack([sar_g, sar_g, sar_g], axis=-1)

    palette = np.array(
        [
            [0, 0, 0],
            [230, 25, 75],
            [60, 180, 75],
            [255, 225, 25],
            [0, 130, 200],
            [245, 130, 48],
            [145, 30, 180],
            [70, 240, 240],
            [240, 50, 230],
            [210, 245, 60],
            [250, 190, 190],
            [0, 128, 128],
            [230, 190, 255],
            [170, 110, 40],
        ],
        dtype=np.uint8,
    )
    gt_idx = np.clip(gt.astype(np.int64, copy=False), 0, palette.shape[0] - 1)
    gt_rgb = palette[gt_idx]

    out_dir = Path("outputs/cross_city_preview")
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(hsi_rgb).save(out_dir / "beijing_sample0_hsi.png")
    Image.fromarray(msi_rgb).save(out_dir / "beijing_sample0_msi.png")
    Image.fromarray(sar_rgb).save(out_dir / "beijing_sample0_sar.png")
    Image.fromarray(gt_rgb).save(out_dir / "beijing_sample0_gt_color.png")

    panels = [hsi_rgb, msi_rgb, sar_rgb, gt_rgb]
    titles = [f"HSI({c_hsi})", f"MSI({c_msi})", f"SAR({c_sar})", "GT"]
    h, w, _ = hsi_rgb.shape
    header_h = 28
    gap = 8
    canvas = Image.new("RGB", (4 * w + 3 * gap, h + header_h), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    x0 = 0
    for title, panel in zip(titles, panels):
        draw.rectangle([x0, 0, x0 + w, header_h], fill=(35, 35, 35))
        draw.text((x0 + 6, 6), title, fill=(255, 255, 255))
        canvas.paste(Image.fromarray(panel), (x0, header_h))
        x0 += w + gap
    canvas.save(out_dir / "beijing_sample0_modal_gt.png")

    print(f"Saved previews to: {out_dir}")
