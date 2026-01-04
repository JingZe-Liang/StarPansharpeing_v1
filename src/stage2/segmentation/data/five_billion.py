"""
Five-Billion China Segmentation Dataset Litdata Dataset
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import Tensor
from kornia.augmentation import (
    AugmentationSequential,
    RandomResizedCrop,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomBoxBlur,
)
from litdata import StreamingDataLoader
from functools import partial
from torch.utils.data import Dataset
import tifffile
from PIL import Image

from src.data import _BaseStreamingDataset


def collate_fn(batch: list[dict], transforms=None) -> dict:
    """
    Collate function that handles string fields like __key__ by collecting them into lists,
    while stacking tensors for other fields.

    Parameters
    ----------
    batch : list of dict
        List of samples from dataset

    Returns
    -------
    dict
        Batched data
    """
    if not batch:
        return {}

    keys = batch[0].keys()
    collated = {}

    for key in keys:
        values = [sample[key] for sample in batch]

        # For __key__ or other string fields, keep as list
        if key == "__key__" or isinstance(values[0], str):
            collated[key] = values
        else:
            # Stack tensors for other fields
            collated[key] = torch.stack(values, dim=0)

    if transforms is not None:
        img, gt = collated["img"], collated["gt"]
        for transf in transforms:
            img, gt = transf(img, gt)
        collated["img"], collated["gt"] = img, gt

    return collated


class FiveBillionStreamingDataset(_BaseStreamingDataset):
    def __init__(
        self,
        img_size: int = 512,
        to_neg_1_1: bool = True,
        augmentation_prob=0.5,
        convert_bg: bool = True,
        norm_const: float = 1024.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.img_size = img_size
        self.to_neg_1_1 = to_neg_1_1
        self.convert_bg = convert_bg
        self.norm_const = norm_const

        # Define Augmentations
        # Kornia AugmentationSequential with "input" and "mask" keys handles
        # interpolation automatically (Bilinear for input, Nearest for mask)
        self.resize = AugmentationSequential(
            RandomResizedCrop(
                size=(img_size, img_size),
                scale=(0.5, 1.0),
                ratio=(3 / 4, 4 / 3),
                p=1.0,
                keepdim=True,
            ),
            data_keys=["input", "mask"],
            keepdim=True,
        )

        # Let collate fn to augment
        self.transforms = None
        if augmentation_prob > 0:
            self.transforms = AugmentationSequential(
                RandomHorizontalFlip(p=augmentation_prob),
                RandomVerticalFlip(p=augmentation_prob),
                # RandomBoxBlur(kernel_size=(7, 7), p=augmentation_prob),
                random_apply=1,
                data_keys=["input", "mask"],
                keepdim=True,
            )

            # Disable auto conversion to ensure we handle tensors directly
            if hasattr(self.transforms, "_disable_features"):
                self.transforms._disable_features = True

    def _apply_transforms(self, img: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Apply transforms to image and mask."""
        # Apply transforms jointly
        # Kornia automatically applies Nearest Neighbor interpolation for "mask" data key
        img, mask = self.resize(img, mask)
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def _preprocess_image(self, img: Tensor) -> Tensor:
        """Normalize image."""
        if not isinstance(img, Tensor):
            img = torch.tensor(img)

        # Assuming img is [C, H, W] or [1, C, H, W]
        img = img.float() / self.norm_const
        img = img.clamp(0.0, 1.0)

        if self.to_neg_1_1:
            img = img * 2.0 - 1.0

        return img

    def __getitem__(self, idx: int) -> dict:
        sample = super().__getitem__(idx)

        img = sample["img"]
        mask = sample["label"]

        # 1. Preprocess Image
        img = self._preprocess_image(img)

        # 2. Preprocess Mask
        if not isinstance(mask, Tensor):
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask)
            else:
                mask = torch.tensor(mask)

        mask = mask.long()

        if self.convert_bg:
            # Convert background (0) to 255, but preserve existing 255 (padding)
            # 1. First shift everything down: x -> x-1 (so 1->0, 2->1).
            #    Note: 0 becomes -1, and originally 255 would become 254.
            # 2. We want original 0 -> 255.
            # 3. We want original 255 -> 255.
            # Logic flow:
            #   mask_shifted = mask - 1
            #   mask = where(mask == 0, 255, mask_shifted)
            #   mask = where(mask == 255, 255, mask) (Wait, original 255 check needs to be on ORIGINAL mask)

            # Let's use the logic from single_mat_loader:
            # fixed_padded_label = where(label == 255, 255, label - 1)
            # result = where(label == 0, 255, fixed_padded_label)

            fixed_shifted = torch.where(mask == 255, torch.tensor(255, device=mask.device), mask - 1)
            mask = torch.where(mask == 0, torch.tensor(255, device=mask.device), fixed_shifted)

        # Ensure dimensions [C, H, W]
        if img.ndim == 2:
            img = img.unsqueeze(0)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)  # [1, H, W]

        # 3. Apply Augmentations
        img, mask = self._apply_transforms(img, mask)

        return {
            "img": img,
            "gt": mask.squeeze(0),  # Return [H, W]
            # "__key__": sample.get("__key__", str(idx)),
        }

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

        # Set custom collate function if not provided
        if "collate_fn" not in loader_kwargs:
            loader_kwargs["collate_fn"] = partial(collate_fn, transforms=None)

        dl = StreamingDataLoader(ds, **loader_kwargs)
        return ds, dl


class FiveBillionLargeImageDataset(Dataset):
    """
    5Billion-ChinaCity large image inference dataset (test mode only).

    - Reads images with ground truth labels.
    - Uses the 30 test images from the paper baseline (consistent with `data/Downstreams/.../readme.md`).
    - Automatically matches `.tif` / `.tiff` extensions for switching between
      `Image_8bit_NirRGB` and `Image_16bit_RGBNir`.
    - Assumes images are loaded in HWC format by default.
    """

    @dataclass(frozen=True, slots=True)
    class _Sample:
        key: str
        img_path: Path
        gt_path: Path

    def __init__(
        self,
        data_dir: str | Path,
        gt_dir: str | Path,
        mode: Literal["test"] = "test",
        *,
        image_variant: Literal["16bit_RGBNir", "8bit_NirRGB"] = "16bit_RGBNir",
        to_neg_1_1: bool = True,
        norm_const: float | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.gt_dir = Path(gt_dir)
        self.mode = mode
        self.image_variant = image_variant
        self.to_neg_1_1 = to_neg_1_1
        self.norm_const = (
            float(norm_const)
            if norm_const is not None
            else {"16bit_RGBNir": 1024.0, "8bit_NirRGB": 255.0}[image_variant]
        )

        if mode != "test":
            raise ValueError("FiveBillionLargeImageDataset only supports mode='test' (for large image inference).")

        self.image_dir = self._resolve_image_dir(self.data_dir, image_variant=image_variant)
        self.samples = self._build_test_samples(self.image_dir, self.gt_dir)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        img = self._read_tif(sample.img_path)
        gt = self._read_gt(sample.gt_path)
        return {
            "__key__": sample.key,
            "img": img,
            "gt": gt,
            # "path": str(sample.img_path),
            "height": int(img.shape[1]),
            "width": int(img.shape[2]),
        }

    def _resolve_image_dir(self, data_dir: Path, *, image_variant: str) -> Path:
        """Resolve image directory from data_dir and image_variant."""
        # 1) data_dir itself is the image directory
        if data_dir.is_dir():
            has_any_tif = any(p.suffix.lower() in {".tif", ".tiff"} for p in data_dir.iterdir() if p.is_file())
            if has_any_tif:
                return data_dir

        candidates = [
            data_dir / f"Image_{image_variant}",
            data_dir / "TrainSet_For_Baseline" / f"Image_{image_variant}",
        ]

        for cand in candidates:
            if cand.is_dir():
                return cand

        raise FileNotFoundError(
            f"Cannot locate FiveBillion image directory. Please check data_dir and image_variant.\n"
            f"data_dir={data_dir}\n"
            f"image_variant={image_variant}\n"
            f"Tried: {', '.join(str(p) for p in candidates)}"
        )

    def _read_tif(self, file_path: Path) -> Tensor:
        """Read tif image file (assumes HWC format by default)."""
        arr = tifffile.imread(file_path)
        x = np.asarray(arr)

        # Assume HWC format by default, convert to CHW
        if x.ndim == 2:
            x = x[:, :, None]  # HW -> HWC
        elif x.ndim != 3:
            raise ValueError(f"Unsupported tif dimensions: {x.shape} (ndim={x.ndim}), file: {file_path}")

        # HWC -> CHW
        t = torch.as_tensor(x).permute(2, 0, 1).contiguous()

        # Normalize and clip
        t = t.float() / self.norm_const
        t = t.clamp(0.0, 1.0)
        if self.to_neg_1_1:
            t = t * 2.0 - 1.0
        return t

    def _read_gt(self, file_path: Path) -> Tensor:
        """Read ground truth label file (supports both TIFF and PNG formats)."""
        # Use PIL for PNG, tifffile for TIFF
        if file_path.suffix.lower() in {".png", ".PNG"}:
            img = Image.open(file_path)
            x = np.array(img)
        else:
            arr = tifffile.imread(file_path)
            x = np.asarray(arr)

        # Assume HW format by default
        if x.ndim == 2:
            t = torch.as_tensor(x).long()
        elif x.ndim == 3:
            # HWC -> HW (take first channel)
            t = torch.as_tensor(x[:, :, 0]).long()
        else:
            raise ValueError(f"Unsupported gt dimensions: {x.shape} (ndim={x.ndim}), file: {file_path}")

        return t

    def _test_set_stems(self) -> list[str]:
        """Get test set stems from paper baseline (Table 4, data/Downstreams/.../readme.md)."""
        return [
            "GF2_PMS1__L1A0001064454-MSS1",
            "GF2_PMS1__L1A0001118839-MSS1",
            "GF2_PMS1__L1A0001344822-MSS1",
            "GF2_PMS1__L1A0001348919-MSS1",
            "GF2_PMS1__L1A0001366278-MSS1",
            "GF2_PMS1__L1A0001366284-MSS1",
            "GF2_PMS1__L1A0001395956-MSS1",
            "GF2_PMS1__L1A0001432972-MSS1",
            "GF2_PMS1__L1A0001670888-MSS1",
            "GF2_PMS1__L1A0001680857-MSS1",
            "GF2_PMS1__L1A0001680858-MSS1",
            "GF2_PMS1__L1A0001757429-MSS1",
            "GF2_PMS1__L1A0001765574-MSS1",
            "GF2_PMS2__L1A0000607677-MSS2",
            "GF2_PMS2__L1A0000607681-MSS2",
            "GF2_PMS2__L1A0000718813-MSS2",
            "GF2_PMS2__L1A0001038935-MSS2",
            "GF2_PMS2__L1A0001038936-MSS2",
            "GF2_PMS2__L1A0001119060-MSS2",
            "GF2_PMS2__L1A0001367840-MSS2",
            "GF2_PMS2__L1A0001378491-MSS2",
            "GF2_PMS2__L1A0001378501-MSS2",
            "GF2_PMS2__L1A0001396036-MSS2",
            "GF2_PMS2__L1A0001396037-MSS2",
            "GF2_PMS2__L1A0001416129-MSS2",
            "GF2_PMS2__L1A0001471436-MSS2",
            "GF2_PMS2__L1A0001517494-MSS2",
            "GF2_PMS2__L1A0001591676-MSS2",
            "GF2_PMS2__L1A0001787564-MSS2",
            "GF2_PMS2__L1A0001821754-MSS2",
        ]

    def _find_existing_image(self, image_dir: Path, *, stem: str) -> Path:
        """Find existing image file with various extensions."""
        for ext in (".tif", ".tiff", ".TIF", ".TIFF"):
            path = image_dir / f"{stem}{ext}"
            if path.exists():
                return path
        raise FileNotFoundError(f"Test image not found: {stem}.tif/.tiff (dir: {image_dir})")

    def _find_existing_gt(self, gt_dir: Path, *, stem: str) -> Path:
        """Find existing ground truth file with various extensions."""
        # Try tif/tiff extensions first
        for ext in (".tif", ".tiff", ".TIF", ".TIFF"):
            path = gt_dir / f"{stem}{ext}"
            if path.exists():
                return path

        # Try PNG format with _24label suffix (for this dataset)
        for ext in (".png", ".PNG"):
            path = gt_dir / f"{stem}_24label{ext}"
            if path.exists():
                return path

        raise FileNotFoundError(f"Ground truth not found: {stem}.tif/.tiff/{stem}_24label.png (dir: {gt_dir})")

    def _build_test_samples(self, image_dir: Path, gt_dir: Path) -> list[_Sample]:
        """Build test samples from image and gt directories."""
        samples: list[FiveBillionLargeImageDataset._Sample] = []
        for stem in self._test_set_stems():
            img_path = self._find_existing_image(image_dir, stem=stem)
            gt_path = self._find_existing_gt(gt_dir, stem=stem)
            samples.append(self._Sample(key=img_path.name, img_path=img_path, gt_path=gt_path))
        return samples


def __test_steaming_dataset():
    from loguru import logger

    # Test block
    # Path to the dataset
    data_path = "data/Downstreams/5Billion-ChinaCity-Segmentation/litdata/test"

    print(f"Testing FiveBillionStreamingDataset with path: {data_path}")

    ds, dl = FiveBillionStreamingDataset.create_dataloader(
        input_dir=data_path, stream_ds_kwargs={"img_size": 512}, loader_kwargs={"batch_size": 4, "num_workers": 0}
    )

    print(f"Dataset length: {len(ds)}")

    # # Try to fetch one sample
    with logger.catch():
        for i, batch in enumerate(dl):
            print(f"Batch {i} keys: {list(batch.keys())}")
            if "img" in batch:
                print(f"Image shape: {batch['img'].shape}, range: [{batch['img'].min():.3f}, {batch['img'].max():.3f}]")
            if "gt" in batch:
                print(f"Mask shape: {batch['gt'].shape}, distinct values: {torch.unique(batch['gt'])}")

            # if i >= 0:  # Check just one batch
            #     break


def __test_large_test_dataset():
    """Test FiveBillionLargeImageDataset."""
    # Data directory
    data_dir = "data/Downstreams/5Billion-ChinaCity-Segmentation/TrainSet_For_Baseline"
    gt_dir = "data/Downstreams/5Billion-ChinaCity-Segmentation/TrainSet_For_Baseline/Annotation_Index"

    print(f"Testing FiveBillionLargeImageDataset")
    print(f"Data directory: {data_dir}")
    print(f"GT directory: {gt_dir}")

    # Create dataset
    ds = FiveBillionLargeImageDataset(
        data_dir=data_dir,
        gt_dir=gt_dir,
        mode="test",
        image_variant="16bit_RGBNir",
        to_neg_1_1=True,
    )

    print(f"Dataset length: {len(ds)}")

    # Test first sample
    sample = ds[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"Image shape: {sample['img'].shape}, range: [{sample['img'].min():.3f}, {sample['img'].max():.3f}]")
    print(f"GT shape: {sample['gt'].shape}, dtype: {sample['gt'].dtype}")
    print(f"GT unique values: {torch.unique(sample['gt'])}")
    print(f"Image size: {sample['height']} x {sample['width']}")


if __name__ == "__main__":
    __test_large_test_dataset()
