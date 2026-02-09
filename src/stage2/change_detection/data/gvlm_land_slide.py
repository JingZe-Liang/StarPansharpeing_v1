from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from kornia.augmentation import AugmentationSequential, RandomHorizontalFlip, RandomVerticalFlip
from kornia.constants import DataKey
from PIL import Image
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class GVLMSamplePaths:
    img1: Path
    img2: Path
    label: Path
    name: str


def get_default_transform(prob: float = 0.5) -> AugmentationSequential:
    return AugmentationSequential(
        RandomHorizontalFlip(p=prob),
        RandomVerticalFlip(p=prob),
        data_keys=[DataKey.INPUT, DataKey.INPUT, DataKey.MASK],
        same_on_batch=True,
        keepdim=True,
    )


def read_list(list_path: Path) -> list[str]:
    if not list_path.exists():
        raise FileNotFoundError(f"List file not found: {list_path}")
    names = [line.strip() for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise ValueError(f"Empty list file: {list_path}")
    return names


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8, copy=True)


def load_label(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.uint8, copy=True)


def build_sample_paths(root: Path, list_dir: str, split: str) -> list[GVLMSamplePaths]:
    list_path = root / list_dir / f"{split}.txt"
    names = read_list(list_path)
    a_dir = root / "A"
    b_dir = root / "B"
    label_dir = root / "label"
    samples: list[GVLMSamplePaths] = []
    for name in names:
        img1 = a_dir / name
        img2 = b_dir / name
        label = label_dir / name
        if not img1.exists() or not img2.exists() or not label.exists():
            raise FileNotFoundError(f"Missing file for sample {name}: {img1}, {img2}, {label}")
        samples.append(GVLMSamplePaths(img1=img1, img2=img2, label=label, name=name))
    return samples


class GVLMLandslideDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        split: str,
        list_dir: str = "list",
        transform: AugmentationSequential | None | str = "default",
        normalize: bool = True,
        img_to_neg1_1: bool = False,
        label_threshold: int = 0,
        return_name: bool = False,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.list_dir = list_dir
        self.samples = build_sample_paths(self.data_root, list_dir, split)
        if transform == "default":
            transform = get_default_transform()
        self.transform = transform
        self.normalize = normalize
        self.img_to_neg1_1 = img_to_neg1_1
        self.label_threshold = label_threshold
        self.return_name = return_name

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        img1 = load_rgb(sample.img1)
        img2 = load_rgb(sample.img2)
        label = load_label(sample.label)

        if self.label_threshold >= 0:
            label = (label > self.label_threshold).astype(np.uint8)

        img1_t = torch.from_numpy(img1).float()
        img2_t = torch.from_numpy(img2).float()
        if self.normalize:
            img1_t = img1_t / 255.0
            img2_t = img2_t / 255.0
        if self.img_to_neg1_1:
            img1_t = img1_t * 2.0 - 1.0
            img2_t = img2_t * 2.0 - 1.0
        img1_t = img1_t.permute(2, 0, 1)
        img2_t = img2_t.permute(2, 0, 1)
        label_t = torch.from_numpy(label).float().unsqueeze(0)

        if self.transform is not None:
            img1_t, img2_t, label_t = self.transform(
                img1_t.unsqueeze(0),
                img2_t.unsqueeze(0),
                label_t.unsqueeze(0),
            )
            img1_t = img1_t.squeeze(0)
            img2_t = img2_t.squeeze(0)
            label_t = label_t.squeeze(0)

        label_t = label_t.long()
        out = {"img1": img1_t, "img2": img2_t, "gt": label_t}
        if self.return_name:
            out["name"] = sample.name
        return out


def create_gvlm_landslide_dataloader(
    data_root: str | Path,
    split: str,
    list_dir: str = "list",
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 4,
    transform: AugmentationSequential | None | str = "default",
    normalize: bool = True,
    img_to_neg1_1: bool = False,
    label_threshold: int = 0,
    return_name: bool = False,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> tuple[GVLMLandslideDataset, DataLoader]:
    dataset = GVLMLandslideDataset(
        data_root=data_root,
        split=split,
        list_dir=list_dir,
        transform=transform,
        normalize=normalize,
        img_to_neg1_1=img_to_neg1_1,
        label_threshold=label_threshold,
        return_name=return_name,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return dataset, dataloader


def test_dataloader():
    """Simple example and test for GVLMLandslideDataset.

    Usage:
        python src/stage2/change_detection/data/gvlm_land_slide.py
    """
    data_root = Path("data/Downstreams/滑坡检测-GVLM/GVLM_CD256_0.3neg")
    if not data_root.exists():
        print(f"Data root not found: {data_root}")
        return

    # Example 1: Basic usage
    print("=" * 50)
    print("Example 1: Basic dataset usage")
    print("=" * 50)
    ds = GVLMLandslideDataset(data_root, split="train", transform=None)
    print(f"Dataset length: {len(ds)}")
    sample = ds[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"img1 shape: {sample['img1'].shape}, dtype: {sample['img1'].dtype}")
    print(f"img2 shape: {sample['img2'].shape}, dtype: {sample['img2'].dtype}")
    print(f"gt shape: {sample['gt'].shape}, dtype: {sample['gt'].dtype}")
    print(f"img1 range: [{sample['img1'].min():.3f}, {sample['img1'].max():.3f}]")

    # Example 2: With data augmentation
    print("\n" + "=" * 50)
    print("Example 2: With default transform (augmentation)")
    print("=" * 50)
    ds_aug = GVLMLandslideDataset(data_root, split="train", transform="default")
    sample_aug = ds_aug[0]
    print(f"img1 shape after aug: {sample_aug['img1'].shape}")

    # Example 3: Return sample name
    print("\n" + "=" * 50)
    print("Example 3: Return sample name")
    print("=" * 50)
    ds_name = GVLMLandslideDataset(data_root, split="train", transform=None, return_name=True)
    sample_name = ds_name[0]
    print(f"Sample name: {sample_name['name']}")

    # Example 4: Without normalization
    print("\n" + "=" * 50)
    print("Example 4: Without normalization")
    print("=" * 50)
    ds_unnorm = GVLMLandslideDataset(data_root, split="train", transform=None, normalize=False)
    sample_unnorm = ds_unnorm[0]
    print(f"img1 range (unnormalized): [{sample_unnorm['img1'].min():.1f}, {sample_unnorm['img1'].max():.1f}]")

    # Example 5: With img_to_neg1_1
    print("\n" + "=" * 50)
    print("Example 5: With img_to_neg1_1 (range [-1, 1])")
    print("=" * 50)
    ds_neg1 = GVLMLandslideDataset(data_root, split="train", transform=None, normalize=True, img_to_neg1_1=True)
    sample_neg1 = ds_neg1[0]
    print(f"img1 range (neg1_1): [{sample_neg1['img1'].min():.3f}, {sample_neg1['img1'].max():.3f}]")

    # Example 6: With DataLoader
    print("\n" + "=" * 50)
    print("Example 6: With DataLoader")
    print("=" * 50)
    from torch.utils.data import DataLoader

    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    print(f"Batch img1 shape: {batch['img1'].shape}")
    print(f"Batch img2 shape: {batch['img2'].shape}")
    print(f"Batch gt shape: {batch['gt'].shape}")

    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    test_dataloader()
