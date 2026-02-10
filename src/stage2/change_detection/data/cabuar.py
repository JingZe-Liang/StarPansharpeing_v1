from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchgeo.datasets import CaBuAr


def _normalize_cabuar_image(image: Tensor, to_neg_1_1: bool) -> Tensor:
    img_min = image.amin(dim=(0, 2, 3), keepdim=True)
    img_max = image.amax(dim=(0, 2, 3), keepdim=True)
    img_range = (img_max - img_min).clamp(min=1e-6)
    image = (image - img_min) / img_range
    image = image.clamp(0.0, 1.0)
    if to_neg_1_1:
        image = image * 2.0 - 1.0
    return image


class CaBuArChangeDetectionDataset(Dataset):
    def __init__(
        self,
        root: str,
        mode: str,
        bands: tuple[str, ...] | None = None,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        normalize: bool = True,
        to_neg_1_1: bool = False,
        return_name: bool = False,
    ) -> None:
        if bands is None:
            bands = CaBuAr.all_bands
        self.dataset = CaBuAr(
            root=root,
            split=mode,
            bands=bands,
            transforms=transforms,
            download=download,
        )
        self.normalize = normalize
        self.to_neg_1_1 = to_neg_1_1
        self.return_name = return_name

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        sample = self.dataset[index]
        image = sample["image"].float()
        mask = sample["mask"].long()
        if self.normalize:
            image = _normalize_cabuar_image(image, to_neg_1_1=self.to_neg_1_1)
        img1 = image[0]
        img2 = image[1]
        out: dict[str, Tensor | str] = {"img1": img1, "img2": img2, "gt": mask}
        if self.return_name:
            out["name"] = self.dataset.uuids[index]
        return out


def get_dataloader(
    batch_size: int,
    num_workers: int,
    mode: str = "train",
    root: str = "data/Downstreams/FireBurn-CaBuAr",
    bands: tuple[str, ...] | None = None,
    transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
    download: bool = False,
    **loader_kwargs: Any,
) -> tuple[CaBuAr, DataLoader]:
    """Get CaBuAr dataloader for wildfire segmentation.

    Args:
        batch_size: Batch size for training.
        num_workers: Number of worker processes for data loading.
        mode: One of 'train', 'val', 'test'.
        root: Root directory where dataset is stored.
        bands: Subset of Sentinel-2 bands to load. Defaults to all 12 bands.
        transforms: Optional transform function applied to samples.
        download: If True, download dataset if not found.
        **loader_kwargs: Additional kwargs passed to DataLoader.

    Returns:
        Tuple of (dataset, dataloader).

    Note:
        The dataset returns samples with:
        - image: Tensor of shape (2, C, H, W) - pre/post fire images
        - mask: Tensor of shape (1, H, W) - binary burned area mask (0/1)
    """
    if bands is None:
        bands = CaBuAr.all_bands

    dataset = CaBuAr(
        root=root,
        split=mode,
        bands=bands,
        transforms=transforms,
        download=download,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=(mode == "train"),
        **loader_kwargs,
    )

    return dataset, dataloader


def create_cabuar_change_detection_dataloader(
    batch_size: int,
    num_workers: int,
    mode: str = "train",
    root: str = "data/Downstreams/FireBurn-CaBuAr",
    bands: tuple[str, ...] | None = None,
    transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
    download: bool = False,
    normalize: bool = True,
    to_neg_1_1: bool = False,
    return_name: bool = False,
    pin_memory: bool = True,
    drop_last: bool = False,
    **loader_kwargs: Any,
) -> tuple[CaBuArChangeDetectionDataset, DataLoader]:
    dataset = CaBuArChangeDetectionDataset(
        root=root,
        mode=mode,
        bands=bands,
        transforms=transforms,
        download=download,
        normalize=normalize,
        to_neg_1_1=to_neg_1_1,
        return_name=return_name,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=(mode == "train") or loader_kwargs.pop("shuffle", False),
        pin_memory=pin_memory,
        drop_last=drop_last,
        **loader_kwargs,
    )
    return dataset, dataloader


def _test_dataloader():
    """Test CaBuAr dataloader."""
    import matplotlib.pyplot as plt

    print("Testing CaBuAr dataloader...")

    # Test dataset loading
    dataset, dataloader = get_dataloader(
        batch_size=2,
        num_workers=0,
        mode="train",
        root="data/Downstreams/FireBurn-CaBuAr",
        download=False,
    )

    print(f"Dataset length: {len(dataset)}")

    # Test single sample
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"Image shape: {sample['image'].shape}")  # (2, C, H, W)
    print(f"Image dtype: {sample['image'].dtype}")
    print(f"Image range: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
    print(f"Mask shape: {sample['mask'].shape}")  # (1, H, W)
    print(f"Mask unique values: {sample['mask'].unique().tolist()}")

    # Test dataloader
    batch = next(iter(dataloader))
    print(f"\nBatch image shape: {batch['image'].shape}")
    print(f"Batch mask shape: {batch['mask'].shape}")

    # Visualize
    fig = dataset.plot(sample)
    fig.savefig("cabuar_sample.webp", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved visualization to cabuar_sample.webp")


if __name__ == "__main__":
    _test_dataloader()
