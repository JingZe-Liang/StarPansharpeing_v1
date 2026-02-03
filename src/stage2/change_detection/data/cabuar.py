from torchgeo.datasets import CaBuAr
from torch.utils.data import DataLoader
from torch import Tensor
from collections.abc import Callable
from typing import Any


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
