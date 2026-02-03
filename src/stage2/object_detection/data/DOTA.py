from collections.abc import Callable
from typing import Any, Literal

from torch.utils.data import DataLoader
from torchgeo.datasets import DOTA


def detection_collate(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return batch


def get_dataloader(
    batch_size: int,
    num_workers: int,
    split: Literal["train", "val"] = "train",
    root: str = "data/Downstreams/DOTA",
    version: Literal["1.0", "1.5", "2.0"] = "2.0",
    bbox_orientation: Literal["horizontal", "oriented"] = "oriented",
    transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    download: bool = False,
    checksum: bool = False,
    collate_fn: Callable[[list[dict[str, Any]]], Any] | None = None,
    **loader_kwargs: Any,
) -> tuple[DOTA, DataLoader]:
    """Get DOTA dataloader for object detection.

    Args:
        batch_size: Batch size for training.
        num_workers: Number of worker processes for data loading.
        split: One of 'train' or 'val'.
        root: Root directory where dataset is stored.
        version: One of '1.0', '1.5', or '2.0'.
        bbox_orientation: One of 'horizontal' or 'oriented'.
        transforms: Optional transform function applied to samples.
        download: If True, download dataset if not found.
        checksum: If True, verify dataset checksums.
        collate_fn: Optional collate function for DataLoader.
        **loader_kwargs: Additional kwargs passed to DataLoader.

    Returns:
        Tuple of (dataset, dataloader).

    Note:
        Sample keys:
        - image: Tensor of shape (3, H, W)
        - bbox: Tensor of shape (N, 8) for oriented boxes
        - bbox_xyxy: Tensor of shape (N, 4) for horizontal boxes
        - labels: Tensor of shape (N,)
    """
    dataset = DOTA(
        root=root,
        split=split,
        version=version,
        bbox_orientation=bbox_orientation,
        transforms=transforms,
        download=download,
        checksum=checksum,
    )

    if collate_fn is None:
        collate_fn = detection_collate

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=(split == "train"),
        collate_fn=collate_fn,
        **loader_kwargs,
    )

    return dataset, dataloader
