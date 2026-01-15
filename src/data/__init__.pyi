"""
Type stubs for src.data module.
Provides type hints for lazy-loaded functions and classes.
"""

from collections.abc import Iterator
from typing import Any, Callable, Generator, Iterable, Literal, Sequence, Union

# Forward declarations for type hints
class WindowSlider:
    """A sliding window generator that extracts patches from full images."""
    def __init__(
        self,
        slide_keys: list[str],
        window_size: int = 64,
        stride: int | None = None,
        overlap: float | None = None,
    ) -> None: ...
    def __call__(
        self, sample: dict[str, Any]
    ) -> Generator[dict[str, Any], None, None]: ...

class _BaseStreamingDataset:
    """Base streaming dataset class."""

    ...

class CombinedStreamingDataset:
    """Combined streaming dataset class."""

    ...

class ConditionsStreamingDataset(_BaseStreamingDataset):
    """Conditions streaming dataset class."""

    ...

class GenerativeStreamingDataset:
    """Generative streaming dataset class."""

    ...

class ImageStreamingDataset(_BaseStreamingDataset):
    """Image streaming dataset class."""

    ...

class IndexedCombinedStreamingDataset(CombinedStreamingDataset):
    """Indexed combined streaming dataset class."""

    ...

class SingleCycleStreamingDataset:
    """Single cycle streaming dataset class."""

    ...

class _TimedLoaderIterator:
    """Timed loader iterator for performance tracking."""

    def __init__(
        self,
        dataloader: Iterable[Any],
        recorder: Any | None = None,
        name: str = 'loader_iter'
    ) -> None: ...

    def __iter__(self) -> "_TimedLoaderIterator": ...

    def __next__(self) -> Any: ...

class SizeBasedBatchsizeStreamingDataloader:
    """Size-based batch size streaming dataloader."""

    ...

class MultimodalityDataloader:
    """Multimodality dataloader class."""

    ...

# Function type hints
def get_fast_test_hyperspectral_data(
    root_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> tuple[Any, Any]: ...
def get_hyperspectral_dataloaders(
    root_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
    train_transform: Callable[[Any], Any] | None = None,
    val_transform: Callable[[Any], Any] | None = None,
    test_transform: Callable[[Any], Any] | None = None,
) -> tuple[Any, Any, Any]: ...
def ms_pan_dir_paired_loader(
    ms_dir: str,
    pan_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
    transform: Callable[[Any], Any] | None = None,
) -> Any: ...
def get_mm_chained_loaders(
    root_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> tuple[Any, Any]: ...
def create_windowed_dataloader(
    dataloader: Any,
    window_size: int,
    stride: int,
    slide_keys: list[str],
    patch_key: str = "patch",
    coord_key: str = "coord",
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Any: ...

# Export all items
__all__: list[str] = [
    "get_fast_test_hyperspectral_data",
    "get_hyperspectral_dataloaders",
    "ms_pan_dir_paired_loader",
    "CombinedStreamingDataset",
    "ConditionsStreamingDataset",
    "GenerativeStreamingDataset",
    "ImageStreamingDataset",
    "IndexedCombinedStreamingDataset",
    "SingleCycleStreamingDataset",
    "_BaseStreamingDataset",
    "_TimedLoaderIterator",
    "SizeBasedBatchsizeStreamingDataloader",
    "MultimodalityDataloader",
    "get_mm_chained_loaders",
    "WindowSlider",
    "create_windowed_dataloader",
]
