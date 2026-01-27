from collections.abc import Callable
from typing import Any

from torch.utils.data import DataLoader, Dataset


def create_dataloader(
    dataset: Dataset[Any],
    loader_kwargs: dict[str, Any],
    collate_fn: Callable | None = None,
) -> tuple[Dataset[Any], DataLoader[Any]]:
    if collate_fn is not None:
        loader_kwargs = {**loader_kwargs, "collate_fn": collate_fn}
    dataloader = DataLoader(dataset, **loader_kwargs)
    return dataset, dataloader
