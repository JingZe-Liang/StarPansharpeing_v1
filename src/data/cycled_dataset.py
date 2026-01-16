from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch


def _samples_in_cycle(num_samples_yielded: int, cycle_length: int) -> int:
    if num_samples_yielded <= 0:
        return 0
    if cycle_length <= 0:
        raise ValueError(f"`cycle_length` must be positive, got {cycle_length}.")
    return ((num_samples_yielded - 1) % cycle_length) + 1


class CycledDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset: Any) -> None:
        super().__init__()
        self._dataset = dataset
        self.current_epoch = 0

    def __len__(self) -> int:
        return int(len(self._dataset))

    def __iter__(self) -> Iterator[Any]:
        while True:
            iterator = iter(self._dataset)
            while True:
                try:
                    yield next(iterator)
                except StopIteration:
                    break

    def set_epoch(self, current_epoch: int) -> None:
        self.current_epoch = current_epoch
        if hasattr(self._dataset, "set_epoch"):
            self._dataset.set_epoch(current_epoch)

    def set_shuffle(self, shuffle: bool) -> None:
        if hasattr(self._dataset, "set_shuffle"):
            self._dataset.set_shuffle(shuffle)

    def set_batch_size(self, batch_size: int) -> None:
        if hasattr(self._dataset, "set_batch_size"):
            self._dataset.set_batch_size(batch_size)

    def set_num_workers(self, num_workers: int) -> None:
        if hasattr(self._dataset, "set_num_workers"):
            self._dataset.set_num_workers(num_workers)

    def set_drop_last(self, drop_last: bool) -> None:
        if hasattr(self._dataset, "set_drop_last"):
            self._dataset.set_drop_last(drop_last)

    def reset_state_dict(self) -> None:
        if hasattr(self._dataset, "reset_state_dict"):
            self._dataset.reset_state_dict()

    def state_dict(self, num_samples_yielded: int, num_workers: int, batch_size: int) -> dict[str, Any]:
        cycle_length = len(self)
        in_cycle = _samples_in_cycle(num_samples_yielded=num_samples_yielded, cycle_length=cycle_length)

        dataset_state = {}
        if hasattr(self._dataset, "state_dict"):
            dataset_state = self._dataset.state_dict(
                num_samples_yielded=in_cycle,
                num_workers=num_workers,
                batch_size=batch_size,
            )

        return {
            "dataset": dataset_state,
            "num_samples_yielded": num_samples_yielded,
            "num_samples_yielded_in_cycle": in_cycle,
            "cycle_length": cycle_length,
            "current_epoch": self.current_epoch,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not state_dict:
            return
        if "current_epoch" in state_dict:
            self.current_epoch = int(state_dict["current_epoch"])
        if hasattr(self._dataset, "load_state_dict") and "dataset" in state_dict:
            self._dataset.load_state_dict(state_dict["dataset"])
