from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch

from src.data.cycled_dataset import CycledDataset, _samples_in_cycle


class _DummyFiniteIterable(torch.utils.data.IterableDataset):
    def __init__(self, length: int) -> None:
        super().__init__()
        self._length = length
        self.last_state_dict_call: tuple[int, int, int] | None = None
        self.loaded_state_dict: dict[str, Any] | None = None

    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> Iterator[int]:
        yield from range(self._length)

    def state_dict(self, num_samples_yielded: int, num_workers: int, batch_size: int) -> dict[str, Any]:
        self.last_state_dict_call = (num_samples_yielded, num_workers, batch_size)
        return {"num_samples_yielded": num_samples_yielded}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.loaded_state_dict = state_dict


def test_samples_in_cycle_boundary() -> None:
    assert _samples_in_cycle(0, 5) == 0
    assert _samples_in_cycle(1, 5) == 1
    assert _samples_in_cycle(5, 5) == 5
    assert _samples_in_cycle(6, 5) == 1


def test_cycled_dataset_cycles_forever() -> None:
    ds = _DummyFiniteIterable(3)
    cyc = CycledDataset(ds)
    it = iter(cyc)
    assert [next(it) for _ in range(5)] == [0, 1, 2, 0, 1]


def test_cycled_dataset_state_dict_uses_in_cycle() -> None:
    ds = _DummyFiniteIterable(5)
    cyc = CycledDataset(ds)

    sd = cyc.state_dict(num_samples_yielded=6, num_workers=2, batch_size=3)
    assert ds.last_state_dict_call == (1, 2, 3)
    assert sd["num_samples_yielded_in_cycle"] == 1


def test_cycled_dataset_load_state_dict_forwards() -> None:
    ds = _DummyFiniteIterable(5)
    cyc = CycledDataset(ds)

    cyc.load_state_dict({"dataset": {"hello": "world"}, "current_epoch": 7})
    assert ds.loaded_state_dict == {"hello": "world"}
    assert cyc.current_epoch == 7


def test_cycled_dataset_empty_iter_yields_none() -> None:
    ds = _DummyFiniteIterable(0)
    cyc = CycledDataset(ds)
    it = iter(cyc)
    assert next(it) is None
