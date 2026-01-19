from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from src.data.litdata_hyperloader import IndexedCombinedStreamingDataset, SingleCycleStreamingDataset


class DummyLeafDataset:
    def __init__(self, *, name: str, length: int) -> None:
        self.name = name
        self.length = length
        self.loaded: dict[str, Any] | None = None

    def __len__(self) -> int:
        return int(self.length)

    def __iter__(self) -> Iterator[int]:
        yield from range(self.length)

    def state_dict(self, num_samples_yielded: int, num_workers: int, batch_size: int) -> dict[str, Any]:
        return {
            "name": self.name,
            "num_samples_yielded": int(num_samples_yielded),
            "num_workers": int(num_workers),
            "batch_size": int(batch_size),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.loaded = state_dict


def test_indexed_combined_state_dict_accepts_int_num_samples_yielded() -> None:
    ds_a = DummyLeafDataset(name="a", length=10)
    ds_b = DummyLeafDataset(name="b", length=30)

    inner = IndexedCombinedStreamingDataset(
        datasets=[ds_a, ds_b],
        iterate_over_all=True,
        batching_method="per_stream",
        seed=2025,
    )

    state = inner.state_dict(num_workers=2, batch_size=4, num_samples_yielded=20)
    assert set(state.keys()) == {"0", "1"}
    assert state["0"]["num_samples_yielded"] == 5
    assert state["1"]["num_samples_yielded"] == 15

    inner.load_state_dict(state)
    assert ds_a.loaded == state["0"]
    assert ds_b.loaded == state["1"]


def test_nested_combined_state_dict_does_not_crash() -> None:
    ds_a = DummyLeafDataset(name="a", length=10)
    ds_b = DummyLeafDataset(name="b", length=30)

    inner = IndexedCombinedStreamingDataset(
        datasets=[ds_a, ds_b],
        iterate_over_all=True,
        batching_method="per_stream",
        seed=2025,
    )
    outer = IndexedCombinedStreamingDataset(
        combined_is_cycled=True,
        datasets=[inner],
        weights=[1.0],
        iterate_over_all=False,
        batching_method="per_stream",
        seed=2025,
    )

    state = outer.state_dict(num_workers=2, batch_size=4, num_samples_yielded=[20])
    assert set(state.keys()) == {"0"}
    assert set(state["0"].keys()) == {"0", "1"}

    cycle = SingleCycleStreamingDataset(dataset=inner)
    cycle_state = cycle.state_dict(num_workers=2, batch_size=4, num_samples_yielded=20)
    assert set(cycle_state.keys()) == {"0"}

    cycle.load_state_dict({"dataset": cycle_state})
    assert ds_a.loaded == cycle_state["0"]["0"]
    assert ds_b.loaded == cycle_state["0"]["1"]
