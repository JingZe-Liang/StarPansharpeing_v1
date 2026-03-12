from collections import defaultdict
from types import SimpleNamespace

import torch
from litdata import StreamingDataLoader

from src.data.litdata_hyperloader import (
    SizeBasedBatchsizeStreamingDataloader,
    _SizedBatchCacheBucket,
)


def _make_batch(batch_size: int, *, start: int = 0, channel: int = 3, size: int = 5) -> dict[str, object]:
    total_values = batch_size * channel * size * size
    img = torch.arange(start, start + total_values, dtype=torch.float32).reshape(batch_size, channel, size, size)
    label = torch.arange(start, start + batch_size)
    keys = [f"sample-{idx}" for idx in range(start, start + batch_size)]
    return {"img": img, "label": label, "__key__": keys}


def _build_loader(
    source_batches: list[dict[str, object]],
    *,
    target_bs: int,
    drop_last: bool,
    cache_minor: bool = True,
) -> SizeBasedBatchsizeStreamingDataloader:
    loader = SizeBasedBatchsizeStreamingDataloader.__new__(SizeBasedBatchsizeStreamingDataloader)
    object.__setattr__(loader, "size_bs_s", {5: target_bs})
    object.__setattr__(loader, "cache_minor", cache_minor)
    object.__setattr__(loader, "dataset", SimpleNamespace(drop_last=drop_last))
    object.__setattr__(loader, "_source_batches", source_batches)

    if cache_minor:
        object.__setattr__(loader, "_size_caches", defaultdict(_SizedBatchCacheBucket))

    return loader


def _streaming_iter(self: SizeBasedBatchsizeStreamingDataloader):
    yield from self._source_batches


def test_size_based_loader_flushes_when_current_batch_completes_cache(monkeypatch) -> None:
    monkeypatch.setattr(StreamingDataLoader, "__iter__", _streaming_iter)

    loader = _build_loader(
        [_make_batch(6, start=0), _make_batch(2, start=6)],
        target_bs=8,
        drop_last=True,
    )

    batches = list(iter(loader))

    assert [batch["img"].shape[0] for batch in batches] == [8]
    assert batches[0]["label"].tolist() == list(range(8))
    assert batches[0]["__key__"] == [f"sample-{idx}" for idx in range(8)]


def test_size_based_loader_reuses_large_batch_remainder_from_cache(monkeypatch) -> None:
    monkeypatch.setattr(StreamingDataLoader, "__iter__", _streaming_iter)

    loader = _build_loader(
        [_make_batch(10, start=0), _make_batch(6, start=10)],
        target_bs=8,
        drop_last=True,
    )

    batches = list(iter(loader))

    assert [batch["img"].shape[0] for batch in batches] == [8, 8]
    assert batches[0]["label"].tolist() == list(range(8))
    assert batches[1]["label"].tolist() == list(range(8, 16))


def test_size_based_loader_drops_cached_tail_when_drop_last_enabled(monkeypatch) -> None:
    monkeypatch.setattr(StreamingDataLoader, "__iter__", _streaming_iter)

    loader = _build_loader(
        [_make_batch(6, start=0), _make_batch(6, start=6)],
        target_bs=8,
        drop_last=True,
    )

    batches = list(iter(loader))

    assert [batch["img"].shape[0] for batch in batches] == [8]
    assert batches[0]["label"].tolist() == list(range(8))


def test_size_based_loader_yields_cached_tail_when_drop_last_disabled(monkeypatch) -> None:
    monkeypatch.setattr(StreamingDataLoader, "__iter__", _streaming_iter)

    loader = _build_loader(
        [_make_batch(6, start=0), _make_batch(6, start=6)],
        target_bs=8,
        drop_last=False,
    )

    batches = list(iter(loader))

    assert [batch["img"].shape[0] for batch in batches] == [8, 4]
    assert batches[1]["label"].tolist() == list(range(8, 12))
