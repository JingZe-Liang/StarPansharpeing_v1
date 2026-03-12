from src.data.litdata_hyperloader import OverfitBatchStreamingLoader


def test_overfit_loader_cycles_cached_batches() -> None:
    base_loader = [{"step": 0}, {"step": 1}, {"step": 2}]
    loader = OverfitBatchStreamingLoader(base_loader, overfit_n_batches=2)

    loader_iter = iter(loader)
    observed = [next(loader_iter)["step"] for _ in range(5)]

    assert observed == [0, 1, 0, 1, 0]
    assert len(loader) == 2


def test_overfit_loader_uses_available_batches_when_source_is_shorter() -> None:
    base_loader = [{"step": 3}]
    loader = OverfitBatchStreamingLoader(base_loader, overfit_n_batches=4)

    loader_iter = iter(loader)
    observed = [next(loader_iter)["step"] for _ in range(3)]

    assert observed == [3, 3, 3]
