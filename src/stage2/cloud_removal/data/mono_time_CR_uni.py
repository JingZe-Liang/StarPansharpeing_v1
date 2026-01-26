from __future__ import annotations

from pathlib import Path
from typing import Literal

import litdata as ld
import torch

from src.data.litdata_hyperloader import IndexedCombinedStreamingDataset
from src.stage2.cloud_removal.data.CUHK_CR import CUHK_CR_StreamingDataset
from src.stage2.cloud_removal.data.SEN12_CR import SEN12_CR_StreamingDataset


type CuhkName = Literal["cr1", "cr2", None]
type CuhkSplit = Literal["train", "test", None]
type Sen12Split = Literal["train", "val", "test", None]
type CuhkDataset = CUHK_CR_StreamingDataset | IndexedCombinedStreamingDataset
type Sen12Dataset = SEN12_CR_StreamingDataset | IndexedCombinedStreamingDataset

_CUHK_NAMES = ("cr1", "cr2")
_CUHK_SPLITS = ("train", "test")
_SEN12_SPLITS = ("train", "val", "test")


def _as_posix(path: str | Path) -> str:
    return Path(path).as_posix()


def _build_combined_kwargs(combined_kwargs: dict | None) -> dict:
    combined_kwargs = dict(combined_kwargs or {})
    combined_kwargs.setdefault("batching_method", "per_stream")
    return combined_kwargs


def _merge_no_s1(sen12_kwargs: dict | None, no_s1: bool) -> dict:
    merged_kwargs = dict(sen12_kwargs or {})
    merged_kwargs.setdefault("no_s1", no_s1)
    return merged_kwargs


def _expand_cuhk_names(name: CuhkName) -> list[str]:
    if name is None:
        return list(_CUHK_NAMES)
    return [name]


def _expand_cuhk_splits(split: CuhkSplit) -> list[str]:
    if split is None:
        return list(_CUHK_SPLITS)
    return [split]


def _expand_sen12_splits(split: Sen12Split) -> list[str]:
    if split is None:
        return list(_SEN12_SPLITS)
    return [split]


def _combine_datasets(datasets: list, combined_kwargs: dict | None) -> IndexedCombinedStreamingDataset:
    combined_kwargs = _build_combined_kwargs(combined_kwargs)
    return IndexedCombinedStreamingDataset(datasets=datasets, **combined_kwargs)


def _cuhk_dir_name(name: str) -> str:
    mapping = {"cr1": "CUHK-CR1", "cr2": "CUHK-CR2"}
    return mapping[name]


def _build_cuhk_dataset(
    *,
    input_dir: str | Path,
    name: CuhkName,
    split: CuhkSplit,
    cuhk_kwargs: dict | None,
) -> CuhkDataset:
    cuhk_kwargs = dict(cuhk_kwargs or {})
    resolved_name = cuhk_kwargs.pop("name", name)
    resolved_split = cuhk_kwargs.pop("split", split)

    datasets: list[CUHK_CR_StreamingDataset] = []
    for name_item in _expand_cuhk_names(resolved_name):
        for split_item in _expand_cuhk_splits(resolved_split):
            stream_kwargs = dict(cuhk_kwargs)
            stream_kwargs["name"] = name_item
            stream_kwargs["split"] = split_item
            datasets.append(CUHK_CR_StreamingDataset.create_dataset(input_dir=_as_posix(input_dir), **stream_kwargs))

    if len(datasets) == 1:
        return datasets[0]
    return _combine_datasets(datasets, None)


def _build_sen12_dataset(
    *,
    input_dir: str | Path,
    split: Sen12Split,
    sen12_kwargs: dict | None,
) -> Sen12Dataset:
    sen12_kwargs = dict(sen12_kwargs or {})
    resolved_split = sen12_kwargs.pop("split", split)

    datasets: list[SEN12_CR_StreamingDataset] = []
    for split_item in _expand_sen12_splits(resolved_split):
        stream_kwargs = dict(sen12_kwargs)
        stream_kwargs["split"] = split_item
        datasets.append(SEN12_CR_StreamingDataset.create_dataset(input_dir=_as_posix(input_dir), **stream_kwargs))

    if len(datasets) == 1:
        return datasets[0]
    return _combine_datasets(datasets, None)


class MonoTimeCRStreamingDataset(IndexedCombinedStreamingDataset):
    def __init__(
        self,
        *,
        cuhk_input_dir: str | Path,
        sen12_input_dir: str | Path,
        cuhk_name: CuhkName = None,
        cuhk_split: CuhkSplit = None,
        sen12_split: Sen12Split = "train",
        no_s1: bool = False,
        cuhk_kwargs: dict | None = None,
        sen12_kwargs: dict | None = None,
        combined_kwargs: dict | None = None,
    ) -> None:
        cuhk_ds = _build_cuhk_dataset(
            input_dir=cuhk_input_dir,
            name=cuhk_name,
            split=cuhk_split,
            cuhk_kwargs=cuhk_kwargs,
        )
        sen12_ds = _build_sen12_dataset(
            input_dir=sen12_input_dir,
            split=sen12_split,
            sen12_kwargs=_merge_no_s1(sen12_kwargs, no_s1),
        )

        self.cuhk_ds = cuhk_ds
        self.sen12_ds = sen12_ds
        combined_kwargs = _build_combined_kwargs(combined_kwargs)
        super().__init__(datasets=[cuhk_ds, sen12_ds], **combined_kwargs)  # type: ignore[invalid-argument-type]

    @classmethod
    def create_dataset(
        cls,
        *,
        cuhk_input_dir: str | Path,
        sen12_input_dir: str | Path,
        cuhk_name: CuhkName = None,
        cuhk_split: CuhkSplit = None,
        sen12_split: Sen12Split = "train",
        no_s1: bool = False,
        cuhk_kwargs: dict | None = None,
        sen12_kwargs: dict | None = None,
        combined_kwargs: dict | None = None,
    ) -> "MonoTimeCRStreamingDataset":
        return cls(
            cuhk_input_dir=cuhk_input_dir,
            sen12_input_dir=sen12_input_dir,
            cuhk_name=cuhk_name,
            cuhk_split=cuhk_split,
            sen12_split=sen12_split,
            no_s1=no_s1,
            cuhk_kwargs=cuhk_kwargs,
            sen12_kwargs=sen12_kwargs,
            combined_kwargs=combined_kwargs,
        )

    @classmethod
    def create_dataloader(
        cls,
        *,
        cuhk_input_dir: str | Path,
        sen12_input_dir: str | Path,
        cuhk_name: CuhkName = None,
        cuhk_split: CuhkSplit = None,
        sen12_split: Sen12Split = "train",
        no_s1: bool = False,
        cuhk_kwargs: dict | None = None,
        sen12_kwargs: dict | None = None,
        combined_kwargs: dict | None = None,
        loader_kwargs: dict | None = None,
    ) -> tuple["MonoTimeCRStreamingDataset", ld.StreamingDataLoader]:
        ds = cls.create_dataset(
            cuhk_input_dir=cuhk_input_dir,
            sen12_input_dir=sen12_input_dir,
            cuhk_name=cuhk_name,
            cuhk_split=cuhk_split,
            sen12_split=sen12_split,
            no_s1=no_s1,
            cuhk_kwargs=cuhk_kwargs,
            sen12_kwargs=sen12_kwargs,
            combined_kwargs=combined_kwargs,
        )
        loader_kwargs = dict(loader_kwargs or {})
        dl = ld.StreamingDataLoader(ds, **loader_kwargs)
        return ds, dl


#  ---------------- Tests ------------------- #

import pytest


class _DummyDataset:
    def __init__(self, length: int = 1) -> None:
        self._length = length

    def __len__(self) -> int:
        return self._length


def _has_cuhk_litdata(input_dir: str | Path) -> bool:
    input_dir = Path(input_dir)
    for name in _CUHK_NAMES:
        for split in _CUHK_SPLITS:
            data_dir = input_dir / _cuhk_dir_name(name) / split
            if not data_dir.exists():
                return False
            if not (data_dir / "index.json").exists():
                return False
    return True


def _infer_dataset_name(batch: dict) -> str:
    if "s1" in batch:
        return "SEN12"

    conditions = batch.get("conditions")
    if isinstance(conditions, list):
        for item in conditions:
            if isinstance(item, torch.Tensor) and item.ndim >= 2 and item.shape[1] == 2:
                return "SEN12"

    img = batch.get("img")
    gt = batch.get("gt")
    if isinstance(img, torch.Tensor) and img.ndim >= 2 and img.shape[1] == 13:
        return "SEN12"
    if isinstance(gt, torch.Tensor) and gt.ndim >= 2 and gt.shape[1] == 13:
        return "SEN12"

    return "CUHK"


def _install_fake_creators(monkeypatch: pytest.MonkeyPatch) -> tuple[list[dict], list[dict]]:
    cuhk_calls: list[dict] = []
    sen12_calls: list[dict] = []

    def fake_cuhk_create_dataset(cls, *, input_dir: str, **kwargs) -> _DummyDataset:  # noqa: ARG001
        cuhk_calls.append({"input_dir": input_dir, **kwargs})
        return _DummyDataset()

    def fake_sen12_create_dataset(cls, *, input_dir: str, **kwargs) -> _DummyDataset:  # noqa: ARG001
        sen12_calls.append({"input_dir": input_dir, **kwargs})
        return _DummyDataset()

    monkeypatch.setattr(CUHK_CR_StreamingDataset, "create_dataset", classmethod(fake_cuhk_create_dataset))
    monkeypatch.setattr(SEN12_CR_StreamingDataset, "create_dataset", classmethod(fake_sen12_create_dataset))
    return cuhk_calls, sen12_calls


def test_merge_no_s1_respects_override() -> None:
    assert _merge_no_s1(None, True)["no_s1"] is True
    assert _merge_no_s1({"no_s1": False}, True)["no_s1"] is False


def test_build_cuhk_dataset_expands_all(monkeypatch: pytest.MonkeyPatch) -> None:
    cuhk_calls, _ = _install_fake_creators(monkeypatch)
    ds = _build_cuhk_dataset(input_dir="cuhk", name=None, split=None, cuhk_kwargs=None)

    assert isinstance(ds, IndexedCombinedStreamingDataset)
    assert len(cuhk_calls) == 4
    assert {call["name"] for call in cuhk_calls} == {"cr1", "cr2"}
    assert {call["split"] for call in cuhk_calls} == {"train", "test"}
    assert len(ds) == 4  # type: ignore[invalid-argument-type]


def test_build_sen12_dataset_expands_all(monkeypatch: pytest.MonkeyPatch) -> None:
    _, sen12_calls = _install_fake_creators(monkeypatch)
    ds = _build_sen12_dataset(input_dir="sen12", split=None, sen12_kwargs=None)

    assert isinstance(ds, IndexedCombinedStreamingDataset)
    assert len(sen12_calls) == 3
    assert {call["split"] for call in sen12_calls} == {"train", "val", "test"}
    assert len(ds) == 3  # type: ignore[invalid-argument-type]


def test_create_dataset_expands_all(monkeypatch: pytest.MonkeyPatch) -> None:
    cuhk_calls, sen12_calls = _install_fake_creators(monkeypatch)
    ds = MonoTimeCRStreamingDataset.create_dataset(
        cuhk_input_dir="cuhk",
        sen12_input_dir="sen12",
        cuhk_name=None,
        cuhk_split=None,
        sen12_split=None,
        no_s1=True,
    )

    assert isinstance(ds, MonoTimeCRStreamingDataset)
    assert len(cuhk_calls) == 4
    assert len(sen12_calls) == 3
    assert all(call["no_s1"] is True for call in sen12_calls)


def test_all_dataset_combined_and_plot_sen12_cuhk() -> None:
    from src.stage2.cloud_removal.data.SEN12_CR import _has_sen12_litdata
    from src.stage2.cloud_removal.data.sen12_vis import plot_sen12_triplet

    cuhk_root = Path("data/Downstreams/CUHK-CR/litdata_out")
    sen12_root = Path("data/SEN12MS-CR")
    tmp_path = Path("./")

    if not _has_cuhk_litdata(cuhk_root):
        pytest.skip("CUHK-CR litdata not found. Run CUHK-CR conversion first.")
    if not all(_has_sen12_litdata(sen12_root, split=split) for split in _SEN12_SPLITS):
        pytest.skip("SEN12MS-CR litdata not found for all splits. Run SEN12MS-CR conversion first.")

    ds = MonoTimeCRStreamingDataset.create_dataset(
        cuhk_input_dir=cuhk_root,
        sen12_input_dir=sen12_root,
        cuhk_name=None,
        cuhk_split=None,
        sen12_split=None,
        no_s1=False,
        cuhk_kwargs={"to_neg_1_1": False, "force_rgb": True},
        sen12_kwargs={"return_meta": False, "to_neg_1_1": False, "rescale_method": "default"},
    )

    cuhk_sample = ds.cuhk_ds[0]
    sen12_sample = ds.sen12_ds[0]

    cuhk_img = cuhk_sample["img"]
    cuhk_gt = cuhk_sample["gt"]
    if not isinstance(cuhk_img, torch.Tensor) or not isinstance(cuhk_gt, torch.Tensor):
        raise TypeError("Expected CUHK samples to be torch.Tensor.")

    cuhk_img = cuhk_img[:3].float()
    cuhk_gt = cuhk_gt[:3].float()
    if cuhk_img.min() < 0 or cuhk_img.max() > 1:
        cuhk_img = (cuhk_img - cuhk_img.min()) / (cuhk_img.max() - cuhk_img.min() + 1e-6)
    if cuhk_gt.min() < 0 or cuhk_gt.max() > 1:
        cuhk_gt = (cuhk_gt - cuhk_gt.min()) / (cuhk_gt.max() - cuhk_gt.min() + 1e-6)

    cuhk_img = cuhk_img.permute(1, 2, 0).cpu().numpy()
    cuhk_gt = cuhk_gt.permute(1, 2, 0).cpu().numpy()

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(cuhk_img)
    axes[0].set_title("CUHK Img")
    axes[1].imshow(cuhk_gt)
    axes[1].set_title("CUHK GT")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    cuhk_path = tmp_path / "cuhk_preview.png"
    print(f"Save at {cuhk_path}")
    fig.savefig(cuhk_path, dpi=150)
    plt.close(fig)

    s1 = sen12_sample["s1"]
    s2 = sen12_sample["gt"]
    s2_cloudy = sen12_sample["img"]
    if not isinstance(s1, torch.Tensor) or not isinstance(s2, torch.Tensor) or not isinstance(s2_cloudy, torch.Tensor):
        raise TypeError("Expected SEN12 samples to be torch.Tensor.")

    sen12_path = tmp_path / "sen12_preview.png"
    plot_sen12_triplet(s1=s1, s2=s2, s2_cloudy=s2_cloudy, save_path=sen12_path)
    print(f"Save at {sen12_path}")

    assert cuhk_path.exists()
    assert sen12_path.exists()


def test_all_dataloader_iter() -> None:
    from src.stage2.cloud_removal.data.SEN12_CR import _has_sen12_litdata

    cuhk_root = Path("data/Downstreams/CUHK-CR/litdata_out")
    sen12_root = Path("data/SEN12MS-CR")

    if not _has_cuhk_litdata(cuhk_root):
        pytest.skip("CUHK-CR litdata not found. Run CUHK-CR conversion first.")
    if not all(_has_sen12_litdata(sen12_root, split=split) for split in _SEN12_SPLITS):
        pytest.skip("SEN12MS-CR litdata not found for all splits. Run SEN12MS-CR conversion first.")

    _, dl = MonoTimeCRStreamingDataset.create_dataloader(
        cuhk_input_dir=cuhk_root,
        sen12_input_dir=sen12_root,
        cuhk_name=None,
        cuhk_split=None,
        sen12_split=None,
        no_s1=False,
        cuhk_kwargs={"to_neg_1_1": False, "force_rgb": True},
        sen12_kwargs={"return_meta": False, "to_neg_1_1": False, "rescale_method": "default"},
        loader_kwargs={"batch_size": 2, "num_workers": 0},
    )

    for batch_idx, batch in enumerate(dl):
        if not isinstance(batch, dict):
            raise TypeError("Expected batch to be a dict.")
        dataset_name = _infer_dataset_name(batch)
        print(f"batch {batch_idx} dataset_name: {dataset_name}")
        for key in ("img", "gt", "s1"):
            value = batch.get(key)
            if isinstance(value, torch.Tensor):
                print(f"{key} shape: {tuple(value.shape)}")

        conditions = batch.get("conditions")
        if isinstance(conditions, list):
            cond_shapes = [tuple(item.shape) for item in conditions if isinstance(item, torch.Tensor)]
            print(f"conditions shapes: {cond_shapes}")
        elif isinstance(conditions, torch.Tensor):
            print(f"conditions shape: {tuple(conditions.shape)}")

        if batch_idx >= 1:
            break


def test_speed():
    from src.stage2.cloud_removal.data.SEN12_CR import _has_sen12_litdata

    cuhk_root = Path("data/Downstreams/CUHK-CR/litdata_out")
    sen12_root = Path("data/SEN12MS-CR")

    if not _has_cuhk_litdata(cuhk_root):
        pytest.skip("CUHK-CR litdata not found. Run CUHK-CR conversion first.")
    if not all(_has_sen12_litdata(sen12_root, split=split) for split in _SEN12_SPLITS):
        pytest.skip("SEN12MS-CR litdata not found for all splits. Run SEN12MS-CR conversion first.")

    _, dl = MonoTimeCRStreamingDataset.create_dataloader(
        cuhk_input_dir=cuhk_root,
        sen12_input_dir=sen12_root,
        cuhk_name=None,
        cuhk_split=None,
        sen12_split=None,
        no_s1=False,
        cuhk_kwargs={"to_neg_1_1": False, "force_rgb": True},
        sen12_kwargs={"return_meta": False, "to_neg_1_1": False, "rescale_method": "default"},
        loader_kwargs={"batch_size": 2, "num_workers": 0},
    )

    from tqdm import tqdm

    for sample in (tbar := tqdm(dl)):
        ds_name = sample["dataset_name"]
        tbar.set_description(f"dataset_name: {ds_name}")
