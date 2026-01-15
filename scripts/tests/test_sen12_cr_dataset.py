from __future__ import annotations

from pathlib import Path

import pytest

from src.stage2.cloud_removal.data.SEN12_CR import SEN12_CR_StreamingDataset

DATA_ROOT = Path("data/SEN12MS-CR")


def _has_sen12_litdata() -> bool:
    required = [
        DATA_ROOT / "litdata_s1/train",
        DATA_ROOT / "litdata_s2_clean/train",
        DATA_ROOT / "litdata_s2_cloudy/train",
        DATA_ROOT / "litdata_meta/train.csv",
    ]
    return all(path.exists() for path in required)


@pytest.mark.skipif(not _has_sen12_litdata(), reason="SEN12MS-CR litdata not found.")
def test_length_and_meta_alignment():
    ds = SEN12_CR_StreamingDataset.create_dataset(
        input_dir=DATA_ROOT.as_posix(),
        split="train",
        return_meta=True,
    )
    assert len(ds) > 0
    sample = ds[0]
    meta = sample["meta"]
    assert meta.index == 0
    assert meta.split == "train"


@pytest.mark.skipif(not _has_sen12_litdata(), reason="SEN12MS-CR litdata not found.")
def test_shapes_and_ranges():
    ds = SEN12_CR_StreamingDataset.create_dataset(
        input_dir=DATA_ROOT.as_posix(),
        split="train",
        return_meta=False,
        to_neg_1_1=False,
        rescale_method="default",
    )
    sample = ds[0]
    s1 = sample["s1"]
    s2 = sample["s2"]
    s2_cloudy = sample["s2_cloudy"]
    assert s2.shape == s2_cloudy.shape
    assert s1.shape[1:] == s2.shape[1:]
    assert s1.shape[0] == 2
    assert s2.shape[0] == 13
    assert s1.min() >= 0 and s1.max() <= 1
    assert s2.min() >= 0 and s2.max() <= 1
    assert s2_cloudy.min() >= 0 and s2_cloudy.max() <= 1


@pytest.mark.skipif(not _has_sen12_litdata(), reason="SEN12MS-CR litdata not found.")
def test_sar_log1p_range():
    ds = SEN12_CR_StreamingDataset.create_dataset(
        input_dir=DATA_ROOT.as_posix(),
        split="train",
        return_meta=False,
        to_neg_1_1=False,
        rescale_method="default",
        sar_log1p=True,
    )
    sample = ds[0]
    s1 = sample["s1"]
    assert s1.min() >= 0 and s1.max() <= 1
