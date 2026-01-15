import torch
import numpy as np
from pathlib import Path
from typing import Literal

from src.data import _BaseStreamingDataset


class CUHK_CR_StreamingDataset(_BaseStreamingDataset):
    def __init__(
        self,
        input_dir: str,
        name: Literal["cr1", "cr2"],
        split: Literal["train", "test"],
        to_neg_1_1: bool = False,
        *args,
        **kwargs,
    ):
        d_name = {"cr1": "CUHK-CR1", "cr2": "CUHK-CR2"}[name]
        input_dir: str = (Path(input_dir) / d_name / split).as_posix()

        super().__init__(input_dir=input_dir, *args, **kwargs)  # type: ignore[unknown-argument]

        self.data_dir = input_dir
        self.split = split
        self.to_neg_1_1 = to_neg_1_1

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        img, gt = sample["img"].float(), sample["gt"].float()
        img /= 255.0
        gt /= 255.0
        if self.to_neg_1_1:
            img = img * 2 - 1
            gt = gt * 2 - 1

        return {"img": img, "gt": gt, "conditions": img}


_TEST = True

if _TEST:
    import pytest

    @pytest.fixture
    def ds():
        return CUHK_CR_StreamingDataset.create_dataset(
            input_dir="data/Downstreams/CUHK-CR/litdata_out",
            name="cr1",
            split="train",
        )

    @pytest.fixture
    def ds_neg():
        return CUHK_CR_StreamingDataset.create_dataset(
            input_dir="data/Downstreams/CUHK-CR/litdata_out",
            name="cr1",
            split="train",
            to_neg_1_1=True,
        )

    def test_equal_img(ds, *args, **kwargs):
        sample = ds[0]
        img, gt = sample["img"], sample["gt"]
        assert img.shape == gt.shape

    def test_value_range(ds, ds_neg):
        sample = ds_neg[1]
        img, gt = sample["img"], sample["gt"]
        assert img.min() >= -1 and img.max() <= 1
        assert gt.min() >= -1 and gt.max() <= 1

    @pytest.mark.parametrize("name, split", [("cr1", "train"), ("cr1", "test"), ("cr2", "train"), ("cr2", "test")])
    def test_all_subdataset(name, split, *args, **kwargs):
        ds = CUHK_CR_StreamingDataset.create_dataset(
            input_dir="data/Downstreams/CUHK-CR/litdata_out",
            name=name,
            split=split,
            to_neg_1_1=True,
        )
        print(f"Test cast: {name=}, {split=}, len of dataset={len(ds)}")
