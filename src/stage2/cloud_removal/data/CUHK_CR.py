import random
from pathlib import Path
from typing import Literal

import litdata as ld
import numpy as np
import torch
from kornia.augmentation import (
    AugmentationSequential,
    RandomBrightness,
    RandomGamma,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomVerticalFlip,
)
from kornia.constants import DataKey
from litdata.utilities import dataset_utilities as litdata_utils
from torch.utils.data import default_collate

from src.data import _BaseStreamingDataset
from src.utilities.config_utils import set_defaults


def rgb_nir_transform(
    p: float,
    *,
    data_keys: list[str | DataKey] = ["input", "input"],
) -> AugmentationSequential:
    return AugmentationSequential(
        RandomHorizontalFlip(p=p),
        RandomVerticalFlip(p=p),
        # RandomBrightness(p=p, brightness=(0.6, 1.6)),
        # RandomGamma(gamma=(0.5, 2.0), gain=(1.5, 1.5), p=p),
        data_keys=data_keys,  # type: ignore[arg-type]
        keepdim=True,
        random_apply=1,
    )


class CUHK_CR_StreamingDataset(_BaseStreamingDataset):
    def __init__(
        self,
        input_dir: str,
        name: Literal["cr1", "cr2", None],
        split: Literal["train", "test", None],
        to_neg_1_1: bool = False,
        interp_to: None | int = None,
        rgb_nir_aug_p: float = 0.0,
        index_file_name: str | None = None,
        force_rgb: bool = False,
        *args,
        **kwargs,
    ):
        if name is not None and split is not None:
            d_name = {"cr1": "CUHK-CR1", "cr2": "CUHK-CR2"}[name]
            input_dir: str = (Path(input_dir) / d_name / split).as_posix()

        self.data_dir = input_dir
        self.split = split
        self.to_neg_1_1 = to_neg_1_1
        self.interp_to = interp_to
        self.rgb_nir_aug_p = rgb_nir_aug_p
        self._rgb_nir_aug: AugmentationSequential | None = None
        self.force_rgb = force_rgb
        if interp_to is not None:
            self._resize_crop = AugmentationSequential(
                RandomResizedCrop(
                    size=(interp_to, interp_to),
                    scale=(0.8, 1.0),
                    ratio=(3 / 4, 4 / 3),
                    resample="BICUBIC",
                    p=1,
                ),
                data_keys=["input", "input"],
                keepdim=True,
            )

        super().__init__(input_dir=input_dir, index_file_name=index_file_name, *args, **kwargs)  # type: ignore[unknown-argument]

    def _apply_rgb_nir_aug(self, img: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.rgb_nir_aug_p <= 0.0 or self.split != "train":
            return img, gt
        if self._rgb_nir_aug is None:
            self._rgb_nir_aug = rgb_nir_transform(self.rgb_nir_aug_p)
        img_aug, gt_aug = self._rgb_nir_aug(img.unsqueeze(0), gt.unsqueeze(0))
        return img_aug.squeeze(0), gt_aug.squeeze(0)

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        img, gt = sample["img"].float(), sample["gt"].float()
        # print(img.min(), img.max(), gt.min(), gt.max())

        # interpolation to 256
        # match the baseline (EMRDM uses bicubic 1/2 resize for 512x512 CUHK)
        if self.interp_to is not None:
            # s = self.interp_to
            # img = torch.nn.functional.interpolate(img[None], size=(s, s), mode="bicubic", align_corners=False)
            # gt = torch.nn.functional.interpolate(gt[None], size=(s, s), mode="bicubic", align_corners=False)
            # img.squeeze_(0)
            # gt.squeeze_(0)
            img, gt = self._resize_crop(img, gt)

        img /= 255.0
        gt /= 255.0
        img = img.clamp(0, 1)
        gt = gt.clamp(0, 1)

        img, gt = self._apply_rgb_nir_aug(img, gt)

        if self.to_neg_1_1:
            img = img * 2 - 1
            gt = gt * 2 - 1

        if self.force_rgb:
            img = img[:3]
            gt = gt[:3]

        return {"img": img, "gt": gt, "conditions": img}


class CUHK_CR_StreamingDataset_RandomKey_For_tokenizerPEFT(_BaseStreamingDataset):
    def __init__(
        self,
        input_dir: str,
        to_neg_1_1: bool = True,
        interp_to: int | None = None,
        getitem_random: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(input_dir=input_dir, *args, **kwargs)  # type: ignore[unknown-argument]
        self.to_neg_1_1 = to_neg_1_1
        self.getitem_random = getitem_random
        self.interp_to = interp_to

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        img, gt = sample["img"].float(), sample["gt"].float()

        img /= 255.0
        gt /= 255.0
        img = img.clamp(0, 1)
        gt = gt.clamp(0, 1)

        if self.interp_to is not None:
            s = self.interp_to
            img = torch.nn.functional.interpolate(img[None], size=(s, s), mode="bilinear", align_corners=False)
            gt = torch.nn.functional.interpolate(gt[None], size=(s, s), mode="bilinear", align_corners=False)
            img.squeeze_(0)
            gt.squeeze_(0)

        if self.to_neg_1_1:
            img = img * 2 - 1
            gt = gt * 2 - 1

        if self.getitem_random:
            k = random.choice(["img", "gt"])  # random get one key
            return {"img": img if k == "img" else gt}
        else:
            return {"img": img, "gt": gt}

    @classmethod
    def create_dataloader(
        cls,
        input_dir: str | list[str],
        stream_ds_kwargs: dict = {},
        combined_kwargs: dict = {"batching_method": "per_stream"},
        loader_kwargs: dict = {},
    ):
        loader_kwargs = set_defaults(
            loader_kwargs,
            collate_fn=random_interp_collate_fn if stream_ds_kwargs.get("getitem_random", True) else default_collate,
        )

        ds = cls.create_dataset(input_dir=input_dir, combined_kwargs=combined_kwargs, **stream_ds_kwargs)
        dl = ld.StreamingDataLoader(ds, **loader_kwargs)
        return ds, dl


def random_interp_collate_fn(batches: list):
    """
    Baselines use 256 px as defaults, for 256/512 px stable reconstruction,
    we randomly choose 256/512 px for training.
    """
    col_batch = default_collate(batches)

    if random.random() < 0.5:
        col_batch["img"] = torch.nn.functional.interpolate(
            col_batch["img"], scale_factor=0.5, mode="bilinear", align_corners=False
        )

    return col_batch


_TEST = True

if _TEST:
    import matplotlib.pyplot as plt
    import pytest

    @pytest.fixture
    def ds():
        ds = CUHK_CR_StreamingDataset.create_dataset(
            input_dir="data/Downstreams/CUHK-CR/litdata_out",
            name="cr1",
            split="train",
        )
        print("cr1 length: ", len(ds))

    @pytest.fixture
    def ds_neg():
        return CUHK_CR_StreamingDataset.create_dataset(
            input_dir="data/Downstreams/CUHK-CR/litdata_out",
            name="cr1",
            split="train",
            to_neg_1_1=True,
        )

    @pytest.fixture
    def ds_rnd_all():
        ds, dl = CUHK_CR_StreamingDataset_RandomKey_For_tokenizerPEFT.create_dataloader(
            input_dir="data/Downstreams/CUHK-CR/litdata_out",
            stream_ds_kwargs=dict(to_neg_1_1=True),
            loader_kwargs={"batch_size": 4},
        )
        return dl

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

    def test_ds_rnd(ds_rnd_all):
        dl = ds_rnd_all
        for sample in dl:
            img = sample["img"]
            print(f"Test cast: {img.shape=}")

    def test_index_file_litdata():
        ds = CUHK_CR_StreamingDataset.create_dataset(
            input_dir="/home/user/zihancao/Project/hyperspectral-1d-tokenizer/data/Downstreams/CUHK-CR/litdata_out/all_train.json",
            name=None,
            split=None,
        )
        print("cr1 length: ", len(ds))

    def test_augmentation():
        # Manual setup to avoid index.json not found error
        # ds_aug = CUHK_CR_StreamingDataset(
        #     input_dir="/home/user/zihancao/Project/hyperspectral-1d-tokenizer/data/Downstreams/CUHK-CR/litdata_out/all_train.json",
        #     name=None,
        #     split=None,
        #     rgb_nir_aug_p=0.0,
        #     interp_to=None,
        #     to_neg_1_1=False,
        # )

        ds_aug = CUHK_CR_StreamingDataset.create_dataset(
            input_dir="data/Downstreams/CUHK-CR/litdata_out",
            name="cr1",
            split="train",
            to_neg_1_1=False,
        )

        dl_aug = ld.StreamingDataLoader(ds_aug, batch_size=4, num_workers=0)  # type: ignore[invalid-argument-type]
        batch = next(iter(dl_aug))
        imgs, gts = batch["img"], batch["gt"]

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        for i in range(4):
            img = imgs[i, :3].permute(1, 2, 0).cpu().numpy()
            gt = gts[i, :3].permute(1, 2, 0).cpu().numpy()
            # print(img.min(), img.max())
            # print(gt.min(), gt.max())

            img = (img - img.min()) / (img.max() - img.min())
            gt = (gt - gt.min()) / (gt.max() - gt.min())

            axes[0, i].imshow(img)
            axes[0, i].set_title(f"Batch {i} - Image")
            axes[0, i].axis("off")

            axes[1, i].imshow(gt)
            axes[1, i].set_title(f"Batch {i} - GT")
            axes[1, i].axis("off")

        plt.tight_layout()
        plt.savefig("augmentation_batch_test.png")
        print("Batch plot saved to augmentation_batch_test.png")
