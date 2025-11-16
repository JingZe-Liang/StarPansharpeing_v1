import random
from collections import defaultdict
from numbers import Number
from typing import Mapping

import litdata
import numpy as np
import torch
import webdataset as wds
from torch import Tensor
from torch.utils.data import DataLoader, default_collate
from tqdm import tqdm

from src.data.hyperspectral_loader import (
    get_hyperspectral_dataloaders,
    get_hyperspectral_img_loaders_with_different_backends_v2,
)
from src.data.litdata_hyperloader import (
    IndexedCombinedStreamingDataset,
    SingleCycleStreamingDataset,
    _BaseStreamingDataset,
)
from src.stage2.denoise.utils.noise_adder import (
    PredefinedNoiseType,
    UniHSINoiseAdderKornia,
)
from src.utilities.logging import log

# 5-kinds of cases
CASES_NOISE_TYPES = [
    ["noniid"],
    ["noniid", "impulse"],
    ["noniid", "deadline"],
    ["noniid", "stripe"],
    ["noniid", "stripe", "deadline", "impulse"],
    # add 'complex' as a whole noise model
    ["complex"],
]


class __CachedDataset(torch.utils.data.Dataset):
    def __init__(self, data: list[Mapping]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class GeneralNoisyLoader:
    def __init__(
        self,
        loader_kwargs: dict,
        # list of list noisers means accumulated the noise on the second noisers' effects
        noise_types: list[PredefinedNoiseType]
        | list[list[PredefinedNoiseType]] = CASES_NOISE_TYPES,  # type: ignore
        noise_adder_kwargs: dict = {},
        get_loader_type: str = "wds",
        cache: bool = True,  # when the dataset is small
    ):
        if get_loader_type == "wds":
            if cache:
                # When caching, we handle resampling ourselves
                self.cache_resample = loader_kwargs.get("resample", True)
                loader_kwargs["resample"] = (
                    False  # disable underlying dataloader resample
                )
            self.dataset, self.dataloader = get_hyperspectral_dataloaders(
                **loader_kwargs
            )
        elif get_loader_type == "litdata":
            raise

        self.is_neg_1_1 = loader_kwargs.get("to_neg_1_1", True)
        self.cache = cache
        if not cache:
            self.cache_resample = loader_kwargs.get("resample", True)

        # Initialize multiple noise adders if multiple noise types are provided
        self.noisers = []
        self.noise_types = noise_types
        for nt in noise_types:
            noise_kwargs_ = noise_adder_kwargs.copy()
            noise_kwargs_.update(noise_type=nt, is_neg_1_1=self.is_neg_1_1)
            noiser = UniHSINoiseAdderKornia(**noise_kwargs_)
            self.noisers.append(noiser)
            log(f"Initialized noise adder with types: {nt}")

        # Cache all of the dataset
        if self.cache:
            log("Caching all dataset batches...")
            self.data = []
            for batch in tqdm(self.dataloader, desc="Caching dataset ..."):
                # Cache clean images only
                cached_batch = {}
                cached_batch["gt"] = batch["img"]
                n_chan = batch["img"].shape[1]
                cached_batch["n_chan"] = n_chan
                self.data.append(cached_batch)

            log(f"Cached {len(self.data)} batches")

    def _add_noise_to_batch(self, clean_img: Tensor) -> Tensor:
        """
        Add noise to clean image batch.

        Parameters
        ----------
        clean_img : Tensor
            Clean image tensor

        Returns
        -------
        Tensor
            Noisy image tensor
        """
        # Randomly select a noise adder for each batch
        noiser = random.choice(self.noisers)

        # Noise the image
        noisy = noiser(clean_img)

        # Clip values
        if self.is_neg_1_1:
            noisy = torch.clamp(noisy, -1, 1)
        else:
            noisy = torch.clamp(noisy, 0, 1)

        return noisy

    def __iter__(self):
        if self.cache:
            if self.cache_resample:
                # Resample mode: infinite loop
                while True:
                    for cached_batch in self.data:
                        batch = cached_batch.copy()
                        batch["noisy"] = self._add_noise_to_batch(batch["gt"])
                        yield batch
            else:
                # Single pass mode: iterate once
                for cached_batch in self.data:
                    batch = cached_batch.copy()
                    batch["noisy"] = self._add_noise_to_batch(batch["gt"])
                    yield batch
        else:
            # Process batches on-the-fly (dataloader already handles resampling)
            for batch in self.dataloader:
                batch["gt"] = batch["img"]
                del batch["img"]
                n_chan = batch["gt"].shape[1]
                batch["noisy"] = self._add_noise_to_batch(batch["gt"])
                batch["n_chan"] = n_chan
                yield batch

    @classmethod
    def create_dataloader(
        cls,
        loader_kwargs: dict,
        noise_types=CASES_NOISE_TYPES,
        noise_adder_kwargs: dict[str, dict] | None = None,
        get_loader_type: str = "wds",
        cache: bool = True,
    ):
        loader = cls(
            loader_kwargs=loader_kwargs,
            noise_types=noise_types,
            noise_adder_kwargs=noise_adder_kwargs or {},
            get_loader_type=get_loader_type,
            cache=cache,
        )

        # is_cached = loader.cache
        # if is_cached:
        #     loader = DataLoader(
        #         loader,
        #         batch_size=loader_kwargs["batch_size"],
        #         num_workers=loader_kwargs["num_workers"],
        #         pin_memory=loader_kwargs.get("pin_memory", False),
        #         persistent_workers=loader_kwargs.get("persistent_workers", False),
        #         shuffle=loader_kwargs.get("shuffle_size", 0) > 0,
        #         drop_last=loader_kwargs.get("drop_last", False),
        #     )

        return loader.dataset, loader


class NoisyCasesDataset(_BaseStreamingDataset):
    def __init__(
        self,
        input_dir: str,
        rename_dict: dict | None = None,
        to_neg_1_1: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(
            input_dir=input_dir,
            *args,
            **kwargs,
        )
        self.rename_dict = rename_dict if rename_dict is not None else {}
        self.to_neg_1_1 = to_neg_1_1

    def _rename_key(self, name: str) -> str:
        return self.rename_dict.get(name, name)

    def __getitem__(self, idx) -> dict:
        d = super().__getitem__(idx)
        for k in d:
            kr = self._rename_key(k)
            d[kr] = d.pop(k)

        # gt and noisy should be clipped values
        d["gt"] = torch.clamp(d["gt"], 0, 1)
        d["noisy"] = torch.clamp(d["noisy"], 0, 1)

        if self.to_neg_1_1:
            d["gt"] = (d["gt"] - 0.5) * 2
            d["noisy"] = (d["noisy"] - 0.5) * 2

        return d

    @classmethod
    def create_dataloader(
        cls,
        input_dir: str,
        rename_dict: dict = {},
        dataset_kwargs: dict | None = None,
        loader_kwargs: dict | None = None,
    ):
        dataset = cls(input_dir, rename_dict, **(dataset_kwargs or {}))
        # Dataloader
        dataloader = litdata.StreamingDataLoader(
            dataset,
            collate_fn=not_pure_tensor_collate_fn,
            **(loader_kwargs or {}),
        )
        return dataset, dataloader


def not_pure_tensor_collate_fn(batch: list[dict]):
    not_tensor_lst = defaultdict(list)
    tensor_samples = []
    collate_not_supported_types = (str, bytes)

    # separate tensor and non-tensor items
    for sample in batch:
        tensor_sample = {}
        for attr, value in sample.items():
            if isinstance(value, collate_not_supported_types):
                not_tensor_lst[attr].append(value)
            else:
                tensor_sample[attr] = value
        tensor_samples.append(tensor_sample)

    # default collate for tensors
    tensor_collated = default_collate(tensor_samples)

    return {**tensor_collated, **not_tensor_lst}


# * --- Test --- #


def test_loader():
    path = "data/Downstreams/WDCDenoise/hyper_images/Washington_DC_mall-191_bands-px_192-0000.tar"
    loader_kwargs = {
        "wds_paths": path,
        "batch_size": 2,
        "num_workers": 1,
        "shuffle_size": 0,
        "to_neg_1_1": False,
        "resample": False,
        "pin_memory": True,
        "shuffle_size": -1,
        "tgt_key": "img",
    }
    noise_adder_kwargs = {
        "p": 1.0,
        "same_on_batch": True,
        "keepdim": True,
        "use_torch": True,
        "clip_value": True,
    }

    loader = GeneralNoisyLoader(
        loader_kwargs=loader_kwargs,
        # noise_types=["complex", "noniid", "deadline", "impulse"],
        noise_adder_kwargs=noise_adder_kwargs,
    )

    print("=== Testing General Noisy DataLoader ===")
    import time

    t1 = time.time()
    for batch in loader:
        print(batch["gt"].shape)
        t2 = time.time()
        print(f"Time taken for one batch: {t2 - t1:.4f} seconds")
        t1 = t2


def test_noisy_cases():
    ds, dl = NoisyCasesDataset.create_dataloader(
        "data/Downstreams/WDCDenoise/_cases_litdata",
        loader_kwargs=dict(
            batch_size=2,
            num_workers=4,
            pin_memory=True,
        ),
    )
    for sample in dl:
        print(sample.keys())


if __name__ == "__main__":
    # test_loader()
    test_noisy_cases()
