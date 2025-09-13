import random

import numpy as np
import torch
import webdataset as wds
from torch import Tensor

from src.data.hyperspectral_loader import (
    get_hyperspectral_dataloaders,
    get_hyperspectral_img_loaders_with_different_backends_v2,
)
from src.stage2.denoise.utils.noise_adder import (
    PredefinedNoiseType,
    UniHSINoiseAdderKornia,
)
from src.utilities.logging import log


class GeneralNoisyLoader:
    def __init__(
        self,
        loader_kwargs: dict,
        # list of list noisers means accumulated the noise on the second noisers' effects
        noise_types: list[PredefinedNoiseType] | list[list[PredefinedNoiseType]] = [
            "complex"
        ],
        noise_adder_kwargs: dict = {},
    ):
        self.dataset, self.dataloader = get_hyperspectral_dataloaders(**loader_kwargs)
        is_neg_1_1 = loader_kwargs.get("to_neg_1_1", False)

        # Initialize multiple noise adders if multiple noise types are provided
        self.noisers = []
        self.noise_types = noise_types
        for nt in noise_types:
            noise_kwargs_ = noise_adder_kwargs.copy()
            noise_kwargs_.update(noise_type=nt, is_neg_1_1=is_neg_1_1)
            noiser = UniHSINoiseAdderKornia(**noise_kwargs_)
            self.noisers.append(noiser)
            log(f"Initialized noise adder with types: {nt}")

    def __iter__(self):
        for batch in self.dataloader:
            # Randomly select a noise adder for each batch
            noiser = random.choice(self.noisers)

            x = batch["clean"] = batch["img"]
            del batch["img"]
            n_chan = x.shape[1]

            noisy = noiser(x)
            batch.update({"n_chan": n_chan, "noisy": noisy})

            yield batch


# * --- Test --- #


def test_loader():
    path = "data/MDAS-HySpex/hyper_images/MDAS-HySpex-368_bands-px_256-MSI-0000.tar"
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
        noise_types=["complex", "noniid", "deadline", "impulse"],
        noise_adder_kwargs=noise_adder_kwargs,
    )

    print("=== Testing General Noisy DataLoader ===")
    import time

    t1 = time.time()
    for batch in loader:
        print(batch["clean"].shape)
        t2 = time.time()
        print(f"Time taken for one batch: {t2 - t1:.4f} seconds")
        t1 = t2


if __name__ == "__main__":
    test_loader()
