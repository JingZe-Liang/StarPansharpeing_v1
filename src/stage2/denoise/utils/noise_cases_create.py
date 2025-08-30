import os
from os import mkdir
from pathlib import Path
from random import shuffle
from typing import Literal

import numpy as np
import torch
import webdataset as wds
from PIL import Image
from scipy.io import loadmat, savemat
from torchvision.utils import make_grid

from src.data.codecs import tiff_codec_io
from src.data.hyperspectral_loader import get_hyperspectral_dataloaders
from src.stage2.denoise.utils.noise_adder import UniHSINoiseAdderKornia
from src.utilities.logging import catch_any, log

MAX_VAL = 32701


def plot_img(img, rgb_chans=[18, 12, 8]):
    # (bs, c, h, w)
    img = (
        make_grid(
            torch.as_tensor(img),
            nrow=1,
            normalize=True,
            scale_each=True,
        )
        .cpu()
        .numpy()
    )
    img = img[rgb_chans].transpose([1, 2, 0])  # (h, w, c)
    img = (img * 255.0).astype("uint8")
    img = Image.fromarray(img)
    return img


def round_img(img, max_val=MAX_VAL):
    img = img * max_val
    img = img.astype("uint16")
    return img


@catch_any()
def add_noise_cases_torch(
    input_path: str,
    save_dir: str,
    device: str | torch.device = "cpu",
    case_names: list[str] | Literal["all"] = "all",
) -> None:
    """
    Generates test cases with different noise types using UniHSINoiseAdderKornia.

    This function reads clean HSI data from .mat files, applies specified noise
    combinations using PyTorch-based noise adders, crops the results, and saves
    them into separate directories for each case.

    Args:
        srcdir: Path to the source directory containing clean .mat files.
                Each file is expected to have a key 'data' with shape (H, W, C).
        dstdir: Path to the destination root directory where noisy cases will be saved.
                Subdirectories for each case will be created under this path.
        device: The torch device ('cpu' or 'cuda') to run the noise addition on.
                Defaults to "cpu".
        case_names: A list of case names to generate (e.g., ["Case2", "Case3"])
                    or "all" to generate all predefined cases (Case2, Case3, Case4).
                    Defaults to "all".
    """
    # Define available test cases and their corresponding noise types.
    # The 'noniid' noise is implicitly included as part of the base setup
    # in the original logic and the UniHSINoiseAdderKornia configuration.
    # Here, we map case names to the additional specific noise types.
    available_cases = {
        "case1": ["blind_gaussian_hypersigma"],
        "case2": ["impulse"],
        "case3": ["deadline", "impulse"],
        "case4": ["stripe", "impulse"],
        "case5": ["stripe", "deadline", "impulse"],
        "case6": ["complex_hypersigma"],
    }

    if case_names == "all":
        case_names = list(available_cases.keys())

    # Create noise adder instances for each specified case.
    noise_adders = {}
    for case_name in case_names:
        if case_name in available_cases:
            # UniHSINoiseAdderKornia is configured to always apply noise (p=1.0),
            # keep input dimensions (keepdim=True), expect [0, 1] input (is_neg_1_1=False),
            # and use the torch implementation (use_torch=True).
            # The 'noniid' noise is handled by the UniHSINoiseAdder internally
            # based on the 'noise_type' list. If the list contains only one type,
            # it wraps it correctly.
            noise_adders[case_name] = UniHSINoiseAdderKornia(
                noise_type=available_cases[case_name],
                p=1.0,
                keepdim=True,
                is_neg_1_1=False,
                use_torch=True,
            ).to(device)
        else:
            print(f"Warning: Unknown case name '{case_name}', skipping.")

    ds, dl = get_hyperspectral_dataloaders(
        input_path,
        batch_size=1,
        shuffle_size=0,
        to_neg_1_1=False,
        resample=False,
        transform_prob=0.0,
        degradation_prob=0.0,
        num_workers=0,
    )
    for name, noise_adder in noise_adders.items():
        log(f"working on case {name}")
        # save img dir
        save_img_dir = Path(save_dir, f"{name}_img")
        save_img_dir.mkdir(parents=True, exist_ok=True)

        save_path = Path(save_dir, f"WDC_191_bands_px_192-{name}.tar").as_posix()
        writter = wds.TarWriter(save_path)

        for sample in dl:
            img = sample["img"]  # (bs, c, h, w)
            img_noisy = noise_adder(img)

            # plot img
            plot_img(img).save(save_img_dir / f"{sample['__key__'][0]}_clean.png")
            plot_img(img_noisy).save(save_img_dir / f"{sample['__key__'][0]}_noisy.png")

            # to (h, w, c)
            img = img[0].permute(1, 2, 0).cpu().numpy()
            img_noisy = img_noisy[0].permute(1, 2, 0).cpu().numpy()

            img = round_img(img)
            img_noisy = round_img(img_noisy)

            img_bytes = tiff_codec_io(
                img,
                compression="jpeg2000",
                compression_args={"reversible": False, "level": 90},
            )
            img_noisy_bytes = tiff_codec_io(
                img_noisy,
                compression="jpeg2000",
                compression_args={"reversible": False, "level": 90},
            )

            writter.write(
                {
                    "__key__": sample["__key__"][0],
                    "noisy": img_noisy_bytes,
                    "clean": img_bytes,
                }
            )

    log("Finished processing all files.")


# Example usage (if run as a script):
if __name__ == "__main__":
    add_noise_cases_torch(
        "data/WDC/hyper_images/Washington_DC_mall-191_bands-px_192-0000.tar",
        "data/WDC/denoise_cases",
    )
