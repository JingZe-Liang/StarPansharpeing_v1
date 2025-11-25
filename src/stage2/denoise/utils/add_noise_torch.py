"""
Noise Adder for hyperspectral image denoising task.
Based on Hypersigma implementation and modified for Kornia and PyTorch.

Author: Zihan Cao
Date: 2025.11.16
Email: iamzihan666@gmail.com

Copyright (c) 2025 Zihan Cao. All Rights Reserved.
Licensed under the MIT License.
"""

from typing import Literal, TypeVar

import numpy as np
import torch

# * --- Basic transformations --- #


class AddNoiseImpulseTorch(object):
    """Add impulse noise to the given torch tensor (B,1,H,W)"""

    def __init__(self, amounts, s_vs_p=0.5):
        self.amounts = amounts if isinstance(amounts, torch.Tensor) else torch.tensor(amounts)
        self.s_vs_p = s_vs_p

    def __call__(self, img):
        # img shape: (B, 1, H, W)
        B, _, H, W = img.shape
        device = img.device
        dtype = img.dtype

        # Move amounts to correct device
        amounts = self.amounts.to(device)
        amount_idx = torch.randint(0, len(amounts), (1,), device=device).item()
        amount = amounts[amount_idx].item()

        # Create noise mask
        noise_mask = torch.rand(B, 1, H, W, device=device, dtype=dtype) < amount
        salt_mask = torch.rand(B, 1, H, W, device=device, dtype=dtype) < self.s_vs_p

        # Apply noise
        img = img.clone()
        img[noise_mask & salt_mask] = 1.0  # Salt
        img[noise_mask & ~salt_mask] = 0.0  # Pepper

        return img


class AddNoiseStripeTorch(object):
    """Add stripe noise to the given torch tensor (B,1,H,W)"""

    def __init__(self, min_amount=0.05, max_amount=0.15):
        assert max_amount > min_amount
        self.min_amount = min_amount
        self.max_amount = max_amount

    def __call__(self, img):
        # img shape: (B, 1, H, W)
        B, _, H, W = img.shape
        device = img.device
        dtype = img.dtype

        num_stripe = torch.randint(
            int(np.ceil(self.min_amount * W)),
            int(np.ceil(self.max_amount * W)),
            (1,),
            device=device,
        ).item()

        # Generate stripe locations
        loc = torch.randperm(W, device=device)[:num_stripe]
        stripe = torch.rand(len(loc), device=device, dtype=dtype) * 0.5 - 0.25

        # Apply stripe noise
        img = img.clone()
        for i, l in enumerate(loc):
            img[:, :, :, l] -= stripe[i]

        return img


class AddNoiseDeadlineTorch(object):
    """Add deadline noise to the given torch tensor (B,1,H,W)"""

    def __init__(self, min_amount=0.05, max_amount=0.15):
        assert max_amount > min_amount
        self.min_amount = min_amount
        self.max_amount = max_amount

    def __call__(self, img):
        # img shape: (B, 1, H, W)
        B, _, H, W = img.shape
        device = img.device

        num_deadline = torch.randint(
            int(np.ceil(self.min_amount * W)),
            int(np.ceil(self.max_amount * W)),
            (1,),
            device=device,
        ).item()

        # Generate deadline locations
        loc = torch.randperm(W, device=device)[:num_deadline]

        # Apply deadline noise
        img = img.clone()
        for l in loc:
            img[:, :, :, l] = 0

        return img


class AddNoiseInpaintingTorch(object):
    """Add inpainting noise to the given torch tensor (B,C,H,W)"""

    def __init__(self, min_amount=0.0, max_amount=0.1, num=4):
        assert max_amount > min_amount
        self.min_amount = min_amount
        self.max_amount = max_amount
        self.num = num

    def __call__(self, img):
        # img shape: (B, C, H, W)
        B, C, H, W = img.shape
        device = img.device

        # Always use self.num instead of random selection
        num = self.num

        # Generate random widths for each inpainting region
        width_deadline = torch.randint(
            int(np.ceil(self.min_amount * W)),
            int(np.ceil(self.max_amount * W)),
            (num,),
            device=device,
        )

        # Generate random start locations for each inpainting region
        max_start_loc = W - int(np.ceil(self.max_amount * W)) - 1
        if max_start_loc <= 0:
            max_start_loc = 1

        loc_start = torch.randint(0, max_start_loc, (num,), device=device)

        # Apply inpainting noise
        img = img.clone()
        for loc, width in zip(loc_start, width_deadline):
            loc = loc.item()
            width = width.item()
            img[:, :, :, loc : loc + width] = 0

        return img


class AddNoiseNoniidTorch(object):
    def __init__(self, sigmas):
        super().__init__()
        self.sigmas = torch.tensor(sigmas) / 255.0

    def __call__(self, img):
        if img.dim() != 4:
            raise ValueError("Input must be a 4D tensor (BCHW), but got {}D".format(img.dim()))

        batch_size = img.shape[0]

        indices = torch.randint(low=0, high=len(self.sigmas), size=(batch_size,))
        selected_sigmas = self.sigmas[indices].to(img.device)
        selected_sigmas = selected_sigmas.view(-1, 1, 1, 1)
        noise = torch.randn_like(img) * selected_sigmas

        return img + noise


class AddNoiseBlindTorch(object):
    def __init__(self, min_sigma, max_sigma):
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    def __call__(self, img):
        sigma = np.random.uniform(self.min_sigma, self.max_sigma) / 255
        noise = torch.randn_like(img) * sigma

        return img + noise


# * --- Mixed transformations --- #

NoisersType = TypeVar(
    "NoisersType",
    AddNoiseStripeTorch,
    AddNoiseDeadlineTorch,
    AddNoiseImpulseTorch,
    AddNoiseInpaintingTorch,
    AddNoiseNoniidTorch,
    AddNoiseBlindTorch,
)
noisers_mapping = {
    "stripe": AddNoiseStripeTorch,
    "deadline": AddNoiseDeadlineTorch,
    "impulse": AddNoiseImpulseTorch,
    "inpainting": AddNoiseInpaintingTorch,
    "noniid": AddNoiseNoniidTorch,
    "blind": AddNoiseBlindTorch,
}
noisers_default_kwargs = {
    "stripe": {"min_amount": 0.05, "max_amount": 0.15, "num_bands": 1 / 3},
    "deadline": {"min_amount": 0.05, "max_amount": 0.15, "num_bands": 1 / 3},
    "impulse": {"amounts": [0.1, 0.3, 0.5, 0.7], "s_vs_p": 0.5, "num_bands": 1 / 3},
    "inpainting": {"min_amount": 0.0, "max_amount": 0.1, "num": 4, "num_bands": 1 / 3},
    "noniid": {"sigmas": [30, 50, 70, 90], "num_bands": 1 / 3},
    "blind": {"min_sigma": 0.01, "max_sigma": 0.1, "num_bands": 1 / 3},
}


class TransformationMixedTorch(object):
    """Add mixed noise to the given torch tensor (B,C,H,W)
    Args:
        noise_bank: list of noise maker (e.g. AddNoiseImpulse)
        num_channels: list of number of channels which is corrupted by each item in noise_bank"""

    def __init__(self, noise_bank: list, num_channels: list):
        assert len(noise_bank) == len(num_channels)
        self.noise_bank = noise_bank
        self.num_channels = num_channels

    @classmethod
    def from_config(
        cls,
        noise_bank: list[str | type[NoisersType]],
        noise_bank_cfg: list[dict],
        num_channels: list[float],
    ):
        """Create an instance from a configuration dictionary."""

        assert len(noise_bank) == len(noise_bank_cfg), "noise_bank and noise_bank_cfg must have the same length"
        # noisers
        transf_s = []
        for transf_cls, transf_cfg in zip(noise_bank, noise_bank_cfg):
            if isinstance(transf_cls, str):
                transf_cls = noisers_mapping[transf_cls]

            transf = transf_cls(**transf_cfg)
            transf_s.append(transf)

        return cls(
            noise_bank=transf_s,
            num_channels=num_channels,
        )

    def __call__(self, img):
        # img shape: (B, C, H, W)
        B, C, H, W = img.shape
        device = img.device
        all_channels = torch.randperm(C, device=device)
        pos = 0
        for noise_maker, num_channel in zip(self.noise_bank, self.num_channels):
            if 0 < num_channel <= 1:
                num_channel = int(np.floor(num_channel * C))
            # Ensure we don't exceed available channels
            num_channel = min(num_channel, C - pos)
            if num_channel <= 0:
                break
            channels = all_channels[pos : pos + num_channel]
            pos += num_channel
            # Apply noise to selected channels
            for c in channels:
                img[:, c, :, :] = noise_maker(img[:, c, :, :].unsqueeze(1)).squeeze(1)
        return img

    def __repr__(self) -> str:
        noise_adder_repr = ",".join([noise_adder.__class__.__name__ for noise_adder in self.noise_bank])

        return f"TransformationMixedTorch(noise_bank={noise_adder_repr}, num_channels={self.num_channels})"


type PredefinedNoiseType = (
    # predefined noise types
    Literal["complex", "complex_hypersigma", "blind_gaussian_hypersigma"]
    # single noise types
    | Literal["stripe", "deadline", "impulse", "inpainting", "noniid", "blind"]
    | str  # for linting
)


def get_default_noise_transformation(
    trans_type: PredefinedNoiseType,
    trans_kwargs: dict | None = None,
):
    """
    Support predefined noise models and single noise models.
    """
    ###### predefined noise models ######
    if trans_type == "complex":
        mixed_noiser = TransformationMixedTorch(
            [
                AddNoiseStripeTorch(0.05, 0.15),
                AddNoiseDeadlineTorch(0.05, 0.15),
                AddNoiseImpulseTorch([0.1, 0.3, 0.5, 0.7]),
            ],
            [1 / 3, 1 / 3, 1 / 3],
        )
    elif trans_type == "complex_hypersigma":
        # config from HyperSIGMA code
        mixed_noiser = TransformationMixedTorch(
            noise_bank=[
                AddNoiseImpulseTorch(amounts=[0.1, 0.3, 0.5, 0.7]),
                AddNoiseImpulseTorch(amounts=[0.1, 0.3, 0.5, 0.7]),
                AddNoiseStripeTorch(),
                AddNoiseDeadlineTorch(),
            ],
            num_channels=[1 / 3, 1 / 3, 1 / 3, 1 / 3],
        )
    elif trans_type == "blind_gaussian_hypersigma":
        mixed_noiser = TransformationMixedTorch([AddNoiseBlindTorch(10, 70)], [1.0])

    ###### Per-kind noise models ######
    else:
        if trans_kwargs is not None:
            cfg = trans_kwargs
        else:
            cfg = noisers_default_kwargs.get(trans_type, None)
        assert cfg is not None, f"{trans_type} is not supported"
        num_bands = cfg.pop("num_bands", 1 / 3)

        mixed_noiser = TransformationMixedTorch.from_config(
            [trans_type],
            noise_bank_cfg=[cfg],
            num_channels=[num_bands],
        )

    return mixed_noiser


# * --- Testing --- #

import pytest


@pytest.mark.parametrize(
    "device",
    ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"],
)
@pytest.mark.parametrize("shape", [(1, 1, 256, 256), (8, 32, 256, 256), (4, 64, 128, 128), (32, 128, 64, 64)])
def test_noisers(device, shape):
    """Test the noise adder"""

    test_noisers = [
        "complex",
        "stripe",
        "deadline",
        "impulse",
        "inpainting",
        "noniid",
        "blind",
    ]

    for noiser in test_noisers:
        noise_adder = get_default_noise_transformation(noiser)
        print(f"Testing noise adder: {noise_adder} on device {device}")
        # Example usage
        img = torch.rand(*shape).to(device)  # Example image with 8 samples, 32 channels, 256x256 pixels
        noisy_img = noise_adder(img)
        print(noisy_img.shape)  # Should be the same shape as input image


# test_noisers("cpu", (1, 3, 256, 256))  # Run a simple test
