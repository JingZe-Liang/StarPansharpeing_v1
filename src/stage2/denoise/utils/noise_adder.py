from typing import cast

import numpy as np
import torch

from src.stage2.denoise.utils.add_noise import (
    AddNoiseBlindv1,
    AddNoiseComplex,
    AddNoiseDeadline,
    AddNoiseImpulse,
    AddNoiseInpainting,
    AddNoiseNoniid,
    AddNoiseStripe,
)
from src.stage2.denoise.utils.add_noise_torch import (
    PredefinedNoiseType,
    get_default_noise_transformation,
)


class UniHSINoiseAdder:
    def __init__(
        self,
        noise_type: PredefinedNoiseType | list[PredefinedNoiseType],
        use_torch: bool = True,
    ):
        self.use_torch = use_torch
        self.noise_models = []
        self._noise_type = noise_type

        if isinstance(noise_type, str):
            noise_type = [noise_type]

        for nt in noise_type:
            ######### Use numpy version #######
            if not use_torch:
                if nt == "complex":
                    # which is strip, deadline, impulse models in.
                    self.noise_models.append(AddNoiseComplex())
                elif nt == "deadline":
                    self.noise_models.append(AddNoiseDeadline())
                elif nt == "impulse":
                    self.noise_models.append(AddNoiseImpulse())
                elif nt == "inpainting":
                    self.noise_models.append(AddNoiseInpainting(4))
                elif nt == "noniid":
                    sigmas = [30, 50, 70, 90]
                    self.noise_models.append(AddNoiseNoniid(sigmas))
                elif nt == "blind_gaussian":
                    sigmas = (10, 70)
                    self.noise_models.append(AddNoiseBlindv1(*sigmas))
                elif nt == "stripe":
                    self.noise_models.append(AddNoiseStripe())
                else:
                    raise ValueError(f"Unknown noise type: {nt}")
            ####### Use Torch version #######
            else:
                self.noise_models.append(get_default_noise_transformation(nt))

    def call_fn_numpy(self, img: np.ndarray | torch.Tensor):
        _squeezed = False
        if isinstance(img, torch.Tensor):
            if img.ndim == 4:
                assert img.shape[0] == 1, "Only single image input is supported."
                img.squeeze_(0)
                _squeezed = True
            img = img.float().cpu().numpy()
        else:
            img = img.astype(np.float32).transpose(2, 0, 1)  # Convert to (c, h, w) format

        # img: (c, h, w), 0 .. 1
        img_noisy_np = img.copy()  # copy it
        for noise_model in self.noise_models:
            img_noisy_np = noise_model(img_noisy_np)

        if _squeezed:
            img_noisy_np = img_noisy_np[None]
        return img_noisy_np

    def call_fn_torch(self, img: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float()
        if img.ndim == 3:
            img = img.unsqueeze(0)

        # img: (c, h, w), 0 .. 1
        img_noisy_th = img.clone()
        for noise_model in self.noise_models:
            img_noisy_th = noise_model(img_noisy_th)

        return img_noisy_th.squeeze(0)

    def __call__(self, img: np.ndarray | torch.Tensor):
        if img.ndim == 3 and torch.is_tensor(img):
            img = img[None]
        elif isinstance(img, np.ndarray):
            img = [img]  # type: ignore[assignment]

        if not self.use_torch:
            imgs = []
            for im in img:
                im = self.call_fn_numpy(im)
                imgs.append(im)
        else:
            imgs = self.call_fn_torch(img)

        if len(imgs) == 1:
            return imgs[0]
        elif not self.use_torch:
            imgs = cast(list[np.ndarray], imgs)
            return np.stack(imgs, axis=0)
        else:
            return imgs

    def __repr__(self):
        return f"{self.__class__.__name__}(noise_types={self._noise_type}, use_torch={self.use_torch})"


# * --- Kornia compatible --- #

from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D


class UniHSINoiseAdderKornia(IntensityAugmentationBase2D):
    def __init__(
        self,
        noise_type: str | list[str] = "complex",
        is_neg_1_1: bool = False,
        p: float = 1.0,
        same_on_batch: bool = False,
        keepdim: bool = True,
        use_torch=True,
        clip_value=False,
    ):
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.is_neg_1_1 = is_neg_1_1
        self.clip_value = clip_value
        self.noise_adder = UniHSINoiseAdder(noise_type=noise_type, use_torch=use_torch)
        self._noise_type = noise_type
        self._use_torch = use_torch

    def apply_transform(self, input: torch.Tensor, params, flags, transform=None):
        """
        Apply the noise addition transformation to the input tensor.

        Args:
            input (torch.Tensor): Input tensor of shape (B, C, H, W) or (C, H, W).
            params: Parameters for the transformation (not used here).
            flags: Additional flags for the transformation (not used here).
            transform: Optional transformation matrix (not used here).

        Returns:
            torch.Tensor: Noisy image tensor of the same shape as input.
        """
        dtype = input.dtype

        if self.is_neg_1_1:
            # to (0, 1)
            input.add_(1).div_(2)

        # Convert input to numpy and apply noise
        noisy_img = self.noise_adder(input)
        noisy_img_th = torch.as_tensor(noisy_img).to(device=input.device, dtype=dtype)
        if self.clip_value:
            noisy_img_th.clamp_(0, 1)
        noisy_img_th.unsqueeze_(0)  # ensure batch dim

        if self.is_neg_1_1:
            # back to (-1, 1)
            noisy_img_th.mul_(2).sub_(1)

        return noisy_img_th

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(noise_type={self._noise_type}, clip_value={self.clip_value},"
            f"use_torch={self._use_torch}, is_neg_1_1={self.is_neg_1_1}, p={self.p})"
        )


# Alias
def get_tokenizer_trainer_noise_adder(p=0.1, is_neg_1_1=True, use_torch=True):
    return UniHSINoiseAdderKornia(
        noise_type="complex",
        is_neg_1_1=is_neg_1_1,
        p=p,
        same_on_batch=True,
        keepdim=True,
        use_torch=use_torch,
    )


if __name__ == "__main__":
    from src.utilities.network_utils.perf_utils import timer

    # Example usage
    # noise_adder = UniHSINoiseAdder(noise_type=["noniid"])
    noise_adder = UniHSINoiseAdderKornia(noise_type=["complex"], keepdim=True, is_neg_1_1=False, use_torch=True)
    # img = np.random.rand(150, 128, 128)  # Example image with 150 bands
    import tifffile

    # img = tifffile.imread("data/OHS/hyper_images/tmp/O1_0010_patch-0.img.tiff")
    # img = (
    #     torch.as_tensor(img.astype(np.float32) / img.max())
    #     .permute(2, 0, 1)[None]
    #     .repeat(4, 1, 1, 1)
    # )

    img = torch.rand(8, 32, 256, 256)

    print(img.shape)  # Should be (1, C, H, W)
    for _ in range(100):
        with timer() as t:
            noisy_img = noise_adder(img)
            print(noisy_img.shape)  # Should be the same shape as input image
