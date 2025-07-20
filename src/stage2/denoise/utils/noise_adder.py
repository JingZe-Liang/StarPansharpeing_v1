import numpy as np
import torch

from src.stage2.denoise.utils.add_noise import (
    AddNoiseComplex,
    AddNoiseDeadline,
    AddNoiseImpulse,
    AddNoiseInpainting,
    AddNoiseNoniid,
    AddNoiseStripe,
)

# TODO: add torch wrapper for these noise models using torch.


class UniHSINoiseAdder:
    def __init__(self, noise_type: str | list[str] = "complex"):
        self.noise_models = []
        if isinstance(noise_type, str):
            noise_type = [noise_type]
        for nt in noise_type:
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
            elif nt == "stripe":
                self.noise_models.append(AddNoiseStripe())
            else:
                raise ValueError(f"Unknown noise type: {nt}")

    def call_fn(self, img: np.ndarray | torch.Tensor):
        _squeezed = False
        if isinstance(img, torch.Tensor):
            if img.ndim == 4:
                assert img.shape[0] == 1, "Only single image input is supported."
                img.squeeze_(0)
                _squeezed = True
            img = img.float().cpu().numpy()
        else:
            img = img.astype(np.float32).transpose(
                2, 0, 1
            )  # Convert to (c, h, w) format

        # img: (c, h, w), 0 .. 1
        img_noisy_np = img.copy()  # copy it
        for noise_model in self.noise_models:
            img_noisy_np = noise_model(img_noisy_np)

        if _squeezed:
            img_noisy_np = img_noisy_np[None]
        return img_noisy_np

    def __call__(self, img: np.ndarray | torch.Tensor):
        if img.ndim == 3 and torch.is_tensor(img):
            img = img[None]
        elif isinstance(img, np.ndarray):
            img = [img]  # type: ignore

        imgs_np = []
        for im in img:
            imgs_np.append(self.call_fn(im))

        if len(imgs_np) == 1:
            return imgs_np[0]
        else:
            return np.stack(imgs_np, axis=0)


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
    ):
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.is_neg_1_1 = is_neg_1_1
        self.noise_adder = UniHSINoiseAdder(noise_type=noise_type)

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
        noisy_img_np = self.noise_adder(input)
        noisy_img_th = (
            torch.from_numpy(noisy_img_np)
            .to(device=input.device, dtype=dtype)
            .clip(0, 1)
        )

        if self.is_neg_1_1:
            # back to (-1, 1)
            noisy_img_th.mul_(2).sub_(1)

        return noisy_img_th


def get_tokenizer_trainer_noise_adder(p=0.1, is_neg_1_1=True):
    return UniHSINoiseAdderKornia(
        noise_type="complex",
        is_neg_1_1=is_neg_1_1,
        p=p,
        same_on_batch=False,
        keepdim=True,
    )


if __name__ == "__main__":
    # Example usage
    # noise_adder = UniHSINoiseAdder(noise_type=["noniid"])
    noise_adder = UniHSINoiseAdderKornia(
        noise_type=["complex"], keepdim=True, is_neg_1_1=False
    )
    # img = np.random.rand(150, 128, 128)  # Example image with 150 bands
    import tifffile

    img = tifffile.imread("data/OHS/hyper_images/tmp/O1_0010_patch-0.img.tiff")
    img = (
        torch.as_tensor(img.astype(np.float32) / img.max())
        .permute(2, 0, 1)[None]
        .repeat(4, 1, 1, 1)
    )

    print(img.shape)  # Should be (1, C, H, W)
    noisy_img = noise_adder(img)
    print(noisy_img.shape)  # Should be the same shape as input image
