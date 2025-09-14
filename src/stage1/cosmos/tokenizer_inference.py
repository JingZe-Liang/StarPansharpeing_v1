from typing import cast

import torch
import torch.nn as nn
from torch import Tensor

from .cosmos_gen_tokenizer import CosmosGenerativeTokenizer
from .cosmos_tokenizer import ContinuousImageTokenizer

# * --- Latent utilities --- #


def dim_match(tensor: Tensor, target: Tensor) -> Tensor:
    if tensor.ndim == 0:
        return tensor
    elif tensor.ndim == 1:
        if target.ndim == 4:  # b, c, h, w
            return tensor.view(1, -1, 1, 1)
        elif target.ndim == 3:  # b, l, c
            return tensor.view(1, 1, -1)
        else:
            raise ValueError(f"Unsupported target ndim: {target.ndim}")
    else:
        raise ValueError(f"Unsupported tensor ndim: {tensor.ndim}")


def scale_shift_latent(
    latent: Tensor, scale: Tensor | float, shift: Tensor | float
) -> Tensor:
    """Scale and shift the latent tensor."""
    if isinstance(scale, float):
        scale = torch.as_tensor(scale, device=latent.device, dtype=latent.dtype)
    if isinstance(shift, float):
        shift = torch.as_tensor(shift, device=latent.device, dtype=latent.dtype)

    scale = cast(Tensor, scale)
    shift = cast(Tensor, shift)

    scale = dim_match(scale, latent)
    shift = dim_match(shift, latent)
    latent.sub_(shift).div_(scale)

    return latent


def un_scale_shift_latent(latent, scale, shift):
    """Un-scale and un-shift the latent tensor."""
    if isinstance(scale, float):
        scale = torch.as_tensor(scale, device=latent.device, dtype=latent.dtype)
    if isinstance(shift, float):
        shift = torch.as_tensor(shift, device=latent.device, dtype=latent.dtype)

    scale = dim_match(scale, latent)
    shift = dim_match(shift, latent)
    latent.mul_(scale).add_(shift)

    return latent


# * --- Inference --- #

type LatentScaleShiftType = tuple[float, float] | tuple[list[float], list[float]] | None


class TokenizerInferenceWrapper(nn.Module):
    _last_x_shape: torch.Size | None = None

    def __init__(
        self,
        tokenizer: ContinuousImageTokenizer | CosmosGenerativeTokenizer,
        tokenizer_scale_shift: LatentScaleShiftType = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer.eval()

        # scale and shift factors
        scale_factor, shift_factor = tokenizer_scale_shift or (1.0, 0.0)
        self.register_buffer("scale_factor", torch.as_tensor(scale_factor))
        self.register_buffer("shift_factor", torch.as_tensor(shift_factor))
        self.scale_factor: nn.Buffer
        self.shift_factor: nn.Buffer
        assert self.scale_factor.numel() in [1, self.tokenizer.latent_channels], (
            "scale_factor must be a float or a list of float with length equal to "
            "the number of latent channels or 1"
        )
        assert self.shift_factor.numel() in [1, self.tokenizer.latent_channels], (
            "shift_factor must be a float or a list of float with length equal to "
            "the number of latent channels or 1"
        )

    @property
    def quantizer(self):
        return self.tokenizer.quantizer

    @property
    def encoder(self):
        return self.tokenizer.encoder

    @property
    def decoder(self):
        return self.tokenizer.decoder

    def _shift_scale_latent(self, latent: Tensor, reverse: bool = False) -> Tensor:
        if reverse:
            return un_scale_shift_latent(latent, self.scale_factor, self.shift_factor)
        else:
            return scale_shift_latent(latent, self.scale_factor, self.shift_factor)

    @torch.no_grad()
    def encode(self, x: Tensor) -> Tensor:
        """Encode an image to latent tokens."""
        self._last_x_shape = x.shape
        z = self.tokenizer.encode(x)
        if not torch.is_tensor(z):
            z = z[0]
        z = self._shift_scale_latent(z, reverse=False)
        return z

    @torch.no_grad()
    def decode(self, z: Tensor, input_shape: torch.Size | None = None) -> Tensor:
        """Decode latent tokens to an image."""
        z = self._shift_scale_latent(z, reverse=True)
        input_shape = input_shape or self._last_x_shape
        assert input_shape is not None, (
            "input_shape must be provided for the first decode call."
        )

        recon = self.tokenizer.decode(z, inp_shape=input_shape, clamp=True)
        self._last_x_shape = None

        if not torch.is_tensor(recon):
            recon = recon[0]
        return recon
