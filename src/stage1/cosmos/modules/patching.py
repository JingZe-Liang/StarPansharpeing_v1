# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The patcher and unpatcher implementation for 2D and 3D data.

The idea of Haar wavelet is to compute LL, LH, HL, HH component as two 1D convolutions.
One on the rows and one on the columns.
For example, in 1D signal, we have [a, b], then the low-freq compoenent is [a + b] / 2 and high-freq is [a - b] / 2.
We can use a 1D convolution with kernel [1, 1] and stride 2 to represent the L component.
For H component, we can use a 1D convolution with kernel [1, -1] and stride 2.
Although in principle, we typically only do additional Haar wavelet over the LL component. But here we do it for all
   as we need to support downsampling for more than 2x.
For example, 4x downsampling can be done by 2x Haar and additional 2x Haar, and the shape would be.
   [3, 256, 256] -> [12, 128, 128] -> [48, 64, 64]
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from fvcore.nn import parameter_count
from jaxtyping import Float
from loguru import logger
from timm.layers import PatchEmbed, create_conv2d
from torch import Tensor

from .blocks import AdaptiveInputConvLayer, AdaptiveOutputConvLayer

_WAVELETS = {
    "haar": torch.tensor([0.7071067811865476, 0.7071067811865476]),
    "rearrange": torch.tensor([1.0, 1.0]),
}
_PERSISTENT = False
_DTYPE_INTERMEDIATE = torch.float32


def _sincos_channel_index_embedding(
    n_channels: int,
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    base: float = 10000.0,
) -> Tensor:
    if dim <= 0:
        raise ValueError(f"{dim=} must be > 0")
    if n_channels <= 0:
        raise ValueError(f"{n_channels=} must be > 0")
    if base <= 0:
        raise ValueError(f"{base=} must be > 0")

    pos = torch.arange(n_channels, device=device, dtype=dtype)
    half_dim = (dim + 1) // 2
    div_term = torch.exp(torch.arange(half_dim, device=device, dtype=dtype) * (-math.log(base) / dim))

    emb = torch.zeros((n_channels, dim), device=device, dtype=dtype)
    sin_inp = pos[:, None] * div_term[None, :]
    emb[:, 0::2] = torch.sin(sin_inp)[:, : emb[:, 0::2].shape[1]]
    if dim > 1:
        emb[:, 1::2] = torch.cos(sin_inp)[:, : emb[:, 1::2].shape[1]]
    return emb


def _compute_resize_matrix(
    old_size: tuple[int, int],
    new_size: tuple[int, int],
    *,
    interpolation: str = "bicubic",
    antialias: bool = True,
    device: torch.device,
    dtype: torch.dtype = _DTYPE_INTERMEDIATE,
) -> Tensor:
    """Compute a resize matrix for PI-resize (Moore-Penrose) patch kernel resampling."""
    old_h, old_w = old_size
    new_h, new_w = new_size
    old_total = old_h * old_w
    new_total = new_h * new_w

    eye_matrix = torch.eye(old_total, device=device, dtype=dtype)
    basis = eye_matrix.reshape(old_total, 1, old_h, old_w)
    resized_basis = F.interpolate(
        basis,
        size=new_size,
        mode=interpolation,
        antialias=antialias,
        align_corners=False,
    )
    resize_matrix = resized_basis.squeeze(1).permute(1, 2, 0).reshape(new_total, old_total)
    return resize_matrix


def _apply_pinv_resampling(
    kernel: Tensor,
    pinv_matrix: Tensor,
    new_size: tuple[int, int],
    *,
    intermediate_dtype: torch.dtype = _DTYPE_INTERMEDIATE,
) -> Tensor:
    """Resample conv kernel weights by multiplying pseudoinverse matrix (PI-resize)."""
    c_out, c_in, _, _ = kernel.shape
    orig_dtype = kernel.dtype
    k_flat = kernel.reshape(c_out, c_in, -1).to(dtype=intermediate_dtype)
    pinv_matrix = pinv_matrix.to(dtype=intermediate_dtype)
    k_new = k_flat @ pinv_matrix
    return k_new.reshape(c_out, c_in, *new_size).to(dtype=orig_dtype)


class PatchKernelResamplerFixedOrigSize(nn.Module):
    """Cache PI-resize pseudoinverse matrices for a fixed orig patch kernel size."""

    def __init__(
        self,
        orig_size: tuple[int, int],
        *,
        interpolation: str = "bicubic",
        antialias: bool = True,
    ) -> None:
        super().__init__()
        self.orig_size = orig_size
        self.interpolation = interpolation
        self.antialias = antialias
        self._pinv_cache_map: dict[tuple[int, int], str] = {}

    def _get_or_create_pinv_matrix(
        self,
        new_size: tuple[int, int],
        *,
        device: torch.device,
        dtype: torch.dtype = _DTYPE_INTERMEDIATE,
    ) -> Tensor:
        buffer_name = self._pinv_cache_map.get(new_size)
        if buffer_name and hasattr(self, buffer_name):
            pinv_matrix = getattr(self, buffer_name)
            if pinv_matrix.device == device and pinv_matrix.dtype == dtype:
                return pinv_matrix

        resize_mat = _compute_resize_matrix(
            self.orig_size,
            new_size,
            interpolation=self.interpolation,
            antialias=self.antialias,
            device=device,
            dtype=dtype,
        )
        pinv_matrix = torch.linalg.pinv(resize_mat)
        buffer_name = f"pinv_{self.orig_size[0]}x{self.orig_size[1]}__{new_size[0]}x{new_size[1]}"
        if hasattr(self, buffer_name):
            delattr(self, buffer_name)
        self.register_buffer(buffer_name, pinv_matrix, persistent=_PERSISTENT)
        self._pinv_cache_map[new_size] = buffer_name
        return pinv_matrix

    def forward(self, kernel: Tensor, new_size: tuple[int, int]) -> Tensor:
        if tuple(kernel.shape[-2:]) != self.orig_size:
            raise ValueError(f"Expected orig kernel size {self.orig_size}, got {tuple(kernel.shape[-2:])}")
        if new_size == self.orig_size:
            return kernel
        pinv_matrix = self._get_or_create_pinv_matrix(new_size, device=kernel.device)
        return _apply_pinv_resampling(kernel, pinv_matrix, new_size)


class Patcher(torch.nn.Module):
    """A module to convert image tensors into patches using torch operations.

    The main difference from `class Patching` is that this module implements
    all operations using torch, rather than python or numpy, for efficiency purpose.

    It's bit-wise identical to the Patching module outputs, with the added
    benefit of being torch.jit scriptable.
    """

    def __init__(self, patch_size=1, patch_method="haar"):
        super().__init__()
        self.patch_size = patch_size
        self.patch_method = patch_method
        self.register_buffer("wavelets", _WAVELETS[patch_method], persistent=_PERSISTENT)
        self.range = range(int(torch.log2(torch.tensor(self.patch_size)).item()))
        self.register_buffer(
            "_arange",
            torch.arange(_WAVELETS[patch_method].shape[0]),
            persistent=_PERSISTENT,
        )
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        if self.patch_method == "haar":
            return self._haar(x)
        elif self.patch_method == "rearrange":
            return self._arrange(x)
        else:
            raise ValueError("Unknown patch method: " + self.patch_method)

    def _dwt(self, x, mode="reflect", rescale=False):
        dtype = x.dtype
        h = self.wavelets

        n = h.shape[0]
        g = x.shape[1]
        hl = h.flip(0).reshape(1, 1, -1).repeat(g, 1, 1)
        hh = (h * ((-1) ** self._arange)).reshape(1, 1, -1).repeat(g, 1, 1)
        hh = hh.to(dtype=dtype)
        hl = hl.to(dtype=dtype)

        x = F.pad(x, pad=(n - 2, n - 1, n - 2, n - 1), mode=mode).to(dtype)
        xl = F.conv2d(x, hl.unsqueeze(2), groups=g, stride=(1, 2))
        xh = F.conv2d(x, hh.unsqueeze(2), groups=g, stride=(1, 2))
        xll = F.conv2d(xl, hl.unsqueeze(3), groups=g, stride=(2, 1))
        xlh = F.conv2d(xl, hh.unsqueeze(3), groups=g, stride=(2, 1))
        xhl = F.conv2d(xh, hl.unsqueeze(3), groups=g, stride=(2, 1))
        xhh = F.conv2d(xh, hh.unsqueeze(3), groups=g, stride=(2, 1))

        out = torch.cat([xll, xlh, xhl, xhh], dim=1)
        if rescale:
            out = out / 2
        return out

    def _haar(self, x):
        for _ in self.range:
            x = self._dwt(x, rescale=True)
        return x

    def _arrange(self, x):
        x = rearrange(
            x,
            "b c (h p1) (w p2) -> b (c p1 p2) h w",
            p1=self.patch_size,
            p2=self.patch_size,
        ).contiguous()
        return x


class Patcher3D(Patcher):
    """A 3D discrete wavelet transform for video data, expects 5D tensor, i.e. a batch of videos."""

    def __init__(self, patch_size=1, patch_method="haar"):
        super().__init__(patch_method=patch_method, patch_size=patch_size)
        self.register_buffer(
            "patch_size_buffer",
            patch_size * torch.ones([1], dtype=torch.int32),
            persistent=_PERSISTENT,
        )

    def _dwt(self, x, wavelet, mode="reflect", rescale=False):
        dtype = x.dtype
        h = self.wavelets

        n = h.shape[0]
        g = x.shape[1]
        hl = h.flip(0).reshape(1, 1, -1).repeat(g, 1, 1)
        hh = (h * ((-1) ** self._arange)).reshape(1, 1, -1).repeat(g, 1, 1)
        hh = hh.to(dtype=dtype)
        hl = hl.to(dtype=dtype)

        # Handles temporal axis.
        x = F.pad(x, pad=(max(0, n - 2), n - 1, n - 2, n - 1, n - 2, n - 1), mode=mode).to(dtype)
        xl = F.conv3d(x, hl.unsqueeze(3).unsqueeze(4), groups=g, stride=(2, 1, 1))
        xh = F.conv3d(x, hh.unsqueeze(3).unsqueeze(4), groups=g, stride=(2, 1, 1))

        # Handles spatial axes.
        xll = F.conv3d(xl, hl.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))
        xlh = F.conv3d(xl, hh.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))
        xhl = F.conv3d(xh, hl.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))
        xhh = F.conv3d(xh, hh.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))

        xlll = F.conv3d(xll, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xllh = F.conv3d(xll, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xlhl = F.conv3d(xlh, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xlhh = F.conv3d(xlh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhll = F.conv3d(xhl, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhlh = F.conv3d(xhl, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhhl = F.conv3d(xhh, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhhh = F.conv3d(xhh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))

        out = torch.cat([xlll, xllh, xlhl, xlhh, xhll, xhlh, xhhl, xhhh], dim=1)
        if rescale:
            out = out / (2 * torch.sqrt(torch.tensor(2.0)))
        return out

    def _haar(self, x):
        xi, xv = torch.split(x, [1, x.shape[2] - 1], dim=2)
        x = torch.cat([xi.repeat_interleave(self.patch_size, dim=2), xv], dim=2)
        for _ in self.range:
            x = self._dwt(x, "haar", rescale=True)
        return x

    def _arrange(self, x):
        xi, xv = torch.split(x, [1, x.shape[2] - 1], dim=2)
        x = torch.cat([xi.repeat_interleave(self.patch_size, dim=2), xv], dim=2)
        x = rearrange().contiguous()
        return x


class UnPatcher(torch.nn.Module):
    """A module to convert patches into image tensors using torch operations.

    The main difference from `class Unpatching` is that this module implements
    all operations using torch, rather than python or numpy, for efficiency purpose.

    It's bit-wise identical to the Unpatching module outputs, with the added
    benefit of being torch.jit scriptable.
            x,
            "b c (t p1) (h p2) (w p3) -> b (c p1 p2 p3) t h w",
            p1=self.patch_size,
            p2=self.patch_size,
            p3=self.patch_size,
    """

    def __init__(self, patch_size=1, patch_method="haar"):
        super().__init__()
        self.patch_size = patch_size
        self.patch_method = patch_method
        self.register_buffer("wavelets", _WAVELETS[patch_method], persistent=_PERSISTENT)
        self.range = range(int(torch.log2(torch.tensor(self.patch_size)).item()))
        self.register_buffer(
            "_arange",
            torch.arange(_WAVELETS[patch_method].shape[0]),
            persistent=_PERSISTENT,
        )
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        if self.patch_method == "haar":
            return self._ihaar(x)
        elif self.patch_method == "rearrange":
            return self._iarrange(x)
        else:
            raise ValueError("Unknown patch method: " + self.patch_method)

    def _idwt(self, x, wavelet="haar", mode="reflect", rescale=False):
        dtype = x.dtype
        h = self.wavelets
        n = h.shape[0]

        g = x.shape[1] // 4
        hl = h.flip([0]).reshape(1, 1, -1).repeat([g, 1, 1])
        hh = (h * ((-1) ** self._arange)).reshape(1, 1, -1).repeat(g, 1, 1)
        hh = hh.to(dtype=dtype)
        hl = hl.to(dtype=dtype)

        xll, xlh, xhl, xhh = torch.chunk(x.to(dtype), 4, dim=1)

        # Inverse transform.
        yl = torch.nn.functional.conv_transpose2d(xll, hl.unsqueeze(3), groups=g, stride=(2, 1), padding=(n - 2, 0))
        yl += torch.nn.functional.conv_transpose2d(xlh, hh.unsqueeze(3), groups=g, stride=(2, 1), padding=(n - 2, 0))
        yh = torch.nn.functional.conv_transpose2d(xhl, hl.unsqueeze(3), groups=g, stride=(2, 1), padding=(n - 2, 0))
        yh += torch.nn.functional.conv_transpose2d(xhh, hh.unsqueeze(3), groups=g, stride=(2, 1), padding=(n - 2, 0))
        y = torch.nn.functional.conv_transpose2d(yl, hl.unsqueeze(2), groups=g, stride=(1, 2), padding=(0, n - 2))
        y += torch.nn.functional.conv_transpose2d(yh, hh.unsqueeze(2), groups=g, stride=(1, 2), padding=(0, n - 2))

        if rescale:
            y = y * 2
        return y

    def _ihaar(self, x):
        for _ in self.range:
            x = self._idwt(x, "haar", rescale=True)
        return x

    def _iarrange(self, x):
        x = rearrange(
            x,
            "b (c p1 p2) h w -> b c (h p1) (w p2)",
            p1=self.patch_size,
            p2=self.patch_size,
        )
        return x


class UnPatcher3D(UnPatcher):
    """A 3D inverse discrete wavelet transform for video wavelet decompositions."""

    def __init__(self, patch_size=1, patch_method="haar"):
        super().__init__(patch_method=patch_method, patch_size=patch_size)

    def _idwt(self, x, wavelet="haar", mode="reflect", rescale=False):
        dtype = x.dtype
        h = self.wavelets
        h.shape[0]

        g = x.shape[1] // 8  # split into 8 spatio-temporal filtered tesnors.
        hl = h.flip([0]).reshape(1, 1, -1).repeat([g, 1, 1])
        hh = (h * ((-1) ** self._arange)).reshape(1, 1, -1).repeat(g, 1, 1)
        hl = hl.to(dtype=dtype)
        hh = hh.to(dtype=dtype)

        xlll, xllh, xlhl, xlhh, xhll, xhlh, xhhl, xhhh = torch.chunk(x, 8, dim=1)

        # Height height transposed convolutions.
        xll = F.conv_transpose3d(xlll, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xll += F.conv_transpose3d(xllh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))

        xlh = F.conv_transpose3d(xlhl, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xlh += F.conv_transpose3d(xlhh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))

        xhl = F.conv_transpose3d(xhll, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhl += F.conv_transpose3d(xhlh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))

        xhh = F.conv_transpose3d(xhhl, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhh += F.conv_transpose3d(xhhh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))

        # Handles width transposed convolutions.
        xl = F.conv_transpose3d(xll, hl.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))
        xl += F.conv_transpose3d(xlh, hh.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))
        xh = F.conv_transpose3d(xhl, hl.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))
        xh += F.conv_transpose3d(xhh, hh.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))

        # Handles time axis transposed convolutions.
        x = F.conv_transpose3d(xl, hl.unsqueeze(3).unsqueeze(4), groups=g, stride=(2, 1, 1))
        x += F.conv_transpose3d(xh, hh.unsqueeze(3).unsqueeze(4), groups=g, stride=(2, 1, 1))

        if rescale:
            x = x * (2 * torch.sqrt(torch.tensor(2.0)))
        return x

    def _ihaar(self, x):
        for _ in self.range:
            x = self._idwt(x, "haar", rescale=True)
        x = x[:, :, self.patch_size - 1 :, ...]
        return x

    def _iarrange(self, x):
        x = rearrange(
            x,
            "b (c p1 p2 p3) t h w -> b c (t p1) (h p2) (w p3)",
            p1=self.patch_size,
            p2=self.patch_size,
            p3=self.patch_size,
        )
        x = x[:, :, self.patch_size - 1 :, ...]
        return x


def create_unpatcher(
    chan: int,
    out_chan: int,
    patch_size: int = 1,
    lora_rank_ratio: int = 0,
    kernel_size=3,
    **conv_kwargs,
):
    out_patch_chan = out_chan * (patch_size**2)
    is_lora = lora_rank_ratio > 0
    if is_lora:
        unpatcher_conv = nn.Sequential(
            create_conv2d(chan, chan // lora_rank_ratio, 1),
            create_conv2d(chan // lora_rank_ratio, out_patch_chan, kernel_size, **conv_kwargs),
        )
    else:
        unpatcher_conv = create_conv2d(chan, out_patch_chan, kernel_size, **conv_kwargs)
    rearranger = Rearrange("bs (c p1 p2) h w -> bs c (h p1) (w p2)", p1=patch_size, p2=patch_size)
    patcher = nn.Sequential(unpatcher_conv, rearranger)
    return patcher


class AdaptivePatchEmbedding(PatchEmbed):
    def __init__(self, in_chans, embed_dim, patch_size: int, mode: str = "interp", **kwargs):
        super().__init__(
            in_chans=in_chans,
            embed_dim=embed_dim,
            patch_size=patch_size,
            strict_img_size=False,
            **kwargs,
        )
        self.proj = AdaptiveInputConvLayer(
            in_chans,
            embed_dim,
            patch_size,
            patch_size,
            padding=0,
            use_bias=kwargs.get("bias", True),
            mode=mode,  # type: ignore
        )


class AdaptiveProgressivePatchEmbedding(PatchEmbed):
    def __init__(
        self,
        in_chans,
        embed_dim,
        progressive_dims: list[int],
        patch_size: int,
        adaptive_mode: str = "interp",
        **kwargs,
    ):
        super().__init__(
            in_chans=in_chans,
            embed_dim=embed_dim,
            patch_size=patch_size,
            strict_img_size=False,
            **kwargs,
        )
        n_scales = math.log2(patch_size)
        assert n_scales.is_integer(), "Patch size must be power of 2."
        self.n_scales = int(n_scales)

        self.progressive_dims = progressive_dims

        # in_chans -> p_dim[0] -> p_dim[1] -> ... -> embed_dim
        #          2x down     2x down     -> ... -> (log2 patch_size)x down
        # log2(patch_size) == len(progressive_dims) + 1
        assert len(self.progressive_dims) == self.n_scales - 1, (
            f"Length of progressive_dims must be {self.n_scales - 1}, but got {len(self.progressive_dims)}"
        )

        # Override self.proj
        dims = [in_chans] + self.progressive_dims + [embed_dim]
        patchers = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i == 0:
                patcher = AdaptiveInputConvLayer(
                    in_dim,
                    out_dim,
                    2,
                    2,
                    padding=0,
                    use_bias=kwargs.get("bias", True),
                    mode=adaptive_mode,  # ['slice', 'interp']
                )
            else:
                patcher = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2, padding=0)
            patchers.append(patcher)
            logger.debug(
                f"[AdaptiveProgressivePatchEmbedding] Patch embedding stage {i}: {in_dim} -> {out_dim} with 2x downsample, "
                f"params: {parameter_count(patcher)[''] / 1024}k"
            )
        self.proj = nn.Sequential(*patchers)


class AdaptiveProgressivePatchUnembedding(nn.Module):
    def __init__(
        self,
        in_chans,
        out_chans,
        progressive_dims: list[int],
        patch_size: int,
        adaptive_mode: str = "interp",
    ):
        super().__init__()
        self.patch_size = patch_size
        n_scales = math.log2(patch_size)
        assert n_scales.is_integer(), "Patch size must be power of 2."
        self.n_scales = int(n_scales)

        self.progressive_dims = progressive_dims

        # in_chans -> p_dim[0] -> p_dim[1] -> ... -> embed_dim
        #          2x up     2x up     -> ... -> (log2 patch_size)x up
        # log2(patch_size) == len(progressive_dims) + 1
        assert len(self.progressive_dims) == self.n_scales - 1, (
            f"Length of progressive_dims must be {self.n_scales - 1}, but got {len(self.progressive_dims)}"
        )

        dims = [in_chans] + self.progressive_dims + [out_chans]
        unpatchers = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # pixel shuffle
            unpatcher = nn.Linear(in_dim, out_dim * 4)
            unpatchers.append(unpatcher)
            logger.debug(
                f"[AdaptiveProgressivePatchUnembedding] Patch Unembedding stage {i}: {in_dim} -> {out_dim} with 2x upsample, "
                f"params: {str(parameter_count(unpatcher)[''] / 1024)}k"
            )

        adaptive_out = AdaptiveOutputConvLayer(
            out_dim,
            out_dim,
            1,
            1,
            padding=0,
            use_bias=True,
            mode=adaptive_mode,  # ['slice', 'interp']
        )
        unpatchers.append(adaptive_out)
        logger.debug(
            f"[AdaptiveProgressivePatchUnembedding] Final adaptive output conv layer: {out_dim} -> {out_dim}, "
            f"params: {parameter_count(adaptive_out)[''] / 1024}k"
        )
        self.unpatchers = nn.ModuleList(unpatchers)

    def forward(self, x: Float[Tensor, "b l c"], out_shape: torch.Size | tuple):
        # out_shape: (b, c, h, w) or (c, h, w)
        c, h, w = out_shape[-3:]
        xh, xw = h // self.patch_size, w // self.patch_size
        x = rearrange(x, "b (h w) c -> b h w c", h=xh, w=xw)

        # np = nw = patch_size
        for layer in self.unpatchers[:-1]:
            x = layer(x)
            x = rearrange(x, "b h w (c p1 p2) -> b (h p1) (w p2) c", p1=2, p2=2)

        x = x.permute(0, -1, 1, 2)  # b, c, h, w
        x = self.unpatchers[-1](x, c)  # to out channels
        return x

    def init_weights(self, zero_out_adaptive_layer=False):
        patcher_backbone = self.unpatchers[:-1]
        patcher_adaptive_last = self.unpatchers[-1]
        for m in patcher_backbone.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # last adaptive layer
        patcher_adaptive_last.init_weights(zero_out=zero_out_adaptive_layer)


class SiTokMAPEPatchEmbedding(nn.Module):
    """SiTok-style spectrum-independent patch embedding with MAPE (multi-scale kernel bank + PI-resize)."""

    def __init__(
        self,
        *,
        img_size: tuple[int, int] | None = None,
        patch_size: int = 16,
        embed_dim: int = 768,
        base_patch_sizes: list[int] | tuple[int, ...] = (16, 32, 64),
        flatten: bool = True,
        bias: bool = True,
        strict_img_size: bool = False,
        dynamic_img_pad: bool = False,
        interpolation: str = "bicubic",
        antialias: bool = True,
        channel_embed_scale: float = 1.0,
        channel_embed_base: float = 10000.0,
        norm_layer: type[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if patch_size <= 0:
            raise ValueError(f"{patch_size=} must be > 0")
        if embed_dim <= 0:
            raise ValueError(f"{embed_dim=} must be > 0")
        if not base_patch_sizes:
            raise ValueError("base_patch_sizes must be non-empty")
        if any(p <= 0 for p in base_patch_sizes):
            raise ValueError(f"Invalid base_patch_sizes: {base_patch_sizes}")
        if channel_embed_base <= 0:
            raise ValueError(f"{channel_embed_base=} must be > 0")

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.flatten = flatten
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad
        self.interpolation = interpolation
        self.antialias = antialias
        self.channel_embed_scale = float(channel_embed_scale)
        self.channel_embed_base = float(channel_embed_base)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        self.base_patch_sizes = sorted({int(p) for p in base_patch_sizes})
        self.kernel_bank = nn.ParameterDict(
            {str(p): nn.Parameter(torch.empty(embed_dim, 1, p, p)) for p in self.base_patch_sizes}
        )
        self.bias = nn.Parameter(torch.zeros(embed_dim)) if bias else None
        for p in self.base_patch_sizes:
            nn.init.trunc_normal_(self.kernel_bank[str(p)], std=0.02)

        self._resamplers = nn.ModuleDict(
            {
                str(p): PatchKernelResamplerFixedOrigSize((p, p), interpolation=interpolation, antialias=antialias)
                for p in self.base_patch_sizes
            }
        )

        self.img_size = img_size
        if img_size is None:
            self.grid_size = None
            self.num_patches = None
        else:
            self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
            self.num_patches = self.grid_size[0] * self.grid_size[1]

    def _choose_base_patch_size(self, patch_size: int) -> int:
        return min(self.base_patch_sizes, key=lambda p: abs(p - patch_size))

    def get_kernel(self, patch_size: int, *, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor | None]:
        base_p = self._choose_base_patch_size(patch_size)
        w = self.kernel_bank[str(base_p)].to(device=device, dtype=dtype)
        if base_p != patch_size:
            resampler = self._resamplers[str(base_p)]
            w = resampler(w, (patch_size, patch_size))
        b = self.bias
        if b is not None:
            b = b.to(device=device, dtype=dtype)
        return w, b

    def forward(self, x: Tensor) -> Tensor:
        bsz, in_chans, h, w = x.shape
        if self.img_size is not None and self.strict_img_size:
            if (h, w) != self.img_size:
                raise ValueError(f"Input size {(h, w)} doesn't match model {self.img_size}")

        patch_size = self.patch_size
        if not self.dynamic_img_pad:
            if (h % patch_size) != 0 or (w % patch_size) != 0:
                raise ValueError(f"Input {(h, w)} must be divisible by patch_size {patch_size}")
        else:
            pad_h = (patch_size - h % patch_size) % patch_size
            pad_w = (patch_size - w % patch_size) % patch_size
            x = F.pad(x, (0, pad_w, 0, pad_h))

        w_shared, b_shared = self.get_kernel(patch_size, device=x.device, dtype=x.dtype)
        w_rep = w_shared.repeat(in_chans, 1, 1, 1)
        if b_shared is None:
            b_rep = None
        else:
            b_rep = b_shared.repeat(in_chans)

        y = F.conv2d(x, w_rep, b_rep, stride=patch_size, padding=0, groups=in_chans)
        y = y.reshape(bsz, in_chans, self.embed_dim, y.shape[-2], y.shape[-1])

        if self.channel_embed_scale != 0:
            ch_emb = _sincos_channel_index_embedding(
                in_chans,
                self.embed_dim,
                device=y.device,
                dtype=y.dtype,
                base=self.channel_embed_base,
            )
            y = y + self.channel_embed_scale * ch_emb[None, :, :, None, None]

        if not self.flatten:
            raise ValueError("SiTokMAPEPatchEmbedding currently expects flatten=True (NLC tokens).")
        y = rearrange(y, "b c d gh gw -> b (c gh gw) d")
        y = self.norm(y)
        return y


# * --- Test --- #


def test_adaptive_progressive_unpatcher():
    batch_size = 2
    in_chans = 300
    out_chans = 300
    progressive_dims = [320, 448, 512]
    embed_dim = 768
    patch_size = 16
    height = 256
    width = 256

    x = torch.randn(batch_size, 301, height, width)

    patcher = AdaptiveProgressivePatchEmbedding(
        in_chans,
        embed_dim=embed_dim,
        progressive_dims=progressive_dims,
        patch_size=patch_size,
        adaptive_mode="interp",
    )
    unpatcher = AdaptiveProgressivePatchUnembedding(
        in_chans=embed_dim,
        out_chans=out_chans,
        progressive_dims=progressive_dims[::-1],
        patch_size=patch_size,
        adaptive_mode="interp",
    )

    patches = patcher(x)
    print("Patches shape:", patches.shape)

    unpatched = unpatcher(patches, out_shape=(batch_size, 301, height, width))
    print("Unpatched shape:", unpatched.shape)


if __name__ == "__main__":
    """
    LOVELY_TENSORS=1 python -m src.stage1.cosmos.modules.patching
    """
    with logger.catch():
        test_adaptive_progressive_unpatcher()
