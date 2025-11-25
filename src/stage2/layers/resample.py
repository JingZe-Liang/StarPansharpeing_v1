from typing import Any, Literal, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast

from .blocks import ResidualBlock
from .conv import ConvLayer


def resize(
    x: torch.Tensor,
    size: Optional[Any] = None,
    scale_factor: Optional[list[float]] = None,
    mode: str = "bicubic",
    align_corners: Optional[bool] = False,
) -> torch.Tensor:
    if mode in {"bilinear", "bicubic"}:
        return F.interpolate(
            x,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
        )
    elif mode in {"nearest", "area"}:
        return F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode)
    else:
        raise NotImplementedError(f"resize(mode={mode}) not implemented.")


def val2list(val: Union[int, tuple, list], n: int) -> list:
    if isinstance(val, int):
        return [val] * n
    elif isinstance(val, (tuple, list)):
        assert len(val) == n
        return list(val)
    else:
        raise ValueError(f"val2list only accepts int/tuple/list, got {type(val)}")


class UpSampleLayer(nn.Module):
    def __init__(
        self,
        mode="bicubic",
        size: Optional[int | tuple[int, int] | list[int]] = None,
        factor=2,
        align_corners=False,
        **_kwargs,
    ):
        super(UpSampleLayer, self).__init__()
        self.mode = mode
        self.size = val2list(size, 2) if size is not None else None
        self.factor = None if self.size is not None else factor
        self.align_corners = align_corners

    @autocast(device_type="cuda", enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (self.size is not None and tuple(x.shape[-2:]) == self.size) or self.factor == 1:
            return x
        if x.dtype in [torch.float16, torch.bfloat16]:
            x = x.float()
        return resize(x, self.size, self.factor, self.mode, self.align_corners)


class ConvPixelUnshuffleDownSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        factor: int,
        **_kwargs,
    ):
        super().__init__()
        self.factor = factor
        out_ratio = factor**2
        assert out_channels % out_ratio == 0
        self.conv = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels // out_ratio,
            kernel_size=kernel_size,
            use_bias=True,
            norm=None,
            act_func=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = F.pixel_unshuffle(x, self.factor)
        return x


class PixelUnshuffleChannelAveragingDownSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        assert in_channels * factor**2 % out_channels == 0
        self.group_size = in_channels * factor**2 // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pixel_unshuffle(x, self.factor)
        B, C, H, W = x.shape
        x = x.view(B, self.out_channels, self.group_size, H, W)
        x = x.mean(dim=2)
        return x


class ConvPixelShuffleUpSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        factor: int,
        **_kwargs,
    ):
        super().__init__()
        self.factor = factor
        out_ratio = factor**2
        self.conv = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels * out_ratio,
            kernel_size=kernel_size,
            use_bias=True,
            norm=None,
            act_func=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = F.pixel_shuffle(x, self.factor)
        return x


class InterpolateConvUpSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        factor: int,
        mode: str = "nearest",
        **_kwargs,
    ) -> None:
        super().__init__()
        self.factor = factor
        self.mode = mode
        self.conv = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            use_bias=True,
            norm=None,
            act_func=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(x, scale_factor=self.factor, mode=self.mode)
        x = self.conv(x)
        return x


class ChannelDuplicatingPixelShuffleUpSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
        **_kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        assert out_channels * factor**2 % in_channels == 0
        self.repeats = out_channels * factor**2 // in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        x = F.pixel_shuffle(x, self.factor)
        return x


UPSAMPLE_LAYERS = {
    "upsample": UpSampleLayer,
    "conv_pixelshuffle": ConvPixelShuffleUpSampleLayer,
    "interpolate_conv": InterpolateConvUpSampleLayer,
}
DOWNSAMPLE_LAYERS = {
    "conv_pixelunshuffle": ConvPixelUnshuffleDownSampleLayer,
}

# *==============================================================
# * interface
# *==============================================================


def create_upsample_layer(
    in_channels: int,
    out_channels: int,
    upsample_type: Literal[
        "upsample",
        "conv_pixelshuffle",
        "interpolate_conv",
        "channelduplicating_pixelshuffle",
    ],
    factor: int,
    kernel_size: int,
    mode: str = "bicubic",
    size: Optional[int | tuple[int, int] | list[int]] = None,
    align_corners: Optional[bool] = False,
    shortcut: Optional[str] = None,
):
    up_layer = UPSAMPLE_LAYERS.get(upsample_type, None)
    if up_layer is None:
        raise NotImplementedError(f"upsample_type {upsample_type} not implemented.")

    up_layer_ins = up_layer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        factor=factor,
        mode=mode,
        size=size,
        align_corners=align_corners,
    )

    if shortcut == "duplicating":
        shortcut_block = ChannelDuplicatingPixelShuffleUpSampleLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            factor=factor,
        )
        up_layer_ins = ResidualBlock(up_layer_ins, shortcut_block)

    return up_layer_ins


def create_downsample_layer(
    in_channels: int,
    out_channels: int,
    downsample_type: Literal[
        "conv_pixelunshuffle",
        "pixelunshuffle_channelaveraging",
    ],
    factor: int,
    kernel_size: int,
    shortcut: Optional[str] = None,
):
    """
    Create a downsample layer based on the specified type.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    downsample_type : Literal["conv_pixelunshuffle", "pixelunshuffle_channelaveraging"]
        Type of downsample layer to create
    factor : int
        Downsampling factor
    kernel_size : int
        Kernel size for convolutional layers

    Returns
    -------
    nn.Module
        The created downsample layer

    Raises
    ------
    NotImplementedError
        If the specified downsample_type is not implemented
    """
    down_layer = DOWNSAMPLE_LAYERS.get(downsample_type, None)
    if down_layer is None:
        raise NotImplementedError(f"downsample_type {downsample_type} not implemented.")

    down_layer_ins = down_layer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        factor=factor,
    )

    if shortcut == "averaging":
        shortcut_block = PixelUnshuffleChannelAveragingDownSampleLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            factor=factor,
        )
        down_layer_ins = ResidualBlock(down_layer_ins, shortcut_block)

    return down_layer_ins
