from typing import Any, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from loguru import logger
from timm.layers import create_norm_act_layer

from .blocks import ResidualBlock, get_same_padding
from .utils import Normalize

# * --- Upsample and Downsample --- #


def resample_norm_keep(x, x_resampled):
    return x_resampled * torch.norm(x) / torch.norm(x_resampled)


class UpsampleRepeatConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        padding_mode: str = "zeros",
        norm_type: str | None = None,
        norm_keep: bool = False,
        interp_type: str = "nearest_interp",  # NOTE: xy_repeat originally
    ):
        super().__init__()
        self.norm_keep = norm_keep
        self.interp_type = interp_type
        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode=padding_mode,
        )
        if interp_type == "transpose_conv":
            self.trans_conv = nn.ConvTranspose2d(
                in_channels,
                in_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode=padding_mode,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.interp_type == "xy_repeat":
            x = x.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
        elif self.interp_type == "nearest_interp":
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        elif self.interp_type == "transpose_conv":
            x = self.trans_conv(x)
        else:
            raise ValueError(f"[Upsample]: Unknown interp_type: {self.interp_type}")

        x_resp = self.conv(x)

        if self.norm_keep:
            x_resp = resample_norm_keep(x, x_resp)
        return x_resp


class DownsamplePadConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        padding_mode: str = "constant",
        padding_in_conv: bool = False,
        norm_type: Optional[str] = None,
        norm_keep: bool = False,
        use_conv: bool = True,
    ):
        # Zihan NOTE: using pad (left and right) align the center of the pixel when downsampling
        # but (may?) cause the boundary artifact when upsampling

        super().__init__()
        self.padding_mode = padding_mode
        self.padding_in_conv = padding_in_conv
        self.norm_keep = norm_keep
        self.use_conv = use_conv

        if self.use_conv:
            padding = 1
            # Not pad in conv, then mannually pad the image
            if not padding_in_conv:
                padding_mode = "zeros"  # 'zeros' as default in sd vae
                padding = 0

            self.conv = nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=2,
                padding=padding,
                padding_mode=padding_mode,
            )
        else:
            self.conv = nn.AvgPool2d(kernel_size=2, stride=2)
            logger.debug(f"[Downsample]: using avg pool downsample instead of conv")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LDM VAE also use the unsymmetric padding
        # cosmos manually pad
        if not self.padding_in_conv and self.use_conv:
            # to align on the center of the downsampled image pixels
            pad = (0, 1, 0, 1)  # NOTE: lower and righter pads, why? inductive bias?
            if self.padding_mode not in ("constant", "zeros"):
                x = F.pad(x, pad, mode=self.padding_mode)
            else:
                x = F.pad(x, pad, mode="constant", value=0)

        x_resp = self.conv(x)

        if self.norm_keep:
            x_resp = resample_norm_keep(x, x_resp)

        return x_resp


class ConvPixelUnshuffleDownSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        factor: int,
        padding_mode: str = "zeros",
        norm_keep: bool = False,
    ):
        super().__init__()
        self.factor = factor
        self.norm_keep = norm_keep
        out_ratio = factor**2
        assert out_channels % out_ratio == 0
        self.conv = nn.Conv2d(
            in_channels,
            out_channels // out_ratio,
            kernel_size,
            padding=get_same_padding(kernel_size),
            padding_mode=padding_mode,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = F.pixel_unshuffle(x, self.factor)
        if self.norm_keep:
            x = resample_norm_keep(x, x)
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
        # hidden = out_channels * group_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pixel_unshuffle(
            x, self.factor
        )  # c * factor ** 2 -> hidden = out_c * group_size
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
        padding_mode: str = "zeros",
        norm_type: Optional[str] = None,
    ):
        super().__init__()
        self.factor = factor
        out_ratio = factor**2
        self.norm = Normalize(
            in_channels=in_channels, num_groups=32, norm_type=norm_type
        )
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * out_ratio,
            kernel_size,
            padding=get_same_padding(kernel_size),
            padding_mode=padding_mode,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(self.norm(x))
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
        padding_mode: str = "zeros",
        norm_type: Optional[str] = None,
        norm_keep: bool = False,
    ) -> None:
        super().__init__()
        self.factor = factor
        self.mode = mode
        self.norm_keep = norm_keep
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            1,
            padding=get_same_padding(kernel_size),
            padding_mode=padding_mode,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(x, scale_factor=self.factor, mode=self.mode)
        x_resp = self.conv(x)
        if self.norm_keep:
            x_resp = resample_norm_keep(x, x_resp)
        return x_resp


class ChannelDuplicatingPixelUnshuffleUpSampleLayer(nn.Module):
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
        assert out_channels * factor**2 % in_channels == 0
        self.repeats = out_channels * factor**2 // in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        x = F.pixel_shuffle(x, self.factor)
        return x


# * --- Mingtok functional --- #


class MingtokDownsampleShortCut(nn.Module):
    """
    Mingtok encoder output projection layer
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_act: tuple[str, str] = ("layernorm", "gelu"),
    ):
        super().__init__()
        self.out_channels = out_channels
        self.norm_act = create_norm_act_layer(norm_act[0], in_channels, norm_act[1])
        self.proj_out = nn.Linear(in_channels, out_channels)
        assert in_channels % out_channels == 0, (
            f"in_channels must be divisible by out_channels, "
            f"but got in_channels={in_channels} and out_channels={out_channels}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (b, l, in_channels)
        """
        # shortcut: mean out
        x_ = rearrange(x, "b l (c h) -> b l c h", c=self.out_channels).mean(dim=-1)
        x = self.norm_act(x)
        x = self.proj_out(x)
        return x + x_


class MingtokUpsampleAverage(nn.Module):
    """
    Mingtok decoder input projection layer
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj_in = nn.Linear(in_channels, out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert out_channels % in_channels == 0, (
            f"out_channels must be divisible by in_channels, "
            f"but got in_channels={in_channels} and out_channels={out_channels}"
        )
        self.rep_factor = self.out_channels // self.in_channels
        assert self.out_channels % self.in_channels == 0, (
            f"out_channels must be divisible by in_channels, "
            f"but got in_channels={in_channels} and out_channels={out_channels}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (bs, len_x, in_channels)
        """
        bs, l, _ = x.shape
        # repeat interleave
        x_ = x[..., None].repeat_interleave(self.rep_factor, dim=-1).view(bs, l, -1)
        x = self.proj_in(x)
        return x + x_


# * --- Upsample and downsample entries --- #


def build_upsample_block(
    block_type: Literal["ConvPixelShuffle", "RepeatConv", "InterpolateConv"],
    in_channels: int,
    out_channels: int,
    shortcut: Literal["duplicating"] | None = None,
    padding_mode: str = "zeros",
    norm_type: str | None = None,  # deprecated
    norm_keep: bool = False,
    **kwargs,
) -> nn.Module:
    logger.debug(
        f"[build_upsample_block] block_type: {block_type}, "
        f"in_channels: {in_channels}, "
        f"out_channels: {out_channels}, "
        f"shortcut: {shortcut}, "
        f"padding_mode: {padding_mode}, "
        f"norm keep: {norm_keep}",
        "debug",
    )

    if block_type == "ConvPixelShuffle":
        block = ConvPixelShuffleUpSampleLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            factor=2,
            padding_mode=padding_mode,
        )
    elif block_type == "RepeatConv":
        block = UpsampleRepeatConv(
            in_channels,
            padding_mode=padding_mode,
            norm_keep=norm_keep,
            interp_type=kwargs.get("interp_type", "nearest_interp"),
        )
    elif block_type == "InterpolateConv":
        block = InterpolateConvUpSampleLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            factor=2,
            padding_mode=padding_mode,
            mode="nearest",
            norm_keep=norm_keep,
        )
    else:
        raise ValueError(f"block_type {block_type} is not supported for upsampling")

    if shortcut is None:
        pass
    elif shortcut == "duplicating":
        shortcut_block = ChannelDuplicatingPixelUnshuffleUpSampleLayer(
            in_channels=in_channels, out_channels=out_channels, factor=2
        )
        block = ResidualBlock(block, shortcut_block)
    else:
        raise ValueError(f"shortcut {shortcut} is not supported for upsample")
    return block


def build_downsample_block(
    block_type: Literal["Conv", "PadConv", "ConvPixelUnshuffle"],
    in_channels: int,
    out_channels: int,
    shortcut: None | Literal["averaging"] = None,
    padding_mode: str = "zeros",
    *,
    padconv_use_manually_pad: bool = True,  # for the compatibility with cosmos checkpoints
    norm_type: str | None = None,
    norm_keep: bool = False,
) -> nn.Module:
    logger.info(
        f"[build_downsample_block] {block_type=}, "
        f"{in_channels=}, "
        f"{out_channels=}, "
        f"{shortcut=}, "
        f"{padding_mode=}, "
        f"{padconv_use_manually_pad=}, "
        f"{norm_keep=}",
    )

    if block_type == "Conv":
        block = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            padding_mode=padding_mode,
            bias=True,
        )
    elif block_type == "PadConv":
        block = DownsamplePadConv(
            in_channels=in_channels,
            padding_in_conv=not padconv_use_manually_pad,
            padding_mode=padding_mode,
            norm_keep=norm_keep,
            use_conv=True,
        )
    elif block_type == "ConvPixelUnshuffle":
        block = ConvPixelUnshuffleDownSampleLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            factor=2,
            padding_mode=padding_mode,
        )
    else:
        raise ValueError(f"block_type {block_type} is not supported for downsampling")

    if shortcut is None:
        pass
    elif shortcut == "averaging":
        shortcut_block = PixelUnshuffleChannelAveragingDownSampleLayer(
            in_channels=in_channels, out_channels=out_channels, factor=2
        )
        block = ResidualBlock(block, shortcut_block)
    else:
        raise ValueError(f"shortcut {shortcut} is not supported for downsample")
    return block


if __name__ == "__main__":
    """
    python -m src.stage1.cosmos.modules.resample
    """
    # downsample = MingtokDownsampleShortCut(256, 16)
    # x = torch.randn(1, 512, 256)
    # y = downsample(x)
    # print(y.shape)

    upsample = MingtokUpsampleAverage(16, 256)
    x = torch.randn(1, 512, 16)
    y = upsample(x)
    print(y.shape)
