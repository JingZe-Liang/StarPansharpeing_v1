from functools import partial
from typing import Literal

import torch
import torch.nn as nn
from accelerate.state import PartialState
from timm.layers import PatchEmbed

from src.stage1.cosmos.modules.blocks import (
    GLUMBConv,
    MoE2DBlock,
    ResidualBlock,
    ResnetBlock,
)
from src.utilities.logging import log_print


def _create_conv_in_module(
    basic_module: str,
    c: int,
    hidden_dim: int,
    padding_mode: str,
    is_patcher: bool = False,
    patch_kwargs: dict | None = None,
):
    if is_patcher:
        module = PatchEmbed(
            img_size=224,
            in_chans=c,
            embed_dim=hidden_dim,
            strict_img_size=False,
            output_fmt="NLC",
            **(patch_kwargs or {}),
        )
        return module

    if basic_module == "conv":
        module = nn.Conv2d(
            in_channels=c,
            out_channels=hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode=padding_mode,
        )
    elif basic_module == "mobile":
        module = nn.Sequential(
            nn.Conv2d(c, hidden_dim, 1, 1, 0),
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                3,
                1,
                1,
                padding_mode=padding_mode,
                groups=hidden_dim,
            ),
        )
    elif basic_module == "inv_bottleneck":
        module = ResidualBlock(
            GLUMBConv(
                in_channels=c,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=1,
                expand_ratio=4,
                use_bias=True,
                norm=(None, None, "rms_triton"),
                act_func=("silu", "silu", None),
            ),
            nn.Conv2d(c, hidden_dim, kernel_size=1, stride=1),
        )
    elif basic_module == "resnet":
        module = ResnetBlock(
            in_channels=c,
            out_channels=hidden_dim,
            num_groups=1,
            dropout=0.0,
            act_checkpoint=False,
            padding_mode=padding_mode,
            norm_type="gn",
        )
    elif basic_module == "moe":
        module = nn.Sequential(
            nn.Conv2d(
                in_channels=c,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="zeros",
            ),
            MoE2DBlock(
                in_channels=hidden_dim,
                hidden_channels=hidden_dim,
                n_experts=4,
                n_selected=1,
                n_shared_experts=1,
                moe_type="tc",
                act_checkpoint=False,
            ),
            # to patcher dim
            # nn.Conv2d(c, hidden_dim, kernel_size=1, stride=1, padding=0),
        )
    else:
        raise ValueError(f"[DiffBandsInputConvIn] Unknown basic_module={basic_module}")
    return module


def _create_conv_out_module(
    basic_module: str,
    c: int,
    hidden_dim: int,
    padding_mode: str,
):
    if basic_module == "conv":
        module = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=c,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode=padding_mode,
        )
    elif basic_module == "mobile":
        module = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                3,
                1,
                1,
                padding_mode=padding_mode,
                groups=hidden_dim,
            ),
            nn.Conv2d(hidden_dim, c, 1, 1, 0),
        )
    elif basic_module == "inv_bottleneck":
        module = ResidualBlock(
            GLUMBConv(
                in_channels=hidden_dim,
                out_channels=c,
                kernel_size=3,
                stride=1,
                expand_ratio=4,
                use_bias=True,
                norm=(None, None, "rms_triton"),
                act_func=("silu", "silu", None),
            ),
            nn.Conv2d(hidden_dim, c, kernel_size=1, stride=1),
        )
    elif basic_module == "resnet":
        module = ResnetBlock(
            in_channels=hidden_dim,
            out_channels=c,
            dropout=0.0,
            act_checkpoint=False,
            num_groups=1,
            norm_type="gn",
        )
    elif basic_module == "moe":
        module = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode=padding_mode,
            ),
            MoE2DBlock(
                in_channels=hidden_dim,
                hidden_channels=hidden_dim,
                n_experts=4,
                n_selected=1,
                n_shared_experts=1,
                moe_type="tc",
                act_checkpoint=False,
            ),
            nn.Conv2d(hidden_dim, c, kernel_size=1, stride=1, padding=0),
        )
    else:
        raise ValueError(f"[DiffBandsInputConvOut] Unknown basic_module={basic_module}")
    return module


class DiffBandsInputConvIn(nn.Module):
    def __init__(
        self,
        band_lst: list[int],
        hidden_dim: int,
        basic_module: Literal["conv", "resnet", "inv_bottleneck", "moe"] = "conv",
        padding_mode: str = "zeros",
        check_grads: bool = True,
        is_patcher: bool = False,
        patch_kwargs: dict | None = None,
    ):
        super().__init__()
        self.band_lst = band_lst
        self.hidden_dim = hidden_dim
        self.is_ddp = PartialState().use_distributed or check_grads
        self._in_module_partial_kwargs = {
            "basic_module": basic_module,
            "hidden_dim": hidden_dim,
            "padding_mode": padding_mode,
            "is_patcher": is_patcher,
            "patch_kwargs": patch_kwargs,
        }

        self.in_modules = nn.ModuleDict()
        for c in band_lst:
            module = _create_conv_in_module(c=c, **self._in_module_partial_kwargs)
            self.in_modules["conv_in_{}".format(c)] = module
            log_print(
                f"[DiffBandsInputConvIn] set conv to hidden module and buffer for channel {c}"
            )

    def add_or_drop_modules(
        self, add_chans: list[int] | None = None, drop_chans: list[int] | None = None
    ):
        """
        Add or remove convolution modules for specific channels.

        Args:
            add_chans: List of channel numbers to add modules for
            drop_chans: List of channel numbers to remove modules for
        """
        if drop_chans is not None:
            for drop_chan in drop_chans:
                if drop_chan in self.band_lst:
                    # Remove module
                    conv_key = f"conv_in_{drop_chan}"
                    if conv_key in self.in_modules:
                        del self.in_modules[conv_key]

                    # Update band list
                    self.band_lst = [c for c in self.band_lst if c != drop_chan]

        if add_chans is not None:
            for add_chan in add_chans:
                if add_chan not in self.band_lst:
                    # Add module using saved kwargs
                    conv_key = f"conv_in_{add_chan}"
                    self.in_modules[conv_key] = _create_conv_in_module(
                        c=add_chan, **self._in_module_partial_kwargs
                    )

                    # Update band list
                    self.band_lst.append(add_chan)
                    self.band_lst.sort()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c_ = x.shape[1]
        module = getattr(self.in_modules, "conv_in_{}".format(c_))
        if module is None:
            raise ValueError(
                f"[DiffBandsInputConvIn] no module for channel {c_}, please check the channel list"
            )
        h = module(x)

        if self.training and self.is_ddp:
            for c in self.band_lst:
                if c != c_:
                    m = self.in_modules["conv_in_{}".format(c)]
                    dummy_loss = sum(p.sum() * 0.0 for p in m.parameters())
                    h = h + dummy_loss

        return h


class DiffBandsInputConvOut(nn.Module):
    def __init__(
        self,
        band_lst: list[int],
        hidden_dim: int,
        basic_module: str = "conv",
        padding_mode: str = "zeros",
        check_grads: bool = True,
    ):
        super().__init__()

        self.band_lst = band_lst
        self.hidden_dim = hidden_dim
        self.basic_module = basic_module
        self.is_ddp = PartialState().use_distributed or check_grads
        self._out_module_partial_kwargs = {
            "basic_module": basic_module,
            "hidden_dim": hidden_dim,
            "padding_mode": padding_mode,
        }

        self.in_modules = nn.ModuleDict()
        for c in band_lst:
            module = _create_conv_out_module(c=c, **self._out_module_partial_kwargs)
            self.in_modules["conv_out_{}".format(c)] = module
            log_print(
                f"[DiffBandsInputConvOut] set conv to hidden module for channel {c}"
            )

        self.out_channel = None

    def add_or_drop_modules(
        self, add_chans: list[int] | None = None, drop_chans: list[int] | None = None
    ):
        """
        Add or remove convolution modules for specific channels.

        Args:
            add_chans: List of channel numbers to add modules for
            drop_chans: List of channel numbers to remove modules for
        """
        if drop_chans is not None:
            for drop_chan in drop_chans:
                if drop_chan in self.band_lst:
                    # Remove module
                    conv_key = f"conv_out_{drop_chan}"
                    if conv_key in self.in_modules:
                        del self.in_modules[conv_key]

                    # Update band list
                    self.band_lst = [c for c in self.band_lst if c != drop_chan]

        if add_chans is not None:
            for add_chan in add_chans:
                if add_chan not in self.band_lst:
                    # Add module using saved kwargs
                    conv_key = f"conv_out_{add_chan}"
                    self.in_modules[conv_key] = _create_conv_out_module(
                        c=add_chan, **self._out_module_partial_kwargs
                    )

                    # Update band list
                    self.band_lst.append(add_chan)
                    self.band_lst.sort()

    def forward(self, x: torch.Tensor, out_channel: int) -> torch.Tensor:
        self.out_channel = out_channel

        module = getattr(self.in_modules, f"conv_out_{out_channel}", None)
        if module is None:
            raise ValueError(
                f"[DiffBandsInputConvOut] No module for out_channel={out_channel}. Available: {list(self.in_modules.keys())}",
            )
        h = module(x)

        if self.training and self.is_ddp:
            for c in self.band_lst:
                if c != out_channel:
                    m = self.in_modules["conv_out_{}".format(c)]
                    dummy_loss = sum(p.sum() * 0.0 for p in m.parameters())
                    h = h + dummy_loss

        return h

    @property
    def weight(self):
        # used to get the weight of the conv_out module for GAN loss
        assert self.out_channel is not None, (
            "out_channel is not set, please call forward first"
        )
        module = getattr(self.in_modules, f"conv_out_{self.out_channel}", None)
        if module is None:
            raise ValueError(
                f"[DiffBandsInputConvOut] No module for out_channel={self.out_channel}. Available: {list(self.in_modules.keys())}",
            )

        match self.basic_module:
            case "conv":
                return module.weight
            case "resnet":
                return module.conv2.weight
            case "moe":
                return module[-1].weight
            case "inv_bottleneck":
                return module.main.depth_conv.conv.weight
            case "mobile":
                return module[-1].weight
            case _:
                raise ValueError(
                    f"[DiffBandsInputConvOut] Unknown basic_module={self.basic_module}. Available: conv, resnet, moe, inv_bottleneck"
                )


# * --- Dinov3 backbone diffbands compatibility --- #


def dinov3_patchembeding_to_diffbands(
    backbone,
    band_lst: list[int],
    hidden_dim: int,
    basic_module="conv",
    padding_mode: str = "zeros",
    check_grads: bool = True,
):
    pe_weights = backbone.patch_embed.proj.weight  # (c, 3, 16, 16)
    assert isinstance(backbone.patch_embed.norm, nn.Identity), (
        "backbone.patch_embed.norm must be nn.Identity"
    )
    assert basic_module == "conv", "only conv is supported for dinov3 compatibility"
    assert 3 in band_lst, "3 must be in band_lst for dinov3 compatibility"

    conv_in = DiffBandsInputConvIn(
        band_lst=band_lst,
        hidden_dim=hidden_dim,
        basic_module=basic_module,
        padding_mode=padding_mode,
        check_grads=check_grads,
        is_patcher=True,
        patch_kwargs={
            "img_size": backbone.patch_size,
        },
    )
    with torch.no_grad():
        conv_in.in_modules["conv_in_3"].proj.weight.data.copy_(pe_weights.data)  # type: ignore
    log_print("[Dinov3 Diffbands]: set patch embedding to hidden module for channel 3")

    backbone.patch_embed = conv_in

    return backbone
