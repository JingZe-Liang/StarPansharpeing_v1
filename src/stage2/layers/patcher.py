import torch.nn as nn
from einops.layers.torch import Rearrange
from timm.layers.create_conv2d import create_conv2d
from timm.layers.helpers import to_2tuple
from timm.layers.patch_embed import PatchEmbed


def create_patcher(
    chan: int,
    out_chan: int,
    patch_size: int = 1,
    img_size: int | tuple = 512,
    out_fmt: str = "NCHW",
):
    patcher = PatchEmbed(
        to_2tuple(img_size),
        patch_size,
        chan,
        out_chan,
        norm_layer=None,
        output_fmt=out_fmt,
        strict_img_size=False,
    )
    return patcher


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
            create_conv2d(
                chan // lora_rank_ratio, out_patch_chan, kernel_size, **conv_kwargs
            ),
        )
    else:
        unpatcher_conv = create_conv2d(chan, out_patch_chan, kernel_size, **conv_kwargs)
    rearranger = Rearrange(
        "bs (c p1 p2) h w -> bs c (h p1) (w p2)", p1=patch_size, p2=patch_size
    )
    patcher = nn.Sequential(unpatcher_conv, rearranger)
    return patcher
