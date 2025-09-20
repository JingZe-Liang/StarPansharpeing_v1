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


def create_unpatcher(chan: int, out_chan: int, patch_size: int = 1):
    out_patch_chan = out_chan * (patch_size**2)
    unpatcher_conv = create_conv2d(chan, out_patch_chan, 3, padding=1, stride=1)
    rearranger = Rearrange(
        "bs (c p1 p2) h w -> bs c (h p1) (w p2)", p1=patch_size, p2=patch_size
    )
    patcher = nn.Sequential(unpatcher_conv, rearranger)
    return patcher
