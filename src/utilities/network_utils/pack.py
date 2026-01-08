from einops import rearrange
from torch import Tensor


def two_d_to_one_d(x: Tensor):
    assert x.ndim == 4, "Input tensor must be 4D (batch, channels, height, width)"
    return rearrange(x, "b c h w -> b (h w) c")


def one_d_to_two_d(x: Tensor, height: int, width: int):
    assert x.ndim == 3, "Input tensor must be 3D (batch, length, channels)"
    return rearrange(x, "b  (h w) c -> b c h w", h=height, w=width)
