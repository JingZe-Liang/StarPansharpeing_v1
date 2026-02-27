from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from src.stage1.utilities.losses.gan_loss.utils import get_rgb_channels_for_model


def ensure_feature_list(features: Tensor | Sequence[Tensor]) -> list[Tensor]:
    if torch.is_tensor(features):
        return [features]
    return list(features)


def maybe_detach_feature_list(features: list[Tensor], detach: bool) -> list[Tensor]:
    if not detach:
        return features
    return [feat.detach() for feat in features]


def next_divisible_of_y(x: int, y: int) -> int:
    return math.ceil(x / y) * y


def select_rgb_channels(
    img: Tensor,
    *,
    rgb_channels: list[int] | str | None,
    use_linstretch: bool,
    pca_fn: Callable[..., Tensor] | None,
) -> Tensor:
    return get_rgb_channels_for_model(
        rgb_channels=rgb_channels,
        img=img,
        use_linstretch=use_linstretch,
        pca_fn=pca_fn,
    )


def maybe_resize_img(
    img: Tensor,
    *,
    img_resize: tuple[int, int] | str | None,
    image_size: int,
    patch_size: int,
) -> Tensor:
    img_size = tuple(img.shape[-2:])
    if tuple([image_size] * 2) == img_size:
        return img

    if img_resize == "dino":
        tgt_size: int | tuple[int, int] = int(image_size)
    elif isinstance(img_resize, tuple):
        tgt_size = (int(img_resize[0]), int(img_resize[1]))
    else:
        tgt_size = (
            next_divisible_of_y(img.shape[-2], patch_size),
            next_divisible_of_y(img.shape[-1], patch_size),
        )

    return F.interpolate(img, size=tgt_size, mode="bilinear", align_corners=False)


def normalize_img(img: Tensor, *, img_is_neg1_1: bool, normalize_fn: Callable[[Tensor], Tensor]) -> Tensor:
    if img_is_neg1_1:
        img = (img + 1) / 2
    return normalize_fn(img)
