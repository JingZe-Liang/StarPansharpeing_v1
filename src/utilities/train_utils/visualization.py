from typing import TypeVar, cast

import beartype
import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Float32, Int, UInt
from matplotlib.cm import get_cmap
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from numpy.typing import NDArray
from PIL import Image
from torch import Tensor

RGB_CHANNELS_BY_BANDS = {
    4: [2, 1, 0],
    8: [4, 2, 0],
    10: [6, 5, 4],
    12: [3, 2, 1],
    13: [4, 3, 2],
    32: [12, 9, 3],
    50: [40, 20, 10],
    150: "mean",  # [37, 28, 13],
    175: "mean",  # [42, 32, 13],
    191: [19, 12, 8],  # WDC mall
    202: "mean",  # [39, 32, 16],
    224: "mean",  # [39, 32, 16],
    242: "mean",  # [66, 40, 13],
    368: "mean",  # [74, 42, 10],
    369: "mean",
    438: "mean",  # [62, 33, 19],
    439: "mean",
}


def get_coco_colors():
    COCO_CATEGORIES = np.array(
        [
            [220, 20, 60],
            [119, 11, 32],
            [0, 0, 142],
            [0, 0, 230],
            [106, 0, 228],
            [0, 60, 100],
            [0, 80, 100],
            [0, 0, 70],
            [0, 0, 192],
            [250, 170, 30],
            [100, 170, 30],
            [220, 220, 0],
            [175, 116, 175],
            [250, 0, 30],
            [165, 42, 42],
            [255, 77, 255],
            [0, 226, 252],
            [182, 182, 255],
            [0, 82, 0],
            [120, 166, 157],
            [110, 76, 0],
            [174, 57, 255],
            [199, 100, 0],
            [72, 0, 118],
            [255, 179, 240],
            [0, 125, 92],
            [209, 0, 151],
            [188, 208, 182],
            [0, 220, 176],
            [255, 99, 164],
            [92, 0, 73],
            [133, 129, 255],
            [78, 180, 255],
            [0, 228, 0],
            [174, 255, 243],
            [45, 89, 255],
            [134, 134, 103],
            [145, 148, 174],
            [255, 208, 186],
            [197, 226, 255],
            [171, 134, 1],
            [109, 63, 54],
            [207, 138, 255],
            [151, 0, 95],
            [9, 80, 61],
            [84, 105, 51],
            [74, 65, 105],
            [166, 196, 102],
            [208, 195, 210],
            [255, 109, 65],
            [0, 143, 149],
            [179, 0, 194],
            [209, 99, 106],
            [5, 121, 0],
            [227, 255, 205],
            [147, 186, 208],
            [153, 69, 1],
            [3, 95, 161],
            [163, 255, 0],
            [119, 0, 170],
            [0, 182, 199],
            [0, 165, 120],
            [183, 130, 88],
            [95, 32, 0],
            [130, 114, 135],
            [110, 129, 133],
            [166, 74, 118],
            [219, 142, 185],
            [79, 210, 114],
            [178, 90, 62],
            [65, 70, 15],
            [127, 167, 115],
            [59, 105, 106],
            [142, 108, 45],
            [196, 172, 0],
            [95, 54, 80],
            [128, 76, 255],
            [201, 57, 1],
            [246, 0, 122],
            [191, 162, 208],
        ]
    )
    COCO_CATEGORIES = np.concatenate(
        (COCO_CATEGORIES, np.ones((COCO_CATEGORIES.shape[0], 1))), axis=1
    )
    return COCO_CATEGORIES / 255.0


def get_rgb_image(img: torch.Tensor, rgb_channels: list[int] | None = None):
    global RGB_CHANNELS_BY_BANDS

    c = img.shape[1]
    if c not in RGB_CHANNELS_BY_BANDS:
        raise ValueError(
            f"Invalid number of channels: {c}. Expected one of {list(RGB_CHANNELS_BY_BANDS.keys())}"
        )

    rgb_channels = rgb_channels or RGB_CHANNELS_BY_BANDS[c]
    if isinstance(rgb_channels, (list, tuple)):
        rgb_img = img[:, rgb_channels, :, :]
    elif rgb_channels == "mean":
        # split three parts
        c_3 = c // 3
        bands = [img[:, i * c_3 : (i + 1) * c_3, :, :].mean(dim=1) for i in range(3)]
        rgb_img = torch.stack(bands, dim=1)
    else:
        raise ValueError(
            f"Invalid RGB channels mapping: {rgb_channels}. Expected list, tuple or 'mean'."
        )

    return rgb_img


type GTMapType = (
    Int[NDArray, "h w"]
    | Int[Tensor, "b c h w"]
    | Int[Tensor, "b h w"]
    | UInt[NDArray, "h w"]
    | UInt[Tensor, "b c h w"]
    | UInt[Tensor, "b h w"]
)
type VisGTMapType = (
    Image.Image
    | list[Image.Image]
    | Float32[NDArray, "b h w"]
    | Float32[NDArray, "h w"]
)


@beartype.beartype
def visualize_segmentation_map(
    gt_map: GTMapType,
    cmap: str = "tab20",
    n_class: int = 20,
    bg_black=True,
    colors: Float32[NDArray, "n_class 4"] | list[list[int | float]] | None = None,
    use_coco_colors=False,
    alpha: float = 1.0,
    to_rgba=False,
    to_pil=False,
    add_channel_dim_1=False,
) -> VisGTMapType:
    if colors is None:
        if use_coco_colors:
            colors = get_coco_colors()
        else:
            colors = np.array(plt.get_cmap(cmap)(np.linspace(0, 1, n_class)))

    # Convert 0-class to be black
    if bg_black:
        alpha_bg = 1.0
        colors[0] = [0, 0, 0, alpha_bg]

    custom_cmap = ListedColormap(colors)
    custom_cmap.colors = custom_cmap.colors * np.array([1, 1, 1, alpha])
    assert n_class <= custom_cmap.N, f"n_class {n_class} > cmap.N {custom_cmap.N}"

    norm = BoundaryNorm(boundaries=np.arange(0, n_class), ncolors=n_class)
    if torch.is_tensor(gt_map):
        # (b, c, h, w)
        gt_map.squeeze_(1)
        assert gt_map.ndim == 3, f"Invalid gt_map shape: {gt_map.shape}"

        gt_map = gt_map.unbind(0)
        gt_map = [g.cpu().numpy() for g in gt_map]
    else:
        gt_map = [gt_map]

    # Convert to RGB/RGBA color maps
    ms = []
    gt_map = cast(list[np.ndarray], gt_map)  # type: ignore
    for m in gt_map:
        m = norm(m)
        m = custom_cmap(m)
        if not to_rgba:
            m = m[..., :3]
        if to_pil:
            mode = "RGBA" if to_rgba else "RGB"
            m = Image.fromarray((m * 255).astype(np.uint8), mode=mode)
        ms.append(m)

    if to_pil:
        return ms[0] if len(ms) == 1 else ms
    else:
        ms = np.stack(ms).squeeze(0)
        if add_channel_dim_1:
            if ms.ndim == 2:
                ms = ms[None, ...]
            elif ms.ndim == 3:
                ms = ms[:, None]
        return ms
