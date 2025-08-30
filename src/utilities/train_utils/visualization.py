import torch

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
