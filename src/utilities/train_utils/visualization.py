import torch

RGB_CHANNELS_BY_BANDS = {
    4: [2, 1, 0],
    8: [4, 2, 0],
    10: [6, 5, 4],
    12: [4, 3, 2],
    13: [4, 3, 2],
    32: [12, 9, 3],
    50: [40, 20, 10],
    150: [37, 28, 13],
    175: [42, 32, 13],
    202: [39, 32, 16],
    224: [39, 32, 16],
    242: [66, 40, 13],
    368: [74, 42, 10],
    438: [62, 33, 19],
}


def get_rgb_image(img: torch.Tensor):
    global RGB_CHANNELS_BY_BANDS

    c = img.shape[1]
    if c not in RGB_CHANNELS_BY_BANDS:
        raise ValueError(
            f"Invalid number of channels: {c}. Expected one of {list(RGB_CHANNELS_BY_BANDS.keys())}"
        )
    rgb_channels = RGB_CHANNELS_BY_BANDS[c]
    rgb_img = img[:, rgb_channels, :, :]
    return rgb_img
