import torch

RGB_CHANNELS_BY_BANDS = {
    12: [4, 3, 2],
    13: [4, 3, 2],
    8: [4, 2, 0],
    4: [2, 1, 0],
    50: [40, 20, 10],
    224: [90, 60, 40],
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
