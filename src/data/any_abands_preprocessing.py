import math

import numpy as np
import torch
from scipy.stats import entropy


def calculate_entropy(band):
    hist, _ = np.histogram(band, bins=256, range=(0, 1))
    prob = hist / hist.sum()
    return entropy(prob)


def hyper_image_choose_any_bands(channel, data, band_choose="order"):
    """
    Choose any bands of HSI data using order or entropy or std or low_correlation

    Args:
        channel (int): number of bands to choose
        data (numpy array): [h, w, channel]
        band_choose (str): 'order','std','entropy','low_correlation'
    Returns:
        x (torch tensor): [1, channel, h, w]
    """
    b = data.shape[-1]
    data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-10)
    if channel > b:
        x = np.tile(data, (1, 1, math.ceil(channel / b)))
        x = x[:, :, :channel]
    elif channel < b:
        if band_choose == "order":
            x = data[:, :, :channel]
        elif band_choose == "std":
            band_variances = np.var(data, axis=(0, 1))
            selected_band_indices = np.argsort(band_variances)[:channel]
            x = data[:, :, selected_band_indices]
        elif band_choose == "entropy":
            band_entropies = np.array([calculate_entropy(data[:, :, i]) for i in range(b)])
            selected_band_indices = np.argsort(band_entropies)[:channel]
            x = data[:, :, selected_band_indices]
        elif band_choose == "low_correlation":
            data_2d = data.reshape(-1, b)
            correlation_matrix = np.corrcoef(data_2d, rowvar=False)
            band_correlation_scores = np.sum(np.abs(correlation_matrix), axis=1)
            selected_band_indices = np.argsort(band_correlation_scores)[:channel]
            x = data[:, :, selected_band_indices]
        else:
            raise ValueError(
                f"Invalid band_choose value: {band_choose}. Expected one of: 'order', 'std', 'entropy', 'low_correlation'"
            )
    else:
        x = data
    x = (x - np.min(x)) / (np.max(x) - np.min(x)) * 2 - 1
    x = x * 0.1
    x = torch.as_tensor(x, dtype=torch.float32)
    x = x.to(torch.float32)
    x = x.permute(2, 0, 1).unsqueeze(0)
    return x
