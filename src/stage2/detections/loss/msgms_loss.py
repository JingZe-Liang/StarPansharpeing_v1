"""
Multi-Scale Gradient Magnitude Similarity (MSGMS) Loss

This module implements the MSGMS loss function for hyperspectral image reconstruction.
The loss computes the similarity between gradient magnitudes of input and reconstructed
images at multiple scales, which is particularly effective for anomaly detection tasks.

The core idea is:
1. Use edge detection operators (Sobel/Prewitt) to extract gradient information
2. Compute gradient magnitude similarity at multiple scales
3. Combine losses across scales for robust reconstruction evaluation

Reference:
    Gradient Magnitude Similarity Deviation: An Highly Efficient Perceptual Image Quality Index
    https://arxiv.org/abs/1308.3052
"""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


class Prewitt(nn.Module):
    """
    Prewitt edge detection operator

    Args:
        device_ids (list): List of device IDs for multi-GPU support
    """

    def __init__(self, device_ids):
        super().__init__()
        self.filter = nn.Conv2d(
            in_channels=1,
            out_channels=2,
            kernel_size=3,
            stride=1,
            padding=0,
            bias=False,
        )
        Gx = torch.tensor([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]]) / 3
        Gy = torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]]) / 3
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1).cuda(device=device_ids[0])
        self.filter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, img):
        """
        Apply Prewitt edge detection

        Args:
            img (torch.Tensor): Input image tensor

        Returns:
            torch.Tensor: Edge magnitude map
        """
        x = self.filter(img)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        return x


class Sobel(nn.Module):
    """
    Sobel edge detection operator with 4 directions

    Args:
        device_ids (list): List of device IDs for multi-GPU support
    """

    def __init__(self, device_ids):
        super().__init__()
        self.filter = nn.Conv2d(
            in_channels=1,
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=0,
            bias=False,
        )
        G1 = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 4
        G2 = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]) / 4
        G3 = torch.tensor([[2.0, 1.0, 0.0], [1.0, 0.0, -1.0], [0.0, -1.0, -2.0]]) / 4
        G4 = torch.tensor([[0.0, -1.0, -2.0], [1.0, 0.0, -1.0], [2.0, 1.0, 0.0]]) / 4
        G = torch.cat([G1.unsqueeze(0), G2.unsqueeze(0), G3.unsqueeze(0), G4.unsqueeze(0)], 0)
        G = G.unsqueeze(1).cuda(device=device_ids[0])
        self.filter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, img):
        """
        Apply Sobel edge detection

        Args:
            img (torch.Tensor): Input image tensor

        Returns:
            torch.Tensor: Edge magnitude map
        """
        x = self.filter(img)
        x = torch.mul(x, x)
        x = torch.mean(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        return x


def GMS(Ii, Ir, edge_filter, c=0.1):
    """
    Compute Gradient Magnitude Similarity map

    Args:
        Ii (torch.Tensor): Input image
        Ir (torch.Tensor): Reconstructed image
        edge_filter (nn.Module): Edge detection filter (Sobel/Prewitt)
        c (float): Constant for numerical stability

    Returns:
        torch.Tensor: Gradient magnitude similarity map
    """
    x = torch.mean(Ii, dim=1, keepdim=True)
    y = torch.mean(Ir, dim=1, keepdim=True)
    g_I = edge_filter(x)
    g_Ir = edge_filter(y)
    g_map = (2 * g_I * g_Ir + c) / (g_I**2 + g_Ir**2 + c)
    return g_map


class MSGMSLoss(nn.Module):
    """
    Multi-Scale Gradient Magnitude Similarity Loss

    This loss function computes the similarity between gradient magnitudes
    of input and reconstructed images at multiple scales, which is effective
    for hyperspectral image reconstruction and anomaly detection.

    Args:
        device_ids (list): List of device IDs for multi-GPU support
        pool_num (int): Number of pooling operations for multi-scale analysis (default: 4)

    Example:
        >>> loss_fn = MSGMS_Loss(device_ids=[0], pool_num=4)
        >>> loss = loss_fn(input_image, reconstructed_image)
    """

    def __init__(self, device_ids, pool_num=4):
        super().__init__()
        self.GMS = partial(GMS, edge_filter=Sobel(device_ids))
        self.pool_num = pool_num

    def GMS_loss(self, Ii, Ir):
        """
        Compute single-scale GMS loss

        Args:
            Ii (torch.Tensor): Input image
            Ir (torch.Tensor): Reconstructed image

        Returns:
            torch.Tensor: GMS loss value
        """
        return torch.mean(1 - self.GMS(Ii, Ir))

    def forward(self, Ii, Ir):
        """
        Compute multi-scale GMS loss

        Args:
            Ii (torch.Tensor): Input image
            Ir (torch.Tensor): Reconstructed image

        Returns:
            torch.Tensor: Multi-scale GMS loss value
        """
        total_loss = self.GMS_loss(Ii, Ir)

        for _ in range(self.pool_num):
            Ii = F.avg_pool2d(Ii, kernel_size=2, stride=2)
            Ir = F.avg_pool2d(Ir, kernel_size=2, stride=2)
            total_loss += self.GMS_loss(Ii, Ir)

        return total_loss / int(1 + self.pool_num)
