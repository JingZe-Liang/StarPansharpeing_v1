import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal
from scipy.signal import windows
from torch import nn

from src.utilities.config_utils import function_config_to_basic_types
from src.utilities.logging import log_print

# * --- Pansharpening simulators --- #


def genMTF(ratio, sensor, nbands):
    """
    Generate MTF-matched Gaussian filters for multispectral bands.

    Args:
        ratio: Scale ratio between PAN and MS (e.g., 4)
        sensor: Sensor type (e.g., 'QB', 'WV2')
        nbands: Number of spectral bands

    Returns:
        h: [N, N, nbands] Filter kernels for each band
    """
    # 1. Define Nyquist frequencies based on sensor
    if sensor == "QB":
        GNyq = [0.34, 0.32, 0.30, 0.22]  # B,G,R,NIR
    elif sensor == "IKONOS":
        GNyq = [0.26, 0.28, 0.29, 0.28]
    elif sensor in ["GeoEye1", "WV4"]:
        GNyq = [0.23] * 4  # All bands same
    elif sensor == "WV2":
        GNyq = [0.35] * 7 + [0.27]  # 7 bands + NIR
    elif sensor == "WV3":
        GNyq = [0.325, 0.355, 0.360, 0.350, 0.365, 0.360, 0.335, 0.315]
    else:
        GNyq = [0.3] * nbands  # Default uniform

    # 2. Parameters
    N = 41  # Kernel size
    nBands = len(GNyq)
    h = np.zeros((N, N, nBands))
    fcut = 1 / ratio

    # 3. Generate per-band filters
    for ii in range(nBands):
        alpha = np.sqrt(((N - 1) * (fcut / 2)) ** 2 / (-2 * np.log(GNyq[ii])))

        # Create Gaussian kernel manually (fspecial alternative)
        ax = np.arange(-N // 2 + 1, N // 2 + 1)
        xx, yy = np.meshgrid(ax, ax)
        H = np.exp(-(xx**2 + yy**2) / (2 * alpha**2))

        # Normalize
        Hd = H / np.max(H)

        # Apply Kaiser window (fwind1 alternative)
        kai = windows.kaiser(N, beta=3)  # Beta=3 approximates MATLAB's default
        h_windowed = Hd * kai[:, None] * kai[None, :]  # Outer product

        h[:, :, ii] = h_windowed

    return h


class MTFConv(nn.Module):
    def __init__(self, ratio: int, sensor: str, nbands: int):
        super(MTFConv, self).__init__()
        self.ratio = ratio  # Store ratio
        self.sensor = sensor  # Store sensor
        self.nbands = nbands  # Store nbands
        # 1. Define Nyquist frequencies based on sensor
        if sensor == "QB":
            GNyq = [0.34, 0.32, 0.30, 0.22]  # B,G,R,NIR
        elif sensor == "IKONOS":
            GNyq = [0.26, 0.28, 0.29, 0.28]
        elif sensor in ["GeoEye1", "WV4"]:
            GNyq = [0.23] * 4  # All bands same
        elif sensor == "WV2":
            # Original code had a potential issue here, assuming 7 bands + NIR = 8 bands total
            # If nbands is different, this needs adjustment.
            # Let's assume nbands includes NIR if WV2 is specified.
            if nbands == 8:
                GNyq = [0.35] * 7 + [0.27]
            else:
                # Fallback or raise error if nbands doesn't match expected WV2 structure
                log_print(
                    f"Warning: WV2 sensor specified with {nbands} bands. Using default GNyq.",
                    level="warning",
                )
                GNyq = [0.3] * nbands
        elif sensor == "WV3":
            # Ensure nbands matches the length of GNyq list for WV3
            wv3_gnqy = [0.325, 0.355, 0.360, 0.350, 0.365, 0.360, 0.335, 0.315]
            if nbands == len(wv3_gnqy):
                GNyq = wv3_gnqy
            else:
                log_print(
                    f"Warning: WV3 sensor specified with {nbands} bands, expected {len(wv3_gnqy)}. Using default GNyq.",
                    level="warning",
                )
                GNyq = [0.3] * nbands
        else:
            log_print(
                f"Sensor {sensor} not explicitly found, using uniform Nyquist frequency 0.3",
                level="warning",
            )
            GNyq = [0.3] * nbands  # Default uniform
        self.GNyq = GNyq

        # 2. Generate per-band filters
        N = 41  # Kernel size
        fcut = 1 / ratio
        kernels = []

        # Ensure GNyq has the correct number of elements
        if len(GNyq) != nbands:
            raise ValueError(
                f"Number of GNyq values ({len(GNyq)}) does not match nbands ({nbands}) for sensor {sensor}"
            )

        for ii in range(nbands):
            # Check if GNyq[ii] is valid for log
            if GNyq[ii] <= 0:
                raise ValueError(
                    f"GNyq value must be positive, but got {GNyq[ii]} for band {ii}"
                )
            alpha_sq = ((N - 1) * (fcut / 2)) ** 2 / (-2 * np.log(GNyq[ii]))
            if alpha_sq <= 0:
                raise ValueError(
                    f"Calculated alpha^2 is non-positive ({alpha_sq}). Check GNyq ({GNyq[ii]}) and ratio ({ratio})."
                )
            alpha = np.sqrt(alpha_sq)

            # Create Gaussian kernel manually
            ax = np.arange(-N // 2 + 1, N // 2 + 1)
            xx, yy = np.meshgrid(ax, ax)
            H = np.exp(-(xx**2 + yy**2) / (2 * alpha**2))

            # Normalize
            Hd = H / np.max(H)

            # Apply Kaiser window
            kai = windows.kaiser(N, beta=3)  # Beta=3 approximates MATLAB's default
            h_windowed = Hd * kai[:, None] * kai[None, :]  # Outer product

            kernels.append(h_windowed)

        # Convert to PyTorch tensor [out_channels=c, in_channels=1, H, W]
        kernels = np.stack(kernels, axis=0)  # [c, H, W]
        kernels = torch.from_numpy(kernels).float().unsqueeze(1)  # [c, 1, H, W]

        # Add normalization to ensure sum=1 for each kernel
        kernels = kernels / kernels.sum(dim=(2, 3), keepdim=True)

        # Register as buffer (not trainable)
        self.register_buffer("kernels", kernels)

    def __repr__(self):
        return (
            f'MTFConv(ratio={self.ratio}, sensor="{self.sensor}", nbands={self.nbands})\n'
            f"GNyq={self.GNyq}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor with shape (bs, c, h, w)
        Returns:
            y: Output tensor with shape (bs, c, h, w)
        """
        bs, c, h, w = x.shape
        if c != self.nbands:
            raise ValueError(
                f"Input channel count ({c}) does not match MTFConv nbands ({self.nbands})"
            )

        # Apply grouped convolution (each kernel for its corresponding channel)
        padding = self.kernels.shape[-1] // 2  # Same output size
        x = F.pad(x, (padding, padding, padding, padding), mode="circular")

        # Reshape kernels for grouped convolution: [out_channels=c, in_channels_per_group=1, kH, kW]
        # Kernels are already in shape [c, 1, H, W]
        y = F.conv2d(
            x,
            self.kernels,  # Shape: [c, 1, H, W]
            padding=0,
            groups=c,  # Apply each filter to one input channel
        )
        return y


class Interp23Tap(nn.Module):
    """
    PyTorch implementation of the interp23tap MATLAB function.

    Interpolates the input tensor using a 23-coefficient polynomial interpolator,
    upsampling by the given ratio. The ratio must be a power of 2.

    Args:
        ratio (int): Scale ratio for upsampling. Must be a power of 2.
    """

    def __init__(self, ratio: int, pad_mode: str = "replicate"):
        super().__init__()

        if not (ratio > 0 and (ratio & (ratio - 1) == 0)):
            raise ValueError("Error: Only resize factors power of 2 are supported.")
        self.ratio = ratio
        self.num_upsamples = int(math.log2(ratio))
        self.pad_mode = pad_mode

        # Define the 23-tap filter coefficients (CDF23 from MATLAB code)
        cdf23_coeffs = 2.0 * np.array(
            [
                0.5,
                0.305334091185,
                0.0,
                -0.072698593239,
                0.0,
                0.021809577942,
                0.0,
                -0.005192756653,
                0.0,
                0.000807762146,
                0.0,
                -0.000060081482,
            ]
        )
        # Make symmetric
        base_coeffs = np.concatenate([np.flip(cdf23_coeffs[1:]), cdf23_coeffs])
        base_coeffs_t = torch.tensor(base_coeffs, dtype=torch.float32)

        # Reshape kernel for 2D convolution (separable filter)
        # Kernel for filtering along height (columns in MATLAB)
        kernel_h = base_coeffs_t.view(1, 1, -1, 1)  # Shape (1, 1, 23, 1)
        # Kernel for filtering along width (rows in MATLAB)
        kernel_w = base_coeffs_t.view(1, 1, 1, -1)  # Shape (1, 1, 1, 23)

        # Register kernels as buffers
        self.register_buffer("kernel_h", kernel_h)
        self.register_buffer("kernel_w", kernel_w)

        # Calculate padding size (kernel_size=23)
        self.padding = (base_coeffs_t.shape[0] - 1) // 2  # Should be 11

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for interpolation.

        Args:
            x (torch.Tensor): Input tensor of shape (bs, c, h, w).

        Returns:
            torch.Tensor: Interpolated tensor of shape (bs, c, h * ratio, w * ratio).
        """
        if self.ratio == 1:
            return x

        current_img = x
        bs, c, h_curr, w_curr = current_img.shape

        for k in range(self.num_upsamples):
            h_curr *= 2
            w_curr *= 2

            # Upsample by inserting zeros
            upsampled = torch.zeros(
                bs, c, h_curr, w_curr, device=x.device, dtype=x.dtype
            )

            # Place original pixels according to MATLAB logic
            if k == 0:
                # I1LRU(2:2:end,2:2:end,:) = I_Interpolated;
                upsampled[..., 1::2, 1::2] = current_img
            else:
                # I1LRU(1:2:end,1:2:end,:) = I_Interpolated;
                upsampled[..., ::2, ::2] = current_img

            # Apply separable convolution with circular padding
            # Grouped convolution: apply filter independently per channel
            # Reshape for grouped conv: (1, bs*c, H, W) or apply channel-wise
            # Using conv2d with groups=c is efficient

            # Pad for horizontal filter (width)
            # Pad width dimension (dim 3) by self.padding on both sides
            padded_w = F.pad(
                upsampled, (self.padding, self.padding, 0, 0), mode=self.pad_mode
            )
            # Apply horizontal filter
            # Input: (bs, c, H, W_padded), Kernel: (1, 1, 1, K) -> Output: (bs, c, H, W)
            # We need kernel shape (c, 1, 1, K) for grouped convolution
            kernel_w_grouped = self.kernel_w.repeat(c, 1, 1, 1)
            filtered_w = F.conv2d(padded_w, kernel_w_grouped, groups=c)

            # Pad for vertical filter (height)
            # Pad height dimension (dim 2) by self.padding on both sides
            padded_h = F.pad(
                filtered_w, (0, 0, self.padding, self.padding), mode="circular"
            )
            # Apply vertical filter
            # Input: (bs, c, H_padded, W), Kernel: (1, 1, K, 1) -> Output: (bs, c, H, W)
            # We need kernel shape (c, 1, K, 1) for grouped convolution
            kernel_h_grouped = self.kernel_h.repeat(c, 1, 1, 1)
            filtered_h = F.conv2d(padded_h, kernel_h_grouped, groups=c)

            current_img = filtered_h  # Update image for next iteration

        return current_img


class PansharpSimulator(nn.Module):
    @function_config_to_basic_types
    def __init__(
        self,
        ratio: int,
        sensor: str,
        nbands: int,
        pan_weight_lst: list[float] | str,  # Allow 'mean' string
        downsample_type: str = "avgpool",
        upsample_type: str = "tap23",
    ):
        super(PansharpSimulator, self).__init__()

        if isinstance(pan_weight_lst, str) and pan_weight_lst == "mean":
            actual_pan_weight_lst = [1.0 / nbands] * nbands
            log_print(f"Using mean pan weights: {actual_pan_weight_lst}")
        elif isinstance(pan_weight_lst, list):
            if len(pan_weight_lst) != nbands:
                raise ValueError(
                    f"Length of pan_weight_lst ({len(pan_weight_lst)}) must match nbands ({nbands})"
                )
            weight_sum = sum(pan_weight_lst)
            if not np.isclose(weight_sum, 1.0):
                log_print(
                    f"Sum of pan_weight_lst ({weight_sum}) is not 1. Normalizing.",
                    level="warning",
                )
                actual_pan_weight_lst = [w / weight_sum for w in pan_weight_lst]
            else:
                actual_pan_weight_lst = pan_weight_lst
        else:
            raise TypeError(
                f"pan_weight_lst must be 'mean' or a list of floats, got {type(pan_weight_lst)}"
            )

        self.mtf_conv = MTFConv(ratio, sensor, nbands)
        self.ratio = ratio
        self.downsample_type = downsample_type.lower()
        if self.downsample_type not in [
            "avgpool",
            "interpolate_bilinear",
            "interpolate_bicubic",
        ]:
            log_print(
                f"Unsupported downsample_type: {downsample_type}. Defaulting to avgpool.",
                level="warning",
            )
            self.downsample_type = "avgpool"

        # Instantiate the upsampler
        # Using nbands for channels, assuming upsampling happens on the MS bands
        if upsample_type == "bilinear":
            self.upsample = lambda x: F.interpolate(
                x,
                scale_factor=ratio,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        else:
            self.upsample = Interp23Tap(ratio=ratio)

        pan_weights = torch.tensor(actual_pan_weight_lst, dtype=torch.float32).view(
            1, -1, 1, 1
        )
        self.register_buffer("pan_weights", pan_weights, persistent=False)

    def forward(
        self, hrms: torch.Tensor, pan: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Simulates LRMS and PAN images from an HRMS image using Wald's protocol.

        Args:
            hrms (torch.Tensor): High-Resolution Multi-Spectral image tensor
                                 Shape: (bs, c, h, w)

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - lrms (torch.Tensor): Low-Resolution Multi-Spectral image tensor,
                                       upsampled back to HRMS size using Interp23Tap.
                                       Shape: (bs, c, h, w)
                - pan (torch.Tensor): Panchromatic image tensor.
                                      Shape: (bs, 1, h, w)
        """
        # Ensure hrms is a tensor
        if not isinstance(hrms, torch.Tensor):
            raise TypeError(
                f"Input 'hrms' must be a torch.Tensor, but got {type(hrms)}"
            )

        # Ensure hrms has 4 dimensions (bs, c, h, w)
        if hrms.ndim != 4:
            raise ValueError(
                f"Input HRMS tensor must have 4 dimensions (bs, c, h, w), but got shape {hrms.shape}"
            )

        bs, c, h, w = hrms.shape
        if pan is not None:
            pan_h, pan_w = pan.shape[-2:]
            assert pan_h // h == self.ratio and pan_w // w == self.ratio, (
                f"Provided PAN dimensions ({pan_h}, {pan_w}) do not match expected size based on HRMS and ratio ({h * self.ratio}, {w * self.ratio})"
            )

        device = hrms.device

        # --- 1. Apply MTF --- (Simulates sensor's optical blurring)
        hrms_mtf = self.mtf_conv(hrms)

        # --- 2. Downsample HRMS to LRMS resolution --- (Simulates spatial resolution difference)
        if self.downsample_type == "avgpool":
            lrms_down = F.avg_pool2d(
                hrms_mtf, kernel_size=self.ratio, stride=self.ratio
            )
        elif self.downsample_type.startswith("interpolate"):
            mode = self.downsample_type.split("_")[-1]  # bilinear or bicubic
            lrms_down = F.interpolate(
                hrms_mtf,
                scale_factor=1 / self.ratio,
                mode=mode,
                align_corners=True,
                antialias=True,
            )
        else:  # Should not happen due to check in init, but as fallback
            lrms_down = F.avg_pool2d(
                hrms_mtf, kernel_size=self.ratio, stride=self.ratio
            )

        # --- 3. Upsample LRMS back to HRMS size --- (Using the defined upsampler)
        lrms_upsampled = self.upsample(lrms_down)

        # Ensure the upsampled size matches the original HRMS size if needed
        # This might be necessary if the upsampler doesn't perfectly restore the size
        if lrms_upsampled.shape[-2:] != (h, w):
            log_print(
                f"Upsampled LRMS size {lrms_upsampled.shape[-2:]} does not match target HRMS size {(h, w)}. Resizing...",
                level="warning",
            )
            lrms_upsampled = F.interpolate(
                lrms_upsampled,
                size=(h, w),
                mode="bilinear",
                align_corners=True,
                antialias=True,
            )

        # --- 4. Generate PAN image --- (Weighted sum of HRMS bands)
        # Ensure pan_weights is on the same device as hrms
        if pan is None:
            self.pan_weights: nn.Buffer
            pan = torch.sum(hrms * self.pan_weights.to(device), dim=1, keepdim=True)
        else:
            pan = pan.to(device)
            if pan.ndim == 3:
                pan = pan.unsqueeze(1)
            pan = F.interpolate(
                pan,
                scale_factor=1 / self.ratio,
                mode="bilinear",
                align_corners=True,
                antialias=True,
            )

        return lrms_upsampled, pan
