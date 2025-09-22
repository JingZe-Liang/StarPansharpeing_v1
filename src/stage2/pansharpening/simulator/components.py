import math

import numpy as np
import torch
import torch.fft
import torch.nn.functional as F
from scipy import signal
from scipy.signal import windows
from torch import nn

from src.utilities.config_utils import function_config_to_basic_types
from src.utilities.logging import log_print

# * --- Helper functions for MTF generation --- #


def _fir_filter_wind(Hd: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Apply FIR filter windowing to frequency response.

    Parameters
    ----------
    Hd : np.ndarray
        Desired frequency response
    w : np.ndarray
        Window function

    Returns
    -------
    h : np.ndarray
        Filter coefficients
    """
    hd = np.rot90(np.fft.fftshift(np.rot90(Hd, 2)), 2)
    h = np.fft.fftshift(np.fft.ifft2(hd))
    h = np.rot90(h, 2)
    h = h * w
    return h


def _gaussian2d(N: int, std: float) -> np.ndarray:
    """
    Generate 2D Gaussian window.

    Parameters
    ----------
    N : int
        Window size
    std : float
        Standard deviation

    Returns
    -------
    w : np.ndarray
        2D Gaussian window
    """
    t = np.arange(-(N - 1) / 2, (N + 1) / 2)
    t1, t2 = np.meshgrid(t, t)
    std = float(std)
    w = np.exp(-0.5 * (t1 / std) ** 2) * np.exp(-0.5 * (t2 / std) ** 2)
    return w


def _kaiser2d(N: int, beta: float) -> np.ndarray:
    """
    Generate 2D Kaiser window.

    Parameters
    ----------
    N : int
        Window size
    beta : float
        Kaiser window parameter

    Returns
    -------
    w : np.ndarray
        2D Kaiser window
    """
    t = np.arange(-(N - 1) / 2, (N + 1) / 2) / np.double(N - 1)
    t1, t2 = np.meshgrid(t, t)
    t12 = np.sqrt(t1 * t1 + t2 * t2)
    w1 = np.kaiser(N, beta)
    w = np.interp(t12, t, w1)
    w[t12 > t[-1]] = 0
    w[t12 < t[0]] = 0
    return w


# * --- Pansharpening simulators --- #


def genMTF(ratio: int, sensor: str, nbands: int) -> np.ndarray:
    """
    Generate MTF-matched Gaussian filters for multispectral bands.

    Reference implementation based on Vivone's pansharpening toolbox.

    Parameters
    ----------
    ratio : int
        Scale ratio between PAN and MS (e.g., 4)
    sensor : str
        Sensor type (e.g., 'QB', 'WV2', 'GF2')
    nbands : int
        Number of spectral bands

    Returns
    -------
    h : np.ndarray
        Filter kernels with shape [N, N, nbands]
    """
    N = 41  # Kernel size

    # Define Nyquist frequencies based on sensor
    if sensor == "QB":
        GNyq = np.asarray([0.34, 0.32, 0.30, 0.22], dtype="float32")  # B,G,R,NIR
    elif sensor == "IKONOS":
        GNyq = np.asarray([0.26, 0.28, 0.29, 0.28], dtype="float32")  # B,G,R,NIR
    elif sensor in ["GeoEye1", "WV4"]:
        GNyq = np.asarray([0.23, 0.23, 0.23, 0.23], dtype="float32")  # B,G,R,NIR
    elif sensor == "WV2":
        if nbands == 8:
            GNyq = np.asarray([0.35] * 7 + [0.27], dtype="float32")
        else:
            GNyq = np.asarray([0.35] * nbands, dtype="float32")
    elif sensor == "WV3":
        GNyq = np.asarray(
            [0.325, 0.355, 0.360, 0.350, 0.365, 0.360, 0.335, 0.315], dtype="float32"
        )
    elif sensor == "GF2":
        GNyq = np.asarray([0.18, 0.18, 0.18, 0.18], dtype="float32")  # GF2 specific
    else:
        GNyq = 0.3 * np.ones(nbands, dtype="float32")

    # Ensure we have the right number of bands
    if len(GNyq) < nbands:
        # Pad with last value if needed
        GNyq = np.pad(GNyq, (0, nbands - len(GNyq)), mode="edge")
    elif len(GNyq) > nbands:
        # Truncate if needed
        GNyq = GNyq[:nbands]

    h = np.zeros((N, N, nbands), dtype="float32")
    fcut = 1 / ratio

    for ii in range(nbands):
        alpha = np.sqrt(((N - 1) * (fcut / 2)) ** 2 / (-2 * np.log(GNyq[ii])))
        H = _gaussian2d(N, alpha)
        Hd = H / np.max(H)
        w = _kaiser2d(N, 0.5)
        h[:, :, ii] = np.real(_fir_filter_wind(Hd, w))

    return h


class MTFConv(nn.Module):
    """
    MTF convolution layer that applies Modulation Transfer Function filters
    to simulate sensor blurring effects.

    Reference implementation based on Vivone's pansharpening toolbox.
    """

    def __init__(self, ratio: int, sensor: str, nbands: int):
        """
        Initialize MTF convolution layer.

        Parameters
        ----------
        ratio : int
            Scale ratio between PAN and MS
        sensor : str
            Sensor type (e.g., 'QB', 'WV2', 'GF2')
        nbands : int
            Number of spectral bands
        """
        super().__init__()
        self.ratio = ratio
        self.sensor = sensor
        self.nbands = nbands

        # Generate MTF filters using the reference implementation
        h = genMTF(ratio, sensor, nbands)

        # Convert to PyTorch tensor and reshape for conv2d
        # h shape: [N, N, nbands] -> [nbands, 1, N, N] for grouped convolution
        kernels = torch.from_numpy(h).float().permute(2, 0, 1).unsqueeze(1)

        # Register as buffer (not trainable)
        self.register_buffer("kernels", kernels)

    def __repr__(self):
        return (
            f"MTFConv(ratio={self.ratio}, sensor='{self.sensor}', nbands={self.nbands})"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply MTF filtering to input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (bs, c, h, w)

        Returns
        -------
        torch.Tensor
            Filtered tensor with shape (bs, c, h, w)
        """
        _, c, _, _ = x.shape
        if c != self.nbands:
            raise ValueError(
                f"Input channel count ({c}) does not match MTFConv nbands ({self.nbands})"
            )

        # Apply padding for 'same' convolution
        padding = self.kernels.shape[-1] // 2
        x_padded = F.pad(x, (padding, padding, padding, padding), mode="replicate")

        # Apply grouped convolution
        # Each channel gets its own kernel
        output = F.conv2d(
            x_padded,
            self.kernels,  # Shape: [c, 1, kH, kW]
            padding=0,
            groups=c,  # One group per channel
        )

        return output


class Interp23Tap(nn.Module):
    """
    PyTorch implementation of the interp23tap MATLAB function.

    Interpolates the input tensor using a 23-coefficient polynomial interpolator,
    upsampling by the given ratio. The ratio must be a power of 2.

    Reference implementation based on Vivone's pansharpening toolbox.

    Parameters
    ----------
    ratio : int
        Scale ratio for upsampling. Must be a power of 2.
    pad_mode : str, optional
        Padding mode for convolution. Default is "replicate".
    """

    def __init__(self, ratio: int, pad_mode: str = "replicate"):
        super().__init__()

        if not (ratio > 0 and (ratio & (ratio - 1) == 0)):
            raise ValueError("Error: Only resize factors power of 2 are supported.")
        self.ratio = ratio
        self.num_upsamples = int(math.log2(ratio))
        self.pad_mode = pad_mode

        # Generate CDF23 coefficients following the reference implementation
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

        # Create symmetric filter as in reference: d = CDF23[::-1], then CDF23 = np.insert(CDF23, 0, d[:-1])
        d = cdf23_coeffs[::-1]
        base_coeffs = np.insert(cdf23_coeffs, 0, d[:-1])
        base_coeffs_t = torch.tensor(base_coeffs, dtype=torch.float32)

        # Reshape kernel for 1D convolution operations
        self.kernel_size = len(base_coeffs)
        self.padding = (self.kernel_size - 1) // 2  # Should be 11 for 23-tap filter

        # Register kernel as buffer
        self.register_buffer("kernel", base_coeffs_t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply 23-tap interpolation to input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (bs, c, h, w).

        Returns
        -------
        torch.Tensor
            Interpolated tensor of shape (bs, c, h * ratio, w * ratio).
        """
        if self.ratio == 1:
            return x

        current_img = x
        bs, c, _, _ = x.shape

        for k in range(self.num_upsamples):
            # Calculate new dimensions
            h_curr = current_img.shape[2] * 2
            w_curr = current_img.shape[3] * 2

            # Create upsampled tensor with zeros
            upsampled = torch.zeros(
                bs, c, h_curr, w_curr, device=x.device, dtype=x.dtype
            )

            # Place original pixels according to MATLAB logic
            # First iteration: place at odd indices (1::2, 1::2)
            # Subsequent iterations: place at even indices (0::2, 0::2)
            if k == 0:
                upsampled[..., 1::2, 1::2] = current_img
            else:
                upsampled[..., ::2, ::2] = current_img

            # Apply 1D correlation along each dimension, following reference implementation
            # Process like MATLAB: for each channel and batch, apply row-wise then column-wise filtering
            filtered_result = torch.zeros_like(upsampled)

            for batch_idx in range(bs):
                for channel_idx in range(c):
                    # Extract 2D slice: (h_curr, w_curr)
                    slice_2d = upsampled[batch_idx, channel_idx]

                    # Apply horizontal filtering (along rows, like MATLAB: t[j, :] = correlate(t[j, :], BaseCoeff))
                    # Pad horizontally for circular wrapping
                    slice_padded_h = F.pad(
                        slice_2d.unsqueeze(0).unsqueeze(0),
                        (self.padding, self.padding, 0, 0),
                        mode="circular",
                    )
                    # Apply 1D convolution along width (dim 3)
                    kernel_h = self.kernel.view(1, 1, 1, -1)
                    filtered_h = F.conv2d(slice_padded_h, kernel_h, padding=0).squeeze()

                    # Apply vertical filtering (along columns, like MATLAB: t[:, k] = correlate(t[:, k], BaseCoeff))
                    # Pad vertically for circular wrapping
                    filtered_h_padded = F.pad(
                        filtered_h.unsqueeze(0).unsqueeze(0),
                        (0, 0, self.padding, self.padding),
                        mode="circular",
                    )
                    # Apply 1D convolution along height (dim 2)
                    kernel_v = self.kernel.view(1, 1, -1, 1)
                    filtered_v = F.conv2d(
                        filtered_h_padded, kernel_v, padding=0
                    ).squeeze()

                    # Store result
                    filtered_result[batch_idx, channel_idx] = filtered_v

            current_img = filtered_result

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
        antialias: bool = True,
        rounding=True,
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
        self.rounding = rounding
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
                antialias=antialias,
            )
        elif upsample_type == "tap23":
            self.upsample = Interp23Tap(ratio=ratio)
        else:
            # Default to bilinear upsampling
            self.upsample = lambda x: F.interpolate(
                x,
                scale_factor=ratio,
                mode="bilinear",
                align_corners=False,
                antialias=antialias,
            )

        pan_weights = torch.tensor(actual_pan_weight_lst, dtype=torch.float32).view(
            1, -1, 1, 1
        )
        self.register_buffer("pan_weights", pan_weights, persistent=False)

    def forward(
        self, hrms: torch.Tensor, pan: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Simulates LRMS and PAN images from an HRMS image using Wald's protocol.

        Wald's protocol simulation steps:
        1. Apply MTF filtering to HRMS to simulate sensor optical blurring
        2. Downsample MTF-filtered HRMS to create LRMS (simulates spatial resolution difference)
        3. Upsample LRMS back to HRMS resolution using specified interpolation method
        4. Generate PAN image from original HRMS using spectral response weights

        Parameters
        ----------
        hrms : torch.Tensor
            High-Resolution Multi-Spectral image tensor with shape (bs, c, h, w)
        pan : torch.Tensor, optional
            Optional PAN image tensor. If provided, will be resized to match HRMS spatial resolution.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            - lrms_upsampled : torch.Tensor
                Low-Resolution Multi-Spectral image upsampled to HRMS size.
                Shape: (bs, c, h, w)
            - pan : torch.Tensor
                Panchromatic image with same spatial resolution as HRMS.
                Shape: (bs, 1, h, w)
        """
        dtype = hrms.dtype

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
                f"Provided PAN dimensions ({pan_h}, {pan_w}) do not match expected size based on "
                "HRMS and ratio ({h * self.ratio}, {w * self.ratio})"
            )
            assert dtype == pan.dtype, (
                f"hrms and pan must have the same dtype. Got {dtype} and {pan.dtype}"
            )

        device = hrms.device

        # --- 1. Apply MTF --- (Simulates sensor's optical blurring)
        hrms_mtf = self.mtf_conv(hrms.float())

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
                f"Upsampled LRMS size {lrms_upsampled.shape[-2:]} does not "
                f"match target HRMS size {(h, w)}. Resizing...",
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
        # According to Wald's protocol, PAN should have the same spatial resolution as HRMS
        if pan is None:
            # Generate PAN from original HRMS (not MTF-filtered) using spectral weights
            pan = torch.sum(hrms * self.pan_weights.to(device), dim=1, keepdim=True)
        else:
            # If PAN is provided, ensure it has the same spatial resolution as HRMS
            pan = pan.to(device).float()
            if pan.ndim == 3:
                pan = pan.unsqueeze(1)

            # If PAN has higher resolution, downsample it to match HRMS
            if pan.shape[-2:] != (h, w):
                pan = F.interpolate(
                    pan,
                    size=(h, w),
                    mode="bilinear",
                    align_corners=True,
                    antialias=True,
                )

        # Rounding
        if self.rounding:
            lrms_upsampled = lrms_upsampled.type(dtype)
            pan = pan.type(dtype)

        return lrms_upsampled, pan


# * --- Test --- #


def test_mtf():
    """
    Test MTF convolution and PansharpSimulator with visualization and unreferenced metrics.

    This function tests the MTF implementation using bilinear upsampling and computes
    unreferenced pansharpening metrics (D_lambda, D_S, HQNR) to evaluate quality.
    """
    import os
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F

    from src.stage2.pansharpening.metrics.metric_pansharpening import AnalysisPanAcc
    from src.utilities.io import read_image

    # Create tmp directory if it doesn't exist
    tmp_dir = Path("tmp/pansharpening_test")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Test data path
    path = "/HardDisk/ZiHanCao/datasets/Pansharpening-NBU_RS_Data/Sat_Dataset/Dataset/6 WorldView-3/6 WorldView-3/MS_256/20.mat"

    print("Loading test image...")
    ms_data = read_image(path)

    # Handle different return types from read_image
    if isinstance(ms_data, dict):
        ms = ms_data["ms"] if "ms" in ms_data else list(ms_data.values())[0]
    elif isinstance(ms_data, (list, tuple)):
        ms = ms_data[0]
    else:
        ms = ms_data

    print(f"Original MS shape: {ms.shape}")

    # Convert to tensor and add batch dimension
    if isinstance(ms, np.ndarray):
        ms_tensor = torch.from_numpy(ms).float()
    else:
        ms_tensor = ms.float()

    # The data is in (h, w, c) format, convert to (c, h, w) for PyTorch
    print(f"Converting from (h, w, c) to (c, h, w)")
    ms_tensor = ms_tensor.permute(2, 0, 1)  # (256, 256, 8) -> (8, 256, 256)

    print(f"MS tensor shape after permute: {ms_tensor.shape}")

    # Add batch dimension
    ms_tensor = ms_tensor.unsqueeze(0)  # (1, 8, 256, 256)
    c = ms_tensor.shape[1]
    print(f"Final tensor shape: {ms_tensor.shape} (bs, c, h, w)")

    # Test MTF convolution
    print("\nTesting MTF convolution...")
    mtf_conv = MTFConv(ratio=4, sensor="WV3", nbands=c)
    ms_mtf = mtf_conv(ms_tensor)
    print(f"MTF filtered shape: {ms_mtf.shape}")

    # Test PansharpSimulator
    print("\nTesting PansharpSimulator...")
    print("Use tap23 upsampling for better quality")
    simulator = PansharpSimulator(
        ratio=4,
        sensor="WV3",
        nbands=c,
        pan_weight_lst="mean",  # Use equal weights
        downsample_type="avgpool",
        upsample_type="tap23",
    )

    # Generate LRMS and PAN from HRMS
    lrms_upsampled, pan = simulator(ms_tensor)
    print(f"LRMS upsampled shape: {lrms_upsampled.shape}")
    print(f"PAN shape: {pan.shape}")

    # Create low-resolution MS for metric computation
    # Downsample original HRMS to get LRMS
    lrms_original = F.avg_pool2d(ms_tensor, kernel_size=4, stride=4)
    print(f"LRMS original shape: {lrms_original.shape}")

    # Initialize metrics computation
    print("\nComputing metrics...")

    # Try reference metrics first (comparing original HRMS vs upsampled LRMS)
    print("Computing reference metrics...")
    ref_metrics_fn = AnalysisPanAcc(
        ref=True,  # Reference mode
        ratio=4,
    )

    ref_metrics_result = ref_metrics_fn(
        ms_tensor / 2047.0,  # Ground truth (original HRMS)
        lrms_upsampled / 2047.0,  # Prediction (upsampled LRMS)
    )
    print("Reference metrics:")
    for key, value in ref_metrics_fn.acc_ave.items():
        print(f"  {key}: {float(value):.4f}")

    # Try unreferenced metrics
    print("\nComputing unreferenced metrics...")
    unref_metrics_fn = AnalysisPanAcc(
        ref=False,  # Unreferenced mode
        ratio=4,
        sensor="WV3",  # WorldView-3 sensor
    )

    # For unreferenced metrics, we need:
    # sr: super-resolved result (our upsampled LRMS)
    # ms: original low-resolution MS
    # lms: upsampled version of ms (should match sr size)
    # pan: panchromatic image

    # Create lms by upsampling the lrms_original to match hrms size
    lms = F.interpolate(
        lrms_original, scale_factor=4, mode="bilinear", align_corners=False
    )

    try:
        # Using 4 parameters: (sr, ms, lms, pan)
        unref_metrics_result = unref_metrics_fn(
            ms_tensor / 2047.0,  # Super-resolved result (sr)
            lrms_original / 2047.0,  # Original low-resolution MS (ms)
            lms / 2047.0,  # Upsampled LRMS (lms)
            pan / 2047.0,  # Panchromatic image (pan)
        )
        print("Unreferenced metrics:")
        for key, value in unref_metrics_fn.acc_ave.items():
            print(f"  {key}: {float(value):.4f}")
        metrics_result = unref_metrics_result
    except Exception as e:
        print(f"Unreferenced metrics failed: {e}")
        print("Falling back to reference metrics only")
        metrics_result = ref_metrics_result

    # Prepare images for visualization
    # Create comparison images: original HRMS vs upsampled LRMS
    comparison_images = []

    # Select RGB bands for visualization (assuming bands 2, 1, 0 for RGB)
    rgb_indices = [2, 1, 0] if c >= 3 else [0, 1, 2] if c >= 2 else [0, 0, 0]

    # Original HRMS (RGB)
    hrms_rgb = ms_tensor[0, rgb_indices].detach().cpu().numpy() / 2047.0
    hrms_rgb = np.transpose(hrms_rgb, (1, 2, 0))  # (h, w, c)
    comparison_images.append(hrms_rgb)

    # Upsampled LRMS (RGB)
    lrms_rgb = lrms_upsampled[0, rgb_indices].detach().cpu().numpy() / 2047.0
    lrms_rgb = np.transpose(lrms_rgb, (1, 2, 0))  # (h, w, c)
    comparison_images.append(lrms_rgb)

    # PAN image
    pan_img = pan[0, 0].detach().cpu().numpy() / 2047.0  # Remove batch and channel dims
    pan_img_rgb = np.stack([pan_img, pan_img, pan_img], axis=-1)  # Convert to 3-channel
    comparison_images.append(pan_img_rgb)

    # MTF filtered HRMS (RGB)
    mtf_rgb = ms_mtf[0, rgb_indices].detach().cpu().numpy() / 2047.0
    mtf_rgb = np.transpose(mtf_rgb, (1, 2, 0))  # (h, w, c)
    comparison_images.append(mtf_rgb)

    # Normalize images for better visualization
    def normalize_image(img):
        # if img.max() > 0:
        #     img = (img - img.min()) / (img.max() - img.min())
        return np.clip(img * 255, 0, 255).astype(np.uint8)

    comparison_images = [normalize_image(img) for img in comparison_images]

    # Save individual images
    print(f"\nSaving images to {tmp_dir}...")
    titles = ["HRMS_Original", "LRMS_Upsampled", "PAN", "HRMS_MTF_Filtered"]

    for i, (img, title) in enumerate(zip(comparison_images, titles)):
        save_path = tmp_dir / f"{title}.png"
        # Using matplotlib to save images
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(8, 8))
            plt.imshow(img)
            plt.title(title)
            plt.axis("off")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved: {save_path}")
        except Exception as e:
            print(f"Failed to save {title}: {e}")

    # Create comparison visualization
    try:
        print("Creating comparison visualization...")
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.flatten()

        for i, (img, title) in enumerate(zip(comparison_images, titles)):
            axes[i].imshow(img)
            axes[i].set_title(title)
            axes[i].axis("off")

        plt.tight_layout()
        comparison_path = tmp_dir / "comparison.png"
        plt.savefig(comparison_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved comparison: {comparison_path}")

    except Exception as e:
        print(f"Failed to create comparison: {e}")

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Original HRMS - Mean: {ms_tensor.mean():.4f}, Std: {ms_tensor.std():.4f}")
    print(f"MTF filtered - Mean: {ms_mtf.mean():.4f}, Std: {ms_mtf.std():.4f}")
    print(
        f"LRMS upsampled - Mean: {lrms_upsampled.mean():.4f}, Std: {lrms_upsampled.std():.4f}"
    )
    print(f"PAN - Mean: {pan.mean():.4f}, Std: {pan.std():.4f}")

    # Calculate PSNR between HRMS and LRMS upsampled
    mse = torch.mean((ms_tensor - lrms_upsampled) ** 2)
    psnr = -20 * torch.log10(1.0 / torch.sqrt(mse))
    print(f"PSNR (HRMS vs LRMS upsampled): {psnr:.2f} dB")

    print(f"\nAll test results saved to: {tmp_dir}")
    print(f"Unreferenced metrics computed: D_lambda, D_S, HQNR")

    return metrics_result


if __name__ == "__main__":
    test_mtf()
