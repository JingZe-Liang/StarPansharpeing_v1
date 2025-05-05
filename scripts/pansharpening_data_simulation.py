import math
import os
import sys
import warnings
from functools import partial
from typing import Any, Callable, Dict, Literal

import hydra
import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
import webdataset as wds
from omegaconf import DictConfig, OmegaConf
from scipy.signal import windows
from tqdm import tqdm

sys.path.insert(0, __file__[: __file__.find("scripts")])
from src.data.codecs import npz_codec_io, safetensors_codec_io, tiff_codec_io
from src.utilities.logging import log_print

warnings.filterwarnings(
    "once",
    "[save fn]: const and dtype, one of them is not provided, save the img using float32",
    append=True,
)
warnings.filterwarnings(
    "ignore",
    module="torch.utils.checkpoint",
)


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

        # Reshape kernels for grouped convolution: [out_channels=c, in_channels_per_group=1, kH, kW]
        # Kernels are already in shape [c, 1, H, W]
        y = F.conv2d(
            x,
            self.kernels,  # Shape: [c, 1, H, W]
            padding=padding,
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

    def __init__(self, ratio: int):
        super().__init__()

        if not (ratio > 0 and (ratio & (ratio - 1) == 0)):
            raise ValueError("Error: Only resize factors power of 2 are supported.")
        self.ratio = ratio
        self.num_upsamples = int(math.log2(ratio))

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
                upsampled, (self.padding, self.padding, 0, 0), mode="circular"
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

    def forward(self, hrms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
                align_corners=False,
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
                lrms_upsampled, size=(h, w), mode="bilinear", align_corners=False
            )

        # --- 4. Generate PAN image --- (Weighted sum of HRMS bands)
        # Ensure pan_weights is on the same device as hrms
        pan = torch.sum(hrms * self.pan_weights.to(device), dim=1, keepdim=True)

        return lrms_upsampled, pan


class PansharpTokenizeProcessor(nn.Module):
    """
    Processes a batch of HRMS images to generate HRMS, LRMS, PAN, and their latent representations.
    Expects input batch dictionary containing at least {'img': hrms_tensor, '__key__': key_string}.
    """

    def __init__(
        self,
        pansharp_simulator: PansharpSimulator,
        tokenizer: nn.Module,
        before_save_fn: Callable | None = None,
        latent_save_backend: Literal["npy", "safetensors", "npz"] = "safetensors",
    ):
        super().__init__()
        self.pansharp_simulator = pansharp_simulator
        self.tokenizer = tokenizer
        self.before_save_fn = before_save_fn
        self.latent_save_backend = latent_save_backend
        assert hasattr(self.tokenizer, "encode"), "Tokenizer must have an encode method"

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a batch of data.

        Args:
            batch (Dict[str, Any]): Input batch dictionary. Must contain:
                'img': HRMS tensor (bs, c, h, w)
                '__key__': List or tuple of keys for each sample in the batch.

        Returns:
            Dict[str, Any]: Output dictionary containing processed data for webdataset,
                            or an empty dict if processing fails for a sample.
                            Keys: '__key__', 'hrms', 'lrms', 'pan', 'hrms_latent',
                                  'lrms_latent', 'pan_latent'.
                            Tensors are moved to CPU.
        """
        if "img" not in batch or "__key__" not in batch:
            log_print(
                "Input batch missing 'img' or '__key__'. Skipping.", level="error"
            )
            return {}

        hrms = batch["img"]
        keys = batch["__key__"]  # keys should be a list/tuple of strings

        # Ensure hrms is a tensor
        if not isinstance(hrms, torch.Tensor):
            log_print(
                f"Input 'img' must be a torch.Tensor, but got {type(hrms)}. Skipping batch.",
                level="error",
            )
            return {}

        # Ensure hrms has 4 dimensions (bs, c, h, w)
        if hrms.ndim != 4:
            log_print(
                f"Input HRMS tensor must have 4 dimensions (bs, c, h, w), but got shape {hrms.shape}. Skipping batch.",
                level="error",
            )
            return {}

        bs, c, h, w = hrms.shape

        # --- 1. Degrade HRMS using Wald's protocol ---
        try:
            lrms, pan = self.pansharp_simulator(hrms)
        except Exception as e:
            log_print(
                f"Error during pansharpening simulation for keys {keys}: {e}. Skipping batch.",
                level="error",
            )
            return {}

        # --- 2. Prepare PAN for tokenizer (repeat single channel) ---
        pan_repeated = pan.repeat(1, c, 1, 1)  # pan_repeated: (bs, c, h, w)

        # --- 3. Tokenize HRMS, LRMS, and repeated PAN ---
        if not hasattr(self.tokenizer, "encode"):
            raise AttributeError(
                "The provided tokenizer object does not have an 'encode' method."
            )

        with torch.no_grad():
            hrms_latent = self.tokenizer.encode(hrms)
            lrms_latent = self.tokenizer.encode(lrms)
            pan_latent = self.tokenizer.encode(pan_repeated)

        # --- 4. Prepare output dictionary for webdataset ---
        # Output structure needs careful handling with webdataset when processing batches.
        # TarWriter typically expects one dictionary per sample.
        # We need to yield one dictionary per item in the batch.
        # This processor is better suited for use with .map before batching, or
        # the main loop needs to unpack the batch and write samples individually.

        # To convert the images to target dtypes
        hrms, lrms, pan = map(lambda x: x.cpu().numpy(), (hrms, lrms, pan))
        if self.before_save_fn is not None:
            hrms, lrms, pan = map(self.before_save_fn, (hrms, lrms, pan))

        # Let's modify this to return a list of dictionaries, one per sample.
        output_list = []
        for i in range(bs):
            # * --- will save the original images or not --- #
            sample_data = {
                "__key__": keys[i],
                # "hrms.tiff": tiff_codec_io(hrms[i], tifffile.PLANARCONFIG.SEPARATE),
                # "lrms.tiff": tiff_codec_io(lrms[i], tifffile.PLANARCONFIG.SEPARATE),
                # "pan.tiff": tiff_codec_io(
                #     pan[i], photometric=tifffile.PHOTOMETRIC.MINISBLACK
                # ),
            }
            # Assuming latent tensors also have a batch dimension
            if self.latent_save_backend == "safetensors":
                _ext = "safetensors"
                sample_data[f"latents.{_ext}"] = safetensors_codec_io(
                    {
                        "hrms_latent": hrms_latent[i],
                        "lrms_latent": lrms_latent[i],
                        "pan_latent": pan_latent[i],
                    }
                )
            elif self.latent_save_backend == "npy":
                # numpy fn
                codec_fn = lambda x: x.detach().cpu().numpy()
                _ext = "npy"
                sample_data[f"hrms_latent.{_ext}"] = codec_fn(hrms_latent[i])
                sample_data[f"lrms_latent.{_ext}"] = codec_fn(lrms_latent[i])
                sample_data[f"pan_latent.{_ext}"] = codec_fn(pan_latent[i])
            elif self.latent_save_backend == "npz":
                _ext = "npz"
                sample_data[f"latents.{_ext}"] = npz_codec_io(
                    {
                        "hrms_latent": hrms_latent[i].cpu().numpy(),
                        "lrms_latent": lrms_latent[i].cpu().numpy(),
                        "pan_latent": pan_latent[i].cpu().numpy(),
                    },
                    do_compression=True,  # small enough data, no need for compression
                )
            else:
                raise ValueError(
                    f"Unsupported latent_save_backend: {self.latent_save_backend}"
                )

            # Add any other metadata from the input batch if needed (assuming it's per-sample)
            # for k, v in batch.items():
            #     if k not in ['img', '__key__'] and k not in sample_data:
            #         if isinstance(v, (list, tuple)) and len(v) == bs:
            #              sample_data[k] = v[i]
            #         # Handle other potential batch structures for metadata

            output_list.append(sample_data)

        # Since the caller (main loop) expects a single dict per batch iteration,
        # returning a list here might break the flow if not handled correctly.
        # A better approach might be to keep the processor working on single samples
        # and use .map before .batched in the dataloader setup.
        # For now, returning the list, assuming the main loop will handle it.
        # *** IMPORTANT: The main loop below needs adjustment to handle this list ***
        return output_list  # Return list of dicts


@hydra.main(
    config_path="configs/pansharpening_simulation",
    config_name="pan_wv3_simulation",
    version_base=None,
)
def process_dataset(cfg: DictConfig) -> None:
    """
    Main processing function driven by Hydra configuration.
    """
    log_print("Starting dataset processing with configuration:")
    log_print(OmegaConf.to_yaml(cfg, resolve=True))

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_print(f"Using device: {device}")

    # 1. Instantiate components based on config
    log_print("Instantiating components...")
    pansharp_sim = PansharpSimulator(
        ratio=cfg.processor.ratio,
        sensor=cfg.processor.sensor,
        nbands=cfg.processor.nbands,  # Get nbands from data config
        pan_weight_lst=(
            OmegaConf.to_container(cfg.processor.pan_weights)
            if not isinstance(cfg.processor.pan_weights, str)
            else cfg.processor.pan_weights
        ),  # Convert to list
        downsample_type=cfg.processor.downsample_type,
    )
    pansharp_sim = pansharp_sim.to(device)

    tokenizer = hydra.utils.instantiate(cfg.tokenizer)
    tokenizer = tokenizer.to(device)
    # load state dict
    if cfg.consts.tokenizer_checkpoint_path is not None:
        _imcompact_keys = tokenizer.load_state_dict(
            torch.load(cfg.consts.tokenizer_checkpoint_path, map_location=device),
            strict=False,
        )
        log_print(f"Imcompact keys: {_imcompact_keys}")
    else:
        log_print(
            f"No checkpoint path provided for {tokenizer.__class__.__name__}\n"
            "make sure you load checkpoint in the tokenizer"
        )

    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad = False  # Freeze parameters for inference
    log_print(f"Loaded tokenizer from {tokenizer.__class__.__name__}")

    def before_save_fn(
        img: np.ndarray, const: float | None = None, dtype: np.dtype | None = None
    ) -> np.ndarray:
        # Convert to float32 and scale to [0, 1]
        if cfg.data.to_neg_1_1:
            img = (img + 1) / 2

        img = np.clip(img, 0, 1)
        if const is not None and dtype is not None:
            img = (img * const).astype(dtype)
        else:
            warnings.warn(
                "[save fn]: const and dtype, one of them is not provided, save the img using float32"
            )
            img = img.astype(np.float32)

        return img

    processor = PansharpTokenizeProcessor(
        pansharp_simulator=pansharp_sim,
        tokenizer=tokenizer,
        before_save_fn=before_save_fn,
        latent_save_backend="safetensors",
    )
    # Note: Processor itself doesn't need .to(device) unless it has its own parameters/buffers
    # The submodules (pansharp_sim, tokenizer) are already moved.

    # 2. Setup WebDataset pipeline using get_hyperspectral_dataloaders
    log_print(
        f"Setting up WebDataset pipeline: {cfg.data.wds_paths} -> {cfg.output.output_dir} TAR files"
    )

    # We need the dataloader, not the dataset object returned by the function
    # The dataloader yields batches of dictionaries like {'img': tensor, '__key__': [keys...]}
    _, input_dataloader = hydra.utils.instantiate(cfg.data)

    # 3. Run the pipeline and write output
    output_dir = cfg.output.output_dir
    os.makedirs(output_dir, exist_ok=True)
    log_print(f"Output directory created: {output_dir}")

    log_print("Starting processing...")
    count = 0
    total_samples = 0  # Keep track of total samples written

    # Wrap the dataloader with tqdm for progress bar
    # Note: If the dataloader length is unknown (common with WebDataset), tqdm might not show total.
    # Consider adding an estimate if possible, or just let it run without total.
    try:
        # Estimate length if possible (might not be accurate with resampled WebDataset)
        # estimated_len = len(input_dataloader)
        # progress_bar = tqdm(input_dataloader, total=estimated_len, desc="Processing Batches")
        progress_bar = tqdm(input_dataloader, desc="Processing Batches", unit="batch")
    except TypeError:
        # If len() is not supported
        progress_bar = tqdm(input_dataloader, desc="Processing Batches", unit="batch")

    sinks: dict[str, wds.TarWriter] = {}
    for batch in progress_bar:
        # Move input batch data to the correct device
        batch_on_device = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Process the batch
        processed_output = processor(batch_on_device)

        # Handle the output (list of sample dicts or empty dict on error)
        if isinstance(processed_output, list):
            for sample_data, tar_path in zip(
                processed_output, batch_on_device["__url__"]
            ):
                name = os.path.basename(tar_path).replace(" ", "_")
                tar_path = os.path.join(output_dir, name)
                output_sink = sinks.get(name, None)
                if output_sink is None:
                    output_sink = sinks[name] = wds.TarWriter(tar_path)
                    log_print(f"Writing to {tar_path}")

                if sample_data:  # Check if the dictionary is not empty
                    output_sink.write(sample_data)
                    total_samples += 1
                    count += 1  # Count samples processed in this batch run
            # Update progress bar description periodically
            if count >= 100:
                progress_bar.set_postfix({"samples_written": total_samples})
                count = 0  # Reset counter for periodic update
        elif isinstance(processed_output, dict) and not processed_output:
            # Batch processing failed entirely
            log_print(
                f"Skipping batch due to processing error (keys: {batch.get('__key__', 'unknown')}).",
                level="warning",
            )
        else:
            # Unexpected output format from processor
            log_print(
                f"Unexpected output format from processor: {type(processed_output)}. Skipping batch.",
                level="warning",
            )

    # close all sinks
    for sink in sinks.values():
        sink.close()
        print("Closed all TAR writters")

    log_print(
        f"Finished processing. {total_samples} samples written to {cfg.output.output_dir}"
    )


# Entry point for Hydra
if __name__ == "__main__":
    from loguru import logger

    with logger.catch():
        process_dataset()
