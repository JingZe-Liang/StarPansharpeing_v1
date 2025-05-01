import sys
from typing import Any, Callable, Dict

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import webdataset as wds
from omegaconf import DictConfig, OmegaConf
from scipy.signal import windows
from tqdm import tqdm

from src.data.codecs import safetensors_codec_io, tiff_codec_io

sys.path.insert(0, __file__[: __file__.find("src")])
from src.utilities.logging import log_print


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
    """Upsamples by a factor of `scale` using 23-tap filters.

    Args:
        channels (int): Number of channels.
        scale (int): Upsampling scale factor.
    """

    def __init__(self, channels: int, scale: int):
        super().__init__()
        self.scale = scale
        self.channels = channels
        # Simplified implementation using bilinear interpolation for now
        # A true 23-tap filter implementation would be more complex
        log_print(
            "Warning: Using simplified bilinear interpolation for Interp23Tap.",
            level="warning",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsamples the input tensor.

        Args:
            x (torch.Tensor): Input tensor (bs, c, h, w).

        Returns:
            torch.Tensor: Upsampled tensor (bs, c, h*scale, w*scale).
        """
        if x.shape[1] != self.channels:
            raise ValueError(
                f"Input channels ({x.shape[1]}) must match Interp23Tap channels ({self.channels})"
            )
        return F.interpolate(
            x, scale_factor=self.scale, mode="bilinear", align_corners=False
        )


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
            self.upsample = Interp23Tap(channels=nbands, scale=ratio)

        pan_weights = torch.tensor(actual_pan_weight_lst, dtype=torch.float32).view(
            1, -1, 1, 1
        )
        self.register_buffer("pan_weights", pan_weights)

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
        latent_save_backend: str = "numpy",
    ):
        super().__init__()
        self.pansharp_simulator = pansharp_simulator
        self.tokenizer = tokenizer
        self.before_save_fn = before_save_fn
        self.latent_save_backend = latent_save_backend

        assert self.latent_save_backend in [
            "numpy",
            "safetensors",
        ], 'latent_save_backend must be "numpy" or "safetensors"'
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
            sample_data = {
                "__key__": keys[i],
                "hrms.tiff": tiff_codec_io(hrms[i], "seperate"),
                "lrms.tiff": tiff_codec_io(lrms[i], "seperate"),
                "pan.tiff": tiff_codec_io(pan[i], photometric="minisblack"),
            }
            # Assuming latent tensors also have a batch dimension
            if self.latent_save_backend == "safetensors":
                codec_fn = lambda x: safetensors_codec_io(
                    {"latent": x.cpu()}
                )  # leave the dtype same as it
                _ext = "safetensors"
            else:
                # numpy fn
                codec_fn = lambda x: x.detach().cpu().numpy()
                _ext = "npy"

            sample_data[f"hrms_latent.{_ext}"] = codec_fn(hrms_latent[i])
            sample_data[f"lrms_latent.{_ext}"] = codec_fn(lrms_latent[i])
            sample_data[f"pan_latent.{_ext}"] = codec_fn(pan_latent[i])

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
    config_path="src/stage2/data/config",
    config_name="pan_wv3_simulation",
    version_base=None,
)
def process_dataset(cfg: DictConfig) -> None:
    """
    Main processing function driven by Hydra configuration.
    """
    log_print("Starting dataset processing with configuration:")
    log_print(OmegaConf.to_yaml(cfg))

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_print(f"Using device: {device}")

    # 1. Instantiate components based on config
    log_print("Instantiating components...")
    pansharp_sim = PansharpSimulator(
        ratio=cfg.processor.ratio,
        sensor=cfg.processor.sensor,
        nbands=cfg.processor.nbands,  # Get nbands from data config
        pan_weight_lst=OmegaConf.to_container(
            cfg.processor.pan_weights
        ),  # Convert to list
        downsample_type=cfg.processor.downsample_type,
    )
    pansharp_sim = pansharp_sim.to(device)

    tokenizer = hydra.utils.instantiate(cfg.model.tokenizer)
    tokenizer = tokenizer.to(device)
    # load state dict
    _imcompact_keys = tokenizer.load_state_dict(
        torch.load(cfg.model.tokenizer_checkpoint_path, map_location=device),
        strict=False,
    )
    tokenizer.eval()
    log_print(f"Loaded tokenizer from {tokenizer.__class__.__name__}")
    log_print(f"Imcompact keys: {_imcompact_keys}")

    def before_save_fn(img: np.ndarray, const: float, dtype: np.dtype) -> np.ndarray:
        # Convert to float32 and scale to [0, 1]
        if cfg.data.to_neg_1_1:
            img = (img + 1) / 2

        img = np.clip(img, 0, 1)
        img = (img * const).astype(dtype)

        return img

    processor = PansharpTokenizeProcessor(
        pansharp_simulator=pansharp_sim,
        tokenizer=tokenizer,
        before_save_fn=before_save_fn,
    )
    # Note: Processor itself doesn't need .to(device) unless it has its own parameters/buffers
    # The submodules (pansharp_sim, tokenizer) are already moved.

    # 2. Setup WebDataset pipeline using get_hyperspectral_dataloaders
    log_print(
        f"Setting up WebDataset pipeline: {cfg.data.wds_paths} -> {cfg.output.output_wds_path}"
    )

    # We need the dataloader, not the dataset object returned by the function
    # The dataloader yields batches of dictionaries like {'img': tensor, '__key__': [keys...]}
    _, input_dataloader = hydra.utils.instantiate(cfg.data)

    # 3. Run the pipeline and write output
    output_sink = wds.ShardWriter(cfg.output.output_wds_path)
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
            for sample_data in processed_output:
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

    output_sink.close()
    log_print(
        f"Finished processing. {total_samples} samples written to {cfg.output.output_wds_path}"
    )


# Entry point for Hydra
if __name__ == "__main__":
    process_dataset()
