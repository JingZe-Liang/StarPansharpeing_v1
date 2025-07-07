import math
import os
import warnings
from contextlib import contextmanager
from functools import partial
from typing import Any, Callable, Dict, Literal, Union

import hydra
import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
import webdataset as wds
from omegaconf import DictConfig, OmegaConf
from peft import PeftConfig
from scipy.signal import windows
from tqdm import tqdm

from src.data.codecs import npz_codec_io, safetensors_codec_io, tiff_codec_io
from src.data.tar_utils import TarSinkManager, tar_sink_manager
from src.stage2.pansharpening.simulator import (
    Interp23Tap,
    MTFConv,
    PansharpSimulator,
    genMTF,
)
from src.utilities.config_utils import function_config_to_basic_types
from src.utilities.logging import log_print
from src.utilities.network_utils import load_diffbands_tokenizer_then_peft_lora

warnings.filterwarnings(
    "once",
    "[save fn]: const and dtype, one of them is not provided, save the img using float32",
    append=True,
)
warnings.filterwarnings(
    "ignore",
    module="torch.utils.checkpoint",
)

# * --- Tokenizer processor --- #


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
                "The provided tokenizer objeqt does not have an 'encode' method."
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
                "hrms.tiff": tiff_codec_io(hrms[i], tifffile.PLANARCONFIG.SEPARATE),
                "lrms.tiff": tiff_codec_io(lrms[i], tifffile.PLANARCONFIG.SEPARATE),
                "pan.tiff": tiff_codec_io(
                    pan[i], photometric=tifffile.PHOTOMETRIC.MINISBLACK
                ),
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


def seperate_pansharpening_latent_pairs(
    sample: dict[str, Any], latent_ext: str = "safetensors"
):
    """
    Separate the latent pairs from the sample dictionary.

    Args:
        sample (dict): Input sample dictionary containing latent pairs.

    Returns:
        dict: Dictionary with separated latent pairs.
    """

    pansharp_sample = {
        "__key__": sample["__key__"],
        "hrms.tiff": sample["hrms.tiff"],
        "lrms.tiff": sample["lrms.tiff"],
        "pan.tiff": sample["pan.tiff"],
    }

    latent_sample = {
        "__key__": sample["__key__"],
    }
    if latent_ext == "safetensors":
        latent_sample["latents.safetensors"] = sample["latents.safetensors"]
    elif latent_ext == "npy":
        latent_sample["hrms_latent.npy"] = sample["hrms_latent.npy"]
        latent_sample["lrms_latent.npy"] = sample["lrms_latent.npy"]
        latent_sample["pan_latent.npy"] = sample["pan_latent.npy"]
    elif latent_ext == "npz":
        latent_sample["latents.npz"] = sample["latents.npz"]
    else:
        raise ValueError(f"Unsupported latent_save_backend: {latent_ext}")

    return pansharp_sample, latent_sample


# * --- Main entry --- #


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
    device = torch.device(
        f"cuda:{cfg.consts.run_device}" if torch.cuda.is_available() else "cpu"
    )
    log_print(f"Using device: {device}")

    # 1. Instantiate components based on config
    log_print("Instantiating components...")
    pansharp_sim = PansharpSimulator(
        ratio=cfg.processor.ratio,
        sensor=cfg.processor.sensor,
        nbands=cfg.processor.nbands,  # Get nbands from data config
        pan_weight_lst=cfg.processor.pan_weights,
        downsample_type=cfg.processor.downsample_type,
    )
    pansharp_sim = pansharp_sim.to(device)

    # loading tokenizer
    tokenizer: tuple[PeftConfig, nn.Module] | nn.Module = hydra.utils.instantiate(
        cfg.tokenizer
    )
    if isinstance(tokenizer, tuple):
        assert len(tokenizer) == 2, (
            "Tokenizer instantiation returns a tuple, must be a PEFT config and a lora-loaded network"
        )
        peft_config, tokenizer = tokenizer
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
    tokenizer.requires_grad_(False)  # Freeze tokenizer parameters
    log_print(f"Prepared tokenizer from {tokenizer.__class__.__name__}")

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
        latent_save_backend="npz",
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

    # sinks: dict[str, wds.TarWriter] = {}
    sink_manager = TarSinkManager(output_dir)
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
                # tar_path = os.path.join(output_dir, name)
                # output_sink = sinks.get(name, None)
                pansharp_sink = sink_manager.get_sink(
                    f"pansharpening_{name}", f"pansharpening_pairs/{name}"
                )
                latent_sink = sink_manager.get_sink(f"latent_{name}", f"latents/{name}")

                # Seperate Pansharpening pairs and latents
                pansharp_sample, latent_sample = seperate_pansharpening_latent_pairs(
                    sample_data, processor.latent_save_backend
                )

                assert pansharp_sample is not None, "Pansharpening sample is None"
                assert latent_sample is not None, "Latent sample is None"

                if sample_data:  # Check if the dictionary is not empty
                    pansharp_sink.write(pansharp_sample)
                    latent_sink.write(latent_sample)

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
    sink_manager.close_all()

    log_print(
        f"Finished processing. {total_samples} samples written to {cfg.output.output_dir}"
    )


# Entry point for Hydra
if __name__ == "__main__":
    from loguru import logger

    with logger.catch():
        process_dataset()
