import math
import os
import warnings
from contextlib import contextmanager
from functools import partial
from typing import Any, Callable, Dict, Literal, Union, cast

import hydra
import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
import webdataset as wds
from omegaconf import DictConfig, OmegaConf
from peft import PeftConfig
from tqdm import tqdm
from transformers import LongformerSelfAttention

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


def valid_value(x):
    return torch.isnan(x).sum() == 0 and torch.isinf(x).sum() == 0


# * --- Tokenizer processor --- #


class PansharpTokenizeProcessor(nn.Module):
    """
    Processes a batch of HRMS images to generate HRMS, LRMS, PAN, and their latent representations.
    Expects input batch dictionary containing at least {'img': hrms_tensor, '__key__': key_string}.
    """

    def __init__(
        self,
        pansharp_simulator: PansharpSimulator,
        tokenizer: nn.Module | None = None,
        before_save_fn: Callable | None = None,
        latent_save_backend: Literal["npy", "safetensors", "npz"] = "safetensors",
    ):
        super().__init__()
        self.pansharp_simulator = pansharp_simulator
        self.tokenizer: nn.Module | None = tokenizer
        self.has_tokenizer = self.tokenizer is not None
        if self.has_tokenizer:
            assert hasattr(self.tokenizer, "encode"), "Tokenizer must have an encode method"
            self.tokenizer.requires_grad_(False)
            self.tokenizer.eval()
            self.tokenizer = cast(nn.Module, self.tokenizer)
        self.before_save_fn = before_save_fn
        self.latent_save_backend = latent_save_backend

    @torch.no_grad()
    def forward(self, batch: Dict[str, Any]):
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
            log_print("Input batch missing 'img' or '__key__'. Skipping.", level="error")
            return None

        hrms = batch["img"]
        keys = batch["__key__"]  # keys should be a list/tuple of strings
        pan = None
        if "pan" in batch:
            pan = batch["pan"]

        # Ensure hrms is a tensor
        if not isinstance(hrms, torch.Tensor):
            log_print(
                f"Input 'img' must be a torch.Tensor, but got {type(hrms)}. Skipping batch.",
                level="error",
            )
            return None

        # Ensure hrms has 4 dimensions (bs, c, h, w)
        if hrms.ndim != 4:
            log_print(
                f"Input HRMS tensor must have 4 dimensions (bs, c, h, w), but got shape {hrms.shape}. Skipping batch.",
                level="error",
            )
            return None

        bs, c, h, w = hrms.shape

        try:
            hrms, pan = map(lambda x: x.to(torch.uint16), (hrms, pan))
            lrms, pan = self.pansharp_simulator(hrms, pan)
        except Exception as e:
            log_print(
                f"Error during pansharpening simulation for keys {keys}: {e}. Skipping batch.",
                level="error",
            )
            return None

        if self.has_tokenizer:
            assert pan.shape[1] == 1, "PAN image must have a single channel"
            pan_repeated = pan.repeat(1, c, 1, 1)  # pan_repeated: (bs, c, h, w)
            self.tokenizer: nn.Module
            hrms_latent = self.tokenizer.encode(hrms)
            lrms_latent = self.tokenizer.encode(lrms)
            pan_latent = self.tokenizer.encode(pan_repeated)

            # if not hasattr(self.tokenizer, "encode"):
            #     raise AttributeError(
            #         "The provided tokenizer object does not have an 'encode' method."
            #     )

        # To convert the images to target dtypes
        if self.before_save_fn is not None:
            hrms = self.before_save_fn(img=hrms)
            lrms = self.before_save_fn(img=lrms)
            pan = self.before_save_fn(img=pan)

        map(valid_value, (hrms, lrms, pan))

        # Let's modify this to return a list of dictionaries, one per sample.
        output_list = []
        for i in range(bs):
            # * --- will save the original images or not --- #
            sample_data = {
                "__key__": keys[i],
                # "hrms.tiff": tiff_codec_io(hrms[i]),
                # "lrms.tiff": tiff_codec_io(lrms[i]),
                # "pan.tiff": tiff_codec_io(pan[i]),
                "pair.npz": {"hrms": hrms[i], "lrms": lrms[i], "pan": pan[i]},
            }
            # Assuming latent tensors also have a batch dimension
            if self.has_tokenizer:
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
                    raise ValueError(f"Unsupported latent_save_backend: {self.latent_save_backend}")

            output_list.append(sample_data)

        return output_list  # Return list of dicts


def seperate_pansharpening_latent_pairs(
    sample: dict[str, Any], latent_ext: str = "safetensors"
) -> tuple[dict[str, Any], dict[str, Any]]:
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


def save_hwc_float_0_1(cfg, img, const: float | None = None, dtype: np.dtype | None = None) -> np.ndarray:
    # Convert to float32 and scale to [0, 1]
    if cfg.consts.to_neg_1_1:
        img = (img + 1) / 2

    img = img.permute(0, 2, 3, 1)  # (bs, c, h, w) -> (bs, h, w, c)
    img = img.cpu().numpy()
    img = np.clip(img, 0, 1)

    if const is not None and dtype is not None:
        img = (img * const).astype(dtype)
    else:
        log_print(
            "[save fn]: const and dtype, one of them is not provided, save the img using float32",
            warn_once=True,
        )
        img = img.astype("float32")

    return img


def save_chw_uint(cfg, img, dtype=np.uint16):
    # bs, c, h, w
    img = img.cpu().numpy()
    return img.astype(dtype)


@hydra.main(
    config_path="../configs/pansharpening_simulation",
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
    device = torch.device(f"cuda:{cfg.consts.run_device}" if torch.cuda.is_available() else "cpu")
    log_print(f"Using device: {device}")

    # 1. Instantiate components based on config
    log_print("Instantiating components...")
    pansharp_sim = PansharpSimulator(
        ratio=cfg.processor.ratio,
        sensor=cfg.processor.sensor,
        nbands=cfg.processor.nbands,  # Get nbands from data config
        pan_weight_lst=cfg.processor.pan_weights,
        downsample_type=cfg.processor.downsample_type,
        upsample_type=cfg.processor.upsample_type,
    )
    pansharp_sim = pansharp_sim.to(device)

    # loading tokenizer
    if cfg.tokenizer is not None:
        tokenizer: tuple[PeftConfig, nn.Module] | nn.Module | None = hydra.utils.instantiate(cfg.tokenizer)
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

    else:
        tokenizer = None
        log_print("No tokenizer provided, skipping latent generation.")

    before_save_fn_ = save_chw_uint if cfg.consts.save_fn == "chw_uint" else save_hwc_float_0_1
    processor = PansharpTokenizeProcessor(
        pansharp_simulator=pansharp_sim,
        tokenizer=tokenizer,
        before_save_fn=partial(before_save_fn_, cfg=cfg),
        latent_save_backend="npz",
    )
    # 2. Setup WebDataset pipeline using get_hyperspectral_dataloaders
    log_print(f"Setting up WebDataset pipeline: {cfg.data.wds_paths} -> {cfg.output.output_dir} TAR files")

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
    progress_bar = tqdm(input_dataloader, desc="Processing Batches", unit="batch")

    sink_manager = TarSinkManager(output_dir)
    for batch in progress_bar:
        # Move input batch data to the correct device
        batch_on_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Process the batch
        processed_output = processor(batch_on_device)

        # Handle the output (list of sample dicts or empty dict on error)
        if isinstance(processed_output, list):
            for sample_data, tar_path in zip(
                processed_output,
                batch_on_device.get("__url__", batch.get("__shard__", None)),
            ):
                if cfg.consts.tar_file_name is not None:
                    name = cfg.consts.tar_file_name
                elif tar_path is not None:
                    name = os.path.basename(tar_path).replace(" ", "_")
                else:
                    name = "default.tar"

                pansharp_sink = sink_manager.get_sink(f"pansharpening_{name}", f"pansharpening_reduced/{name}")

                if tokenizer is not None:
                    latent_sink = sink_manager.get_sink(f"latent_{name}", f"pansharpening_latents/{name}")

                    # Seperate Pansharpening pairs and latents
                    pansharp_sample, latent_sample = seperate_pansharpening_latent_pairs(
                        sample_data, processor.latent_save_backend
                    )

                    assert latent_sample is not None, "Latent sample is None"
                    assert pansharp_sample is not None, "Pansharpening sample is None"
                else:
                    pansharp_sample = sample_data
                    latent_sample = None

                if sample_data:  # Check if the dictionary is not empty
                    pansharp_sink.write(pansharp_sample)
                    if latent_sample is not None:
                        latent_sink.write(latent_sample)

                    total_samples += 1
                    count += 1  # Count samples processed in this batch run
            # Update progress bar description periodically
            if count >= 100:
                progress_bar.set_postfix({"samples_written": total_samples})
                count = 0  # Reset counter for periodic update
        elif processed_output is None:
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

    log_print(f"Finished processing. {total_samples} samples written to {cfg.output.output_dir}")


# Entry point for Hydra
if __name__ == "__main__":
    from loguru import logger

    with logger.catch():
        process_dataset()
