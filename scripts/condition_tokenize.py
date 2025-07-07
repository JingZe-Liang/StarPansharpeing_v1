"""
Script to tokenize condition images using cosmos_tokenizer and save as webdataset.
Reads condition images from wids multimodal loader and tokenizes them using cosmos_tokenizer.
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Literal, Union

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from peft import PeftConfig
from tqdm import tqdm

from src.data.codecs import npy_codec_io, npz_codec_io, safetensors_codec_io
from src.data.panshap_loader import MultimodalityDataloader
from src.data.tar_utils import TarSinkManager
from src.utilities.logging import log_print, set_logger_file

warnings.filterwarnings("ignore", module="torch.utils.checkpoint")


class ConditionTokenizeProcessor(nn.Module):
    """
    Processes condition images to generate their latent representations using cosmos_tokenizer.
    """

    def __init__(
        self,
        tokenizer: nn.Module,
        condition_keys: List[str],
        latent_save_backend: Literal["safetensors", "npz", "npy"] = "safetensors",
        dtype=torch.bfloat16,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.condition_keys = condition_keys
        self.latent_save_backend = latent_save_backend
        assert hasattr(self.tokenizer, "encode"), "Tokenizer must have an encode method"

    def forward(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Processes a batch of condition images.

        Args:
            batch (Dict[str, Any]): Input batch dictionary containing condition images.
                Expected keys: '__key__', '__index__', condition keys (e.g., 'hed', 'mlsd', etc.)

        Returns:
            List[Dict[str, Any]]: List of output dictionaries containing tokenized latents.
        """
        if "__key__" not in batch:
            log_print("Input batch missing '__key__'. Skipping.", level="error")
            return []

        keys = batch["__key__"]
        batch_size = len(keys)

        # Process each condition type
        condition_latents = {}

        for condition_key in self.condition_keys:
            if condition_key not in batch:
                log_print(
                    f"Condition key '{condition_key}' not found in batch. Skipping.",
                    level="warning",
                )
                continue

            condition_imgs = batch[condition_key]

            # Ensure condition_imgs is a tensor
            if not isinstance(condition_imgs, torch.Tensor):
                log_print(
                    f"Condition '{condition_key}' must be a torch.Tensor, got {type(condition_imgs)}. Skipping.",
                    level="error",
                )
                continue  # Tokenize the condition images

            try:
                with torch.no_grad():
                    latents = self.tokenizer.encode(condition_imgs)  # type: ignore
                    condition_latents[condition_key] = latents
            except Exception as e:
                log_print(
                    f"Error tokenizing condition '{condition_key}': {e}. Skipping.",
                    level="error",
                )
                continue

        # Prepare output list
        output_list = []
        for i in range(batch_size):
            sample_data = {
                "__key__": keys[i],
            }

            # Add latents for each condition
            if self.latent_save_backend == "safetensors":
                latent_dict = {}
                for condition_key, latents in condition_latents.items():
                    latent_dict[f"{condition_key}_latent"] = (
                        latents[i].to(torch.bfloat16).cpu()
                    )

                if latent_dict:
                    sample_data["condition_latents.safetensors"] = safetensors_codec_io(
                        latent_dict
                    )

            elif self.latent_save_backend == "npz":
                latent_dict = {}
                for condition_key, latents in condition_latents.items():
                    latent_dict[f"{condition_key}_latent"] = (
                        latents[i].float().cpu().numpy()
                    )

                if latent_dict:
                    sample_data["condition_latents.npz"] = npz_codec_io(
                        latent_dict, do_compression=True
                    )

            elif self.latent_save_backend == "npy":
                latent_dict = {}
                for condition_key, latents in condition_latents.items():
                    latent_dict[f"{condition_key}_latent.npy"] = npy_codec_io(
                        latents[i].float().cpu().numpy()  # up-cast to float32
                    )
                if latent_dict:
                    sample_data.update(latent_dict)

            else:
                raise ValueError(
                    f"Unsupported latent_save_backend: {self.latent_save_backend}"
                )

            output_list.append(sample_data)

        return output_list


def tokenize_conditions_from_wids(
    wids_paths: Dict[str, Union[str, Path]],
    output_dir: str,
    tokenizer: nn.Module,
    condition_keys: List[str],
    batch_size: int = 8,
    num_workers: int = 1,
    latent_save_backend: Literal["safetensors", "npz", "npy"] = "safetensors",
    device: str = "cuda",
) -> int:
    """
    Tokenize condition images from wids multimodal loader and save as webdataset.

    Args:
        wids_paths: Dictionary mapping modality names to wids index file paths
        output_dir: Output directory for saving tokenized latents
        tokenizer: Tokenizer model for encoding conditions
        condition_keys: List of condition keys to tokenize
        batch_size: Batch size for processing
        num_workers: Number of workers for data loading
        latent_save_backend: Backend for saving latents
        device: Device for processing

    Returns:
        int: Total number of samples processed
    """
    log_print(f"Starting condition tokenization...")
    log_print(f"Condition keys: {condition_keys}")
    log_print(f"Output directory: {output_dir}")
    log_print(f"Device: {device}")

    # Create processor
    processor = ConditionTokenizeProcessor(
        tokenizer=tokenizer,
        condition_keys=condition_keys,
        latent_save_backend=latent_save_backend,
    )

    # Create multimodal dataloader
    datasets, dataloader = MultimodalityDataloader.create_loader(
        wds_paths=wids_paths,
        batch_size=batch_size,
        num_workers=num_workers,
        to_neg_1_1=False,  # Keep original image values
        shuffle_size=-1,  # No shuffling for processing
    )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    log_print(f"Output directory created: {output_dir}")

    # Initialize sink manager
    sink_manager = TarSinkManager(output_dir)

    total_samples = 0
    progress_bar = tqdm(
        dataloader,
        desc="Processing condition batches",
        unit="batch",
        total=len(datasets) // batch_size,
    )

    for batch in progress_bar:
        # Move batch to device
        batch_on_device = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch_on_device[k] = v.to(device)
            else:
                batch_on_device[k] = v

        # Process the batch
        processed_output = processor(batch_on_device)

        # Save processed samples
        if processed_output:
            for sample_data in processed_output:
                # Use original tar name structure
                sample_key = sample_data["__key__"]
                # Extract original tar name from batch metadata if available
                tar_name = "condition_latents.tar"  # Default name
                if "__shard__" in batch:
                    # Extract original shard name
                    shard_info = (
                        batch["__shard__"][0]
                        if isinstance(batch["__shard__"], list)
                        else batch["__shard__"]
                    )
                    if isinstance(shard_info, str):
                        original_tar_name = Path(shard_info).name
                        tar_name = original_tar_name

                # Get sink for this tar in condition_latents folder
                sink = sink_manager.get_sink(
                    tar_name.replace(".tar", ""), f"condition_latents/{tar_name}"
                )

                # Create clean sample with only latents and key
                clean_sample = {"__key__": sample_data["__key__"]}

                # Add only latent data
                for key, value in sample_data.items():
                    if key != "__key__":  # Only add latent files
                        clean_sample[key] = value

                # Write sample
                sink.write(clean_sample)
                total_samples += 1

        # Update progress
        progress_bar.set_postfix({"samples_processed": total_samples})

    # Close all sinks
    sink_manager.close_all()
    progress_bar.close()

    log_print(f"Finished processing. {total_samples} samples written to {output_dir}")
    return total_samples


@hydra.main(
    config_path="configs/condition_tokenization",
    config_name="condition_tokenize_config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """
    Main function for condition tokenization with Hydra configuration.
    """
    log_print("Starting condition tokenization with configuration:")
    log_print(OmegaConf.to_yaml(cfg, resolve=True))

    # Determine device
    device = torch.device(
        f"cuda:{cfg.consts.run_device}" if torch.cuda.is_available() else "cpu"
    )
    log_print(f"Using device: {device}")

    # Load tokenizer
    log_print("Loading tokenizer...")
    tokenizer: Union[tuple[PeftConfig, nn.Module], nn.Module] = hydra.utils.instantiate(
        cfg.tokenizer
    )

    if isinstance(tokenizer, tuple):
        assert len(tokenizer) == 2, (
            "Tokenizer instantiation returns a tuple, must be a PEFT config and a lora-loaded network"
        )
        peft_config, tokenizer = tokenizer

    tokenizer = tokenizer.to(device)

    # Load state dict if provided
    if cfg.consts.tokenizer_checkpoint_path is not None:
        checkpoint = torch.load(
            cfg.consts.tokenizer_checkpoint_path, map_location=device
        )
        incompatible_keys = tokenizer.load_state_dict(checkpoint, strict=False)
        log_print(f"Loaded checkpoint with incompatible keys: {incompatible_keys}")
    else:
        log_print("No checkpoint path provided, using tokenizer as-is")

    tokenizer.eval()
    tokenizer.requires_grad_(False)
    log_print(f"Prepared tokenizer: {tokenizer.__class__.__name__}")

    # Get condition keys from config or auto-detect
    condition_keys = cfg.processor.get("condition_keys", None)
    if condition_keys is None:
        # Auto-detect from sample data if not specified
        log_print("Auto-detecting condition keys from sample data...")
        sample_datasets, sample_loader = MultimodalityDataloader.create_loader(
            wds_paths=cfg.data.wids_paths,
            batch_size=1,
            num_workers=0,
            shuffle_size=-1,
        )

        # Get first sample to detect keys
        sample = next(iter(sample_loader))
        condition_keys = [k for k in sample.keys() if not k.startswith("__")]
        log_print(f"Auto-detected condition keys: {condition_keys}")
    else:
        log_print(f"Using configured condition keys: {condition_keys}")

    # Run tokenization
    total_samples = tokenize_conditions_from_wids(
        wids_paths=cfg.data.wids_paths,
        output_dir=cfg.output.output_dir,
        tokenizer=tokenizer,
        condition_keys=condition_keys,
        batch_size=cfg.data.get("batch_size", 8),
        num_workers=cfg.data.get("num_workers", 1),
        latent_save_backend=cfg.processor.get("latent_save_backend", "safetensors"),
        device=str(device),
    )

    log_print(
        f"Condition tokenization completed. Total samples processed: {total_samples}"
    )


if __name__ == "__main__":
    from loguru import logger

    with logger.catch():
        main()
