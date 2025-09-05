from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Callable, cast

import accelerate
import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, set_peft_model_state_dict

from src.stage1.cosmos.modules.blocks import (
    DiffBandsInputConvIn,
    DiffBandsInputConvOut,
)
from src.utilities.logging import log_print

# * --- Utilites --- #


def get_conv_in_out_modules(
    current_model: nn.Module,
    conv_in_name: str = "encoder.encoder.conv_in",
    conv_out_name: str = "decoder.decoder.conv_out",
) -> tuple[DiffBandsInputConvIn, DiffBandsInputConvOut]:
    conv_in_module = current_model.get_submodule(conv_in_name)
    conv_out_module = current_model.get_submodule(conv_out_name)

    return conv_in_module, conv_out_module  # type: ignore


def in_out_chans_from_modules(
    conv_in_module: DiffBandsInputConvIn, conv_out_module: DiffBandsInputConvOut
):
    in_chans = conv_in_module.band_lst
    out_chans = conv_out_module.band_lst

    assert in_chans == out_chans, f"{in_chans=}, but got {out_chans=}"

    return in_chans


def change_new_conv_in_out_modules(
    current_model: nn.Module,
    conv_in_name: str = "encoder.encoder.conv_in",
    conv_out_name: str = "decoder.decoder.conv_out",
    add_chans: list[int] | None = None,
    drop_chans: list[int] | None = None,
):
    """
    Dynamically add or drop convolution modules for specific channels.

    Args:
        current_model: The model to modify
        conv_in_name: Path to the input convolution module
        conv_out_name: Path to the output convolution module
        add_chans: List of channel numbers to add modules for
        drop_chans: List of channel numbers to remove modules for
    """
    conv_in_module, conv_out_module = get_conv_in_out_modules(
        current_model, conv_in_name, conv_out_name
    )

    # Call module-level methods to handle add/drop operations
    conv_in_module.add_or_drop_modules(add_chans=add_chans, drop_chans=drop_chans)
    conv_out_module.add_or_drop_modules(add_chans=add_chans, drop_chans=drop_chans)


# * --- LoRA Mixin --- #


class TokenizerLoRAMixin(nn.Module):
    def __init__(
        self,
        tokenizer: nn.Module,
        lora_weights: dict[str, str | Path],
        lora_hyper_chans: dict[str, int],
        active_lora: str | None = None,
        # Optional, since configs are in directories
        lora_cfg: dict | LoraConfig | None = None,
    ):
        """
        Initialize the TokenizerLoRAMixin with a base tokenizer and LoRA weights.

        Args:
            tokenizer: Base tokenizer model to be augmented with LoRA adapters
            lora_weights: Dictionary mapping LoRA names to their weight file paths
            active_lora: Name of the currently active LoRA adapter
            lora_cfg: LoRA configuration, can be a dictionary or LoraConfig object

        Note: if LoRA state dict has some key that tokenizer instance does not have,
              the 'from_pretrained' method will not even print any warning. Make sure
              all keys are loaded, especially the multi-experts Conv-In and Conv-Out in
              Cosmos tokenizer.
        """
        super().__init__()
        self._offload_model: nn.Module | None = None
        self.base_model: nn.Module | None = None
        self.model_peft: PeftModel | None = None
        self.current_lora: str | None = None
        self.current_lora_chan: int | None = None

        self.encode: Callable
        self.decode: Callable
        self.forward: Callable

        # Store LoRA info for lazy loading
        self.lora_weights = {
            str(name): str(path) for name, path in lora_weights.items()
        }

        # Lora channels
        self.base_model_chans = in_out_chans_from_modules(
            *get_conv_in_out_modules(tokenizer)
        )
        self.lora_names = list(self.lora_weights.keys())
        assert len(lora_hyper_chans) == len(self.lora_weights), (
            f"Lora channels={lora_hyper_chans} and lora weights={self.lora_weights} "
            "should have the equal length"
        )
        assert set(self.lora_names) == set(lora_hyper_chans.keys()), (
            f"Lora names={self.lora_names} and lora channels={list(lora_hyper_chans.keys())} "
            "should be equal"
        )
        self.lora_changed_chans = lora_hyper_chans

        # Keep for backward compatibility, but may not be needed
        self.lora_cfg = lora_cfg

        # Initialize with base model (no LoRA)
        self.set_base_model(tokenizer)

        # Activate one lora
        if active_lora is not None:
            self.change_lora(active_lora, merge=False)

    def set_base_model(self, model):
        """Set the base tokenizer"""
        assert model is not None, "model should be a nn.Module"
        model.requires_grad_(False)
        self._offload_model = deepcopy(model.cpu())  # offload to CPU
        self.base_model = model.cuda()
        log_print(f"Offload one copy of model into CPU and one base model to CUDA.")

    @property
    def actived_model(self):
        """Return the currently active model (base or PEFT)"""
        if self.model_peft is not None:
            return self.model_peft
        return self._tokenizer

    @property
    def offload_model(self):
        return self._offload_model

    @property
    def peft_config(self):
        if self.model_peft is not None:
            return self.model_peft.peft_config
        return {}

    def _update_methods_from_model(self, model):
        """Update encode/decode/forward methods from given model"""
        attrs = ["encode", "decode", "forward", "to", "cuda", "cpu", "train", "eval"]
        for attr in attrs:
            setattr(self, attr, getattr(model, attr))

    def _load_single_lora(self, lora_name: str):
        """Load a single LoRA adapter on demand"""
        if lora_name not in self.lora_weights:
            raise ValueError(
                f"LoRA '{lora_name}' not found in available weights: {list(self.lora_weights.keys())}"
            )
        assert self.base_model is not None, "base model is None"
        log_print(f"Loading LoRA '{lora_name}' from: {self.lora_weights[lora_name]}")

        # Add new lora channel
        lora_chan = self.lora_changed_chans[lora_name]
        _need_change_module = lora_chan not in self.base_model_chans
        if _need_change_module:
            change_new_conv_in_out_modules(self.base_model, add_chans=[lora_chan])

        # Load the specific LoRA directly from directory
        sd_path = Path(self.lora_weights[lora_name])
        if sd_path.is_dir():
            # Direct loading from directory - config is read automatically
            self.model_peft = PeftModel.from_pretrained(
                self.base_model,
                str(sd_path),
                adapter_name=lora_name,
                is_trainable=False,
                ignore_mismatched_sizes=True,  # load it anyway
            )
            loaded_result = "Loaded from directory with auto config"
        elif (
            sd_path.with_suffix(".safetensors").exists()
            or sd_path.with_suffix(".pt").exists()
        ):
            # Load from state dict file
            sd = accelerate.utils.load_state_dict(str(sd_path))
            # Try to load config from same directory
            config_path = sd_path.parent / "adapter_config.json"
            if config_path.exists():
                # Use config from file
                self.model_peft = PeftModel.from_pretrained(
                    self.base_model,
                    str(sd_path.parent),
                    adapter_name=lora_name,
                    is_trainable=False,
                    ignore_mismatched_sizes=True,  # load it anyway
                )
                # Then load the weights
                self.model_peft.add_adapter(
                    adapter_name=lora_name,
                    peft_config=self.model_peft.peft_config[lora_name],
                    low_cpu_mem_usage=False,
                )
                set_peft_model_state_dict(
                    self.model_peft,
                    sd,
                    adapter_name=lora_name,
                    ignore_mismatched_sizes=True,
                    low_cpu_mem_usage=False,
                )
                loaded_result = "Loaded weights with config from directory"
            else:
                raise ValueError(f"No config found for LoRA weights at {sd_path}")
        else:
            raise ValueError(f"Unsupported LoRA loading path: {sd_path}")

        log_print(
            f"Successfully loaded LoRA adapter: {lora_name}, result: {loaded_result}"
        )

        self.current_lora = lora_name
        self.current_lora_chan = self.lora_changed_chans[lora_name]

        # Set the active adapter
        self.model_peft.set_adapter(lora_name)
        self.current_lora_chan = self.lora_changed_chans[lora_name]

        # Update methods to use PEFT model instead of base tokenizer
        self._update_methods_from_model(self.model_peft)

    def change_lora(
        self, lora_name: str, merge=False, not_cache_action: str = "warning"
    ):
        """Change to a different LoRA adapter, loading it if necessary"""
        if lora_name == self.current_lora:
            log_print(f"Already using LoRA adapter: {lora_name}")
            return

        if lora_name not in self.lora_weights:
            available_adapters = list(self.lora_weights.keys())
            string = f"Adapter '{lora_name}' not found. Available adapters: {available_adapters}"
            if not_cache_action == "warning":
                log_print(string, "warning")
                return
            elif not_cache_action == "fallback":
                self.drop_current_lora()
                log_print(
                    f"Can not change lora {lora_name}, fall back to no lora base model."
                )
            elif not_cache_action in ("error", "raise"):
                raise ValueError(string)
            else:
                raise ValueError(f"Can not handle no_change_action: {not_cache_action}")

        # Load the new LoRA (this will replace any existing PEFT model)
        load_res = False
        try:
            self._load_single_lora(lora_name)
            load_res = True
        except Exception as e:
            log_print(
                f"Change lora {lora_name} failed, try to disable the current lora {self.current_lora}. "
                f"Then try to reload the required LoRA. "
                f"Error: {e}",
                level="warning",
            )
            self.drop_current_lora()
            self._load_single_lora(lora_name)
            load_res = True
        finally:
            if not load_res:
                raise RuntimeError(f"Failed to load LoRA adapter: {lora_name}")
            else:
                log_print(f"Change LoRA adapter <green>{lora_name}</>")

        if merge:
            self.merge_lora_weights()

    def merge_lora_weights(self):
        """Merge current LoRA weights into base model"""
        if self.model_peft is not None:
            self.model_peft = self.model_peft.merge_and_unload()
            # Update base tokenizer reference to the merged model
            self._tokenizer = self.model_peft
            self.model_peft = None
            self.current_lora = None
            # Update methods to use merged model
            self._update_methods_from_model(self._tokenizer)
            log_print("Merged current LoRA weights into base model")

    def merge_specific_lora(self, adapter_name: str | None = None):
        """Load and merge a specific LoRA adapter"""
        if adapter_name:
            self.change_lora(adapter_name)

        if self.model_peft is not None:
            self.model_peft = self.model_peft.merge_and_unload()
            # Update base tokenizer reference to the merged model
            self._tokenizer = self.model_peft
            self.model_peft = None
            self.current_lora = None
            # Update methods to use merged model
            self._update_methods_from_model(self._tokenizer)
            log_print(f"Merged LoRA weights: {adapter_name or 'current'}")

    @contextmanager
    def disable_lora(self):
        """Context manager to temporarily disable LoRA adapters."""
        current_state = self.current_lora
        try:
            self.model_peft.disable_adapter()
            yield
        finally:
            if current_state:
                self.change_lora(current_state)

    def drop_current_lora(self):
        """Disable LoRA and revert to base model"""
        if self.model_peft is not None:
            # Get the base model from PEFT model
            self.model_peft = None
            self.base_model = None
            self.base_model = deepcopy(self._offload_model)
            self.base_model.to("cuda")

            # Update methods to use base model
            self._update_methods_from_model(self.base_model)
            log_print(
                f"Drop LoRA <green>{self.current_lora}</>, reverted to base model"
            )

            self.current_lora = None
            self.current_lora_chan = None

            # Remove the added conv-in-out expert
            if self.current_lora_chan in self.base_model_chans:
                change_new_conv_in_out_modules(
                    self.base_model, drop_chans=[self.current_lora_chan]
                )

    def _enable_lora(self):
        """Re-enable current LoRA if exists"""
        if self.current_lora:
            self._load_single_lora(self.current_lora)

    def get_available_loras(self) -> list[str]:
        """Get list of available LoRA adapters"""
        return list(self.lora_weights.keys())

    def get_current_lora(self) -> str | None:
        """Get currently active LoRA adapter"""
        return self.current_lora

    def get_base_model(self):
        return self.base_model
