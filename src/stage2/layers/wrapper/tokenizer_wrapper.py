import inspect
from collections import deque
from functools import partial
from types import MethodType
from typing import Callable, cast

import torch
import torch.nn as nn
from loguru import logger
from peft import PeftMixedModel, PeftModel, get_peft_model
from peft.tuners.tuners_utils import BaseTuner
from torch import Tensor

from src.stage1.cosmos.cosmos_tokenizer import ContinuousImageTokenizer
from src.stage1.cosmos.lora_mixin import TokenizerLoRAMixin
from src.stage1.cosmos.tokenizer_inference import (
    scale_shift_latent,
    un_scale_shift_latent,
)
from src.stage2.utilities.amotized.amotized_model_wrapper import AmotizedModelMixin
from src.utilities.config_utils import function_config_to_basic_types
from src.utilities.logging import log

type LatentScaleShiftType = tuple[float, float] | tuple[list[float], list[float]] | None


############### Utilities functions ##############


def _not_lora_not_implemented_raise(self, *args, **kwargs):
    raise AttributeError(
        f"This method is only available for LoRA tokenizer of class {self.__class__.__name__}."
    )


def _partial_amotized_model_decode_fn(tokenizer):
    def decode_fn(x, inp_shape):
        return tokenizer.decode(x, inp_shape)

    return decode_fn


############# Tokenizer wrapper with downstream model ##############


class DownstreamModelTokenizerWrapper(nn.Module):
    @function_config_to_basic_types
    def __init__(
        self,
        tokenizer: ContinuousImageTokenizer
        | TokenizerLoRAMixin
        | PeftModel
        | PeftMixedModel,
        downstream_model: nn.Module | AmotizedModelMixin | partial,
        froze_tokenizer: bool = True,
        n_img_encoded: int = 1,
        tokenizer_scale_shift: LatentScaleShiftType = None,
        tokenizer_img_processor: Callable | None = None,
        detokenizer_img_processor: Callable | None = None,
        img_is_neg_1_1: bool = False,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.downstream_model: nn.Module = downstream_model
        self.is_scale_latent = tokenizer_scale_shift is not None
        self.img_is_neg_1_1 = img_is_neg_1_1

        # Merge the adpter weights
        if isinstance(self.tokenizer, (PeftMixedModel, PeftModel)):
            self.tokenizer = self.tokenizer.base_model.merge_and_unload()
            log("Merge and unload the PEFT adapter weights.")

        # Latent value scaler
        if self.is_scale_latent:
            scale, shift = tokenizer_scale_shift
            scale, shift = torch.as_tensor(scale), torch.as_tensor(shift)
            self.scale = nn.Buffer(scale, persistent=False)
            self.shift = nn.Buffer(shift, persistent=False)

        # Image processors
        self.tokenizer_img_processor = tokenizer_img_processor
        if tokenizer_img_processor is not None:
            assert callable(tokenizer_img_processor), (
                "tokenizer_img_processor must be callable"
            )

        self.detokenizer_img_processor = detokenizer_img_processor
        if detokenizer_img_processor is not None:
            assert callable(detokenizer_img_processor), (
                "detokenizer_img_processor must be callable"
            )

        # low-level tasks wrapper
        self.is_downstream_amotized = isinstance(
            self.downstream_model, AmotizedModelMixin
        )
        # is a partial inited class
        if isinstance(self.downstream_model, partial):
            decoder_fn = _partial_amotized_model_decode_fn(self.tokenizer)
            self.downstream_model = self.downstream_model(decoder_fn=decoder_fn)
        self.downstream_model = cast(nn.Module, self.downstream_model)

        self._check_downstream_model_args()

        assert hasattr(self.tokenizer, "encode"), "tokenizer must have an encode method"
        assert hasattr(self.tokenizer, "decode"), "tokenizer must have a decode method"
        self._encode_fn = self.tokenizer.encode
        self._decode_fn = self.tokenizer.decode
        self.is_tokenizer_lora = isinstance(self.tokenizer, TokenizerLoRAMixin)
        self.is_peft_lora = isinstance(self.tokenizer, (PeftMixedModel, PeftModel))
        if self.is_peft_lora:
            self._base_peft_model: BaseTuner = self.tokenizer.base_model

        # Freeze tokenizer by default
        self.tokenizer_decoder_learnt = getattr(
            self.downstream_model, "learn_decoder", False
        )
        if froze_tokenizer:
            self.tokenizer.eval()
            self.tokenizer.requires_grad_(False)
            if self.is_downstream_amotized and self.tokenizer_decoder_learnt:
                assert not self.is_tokenizer_lora, (
                    'When using LoRA tokenizer, "learn_decoder" must be false.'
                )
                self.tokenizer.decoder.requires_grad_(True)
            logger.info(f"TokenizerWrapper: froze_tokenizer={froze_tokenizer}")

        self._img_chans_cache = deque(maxlen=n_img_encoded)

        # Register lora methods to the wrapper
        self._update_lora_methods()
        # hints
        self.actived_model: Callable
        self.offload_model: Callable
        self.peft_config: Callable
        self.change_lora: Callable
        self.merge_lora_weights: Callable
        self.merge_specific_lora: Callable
        self.disable_lora: Callable
        self.drop_current_lora: Callable
        self.get_available_loras: Callable
        self.get_current_lora: Callable
        self.get_base_model: Callable

    def change_lora(self, adapter_name: str, merge=True):
        """Change the active LoRA adapter for the tokenizer."""
        if self.is_tokenizer_lora:
            self.tokenizer.change_lora(adapter_name)  # Online loading adapter
        elif self.is_peft_lora:
            self.tokenizer = cast(PeftModel, self.tokenizer)
            # Unmerge the activated adapter, set the new adapter, and merge again
            if merge:
                assert getattr(self.tokenizer, "_base_peft_model", None) is not None, (
                    "_base_peft_model not found in the PeftModel tokenizer."
                )
                self._base_peft_model.unmerge_adapter()
            self.tokenizer.set_adapter(adapter_name)
            if merge:
                self._base_peft_model.merge_and_unload()
        else:
            raise ValueError(
                f"Tokenizer type {type(self.tokenizer)} not supported change LoRA weights."
            )

    def load_lora(self, path_to_adapter: str, adapter_name: str):
        if self.is_tokenizer_lora:
            raise ValueError(
                f"The LoraMixin tokenizer should load LoRA adapters in __init__"
            )
            # TODO: add the tokenizer.lora_weights may work? since it's online adapter loading.
        elif isinstance(self.tokenizer, PeftModel):
            self.tokenizer.load_adapter(
                path_to_adapter,
                adapter_name=adapter_name,
                is_trainable=False,
                ignore_mismatched_sizes=True,  # load it anyway
            )
            logger.debug(f"Load the Peft adapter: {adapter_name}")
        elif isinstance(self.tokenizer, PeftMixedModel):
            self.tokenizer.load_adapter(
                path_to_adapter,
                adapter_name=adapter_name,
                is_trainable=False,
                ignore_mismatched_sizes=True,  # load it anyway
            )
            logger.debug(f"Load the PeftMixed adapter: {adapter_name}")
        else:
            raise ValueError(
                f"Unsupported tokenizer type: {type(self.tokenizer)} for LoRA loading."
            )

    def _check_downstream_model_args(self):
        # Check if downstream model forward has two args, ignore **kwargs
        sig = inspect.signature(self.downstream_model.forward)
        params = sig.parameters
        n_params = len(
            [
                p
                for p in params.values()
                if p.kind
                in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL)
            ]
        )
        assert n_params >= 2, (
            "downstream_model.forward must have more than two positional args "
            f"(pixel_in, latent_in, ...), found {n_params}"
        )

    def _update_lora_methods(self):
        if self.is_tokenizer_lora:
            lora_tokenizer_methods = [
                "actived_model",
                "offload_model",
                "peft_config",
                "change_lora",
                "merge_lora_weights",
                "merge_specific_lora",
                "disable_lora",
                "drop_current_lora",
                "get_available_loras",
                "get_current_lora",
                "get_base_model",
            ]
            self._update_methods_from_model(self.tokenizer, lora_tokenizer_methods)
        else:
            # for method in lora_tokenizer_methods:
            #     setattr(self, method, MethodType(_not_lora_not_implemented_raise, self))

            # Do not attach methods
            pass

    def _update_methods_from_model(self, model, model_methods: list[str]):
        if not self.is_tokenizer_lora:
            return

        for method in model_methods:
            if hasattr(model, method):
                # bind the method to the current model but the function is from the lora model
                setattr(self, method, getattr(model, method))
            else:
                logger.debug(f"Method {method} not found in the lora model")

    def _cache_img_chans(self, img: Tensor):
        self._img_chans_cache.appendleft(img.shape[1])

    @property
    def img_cached_channels(self):
        return list(self._img_chans_cache)

    def _map_to_neg_1_1(self, x):
        if self.img_is_neg_1_1:
            return x
        return x * 2.0 - 1.0

    def _map_to_0_1(self, x):
        if self.img_is_neg_1_1:
            return (x + 1.0) / 2
        return

    @property
    def img_fifo_channel(self):
        if len(self._img_chans_cache) == 0:
            raise ValueError(
                "No image channels cached. Please run forward at least once."
            )
        return self._img_chans_cache.pop()  # left in, right out: FIFO

    def _clear_cached_img_channels(self):
        self._img_chans_cache.clear()

    @torch.no_grad()
    def _forward_tokenizer_encode_single_img(self, img: Tensor) -> Tensor:
        img = self._map_to_neg_1_1(img)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            latent_ret_ = self._encode_fn(img)

        # only return latent
        if isinstance(latent_ret_, (tuple, list)):  # latent, q_loss, loss_breakdown
            latent = latent_ret_[0]
        else:
            latent = latent_ret_
        # scale and shift latent
        if self.is_scale_latent:
            latent = scale_shift_latent(latent, self.scale, self.shift)

        return latent.type_as(img)

    def _forward_tokenizer_decode_single_latent(
        self, latent: Tensor, input_shape: torch.Size | int
    ) -> Tensor:
        # un-scale and un-shift latent
        if self.is_scale_latent:
            latent = un_scale_shift_latent(latent, self.scale, self.shift)
        ctx = (
            torch.no_grad()
            if not self.downstream_model.learn_decoder
            else torch.enable_grad()
        )
        with ctx:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                decoded = self._decode_fn(latent, input_shape)

        if isinstance(decoded, tuple):
            return decoded[0]

        return decoded

    def _forward_tokenizer_encode(
        self,
        pixel_in: Tensor | list[Tensor],
        latent_in: Tensor | list[Tensor] | None = None,
    ) -> Tensor | list[Tensor]:
        if latent_in is not None:
            return latent_in

        # process image for tokenizer
        if self.tokenizer_img_processor is not None:
            pixel_in = self.tokenizer_img_processor(*pixel_in)

        if torch.is_tensor(pixel_in):
            self._cache_img_chans(pixel_in)
            latent = self._forward_tokenizer_encode_single_img(pixel_in)
            return latent
        else:
            latents = []
            for img_ in pixel_in:
                self._cache_img_chans(img_)
                latents.append(self._forward_tokenizer_encode_single_img(img_))
            return latents

    def _forward_tokenizer_decode(
        self, latents: Tensor | list[Tensor], chans: int | list[int]
    ) -> Tensor | list[Tensor]:
        if isinstance(latents, torch.Tensor):
            assert isinstance(chans, int), "chans must be int if latents is Tensor"
            decoded = self._forward_tokenizer_decode_single_latent(latents, chans)
        else:
            decoded = []
            assert isinstance(chans, (tuple, list)), (
                "chans must be list or tuple of int"
            )
            for latent_, chan_ in zip(latents, chans):
                decoded.append(
                    self._forward_tokenizer_decode_single_latent(latent_, chan_)
                )
        # process image for detokenizer
        if self.detokenizer_img_processor is not None:
            decoded = self.detokenizer_img_processor(decoded)
        return decoded

    def forward(
        self,
        pixel_in: Tensor | list[Tensor],
        # already and scale-shifted latents
        latent_in: Tensor | list[Tensor] | None = None,
        **kwargs,
    ):
        latents = self._forward_tokenizer_encode(pixel_in, latent_in)
        # inputs should be img and latent
        return self.downstream_model(pixel_in, latents, **kwargs)

    def encode(self, img: Tensor) -> Tensor:
        return self._forward_tokenizer_encode_single_img(img)

    def decode(
        self,
        latents: Tensor | list[Tensor],
        chans: int | list[int] | None = None,
    ) -> Tensor | list[Tensor]:
        if chans is None:
            chans = self.img_cached_channels
            assert 0 < len(chans) < len(latents)
        else:
            # if isinstance(chans, int) and torch.is_tensor(latents):
            #     chans = [chans] * len(latents)
            if isinstance(chans, list):
                assert len(chans) == len(latents), (
                    "Channels should be a list of length equal to the number of latents"
                )

        return self._forward_tokenizer_decode(latents, chans)

    def _only_first_from_tuple(self, tok: tuple[torch.Tensor, ...], disable=False):
        if disable:
            return tok
        if isinstance(tok, (tuple, list)):
            return tok[0]
        return tok

    def state_dict(self, *args, **kwargs):
        # only save the first tokenizer if it is a lora tokenizer with multiple models
        # learn_detok = getattr(self.downstream_model, "learn_decoder", False)

        # Filter out
        filter_out_fn = lambda k: k.startswith("downstream_model")
        if self.tokenizer_decoder_learnt:
            if self.is_peft_lora:
                filter_out_fn = lambda k: not k.startswith("tokenizer.decoder")
            elif self.is_tokenizer_lora:
                filter_out_fn = lambda k: not k.startswith("active_model.decoder")

        sd_out = {}
        for k, v in super().state_dict().items():
            if filter_out_fn(k):
                sd_out[k] = v

        return sd_out

    def load_state_dict(self, state_dict, strict=False, **kwargs):
        try:
            super().load_state_dict(state_dict, strict=strict)
        except Exception as e:
            logger.error(f"[Load state dict]: {e}")
            from src.utilities.network_utils import load_weights_with_shape_check

            only_load_downstream = True
            if only_load_downstream:
                d = {
                    k[len("downstream_model.") :]: v
                    for k, v in state_dict.items()
                    if k.startswith("downstream_model")
                }
                load_weights_with_shape_check(self.downstream_model, d)

    def set_grad_checkpointing(self, mode=True):
        for module in self.modules():
            if hasattr(module, "grad_checkpointing"):
                module.grad_checkpointing = mode
                log(f"Set grad_checkpointing={mode} for {module.__class__.__name__}")

    def parameters(self):
        if self.tokenizer_decoder_learnt:
            raise NotImplementedError("Not implemented yet")
        else:
            # return only downstream model parameters
            return self.downstream_model.parameters()

    def named_parameters(self):
        if self.tokenizer_decoder_learnt:
            raise NotImplementedError("Not implemented yet")
        else:
            # return only downstream model parameters
            named_ps_downstream = self.downstream_model.named_parameters()
            # add the prefix
            # named_ps = [(f"downstream_model.{n}", p) for n, p in named_ps_downstream]
            # return named_ps
            for n, p in named_ps_downstream:
                yield f"downstream_model.{n}", p
