import inspect
from collections import deque
from types import MethodType
from typing import Sequence

import torch
import torch.nn as nn
from param import Callable
from torch import Tensor

from src.stage1.cosmos.cosmos_tokenizer import ContinuousImageTokenizer
from src.stage1.cosmos.lora_mixin import TokenizerLoRAMixin
from src.stage2.utilities.amotized.amotized_model_wrapper import AmotizedModelMixin
from src.utilities.logging import log


def _not_lora_not_implemented_raise(self, *args, **kwargs):
    raise AttributeError(
        f"This method is only available for LoRA tokenizer of class {self.__class__.__name__}."
    )


class ModelTokenizerWrapper(nn.Module):
    def __init__(
        self,
        tokenizer: ContinuousImageTokenizer | TokenizerLoRAMixin,
        downstream_model: nn.Module | AmotizedModelMixin,
        froze_tokenizer: bool = True,
        n_img_encoded: int = 1,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.downstream_model = downstream_model

        self.is_downstream_amotized = isinstance(
            self.downstream_model, AmotizedModelMixin
        )
        self._check_downstream_model_args()

        assert hasattr(self.tokenizer, "encode"), "tokenizer must have an encode method"
        assert hasattr(self.tokenizer, "decode"), "tokenizer must have a decode method"
        self._encode_fn = self.tokenizer.encode
        self._decode_fn = self.tokenizer.decode
        self.is_tokenizer_lora = isinstance(self.tokenizer, TokenizerLoRAMixin)

        if froze_tokenizer:
            self.tokenizer.eval()
            self.tokenizer.requires_grad_(False)
            if self.is_downstream_amotized and downstream_model.learn_decoder:
                self.downstream_model.decoder.requires_grad_(True)

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
        assert n_params == 2, (
            "downstream_model.forward must have two positional args (pixel_in, latent_in), found {n_params}"
        )

    def _update_lora_methods(self):
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
        if self.is_tokenizer_lora:
            self._update_methods_from_model(self.tokenizer, lora_tokenizer_methods)
        else:
            for method in lora_tokenizer_methods:
                setattr(self, method, MethodType(_not_lora_not_implemented_raise, self))

    def _update_methods_from_model(self, model, model_methods: list[str]):
        if not self.is_tokenizer_lora:
            return

        for method in model_methods:
            if hasattr(model, method):
                # bind the method to the current model but the function is from the lora model
                setattr(self, method, getattr(model, method))
            else:
                log(f"Method {method} not found in the lora model", level="warning")

    def _cache_img_chans(self, img: Tensor):
        self._img_chans_cache.appendleft(img.shape[1])

    @property
    def img_cached_channels(self):
        return list(self._img_chans_cache)

    @property
    def img_fifo_channel(self):
        if len(self._img_chans_cache) == 0:
            raise ValueError("No image channels cached. Please run forward once.")
        return self._img_chans_cache.pop()  # left in, right out: FIFO

    def clear_cached_img_channels(self):
        self._img_chans_cache.clear()

    def _forward_tokenizer_encode_single_img(self, img: Tensor) -> Tensor:
        latent_ret_ = self._encode_fn(img)
        # only return latent
        if isinstance(latent_ret_, tuple):  # latent, q_loss, loss_breakdown
            return latent_ret_[0]
        return latent_ret_

    def _forward_tokenizer_decode_single_latent(
        self, latent: Tensor, input_shape: torch.Size
    ) -> Tensor:
        decoded = self._decode_fn(latent, input_shape)
        if isinstance(decoded, tuple):
            return decoded[0]
        return decoded

    def _forward_tokenizer_encode(
        self,
        pixel_in: Tensor | Sequence[Tensor],
        latent_in: Tensor | Sequence[Tensor] | None = None,
    ) -> Tensor | Sequence[Tensor]:
        if latent_in is not None:
            return latent_in

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
        self,
        latents: Tensor | Sequence[Tensor],
        chans: int | Sequence[int],
    ) -> Tensor | Sequence[Tensor]:
        if isinstance(latents, torch.Tensor):
            assert isinstance(chans, int), "chans must be int if latents is Tensor"
            input_shape = torch.Size([latents.shape[0], chans])
            return self._forward_tokenizer_decode_single_latent(latents, input_shape)
        else:
            decoded = []
            assert isinstance(chans, (tuple, list)), (
                "chans must be list or tuple of int"
            )
            for latent_, chan_ in zip(latents, chans):
                input_shape = torch.Size([latent_.shape[0], chan_])
                decoded.append(
                    self._forward_tokenizer_decode_single_latent(latent_, input_shape)
                )
            return decoded

    def forward(
        self,
        pixel_in: torch.Tensor | tuple[torch.Tensor, ...],
        latent_in: Tensor | Sequence[Tensor] | None = None,
        **kwargs,
    ):
        latents = self._forward_tokenizer_encode(pixel_in, latent_in)
        # inputs should be img and latent
        return self.downstream_model(pixel_in, latents, **kwargs)

    def decode(
        self, latents: Tensor | Sequence[Tensor], chans: int | list[int] | None = None
    ):
        if chans is None:
            chans = self.img_cached_channels
            assert 0 < len(chans) < len(latents)
        else:
            assert len(chans) == len(latents)

        if torch.is_tensor(latents):
            if isinstance(chans, list):
                chans = chans[0]
            return self._decode_fn(latents, chans)
        else:
            assert isinstance(chans, list) and len(chans) == len(latents), (
                "chans must be a list of same length as latents"
            )
            recons = []
            for latent_, chan_ in zip(latents, chans):
                recons.append(self._decode_fn(latent_, chan_))
            return recons
