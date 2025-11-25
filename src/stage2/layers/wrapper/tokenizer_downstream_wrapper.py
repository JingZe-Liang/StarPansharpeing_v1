"""
Tokenizer Wrapper to train a latent processing networks:
compatible with tasks: pansharpening, denoising, super-resolution

Deprecated: use DownstreamModelTokenizerWrapper
"""

import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing_extensions import deprecated

from src.stage1.cosmos.tokenizer_inference import (
    TokenizerInferenceWrapper,
    scale_shift_latent,
    un_scale_shift_latent,
)
from stage2.utilities.amotized.amotized_model_wrapper import AmotizedModelMixin

type LatentScaleShiftType = tuple[float, float] | tuple[list[float], list[float]] | None


@deprecated("Use DownstreamModelTokenizerWrapper")
class TokenizerHyperDownstreamWrapper(TokenizerInferenceWrapper, AmotizedModelMixin):
    def __init__(
        self,
        tokenizer: nn.Module,
        pixel_model: nn.Module,
        amotized_model: nn.Module,
        amotize_type: str,
        backward_decoder: bool = False,
        learn_decoder: bool = False,
        tokenizer_scale_shift: LatentScaleShiftType = None,
    ):
        TokenizerInferenceWrapper.__init__(tokenizer, tokenizer_scale_shift)
        decoder_fn = self.tokenizer.decoder if learn_decoder else self.tokenizer.decode
        AmotizedModelMixin.__init__(
            self,
            pixel_model,
            amotized_model,
            decoder_fn,
            amotize_type,
            backward_decoder,
            learn_decoder,
        )
        if self.learn_decoder:
            self.tokenizer.decoder.train()
            self.tokenizer.encoder.eval()
            self.tokenizer.requires_grad_(False)
        else:
            self.tokenizer.eval()
            self.tokenizer.requires_grad_(False)

    def _tuple_take_first(self, x: tuple | Tensor):
        if isinstance(x, tuple):
            return x[0]
        return x

    def _forward_downstream_model(self, pixel_in: tuple, latent_in: tuple):
        return self.forward_all_amotized_types(pixel_in, latent_in)

    def forward(self, pixel_in: Tensor | tuple, latent_in: Tensor | tuple | None = None):
        pixel_in = self._single_tensor_to_tuple(pixel_in)
        if latent_in is None:
            pixel_latents = []
            for px in pixel_in:
                with torch.no_grad():
                    latent = self.encode(px)
                pixel_latents.append(latent)

        return self._forward_downstream_model(pixel_in, pixel_latents)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        sd = {
            "scale_factor": self.scale_factor.data,
            "shift_factor": self.shift_factor.data,
            "pixel_model": self.pixel_model.state_dict(),
            "amotized_model": self.amotized_model.state_dict(),
        }
        if self.learn_decoder:
            sd["decoder"] = self.tokenizer.decoder.state_dict()
        return sd

    def load_state_dict(self, state_dict, strict=True):
        self.scale_factor.data = state_dict["scale_factor"]
        self.shift_factor.data = state_dict["shift_factor"]
        rets_pixel = self.pixel_model.load_state_dict(state_dict["pixel_model"], strict=strict)
        rets_amotize = self.amotized_model.load_state_dict(state_dict["amotized_model"], strict=strict)
        rets = {
            "pixel_model": rets_pixel,
            "amotized_model": rets_amotize,
        }
        if self.learn_decoder and "decoder" in state_dict:
            rets_decoder = self.tokenizer.decoder.load_state_dict(state_dict["decoder"], strict=strict)
            rets["decoder"] = rets_decoder
        return rets
