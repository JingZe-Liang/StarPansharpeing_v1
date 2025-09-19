import functools
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Literal, TypedDict, no_type_check

import torch
from torch import Tensor, nn
from torch.nn import Module

from src.utilities.config_utils import dataclass_from_dict
from src.utilities.logging import log


def amotizing_call_model(
    detokenizer: Module,
    pansp_model: Module,
    amotize_fn: Callable | str,
):
    def wrapper(lrms_latent, pan_latent, lrms, pan):
        amotize_call = (
            amotize_fn
            if isinstance(amotize_fn, Callable)
            else getattr(pansp_model, amotize_fn)
        )
        assert callable(amotize_call), "amotize_fn must be callable"

        latent_sr, pixel_amot = pansp_model(lrms_latent, pan_latent, lrms, pan)
        sr_detok = detokenizer(latent_sr)
        sr = amotize_call(sr_detok, pixel_amot)

        return latent_sr, sr

    return wrapper


# *==============================================================
# * Amotized Model (two branches)
# * latent branch amotizes the pixel branch
# * to get the final super-resolution image
# *==============================================================


ModelOutput = TypedDict(
    "ModelOutput",
    {
        "latent_out": Tensor,
        "pixel_from_latent": Tensor | None,
        "pixel_out": Tensor,
    },
)


class AmotizedForwardType(str, Enum):
    SIMPLE_MERGE_ADD = "simple_merge_add"
    PIXEL_TO_PIXEL_FUSION = "pixel_to_pixel_fusion"
    LATENT_TO_PIXEL_FUSION = "latent_to_pixel_fusion"
    ONE_MODEL = "one_model"


class AmotizedModelMixin(nn.Module):
    """
    # Amotized model

    # < version 1: simple merging
    #                                |-------> latent loss
    # latent -> latent branch -> sr latent -> de-tokenizer --- sr (not accurate) -> sr -> pixel loss
    # (lrms, pan) -> pixel branch ---------------------------------↑ (e.g., add)

    # < version 2: pixel to pixel fusion
    # latent -> latent branch -> sr latent -> de-tokenizer --- sr (not accurate)
    #                        ↓ -------------------------------------|
    # (lrms, pan) ---> pixel branch -> sr

    # < version 3: latent to pixel fusion
    # latent -> latent branch -> sr latent -> de-tokenizer --- sr (not accurate)
    #                        ↓ ------|
    # (lrms, pan) ---> pixel branch -> sr -> pixel loss

    # < version 4: pixel to latent fusion
    # latent -------------------------> big transformer -----> latent -> de-tokenizer -> pixel loss
    # pixels -----> CNN ---> patcher -----↑                     ↓----> latent loss (optional)

    """

    amotizing_pixels = True  # do not change

    def __init__(
        self,
        pixel_model: Module,
        amotized_model: Module,
        # take latent and size of orignal pixels
        decoder_fn: Module | Callable[[Tensor, torch.Size], Tensor],
        amotize_type: AmotizedForwardType
        | str = AmotizedForwardType.LATENT_TO_PIXEL_FUSION,
        backward_decoder: bool = False,
        learn_decoder: bool = False,
    ):
        super().__init__()
        self.pixel_model = pixel_model
        self.amotized_model = amotized_model
        self.amotize_type = amotize_type
        self.decoder = decoder_fn  # not saved in state_dict if not learn_decoder

        self.backward_decoder = backward_decoder
        self.learn_decoder = learn_decoder
        if self.learn_decoder:
            assert isinstance(self.decoder, nn.Module), (
                "decoder_fn must be a nn.Module if learn_decoder is True"
            )

        # amotizing type check
        if not (
            self.amotize_type
            in (
                "pixel_to_pixel_fusion",
                "latent_to_pixel_fusion",
                "one_model_conditioning",
            )
            or self.amotize_type[:11] == "simple_merge"
        ):
            raise ValueError(
                f"amotize_type {self.amotize_type} is not supported for AmotizedModel"
            )

    def _single_tensor_to_tuple(self, x: Tensor | tuple) -> tuple[Tensor | Any, ...]:
        return (x,) if torch.is_tensor(x) else x

    def _get_pixel_shape(self, pixel_in: tuple) -> torch.Size:
        # assume the first element is the main pixel input
        return pixel_in[0].shape

    def _decoded_to_pixel(self, decoded: Tensor | tuple) -> Tensor:
        return decoded if torch.is_tensor(decoded) else decoded[0]

    def simple_merge_forward(
        self, pixel_in: tuple, latent_in: tuple, amotize_type: str
    ):
        latent_out = self.amotized_model(*latent_in)
        shape = self._get_pixel_shape(pixel_in)
        # latent loss ? (if use, the pixel branch is just residual)
        amotized_pixel_from_latent = self.decoder(latent_out, shape)
        amotized_pixel_from_latent = self._decoded_to_pixel(amotized_pixel_from_latent)
        pixel_out = self.pixel_model(*pixel_in)
        # todo: add more complex merging logic
        if amotize_type == "add":
            out = (
                pixel_out + amotized_pixel_from_latent  # -> pixel loss
            )
        else:
            raise ValueError(f"Unknown amotize_type: {amotize_type}")

        return {
            "latent_out": latent_out,
            "pixel_from_latent": amotized_pixel_from_latent,
            "pixel_out": out,
        }

    def pixel_to_pixel_fusion_forward(self, pixel_in: tuple, latent_in: tuple):
        latent_out = self.amotized_model(*latent_in)  # -> latent loss
        shape = self._get_pixel_shape(pixel_in)
        amotized_pixel_from_latent = self.decoder(latent_out, shape)  # -> pixel loss
        amotized_pixel_from_latent = self._decoded_to_pixel(amotized_pixel_from_latent)
        # args: (pixel_in, amotized_pixel_from_latent)
        pixel_ins = pixel_in + self._single_tensor_to_tuple(amotized_pixel_from_latent)
        pixel_out = self.pixel_model(*pixel_ins)  # -> pixel loss

        return {
            "latent_out": latent_out,
            "pixel_from_latent": amotized_pixel_from_latent,
            "pixel_out": pixel_out,
        }

    def latent_to_pixel_fusion_forward(self, pixel_in: tuple, latent_in: tuple):
        latent_out = self.amotized_model(*latent_in)  # -> latent loss
        # args: pixel_in, latent_out
        pixel_model_ins = pixel_in + self._single_tensor_to_tuple(latent_out)
        pixel_out = self.pixel_model(*pixel_model_ins)  # -> pixel loss

        return {
            "latent_out": latent_out,
            "pixel_from_latent": None,  # No pixel from latent in this case, left to decoder out from forward
            "pixel_out": pixel_out,
        }

    def pixel_to_latent_forward(self, pixel_in: tuple, latent_in: tuple):
        pixel_out_as_cond = self.pixel_model(*pixel_in)
        model_conds = latent_in + self._single_tensor_to_tuple(pixel_out_as_cond)
        latent_out = self.amotized_model(*model_conds)  # -> latent loss

        pixel_out = None
        if self.backward_decoder or self.learn_decoder:
            shape = self._get_pixel_shape(pixel_in)
            pixel_out = self.decoder(latent_out, shape)  # -> pixel loss
            pixel_out = self._decoded_to_pixel(pixel_out)

        return {
            "latent_out": latent_out,
            "pixel_from_latent": pixel_out,
            "pixel_out": pixel_out,
        }

    @no_type_check
    def forward(
        self, pixel_in: Tensor | tuple, latent_in: Tensor | tuple
    ) -> ModelOutput:
        pixel_in = self._single_tensor_to_tuple(pixel_in)
        latent_in = self._single_tensor_to_tuple(latent_in)

        if (
            self.amotize_type[:11] == "simple_merge"
            or self.amotize_type == AmotizedForwardType.SIMPLE_MERGE_ADD
        ):
            return self.simple_merge_forward(
                pixel_in, latent_in, self.amotize_type.rsplit("_")[-1]
            )

        elif self.amotize_type == AmotizedForwardType.PIXEL_TO_PIXEL_FUSION:
            return self.pixel_to_pixel_fusion_forward(pixel_in, latent_in)

        elif self.amotize_type == AmotizedForwardType.LATENT_TO_PIXEL_FUSION:
            return self.latent_to_pixel_fusion_forward(pixel_in, latent_in)

        elif self.amotize_type == AmotizedForwardType.ONE_MODEL:
            return self.one_model_conditioning_forward(pixel_in, latent_in)

        else:
            raise ValueError(f"Unknown amotize_type: {self.amotize_type}")

    @classmethod
    def create_model(
        cls,
        amotized_model_class: type[nn.Module],
        pixel_model_class: type[nn.Module],
        amotized_cfg_cls: "type[dataclass]",
        pixel_cfg_cls: "type[dataclass]",
        amotized_kwargs: dict = {},
        pixel_kwargs: dict = {},
        mixin_kwargs: dict = {},
    ):
        if amotized_cfg_cls is not None:
            amotized_cfg = dataclass_from_dict(amotized_cfg_cls, amotized_kwargs)
            amotized_model = amotized_model_class(amotized_cfg)
        else:
            amotized_model = amotized_model_class(**amotized_kwargs)

        if pixel_cfg_cls is not None:
            pixel_cfg = dataclass_from_dict(pixel_cfg_cls, pixel_kwargs)
            pixel_model = pixel_model_class(pixel_cfg)
        else:
            pixel_model = pixel_model_class(**pixel_kwargs)

        return cls(
            pixel_model=pixel_model,
            amotized_model=amotized_model,
            **mixin_kwargs,
        )

    def set_checkpoint_mode(self, mode=True):
        for module in self.modules():
            if hasattr(module, "grad_checkpointing"):
                module.grad_checkpointing = mode
                log(f"Set grad_checkpointing={mode} for {module.__class__.__name__}")
