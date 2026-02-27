from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn

from .base import TeacherAdapter
from .dino_adapter import DinoTeacherAdapter, load_repa_dino_v2_model, load_repa_dino_v3_model
from .pe_adapter import PETeacherAdapter, load_perception_model
from .siglip_adapter import SiglipTeacherAdapter, load_siglip2_model


def load_repa_encoder(
    repa_name: str = "dinov2",
    model_name: str = "dinov2_vitb14",
    weight_path: str | Path | None = None,
    *,
    load_from: str = "torch",
    dino_v3_pretrained_on: Literal["satellite", "web"] = "satellite",
    compile: bool = True,
):
    if repa_name == "dinov2":
        return load_repa_dino_v2_model(load_from, model_name, weight_path, compile)
    if repa_name == "dinov3":
        return load_repa_dino_v3_model(
            weight_path,
            model_name,
            pretrained_on=dino_v3_pretrained_on,
            compile=compile,
        )
    if repa_name == "pe":
        assert weight_path is not None, "weight_path should not be None when loading PE model"
        return load_perception_model(weight_path, model_name, compile=compile)
    if repa_name == "siglip2":
        return load_siglip2_model(model_name)
    raise ValueError(f"Unknown DINO/PE version {repa_name}")


def build_teacher_adapter(
    *,
    repa_model_type: str,
    repa_model_name: str,
    repa_model_load_path: str | None,
    repa_encoder: nn.Module | tuple[nn.Module, object] | None,
    dino_load_type: str,
    dino_pretrained_on: Literal["satellite", "web"],
    c_dim_first: bool,
    img_is_neg1_1: bool,
    rgb_channels: list[int] | str | None,
    img_resize: tuple[int, int] | str | None,
    repa_img_size: int,
    dtype: torch.dtype,
    pca_fn: Callable[..., torch.Tensor] | None,
) -> TeacherAdapter:
    encoder_input = repa_encoder
    if encoder_input is None:
        encoder_input = load_repa_encoder(
            repa_name=repa_model_type,
            model_name=repa_model_name,
            weight_path=repa_model_load_path,
            load_from=dino_load_type,
            dino_v3_pretrained_on=dino_pretrained_on,
            compile=False,
        )

    if repa_model_type in {"dinov2", "dinov3"}:
        assert isinstance(encoder_input, nn.Module)
        encoder = encoder_input.to(dtype)
        encoder.requires_grad_(False)
        encoder.eval()
        return DinoTeacherAdapter(
            repa_encoder=encoder,
            dino_type=dino_load_type,
            repa_model_name=repa_model_name,
            c_dim_first=c_dim_first,
            img_is_neg1_1=img_is_neg1_1,
            rgb_channels=rgb_channels,
            img_resize=img_resize,
            pca_fn=pca_fn,
            dino_pretrained_on=dino_pretrained_on,
        )

    if repa_model_type == "siglip2":
        if isinstance(encoder_input, tuple):
            encoder, processor = encoder_input
        else:
            raise TypeError("Siglip2 adapter requires (encoder, processor) tuple")
        encoder = encoder.to(dtype)
        encoder.requires_grad_(False)
        encoder.eval()
        return SiglipTeacherAdapter(
            repa_encoder=encoder,
            processor=processor,
            repa_img_size=repa_img_size,
            img_is_neg1_1=img_is_neg1_1,
            rgb_channels=rgb_channels,
            pca_fn=pca_fn,
        )

    if repa_model_type == "pe":
        assert isinstance(encoder_input, nn.Module)
        encoder = encoder_input.to(dtype)
        encoder.requires_grad_(False)
        encoder.eval()
        return PETeacherAdapter(
            repa_encoder=encoder,
            repa_model_name=repa_model_name,
            img_is_neg1_1=img_is_neg1_1,
            rgb_channels=rgb_channels,
            img_resize=img_resize,
            pca_fn=pca_fn,
        )

    raise ValueError(f"Unknown model type: {repa_model_type}")
