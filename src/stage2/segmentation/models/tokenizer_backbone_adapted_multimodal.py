from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
from jaxtyping import Float
from loguru import logger
from omegaconf import OmegaConf
from torch import Tensor

from src.utilities.config_utils import function_config_to_basic_types

from .tokenizer_backbone_adapted import TokenizerHybridUNet, _create_default_cfg as _create_base_cfg


def split_modal_tensor(x: Tensor, modal_channels: Sequence[int]) -> list[Tensor]:
    if x.ndim != 4:
        raise ValueError(f"Expected [B, C, H, W], got shape {tuple(x.shape)}")
    expected_channels = int(sum(int(c) for c in modal_channels))
    if x.shape[1] != expected_channels:
        raise ValueError(
            f"Input channels mismatch: got {x.shape[1]}, expected {expected_channels} from modal_channels={list(modal_channels)}"
        )
    split_sizes = [int(c) for c in modal_channels]
    return list(torch.split(x, split_sizes, dim=1))


def _create_default_cfg() -> Any:
    cfg = _create_base_cfg()
    cfg.multimodal = OmegaConf.create(
        {
            "modalities": ["hsi", "msi", "sar"],
            "modal_channels": [10, 4, 2],
            "fusion_type": "concat",
        }
    )
    return cfg


class MultimodalTokenizerHybridUNet(TokenizerHybridUNet):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        self.mm_cfg = cfg.get("multimodal", None)
        if self.mm_cfg is None:
            raise ValueError("Missing multimodal config.")

        self.modalities: tuple[str, ...] = tuple(self.mm_cfg.get("modalities", []))
        self.modal_channels: list[int] = [int(c) for c in self.mm_cfg.get("modal_channels", [])]
        self.fusion_type: str = str(self.mm_cfg.get("fusion_type", "concat"))

        if len(self.modalities) == 0:
            raise ValueError("multimodal.modalities must not be empty.")
        if len(self.modalities) != len(self.modal_channels):
            raise ValueError(
                f"Length mismatch: modalities={len(self.modalities)}, modal_channels={len(self.modal_channels)}"
            )
        if self.fusion_type not in {"concat", "mean"}:
            raise ValueError(f"Unsupported fusion_type={self.fusion_type}, expected concat/mean")

        self.encoder_in_channels: int = int(self.tok_f_cfg.in_channels)
        if self.encoder_in_channels <= 0:
            raise ValueError(f"tokenizer_feature.in_channels must be > 0, got {self.encoder_in_channels}")

        n_modal = len(self.modal_channels)
        skip_channels = [int(c) for c in self.tok_f_cfg.features_per_stage]
        latent_channels = int(self.cfg.tokenizer.cnn_cfg.model.latent_channels)

        if self.fusion_type == "concat":
            self.skip_fusions = nn.ModuleList(
                [nn.Conv2d(ch * n_modal, ch, kernel_size=1, bias=False) for ch in skip_channels]
            )
            self.latent_fusion = nn.Conv2d(latent_channels * n_modal, latent_channels, kernel_size=1, bias=False)
        else:
            self.skip_fusions = nn.ModuleList([nn.Identity() for _ in skip_channels])
            self.latent_fusion = nn.Identity()

        self._init_multimodal_weights()
        logger.info(
            "Created multimodal tokenizer UNet: "
            f"modalities={self.modalities}, modal_channels={self.modal_channels}, fusion={self.fusion_type}"
        )

    def _init_multimodal_weights(self) -> None:
        from timm.layers.weight_init import lecun_normal_

        for m in self.skip_fusions.modules():
            if isinstance(m, nn.Conv2d):
                lecun_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if isinstance(self.latent_fusion, nn.Conv2d):
            lecun_normal_(self.latent_fusion.weight)
            if self.latent_fusion.bias is not None:
                nn.init.zeros_(self.latent_fusion.bias)

    def _prepare_modal_inputs(self, x: Tensor | dict[str, Tensor] | tuple[Tensor, ...] | list[Tensor]) -> list[Tensor]:
        if isinstance(x, Tensor):
            return split_modal_tensor(x, self.modal_channels)

        if isinstance(x, dict):
            if "img" in x and isinstance(x["img"], Tensor):
                return split_modal_tensor(x["img"], self.modal_channels)

            inputs: list[Tensor] = []
            for name in self.modalities:
                if name in x and isinstance(x[name], Tensor):
                    inputs.append(x[name])
                    continue
                prefixed_key = f"img_{name}"
                if prefixed_key in x and isinstance(x[prefixed_key], Tensor):
                    inputs.append(x[prefixed_key])
                    continue
                raise KeyError(f"Missing modality tensor for '{name}' in input dict keys={list(x.keys())}")
            return inputs

        if isinstance(x, (tuple, list)):
            inputs = list(x)
            if len(inputs) != len(self.modal_channels):
                raise ValueError(f"Input modalities mismatch: got {len(inputs)}, expected {len(self.modal_channels)}")
            return inputs

        raise TypeError(f"Unsupported input type: {type(x)}")

    def _fuse_skips(self, all_skips: list[list[Tensor]]) -> list[Tensor]:
        n_scales = len(all_skips[0])
        fused: list[Tensor] = []
        for s in range(n_scales):
            scale_feats = [modal_skips[s] for modal_skips in all_skips]
            if self.fusion_type == "mean":
                fused_feat = torch.stack(scale_feats, dim=0).mean(dim=0)
            else:
                fused_feat = self.skip_fusions[s](torch.cat(scale_feats, dim=1))
            fused.append(fused_feat)
        return fused

    def _fuse_latent(self, all_latents: list[Tensor | None]) -> Tensor | None:
        if all_latents[0] is None:
            return None
        if any(latent is None for latent in all_latents):
            raise ValueError("Latent fusion received mixed None/non-None values.")

        latent_list = [latent for latent in all_latents if latent is not None]
        if self.fusion_type == "mean":
            return torch.stack(latent_list, dim=0).mean(dim=0)
        return self.latent_fusion(torch.cat(latent_list, dim=1))

    def _align_modal_channels(self, modal_x: Tensor) -> Tensor:
        in_ch = int(modal_x.shape[1])
        tgt_ch = self.encoder_in_channels
        if in_ch == tgt_ch:
            return modal_x
        if in_ch > tgt_ch:
            raise ValueError(
                f"Modal channels {in_ch} exceed encoder in_channels {tgt_ch}. "
                "Current multimodal path only supports zero-padding alignment."
            )
        pad = modal_x.new_zeros((modal_x.shape[0], tgt_ch - in_ch, modal_x.shape[2], modal_x.shape[3]))
        return torch.cat([modal_x, pad], dim=1)

    def forward(self, x: Float[Tensor, "b c h w"] | dict[str, Tensor] | tuple[Tensor, ...] | list[Tensor]) -> Tensor:
        modal_inputs = self._prepare_modal_inputs(x)

        all_skips: list[list[Tensor]] = []
        all_latents: list[Tensor | None] = []
        for modal_x in modal_inputs:
            modal_x = self._align_modal_channels(modal_x)
            skips, latent = self.encoder(modal_x)
            all_skips.append(skips)
            all_latents.append(latent)

        fused_skips = self._fuse_skips(all_skips)
        fused_latent = self._fuse_latent(all_latents)
        return self.decoder(fused_skips, fused_latent)

    @classmethod
    @function_config_to_basic_types
    def create_model(cls, cfg=_create_default_cfg(), **overrides):
        if overrides is not None:
            cfg.merge_with(overrides)
        if cfg.tokenizer_pretrained_path is None:
            logger.warning("[MultimodalTokenizerHybridUNet]: No pretrained weights provided. Using random init.")
        return cls(cfg)
