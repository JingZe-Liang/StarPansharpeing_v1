from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import MethodType
from typing import cast

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .base import TeacherAdapter
from .utils import maybe_detach_feature_list, maybe_resize_img, select_rgb_channels


PE_INTERACTION_INDEXES = {
    "PE-Spatial-L14-448": [5, 11, 17, 23],
    "PE-Spatial-G14-448": [9, 21, 33, 48],
    "PE-Core-B16-224": [3, 5, 8, 11],
    "PE-Core-G14-448": [9, 21, 33, 48],
    "PE-Lang-L14-448": [5, 11, 17, 22],
    "PE-Lang-G14-448": [9, 21, 33, 46],
    "PE-Lang-L14-448-Tiling": [5, 11, 17, 22],
    "PE-Lang-G14-448-Tiling": [9, 21, 33, 46],
}


def _default_interaction_indexes(n_layers: int) -> list[int]:
    if n_layers <= 0:
        raise ValueError(f"{n_layers=} must be > 0")
    if n_layers == 1:
        return [0]
    ratios = (0.25, 0.5, 0.75, 1.0)
    idxs = [max(int(n_layers * ratio) - 1, 0) for ratio in ratios]
    idxs[-1] = n_layers - 1
    return idxs


def _normalize_layer_indexes(layer_indexes: list[int], *, n_layers: int) -> list[int]:
    if n_layers <= 0:
        raise ValueError(f"{n_layers=} must be > 0")
    if len(layer_indexes) == 0:
        return [n_layers - 1]

    normalized: list[int] = []
    for idx in layer_indexes:
        normalized_idx = idx
        if normalized_idx < 0:
            normalized_idx = (n_layers + normalized_idx) % n_layers
        normalized_idx = min(max(normalized_idx, 0), n_layers - 1)
        normalized.append(normalized_idx)

    deduped: list[int] = []
    seen: set[int] = set()
    for idx in normalized:
        if idx in seen:
            continue
        seen.add(idx)
        deduped.append(idx)
    return deduped


def _pe_interaction_indexes_for_model(model_name: str, *, n_layers: int) -> list[int]:
    idxs = PE_INTERACTION_INDEXES.get(model_name)
    if idxs is None:
        idxs = _default_interaction_indexes(n_layers)
    return _normalize_layer_indexes([int(x) for x in idxs], n_layers=n_layers)


def _pe_model_multi_features_patcher(
    self,
    x: torch.Tensor,
    norm: bool = False,
    layer_idx: int | list[int] = -1,
    strip_cls_token: bool = True,
) -> Tensor | list[Tensor]:
    batch, _, h, w = x.shape
    grid_h, grid_w = h // self.patch_size, w // self.patch_size
    is_multi_feats_out = isinstance(layer_idx, list)

    x = self.conv1(x)
    x = x.permute(0, 2, 3, 1).reshape(batch, -1, self.width)

    if self.use_cls_token:
        x = torch.cat(
            [self.class_embedding.view(1, 1, -1).expand(batch, -1, -1), x],
            dim=1,
        )

    if self.use_abs_posemb:
        x = x + self._sample_abs_posemb(grid_h, grid_w)

    if self.use_rope2d:
        self.rope.update_grid(x.device, grid_h, grid_w)

    x = self.ln_pre(x)

    backbone = self.transformer
    output_feats: list[Tensor] = []

    if isinstance(layer_idx, int):
        stop_idx = (backbone.layers + layer_idx) % backbone.layers
    else:
        stop_idx = max(layer_idx) if layer_idx else backbone.layers - 1

    attn_mask = None
    for i, block in enumerate(backbone.resblocks):
        if backbone.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint(block, x, None, None, attn_mask)
        else:
            x = block(x, attn_mask=attn_mask)

        if is_multi_feats_out:
            assert isinstance(layer_idx, list)
            if i in layer_idx:
                output_feats.append(x)

        if i == stop_idx:
            break

    if not is_multi_feats_out:
        return_feats: Tensor | list[Tensor] = x
    else:
        return_feats = output_feats

    if norm:
        if is_multi_feats_out:
            for i, feats in enumerate(return_feats):
                return_feats[i] = self.ln_post(feats)
        else:
            return_feats = self.ln_post(return_feats)

    if strip_cls_token and self.use_cls_token:
        if is_multi_feats_out:
            for i, feats in enumerate(return_feats):
                return_feats[i] = feats[:, 1:, :]
        else:
            return_feats = cast(Tensor, return_feats)[:, 1:, :]

    return return_feats


def load_perception_model(
    weight_path: str | Path,
    model_name: str = "PE-Core-L14-336",
    pretrained_on: str = "core",
    compile: bool = True,
):
    _ = pretrained_on
    import sys

    sys.path.insert(0, "src/stage1/perception_models")
    import core.vision_encoder.pe as pe  # ty: ignore[unresolved-import]

    if model_name is None:
        model_name = Path(weight_path).stem

    visual_available_cfgs = pe.VisionTransformer.available_configs()
    assert model_name in visual_available_cfgs, (
        f"Model {model_name} not available. Available models are {visual_available_cfgs}"
    )
    assert weight_path is not None, "weight path should not be None when loading the perception encoder model"

    model = pe.VisionTransformer.from_config(model_name, pretrained=True, checkpoint_path=str(weight_path))
    model.forward_features = MethodType(_pe_model_multi_features_patcher, model)

    if compile:
        model = torch.compile(model, mode="reduce-overhead")

    return model


class PETeacherAdapter(TeacherAdapter):
    def __init__(
        self,
        *,
        repa_encoder: nn.Module,
        repa_model_name: str,
        img_is_neg1_1: bool,
        rgb_channels: list[int] | str | None,
        img_resize: tuple[int, int] | str | None,
        pca_fn: Callable[..., Tensor] | None,
    ) -> None:
        super().__init__(repa_encoder, processor=None)
        self.repa_model_name = repa_model_name
        self.img_is_neg1_1 = img_is_neg1_1
        self.rgb_channels = rgb_channels
        self.img_resize = img_resize
        self.pca_fn = pca_fn
        self.mean = (0.5, 0.5, 0.5)
        self.std = (0.5, 0.5, 0.5)

    def _normalize(self, img: Tensor) -> Tensor:
        mean = img.new_tensor(self.mean).view(1, 3, 1, 1)
        std = img.new_tensor(self.std).view(1, 3, 1, 1)
        return (img - mean) / std

    def _unwrap_encoder(self) -> nn.Module:
        if isinstance(self.encoder, torch._dynamo.OptimizedModule):
            return self.encoder._orig_mod
        return self.encoder

    def forward_features(self, x: Tensor | dict, *, get_interm_feats: bool, detach: bool) -> list[Tensor]:
        if isinstance(x, dict):
            raise TypeError("PE adapter expects tensor inputs")

        img = x
        encoder = self._unwrap_encoder()
        n_layers = int(getattr(encoder, "layers"))
        if get_interm_feats:
            layers_to_take: int | list[int] = _pe_interaction_indexes_for_model(self.repa_model_name, n_layers=n_layers)
        else:
            layers_to_take = -1

        out_feats = self.encoder.forward_features(  # type: ignore[attr-defined]
            img,
            layer_idx=layers_to_take,
            strip_cls_token=True,
            norm=True,
        )

        patch_size = int(getattr(encoder, "patch_size", 16))
        h, w = img.shape[2] // patch_size, img.shape[3] // patch_size

        if isinstance(out_feats, (list, tuple)):
            feats = [rearrange(feat, "b (h w) c -> b c h w", h=h, w=w) for feat in out_feats]
        else:
            feats = [rearrange(out_feats, "b (h w) c -> b c h w", h=h, w=w)]

        return maybe_detach_feature_list(feats, detach)

    def encode(
        self,
        img: Tensor,
        *,
        get_interm_feats: bool,
        use_linstretch: bool,
        detach: bool,
        repa_fixed_bs: int | None,
    ) -> list[Tensor]:
        _ = repa_fixed_bs
        img = select_rgb_channels(
            img,
            rgb_channels=self.rgb_channels,
            use_linstretch=use_linstretch,
            pca_fn=self.pca_fn,
        )
        encoder = self._unwrap_encoder()
        image_size = int(getattr(encoder, "image_size", 224))
        patch_size = int(getattr(encoder, "patch_size", 16))
        img = maybe_resize_img(
            img,
            img_resize=self.img_resize,
            image_size=image_size,
            patch_size=patch_size,
        )
        if self.img_is_neg1_1:
            img = (img + 1) / 2
        img = self._normalize(img)
        return self.forward_features(img, get_interm_feats=get_interm_feats, detach=detach)
