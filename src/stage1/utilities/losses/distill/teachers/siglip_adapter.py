from __future__ import annotations

from collections.abc import Callable, Mapping
import inspect
import math
from pathlib import Path
from types import MethodType
from typing import Any, cast

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig, SiglipProcessor
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling

from .base import TeacherAdapter
from .utils import ensure_feature_list, maybe_detach_feature_list, select_rgb_channels


SIGLIP2_INTERACTION_INDEXES = {
    "google/siglip2-base-patch16-512": [2, 5, 8, 11],
    "google/siglip2-large-patch16-512": [5, 11, 17, 23],
    "google/siglip2-base-patch16-naflex": [2, 5, 8, 11],
    "google/siglip2-large-patch16-naflex": [5, 11, 17, 23],
    "google/siglip2-so400m-patch14-224": [5, 11, 17, 23],
    "google/siglip2-so400m-patch16-naflex": [7, 16, 22, 26],
}


SIGLIP2_FEATURE_INDEX: list[int] | None = None


def _siglip_vit_encoder_forward_features_patcher(
    self,
    inputs_embeds,
    attention_mask: torch.Tensor | None = None,
    **kwargs,
):
    output_last_hs = kwargs.pop("output_hidden_states", False)
    intermidate_layer_indices = []
    features = []

    global SIGLIP2_FEATURE_INDEX
    if output_last_hs:
        intermidate_layer_indices = SIGLIP2_FEATURE_INDEX
        assert intermidate_layer_indices is not None, "Siglip2 feature index is not set"

    hidden_states = inputs_embeds
    for i, encoder_layer in enumerate(self.layers):
        hidden_states = encoder_layer(hidden_states, attention_mask, **kwargs)
        if output_last_hs and i in intermidate_layer_indices:
            features.append(hidden_states)
    assert len(features) == len(intermidate_layer_indices), (
        f"Extracted features do not match expected {len(intermidate_layer_indices)=} but got {len(features)=}"
    )
    return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=tuple(features))


def _siglip_vit_forward_features_patcher(self, pixel_values, interpolate_pos_encoding: bool | None = False, **kwargs):
    hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

    encoder_outputs: BaseModelOutput = self.encoder(
        inputs_embeds=hidden_states,
        **kwargs,
    )

    last_hidden_state = encoder_outputs.last_hidden_state
    last_hidden_state = self.post_layernorm(last_hidden_state)

    pooler_output = self.head(last_hidden_state) if self.use_head else None

    return BaseModelOutputWithPooling(
        hidden_states=encoder_outputs.hidden_states,
        last_hidden_state=last_hidden_state,
        pooler_output=pooler_output,
    )


def load_siglip2_model(
    name: str = "google/siglip2-so400m-patch16-naflex",
    use_bnb: bool = False,
    attn_implem: str = "sdpa",
    use_automodel: bool = True,
    cache_dir: str | Path | None = None,
    local_files_only: bool = True,
    *,
    local_file_path: str | None = "/Data2/ZihanCao/Checkpoints/Siglip2-so400m-patch16-naflex/checkpoint",
) -> tuple[nn.Module, SiglipProcessor]:
    if use_bnb:
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)
    else:
        bnb_config = None

    if not use_automodel:
        raise NotImplementedError("Directly load from Siglip2 class is not implemented yet")

    if cache_dir is None:
        cache_dir = Path.home() / ".cache/huggingface/hub"

    model = AutoModel.from_pretrained(
        name if local_file_path is None else local_file_path,
        quantization_config=bnb_config,
        device_map=None,
        cache_dir=cache_dir,
        attn_implementation=attn_implem,
        local_files_only=local_files_only,
    )
    model.text_model = None
    vision_model = model.vision_model
    processor = AutoProcessor.from_pretrained(
        name if local_file_path is None else local_file_path, cache_dir=cache_dir, local_files_only=local_files_only
    )

    global SIGLIP2_FEATURE_INDEX
    SIGLIP2_FEATURE_INDEX = SIGLIP2_INTERACTION_INDEXES[name]

    vision_model.encoder.forward = MethodType(_siglip_vit_encoder_forward_features_patcher, vision_model.encoder)
    if "naflex" not in name:
        vision_model.forward = MethodType(_siglip_vit_forward_features_patcher, vision_model)

    return vision_model, processor


def patch_siglip_processor(
    processor: SiglipProcessor,
    *,
    interp_pe: bool = False,
    size: dict | int | None = None,
    do_resize: bool | None = None,
    max_num_patches: int | None = None,
):
    if size is not None:
        if isinstance(size, int):
            size = {"height": size, "width": size}
        processor.image_processor.size = size
    if do_resize is not None:
        processor.image_processor.do_resize = do_resize

    processor.image_processor.do_rescale = False

    call_sig = inspect.signature(processor.__call__)
    allow_max_num_patches = "max_num_patches" in call_sig.parameters

    def _processor(*args, **kwargs):
        if max_num_patches is not None and allow_max_num_patches:
            kwargs["max_num_patches"] = max_num_patches

        kwargs.setdefault("return_tensors", "pt")
        inputs = processor(*args, **kwargs)
        inputs["attention_mask"] = inputs.pop("pixel_attention_mask", None)
        if interp_pe:
            inputs["interpolate_pos_encoding"] = True
        return cast(dict[str, Any], inputs)

    return _processor


class SiglipTeacherAdapter(TeacherAdapter):
    def __init__(
        self,
        *,
        repa_encoder: nn.Module,
        processor,
        repa_img_size: int,
        img_is_neg1_1: bool,
        rgb_channels: list[int] | str | None,
        pca_fn: Callable[..., Tensor] | None,
    ) -> None:
        max_n_patches = (repa_img_size // 16) ** 2
        patched_processor = patch_siglip_processor(
            processor,
            interp_pe=False,
            max_num_patches=max_n_patches,
        )
        super().__init__(repa_encoder, processor=patched_processor)
        self.img_is_neg1_1 = img_is_neg1_1
        self.rgb_channels = rgb_channels
        self.pca_fn = pca_fn

    def _select_hidden_layers(self, hidden_states: list[Tensor]) -> list[Tensor]:
        if SIGLIP2_FEATURE_INDEX is None:
            return hidden_states
        if len(hidden_states) == len(SIGLIP2_FEATURE_INDEX):
            return hidden_states

        n_layers = getattr(getattr(self.encoder, "config", None), "num_hidden_layers", None)
        offset = 0
        if isinstance(n_layers, int) and len(hidden_states) == n_layers + 1:
            offset = 1

        selected: list[Tensor] = []
        for idx in SIGLIP2_FEATURE_INDEX:
            mapped_idx = idx + offset
            if mapped_idx >= len(hidden_states):
                break
            selected.append(hidden_states[mapped_idx])
        if len(selected) == len(SIGLIP2_FEATURE_INDEX):
            return selected
        return hidden_states

    def forward_features(self, x: Tensor | dict, *, get_interm_feats: bool, detach: bool) -> list[Tensor]:
        if not isinstance(x, Mapping):
            raise TypeError("Siglip adapter expects mapping inputs from processor")

        inputs = dict(x)
        is_naflex = "spatial_shapes" in inputs
        shapes = inputs.get("spatial_shapes")
        dev = next(self.encoder.parameters()).device
        inputs = {k: v.to(dev) if hasattr(v, "to") else v for k, v in inputs.items()}
        if get_interm_feats:
            inputs["output_hidden_states"] = True
        model_out = self.encoder(**inputs)
        if get_interm_feats:
            hidden_states = model_out.hidden_states
            if hidden_states is None:
                raise RuntimeError("Siglip model returned no hidden_states while output_hidden_states=True")
            out_feats = self._select_hidden_layers(ensure_feature_list(hidden_states))
        else:
            out_feats = model_out.last_hidden_state

        feats = ensure_feature_list(out_feats)
        bs = int(inputs["pixel_values"].shape[0])

        def _to_2d(feat: Tensor) -> Tensor:
            if is_naflex:
                assert shapes is not None
                feat_valid = []
                for i in range(bs):
                    shape_i = shapes[i]
                    n_patches_i = shape_i.prod().item()
                    h_sample = feat[i, :n_patches_i].view(1, *shape_i, -1).permute(0, 3, 1, 2)
                    feat_valid.append(h_sample)
                return torch.cat(feat_valid, dim=0)

            if feat.ndim == 3:
                h = w = int(math.sqrt(feat.shape[1]))
                return feat.reshape(bs, h, w, feat.shape[-1]).permute(0, 3, 1, 2)
            return feat

        feats_2d = [_to_2d(feat) for feat in feats]
        return maybe_detach_feature_list(feats_2d, detach)

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
        if self.img_is_neg1_1:
            img = (img + 1) / 2
        img = img.clamp(0, 1)
        assert self.processor is not None
        inputs_any = self.processor(images=img)
        if not isinstance(inputs_any, Mapping):
            raise TypeError(f"Siglip processor output must be mapping, got {type(inputs_any)}")
        inputs = dict(inputs_any)
        return self.forward_features(inputs, get_interm_feats=get_interm_feats, detach=detach)
