from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal, cast

import timm
import torch
import torch.nn as nn
from accelerate.state import AcceleratorState
from torch import Tensor
from torch.distributed.tensor import DTensor, Shard
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from .base import TeacherAdapter
from .utils import ensure_feature_list, maybe_detach_feature_list, maybe_resize_img, normalize_img, select_rgb_channels


DINO_V3_INTERACTION_INDEXES = {
    "dinov3_vits16": [2, 5, 8, 11],
    "dinov3_vits16plus": [2, 5, 8, 11],
    "dinov3_vitb16": [2, 5, 8, 11],
    "dinov3_vitl16": [4, 11, 17, 23],
    "dinov3_vith16plus": [4, 16, 19, 31],
    "dinov3_vit7b16": [9, 19, 29, 39],
}


def load_repa_dino_v3_model(
    weight_path: str | Path | None = None,
    model_name: str | None = "dinov3_vitl16",
    pretrained_on: Literal["satellite", "web"] = "satellite",
    compile: bool = True,
) -> torch.nn.Module | torch._dynamo.OptimizedModule:
    repo_dir = Path(__file__).parents[6] / "dinov3"
    assert repo_dir.exists(), (
        f"DINOv3 repo directory {repo_dir} does not exist. Please git clone from https://github.com/facebookresearch/dinov3"
    )

    if model_name is None and weight_path is not None:
        stem = Path(weight_path).stem
        model_name = "_".join(stem.split("_", 2))
    elif weight_path is None and model_name is not None:
        model_type_dir = {
            "web": "web_image_pretrained_lvd",
            "satellite": "remote_sensing_image_pretrained_SAT_493M",
        }[pretrained_on]
        weight_dir = repo_dir / "weights" / model_type_dir
        paths = weight_dir.rglob("*.pth")
        for path in paths:
            search_name = model_name + "_pretrain"
            if search_name in path.stem:
                weight_path = str(path)
                break
        assert weight_path is not None, f"can not find weight {model_name=} at {weight_dir}"
    elif weight_path is None and model_name is None:
        raise ValueError("Either model_name or weight_path must be specified.")

    assert weight_path is not None, f"{weight_path=} does not exists"
    assert Path(weight_path).exists(), "Dino v3 model weight path does not exists"

    dino_model = torch.hub.load(repo_dir, model_name, source="local", weights=weight_path)
    dino_model = cast(nn.Module, dino_model)
    if compile:
        dino_model = torch.compile(dino_model, mode="reduce-overhead")
        dino_model = cast(torch._dynamo.OptimizedModule, dino_model)
    return dino_model


def load_repa_dino_v2_model(
    load_from: str = "torch",
    model_name: str = "dinov2_vitb14",
    weight_path: str | Path | None = None,
    compile: bool = True,
) -> torch.nn.Module:
    _ = weight_path
    if load_from == "timm":
        model = timm.create_model(
            model_name,
            pretrained=True,
            dynamic_img_size=True,
        )
    elif load_from == "torch":
        model = torch.hub.load("facebookresearch/dinov2", model_name)
        model = cast(nn.Module, model)
    else:
        raise ValueError(f"Unknown model loading source {load_from}, must be 'torch' or 'timm'.")
    if compile:
        model = cast(nn.Module, torch.compile(model))
    return model


class DinoTeacherAdapter(TeacherAdapter):
    def __init__(
        self,
        *,
        repa_encoder: nn.Module,
        dino_type: str,
        repa_model_name: str,
        c_dim_first: bool,
        img_is_neg1_1: bool,
        rgb_channels: list[int] | str | None,
        img_resize: tuple[int, int] | str | None,
        pca_fn: Callable[..., Tensor] | None,
        dino_pretrained_on: str,
    ) -> None:
        super().__init__(repa_encoder, processor=None)
        self.dino_type = dino_type
        self.repa_model_name = repa_model_name
        self.c_dim_first = c_dim_first
        self.img_is_neg1_1 = img_is_neg1_1
        self.rgb_channels = rgb_channels
        self.img_resize = img_resize
        self.pca_fn = pca_fn

        if dino_pretrained_on == "satellite" and repa_model_name.startswith("dinov3"):
            self.mean = (0.430, 0.411, 0.296)
            self.std = (0.213, 0.156, 0.143)
        else:
            self.mean = IMAGENET_DEFAULT_MEAN
            self.std = IMAGENET_DEFAULT_STD

    def _normalize(self, img: Tensor) -> Tensor:
        mean = img.new_tensor(self.mean).view(1, 3, 1, 1)
        std = img.new_tensor(self.std).view(1, 3, 1, 1)
        return (img - mean) / std

    def _unwrap_encoder(self) -> nn.Module:
        if isinstance(self.encoder, torch._dynamo.OptimizedModule):
            return self.encoder._orig_mod
        return self.encoder

    def _to_dtensor(self, img: Tensor) -> Tensor:
        to_dt = False
        try:
            to_dt = AcceleratorState().is_fsdp2
        except Exception:
            pass

        if to_dt:
            try:
                dm = self.encoder.patch_embed.proj.weight.device_mesh
                return DTensor.from_local(img, dm, placements=(Shard(0),))  # ty: ignore[invalid-argument-type]
            except Exception:
                return img
        return img

    def forward_features(self, x: Tensor | dict, *, get_interm_feats: bool, detach: bool) -> list[Tensor]:
        if isinstance(x, dict):
            raise TypeError("DINO adapter expects tensor inputs")

        img = self._to_dtensor(x)

        if self.dino_type == "torch":
            layers_to_take = DINO_V3_INTERACTION_INDEXES.get(self.repa_model_name, 1) if get_interm_feats else 1
            img_feats = self.encoder.get_intermediate_layers(  # type: ignore[attr-defined]
                img, n=layers_to_take, reshape=self.c_dim_first, norm=True
            )
            if len(img_feats) == 1:
                img_feats = img_feats[0]
        elif self.dino_type == "timm":
            b, _, h, w = img.shape
            assert h % 16 == 0 and w % 16 == 0, "image size must be divisible by 16"
            img_feats = self.encoder.forward_features(img)[:, 1:].reshape(b, h // 16, w // 16, -1).permute(0, 3, 1, 2)  # type: ignore[attr-defined]
        else:
            raise ValueError(f"Unknown model type: {self.dino_type}. Must be 'torch' or 'timm'.")

        if isinstance(img_feats, DTensor):
            img_feats = img_feats.full_tensor()
        elif isinstance(img_feats, (tuple, list)) and len(img_feats) > 0 and isinstance(img_feats[0], DTensor):
            img_feats = [img_feat.full_tensor() for img_feat in img_feats]

        feats = ensure_feature_list(img_feats)
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
        img = normalize_img(img, img_is_neg1_1=self.img_is_neg1_1, normalize_fn=self._normalize)

        bs = int(img.shape[0])
        if repa_fixed_bs is None or bs < repa_fixed_bs:
            return self.forward_features(img, get_interm_feats=get_interm_feats, detach=detach)

        feat_chunks: list[list[Tensor]] = []
        for start in range(0, bs, repa_fixed_bs):
            mb = img[start : start + repa_fixed_bs]
            feat_chunks.append(self.forward_features(mb, get_interm_feats=get_interm_feats, detach=detach))

        n_feats = len(feat_chunks[0])
        return [torch.cat([chunk[i] for chunk in feat_chunks], dim=0) for i in range(n_feats)]
