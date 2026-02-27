import math
import sys
from functools import partial
from pathlib import Path
from types import MethodType
from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    Sequence,
    cast,
    no_type_check,
    overload,
)

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.state import AcceleratorState
from einops import rearrange
from jaxtyping import Float
from loguru import logger
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers.create_norm_act import get_norm_act_layer
from torch import Tensor, is_tensor
from torch.distributed.tensor import DTensor, Shard
from torch.utils.checkpoint import checkpoint
from torchvision.transforms import Normalize
from transformers import (
    AutoModel,
    AutoProcessor,
    BitsAndBytesConfig,
    Siglip2VisionModel,
    SiglipProcessor,
)
from transformers.feature_extraction_utils import BatchFeature
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.models.siglip2.modeling_siglip2 import Siglip2VisionTransformer

from ..distill.phi_s import MultiLayersPhiSDistillLoss
from ..distill.hadamard import get_hadamard_matrix
from ..distill.teachers import (
    build_teacher_adapter,
    ensure_feature_list,
    load_perception_model as _load_perception_model_impl,
    load_repa_dino_v2_model as _load_repa_dino_v2_model_impl,
    load_repa_dino_v3_model as _load_repa_dino_v3_model_impl,
    load_repa_encoder as _load_repa_encoder_impl,
    load_siglip2_model as _load_siglip2_model_impl,
    patch_siglip_processor as _siglip_processor_patcher_impl,
)

from src.stage1.utilities.losses.gan_loss.utils import get_rgb_channels_for_model
from src.stage1.utilities.losses.repa.feature_pca import (
    feature_pca_sk,
    feature_pca_torch,
)
from src.utilities.train_utils import time_recorder
from src.utilities.config_utils import function_config_to_basic_types


# Dino collections
DINOV3_TO_NUM_LAYERS = {
    "dinov3_vits16": 12,
    "dinov3_vits16plus": 12,
    "dinov3_vitb16": 12,
    "dinov3_vitl16": 24,
    "dinov3_vith16plus": 32,
    "dinov3_vit7b16": 40,
}
DINOv3_INTERACTION_INDEXES = {
    "dinov3_vits16": [2, 5, 8, 11],
    "dinov3_vits16plus": [2, 5, 8, 11],
    "dinov3_vitb16": [2, 5, 8, 11],
    "dinov3_vitl16": [4, 11, 17, 23],
    "dinov3_vith16plus": [4, 16, 19, 31],
    "dinov3_vit7b16": [9, 19, 29, 39],
}

# Siglip2 collections
# https://huggingface.co/collections/google/siglip2-67b5dcef38c175486e240107
SIGLIP2_INTERACTION_INDEXES = {
    "google/siglip2-base-patch16-512": [2, 5, 8, 11],
    "google/siglip2-large-patch16-512": [5, 11, 17, 23],
    "google/siglip2-base-patch16-naflex": [2, 5, 8, 11],
    "google/siglip2-large-patch16-naflex": [5, 11, 17, 23],
    "google/siglip2-so400m-patch14-224": [5, 11, 17, 23],  # patch=[14,16]; shape=[224,384,512]
    "google/siglip2-so400m-patch16-naflex": [7, 16, 22, 26],  # naflex backbone
}
SIGLIP2_FEATURE_INDEX: list[int] | None = None

# PE collections
PE_INTERACTION_INDEXES = {
    "PE-Spatial-L14-448": [5, 11, 17, 23],
    "PE-Spatial-G14-448": [9, 21, 33, 48],
    "PE-Core-B16-224": [3, 5, 8, 11],
    "PE-Core-G14-448": [9, 21, 33, 48],
    # PE lang checkpoints have fewer layers than their PE core counterparts.
    # These indices follow the same "4-tap" convention used elsewhere in this file,
    # but are clamped to the valid layer range at runtime.
    "PE-Lang-L14-448": [5, 11, 17, 22],
    "PE-Lang-G14-448": [9, 21, 33, 46],
    "PE-Lang-L14-448-Tiling": [5, 11, 17, 22],
    "PE-Lang-G14-448-Tiling": [9, 21, 33, 46],
}

# types
type InterpType = tuple[int, ...] | torch.Size


def _default_interaction_indexes(n_layers: int) -> list[int]:
    if n_layers <= 0:
        raise ValueError(f"{n_layers=} must be > 0")
    if n_layers == 1:
        return [0]
    ratios = (0.25, 0.5, 0.75, 1.0)
    idxs = [max(int(n_layers * r) - 1, 0) for r in ratios]
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

    # Keep order stable, but deduplicate.
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


def interpolate_features_2d_stacked(
    x: Tensor | list[Tensor], tgt_size: InterpType | list[InterpType]
) -> Tensor | list[Tensor]:
    assert is_tensor(x) or isinstance(x, list), (
        f"feature shape should be [layers, bs, c, h, w] or list of [bs, c, h, w] but got {type(x)}"
    )
    if is_tensor(x):
        assert x.ndim == 5

    n_ft = len(x)
    for i in range(n_ft):
        x[i] = interpolate_features_2d(
            x[i],
            tgt_size if not isinstance(tgt_size, list) else tgt_size[i],
        )

    return x


def interpolate_features_2d(x: Tensor, tgt_size: InterpType) -> Tensor:
    B, D, H, W = x.shape
    H_tgt, W_tgt = tgt_size

    if H == H_tgt and W == W_tgt:
        return x
    else:
        # use bicubic may cause low GPU utilitzaiton
        return F.interpolate(x, size=(H_tgt, W_tgt), mode="bilinear", align_corners=False)


def interpolate_features_1d(x, target_len: int):
    """Interpolate features to match target sequence length.
    Args:
        x: tensor of shape (B, T1, D)
        target_len: desired sequence length T2
    Returns:
        tensor of shape (B, T2, D)
    """
    B, T1, D = x.shape

    if T1 == target_len:
        return x
    else:
        H1 = W1 = int(math.sqrt(T1))
        H2 = W2 = int(math.sqrt(target_len))

        # Reshape to 2D spatial dimensions and move channels to second dimension
        x = x.reshape(B, H1, W1, D).permute(0, 3, 1, 2)

        # Interpolate
        x = F.interpolate(x, size=(H2, W2), mode="bilinear", align_corners=False)

        # Reshape back to sequence
        return x.permute(0, 2, 3, 1).reshape(B, target_len, D)


def mean_flat(x: Tensor):
    return x.mean(dim=list(range(1, len(x.shape))))


def rearrange_lst(x: list[Tensor] | Tensor, pattern: str, **pattern_kwargs: dict[str, int]):
    if isinstance(x, list):
        return [rearrange(f, pattern, **pattern_kwargs) for f in x]
    else:
        return rearrange(x, pattern, **pattern_kwargs)


def norm_feature(feat: Tensor | list[Tensor], dim: int = 1, norm_type: str = "l2"):
    def _norm_fn(x):
        if norm_type == "l2":
            return F.normalize(x, dim=dim)
        elif norm_type == "spatial":
            # from iREPA paper
            gamma = 0.6  # (0.6, 0.8) from its paper
            # shaped as [b, c, l]
            spatial_dim = -1 if dim in (1, -2) else 1
            x = x - gamma * x.mean(dim=spatial_dim, keepdim=True)
            x = x / (x.std(dim=spatial_dim, keepdim=True) + 1e-6)
            return x
        else:
            raise ValueError(f"Unknown {norm_type=}")

    if isinstance(feat, list):
        return [_norm_fn(f) for f in feat]
    else:
        return _norm_fn(feat)


def build_mlp(
    hidden_size: int,
    projector_dim: int,
    z_dim: int,
    is_1d=False,
    norm_type: str = "layernorm2d",
    act_type: str = "gelu",
):
    linear_cls = nn.Linear if is_1d else partial(nn.Conv2d, kernel_size=1)
    norm_act = get_norm_act_layer(norm_type, act_layer=act_type)
    assert norm_act is not None, "norm_act should not be None"
    mlp = nn.Sequential(
        norm_act(hidden_size),
        linear_cls(hidden_size, projector_dim),
        norm_act(projector_dim),
        linear_cls(projector_dim, z_dim),
    )
    return mlp


def next_divisble_of_y(x, y):
    return math.ceil(x / y) * y


def is_list_tuple(x):
    return isinstance(x, (list, tuple))


def repa_loss_(feature_teacher, feature_student, dim=1):
    assert feature_teacher.shape == feature_student.shape
    return -torch.sum(feature_teacher * feature_student, dim=dim).mean()


def am_radio_spatial_loss(feature_teacher, feature_student, l1_ratio=0.0, dim=1):
    # b, c, l
    assert feature_teacher.shape == feature_student.shape
    un_sim = 1 - torch.cosine_similarity(feature_teacher, feature_student, dim=dim)
    loss = un_sim.mean()

    if l1_ratio > 0:
        smooth_l1 = F.smooth_l1_loss(feature_teacher, feature_student)
        loss = loss + l1_ratio * smooth_l1

    return loss


def token_relation_loss(
    feature_teacher,
    feature_student,
    dim=-1,  # TODO: change args name to c_dim=-1
    norm=False,
    img_level=True,
    remove_negative=True,
    remove_only_teacher_neg=False,
    reduction: str = "mean",
):
    """
    Gram loss actually
    feature shape: [b * nf, l, c], dim=-1; [b * nf, c, l], dim=1,
        means where the channel dim at which dimension.
    """
    if dim in (1, -2):
        feature_teacher = feature_teacher.transpose(1, 2)  # bcl->blc
        feature_student = feature_student.transpose(1, 2)

    if not img_level:
        feature_teacher = feature_teacher.flatten(0, 1)  # (bs * l, c)
        feature_student = feature_student.flatten(0, 1)  # (bs * l, c)

    # Normalize features before computing Gram matrix (consistent with DINOv3 anchor gram loss)
    if norm:
        feature_teacher = F.normalize(feature_teacher, dim=-1)
        feature_student = F.normalize(feature_student, dim=-1)

    if img_level:
        relation_t = torch.einsum("bsc,blc->bsl", feature_teacher, feature_teacher)
        relation_s = torch.einsum("bsc,blc->bsl", feature_student, feature_student)
    else:
        relation_t = torch.einsum("Bc,Nc->BN", feature_teacher, feature_teacher)
        relation_s = torch.einsum("Bc,Nc->BN", feature_student, feature_student)

    if remove_negative:
        relation_t = relation_t.clamp(min=0.0)
        relation_s = relation_s.clamp(min=0.0)
    elif remove_only_teacher_neg:
        relation_t = relation_t.clamp(min=0.0)
        relation_s[(relation_s < 0) & (relation_t < 0)] = 0.0  # ? Check this, if is correct after do teacher clamp

    loss = F.mse_loss(relation_t, relation_s, reduction=reduction)
    return loss


def multiple_features_apply(
    function: Callable,
    feature_teacher,
    feature_student,
    disable=False,
    **kwargs,
):
    if disable:
        return function(feature_teacher, feature_student, **kwargs)

    assert len(feature_teacher) == len(feature_student), (
        "Teacher and student feature lists/Tensors should have the same length"
    )
    loss = 0.0
    for ft, fs in zip(feature_teacher, feature_student):
        loss += function(ft, fs, **kwargs)
    loss = loss / len(feature_teacher)
    return loss


def hier_distillation_loss(
    feature_teacher,
    feature_student,
    dim: int = -2,
    beta: float = 2.0,
    implem: Literal["new", "legacy"] = "new",
    loss_type: Literal["cosine", "gram"] = "cosine",
    layer_weight_type: Literal["softmax", "equal", "linear"] = "softmax",
):
    """Hierarchical distillation loss for multi-layer feature alignment.

    This function computes a weighted distillation loss across multiple layers,
    where higher layers get higher base weights and the weights are further
    adjusted based on the dissimilarity between teacher and student features.

    Parameters
    ----------
    feature_teacher : Tensor or list[Tensor]
        Teacher features with shape (layers, bs, c, l) or list of [bs, c, l]
    feature_student : Tensor or list[Tensor]
        Student features with same structure as teacher features
    dim : int, optional
        Dimension along which to compute cosine similarity. Default is -2 (channel dim).
    beta : float, optional
        Temperature parameter for weight adjustment. Higher values increase
        the influence of dissimilarity on weights. Default is 2.0.
    implem : str, optional
        Implementation method: "legacy" or "new". Default is "new".
    loss_type: str, optional
        Loss type: "cosine" or "gram". Default is "cosine".
    layer_weight_type: str, optional
        Layer weight type: "softmax", "equal", or "linear". Default is "softmax".

    Returns
    -------
    Tensor
        Computed hierarchical distillation loss (scalar).

    Note
    ----
    Assumes input features are in format [bs, c, l] where dim=-2 corresponds
    to the channel dimension for cosine similarity computation.
    """
    # Assertions
    is_lst = is_list_tuple(feature_teacher) and is_list_tuple(feature_student)
    is_stacked = is_tensor(feature_teacher) and is_tensor(feature_student)
    assert is_lst or is_stacked, "feature_teacher and feature_student should be lists or tensors"
    layers_n = len(feature_teacher)
    assert layers_n == len(feature_student), "Teacher and student should have the same number of layers"

    device = feature_teacher.device if is_tensor(feature_teacher) else feature_teacher[0].device
    weight_base = torch.arange(1, layers_n + 1, dtype=torch.float32, device=device) / layers_n

    # Compatibility: if the feature size are different across layers, use list form
    if is_stacked:
        feature_teacher = feature_teacher.unbind(dim=0)
        feature_student = feature_student.unbind(dim=0)

    ####### Cosine loss
    if loss_type == "cosine":
        cosine_sims = []
        for i in range(layers_n):
            ft, fs = feature_teacher[i], feature_student[i]
            if implem == "legacy":
                t_norm = F.normalize(ft, p=2, dim=dim)  # (bs, c, l)
                s_norm = F.normalize(fs, p=2, dim=dim)  # (bs, c, l)
                cosine_sim_map = (t_norm * s_norm).sum(dim=dim)  # (bs, l)
            else:
                cosine_sim_map = F.cosine_similarity(ft, fs, dim=dim)  # (bs, l)
            cosine_sims.append(cosine_sim_map)
        cosine_sims = torch.stack(cosine_sims, dim=0)
        unsim_loss: Tensor = 1.0 - cosine_sims.mean(dim=(1, 2))  # (layers,)
    elif loss_type == "gram":
        gram_losses = []
        for i in range(layers_n):
            ft, fs = feature_teacher[i], feature_student[i]
            gram_loss = token_relation_loss(ft, fs, dim=dim, norm=True, remove_negative=True)
            gram_losses.append(gram_loss)
        unsim_loss = torch.stack(gram_losses, dim=0)  # name hack for the code compatibility.
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    if layer_weight_type == "softmax":
        if implem == "legacy" and loss_type == "cosine":
            un_sim_exp = (beta * unsim_loss).exp()
            w_per_layer = weight_base * un_sim_exp
            w_per_layer = w_per_layer / (w_per_layer.sum() + 1e-6)
        else:
            w_per_layer = weight_base * torch.softmax(beta * unsim_loss, dim=0)
    elif layer_weight_type == "equal":
        w_per_layer = torch.ones_like(unsim_loss) / layers_n
    elif layer_weight_type == "linear":
        w_per_layer = weight_base / weight_base.sum()
    else:
        raise ValueError(f"Unknown layer weight type {layer_weight_type}")

    loss = (w_per_layer * unsim_loss).sum()

    return loss


# *==============================================================
# * Load DINOv2/v3 or PE models
# *==============================================================


def _pe_model_multi_features_patcher(
    self,
    x: torch.Tensor,
    norm: bool = False,
    layer_idx: int | list[int] = -1,
    strip_cls_token: bool = True,
) -> Tensor | list[Tensor]:
    """Forward method patcher for PE models"""
    batch, _, h, w = x.shape
    grid_h, grid_w = h // self.patch_size, w // self.patch_size
    _is_multi_feats_out = isinstance(layer_idx, list)

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

    ###### Forward transformer layers
    # x = self.transformer(x, layer_idx=layer_idx)
    backbone = self.transformer
    output_feats: list[Tensor] = []

    if isinstance(layer_idx, int):
        stop_idx = (backbone.layers + layer_idx) % backbone.layers
    else:
        # For list of layer indices, we need to go through all specified layers
        stop_idx = max(layer_idx) if layer_idx else backbone.layers - 1

    attn_mask = None
    for i, r in enumerate(backbone.resblocks):
        if backbone.grad_checkpointing and not torch.jit.is_scripting():
            # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
            # x = checkpoint(r, x, attn_mask)
            x = checkpoint(r, x, None, None, attn_mask)
        else:
            x = r(x, attn_mask=attn_mask)

        # Collect features at specified layers
        if _is_multi_feats_out:
            assert isinstance(layer_idx, list)
            if i in layer_idx:
                output_feats.append(x)

        if i == stop_idx:
            break

    if not _is_multi_feats_out:
        return_feats: Tensor | list[Tensor] = x
    else:
        return_feats = output_feats

    ######### Post ln norm
    if norm:
        if _is_multi_feats_out:
            for i, feats in enumerate(return_feats):
                return_feats[i] = self.ln_post(feats)
        else:
            return_feats = self.ln_post(return_feats)

    if strip_cls_token and self.use_cls_token:
        if _is_multi_feats_out:
            for i, feats in enumerate(return_feats):
                return_feats[i] = feats[:, 1:, :]
        else:
            return_feats_t = cast(Tensor, return_feats)
            return_feats = return_feats_t[:, 1:, :]

    return return_feats


def load_perception_model(
    weight_path: str | Path,
    model_name: str = "PE-Core-L14-336",
    pretrained_on: Literal["core", "lang", "spatial"] = "core",
    compile=True,
):
    """
    Load the Perception Encoder (PE) vision encoder (core / lang / spatial).

    Reference: https://github.com/facebookresearch/perception_models

    Supported `model_name` values (aligned with `VisionTransformer.available_configs()`):
    - PE core:
      - `PE-Core-G14-448`
      - `PE-Core-L14-336`
      - `PE-Core-B16-224`
      - `PE-Core-S16-384`
      - `PE-Core-T16-384`
    - PE lang:
      - `PE-Lang-G14-448`
      - `PE-Lang-L14-448`
      - `PE-Lang-G14-448-Tiling`
      - `PE-Lang-L14-448-Tiling`
    - PE spatial:
      - `PE-Spatial-G14-448`
      - `PE-Spatial-L14-448`
      - `PE-Spatial-B16-512`
      - `PE-Spatial-S16-512`
      - `PE-Spatial-T16-512`

    Notes:
    - `weight_path` must point to the corresponding checkpoint (e.g. a local `*.pt`).
    - This function patches `forward_features` to support multi-layer outputs via `layer_idx: list[int]`
      (useful for hierarchical distillation).
    """
    sys.path.insert(0, "src/stage1/perception_models")
    import core.vision_encoder.pe as pe  # ty: ignore[unresolved-import]

    if model_name is None:
        model_name = Path(weight_path).stem

    visual_available_cfgs = pe.VisionTransformer.available_configs()
    assert model_name in visual_available_cfgs, (
        f"Model {model_name} not available. Available models are {visual_available_cfgs}"
    )
    assert weight_path is not None, "weight path should not be None when loading the perception encoder model"

    # Load the model
    # cfg = pe.PE_VISION_CONFIG[model_name]
    model = pe.VisionTransformer.from_config(model_name, pretrained=True, checkpoint_path=str(weight_path))
    logger.info(f"[PE Model]: Load model {model_name} with model_name {model_name}")

    # Patch forward method to support multiple-layer feature output
    model.forward_features = MethodType(_pe_model_multi_features_patcher, model)

    if compile:
        model = torch.compile(model, mode="reduce-overhead")
        logger.info("[PE model]: Model compiled with reduce-overhead")

    return model


def load_repa_dino_v3_model(
    weight_path: str | Path | None = None,
    model_name: str | None = "dinov3_vitl16",
    pretrained_on: Literal["satellite", "web"] = "satellite",
    compile=True,
) -> torch.nn.Module | torch._dynamo.OptimizedModule:
    """
    import torch

    REPO_DIR = <PATH/TO/A/LOCAL/DIRECTORY/WHERE/THE/DINOV3/REPO/WAS/CLONED>

    # DINOv3 ViT models pretrained on web images
    dinov3_vits16 = torch.hub.load(REPO_DIR, 'dinov3_vits16', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
    dinov3_vits16plus = torch.hub.load(REPO_DIR, 'dinov3_vits16plus', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
    dinov3_vitb16 = torch.hub.load(REPO_DIR, 'dinov3_vitb16', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
    dinov3_vitl16 = torch.hub.load(REPO_DIR, 'dinov3_vitl16', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
    dinov3_vith16plus = torch.hub.load(REPO_DIR, 'dinov3_vith16plus', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
    dinov3_vit7b16 = torch.hub.load(REPO_DIR, 'dinov3_vit7b16', source='local', weights=<CHECKPOINT/URL/OR/PATH>)

    # DINOv3 ConvNeXt models pretrained on web images
    dinov3_convnext_tiny = torch.hub.load(REPO_DIR, 'dinov3_convnext_tiny', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
    dinov3_convnext_small = torch.hub.load(REPO_DIR, 'dinov3_convnext_small', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
    dinov3_convnext_base = torch.hub.load(REPO_DIR, 'dinov3_convnext_base', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
    dinov3_convnext_large = torch.hub.load(REPO_DIR, 'dinov3_convnext_large', source='local', weights=<CHECKPOINT/URL/OR/PATH>)

    # DINOv3 ViT models pretrained on satellite imagery
    dinov3_vitl16 = torch.hub.load(REPO_DIR, 'dinov3_vitl16', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
    dinov3_vit7b16 = torch.hub.load(REPO_DIR, 'dinov3_vit7b16', source='local', weights=<CHECKPOINT/URL/OR/PATH>)

    The pretrained weights are placed as follows:

    src/stage1/utilities/losses/dinov3/weights
    ├── remote_sensing_image_pretrained_SAT_493M
    │   ├── dinov3_vit7b16_pretrain_sat493m-a6675841.pth
    │   └── dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth
    └── web_image_pretrained_lvd
        ├── dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth
        ├── dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth
        ├── dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth
        ├── dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth
        ├── dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
        ├── dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth
        ├── dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
        ├── dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth
        ├── dinov3_vits16_pretrain_lvd1689m-08c60483.pth
        └── download_dinov3_weights.py
    """
    repo_dir = Path(__file__).parents[1] / "dinov3"
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
        # search the weight path
        paths = weight_dir.rglob("*.pth")
        for p in paths:
            # avoid 'dinov3_vits16' and 'dinov3_vits16plus'
            search_name = model_name + "_pretrain"
            if search_name in p.stem:
                weight_path = str(p)
                break
        assert weight_path is not None, f"can not find weight {model_name=} at {weight_dir}"
    elif weight_path is None and model_name is None:
        raise ValueError("Either model_name or weight_path must be specified.")

    assert weight_path is not None, f"{weight_path=} does not exists"
    logger.info(f"[Dino v3 in REPA]: use Dino v3 model: {model_name} loaded from {weight_path}.")
    assert Path(weight_path).exists(), "Dino v3 model weight path does not exists"
    sys.path.append(str(repo_dir))
    dino_model = torch.hub.load(repo_dir, model_name, source="local", weights=weight_path)
    dino_model = cast(nn.Module, dino_model)
    if compile:
        dino_model = torch.compile(dino_model, mode="reduce-overhead")
        dino_model = cast(torch._dynamo.OptimizedModule, dino_model)
        logger.info("[Dino v3 model]: compiled model done")
    return dino_model


def load_repa_dino_v2_model(
    load_from: str = "torch",
    model_name: str = "dinov2_vitb14",
    weight_path: str | Path | None = None,
    compile=True,
) -> torch.nn.Module:
    # This is Dino v2 models
    if load_from == "timm":
        model = timm.create_model(
            model_name,  # "hf-hub:timm/vit_large_patch14_dinov2.lvd142m",
            pretrained=True,
            dynamic_img_size=True,
        )
    elif load_from == "torch":  # vit base
        # TODO: add the local weight path
        model = torch.hub.load("facebookresearch/dinov2", model_name)
        model = cast(nn.Module, model)
    else:
        raise ValueError(f"Unknown model loading source {load_from}, must be 'torch' or 'timm'.")
    if compile:
        model = torch.compile(model)
        logger.info("[Dino v2 model]: compiled model done")
    return model  # type: ignore[return-value]


def _siglip_vit_encoder_forward_features_patcher(
    self,
    inputs_embeds,
    attention_mask: Optional[torch.Tensor] = None,
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


def _siglip_vit_forward_features_patcher(
    self, pixel_values, interpolate_pos_encoding: Optional[bool] = False, **kwargs
):
    hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

    encoder_outputs: BaseModelOutput = self.encoder(
        inputs_embeds=hidden_states,
        **kwargs,
    )

    last_hidden_state = encoder_outputs.last_hidden_state
    last_hidden_state = self.post_layernorm(last_hidden_state)

    pooler_output = self.head(last_hidden_state) if self.use_head else None

    return BaseModelOutputWithPooling(
        hidden_states=encoder_outputs.hidden_states,  # add hidden_states here
        last_hidden_state=last_hidden_state,
        pooler_output=pooler_output,
    )


def load_siglip2_model(
    name="google/siglip2-so400m-patch16-naflex",
    use_bnb=False,
    attn_implem="sdpa",  # 'sdpa' or 'flash_attention_2'
    use_automodel=True,
    cache_dir=None,
    local_files_only=True,
) -> tuple[Siglip2VisionTransformer, SiglipProcessor]:
    if use_bnb:
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)
    else:
        bnb_config = None

    if not use_automodel:
        # model, processor = None, None
        raise NotImplementedError("Directly load from Siglip2 class is not implemented yet")
    else:
        if cache_dir is None:
            # default cache dir
            cache_dir = Path.home() / ".cache/huggingface/hub"
        model = AutoModel.from_pretrained(
            name,
            quantization_config=bnb_config,
            device_map="auto",
            cache_dir=cache_dir,
            attn_implementation=attn_implem,
            local_files_only=local_files_only,
        )
        # remove the text model
        model.text_model = None
        vision_model = model.vision_model
        processor = AutoProcessor.from_pretrained(name, cache_dir=cache_dir, local_files_only=local_files_only)

    model = cast(Siglip2VisionModel, model)

    global SIGLIP2_FEATURE_INDEX
    SIGLIP2_FEATURE_INDEX = SIGLIP2_INTERACTION_INDEXES[name]
    logger.info(f"[Siglip2]: using feature index {SIGLIP2_FEATURE_INDEX=}")

    # Override the forward_features method
    vision_model.encoder.forward = MethodType(_siglip_vit_encoder_forward_features_patcher, vision_model.encoder)
    logger.info("[Siglip2]: override forward_features method for Siglipv2 ViT encoder")
    if "naflex" not in name:
        vision_model.forward = MethodType(_siglip_vit_forward_features_patcher, vision_model)
        logger.info("[Siglip2]: override forward method for Siglipv2 ViT")

    return vision_model, processor  # type: ignore[return-value]


def _siglip_processor_patcher(
    processor: SiglipProcessor,
    interp_pe=False,
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

    # the image should be [0, 1] and not to rescale
    processor.image_processor.do_rescale = False

    def _processor(*args, **kwargs):
        if max_num_patches is not None:
            kwargs["max_num_patches"] = max_num_patches

        kwargs.setdefault("return_tensors", "pt")
        inputs = processor(*args, **kwargs)
        inputs["attention_mask"] = inputs.pop("pixel_attention_mask", None)
        if interp_pe:
            inputs["interpolate_pos_encoding"] = True
        return inputs

    return _processor


def load_repa_encoder(
    repa_name: str = "dinov2",
    model_name: str = "dinov2_vitb14",
    weight_path: str | Path | None = None,
    *,
    load_from="torch",
    dino_v3_pretrained_on: Literal["satellite", "web"] = "satellite",
    compile=True,
):
    if repa_name == "dinov2":
        return load_repa_dino_v2_model(load_from, model_name, weight_path, compile)
    elif repa_name == "dinov3":
        return load_repa_dino_v3_model(
            weight_path,
            model_name,
            pretrained_on=dino_v3_pretrained_on,
            compile=compile,
        )
    elif repa_name == "pe":
        assert weight_path is not None, "weight_path should not be None when loading PE model"
        pe_model = load_perception_model(weight_path, model_name, compile=compile)
        return pe_model
    elif repa_name == "siglip2":
        # assert weight_path is not None, (
        #     "weight_path should not be None when loading Siglip2 model"
        # )
        model, processor = load_siglip2_model(model_name)
        return model, processor
    else:
        raise ValueError(f"Unknown DINO/PE version {repa_name}")


# Rebind legacy helper names to the shared teacher adapter implementations.
load_perception_model = _load_perception_model_impl
load_repa_dino_v3_model = _load_repa_dino_v3_model_impl
load_repa_dino_v2_model = _load_repa_dino_v2_model_impl
load_siglip2_model = _load_siglip2_model_impl
_siglip_processor_patcher = _siglip_processor_patcher_impl
load_repa_encoder = _load_repa_encoder_impl


# *==============================================================
# * Loss modules
# *==============================================================


class REPALoss(torch.nn.Module):
    @function_config_to_basic_types
    def __init__(
        self,
        repa_encoder: nn.Module | None = None,
        c_dim_first=True,  # teacher model output is [b, l, c] or [b, c, h, w]
        build_proj=False,
        img_is_neg1_1=True,
        rgb_channels: list[int] | Literal["random", "mean", "largest"] | str | None = None,
        img_resize: Literal["dino"] | tuple[int, int] | int | None = "dino",
        feature_normalize: str | None = "spatial",
        loss_type: str = "repa_original",
        get_hier_teacher_feature: bool = False,
        feature_resample_type: Literal["match_teacher", "match_student"] = "match_student",
        dtype: torch.dtype = torch.bfloat16,
        repa_fixed_bs: int | None = None,
        repa_img_size: int = 224,
        repa_model_type: str = "dinov3",  # dinov2, dinov3, pe, siglip2
        repa_model_name: str = "dinov3_vitl16",  # dino series (v2, v3), pe series, siglip2 series
        repa_model_load_path: str | None = None,
        # DINO model specifics
        dino_load_type: str = "torch",  # [torch, timm]
        dino_version: int = 3,
        dino_pretrained_on: str | Literal["satellite", "web"] | None = "satellite",
    ):
        super().__init__()
        self.rgb_channels = rgb_channels
        self.feature_normalize = feature_normalize
        self.loss_type = loss_type
        self.feature_resample_type = feature_resample_type
        self.get_hier_teacher_feature = get_hier_teacher_feature

        # Assertions
        assert loss_type in (
            "repa_original",
            "am_radio",
            "token_relation",
            "hier_distillation",
            "hier_distillation@gram",
            "hier_distillation@cosine",
        )
        assert repa_model_type in ("dinov2", "dinov3", "pe", "siglip2")
        assert dino_load_type in ("torch", "timm")
        assert dino_pretrained_on in ("satellite", "web")

        if isinstance(img_resize, int):
            img_resize = (img_resize, img_resize)
        if isinstance(img_resize, (tuple, list)):
            assert img_resize[0] % 16 == 0 and img_resize[1] % 16 == 0, f"{img_resize=} must be divisible by patch size"
        self.img_resize = img_resize

        self.repa_fixed_bs = repa_fixed_bs
        if self.rgb_channels is not None:
            if isinstance(self.rgb_channels, str):
                assert self.rgb_channels in (
                    "random",
                    "mean",
                    "largest",
                ), "rgb_channels must be randomly selected"
            elif isinstance(self.rgb_channels, Sequence):
                assert len(self.rgb_channels) == 3, "rgb_channels must be 3 channels"
            else:
                raise TypeError(f"rgb_channels must be list or tuple or str, but got {type(rgb_channels)}")

        # encoder
        self.dino_type = dino_load_type
        self.repa_model_name = repa_model_name
        self.repa_model_type = repa_model_type
        if dino_version == 3:
            self.dino_type = "torch"
        self.teacher_adapter = build_teacher_adapter(
            repa_model_type=repa_model_type,
            repa_model_name=repa_model_name,
            repa_model_load_path=repa_model_load_path,
            repa_encoder=repa_encoder,
            dino_load_type=self.dino_type,
            dino_pretrained_on=cast(Literal["satellite", "web"], dino_pretrained_on),
            c_dim_first=c_dim_first,
            img_is_neg1_1=img_is_neg1_1,
            rgb_channels=cast(list[int] | str | None, rgb_channels),
            img_resize=cast(tuple[int, int] | str | None, img_resize),
            repa_img_size=repa_img_size,
            dtype=dtype,
            pca_fn=feature_pca_torch if rgb_channels == "pca" else None,
        )
        self.repa_encoder = self.teacher_adapter.encoder
        if getattr(self.teacher_adapter, "processor", None) is not None:
            self.processor = self.teacher_adapter.processor

        logger.info("[REPA Loss]: load pretrained dino visual foundation model done")
        self.c_dim_first = c_dim_first

        # mlp projection to aligh the dim
        self.build_proj = build_proj
        if build_proj:
            encoder = self._get_encoder()
            self.projector = build_mlp(
                encoder.embed_dim,
                2048,
                encoder.embed_dim,
                is_1d=not c_dim_first,
            )
            logger.warning(
                "build repa loss in loss class, remember to optimize the projector, "
                "or match to repa dim in model forward",
                "warning",
            )
        else:
            self.projector = nn.Identity()

        self.img_is_neg1_1 = img_is_neg1_1

    def _get_encoder(self):
        """Unwrap compiled model to access original attributes."""
        if isinstance(self.repa_encoder, torch._dynamo.OptimizedModule):
            return self.repa_encoder._orig_mod
        return self.repa_encoder

    def __repr__(self):
        return (
            f"\nREPA Loss: {self.repa_encoder.__class__.__name__}\n"
            + f"    img_is_neg1_1: {self.img_is_neg1_1}\n"
            + f"    c_dim_first: {self.c_dim_first}\n"
            + f"    img_resize: {self.img_resize}\n"
            + f"    rgb_channels: {self.rgb_channels}\n"
            + f"    feature_normalize: {self.feature_normalize}\n"
            + f"    loss_type: {self.loss_type}\n"
            + f"    dino_type: {self.dino_type}\n"
            + f"    feature_resample_type: {self.feature_resample_type}\n"
            + f"    build_proj: {self.build_proj}"
        )

    def _forward_features(self, x: BatchFeature | Tensor, get_interm_feats=False, detach=True, **kwargs):
        _ = kwargs
        x_in = cast(Tensor | dict[str, Tensor], x)
        feats = self.teacher_adapter.forward_features(x_in, get_interm_feats=get_interm_feats, detach=detach)
        if get_interm_feats:
            return feats
        return feats[0]

    @time_recorder.record("repa_encode_img")
    @torch.no_grad()
    def _encode_img(
        self,
        img: Float[Tensor, "bs c h w"],
        get_interm_feats=False,
        use_linstretch=True,
        detach=True,
    ):
        assert img.ndim == 4, f"img must be 4d tensor but got {img.ndim}d"
        feats = self.teacher_adapter.encode(
            img,
            get_interm_feats=get_interm_feats,
            use_linstretch=use_linstretch,
            detach=detach,
            repa_fixed_bs=self.repa_fixed_bs,
        )
        if get_interm_feats:
            return feats
        return feats[0]

    @overload
    def _interp_teacher_or_student_features(
        self, teacher_feat: Tensor, student_feat: Tensor
    ) -> tuple[Tensor, Tensor]: ...

    @overload
    def _interp_teacher_or_student_features(
        self, teacher_feat: list[Tensor], student_feat: list[Tensor]
    ) -> tuple[list[Tensor], list[Tensor]]: ...

    def _interp_teacher_or_student_features(
        self, teacher_feat: Tensor | list[Tensor], student_feat: Tensor | list[Tensor]
    ):
        # Feature list or stacked feature
        is_t_stacked = is_tensor(teacher_feat) and teacher_feat.ndim == 5
        is_s_stacked = is_tensor(student_feat) and student_feat.ndim == 5
        is_s_feat_lst = is_list_tuple(student_feat) or is_s_stacked
        is_t_feat_lst = is_list_tuple(teacher_feat) or is_t_stacked
        is_feat_lst = is_t_feat_lst and is_s_feat_lst

        if is_feat_lst:
            assert len(teacher_feat) == len(student_feat), (
                f"len(teacher_feat) != len(student_feat), {len(teacher_feat)=}, {len(student_feat)=}"
            )
            assert (is_t_stacked and is_s_stacked) or (is_list_tuple(teacher_feat) and is_list_tuple(student_feat)), (
                "all list features or all stacked features"
            )
        is_feat_stacked = is_s_stacked and is_t_stacked

        # Match teacher features to student features
        if self.feature_resample_type == "match_student":

            def _interp_feat_match_stu(t_feat, s_feat):
                st_ndim = s_feat.ndim
                assert st_ndim in (3, 4), "student feature must be 1d or 2d feature"
                func_mapping_ = {3: interpolate_features_1d, 4: interpolate_features_2d}
                tgt_size = s_feat.shape[-2:] if st_ndim == 4 else s_feat.shape[-2]
                t_feat = func_mapping_[st_ndim](t_feat, tgt_size)
                return t_feat

            if is_feat_lst:
                teacher_feat = [
                    _interp_feat_match_stu(t_feat, s_feat) for t_feat, s_feat in zip(teacher_feat, student_feat)
                ]
                if is_feat_stacked:
                    teacher_feat = torch.stack(teacher_feat)
            else:
                teacher_feat = _interp_feat_match_stu(teacher_feat, student_feat)

        else:  # student feature matched to teacher feature

            def _interp_feat_match_tch(t_feat, s_feat):
                # tgt size is the teacher feature size
                if self.c_dim_first:
                    _tgt_sz = t_feat.shape[-2:]
                    interpolate_fn = interpolate_features_2d
                else:
                    _tgt_sz = t_feat.shape[-2]
                    interpolate_fn = interpolate_features_1d
                s_feat = interpolate_fn(s_feat, _tgt_sz)
                return s_feat

            if is_feat_lst:
                student_feat = [
                    _interp_feat_match_tch(t_feat, s_feat) for t_feat, s_feat in zip(teacher_feat, student_feat)
                ]
                if is_feat_stacked:
                    student_feat = torch.stack(student_feat)
            else:
                student_feat = _interp_feat_match_tch(teacher_feat, student_feat)

        # one layer feature case: [bs, c, h, w] or [bs, l, c]
        # multiple layer feature case: [n_layers, bs, c, h, w] or [n_layers, bs, l, c] or list of which
        return teacher_feat, student_feat

    @time_recorder.record("call_repa_loss")
    def repa_loss(self, teacher_feat: Tensor | list[Tensor], student_feat: Tensor | list[Tensor]):
        # [bs, l, c] or [bs, c, h, w]
        # 1. interpolate
        teacher_feat = ensure_feature_list(teacher_feat)
        student_feat = ensure_feature_list(student_feat)
        teacher_feat, student_feat = self._interp_teacher_or_student_features(  # type: ignore
            teacher_feat, student_feat
        )

        # 2. dino feature -> Featup -> model feature
        # TODO: use featup to upsample the dino feature into (high-resolution) model (or says)
        # tokenizer feature
        """
        some featup upsampling methods
        see: FeatUp, LoftUp, AnyUp

        self.featup: nn.Module
        teacher_feat = self.featup(teacher_feat)
        """

        # flatten the model feature
        # FIXME: no used, removed this
        student_feat = self.projector(student_feat)

        # 3. repa loss
        # stack all n_feature and bs dim together
        if self.c_dim_first:
            rarg_pattn = "... c h w -> ... c (h w)"
        else:
            rarg_pattn = "... h w c -> ... c (h w)"
        teacher_feat = rearrange_lst(teacher_feat, rarg_pattn)
        student_feat = rearrange_lst(student_feat, rarg_pattn)

        # 3.1 Whether to keep the feature magnitude (norm or not)
        dim = 1
        if self.feature_normalize is not None:
            # if hier_distillation, normalization in the function, see 'hier_distillation_loss'
            teacher_feat = norm_feature(teacher_feat, dim=dim, norm_type=self.feature_normalize)
            student_feat = norm_feature(student_feat, dim=dim, norm_type=self.feature_normalize)

        # Shape assertion
        _assertion_msg = lambda tf, sf: (
            f"img_feat and model_feat must have the same shape to compute the repa loss, "
            f"but got {tuple(tf.shape)} and {tuple(sf.shape)}"
        )
        if isinstance(teacher_feat, Tensor) and isinstance(student_feat, Tensor):
            assert teacher_feat.shape == student_feat.shape, _assertion_msg(teacher_feat, student_feat)
        else:  # both are list
            for t_feat, s_feat in zip(teacher_feat, student_feat):
                assert t_feat.shape == s_feat.shape, _assertion_msg(t_feat, s_feat)

        # 3.2 loss
        if self.loss_type.startswith("hier_distillation"):
            args = self.loss_type.split("@")  # hier_distillation@cosine/gram
            hd_loss_type: Literal["cosine", "gram"] = "cosine"
            if len(args) > 1:
                if args[1] not in ("cosine", "gram"):
                    raise ValueError(f"Unknown hier_distillation loss type: {args[1]!r}")
                hd_loss_type = cast(Literal["cosine", "gram"], args[1])
            # hier_distillation_loss needs all list features for weighted computation
            repa_loss = hier_distillation_loss(teacher_feat, student_feat, dim=dim, loss_type=hd_loss_type)
        else:
            # Other loss functions use multiple_features_apply
            loss_functions = {
                "repa_original": repa_loss_,
                "am_radio": am_radio_spatial_loss,
                "token_relation": token_relation_loss,
            }

            if self.loss_type not in loss_functions:
                raise ValueError(f"Unknown repa loss type {self.loss_type}")

            loss_fn = loss_functions[self.loss_type]

            # Use multiple_features_apply with disable=False for list inputs
            # For single tensor inputs, disable multiple_features_apply and call directly
            repa_loss = multiple_features_apply(
                loss_fn,
                teacher_feat,
                student_feat,
                disable=not is_list_tuple(teacher_feat),
                dim=dim,
            )

        return repa_loss

    @time_recorder.record("repa_forward")
    def forward(self, img: Tensor, student_feature: Tensor | list[Tensor]):
        teacher_feat = self._encode_img(img, get_interm_feats=self.get_hier_teacher_feature)
        teacher_feat = teacher_feat.detach() if is_tensor(teacher_feat) else [f.detach() for f in teacher_feat]
        if self.get_hier_teacher_feature and len(teacher_feat) > len(student_feature):
            assert is_list_tuple(teacher_feat), "teacher_feat must be a list when get_hier_teacher_feature is True"
            assert is_list_tuple(student_feature), "student_feature must be a list"
            teacher_feat = teacher_feat[-len(student_feature) :]  # last n teacher features
        repa_loss = self.repa_loss(teacher_feat, student_feature)
        return repa_loss

    @no_type_check
    def named_parameters(self, *args, **kwargs):
        if self.build_proj:
            return self.projector.named_parameters(*args, **kwargs)
        else:
            return []

    @no_type_check
    def parameters(self, *args, **kwargs):
        if self.build_proj:
            return self.projector.parameters(*args, **kwargs)
        else:
            return []

    def _to_dtensor(self, img: Tensor):
        to_dt = False
        try:
            to_dt = AcceleratorState().is_fsdp2
        except:
            pass

        if to_dt:
            dm = self.repa_encoder.patch_embed.proj.weight.device_mesh  # type: ignore
            img = DTensor.from_local(img, dm, placements=(Shard(0),))  # ty: ignore[invalid-argument-type]

        return img


class PhiSMultipleTeacherDistillLoss(nn.Module):
    def __init__(
        self,
        *,
        teacher_configs: dict[str, dict[str, Any]],
        teacher_dims: dict[str, list[int]],
        teacher_weights: dict[str, float] | None = None,
        layer_weights: dict[str, list[float]] | None = None,
        c_dim_first: bool = True,
        img_is_neg1_1: bool = True,
        rgb_channels: list[int] | Literal["random", "mean", "largest"] | str | None = None,
        img_resize: Literal["dino"] | tuple[int, int] | int | None = "dino",
        feature_resample_type: Literal["match_teacher", "match_student"] = "match_student",
        dtype: torch.dtype = torch.bfloat16,
        repa_fixed_bs: int | None = None,
        repa_img_size: int = 224,
        dino_load_type: Literal["torch", "timm"] = "torch",
        dino_pretrained_on: Literal["satellite", "web"] = "satellite",
        phi_loss_type: Literal["mse", "smoothl1"] = "mse",
        phi_eps: float = 1e-6,
        hadamard_allow_approx: bool = True,
    ) -> None:
        super().__init__()
        if len(teacher_configs) == 0:
            raise ValueError("teacher_configs must not be empty")
        if len(teacher_dims) == 0:
            raise ValueError("teacher_dims must not be empty")

        if isinstance(img_resize, int):
            img_resize = (img_resize, img_resize)

        self.c_dim_first = c_dim_first
        self.feature_resample_type = feature_resample_type
        self.repa_fixed_bs = repa_fixed_bs
        self.teacher_dims = {k: list(v) for k, v in teacher_dims.items()}

        hadamard_fn = lambda n, device=None, dtype=None: get_hadamard_matrix(  # noqa: E731
            n,
            allow_approx=hadamard_allow_approx,
        ).to(device=device, dtype=dtype)
        self.phi_loss = MultiLayersPhiSDistillLoss(
            teacher_dims=self.teacher_dims,
            hadamard_fn=hadamard_fn,
            loss_type=phi_loss_type,
            eps=phi_eps,
            teacher_weights=teacher_weights,
            layer_weights=layer_weights,
        )

        adapters: dict[str, Any] = {}
        encoder_modules: dict[str, nn.Module] = {}
        for teacher_name, cfg in teacher_configs.items():
            cfg_local = dict(cfg)
            repa_model_type = str(cfg_local.pop("repa_model_type", "dinov3"))
            repa_model_name = str(cfg_local.pop("repa_model_name", "dinov3_vitl16"))
            repa_model_load_path = cfg_local.pop("repa_model_load_path", None)
            repa_encoder = cfg_local.pop("repa_encoder", None)
            teacher_dino_load_type = str(cfg_local.pop("dino_load_type", dino_load_type))
            teacher_dino_pretrained_on = str(cfg_local.pop("dino_pretrained_on", dino_pretrained_on))
            teacher_rgb_channels = cfg_local.pop("rgb_channels", rgb_channels)
            teacher_img_resize = cfg_local.pop("img_resize", img_resize)
            teacher_img_is_neg1_1 = bool(cfg_local.pop("img_is_neg1_1", img_is_neg1_1))
            teacher_repa_img_size = int(cfg_local.pop("repa_img_size", repa_img_size))
            teacher_dtype = cfg_local.pop("dtype", dtype)

            adapter = build_teacher_adapter(
                repa_model_type=repa_model_type,
                repa_model_name=repa_model_name,
                repa_model_load_path=repa_model_load_path,
                repa_encoder=repa_encoder,
                dino_load_type=teacher_dino_load_type,
                dino_pretrained_on=cast(Literal["satellite", "web"], teacher_dino_pretrained_on),
                c_dim_first=c_dim_first,
                img_is_neg1_1=teacher_img_is_neg1_1,
                rgb_channels=cast(list[int] | str | None, teacher_rgb_channels),
                img_resize=cast(tuple[int, int] | str | None, teacher_img_resize),
                repa_img_size=teacher_repa_img_size,
                dtype=cast(torch.dtype, teacher_dtype),
                pca_fn=feature_pca_torch if teacher_rgb_channels == "pca" else None,
            )
            adapters[teacher_name] = adapter
            encoder_modules[teacher_name] = adapter.encoder

        if set(adapters.keys()) != set(self.teacher_dims.keys()):
            raise ValueError(
                f"teacher_configs keys {sorted(adapters.keys())} must match teacher_dims keys {sorted(self.teacher_dims.keys())}"
            )

        self.teacher_adapters = adapters
        self.teacher_encoders = nn.ModuleDict(encoder_modules)
        self.teacher_processors: dict[str, Any] = {
            key: adapter.processor for key, adapter in self.teacher_adapters.items() if adapter.processor is not None
        }

    def _interp_teacher_or_student_features(
        self,
        teacher_feat: list[Tensor],
        student_feat: list[Tensor],
    ) -> tuple[list[Tensor], list[Tensor]]:
        if len(teacher_feat) != len(student_feat):
            if len(teacher_feat) > len(student_feat):
                teacher_feat = teacher_feat[-len(student_feat) :]
            else:
                raise ValueError(
                    f"Teacher/student layer count mismatch: teacher={len(teacher_feat)}, student={len(student_feat)}"
                )

        if self.feature_resample_type == "match_student":

            def _interp_feat_match_stu(t_feat: Tensor, s_feat: Tensor) -> Tensor:
                st_ndim = s_feat.ndim
                if st_ndim == 3:
                    return interpolate_features_1d(t_feat, s_feat.shape[-2])
                if st_ndim == 4:
                    return interpolate_features_2d(t_feat, s_feat.shape[-2:])
                raise ValueError(f"student feature must be 1d or 2d, got ndim={st_ndim}")

            teacher_feat = [
                _interp_feat_match_stu(t_feat, s_feat) for t_feat, s_feat in zip(teacher_feat, student_feat)
            ]
        else:

            def _interp_feat_match_tch(t_feat: Tensor, s_feat: Tensor) -> Tensor:
                if t_feat.ndim == 3:
                    return interpolate_features_1d(s_feat, t_feat.shape[-2])
                if t_feat.ndim == 4:
                    return interpolate_features_2d(s_feat, t_feat.shape[-2:])
                raise ValueError(f"teacher feature must be 1d or 2d, got ndim={t_feat.ndim}")

            student_feat = [
                _interp_feat_match_tch(t_feat, s_feat) for t_feat, s_feat in zip(teacher_feat, student_feat)
            ]

        return teacher_feat, student_feat

    def _encode_teachers(
        self,
        img: Tensor,
        *,
        detach: bool,
    ) -> dict[str, list[Tensor]]:
        teacher_feats: dict[str, list[Tensor]] = {}
        for teacher_name, adapter in self.teacher_adapters.items():
            teacher_feats[teacher_name] = adapter.encode(
                img,
                get_interm_feats=True,
                use_linstretch=True,
                detach=detach,
                repa_fixed_bs=self.repa_fixed_bs,
            )
        return teacher_feats

    def _prepare_student_feats(
        self,
        student_feature: dict[str, Tensor | list[Tensor]],
    ) -> dict[str, list[Tensor]]:
        return {teacher_name: ensure_feature_list(feats) for teacher_name, feats in student_feature.items()}

    def _align_teacher_student(
        self,
        teacher_feats: dict[str, list[Tensor]],
        student_feats: dict[str, list[Tensor]],
    ) -> tuple[dict[str, list[Tensor]], dict[str, list[Tensor]]]:
        teacher_keys = set(teacher_feats.keys())
        student_keys = set(student_feats.keys())
        if teacher_keys != student_keys:
            missing_in_student = sorted(teacher_keys - student_keys)
            missing_in_teacher = sorted(student_keys - teacher_keys)
            raise KeyError(
                f"Teacher/student key mismatch: missing_in_student={missing_in_student}, missing_in_teacher={missing_in_teacher}"
            )

        aligned_teacher: dict[str, list[Tensor]] = {}
        aligned_student: dict[str, list[Tensor]] = {}
        for teacher_name in sorted(teacher_feats.keys()):
            t_feats = teacher_feats[teacher_name]
            s_feats = student_feats[teacher_name]
            t_feats, s_feats = self._interp_teacher_or_student_features(t_feats, s_feats)
            aligned_teacher[teacher_name] = t_feats
            aligned_student[teacher_name] = s_feats
        return aligned_teacher, aligned_student

    @torch.no_grad()
    def update_phi_stats_from_image(self, img: Tensor) -> None:
        teacher_feats = self._encode_teachers(img, detach=True)
        self.phi_loss.update_phi_stats(teacher_feats)

    @torch.no_grad()
    def reset_phi_stats(self) -> None:
        self.phi_loss.reset_phi_stats()

    @torch.no_grad()
    def finalize_phi_from_stats(self, *, distributed: bool = True) -> None:
        self.phi_loss.finalize_phi_from_stats(distributed=distributed)

    @torch.no_grad()
    def load_phi_from_cache(self, cache_path: str | Path, *, broadcast: bool = True) -> None:
        self.phi_loss.load_phi_from_cache(cache_path, broadcast=broadcast)

    @torch.no_grad()
    def save_phi_to_cache(self, cache_path: str | Path) -> None:
        self.phi_loss.save_phi_to_cache(cache_path)

    def forward(self, img: Tensor, student_feature: dict[str, Tensor | list[Tensor]]) -> Tensor:
        teacher_feats = self._encode_teachers(img, detach=True)
        student_feats = self._prepare_student_feats(student_feature)
        teacher_feats, student_feats = self._align_teacher_student(teacher_feats, student_feats)
        return self.phi_loss(student_feats, teacher_feats)


class VFLoss(REPALoss):
    def __init__(
        self,
        repa_encoder=None,
        distmat_margin: float = 0.25,
        cos_margin: float = 0.5,
        distmat_weight: float = 1.0,
        cos_weight: float = 1.0,
        c_dim_first=False,
        build_proj=False,
        img_is_neg1_1=True,
        feature_normalize: bool = False,
        loss_type: str = "vf_original",
        rgb_channels: list | Literal["random"] | str | None = None,
        img_resize: Literal["dino"] | tuple | None = "dino",
        dino_fixed_bs: int | None = None,
        dino_img_size: int = 224,
        dtype: torch.dtype = torch.bfloat16,
        vf_weight: float = 1.0,
        feature_resample_type: Literal["match_teacher", "match_student"] = "match_student",
    ):
        """Visual foundation model loss for tokenizer feature alignment.

        Args:
            distmat_margin (float, optional): Margin for distance matrix loss. Defaults to 0.25.
            cos_margin (float, optional): Margin for cosine similarity loss. Defaults to 0.5.
            distmat_weight (float, optional): Weight for distance matrix loss. Defaults to 1.0.
            cos_weight (float, optional): Weight for cosine similarity loss. Defaults to 1.0.
            c_dim_first (bool, optional): Whether features are 2D (True) or 1D (False). Defaults to False.
            build_proj (bool, optional): Whether to build projection in this loss class (not recommended). Defaults to False.
            img_is_neg1_1 (bool, optional): Whether image values range from (-1, 1). Defaults to True.
            rgb_channels (list | Literal["random"] | str | None, optional): RGB channel indices or "random"/"mean" selection from hyperspectral images. Defaults to None.
            img_resize (Literal["dino"] | tuple | None, optional): Input image resize strategy. Defaults to "dino".
            dino_fixed_bs (int | None, optional): Batch size for DINO forward pass. Defaults to None.
            dino_img_size (int, optional): DINO image size. Defaults to 224.
            dtype (torch.dtype, optional): Data type for the DINO encoder. Defaults to torch.bfloat16.
            vf_weight (float, optional): Weight for visual foundation loss. Defaults to 1.0.
        """

        super().__init__(
            repa_encoder,
            c_dim_first,
            build_proj,
            img_is_neg1_1,
            rgb_channels,
            img_resize,
            feature_normalize,
            loss_type,
            dino_fixed_bs,
            dino_img_size,
            dtype,
            feature_resample_type=feature_resample_type,
        )
        self.distmat_margin = distmat_margin
        self.cos_margin = cos_margin
        self.distmat_weight = distmat_weight
        self.cos_weight = cos_weight
        self.vf_weight = vf_weight

        logger.debug(repr(self))

    def __repr__(self):
        return "VFLoss: " + (
            f"\n(\n    distmat_margin: {self.distmat_margin}\n"
            f"    cos_margin: {self.cos_margin}\n"
            f"    distmat_weight: {self.distmat_weight}\n"
            f"    cos_weight: {self.cos_weight}\n"
            f"    vf_weight: {self.vf_weight}\n)"
        )

    def calculate_adaptive_weight_vf(self, nll_loss, vf_loss, last_layer=None):
        vf_weight = self.vf_weight

        if last_layer is not None and nll_loss is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            vf_grads = torch.autograd.grad(vf_loss, last_layer, retain_graph=True)[0]

            vf_weight = torch.norm(nll_grads) / (torch.norm(vf_grads) + 1e-4)
            vf_weight = torch.clamp(vf_weight, 0.0, 1e8).detach()
            vf_weight = vf_weight * self.vf_weight
        else:
            vf_weight = self.vf_weight

        return vf_weight

    def vf_loss(self, z: Tensor, aux_feature: Tensor):
        z_flat = rearrange(z, "b c h w -> b c (h w)")
        aux_feature_flat = rearrange(aux_feature, "b c h w -> b c (h w)")
        z_norm = torch.nn.functional.normalize(z_flat, dim=1)
        aux_feature_norm = torch.nn.functional.normalize(aux_feature_flat, dim=1)
        z_cos_sim = torch.einsum("bci,bcj->bij", z_norm, z_norm)
        aux_feature_cos_sim = torch.einsum("bci,bcj->bij", aux_feature_norm, aux_feature_norm)
        diff = torch.abs(z_cos_sim - aux_feature_cos_sim)
        vf_loss_1 = torch.nn.functional.relu(diff - self.distmat_margin).mean()
        vf_loss_2 = torch.nn.functional.relu(
            1 - self.cos_margin - torch.nn.functional.cosine_similarity(aux_feature, z)
        ).mean()
        vf_loss = vf_loss_1 * self.distmat_weight + vf_loss_2 * self.cos_weight

        return vf_loss

    def forward(self, img, feature, nll_loss=None, last_layer=None):
        """
        Forward pass for the visual foundation model loss.

        Args:
            z (Tensor): The main feature tensor.
            img (Tensor): The input image tensor.
            nll_loss (Tensor, optional): The negative log-likelihood loss. Defaults to None.
            last_layer (Tensor, optional): The last layer of the encoder for gradient calculation. Defaults to None.

        Returns:
            Tensor: The computed VF loss.
        """
        # encode
        aux_feature = self._encode_img(img)
        assert torch.is_tensor(aux_feature), "aux_feature must be a tensor not a list, but got {}".format(
            type(aux_feature)
        )
        aux_feature = aux_feature.detach()

        # iterpolate features
        # _tgt_sz = feature.shape[-2:] if self.c_dim_first else feature.shape[-2]
        # aux_feature = self.interp_feat(aux_feature, _tgt_sz)
        aux_feature, feature = self._interp_teacher_or_student_features(aux_feature, feature)

        feature = self.projector(feature)
        assert feature.shape == aux_feature.shape, (
            "features and aux_feature must have the same shape to compute the repa loss"
        )

        # loss
        vf_loss = self.vf_loss(feature, aux_feature)

        # weighted
        vf_weight = self.calculate_adaptive_weight_vf(nll_loss, vf_loss, last_layer)

        return vf_loss * vf_weight


class LatentGramLoss(REPALoss):
    def __init__(
        self,
        repa_encoder=None,
        apply_norm=True,
        img_level=True,
        remove_neg=True,
        remove_only_teacher_neg=False,
        weight: float = 1.0,
        **repa_init_kwargs,
    ):
        super().__init__(repa_encoder, **repa_init_kwargs)
        # Loss
        self.mse_loss = torch.nn.MSELoss()

        # Parameters
        self.apply_norm = apply_norm
        self.remove_neg = remove_neg
        self.remove_only_teacher_neg = remove_only_teacher_neg
        self.img_level = img_level
        self.weight = weight

        if self.remove_neg or self.remove_only_teacher_neg:
            assert self.remove_neg != self.remove_only_teacher_neg

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            "  apply_norm={self.apply_norm},\n"
            "  remove_neg={self.remove_neg},\n"
            "  remove_only_teacher_neg={self.remove_only_teacher_neg},\n"
            "  img_level={self.img_level},\n"
            "  weight={self.weight}\n"
            ")"
        )

    # def calculate_adaptive_weight(self, nll_loss, loss, last_layer=None):
    #     if last_layer is not None and nll_loss is not None:
    #         nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
    #         loss_grads = torch.autograd.grad(loss, last_layer, retain_graph=True)[0]

    #         g_weight = torch.norm(nll_grads) / (torch.norm(loss_grads) + 1e-4)
    #         g_weight = torch.clamp(g_weight, 0.0, 1e8).detach()
    #         g_weight = g_weight * self.weight
    #     else:
    #         g_weight = self.weight

    #     return g_weight

    def gram_loss(self, output_feats, target_feats, img_level=True):
        """Comes from DINO v3 training loss"""

        # Dimensions of the tensor should be (B, N, dim)
        if img_level:
            assert len(target_feats.shape) == 3 and len(output_feats.shape) == 3

        # Float casting
        output_feats = output_feats.float()
        target_feats = target_feats.float()

        # SSL correlation
        if self.apply_norm:
            target_feats = F.normalize(target_feats, dim=-1)

        if not img_level and len(target_feats.shape) == 3:
            # Flatten (B, N, D) into  (B*N, D)
            target_feats = target_feats.flatten(0, 1)

        # Compute similarities
        # [B, N, D] x [B, N, D] -> [B, N, N] or [B*N, D] x [B*N, D] -> [B*N, B*N]
        target_sim = torch.matmul(target_feats, target_feats.transpose(-1, -2))

        # Patch correlation
        if self.apply_norm:
            output_feats = F.normalize(output_feats, dim=-1)

        if not img_level and len(output_feats.shape) == 3:
            # Flatten (B, N, D) into  (B*N, D)
            output_feats = output_feats.flatten(0, 1)

        # Compute similarities
        student_sim = torch.matmul(output_feats, output_feats.transpose(-1, -2))

        if self.remove_neg:
            target_sim[target_sim < 0] = 0.0
            student_sim[student_sim < 0] = 0.0

        elif self.remove_only_teacher_neg:
            # Remove only the negative sim values of the teacher
            target_sim[target_sim < 0] = 0.0
            student_sim[(student_sim < 0) & (target_sim < 0)] = 0.0

        return self.mse_loss(student_sim, target_sim)

    def _to_1d_seq(self, x, t="c"):
        if x.ndim == 4:
            x = rearrange(x, "b c h w -> b (h w) c")
            if t == "hw":
                x = x.transpose(1, 2)  # [b, c, hw]

        return x

    def forward(self, img, recon_img, nll_loss=None, last_layer=None):
        # encode
        target_features = self._encode_img(img, get_interm_feats=True, detach=True)
        student_features = self._encode_img(recon_img, get_interm_feats=True, detach=False)

        assert isinstance(target_features, (tuple, list)), (
            f"teacher features must be a list but got {type(student_features)}"
        )

        # Gram loss
        loss = 0.0
        for sf, tf in zip(student_features, target_features):
            sf, tf = map(self._to_1d_seq, (sf, tf))
            gram_loss = self.gram_loss(sf, tf, img_level=self.img_level)
            g_weight = 1.0  # self.calculate_adaptive_weight(nll_loss, gram_loss, last_layer)
            loss += g_weight * gram_loss

        return loss / len(student_features)


# * --- Test --- * #


def test_dinov3_pca():
    # Load the dino v3 model
    model_name = "dinov3_vitl16"
    model = load_repa_dino_v3_model(
        # "src/stage1/utilities/losses/dinov3/weights/remote_sensing_image_pretrained_SAT_493M/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
        model_name=model_name,
        pretrained_on="satellite",
    )
    logger.info("load model done.")
    model.eval()
    model = model.cuda().to(torch.bfloat16)
    model.requires_grad_(False)

    import PIL.Image

    img = "/Data4/cao/ZiHanCao/exps/HyperspectralTokenizer/data/YuZhongDataset/LoveDA/Train/Rural/images_png/157.png"
    img = PIL.Image.open(img)

    x = torch.as_tensor(np.array(img), dtype=torch.float32)
    x = x.permute(2, 0, 1)[None] / 255.0
    # norm using imagenet mean and std
    from torchvision.transforms import Normalize

    norm = Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    x = norm(x)
    logger.info(x.shape)

    x = x.cuda().to(torch.bfloat16)
    with torch.inference_mode() and torch.autocast("cuda", torch.bfloat16):
        y = model.get_intermediate_layers(  # type: ignore
            x, n=1, reshape=True, norm=True
        )[0]  # the last layer feature
        logger.info(y.shape)

    # pca
    from src.stage1.utilities.losses.repa.feature_pca import feature_pca_sk

    pca_img = feature_pca_sk(y, 3)
    # save pca img
    pca_img = pca_img.cpu().numpy()[0].transpose(1, 2, 0)
    pca_img = pca_img - pca_img.min()
    pca_img = pca_img / pca_img.max()
    PIL.Image.fromarray((pca_img * 255).astype(np.uint8)).save(f"pca_img_{model_name}.png")


def test_repa_loss():
    loss_fn = REPALoss(c_dim_first=True, build_proj=False, img_is_neg1_1=True).cuda()
    img = torch.randn(2, 3, 224, 224).cuda()
    features = torch.randn(2, 768, 28, 28).cuda()
    loss = loss_fn(img, features)
    print("Loss:", loss.item())


def test_repa_loss_hier():
    loss_fn = REPALoss(
        c_dim_first=True,
        build_proj=False,
        img_is_neg1_1=True,
        rgb_channels="mean",
        get_hier_teacher_feature=True,
        loss_type="hier_distillation",
        repa_model_type="siglip2",
        repa_model_name="google/siglip2-large-patch16-512",  # "google/siglip2-so400m-patch16-naflex",
    ).cuda()
    print("repa loss init done.")

    # 4 hier features
    student_features = [
        torch.randn(2, 1152, 28, 28).cuda(),
    ] * 4
    img_for_teacher = torch.randn(2, 3, 224, 224).cuda()
    loss = loss_fn(img_for_teacher, student_features)
    print(loss)


def test_vf_loss():
    loss_fn = (
        VFLoss(
            distmat_margin=0.1,
            cos_margin=0.1,
            distmat_weight=1.0,
            cos_weight=1.0,
            c_dim_first=True,
            build_proj=False,
            img_is_neg1_1=True,
            rgb_channels="random",
            img_resize="dino",
            dino_fixed_bs=None,
            dino_img_size=224,
        )
        .to(torch.bfloat16)
        .cuda()
    )
    dtype = torch.bfloat16
    img = torch.randn(2, 3, 224, 224).to(dtype).cuda()
    z = torch.randn(2, 768, 32, 32).to(dtype).cuda()
    with torch.autocast("cuda", dtype):
        loss = loss_fn(img, z)
    print("VFLoss:", loss.item())


def test_gram_loss():
    from tqdm import trange

    loss_fn = LatentGramLoss(c_dim_first=True, build_proj=False, img_is_neg1_1=True, rgb_channels="mean")
    loss_fn = loss_fn.to(torch.bfloat16).cuda()
    img = torch.randn(2, 3, 224, 224).cuda()
    feature = torch.randn(2, 1024, 28, 28).cuda()
    with torch.autocast("cuda", torch.bfloat16):
        for _ in trange(100):
            gram_loss = loss_fn(img, feature)
            # print(f"{gram_loss=}")


def test_siglip2_model():
    import math

    import lovely_tensors as lt
    from einops import rearrange
    from PIL import Image

    lt.monkey_patch()
    model, processor = load_siglip2_model(
        name="google/siglip2-so400m-patch16-naflex",
        cache_dir="/home/user/zihancao/Checkpoints/",
    )
    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open("000000039769.jpg")
    print(image.size)
    model = model.cuda()

    # inputs = processor(
    #     # text="",
    #     images=[image],
    #     do_rescale=True,
    #     # max_num_patches=1024,
    #     return_tensors="pt",
    # )
    processor_ = _siglip_processor_patcher(
        processor,
        interp_pe=False,
        # do_resize=False,
        # size=(512, 512),
        max_num_patches=2048,  # 32 * 32
    )
    inputs = processor_(images=image)
    x = inputs["pixel_values"] = inputs["pixel_values"].cuda()
    mask = shapes = None
    if "attention_mask" in inputs:
        mask = inputs["attention_mask"] = inputs["attention_mask"].cuda()
        shapes = inputs["spatial_shapes"]
    # inputs["interpolate_pos_encoding"] = True

    with torch.autocast("cuda", torch.bfloat16):
        with torch.no_grad():
            model_out = model(**inputs)
            last_hidden_state = model_out.last_hidden_state
    print(last_hidden_state.shape)

    if "attention_mask" in inputs:
        # mask: bs, n; pixel_values: bs, n, c;
        assert shapes is not None
        hiddens_2d = []
        for i in range(last_hidden_state.shape[0]):
            n = shapes[i].prod()
            hs = last_hidden_state[i, :n].view(1, *shapes[i], -1)  # (1, h, w, c)
            hs_2d = hs.permute(0, 3, 1, 2)  # (1, c, h, w)
            hiddens_2d.append(hs_2d)
        last_hidden_state = hiddens_2d[0]
    if last_hidden_state.ndim == 3:
        # square image
        last_hidden_state = rearrange(
            last_hidden_state,
            "b (h w) c -> b c h w",
            h=int(math.sqrt(last_hidden_state.shape[1])),
        )

    # xx = feature_pca_torch(xx, 3)
    xx = feature_pca_sk(last_hidden_state, 3)
    xx = (xx - xx.min()) / (xx.max() - xx.min())
    xx.rgb.fig.savefig("siglip2_so400m_patch16_feat.png")
    print("test_siglip2_model done.")


def test_pe_model():
    from src.data.litdata_hyperloader import ImageStreamingDataset
    from src.stage1.utilities.losses.repa.feature_pca import feature_pca_sk, pca_list

    ds = ImageStreamingDataset(input_dir="data2/RemoteSAM270k/LitData_hyper_images", to_neg_1_1=False)
    x = ds[10]["img"][None]  # [1, 3, H, W]
    x = x.to("cuda", torch.bfloat16)
    x = F.interpolate(x, size=(448, 448))

    pe_model = load_perception_model(
        "/Data/ZiHanCao/checkpoints/perception_models/PE-Spatial-L14-448.pt",
        model_name="PE-Spatial-L14-448",
        compile=False,
    )
    layer_take = PE_INTERACTION_INDEXES["PE-Spatial-L14-448"]
    print(f"Loaded model: {pe_model.__class__.__name__}")
    pe_model = pe_model.cuda()

    with torch.no_grad():
        with torch.autocast("cuda", torch.bfloat16):
            feats = pe_model.forward_features(x, layer_idx=layer_take, norm=True)
            # to 2d
            h, w = x.shape[2] // 14, x.shape[3] // 14
            feats = [rearrange(f, "b (h w) c -> b c h w", h=h, w=w) for f in feats]
    feats_pca, _ = pca_list(feats, 3)
    feats_pca_b = torch.cat(feats_pca, dim=0)

    for i, f in enumerate(feats):
        print(f"Feature {i}: {f.shape}")


if __name__ == "__main__":
    import lovely_tensors as lt
    from rich.traceback import install

    lt.monkey_patch()
    install(show_locals=True, code_width=120, width=200)

    # test_gram_loss()
    # test_siglip2_model()
    # test_repa_loss_hier()
    test_pe_model()

    # load_siglip2_model("google/siglip2-large-patch16-512")
