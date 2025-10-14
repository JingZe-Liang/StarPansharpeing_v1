import math
import sys
import warnings
from functools import lru_cache, partial
from pathlib import Path
from types import MethodType
from typing import Any, Callable, Literal, Optional, Sequence, TypeVar, cast, overload

import einx
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.state import AcceleratorState
from einops import einsum, rearrange, reduce
from jaxtyping import Float
from numpy.typing import NDArray
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers.create_norm_act import get_norm_act_layer
from torch import Tensor, is_tensor
from torch.distributed.tensor import DTensor, Shard
from torch.utils.file_baton import FileBaton
from torchvision.transforms import Normalize
from transformers import (
    AutoModel,
    AutoProcessor,
    BitsAndBytesConfig,
    Siglip2Model,
    Siglip2VisionModel,
    SiglipImageProcessor,
    SiglipProcessor,
)
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.siglip2.modeling_siglip2 import Siglip2VisionTransformer

from src.stage1.utilities.losses.gan_loss.utils import get_rgb_channels_for_model
from src.stage1.utilities.losses.repa.feature_pca import (
    feature_pca_sk,
    feature_pca_torch,
)
from src.utilities.config_utils import function_config_to_basic_types
from src.utilities.logging.print import log_print

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
    "google/siglip2-so400m-patch14-224": [
        5,
        11,
        17,
        23,
    ],  # patch=[14,16]; shape=[224,384,512]
    "google/siglip2-so400m-patch16-naflex": [7, 16, 22, 26],  # naflex backbone
}
SIGLIP2_FEATURE_INDEX: list[int] | None = None


def interpolate_features_2d_stacked(x, tgt_size: tuple[int, ...] | torch.Size):
    n_stacked = x.shape[0]
    for i in range(n_stacked):
        x[i] = interpolate_features_2d(x[i], tgt_size)
    return x


def interpolate_features_2d(x: Tensor, tgt_size: tuple[int, ...] | torch.Size):
    B, D, H, W = x.shape
    H_tgt, W_tgt = tgt_size

    if H == H_tgt and W == W_tgt:
        return x
    else:
        return F.interpolate(
            x, size=(H_tgt, W_tgt), mode="bicubic", align_corners=False
        )


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
        x = F.interpolate(x, size=(H2, W2), mode="bicubic", align_corners=False)

        # Reshape back to sequence
        return x.permute(0, 2, 3, 1).reshape(B, target_len, D)


def mean_flat(x: Tensor):
    return x.mean(dim=list(range(1, len(x.shape))))


def norm_feature(feat: Tensor, dim=1):
    return F.normalize(feat, dim=dim)


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


def repa_loss_(feature_teacher, feature_student, dim=-1):
    assert feature_teacher.shape == feature_student.shape
    return -torch.sum(feature_teacher * feature_student, dim=dim)


def am_ratio_spatial_loss(feature_teacher, feature_student, dim=-1):
    assert feature_teacher.shape == feature_student.shape
    un_sim = 1 - torch.cosine_similarity(feature_teacher, feature_student, dim=1)
    smooth_l1 = F.smooth_l1_loss(feature_teacher, feature_student)
    return un_sim + smooth_l1 * 0.1


def token_relation_loss(
    feature_teacher,
    feature_student,
    dim=-1,
    norm=False,
    img_level=True,
    remove_negative=True,
    remove_only_teacher_neg=False,
):
    # gram loss actually
    # feature shape: [b * nf, l, c], dim=-1; [b * nf, c, l], dim=1
    if dim == 1:
        feature_teacher = feature_teacher.transpose(1, 2)
        feature_student = feature_student.transpose(1, 2)
    if not img_level:
        feature_teacher = feature_teacher.flatten(0, 1)  # (bs * l, c)
        feature_student = feature_student.flatten(0, 1)  # (bs * l, c)

    if img_level:
        relation_t = torch.einsum("blc,blc->bll", feature_teacher, feature_teacher)
        relation_s = torch.einsum("blc,blc->bll", feature_student, feature_student)
    else:
        relation_t = torch.einsum("Nc,Nc->NN", feature_teacher, feature_teacher)
        relation_s = torch.einsum("Nc,Nc->NN", feature_student, feature_student)

    if norm:
        # relation_t = F.normalize(relation_t, dim=-1)
        relation_s = F.normalize(relation_s, dim=-1)

    if remove_negative:
        relation_t = relation_t.clamp(min=0.0)
        relation_s = relation_s.clamp(min=0.0)
    elif remove_only_teacher_neg:
        relation_t = relation_t.clamp(min=0.0)
        relation_s[(relation_s < 0) & (relation_t < 0)] = (
            0.0  # ? Check this, if is correct after do teacher clamp
        )

    loss = F.mse_loss(relation_t, relation_s)
    return loss

    loss = F.mse_loss(relation_t, relation_s)
    return loss


# *==============================================================
# * Load DINOv2/v3 or PE models
# *==============================================================


def load_perception_model(
    weight_path: str | Path,
    model_name: str = "PE-Core-L14-336",
    pretrained_on: Literal["core", "llm", "spatial"] = "core",
    compile=True,
):
    """
    Perception Encoder: https://github.com/facebookresearch/perception_models
    """

    import src.stage1.perception_models.core.vision_encoder.pe as pe

    visual_available_cfgs = pe.VisionTransformer.available_configs()
    assert model_name in visual_available_cfgs, (
        f"Model {model_name} not available. Available models are {visual_available_cfgs}"
    )
    assert weight_path is not None, (
        "weight path should not be None when loading the perception encoder model"
    )

    # Load the model
    cfg = pe.PE_VISION_CONFIG[model_name]
    model = pe.VisionTransformer.from_config(
        cfg, pretrained=True, checkpoint_path=str(weight_path)
    )
    log_print(f"[PE Model]: Load model {model_name} with config {cfg}")

    if compile:
        model = torch.compile(model, mode="reduce-overhead")
        log_print("[PE model]: Model compiled with reduce-overhead")

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
        assert weight_path is not None, (
            f"can not find weight {model_name=} at {weight_dir}"
        )
    elif weight_path is None and model_name is None:
        raise ValueError("Either model_name or weight_path must be specified.")

    assert weight_path is not None, f"{weight_path=} does not exists"
    log_print(
        f"[Dino v3 in REPA]: use Dino v3 model: {model_name} loaded from {weight_path}."
    )
    assert Path(weight_path).exists(), "Dino v3 model weight path does not exists"
    sys.path.append(str(repo_dir))
    dino_model = torch.hub.load(
        repo_dir, model_name, source="local", weights=weight_path
    )
    dino_model = cast(nn.Module, dino_model)
    if compile:
        dino_model = torch.compile(dino_model, mode="reduce-overhead")
        dino_model = cast(torch._dynamo.OptimizedModule, dino_model)
        log_print("[Dino v3 model]: compiled model done")
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
        raise ValueError(
            f"Unknown model loading source {load_from}, must be 'torch' or 'timm'."
        )
    if compile:
        model = torch.compile(model)
        log_print(f"[Dino v2 model]: compiled model done")
    return model  # type: ignore[return-value]


def _siglip_vit_encoder_forward_features_patcher(
    self,
    inputs_embeds,
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
):
    output_last_hs = kwargs.pop("output_hidden_states", False)
    index = []

    global SIGLIP2_FEATURE_INDEX
    if output_last_hs:
        index = SIGLIP2_FEATURE_INDEX
        assert index is not None, "Siglip2 feature index is not set"

    hidden_states = inputs_embeds
    features = []
    for i, encoder_layer in enumerate(self.layers):
        hidden_states = encoder_layer(hidden_states, attention_mask, **kwargs)
        if output_last_hs and i in index:
            features.append(hidden_states)
    return BaseModelOutput(
        last_hidden_state=hidden_states,
        hidden_states=features if len(features) > 0 else None,
    )


def load_siglip2_model(
    name="google/siglip2-so400m-patch16-naflex",
    use_bnb=False,
    attn_implem="sdpa",
    use_automodel=True,
    cache_dir=None,
) -> tuple[Siglip2VisionTransformer, SiglipProcessor]:
    if use_bnb:
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)
    else:
        bnb_config = None

    if not use_automodel:
        # model, processor = None, None
        raise NotImplementedError("AutoModel is not supported yet")
    else:
        model = AutoModel.from_pretrained(
            name,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation=attn_implem,
            cache_dir=cache_dir,
        )
        # remove the text model
        model.text_model = None
        vision_model = model.vision_model
        processor = AutoProcessor.from_pretrained(name, cache_dir=cache_dir)

    model = cast(Siglip2VisionModel, model)

    global SIGLIP2_FEATURE_INDEX
    SIGLIP2_FEATURE_INDEX = SIGLIP2_INTERACTION_INDEXES[name]
    log_print(f"[Siglip2]: using feature index {SIGLIP2_FEATURE_INDEX}")

    # Override the forward_features method
    vision_model.encoder.forward = MethodType(
        _siglip_vit_encoder_forward_features_patcher, vision_model.encoder
    )

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
    model_name: str = "dinov2_vitb14",
    weight_path: str | Path | None = None,
    load_from="torch",
    version: int | str = 2,
    dino_v3_pretrained_on: Literal["satellite", "web"] = "satellite",
    compile=True,
):
    if version == 2 or version == "dinov2":
        return load_repa_dino_v2_model(load_from, model_name, weight_path, compile)
    elif version == 3 or version == "dinov3":
        return load_repa_dino_v3_model(
            weight_path,
            model_name,
            pretrained_on=dino_v3_pretrained_on,
            compile=compile,
        )
    elif version == "pe":
        assert weight_path is not None, (
            "weight_path should not be None when loading PE model"
        )
        return load_perception_model(
            weight_path,
            model_name,
            compile=compile,
        )
    elif version == "siglip2":
        assert weight_path is not None, (
            "weight_path should not be None when loading Siglip2 model"
        )
        model, processor = load_siglip2_model(weight_path)
        return model, processor
    else:
        raise ValueError(f"Unknown DINO/PE version {version}")


# *==============================================================
# * Loss modules
# *==============================================================


class REPALoss(torch.nn.Module):
    @function_config_to_basic_types
    def __init__(
        self,
        repa_encoder: nn.Module | None = None,
        c_dim_first=False,  # teacher model output is [b, l, c] or [b, c, h, w]
        build_proj=False,
        img_is_neg1_1=True,
        rgb_channels: list[int]
        | Literal["random", "mean", "largest"]
        | str
        | None = None,
        img_resize: Literal["dino"] | tuple | None = "dino",
        feature_normalize: bool = False,
        loss_type: str = "repa_original",
        # FIXME: rename these args, not dino but for repa encoder options.
        dino_fixed_bs: int | None = None,
        dino_img_size: int = 224,
        dtype: torch.dtype = torch.bfloat16,
        dino_type: str = "torch",  # [torch, timm, siglip2]
        dino_name="dinov3_vitl16",
        dino_version: int = 3,
        dino_pretrained_path: str | None = None,
        dino_pretrained_on: str | Literal["satellite", "web"] | None = "satellite",
        feature_resample_type: Literal[
            "match_teacher", "match_student"
        ] = "match_student",
    ):
        """Align the tokenizer/model feature with DINO pretrained model feature.

        This loss function implements REPA (Representation Alignment) to align the features
        from a tokenizer/model with features extracted from DINO pretrained models. It supports
        both 1D (sequence) and 2D (spatial) feature representations and provides various
        configuration options for flexible usage.

        Args:
            repa_encoder (nn.Module | None, optional): Pre-loaded DINO encoder. If None,
                will load based on other parameters. Defaults to None.
            c_dim_first (bool, optional): Whether features are 2D (True) or 1D (False).
                If True, features have shape [b, c, h, w]; if False, [b, l, c].
                Defaults to False.
            build_proj (bool, optional): Whether to build projection layer in this loss class.
                Not recommended as it's better to handle projection in the model.
                Defaults to False.
            img_is_neg1_1 (bool, optional): Whether input image values range from (-1, 1).
                If False, assumes (0, 1) range. Defaults to True.
            rgb_channels (list[int] | Literal["random", "mean"] | str | None, optional):
                RGB channel selection strategy. Can be specific indices, "random" for random
                selection, "mean" for mean pooling, or None for original channels.
                Defaults to None.
            img_resize (Literal["dino"] | tuple | None, optional): Input image resize strategy.
                Can be "dino" for DINO's pretrained size, tuple for specific size,
                or None for auto-sizing. Defaults to "dino".
            feature_normalize (bool, optional): Whether to normalize features before
                computing loss. Defaults to False.
            loss_type (str, optional): Type of loss to compute. Supports "repa_original"
                and other variants. Defaults to "repa_original".
            dino_fixed_bs (int | None, optional): Fixed batch size for DINO forward pass.
                If None, uses input batch size. Defaults to None.
            dino_img_size (int, optional): DINO model's expected image size. Defaults to 224.
            dtype (torch.dtype, optional): Data type for DINO encoder computations.
                Defaults to torch.bfloat16.
            dino_type (str, optional): DINO model type ("torch" or "timm").
                Defaults to "torch".
            dino_name (str, optional): DINO model name/identifier.
                Defaults to "dinov3_vitl16".
            dino_version (int, optional): DINO version (2 or 3). Defaults to 3.
            dino_pretrained_path (str | None, optional): Path to custom DINO pretrained weights.
                If None, uses default weights based on model name. Defaults to None.
            dino_pretrained_on (str | Literal["satellite", "web"] | None, optional):
                Dataset used for DINO pretraining. Affects default weight selection.
                Defaults to "satellite".
            feature_resample_type (Literal["match_teacher", "match_student"], optional):
                Strategy for resampling features to match dimensions. "match_teacher"
                resamples student to teacher size, "match_student" does the opposite.
                Defaults to "match_student".
        """
        super().__init__()
        self.rgb_channels = rgb_channels
        self.img_resize = img_resize
        self.feature_normalize = feature_normalize
        self.loss_type = loss_type
        self.feature_resample_type = feature_resample_type
        assert loss_type in ["repa_original", "am_ratio_spatial", "token_relation"], (
            f"loss_type {loss_type} not supported, must be in "
            f"['repa_original', 'am_ratio_spatial', 'token_relation']"
        )

        if isinstance(img_resize, (tuple, list)):
            assert img_resize[0] % 14 == 0 and img_resize[1] % 14 == 0, (
                "img_size[0] must be divisible by patch size"
            )
        self.dino_fixed_bs = dino_fixed_bs
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
                raise TypeError(
                    f"rgb_channels must be list or tuple or str, but got {type(rgb_channels)}"
                )

        # encoder
        self.dino_type = dino_type
        if repa_encoder is not None:
            self.repa_encoder = repa_encoder
        else:
            baton = FileBaton(
                "/tmp/repa_model_loading.lock"
            )  # Use a file baton to ensure single process access
            # load repa encoder in multiprocessing context
            if dino_version == 3:
                self.dino_type = "torch"
            load_kwargs = dict(
                model_name=dino_name,
                weight_path=dino_pretrained_path,
                load_from=dino_type,
                version=dino_version,
                dino_v3_pretrained_on=dino_pretrained_on,
                compile=False,
            )
            self.dino_name = dino_name
            if baton.try_acquire():
                try:
                    repa_encoder = load_repa_encoder(**load_kwargs)
                finally:
                    baton.release()
            else:
                baton.wait()
                repa_encoder = load_repa_encoder(**load_kwargs)
            if isinstance(repa_encoder, tuple):
                self.repa_encoder = repa_encoder[0]
                processor = repa_encoder[1]
                # do resize in the processor
                max_n_patches = (dino_img_size // 16) ** 2
                self.processor = _siglip_processor_patcher(
                    processor,
                    interp_pe=False,
                    # size=(dino_img_size, dino_img_size),
                    max_num_patches=max_n_patches,
                )

            self.repa_encoder.image_size = dino_img_size
            self.repa_encoder = self.repa_encoder.to(dtype)
            self.repa_encoder.requires_grad_(False)
            self.repa_encoder.eval()

        log_print("[REPA Loss]: load pretrained dino visual foundation model done")
        self.c_dim_first = c_dim_first
        # if c_dim_first:
        #     self.interp_feat = interpolate_features_2d
        # else:
        #     self.interp_feat = interpolate_features_1d

        # mlp projection to aligh the dim
        self.build_proj = build_proj
        if build_proj:
            self.projector = build_mlp(
                self.repa_encoder.embed_dim,
                2048,
                self.repa_encoder.embed_dim,
                is_1d=not c_dim_first,
            )
            log_print(
                "build repa loss in loss class, remember to optimize the projector, "
                "or match to repa dim in model forward",
                "warning",
            )
        else:
            self.projector = nn.Identity()

        self.img_is_neg1_1 = img_is_neg1_1
        self.normalize = Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

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

    def _norm_img_before_repa(self, img):
        if self.img_is_neg1_1:
            img = (img + 1) / 2

        # norm, image dim must be 4d
        assert img.ndim == 4 and img.shape[1] == 3, (
            "img dim must be 4d and have 3 channels"
        )
        img = self.normalize(img)
        return img

    def _forward_features(
        self, x: dict | Tensor, get_interm_feats=False, detach=True, **kwargs
    ):
        if self.dino_type == "siglip2":
            assert isinstance(x, dict), "inputs must be a dict for Siglip2 model"
            return self._forward_siglip_features(x, get_interm_feats, detach=detach)
        else:
            assert is_tensor(x), "inputs must be a tensor for DINOv2/v3 model"
            return self._forward_dino_features(
                x, get_interm_feats, detach=detach, **kwargs
            )

    @torch.autocast("cuda", torch.bfloat16)
    def _forward_siglip_features(
        self, inputs: dict, get_interm_feats=False, detach=True
    ):
        assert self.dino_type == "siglip2", (
            "dino_type must be 'siglip2' when using Siglip2 model"
        )

        # image is the output of the processor
        _is_naflex = False
        if "spatial_shapes" in inputs:
            shapes = inputs["spatial_shapes"]  # (bs, 2)
            _is_naflex = True
        else:
            shapes = inputs["pixel_values"].shape  # (bs, c, h, w)
        model_out_ = self.repa_encoder(**inputs)  # Tensor (bs, n, c) or (bs, c, h, w)

        out_feats: Tensor | list[Tensor]
        if get_interm_feats:
            out_feats = model_out_.hidden_states  # list of Tensor
        else:
            out_feats = model_out_.last_hidden_state  # Tensor

        n_feats = 1
        if isinstance(out_feats, list):
            n_feats = len(out_feats)

        # Post-processing
        # Per-tensor process
        bs = inputs["pixel_values"].shape[0]

        def _per_tensor_process(feat: Tensor) -> Tensor:
            if _is_naflex:
                feat_valid = []
                for i in range(n_feats):
                    # Usually, the batch of tensor has the same shape and
                    # valid length, but in general, we keep the for-loop
                    s = shapes[i]  # (2,): mean h, w
                    n_patches_i = s.prod().item()
                    h_sample = (
                        feat[i, :n_patches_i].view(1, *s, -1).permute(0, 3, 1, 2)
                    )  # (1, c, h, w)
                    feat_valid.append(h_sample)  # 2d feature
                # Stack back to a tensor
                feat = torch.stack(feat_valid, dim=0)  # (bs, c, h, w)
            else:
                # assert feat.ndim == 4, f"Siglip2 (not naflex version) output feature must be 4d but got {feat.ndim}d"
                if feat.ndim == 3:
                    # Reshape back to 2d
                    h = w = int(math.sqrt(feat.shape[1]))
                    feat = rearrange(feat, "b (h w) c -> b c h w", h=h, w=w)
            return feat

        if get_interm_feats:
            # is a list of features
            img_feats = [
                _per_tensor_process(feat) for feat in out_feats
            ]  # 2d list features
        else:
            img_feats = _per_tensor_process(out_feats)  # 2d features

        # Detach and return 1 feature out?
        if isinstance(img_feats, list) and len(img_feats) == 1:
            img_feats = img_feats[-1]
        elif isinstance(img_feats, Tensor):
            pass
        else:  # is a list of Tensors
            # Stack the Siglip2 features
            img_feats = torch.stack(img_feats, dim=0)  # [4, B, C, H, W]

        return img_feats

    @torch.autocast("cuda", torch.bfloat16)
    def _forward_dino_features(
        self, img, get_interm_feats=False, detach=True
    ) -> Tensor | list[Tensor]:
        img = self._to_dtensor(img)

        if self.dino_type == "torch":
            # if not self.c_dim_first:  # distill for vit 1d features
            #     img_feats = self.repa_encoder.forward_features(img)[  # type: ignore
            #         "x_norm_patchtokens"
            #     ]  # (bs, 256, 768)
            # else:  # distill for cnn 2d features
            layers_to_take = (
                DINOv3_INTERACTION_INDEXES.get(self.dino_name, 1)
                if get_interm_feats
                else 1
            )
            img_feats = self.repa_encoder.get_intermediate_layers(  # type: ignore
                img, n=layers_to_take, reshape=self.c_dim_first, norm=True
            )  # the last layer feature
            if len(img_feats) == 1:
                img_feats = img_feats[0]
        elif self.dino_type == "timm":
            b, c, h, w = img.shape
            assert h % 16 == 0 and w % 16 == 0, "image size must be divisible by 16"
            img_feats = (
                self.repa_encoder.forward_features(img)[:, 1:]  # type: ignore
                .reshape(b, h // 16, w // 16, -1)
                .permute(0, 3, 1, 2)
            )  # (bs, c, h, w)
        else:
            raise ValueError(
                f"Unknown model type: {self.dino_type}. Must be 'torch', 'timm', or 'siglip2'"
            )

        # Dtensor to Tensor
        if isinstance(img_feats, DTensor):
            img_feats = img_feats.full_tensor()
        elif isinstance(img_feats, (tuple, list)) and isinstance(img_feats[0], DTensor):
            img_feats = list(img_feats)
            for i, img_feat in enumerate(img_feats):
                img_feats[i] = img_feat.full_tensor()

        # Detach all tensors
        if detach:
            if isinstance(img_feats, Tensor):
                img_feats = img_feats.detach()
            elif isinstance(img_feats, (tuple, list)):
                img_feats = [f.detach() for f in img_feats]

        if get_interm_feats:
            # Stack all features
            assert isinstance(img_feats, (tuple, list)), (
                "img_feats must be a list when get_interm_feats is True"
            )
            img_feats = torch.stack(img_feats, dim=0)  # [4, B, C, H, W]

        return img_feats

    def _resize_img(self, img, size) -> Any:
        assert img.shape[1] == 3, (
            f"img must be rgb images but got image shaped as {img.shape}"
        )

        # resize images
        _interp_kwargs = {"input": img, "mode": "bicubic", "align_corners": False}
        if tuple([self.repa_encoder.image_size] * 2) != size:
            if self.img_resize == "dino":  # to 224 x 224 pretrained size
                _interp_kwargs["size"] = self.repa_encoder.image_size
            elif isinstance(self.img_resize, (tuple, list)):
                assert len(self.img_resize) == 2, "img_resize must be 2d tuple"
                _interp_kwargs["size"] = self.img_resize
            else:
                _interp_kwargs["size"] = (
                    next_divisble_of_y(img.shape[-2], 14),
                    next_divisble_of_y(img.shape[-1], 14),
                )
                log_print(
                    "[Repa Resize]: image not resize into dino pretrained size",
                    "warning",
                    warn_once=True,
                )
            img = F.interpolate(**_interp_kwargs)

        img = self._norm_img_before_repa(img)
        return img

    @torch.no_grad()
    def _encode_img(
        self,
        img: Float[Tensor, "bs c h w"],
        get_interm_feats=False,
        use_linstretch=True,
        detach=True,
    ):
        img_size = tuple(img.shape[-2:])
        bs = img.shape[0]
        assert img.ndim == 4, f"img must be 4d tensor but got {img.ndim}d"

        # Use the reusable function to handle RGB channel selection
        img = get_rgb_channels_for_model(
            rgb_channels=self.rgb_channels,
            img=img,
            use_linstretch=use_linstretch,  # Keep original behavior without linear stretching
            pca_fn=feature_pca_torch if self.rgb_channels == "pca" else None,
        )

        # Resize image / Preprocess image
        inputs = None
        if hasattr(self, "processor"):  # Siglip2 processor
            if self.img_is_neg1_1:
                img = (img + 1) / 2
            # make 1d patches nor downsample by AutoProcessor
            inputs = self.processor(images=img)["pixel_values"]
        else:
            inputs = self._resize_img(img, img_size)

        # loop to get the features
        if self.dino_fixed_bs is None or bs < self.dino_fixed_bs:
            img_feats = self._forward_features(
                inputs,
                detach=detach,
                get_interm_feats=get_interm_feats,
            )
        else:
            # TODO: add support for siglip2
            assert self.dino_type != "siglip2", (
                "Siglip2 model does not support fixed batch size inference"
            )

            # macro-batch-size inference
            img_feats: list[torch.Tensor] = []
            ret_lst = False
            for i in range(0, img.shape[0], self.dino_fixed_bs):
                img_mb = img[i : i + self.dino_fixed_bs]
                img_feats_mb = list(
                    self._forward_features(
                        img_mb, get_interm_feats=get_interm_feats, detach=detach
                    )
                )  # 4 feature if is a list
                if isinstance(img_feats_mb, (tuple, list)):
                    ret_lst = True
                img_feats.append(img_feats_mb)  # type: ignore[list-item]

            if ret_lst:
                n_feats = len(img_feats[0])
                for i in range(n_feats):
                    img_feats[i] = torch.cat(img_feats[i], dim=0)
            else:
                img_feats = torch.cat(img_feats, dim=0)

        return img_feats

    @overload
    def _interp_teacher_or_student_features(
        self, teacher_feat: Tensor, student_feat: Tensor
    ) -> tuple[Tensor, Tensor]: ...

    @overload
    def _interp_teacher_or_student_features(
        self, teacher_feat: list[Tensor], student_feat: list[Tensor]
    ) -> tuple[list[Tensor], list[Tensor]]: ...

    def _interp_teacher_or_student_features(self, teacher_feat, student_feat):
        is_feat_lst = isinstance(teacher_feat, (tuple, list)) or teacher_feat.ndim == 5
        if is_feat_lst:
            assert isinstance(student_feat, (tuple, list)) or student_feat.ndim == 5, (
                f"if teacher_feat is a list, student_feat must be a list or stacked tensor too, "
                f"but got {type(student_feat)} and {len(student_feat)=}"
            )
            assert len(teacher_feat) == len(student_feat), (
                f"len(teacher_feat) != len(student_feat), {len(teacher_feat)}, {len(student_feat)}"
            )

        # Match teacher features to student features
        if self.feature_resample_type == "match_student":

            def _interp_feat_match_stu(t_feat, s_feat):
                st_ndim = s_feat.ndim
                assert st_ndim in (3, 4, 5), "student feature must be 1d or 2d feature"
                func_mapping_ = {
                    3: interpolate_features_1d,
                    4: interpolate_features_2d,
                    5: interpolate_features_2d_stacked,
                }
                tgt_size = s_feat.shape[-2:] if st_ndim in (4, 5) else s_feat.shape[-2]
                t_feat = func_mapping_[st_ndim](t_feat, tgt_size)
                return t_feat

            if is_feat_lst:
                _is_stacked = isinstance(teacher_feat, torch.Tensor)
                teacher_feat = [
                    _interp_feat_match_stu(t_feat, s_feat)
                    for t_feat, s_feat in zip(teacher_feat, student_feat)
                ]
                if _is_stacked:
                    teacher_feat = torch.stack(teacher_feat)
            else:
                teacher_feat = _interp_feat_match_stu(teacher_feat, student_feat)

        else:  # student feature matched to teacher feature

            def _interp_feat_match_tch(t_feat, s_feat):
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
                    _interp_feat_match_tch(t_feat, s_feat)
                    for t_feat, s_feat in zip(teacher_feat, student_feat)
                ]
            else:
                student_feat = _interp_feat_match_tch(teacher_feat, student_feat)

        return teacher_feat, student_feat

    def repa_loss(self, teacher_feat: Tensor, student_feat: Tensor):
        # [bs, l, c] or [bs, c, h, w]
        # 1. interpolate
        teacher_feat, student_feat = self._interp_teacher_or_student_features(
            teacher_feat, student_feat
        )

        # 2. dino feature -> Featup -> model feature
        # TODO: use featup to upsample the dino feature into (high-resolution) model (or says)
        # tokenizer feature
        """
        some featup upsampling methods

        self.featup: nn.Module
        teacher_feat = self.featup(teacher_feat)
        """

        # flatten the model feature
        student_feat = self.projector(student_feat)

        # 3. repa loss
        # stack all n_feature and bs dim together
        if self.c_dim_first:
            dim = 1
            rarg_pattn = "... c h w -> (...) c (h w)"
        else:
            dim = -1
            rarg_pattn = "... h w c -> (...) (h w) c"

        teacher_feat = rearrange(teacher_feat, rarg_pattn)
        student_feat = rearrange(student_feat, rarg_pattn)

        # 3.1 Whether to keep the feature magnitude
        if self.feature_normalize:
            teacher_feat = norm_feature(teacher_feat, dim=dim)
            student_feat = norm_feature(student_feat, dim=dim)

        assert teacher_feat.shape == student_feat.shape, (
            "img_feat and model_feat must have the same shape to compute the repa loss, "
            f"but got {tuple(teacher_feat.shape)} and {tuple(student_feat.shape)}"
        )

        # 3.2 loss
        if self.loss_type == "repa_original":
            repa_loss = repa_loss_(teacher_feat, student_feat, dim=dim)
        elif self.loss_type == "am_ratio":
            repa_loss = am_ratio_spatial_loss(teacher_feat, student_feat, dim=dim)
        elif self.loss_type == "token_relation":
            repa_loss = token_relation_loss(teacher_feat, student_feat, dim=dim)
        else:
            raise ValueError(f"Unknown repa loss type {self.loss_type}")
        return repa_loss.mean()

    def forward(self, img: Tensor, student_feature: Tensor):
        teacher_feat = self._encode_img(img)
        assert torch.is_tensor(teacher_feat), (
            "teacher feature must be a tensor not a list, but got {}".format(
                type(teacher_feat)
            )
        )
        teacher_feat = teacher_feat.detach()
        repa_loss = self.repa_loss(teacher_feat, student_feature)
        return repa_loss

    def named_parameters(self, *args, **kwargs):
        if self.build_proj:
            return self.projector.named_parameters(*args, **kwargs)
        else:
            return []

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
            dm = self.repa_encoder.patch_embed.proj.weight.device_mesh
            img = DTensor.from_local(img, dm, placements=(Shard(0),))

        return img


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
        feature_resample_type: Literal[
            "match_teacher", "match_student"
        ] = "match_student",
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

        log_print(repr(self), "debug")

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
        aux_feature_cos_sim = torch.einsum(
            "bci,bcj->bij", aux_feature_norm, aux_feature_norm
        )
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
        assert torch.is_tensor(aux_feature), (
            "aux_feature must be a tensor not a list, but got {}".format(
                type(aux_feature)
            )
        )
        aux_feature = aux_feature.detach()

        # iterpolate features
        # _tgt_sz = feature.shape[-2:] if self.c_dim_first else feature.shape[-2]
        # aux_feature = self.interp_feat(aux_feature, _tgt_sz)
        aux_feature, feature = self._interp_teacher_or_student_features(
            aux_feature, feature
        )

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
        student_features = self._encode_img(
            recon_img, get_interm_feats=True, detach=False
        )

        assert isinstance(target_features, (tuple, list)), (
            f"teacher features must be a list but got {type(student_features)}"
        )

        # Gram loss
        loss = 0.0
        for sf, tf in zip(student_features, target_features):
            sf, tf = map(self._to_1d_seq, (sf, tf))
            gram_loss = self.gram_loss(sf, tf, img_level=self.img_level)
            g_weight = (
                1.0  # self.calculate_adaptive_weight(nll_loss, gram_loss, last_layer)
            )
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
    log_print("load model done.")
    model.eval()
    model = model.cuda().to(torch.bfloat16)
    model.requires_grad_(False)

    import numpy as np
    import PIL.Image

    img = "/Data4/cao/ZiHanCao/exps/HyperspectralTokenizer/data/YuZhongDataset/LoveDA/Train/Rural/images_png/157.png"
    img = PIL.Image.open(img)

    x = torch.as_tensor(np.array(img), dtype=torch.float32)
    x = x.permute(2, 0, 1)[None] / 255.0
    # norm using imagenet mean and std
    from torchvision.transforms import Normalize

    norm = Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    x = norm(x)
    log_print(x.shape)

    x = x.cuda().to(torch.bfloat16)
    with torch.inference_mode() and torch.autocast("cuda", torch.bfloat16):
        y = model.get_intermediate_layers(  # type: ignore
            x, n=1, reshape=True, norm=True
        )[0]  # the last layer feature
        log_print(y.shape)

    # pca
    from src.stage1.utilities.losses.repa.feature_pca import feature_pca_sk

    pca_img = feature_pca_sk(y, 3)
    # save pca img
    pca_img = pca_img.cpu().numpy()[0].transpose(1, 2, 0)
    pca_img = pca_img - pca_img.min()
    pca_img = pca_img / pca_img.max()
    PIL.Image.fromarray((pca_img * 255).astype(np.uint8)).save(
        f"pca_img_{model_name}.png"
    )


def test_repa_loss():
    loss_fn = REPALoss(c_dim_first=True, build_proj=False, img_is_neg1_1=True).cuda()
    img = torch.randn(2, 3, 224, 224).cuda()
    features = torch.randn(2, 768, 28, 28).cuda()
    loss = loss_fn(img, features)
    print("Loss:", loss.item())


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

    loss_fn = LatentGramLoss(
        c_dim_first=True, build_proj=False, img_is_neg1_1=True, rgb_channels="mean"
    )
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
    import requests
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


if __name__ == "__main__":
    # test_gram_loss()
    test_siglip2_model()
