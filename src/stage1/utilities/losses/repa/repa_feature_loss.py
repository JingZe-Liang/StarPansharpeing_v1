import math
import sys
import warnings
from functools import lru_cache, partial
from pathlib import Path
from typing import Literal, Sequence, cast

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.state import AcceleratorState
from einops import rearrange, reduce
from jaxtyping import Float
from numpy.typing import NDArray
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers.create_norm_act import get_norm_act_layer
from torch import Tensor
from torch.distributed.tensor import DTensor, Shard
from torch.utils.file_baton import FileBaton
from torchvision.transforms import Normalize

from src.stage1.utilities.losses.repa.feature_pca import feature_pca_torch
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


def choose_largest_bands(img: Float[Tensor, "... c h w"]):
    # Warning: this function is expensive
    mean_cs = img.view(-1, *img.shape[-3:]).mean((0, -2, -1)).detach()
    mean_cs = np.asarray(mean_cs.cpu())
    indices = np.argsort(mean_cs)[::-1][:3].tolist()
    assert indices[-1] < img.shape[1], (
        f"Invalid channel index {indices[-1]} for image with {img.shape[1]} channels."
    )
    return indices


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
        f"[Dino v3 in REPA]: use Dino v3 model: {model_name} loaded from {weight_path} "
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
        dino_fixed_bs: int | None = None,
        dino_img_size: int = 224,
        dtype: torch.dtype = torch.bfloat16,
        dino_type: str = "torch",  # [torch, timm]
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
            if baton.try_acquire():
                try:
                    self.repa_encoder = load_repa_encoder(**load_kwargs)
                finally:
                    baton.release()
            else:
                baton.wait()
                self.repa_encoder = load_repa_encoder(**load_kwargs)
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

    @torch.autocast("cuda", torch.bfloat16)
    def _forward_features(self, img):
        # from torch.distributed.tensor.device_mesh import init_device_mesh, DeviceMesh
        # from torch.distributed.tensor import DTensor
        img = self._to_dtensor(img)

        if self.dino_type == "torch":
            if not self.c_dim_first:  # distill for vit 1d features
                img_feats = self.repa_encoder.forward_features(img)[  # type: ignore
                    "x_norm_patchtokens"
                ]  # (bs, 256, 768)
            else:  # distill for cnn 2d features
                img_feats = self.repa_encoder.get_intermediate_layers(  # type: ignore
                    img, n=1, reshape=True, norm=True
                )[0]  # the last layer feature
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
                f"Unknown dino_type: {self.dino_type}. Must be 'torch' or 'timm'."
            )

        if isinstance(img_feats, DTensor):
            img_feats = img_feats.full_tensor()

        return img_feats

    @torch.no_grad()
    def _encode_img(self, img: Float[Tensor, "bs c h w"]):
        _img_sz = tuple(img.shape[-2:])

        if self.rgb_channels is not None and img.shape[1] > 3:
            assert img.shape[1] >= 3, "img must be hyperspectral images"
            if self.rgb_channels == "random":
                _rgb_chan_select = torch.randperm(img.shape[1])[:3]
                rgb_channels = _rgb_chan_select.tolist()
                img = img[:, rgb_channels]
            elif (
                not isinstance(self.rgb_channels, (list, tuple))
                and self.rgb_channels.startswith("random")
                and self.rgb_channels != "random"
            ):
                # e.g., random_5_12, means select 3 of channels from 5 to 12 channel index
                _lft_idx = self.rgb_channels.split("_")[1]
                _rgt_idx = self.rgb_channels.split("_")[2]
                _lft_idx = int(_lft_idx)
                _rgt_idx = int(_rgt_idx)
                assert _lft_idx < _rgt_idx, (
                    "rgb_channels must be in the range of [lft, rgt)"
                )
                assert _rgt_idx < img.shape[1], (
                    "rgb_channels must be in the range of [lft, rgt)"
                )
                _rgb_chan_select = torch.randperm(_rgt_idx - _lft_idx)[:3] + _lft_idx
                rgb_channels = torch.tensor(
                    [_rgb_chan_select[0], _rgb_chan_select[1], _rgb_chan_select[2]]
                )
                img = img[:, rgb_channels]
            elif self.rgb_channels == "mean":
                # mean three splitted bands
                c = img.shape[1]
                c_3 = c // 3
                # list(range(c))[::3]
                bands = [
                    img[:, i * c_3 : (i + 1) * c_3, :, :].mean(dim=1) for i in range(3)
                ]
                img = torch.stack(bands, dim=1)
            elif self.rgb_channels == "largest":
                bands = choose_largest_bands(img)
                img = img[:, bands]
            elif self.rgb_channels == "pca":
                assert img.ndim == 4
                img = feature_pca_torch(img, 3)
            else:
                rgb_channels = self.rgb_channels
                img = img[:, rgb_channels]

        assert img.shape[1] == 3, (
            f"img must be rgb images but got image shaped as {img.shape}"
        )

        # resize images
        _interp_kwargs = {"input": img, "mode": "bicubic", "align_corners": False}
        if tuple([self.repa_encoder.image_size] * 2) != _img_sz:
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

        # loop to get the features
        if self.dino_fixed_bs is None or img.shape[0] < self.dino_fixed_bs:
            img_feats = self._forward_features(img)
        else:
            # macro-batch-size inference
            img_feats = torch.cat(
                [
                    self._forward_features(img[i : i + self.dino_fixed_bs])
                    for i in range(0, img.shape[0], self.dino_fixed_bs)
                ],
                dim=0,
            )

        return img_feats

    def repa_loss(self, teacher_feat: Tensor, student_feat: Tensor):
        # [bs, l, c] or [bs, c, h, w]
        # 1. interpolate
        if self.feature_resample_type == "match_student":
            st_ndim = student_feat.ndim
            assert st_ndim in (3, 4), "student feature must be 1d or 2d feature"
            tgt_size = (
                student_feat.shape[-2:] if st_ndim == 4 else student_feat.shape[-2]
            )
            func_mapping_ = {
                3: interpolate_features_1d,
                4: interpolate_features_2d,
            }
            teacher_feat = func_mapping_[st_ndim](teacher_feat, tgt_size)
        else:  # student feature matched to teacher feature
            if self.c_dim_first:
                _tgt_sz = teacher_feat.shape[-2:]
                interpolate_fn = interpolate_features_2d
            else:
                _tgt_sz = teacher_feat.shape[-2]
                interpolate_fn = interpolate_features_1d

            teacher_feat = interpolate_fn(student_feat, _tgt_sz)

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
        if self.c_dim_first:
            dim = 1
            rarg_pattn = "b c h w -> b c (h w)"
        else:
            dim = -1
            rarg_pattn = "b h w c -> b (h w) c"

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
        else:
            repa_loss = am_ratio_spatial_loss(teacher_feat, student_feat, dim=dim)
        return repa_loss.mean()

    def forward(self, img: Tensor, student_feature: Tensor):
        teacher_feat = self._encode_img(img).detach()
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
        aux_feature = self._encode_img(img).detach()

        # iterpolate features
        _tgt_sz = feature.shape[-2:] if self.c_dim_first else feature.shape[-2]
        aux_feature = self.interp_feat(aux_feature, _tgt_sz)

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

    def calculate_adaptive_weight(self, nll_loss, loss, last_layer=None):
        if last_layer is not None and nll_loss is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            loss_grads = torch.autograd.grad(loss, last_layer, retain_graph=True)[0]

            g_weight = torch.norm(nll_grads) / (torch.norm(loss_grads) + 1e-4)
            g_weight = torch.clamp(g_weight, 0.0, 1e8).detach()
            g_weight = g_weight * self.weight
        else:
            g_weight = self.weight

        return g_weight

    def gram_loss(self, output_feats, target_feats, img_level=True):
        """Compute the MSE loss between the gram matrix of the input and target features.

        Args:
            output_feats: Pytorch tensor (B, N, dim) or (B*N, dim) if img_level == False
            target_feats: Pytorch tensor (B, N, dim) or (B*N, dim) if img_level == False
            img_level: bool, if true gram computed at the image level only else over the entire batch
        Returns:
            loss: scalar
        """

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

    def to_1d_seq(self, x):
        if x.ndim == 4:
            x = rearrange(x, "b c h w -> b (h w) c")
        return x

    def forward(self, img, feature, nll_loss=None, last_layer=None):
        # encode
        aux_feature = self._encode_img(img).detach()

        # iterpolate features
        _tgt_sz = feature.shape[-2:] if self.c_dim_first else feature.shape[-2]
        interp_feat_fn = (
            interpolate_features_2d if self.c_dim_first else interpolate_features_1d
        )
        aux_feature = interp_feat_fn(aux_feature, _tgt_sz)

        feature = self.projector(feature)
        assert feature.shape == aux_feature.shape, (
            "z and aux_feature must have the same shape to compute the repa loss"
        )

        # Gram loss
        feature, aux_feature = map(self.to_1d_seq, (feature, aux_feature))
        gram_loss = self.gram_loss(feature, aux_feature, img_level=self.img_level)
        g_weight = (
            1.0  # self.calculate_adaptive_weight(nll_loss, gram_loss, last_layer)
        )

        return gram_loss * g_weight


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


if __name__ == "__main__":
    test_gram_loss()
