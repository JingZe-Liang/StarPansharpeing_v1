import math
import warnings
from functools import partial
from typing import Literal, Sequence

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.state import AcceleratorState
from einops import rearrange
from loguru import logger
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import Tensor
from torch.distributed.tensor import DTensor, Shard
from torch.utils.file_baton import FileBaton

from src.utilities.config_utils import function_config_to_basic_types
from src.utilities.logging.print import log_print


def interpolate_features_2d(x: Tensor, tgt_size: tuple[int] | torch.Size):
    B, D, H, W = x.shape
    H_tgt, W_tgt = tgt_size

    if H == H_tgt and W == W_tgt:
        return x
    else:
        return F.interpolate(
            x, size=(H_tgt, W_tgt), mode="bicubic", align_corners=False
        )


def interpolate_features_1d(x, target_len):
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


def build_mlp(hidden_size, projector_dim, z_dim, is_1d=True):
    ln_cls = nn.Linear if is_1d else partial(nn.Conv2d, kernel_size=1)
    return nn.Sequential(
        ln_cls(hidden_size, projector_dim),
        nn.SiLU(),
        ln_cls(projector_dim, projector_dim),
        nn.SiLU(),
        ln_cls(projector_dim, z_dim),
    )


def next_divisble_of_y(x, y):
    return math.ceil(x / y) * y


def load_repa_encoder(dino_type: str) -> nn.Module:
    if dino_type == "timm":
        model = timm.create_model(
            "hf-hub:timm/vit_large_patch14_dinov2.lvd142m",
            pretrained=True,
            dynamic_img_size=True,
        )
    elif dino_type == "torch":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")  # type: ignore
    else:
        raise ValueError(f"Unknown dino type {dino_type}, must be 'torch' or 'timm'.")

    return model


class REPALoss(torch.nn.Module):
    @function_config_to_basic_types
    def __init__(
        self,
        c_dim_first=False,
        build_proj=False,
        img_is_neg1_1=True,
        rgb_channels: list | Literal["random"] | str | None = None,
        img_resize: Literal["dino"] | tuple | None = "dino",
        dino_fixed_bs: int | None = None,
        dino_img_size: int = 224,
        dtype: torch.dtype = torch.bfloat16,
        dino_type: str = "torch",  # [torch, timm]
    ):
        """Align the tokenizer/model feature with DINO v2 pretrained model feature.

        Args:
            c_dim_first (bool, optional): is 2d or 1d features. Defaults to False.
            build_proj (bool, optional): building projection in this loss class (not recommended). Defaults to False.
            img_is_neg1_1 (bool, optional): is image value ranging (-1, 1). Defaults to True.
            rgb_channels (list | str, optional): rgb channel indices or `random` selected from hyperspectral images. Defaults to None.
            img_resize (str | tuple | None, optional): input image will be resized into . Defaults to "dino".
            dino_fixed_bs (int | None, optional): batch size for dino forward. Defaults to None.
            dino_img_size (int, optional): dino image size. Defaults to 224.
        """
        super().__init__()
        self.rgb_channels = rgb_channels
        self.img_resize = img_resize
        if isinstance(img_resize, Sequence):
            assert img_resize[0] % 14 == 0 and img_resize[1] % 14 == 0, (
                "img_size[0] must be divisible by patch size"
            )
        self.dino_fixed_bs = dino_fixed_bs
        if self.rgb_channels is not None:
            if isinstance(self.rgb_channels, str):
                assert self.rgb_channels in (
                    "random",
                    "mean",
                ), "rgb_channels must be randomly selected"
            elif isinstance(self.rgb_channels, Sequence):
                assert len(self.rgb_channels) == 3, "rgb_channels must be 3 channels"
            else:
                raise TypeError(
                    f"rgb_channels must be list or tuple or str, but got {type(rgb_channels)}"
                )

        # encoder
        self.dino_type = dino_type
        baton = FileBaton(
            "/tmp/repa_model_reading.lock"
        )  # Use a file baton to ensure single process access

        # load repa encoder in multiprocessing context
        if baton.try_acquire():
            try:
                self.repa_encoder = load_repa_encoder(self.dino_type)
            finally:
                baton.release()
        else:
            baton.wait()
            self.repa_encoder = load_repa_encoder(self.dino_type)

        log_print("[REPA Loss]: load pretrained dino visual foundation model done")
        self.repa_encoder.image_size = dino_img_size
        self.repa_encoder = self.repa_encoder.to(dtype)

        self.repa_encoder.requires_grad_(False)
        self.repa_encoder.eval()

        self.c_dim_first = c_dim_first
        if c_dim_first:
            self.interp_feat = interpolate_features_2d
        else:
            self.interp_feat = interpolate_features_1d

        self.build_proj = build_proj
        if build_proj:
            self.projector = build_mlp(
                self.repa_encoder.embed_dim,
                2048,
                self.repa_encoder.embed_dim,
                is_1d=not c_dim_first,
            )
            logger.warning(
                "build repa loss in loss class, remember to optimize the projector, "
                "or match to repa dim in model forward"
            )
        else:
            self.projector = nn.Identity()

        self.img_is_neg1_1 = img_is_neg1_1
        self.register_buffer(
            "dino_mean", torch.as_tensor(IMAGENET_DEFAULT_MEAN, dtype=torch.float)
        )
        self.register_buffer(
            "dino_std", torch.as_tensor(IMAGENET_DEFAULT_STD, dtype=torch.float)
        )

    def __repr__(self):
        return (
            f"REPA Loss: {self.repa_encoder.__class__.__name__}\n"
            + f"img_is_neg1_1: {self.img_is_neg1_1}\n"
            + f"c_dim_first: {self.c_dim_first}\n"
            + f"img_resize: {self.img_resize}\n"
            + f"rgb_channels: {self.rgb_channels}"
        )

    def _norm_img_before_repa(self, img):
        if self.img_is_neg1_1:
            img = (img + 1) / 2

        # norm, image dim must be 4d
        assert img.ndim == 4 and img.shape[1] == 3, (
            "img dim must be 4d and have 3 channels"
        )
        img = (img - self.dino_mean[None, :, None, None]) / self.dino_std[
            None, :, None, None
        ]

        return img

    @torch.no_grad()
    def _encode_img(self, img):
        _img_sz = tuple(img.shape[-2:])
        if self.rgb_channels is not None and img.shape[1] > 3:
            assert img.shape[1] >= 3, "img must be hyperspectral images"
            if self.rgb_channels == "random":
                _rgb_chan_select = torch.randperm(img.shape[1])[:3]
                rgb_channels = _rgb_chan_select.tolist()
                img = img[:, rgb_channels]
            elif (
                not isinstance(self.rgb_channels, Sequence)
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
            else:
                rgb_channels = self.rgb_channels
                img = img[:, rgb_channels]

        assert img.shape[1] == 3, "img must be rgb images"

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

        @torch.autocast("cuda", torch.bfloat16)
        def _forward_features(img):
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
                        img, 1, reshape=True, norm=True
                    )[0]  # last layer feature
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

        # loop to get the features
        if self.dino_fixed_bs is None or img.shape[0] < self.dino_fixed_bs:
            img_feats = _forward_features(img)
        else:
            img_feats = torch.cat(
                [
                    _forward_features(img[i : i + self.dino_fixed_bs])
                    for i in range(0, img.shape[0], self.dino_fixed_bs)
                ],
                dim=0,
            )

        return img_feats

    def _repa_loss(self, img_feat: Tensor, model_feat: Tensor):
        # [bs, l, c] or [bs, c, h, w]
        # 1. interpolate: image feature -> dino feature size
        _tgt_sz = img_feat.shape[-2:] if self.c_dim_first else img_feat.shape[-2]
        model_feat = self.interp_feat(model_feat, _tgt_sz)

        # 2. dino feature -> Featup -> model feature
        # TODO: use featup to upsample the dino feature into (high-resolution) model (or says)
        # tokenizer feature

        # flatten the model feature
        model_feat = self.projector(model_feat)

        # repa loss
        if self.c_dim_first:
            dim = 1
            rarg_pattn = "b c h w -> b c (h w)"
        else:
            dim = -1
            rarg_pattn = "b h w c -> b (h w) c"

        img_feat = rearrange(img_feat, rarg_pattn)
        model_feat = rearrange(model_feat, rarg_pattn)
        img_feat = norm_feature(img_feat, dim=dim)
        model_feat = norm_feature(model_feat, dim=dim)

        assert img_feat.shape == model_feat.shape, (
            "img_feat and model_feat must have the same shape to compute the repa loss"
        )
        repa_loss = -torch.sum(img_feat * model_feat, dim=dim)

        # repa_loss = (
        #     torch.cosine_similarity(img_feat.flatten(1), model_feat.flatten(1), dim=1)
        #     / img_feat.shape[0]
        # )
        return repa_loss.mean()

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

    def forward(self, img: Tensor, features: Tensor):
        img_feat = self._encode_img(img).detach()
        repa_loss = self._repa_loss(img_feat, features)
        return repa_loss

    def named_parameters(self):
        if self.build_proj:
            return self.projector.named_parameters()
        else:
            return []

    def parameters(self):
        if self.build_proj:
            return self.projector.parameters()
        else:
            return []


class VFLoss(REPALoss):
    def __init__(
        self,
        distmat_margin: float = 0.25,
        cos_margin: float = 0.5,
        distmat_weight: float = 1.0,
        cos_weight: float = 1.0,
        c_dim_first=False,
        build_proj=False,
        img_is_neg1_1=True,
        rgb_channels: list | Literal["random"] | str | None = None,
        img_resize: Literal["dino"] | tuple | None = "dino",
        dino_fixed_bs: int | None = None,
        dino_img_size: int = 224,
        dtype: torch.dtype = torch.bfloat16,
        vf_weight: float = 1.0,
    ):
        """
        visual foundation model loss for tokenizer feature alignment.
        """

        super().__init__(
            c_dim_first,
            build_proj,
            img_is_neg1_1,
            rgb_channels,
            img_resize,
            dino_fixed_bs,
            dino_img_size,
            dtype,
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

    def forward(self, z, img, nll_loss=None, last_layer=None):
        """
        Forward pass for the visual foundation model loss.

        Args:
            z (Tensor): The main feature tensor.
            aux_feature (Tensor): The auxiliary feature tensor.
            nll_loss (Tensor, optional): The negative log-likelihood loss. Defaults to None.
            last_layer (Tensor, optional): The last layer of the encoder for gradient calculation. Defaults to None.

        Returns:
            Tensor: The computed VF loss.
        """
        # encode
        aux_feature = self._encode_img(img).detach()

        # iterpolate features
        _tgt_sz = z.shape[-2:] if self.c_dim_first else z.shape[-2]
        aux_feature = self.interp_feat(aux_feature, _tgt_sz)

        z = self.projector(z)
        assert z.shape == aux_feature.shape, (
            "z and aux_feature must have the same shape to compute the repa loss"
        )

        # loss
        vf_loss = self.vf_loss(z, aux_feature)

        # weighted
        vf_weight = self.calculate_adaptive_weight_vf(nll_loss, vf_loss, last_layer)

        return vf_loss * vf_weight


if __name__ == "__main__":
    # Example usage
    # loss_fn = REPALoss(c_dim_first=True, build_proj=False, img_is_neg1_1=True).cuda()
    # img = torch.randn(2, 3, 224, 224).cuda()
    # features = torch.randn(2, 768, 28, 28).cuda()
    # loss = loss_fn(img, features)
    # print("Loss:", loss.item())

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
        loss = loss_fn(z, img)
    print("VFLoss:", loss.item())
