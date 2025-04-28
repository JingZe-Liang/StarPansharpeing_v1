import math
import warnings
from functools import partial
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.state import AcceleratorState, PartialState
from einops import rearrange
from loguru import logger
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import Tensor
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.device_mesh import DeviceMesh, init_device_mesh

warnings.filterwarnings(
    "once",
    message="[Repa Resize]: image not resize into dino pretrained size",
    append=True,
)


def interpolate_features_2d(x: Tensor, tgt_size: tuple[int]):
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


class REPALoss(torch.nn.Module):
    def __init__(
        self,
        c_dim_first=False,
        build_proj=False,
        img_is_neg1_1=True,
        rgb_channels: list | Literal["random"] | str = None,
        img_resize: Literal["dino"] | tuple | None = "dino",
        dino_fixed_bs: int | None = None,
        dino_img_size: int = 224,
        dtype: torch.dtype = torch.bfloat16,
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
        if isinstance(img_resize, (tuple, list)):
            assert (
                img_resize[0] % 14 == 0 and img_resize[1] % 14 == 0
            ), "img_size[0] must be divisible by patch size"
        self.dino_fixed_bs = dino_fixed_bs
        if self.rgb_channels is not None:
            if isinstance(self.rgb_channels, (list, tuple)):
                assert len(self.rgb_channels) == 3, "rgb_channels must be 3 channels"
            elif isinstance(self.rgb_channels, str):
                assert self.rgb_channels in (
                    "random",
                ), "rgb_channels must be randomly selected"
            else:
                raise TypeError("rgb_channels must be list or tuple or str")

        # encoder
        self.repa_encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        self.repa_encoder.image_size = dino_img_size
        self.repa_encoder = self.repa_encoder.to(dtype)

        for param in self.repa_encoder.parameters():
            param.requires_grad = False
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
                "build repa loss in loss class, remembe to optimize the projector, "
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
        assert (
            img.ndim == 4 and img.shape[1] == 3
        ), "img dim must be 4d and have 3 channels"
        img = (img - self.dino_mean[None, :, None, None]) / self.dino_std[
            None, :, None, None
        ]

        return img

    @torch.no_grad()
    def _encode_img(self, img):
        _img_sz = tuple(img.shape[-2:])
        if self.rgb_channels is not None:
            assert img.shape[1] >= 3, "img must be hyperspectral images"
            if self.rgb_channels == "random":
                _rgb_chan_select = torch.randperm(img.shape[1])[:3]
                rgb_channels = _rgb_chan_select.tolist()
            elif (
                self.rgb_channels.startswith("random") and self.rgb_channels != "random"
            ):
                # e.g., random_5_12, means select 3 of channels from 5 to 12 channel index
                _lft_idx = self.rgb_channels.split("_")[1]
                _rgt_idx = self.rgb_channels.split("_")[2]

                _lft_idx = int(_lft_idx)
                _rgt_idx = int(_rgt_idx)

                assert (
                    _lft_idx < _rgt_idx
                ), "rgb_channels must be in the range of [lft, rgt)"
                assert (
                    _rgt_idx < img.shape[1]
                ), "rgb_channels must be in the range of [lft, rgt)"
                _rgb_chan_select = torch.randperm(_rgt_idx - _lft_idx)[:3] + _lft_idx
                rgb_channels = torch.tensor(
                    [_rgb_chan_select[0], _rgb_chan_select[1], _rgb_chan_select[2]]
                )
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
                warnings.warn(
                    "[Repa Resize]: image not resize into dino pretrained size"
                )
            img = F.interpolate(**_interp_kwargs)

        img = self._norm_img_before_repa(img)

        @torch.autocast("cuda", torch.bfloat16)
        def _forward_featurs(img):
            # from torch.distributed.tensor.device_mesh import init_device_mesh, DeviceMesh
            # from torch.distributed.tensor import DTensor
            img = self._to_dtensor(img)

            if not self.c_dim_first:  # distill for vit 1d features
                img_feats = self.repa_encoder.forward_features(img)[
                    "x_norm_patchtokens"
                ]  # (bs, 256, 768)
            else:  # distill for cnn 2d features
                img_feats = self.repa_encoder.get_intermediate_layers(
                    img, 1, reshape=True, norm=True
                )[0]  # last layer feature

            if isinstance(img_feats, DTensor):
                img_feats = img_feats.full_tensor()

            return img_feats

        # loop to get the features
        if self.dino_fixed_bs is None or img.shape[0] < self.dino_fixed_bs:
            img_feats = _forward_featurs(img)
        else:
            img_feats = torch.cat(
                [
                    _forward_featurs(img[i : i + self.dino_fixed_bs])
                    for i in range(0, img.shape[0], self.dino_fixed_bs)
                ],
                dim=0,
            )

        return img_feats.detach()

    def _repa_loss(self, img_feat: Tensor, model_feat: Tensor):
        # [bs, l, c] or [bs, c, h, w]
        # 1. interpolate: image feature -> dino feature size
        _tgt_sz = img_feat.shape[-2:] if self.c_dim_first else img_feat.shape[-2]
        model_feat = self.interp_feat(model_feat, _tgt_sz)

        # 2. dino feature -> Featup -> model feature
        # TODO: use featup to upsampel the dino feature into (high-resolution) model (or says)
        # tokenizer feature

        # flatten the model feature
        model_feat = self.projector(model_feat)

        # repa loss
        dim = 1 if self.c_dim_first else -1
        img_feat = norm_feature(img_feat, dim=dim)
        model_feat = norm_feature(model_feat, dim=dim)

        assert (
            img_feat.shape == model_feat.shape
        ), "img_feat and model_feat must have the same shape to compute the repa loss"
        repa_loss = -torch.sum(img_feat * model_feat, dim=dim)
        # repa_loss = (
        #     torch.cosine_similarity(img_feat.flatten(1), model_feat.flatten(1), dim=1)
        #     / img_feat.shape[0]
        # )
        return repa_loss.mean()

    def _to_dtensor(self, img: Tensor):
        if AcceleratorState().is_fsdp2:
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


if __name__ == "__main__":
    # Example usage
    loss_fn = REPALoss(c_dim_first=True, build_proj=False, img_is_neg1_1=True).cuda()
    img = torch.randn(2, 3, 224, 224).cuda()
    features = torch.randn(2, 768, 28, 28).cuda()
    loss = loss_fn(img, features)
    print("Loss:", loss.item())
