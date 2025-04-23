import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from loguru import logger
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import Tensor


def interpolate_features_2d(x: Tensor, tgt_size: int):
    B, D, H, W = x.shape
    H_tgt = W_tgt = math.sqrt(tgt_size)

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


def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
        nn.Linear(hidden_size, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, z_dim),
    )


class REPALoss(torch.nn.Module):
    def __init__(
        self,
        c_dim_first=False,
        build_proj=False,
        img_is_neg1_1=True,
        rgb_channels: list | str = None,
    ):
        super().__init__()
        self.rgb_channels = rgb_channels
        if self.rgb_channels is not None:
            if isinstance(self.rgb_channels, (list, tuple)):
                assert len(self.rgb_channels) == 3, "rgb_channels must be 3 channels"
            elif isinstance(self.rgb_channels, str):
                assert self.rgb_channels in (
                    "random",
                ), "rgb_channels must be randomly selected"
            else:
                raise TypeError("rgb_channels must be list or tuple or str")

        self.repa_encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        self.repa_encoder.image_size = 224
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
                self.repa_encoder.embed_dim, 2048, self.repa_encoder.embed_dim
            )
            logger.warning(
                "build repa loss in loss class, remembe to optimize the projector, "
                "or match to repa dim in model forward"
            )
        else:
            self.projector = nn.Identity()

        self.img_is_neg1_1 = img_is_neg1_1
        self.register_buffer("dino_mean", IMAGENET_DEFAULT_MEAN)
        self.register_buffer("dino_std", IMAGENET_DEFAULT_STD)

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
                _rgb_chan_select = torch.randint(0, img.shape[1], (3,))
                rgb_channels = _rgb_chan_select.tolist()
            else:
                rgb_channels = self.rgb_channels
            img = img[:, rgb_channels]

        assert img.shape[1] == 3, "img must be rgb images"

        if tuple(list(self.repa_encoder.img_size) * 2) != _img_sz:
            img = F.interpolate(
                img,
                size=self.repa_encoder.img_size,
                mode="bicubic",
                align_corners=False,
            )

        img = self._norm_img_before_repa(img)
        img_feats = self.repa_encoder.forward_features(img)[
            "x_norm_patchtokens"
        ]  # (bs, 256, 768)

        return img_feats.detach()

    def _repa_loss(self, img_feat: Tensor, model_feat: Tensor):
        _tgt_sz = img_feat.shape[1]
        img_feat = self.interp_feat(img_feat, _tgt_sz)

        # flatten the model feature
        if self.c_dim_first:
            assert model_feat.ndim == 4
            # (bs, c, h, w) -> (bs, h * w, c)
            model_feat = rearrange(model_feat, "b c h w -> b (h w) c")
            model_feat = self.projector(model_feat)
        else:
            assert model_feat.ndim == 3

        # repa loss
        img_feat = norm_feature(img_feat, dim=-1)
        model_feat = norm_feature(model_feat, dim=-1)

        repa_loss = -torch.sum(img_feat * model_feat, dim=-1)
        return repa_loss.mean()

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
    loss_fn = REPALoss(c_dim_first=True, build_proj=False, img_is_neg1_1=True)
    img = torch.randn(2, 3, 224, 224)
    features = torch.randn(2, 768, 14 * 14)
    loss = loss_fn(img, features)
    print("Loss:", loss.item())
