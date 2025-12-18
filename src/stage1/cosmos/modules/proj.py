from functools import partial

import torch.nn as nn
from einops import rearrange
from timm.layers.create_norm_act import get_norm_act_layer
from timm.layers.mlp import ConvMlp, Mlp


def build_mlp_legacy_(hidden_size, projector_dim, z_dim, is_1d=False):
    ln_cls = nn.Linear if is_1d else partial(nn.Conv2d, kernel_size=1)
    return nn.Sequential(
        ln_cls(hidden_size, projector_dim),
        nn.SiLU(),
        ln_cls(projector_dim, projector_dim),
        nn.SiLU(),
        ln_cls(projector_dim, z_dim),
    )


def build_mlp_norm_first_(
    hidden_size: int,
    projector_dim: int,
    z_dim: int,
    is_1d=False,
    norm_type: str = "layernorm",
    act_type: str = "gelu",
):
    linear_cls = nn.Linear if is_1d else partial(nn.Conv2d, kernel_size=1)
    if not is_1d:
        norm_type = norm_type + "2d"
    norm_act = get_norm_act_layer(norm_type, act_layer=act_type)
    assert norm_act is not None, "norm_act should not be None"
    # two layers Mlp
    mlp = nn.Sequential(
        norm_act(hidden_size),
        linear_cls(hidden_size, projector_dim),
        norm_act(projector_dim),
        linear_cls(projector_dim, z_dim),
    )
    return mlp


def build_mlp_norm_first_timm_(hidden_size, projector_dim, z_dim, is_1d=False):
    mlp_cls = Mlp if is_1d else ConvMlp
    mlp = mlp_cls(
        in_features=hidden_size,
        hidden_features=projector_dim,
        out_features=z_dim,
        act_layer=nn.GELU,
    )
    return mlp


class MlpConvForce2D(nn.Module):
    """
    Force the input to 2D feature that reduce the spatial information loss
    Taken from iREPA paper.
    """

    def __init__(self, hidden_size: int, projector_dim: int, z_dim: int, hw: tuple | None = None):
        super().__init__()
        self.hw = hw

        self.mlp = ConvMlp(
            hidden_size,
            hidden_features=projector_dim,
            out_features=z_dim,
            act_layer=nn.GELU,
        )

    def forward(self, x, hw: tuple | None = None):
        assert x.ndim in (3, 4), f"Input must be 3D or 4D tensor, but got {x.ndim}D tensor."
        img_hw = hw or self.hw
        if img_hw is None and x.ndim == 3:
            img_hw = (int(x.shape[1] ** 0.5), int(x.shape[1] ** 0.5))
        if x.ndim == 3:  # blc
            x = rearrange(x, "b (h w) c -> b c h w", h=img_hw[0], w=img_hw[1])
        return self.mlp(x)


# Interface


def build_mlp(
    hidden_size,
    projector_dim,
    z_dim,
    is_1d=False,
    proj_type: str = "norm_first",  # 'legacy', 'norm_first', 'norm_first_timm'
):
    """Mlp for REPA projector"""
    if proj_type == "legacy":
        return build_mlp_legacy_(hidden_size, projector_dim, z_dim, is_1d=is_1d)
    elif proj_type == "norm_first":
        return build_mlp_norm_first_(hidden_size, projector_dim, z_dim, is_1d=is_1d)
    elif proj_type == "norm_first_timm":
        return build_mlp_norm_first_timm_(hidden_size, projector_dim, z_dim, is_1d=is_1d)
    elif proj_type == "norm_first_force_conv":
        # taken from irepa paper
        # force the projector is a convolution
        return MlpConvForce2D(hidden_size, projector_dim, z_dim)
    else:
        raise ValueError(f"Unknown proj_type: {proj_type}")
