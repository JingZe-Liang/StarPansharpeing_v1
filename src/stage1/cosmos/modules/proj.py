from functools import partial

import torch.nn as nn
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


# Interface


def build_mlp(
    hidden_size,
    projector_dim,
    z_dim,
    is_1d=False,
    proj_type: str = "norm_first",  # 'legacy', 'norm_first', 'norm_first_timm'
):
    if proj_type == "legacy":
        return build_mlp_legacy_(hidden_size, projector_dim, z_dim, is_1d=is_1d)
    elif proj_type == "norm_first":
        return build_mlp_norm_first_(hidden_size, projector_dim, z_dim, is_1d=is_1d)
    elif proj_type == "norm_first_timm":
        return build_mlp_norm_first_timm_(
            hidden_size, projector_dim, z_dim, is_1d=is_1d
        )
    else:
        raise ValueError(f"Unknown proj_type: {proj_type}")
