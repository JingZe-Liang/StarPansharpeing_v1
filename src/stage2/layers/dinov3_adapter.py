# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

# Modified by Zihan Cao
# Date: 2025.09.27
# UESTC. All Rights Reserved.

import sys
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from einops import rearrange
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import create_norm_act_layer, create_norm_layer
from torchvision.transforms import Normalize

sys.path.append("src/stage1/utilities/losses/dinov3")
from dinov3.eval.segmentation.models.utils.ms_deform_attn import (  # type: ignore
    MSDeformAttn,
)


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
        )
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    return reference_points


def deform_inputs(x, patch_size):
    bs, c, h, w = x.shape
    spatial_shapes = torch.as_tensor(
        [(h // 8, w // 8), (h // 16, w // 16), (h // 32, w // 32)],
        dtype=torch.long,
        device=x.device,
    )
    level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // patch_size, w // patch_size)], x.device)
    deform_inputs1 = [reference_points, spatial_shapes, level_start_index]

    spatial_shapes = torch.as_tensor([(h // patch_size, w // patch_size)], dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 8, w // 8), (h // 16, w // 16), (h // 32, w // 32)], x.device)
    deform_inputs2 = [reference_points, spatial_shapes, level_start_index]

    return deform_inputs1, deform_inputs2


def deform_inputs_v2(x, patch_size: int, downsample_ratios: list[int]):
    """
    Generalized version of deform_inputs that takes a list of downsample ratios.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (B, C, H, W)
    patch_size : int
        Patch size for the backbone
    downsample_ratios : list[int]
        List of downsample ratios for different feature levels

    Returns
    -------
    tuple
        (deform_inputs1, deform_inputs2) where:
        - deform_inputs1: for attention query reference points
        - deform_inputs2: for multi-scale feature attention
    """
    bs, c, h, w = x.shape

    # Create spatial shapes based on downsample ratios
    spatial_shapes_list = [(h // ratio, w // ratio) for ratio in downsample_ratios]

    # deform_inputs1: reference points for patch-level features
    reference_points1 = get_reference_points([(h // patch_size, w // patch_size)], x.device)
    spatial_shapes = torch.as_tensor(spatial_shapes_list, dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    deform_inputs1 = [reference_points1, spatial_shapes, level_start_index]

    # deform_inputs2: reference points for multi-scale features
    reference_points2 = get_reference_points(spatial_shapes_list, x.device)
    spatial_shapes = torch.as_tensor([(h // patch_size, w // patch_size)], dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    deform_inputs2 = [reference_points2, spatial_shapes, level_start_index]

    return deform_inputs1, deform_inputs2


class ConvFFN(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        dw_ratios=[2, 1, 0.5],
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        # self.dwconv = DWConv(hidden_features)
        self.dwconv = SharedDWConv(hidden_features, dw_ratios)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        # 256 / 8 = 32
        # 256 / 16 = 16
        # 256 / 32 = 8
        # tokens: 32 ** 2+ 16 ** 2 + 8 ** 2 = 1344
        # N // 21 = 64, 16 * 64 = 32 * 32, 8 * 64 = 16 * 16, 4 * 64 = 8 * 8
        # into different input 2D tokens, shared with one conv.
        n = N // 21
        x1 = x[:, 0 : 16 * n, :].transpose(1, 2).view(B, C, H * 2, W * 2).contiguous()
        x2 = x[:, 16 * n : 20 * n, :].transpose(1, 2).view(B, C, H, W).contiguous()
        x3 = x[:, 20 * n :, :].transpose(1, 2).view(B, C, H // 2, W // 2).contiguous()
        x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
        x2 = self.dwconv(x2).flatten(2).transpose(1, 2)
        x3 = self.dwconv(x3).flatten(2).transpose(1, 2)
        x = torch.cat([x1, x2, x3], dim=1)
        assert x.shape[1] == N
        return x


class SharedDWConv(DWConv):
    """
    Re-implementation of DWConv with shared weights.
    Avoid the hard-coded factor.
    """

    def __init__(self, dim, ratios=[2, 1, 1 / 2]):
        super().__init__(dim)
        self.ratios = ratios
        self._checked = False

    def _to_2d_img(self, x, idx, H, W, st: int = 0):
        r = self.ratios[idx]
        h, w = int(H * r), int(W * r)
        l = h * w
        ed = st + l
        x_r = x[:, st:ed]
        x_r = rearrange(x_r, "b (h w) c -> b c h w", h=h, w=w)
        return x_r, l

    def _to_1d_img(self, x):
        return rearrange(x, "b c h w -> b (h w) c")

    def _assertion(self, x, H, W):
        len_exp_rs = [int(H * r * W * r) for r in self.ratios]
        len_expected = sum(len_exp_rs)
        assert x.shape[1] == len_expected, (
            f"Expected {len_expected} length, got {x.shape[1]}, levels of image size "
            f"should be {len_exp_rs} for ratios {self.ratios}"
        )
        self._checked = True

    def forward(self, x, H, W):
        if not self._checked:
            self._assertion(x, H, W)

        N = x.shape[1]
        st = 0
        xs = []
        for i in range(len(self.ratios)):
            x_2d, length = self._to_2d_img(x, i, H, W, st)
            x_2d = self.dwconv(x_2d)
            xs.append(self._to_1d_img(x_2d))
            st = st + length
        x = torch.cat(xs, dim=1)
        assert x.shape[1] == N, f"{x.shape[1]} != {N}"
        return x


class Extractor(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=6,
        n_points=4,
        n_levels=1,
        deform_ratio=1.0,
        with_cffn=True,
        cffn_ratio=0.25,
        drop=0.0,
        drop_path=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        dw_ratios=[2, 1, 0.5],
        with_cp=False,
    ):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(
            d_model=dim,
            n_levels=n_levels,
            n_heads=num_heads,
            n_points=n_points,
            ratio=deform_ratio,
        )
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn = ConvFFN(
                in_features=dim,
                hidden_features=int(dim * cffn_ratio),
                drop=drop,
                dw_ratios=dw_ratios,
            )
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W):
        def _inner_forward(query, feat):
            attn = self.attn(
                self.query_norm(query),
                reference_points,
                self.feat_norm(feat),
                spatial_shapes,
                level_start_index,
                None,
            )
            query = query + attn

            if self.with_cffn:
                query = query + self.drop_path(self.ffn(self.ffn_norm(query), H, W))
            return query

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)

        return query


class InteractionBlockWithCls(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=6,
        n_points=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        drop=0.0,
        drop_path=0.0,
        with_cffn=True,
        cffn_ratio=0.25,
        init_values=0.0,
        dw_ratios=[2, 1, 0.5],
        deform_ratio=1.0,
        extra_extractor=False,
        with_cp=False,
    ):
        super().__init__()
        self.extractor = Extractor(
            dim=dim,
            n_levels=1,
            num_heads=num_heads,
            n_points=n_points,
            norm_layer=norm_layer,
            deform_ratio=deform_ratio,
            with_cffn=with_cffn,
            cffn_ratio=cffn_ratio,
            drop=drop,
            drop_path=drop_path,
            with_cp=with_cp,
            dw_ratios=dw_ratios,
        )
        if extra_extractor:
            self.extra_extractors = nn.Sequential(
                *[
                    Extractor(
                        dim=dim,
                        num_heads=num_heads,
                        n_points=n_points,
                        norm_layer=norm_layer,
                        with_cffn=with_cffn,
                        cffn_ratio=cffn_ratio,
                        deform_ratio=deform_ratio,
                        drop=drop,
                        drop_path=drop_path,
                        with_cp=with_cp,
                        dw_ratios=dw_ratios,
                    )
                    for _ in range(2)
                ]
            )
        else:
            self.extra_extractors = None

    def forward(self, x, c, cls, deform_inputs1, deform_inputs2, H_c, W_c, H_toks, W_toks):
        c = self.extractor(
            query=c,
            reference_points=deform_inputs2[0],
            feat=x,
            spatial_shapes=deform_inputs2[1],
            level_start_index=deform_inputs2[2],
            H=H_c,
            W=W_c,
        )
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(
                    query=c,
                    reference_points=deform_inputs2[0],
                    feat=x,
                    spatial_shapes=deform_inputs2[1],
                    level_start_index=deform_inputs2[2],
                    H=H_c,
                    W=W_c,
                )
        return x, c, cls


class SpatialPriorModule(nn.Module):
    def __init__(self, inplanes=64, embed_dim=384, with_cp=False, final_downsample=32):
        """
        Spatial Prior Module for generating multi-scale features.

        Parameters
        ----------
        inplanes : int
            Number of input planes
        embed_dim : int
            Embedding dimension
        with_cp : bool
            Whether to use checkpointing
        final_downsample : int
            Final downsampling ratio (16 or 32)
        """
        super().__init__()
        self.with_cp = with_cp
        self.final_downsample = final_downsample

        self.stem = nn.Sequential(
            *[
                nn.Conv2d(
                    3,
                    inplanes,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.SyncBatchNorm(inplanes),
                nn.ReLU(inplace=True),
                nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.SyncBatchNorm(inplanes),
                nn.ReLU(inplace=True),
                nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.SyncBatchNorm(inplanes),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ]
        )
        self.conv2 = nn.Sequential(
            *[
                nn.Conv2d(
                    inplanes,
                    2 * inplanes,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.SyncBatchNorm(2 * inplanes),
                nn.ReLU(inplace=True),
            ]
        )
        self.conv3 = nn.Sequential(
            *[
                nn.Conv2d(
                    2 * inplanes,
                    4 * inplanes,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.SyncBatchNorm(4 * inplanes),
                nn.ReLU(inplace=True),
            ]
        )

        # Conditionally create conv4 based on final_downsample
        if final_downsample == 32:
            self.conv4 = nn.Sequential(
                *[
                    nn.Conv2d(
                        4 * inplanes,
                        4 * inplanes,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.SyncBatchNorm(4 * inplanes),
                    nn.ReLU(inplace=True),
                ]
            )
        else:
            # For 16x downsampling, conv4 is just identity
            self.conv4 = nn.Sequential(
                *[
                    nn.Conv2d(
                        4 * inplanes,
                        4 * inplanes,
                        kernel_size=3,
                        stride=1,  # stride=1
                        padding=1,
                        bias=False,
                    ),
                    nn.SyncBatchNorm(4 * inplanes),
                    nn.ReLU(inplace=True),
                ]
            )

        self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        def _inner_forward(x):
            c1 = self.stem(x)
            c2 = self.conv2(c1)
            c3 = self.conv3(c2)
            c4 = self.conv4(c3)
            c1 = self.fc1(c1)
            c2 = self.fc2(c2)
            c3 = self.fc3(c3)
            c4 = self.fc4(c4)

            bs, dim, _, _ = c1.shape
            # c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s
            c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
            c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
            c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s

            return c1, c2, c3, c4

        if self.with_cp and self.training:
            outs = cp.checkpoint(_inner_forward, x)
        else:
            outs = _inner_forward(x)
        return outs


class DINOv3_Adapter(nn.Module):
    def __init__(
        self,
        backbone,
        interaction_indexes=[9, 19, 29, 39],
        pretrain_size=512,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=16,
        drop_path_rate=0.3,
        init_values=0.0,
        with_cffn=True,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        add_vit_feature=True,
        use_extra_extractor=True,
        dw_ratios=[2, 1, 0.5],
        with_cp=True,
        use_bn=True,
        freeze_backbone=True,
    ):
        super(DINOv3_Adapter, self).__init__()

        embed_dim = self._setup_backbone(
            backbone,
            pretrain_size,
            interaction_indexes,
            add_vit_feature,
            freeze_backbone,
        )

        self._setup_interactions(
            embed_dim=embed_dim,
            interaction_indexes=self.interaction_indexes,
            pretrain_size=pretrain_size,
            conv_inplane=conv_inplane,
            n_points=n_points,
            deform_num_heads=deform_num_heads,
            drop_path_rate=drop_path_rate,
            init_values=init_values,
            with_cffn=with_cffn,
            cffn_ratio=cffn_ratio,
            deform_ratio=deform_ratio,
            add_vit_feature=add_vit_feature,
            use_extra_extractor=use_extra_extractor,
            dw_ratios=dw_ratios,
            with_cp=with_cp,
            use_bn=use_bn,
        )

    def _setup_backbone(
        self,
        backbone,
        pretrain_size: int,
        interaction_indexes: list[int],
        add_vit_feature=True,
        freeze_backbone=True,
    ):
        self.backbone = backbone
        # Important: we freeze the backbone
        if freeze_backbone:
            self.backbone.requires_grad_(False)

        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        embed_dim = self.backbone.embed_dim
        self.patch_size = self.backbone.patch_size
        print("embed dim", embed_dim)
        print("interaction_indexes", self.interaction_indexes)
        print("patch_size", self.patch_size)

        return embed_dim

    def _setup_interactions(
        self,
        embed_dim: int,
        interaction_indexes=[9, 19, 29, 39],
        pretrain_size=512,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=16,
        drop_path_rate=0.3,
        init_values=0.0,
        with_cffn=True,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        add_vit_feature=True,
        use_extra_extractor=True,
        dw_ratios=[2, 1, 0.5],
        with_cp=True,
        use_bn=True,
        inp_mean_std=(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ):
        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=embed_dim, with_cp=False)
        self.interactions = nn.Sequential(
            *[
                InteractionBlockWithCls(
                    dim=embed_dim,
                    num_heads=deform_num_heads,
                    n_points=n_points,
                    init_values=init_values,
                    drop_path=drop_path_rate,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    with_cffn=with_cffn,
                    cffn_ratio=cffn_ratio,
                    deform_ratio=deform_ratio,
                    dw_ratios=dw_ratios,
                    extra_extractor=(
                        (True if i == len(self.interaction_indexes) - 1 else False) and use_extra_extractor
                    ),
                    with_cp=with_cp,
                )
                for i in range(len(self.interaction_indexes))
            ]
        )
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.norm1 = nn.SyncBatchNorm(embed_dim) if use_bn else create_norm_layer("layernorm2d", embed_dim)
        self.norm2 = nn.SyncBatchNorm(embed_dim) if use_bn else create_norm_layer("layernorm2d", embed_dim)
        self.norm3 = nn.SyncBatchNorm(embed_dim) if use_bn else create_norm_layer("layernorm2d", embed_dim)
        self.norm4 = nn.SyncBatchNorm(embed_dim) if use_bn else create_norm_layer("layernorm2d", embed_dim)

        self._input_norm = Normalize(*inp_mean_std)

        self.init_weights()

    def init_weights(self):
        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        torch.nn.init.normal_(self.level_embed)

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            # fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            # fan_out //= m.groups
            # m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            from timm.layers.weight_init import lecun_normal_

            lecun_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1,
            self.pretrain_size[0] // self.patch_size,
            self.pretrain_size[1] // self.patch_size,
            -1,
        ).permute(0, 3, 1, 2)
        pos_embed = (
            F.interpolate(pos_embed, size=(H, W), mode="bicubic", align_corners=False)
            .reshape(1, -1, H * W)
            .permute(0, 2, 1)
        )
        return pos_embed

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def _forward_backbone_intermediate_features(self, x):
        with torch.autocast("cuda", torch.bfloat16):
            with torch.no_grad():
                all_layers = self.backbone.get_intermediate_layers(
                    x, n=self.interaction_indexes, return_class_token=True
                )

        return all_layers

    def forward(self, x):
        x = self._input_norm(x)  # for input is 0-1
        deform_inputs1, deform_inputs2 = deform_inputs(x, self.patch_size)

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)

        c = torch.cat([c2, c3, c4], dim=1)

        # Code for matching with oss
        H_c, W_c = x.shape[2] // 16, x.shape[3] // 16
        H_toks, W_toks = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size
        bs, C, h, w = x.shape

        all_layers = self._forward_backbone_intermediate_features(x)
        others = None
        if isinstance(all_layers, tuple):
            all_layers, others = all_layers
        bs, _, dim = all_layers[0][0].shape  # [x, cls] per layer out

        outs = list()
        for i, layer in enumerate(self.interactions):
            x, cls = all_layers[i]
            # print(f"{x.shape=}, {cls.shape=}, ")
            _, c, _ = layer(
                x,
                c,
                cls,
                deform_inputs1,
                deform_inputs2,
                H_c,
                W_c,
                H_toks,
                W_toks,
            )
            outs.append(x.transpose(1, 2).view(bs, dim, H_toks, W_toks).contiguous())

        # Split & Reshape
        c2 = c[:, 0 : c2.size(1), :]
        c3 = c[:, c2.size(1) : c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1) :, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H_c * 2, W_c * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H_c, W_c).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H_c // 2, W_c // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs

            x1 = F.interpolate(x1, size=(4 * H_c, 4 * W_c), mode="bilinear", align_corners=False)
            x2 = F.interpolate(x2, size=(2 * H_c, 2 * W_c), mode="bilinear", align_corners=False)
            x3 = F.interpolate(x3, size=(1 * H_c, 1 * W_c), mode="bilinear", align_corners=False)
            x4 = F.interpolate(x4, size=(H_c // 2, W_c // 2), mode="bilinear", align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)

        ret = {"1": f1, "2": f2, "3": f3, "4": f4}
        if others is None:
            return ret
        return ret, others


class UpsampleBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            create_norm_act_layer("layernorm2d", dim, "gelu"),
        )

    def forward(self, x_main, x_low):
        x_low = self.upsample(x_low)
        return x_main + x_low


class DownsampleBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(dim * 4, dim, 3, 1, 1),
            create_norm_act_layer("layernorm2d", dim, "gelu"),
        )

    def forward(self, x_main, x_high):
        x_high = self.downsample(x_high)
        return x_main + x_high


class DINOv3_Adapter_MS_Down(DINOv3_Adapter):
    def __init__(
        self,
        backbone,
        interaction_indexes=[9, 19, 29, 39],
        pretrain_size=512,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=16,
        drop_path_rate=0.3,
        init_values=0.0,
        with_cffn=True,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        add_vit_feature=True,
        dw_ratios=[2, 1, 1],
        use_extra_extractor=True,
        with_cp=True,
        use_bn=True,
    ):
        # Call parent init first
        super().__init__(
            backbone=backbone,
            interaction_indexes=interaction_indexes,
            pretrain_size=pretrain_size,
            conv_inplane=conv_inplane,
            n_points=n_points,
            deform_num_heads=deform_num_heads,
            drop_path_rate=drop_path_rate,
            init_values=init_values,
            with_cffn=with_cffn,
            cffn_ratio=cffn_ratio,
            deform_ratio=deform_ratio,
            add_vit_feature=add_vit_feature,
            use_extra_extractor=use_extra_extractor,
            dw_ratios=dw_ratios,  # the last layer scale is only 16, c4 = c3
            with_cp=with_cp,
            use_bn=use_bn,
        )

        # Replace SPM with 16x downsampling version
        self.spm = SpatialPriorModule(
            inplanes=conv_inplane,
            embed_dim=self.backbone.embed_dim,
            with_cp=False,
            final_downsample=16,  # Use 16x downsampling instead of 32x
        )

        delattr(self, "up")
        self.downs = nn.ModuleList([DownsampleBlock(self.backbone.embed_dim) for _ in range(2)])

    def forward(self, x, ret_lvls=True):
        """
        Forward pass for DINOv3_Adapter_MS_Down with progressive feature fusion.

        This method implements a reversed feature fusion process where shallow features
        (c1, c2, c3) are progressively fused into deeper features (c4), resulting in
        enhanced deep feature representation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W)

        Returns
        -------
        dict
            Dictionary containing only the final enhanced c4 feature
        """
        x = self._input_norm(x)  # for input is 0-1
        # Use deform_inputs_v2 with 16x downsample ratios for MS_Down variant
        deform_inputs1, deform_inputs2 = deform_inputs_v2(x, self.patch_size, downsample_ratios=[8, 16, 16])

        # SPM forward - generate multi-scale features
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)

        c = torch.cat([c2, c3, c4], dim=1)

        # Code for matching with oss
        H_c, W_c = x.shape[2] // 16, x.shape[3] // 16
        H_toks, W_toks = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size
        bs, _, _, _ = x.shape

        with torch.autocast("cuda", torch.bfloat16):
            with torch.no_grad():
                all_layers = self.backbone.get_intermediate_layers(
                    x, n=self.interaction_indexes, return_class_token=True
                )

        bs, _, dim = all_layers[0][0].shape  # [x, cls] per layer out

        outs = list()
        for i, layer in enumerate(self.interactions):
            x_layer, cls = all_layers[i]
            _, c, _ = layer(
                x_layer,
                c,
                cls,
                deform_inputs1,
                deform_inputs2,
                H_c,
                W_c,
                H_toks,
                W_toks,
            )
            outs.append(x_layer.transpose(1, 2).view(bs, dim, H_toks, W_toks).contiguous())

        # Split & Reshape distilled features
        c2 = c[:, 0 : c2.size(1), :]
        c3 = c[:, c2.size(1) : c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1) :, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H_c * 2, W_c * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H_c, W_c).contiguous()
        # For 16x downsampling, c4 has the same spatial size as c3
        c4 = c4.transpose(1, 2).view(bs, dim, H_c, W_c).contiguous()

        # Progressive feature fusion: c1 -> c2 -> c3 -> c4
        # First downsample c1 to match c2 resolution and fuse
        c2 = self.downs[0](c2, c1)

        # Then downsample c2 to match c3 resolution and fuse
        c3 = self.downs[1](c3, c2)

        # For 16x downsampling, c3 and c4 have the same resolution, so direct fusion
        c4_enhanced = c4 + c3

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs

            # Interpolate ViT features to match corresponding scales
            x1 = F.interpolate(x1, size=(4 * H_c, 4 * W_c), mode="bilinear", align_corners=False)
            x2 = F.interpolate(x2, size=(2 * H_c, 2 * W_c), mode="bilinear", align_corners=False)
            x3 = F.interpolate(x3, size=(1 * H_c, 1 * W_c), mode="bilinear", align_corners=False)
            # For 16x downsampling, x4 should match c4 resolution (same as c3)
            x4 = F.interpolate(x4, size=(H_c, W_c), mode="bilinear", align_corners=False)

            # Add ViT features to enhanced features
            c2 = c2 + x2
            c3 = c3 + x3
            c4 = c4 + x4

        # Final Norm - only for c4
        f4 = self.norm4(c4_enhanced)

        return {"4": f4} if ret_lvls else f4


# * --- Test --- #


def test_ms_adapter():
    """Test DINOv3_Adapter_MS_Down functionality"""
    import sys
    from pathlib import Path

    import torch

    # Add path for loading dinov3 model
    from src.stage1.utilities.losses.repa.repa_feature_loss import (
        load_repa_dino_v3_model,
    )

    print("=== Testing DINOv3_Adapter_MS_Down ===")

    try:
        # Load DINOv3 backbone (using small model for testing)
        print("Loading DINOv3 backbone...")
        backbone = load_repa_dino_v3_model(
            weight_path=None,
            model_name="dinov3_vits16",
            pretrained_on="web",
            compile=False,
        )
        print(f"Backbone loaded successfully: {type(backbone)}")
        device = "cuda"

        # Create adapter instance
        print("Creating DINOv3_Adapter_MS_Down...")
        adapter = DINOv3_Adapter_MS_Down(
            backbone=backbone,
            interaction_indexes=[2, 5, 8, 11],  # For vit-small
            pretrain_size=512,
            conv_inplane=64,
            n_points=4,
            deform_num_heads=16,
            drop_path_rate=0.1,
            init_values=0.0,
            with_cffn=True,
            cffn_ratio=0.25,
            deform_ratio=0.5,
            add_vit_feature=True,
            use_extra_extractor=True,
            with_cp=True,
            use_bn=True,
        ).to(device)
        print("Adapter created successfully")

        # Parameters
        # print("Testing parameters...")
        # from fvcore.nn import FlopCountAnalysis, flop_count_table

        # x = torch.randn(1, 3, 224, 224, device=device)
        # print(flop_count_table(FlopCountAnalysis(adapter, x)))

        # Test forward pass
        print("Testing forward pass...")
        batch_size = 1
        input_size = 256
        x = torch.randn(batch_size, 3, input_size, input_size, device=device)

        output = adapter(x)
        print(f"Output keys: {output.keys()}")
        print(f"Output shape: {output['4'].shape}")

        # Expected output size should be 16x downsampled
        expected_size = input_size // 16
        actual_shape = output["4"].shape

        assert actual_shape == (
            batch_size,
            backbone.embed_dim,
            expected_size,
            expected_size,
        ), f"Expected shape ({batch_size}, {backbone.embed_dim}, {expected_size}, {expected_size}), got {actual_shape}"

        # Check if output is valid
        assert not torch.isnan(output["4"]).any(), "Output contains NaN values"
        assert torch.isfinite(output["4"]).all(), "Output contains infinite values"

        print("✓ Forward pass test passed")
        print(f"✓ Output shape: {actual_shape} (16x downsampling from {input_size}x{input_size})")

        # Test backward pass
        print("Testing backward pass...")
        loss = output["4"].mean()
        loss.backward()

        # Check if gradients are computed
        grad_count = 0
        for name, param in adapter.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_count += 1

        print(f"✓ Backward pass test passed ({grad_count} parameters with gradients)")

        # Test with different input sizes
        print("Testing with different input sizes...")
        test_sizes = [128, 256, 512]

        for size in test_sizes:
            x_test = torch.randn(1, 3, size, size, device=device)
            output_test = adapter(x_test)
            expected_size = size // 16
            actual_shape = output_test["4"].shape

            assert actual_shape[2] == expected_size and actual_shape[3] == expected_size, (
                f"Size {size}x{size}: expected {expected_size}x{expected_size}, got {actual_shape[2]}x{actual_shape[3]}"
            )

            print(f"✓ Input {size}x{size} -> Output {actual_shape[2]}x{actual_shape[3]} (16x downsampling)")

        print("✓ All size tests passed")

        # Test with add_vit_feature=False
        print("Testing with add_vit_feature=False...")
        adapter_no_vit = DINOv3_Adapter_MS_Down(
            backbone=backbone,
            interaction_indexes=[2, 5, 8, 11],
            add_vit_feature=False,
        ).to(device)

        output_no_vit = adapter_no_vit(x)
        assert output_no_vit["4"].shape == output["4"].shape, (
            "Output shape should be the same with/without ViT features"
        )

        print("✓ add_vit_feature=False test passed")

        print("\n🎉 All tests passed! DINOv3_Adapter_MS_Down is working correctly.")

        return True

    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def test_dinov3_adapter():
    """Test DINOv3_Adapter functionality"""
    import sys
    from pathlib import Path

    import torch

    print("=== Testing DINOv3_Adapter ===")

    try:
        # Load real DINOv3 backbone
        print("Loading DINOv3 backbone...")
        from src.stage1.utilities.losses.repa.repa_feature_loss import (
            load_repa_dino_v3_model,
        )

        backbone = load_repa_dino_v3_model(
            weight_path=None,
            model_name="dinov3_vits16",
            pretrained_on="web",
            compile=False,
        )
        print(f"DINOv3 backbone loaded successfully: {type(backbone)}")

        # Create adapter instance
        print("Creating DINOv3_Adapter...")
        adapter = DINOv3_Adapter(
            backbone=backbone,
            interaction_indexes=[2, 5, 8, 11],
            pretrain_size=512,
            conv_inplane=64,
            n_points=4,
            deform_num_heads=16,
            drop_path_rate=0.1,
            init_values=0.0,
            with_cffn=True,
            cffn_ratio=0.25,
            deform_ratio=0.5,
            add_vit_feature=True,
            use_extra_extractor=True,
            with_cp=False,  # Disable checkpointing for testing
            use_bn=True,
        )
        print("Adapter created successfully")

        # Move to CUDA
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        adapter = adapter.to(device)
        backbone = backbone.to(device)

        # Test forward pass
        print("Testing forward pass...")
        batch_size = 1
        input_size = 256
        x = torch.randn(batch_size, 3, input_size, input_size, device=device)

        output = adapter(x)
        print(f"Output keys: {output.keys()}")
        print(f"Output shapes: {[f'{k}: {v.shape}' for k, v in output.items()]}")

        # Expected output sizes should be 4x, 8x, 16x, 32x downsampled
        expected_sizes = {
            "1": input_size // 4,
            "2": input_size // 8,
            "3": input_size // 16,
            "4": input_size // 32,
        }

        for key in output.keys():
            actual_shape = output[key].shape
            expected_size = expected_sizes[key]

            assert actual_shape == (
                batch_size,
                backbone.embed_dim,
                expected_size,
                expected_size,
            ), (
                f"Expected shape ({batch_size}, {backbone.embed_dim}, {expected_size}, {expected_size}), got {actual_shape}"
            )

        # Check if outputs are valid
        for key, out in output.items():
            assert not torch.isnan(out).any(), f"Output {key} contains NaN values"
            assert torch.isfinite(out).all(), f"Output {key} contains infinite values"

        print("✓ Forward pass test passed")
        for key, out in output.items():
            print(f"✓ Output {key}: {out.shape} ({input_size // out.shape[2]}x downsampling)")

        # Test backward pass
        print("Testing backward pass...")
        loss = sum(out.mean() for out in output.values())
        loss.backward()

        # Check if gradients are computed
        grad_count = 0
        for name, param in adapter.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_count += 1

        print(f"✓ Backward pass test passed ({grad_count} parameters with gradients)")

        # Test with different input sizes
        print("Testing with different input sizes...")
        test_sizes = [128, 256, 512]

        for size in test_sizes:
            x_test = torch.randn(1, 3, size, size, device=device)
            output_test = adapter(x_test)

            expected_sizes = {
                "1": size // 4,
                "2": size // 8,
                "3": size // 16,
                "4": size // 32,
            }

            for key in output_test.keys():
                actual_shape = output_test[key].shape
                expected_size = expected_sizes[key]

                assert actual_shape[2] == expected_size and actual_shape[3] == expected_size, (
                    f"Size {size}x{size}, key {key}: expected {expected_size}x{expected_size}, got {actual_shape[2]}x{actual_shape[3]}"
                )

            print(
                f"✓ Input {size}x{size} -> Outputs {[f'{k}: {v.shape[2]}x{v.shape[3]}' for k, v in output_test.items()]}"
            )

        print("✓ All size tests passed")

        # Test with add_vit_feature=False
        print("Testing with add_vit_feature=False...")
        adapter_no_vit = DINOv3_Adapter(
            backbone=backbone,
            interaction_indexes=[2, 5, 8, 11],
            add_vit_feature=False,
            with_cp=False,
        )

        adapter_no_vit = adapter_no_vit.to(device)
        output_no_vit = adapter_no_vit(x)
        for key in output.keys():
            assert output_no_vit[key].shape == output[key].shape, (
                f"Output shape should be the same with/without ViT features for key {key}"
            )

        print("✓ add_vit_feature=False test passed")

        # Test output value ranges
        print("Testing output value ranges...")
        for key, out in output.items():
            mean_val = out.mean().item()
            std_val = out.std().item()
            min_val = out.min().item()
            max_val = out.max().item()
            print(f"✓ Output {key}: mean={mean_val:.4f}, std={std_val:.4f}, min={min_val:.4f}, max={max_val:.4f}")

        print("\n🎉 All DINOv3_Adapter tests passed!")

        return True

    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    # # Test the adapter
    test_ms_adapter()

    # Test DINOv3_Adapter
    # test_dinov3_adapter()
