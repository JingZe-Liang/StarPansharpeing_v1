import sys
from typing import List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from loguru import logger
from timm.layers import (
    create_conv2d,
    create_norm_act_layer,
    get_act_layer,
    get_norm_act_layer,
    get_norm_layer,
)
from timm.layers.norm import LayerNorm2d
from timm.layers.squeeze_excite import SqueezeExcite
from timm.layers.weight_init import lecun_normal_
from torch import Tensor
from torch.nn.modules.conv import _ConvNd

from ...layers.dinov3_adapter import DINOv3_Adapter
from ...layers.stages import MbConvSequentialCond, Spatial2DNatStage

# sys.path.append("src/stage1/utilities/losses/dinov3")  # load dinov3 self-holded adapter
# from dinov3.eval.segmentation.models.backbone.dinov3_adapter import (  # type: ignore
#     DINOv3_Adapter,
# )
# from dinov3.models.vision_transformer import (  # type: ignore
#     DinoVisionTransformer,
# )


def initialize(module) -> None:
    if isinstance(module, _ConvNd):
        if module.weight.requires_grad:
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    norm_cls = (
        nn.LayerNorm,
        nn.RMSNorm,
        *[get_norm_layer(n) for n in ["layernorm", "layernorm2d", "rmsnorm", "rmsnorm2d"]],
    )
    if isinstance(module, norm_cls):
        if module.weight.requires_grad:
            nn.init.ones_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)

    if isinstance(module, nn.Linear):
        if module.weight.requires_grad:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


class GatedChannelSelection(nn.Module):
    """Soft gating before projection to suppress redundant channels."""

    def __init__(self, in_ch: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.gate(x)
        return x * w


class DualBranchSharedBasis(nn.Module):
    """
    Dual-branch shared basis module.
    - Shared Branch: Captures cross-scale common information.
    - Specific Branch: Captures scale-specific information.
    """

    def __init__(
        self,
        in_ch: int,
        shared_rank: int,
        specific_rank: int,
        num_scales: int,
        bias: bool = False,
    ):
        """
        Args:
            in_ch: Input channel count (from DINOv3).
            shared_rank: Output channel count of shared branch.
            specific_rank: Output channel count of specific branch.
            num_scales: Number of scales (e.g., 4 scales).
            bias: Whether to use bias.
        """
        super().__init__()
        self.num_scales = num_scales

        # 1. Shared branch: a 1x1 convolution shared across all scales
        self.shared_branch = nn.Conv2d(in_ch, shared_rank, kernel_size=1, bias=bias)

        # 2. Specific branch: a ModuleList creating independent 1x1 convolutions for each scale
        self.specific_branches = nn.ModuleList(
            [nn.Conv2d(in_ch, specific_rank, kernel_size=1, bias=bias) for _ in range(num_scales)]
        )

    def forward(self, x: torch.Tensor, scale_idx: int) -> torch.Tensor:
        """
        Args:
            x: Input feature map.
            scale_idx: Current scale index (0, 1, 2, ...), used to select the correct specific branch.

        Returns:
            Fused feature map.
        """
        # Compute shared features
        z_shared = self.shared_branch(x)

        # Compute specific features
        # Select corresponding specific branch based on scale_idx
        z_specific = self.specific_branches[scale_idx](x)

        # Concatenate along channel dimension to fuse both types of information
        z_combined = torch.cat([z_shared, z_specific], dim=1)

        return z_combined


class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
        norm: type[nn.Module] | str = LayerNorm2d,
        act: type[nn.Module] | str = nn.ReLU,
        norm_kwargs: dict | None = None,
        act_kwargs: dict | None = None,
        inplace=False,
    ):
        super().__init__()
        norm_kwargs = {} if norm_kwargs is None else norm_kwargs
        act_kwargs = {"inplace": True} if act_kwargs is None else act_kwargs
        self.depthwise = create_conv2d(in_ch, in_ch, kernel_size, bias=bias, depthwise=True)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)
        if norm is not None:
            self.norm_act = create_norm_act_layer(norm, out_ch, act, inplace=inplace)
        else:
            self.norm_act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm_act(x)
        return x


class SharedBasisProjector(nn.Module):
    """Low-rank shared basis across scales: x -> U (shared) -> V_s (per-scale) -> target."""

    def __init__(
        self,
        in_ch: int,
        rank: int,
        out_ch_list: List[int],
        norm: Type[nn.Module] = LayerNorm2d,
        act: Type[nn.Module] = nn.ReLU,
        norm_kwargs: dict | None = None,
        act_kwargs: dict | None = None,
        bias: bool = False,
    ):
        super().__init__()
        norm_kwargs = {} if norm_kwargs is None else norm_kwargs
        act_kwargs = {"inplace": True} if act_kwargs is None else act_kwargs
        self.shared = nn.Conv2d(in_ch, rank, kernel_size=1, bias=bias)
        self.projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(rank, oc, kernel_size=1, bias=bias),
                    create_norm_act_layer(norm, oc, act) if norm else nn.Identity(),
                )
                for oc in out_ch_list
            ]
        )

    def forward(self, x_list: List[torch.Tensor]) -> List[torch.Tensor]:
        out = []
        for i, x in enumerate(x_list):
            z = self.shared(x)
            out.append(self.projs[i](z))
        return out


class FAPM(nn.Module):
    """
    Feature Adaptive Projection Module
    """

    def __init__(
        self,
        in_ch: int,
        inner_ch: int,
        out_ch_list: list[int],
        norm: type[nn.Module] | str = LayerNorm2d,
        act: type[nn.Module] | str = nn.ReLU,
        norm_kwargs: dict | None = None,
        act_kwargs: dict | None = None,
        bias: bool = False,
    ):
        super().__init__()
        norm_kwargs = {} if norm_kwargs is None else norm_kwargs
        act_kwargs = {"inplace": True} if act_kwargs is None else act_kwargs

        # --- Stage 1: Dual-branch feature extraction ---
        self.shared_basis = nn.Conv2d(in_ch, inner_ch, kernel_size=1, bias=bias)
        self.specific_bases = nn.ModuleList([nn.Conv2d(in_ch, inner_ch, kernel_size=1, bias=bias) for _ in out_ch_list])

        # --- FiLM parameter generators ---
        self.modulations = nn.ModuleList(
            [nn.Conv2d(inner_ch, inner_ch * 2, kernel_size=1, bias=bias) for _ in out_ch_list]
        )

        # --- Stage 2: Scale-wise progressive refinement ---
        self.refinement_blocks = nn.ModuleList()
        # --- New: Shortcut projection layers for residual connections ---
        self.shortcut_projections = nn.ModuleList()

        for oc in out_ch_list:
            # --- Refinement module backbone ---
            reduce = nn.Conv2d(inner_ch, oc, kernel_size=1, bias=bias)
            dw = DepthwiseSeparableConv(
                oc,
                oc,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
                norm=norm,
                act=act,
                norm_kwargs=norm_kwargs,
                act_kwargs=act_kwargs,
            )
            refine = nn.Conv2d(oc, oc, kernel_size=1, bias=bias)
            se = SqueezeExcite(oc)

            self.refinement_blocks.append(
                nn.Sequential(
                    reduce,
                    create_norm_act_layer(norm, oc, act, inplace=False) if norm else nn.Identity(),
                    dw,
                    refine,
                    se,
                )
            )

            # --- Shortcut branch ---
            # If refinement block input/output channel counts differ, need 1x1 conv to match dimensions
            if inner_ch != oc:
                self.shortcut_projections.append(nn.Conv2d(inner_ch, oc, kernel_size=1, bias=bias))
            else:
                # If dimensions are the same, no operation needed
                self.shortcut_projections.append(nn.Identity())

    def forward(self, x_list: List[torch.Tensor]) -> List[torch.Tensor]:
        out = []
        for i, x in enumerate(x_list):
            # --- Stage 1: Get context features and main features ---
            z_shared = self.shared_basis(x)
            z_specific = self.specific_bases[i](x)

            # --- FiLM modulation process ---
            gamma_beta = self.modulations[i](z_shared)
            gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
            z_modulated = gamma * z_specific + beta

            # --- Stage 2: Refine the modulated features ---
            refined = self.refinement_blocks[i](z_modulated)

            # --- Correct residual connection ---
            # 1. Project input (shortcut) to match dimensions
            shortcut = self.shortcut_projections[i](z_modulated)
            # 2. Add projected shortcut with refinement block output
            final_output = refined + shortcut

            out.append(final_output)
        return out


class LearnableUpsampleBlock(nn.Module):
    """Lightweight learnable upsampling (transpose conv) as an alternative to bilinear."""

    def __init__(self, channels: int):
        super().__init__()
        self.up2 = nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2, bias=True)

    def forward(self, x: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        h, w = x.shape[2], x.shape[3]
        out = x
        # Upsample by factors of 2 until we reach or exceed target, then final bilinear to exact size
        while h * 2 <= target_size[0] and w * 2 <= target_size[1]:
            out = self.up2(out)
            h, w = out.shape[2], out.shape[3]
        if (h, w) != target_size:
            out = F.interpolate(out, size=target_size, mode="bilinear", align_corners=False)
        return out


class DINOv3EncoderAdapter(nn.Module):
    def __init__(
        self,
        dinov3_adapter: DINOv3_Adapter,
        target_channels: list[int],
        adapter_type: str = "default",
        rank: int = 256,
        conv_op=nn.Conv2d,
        norm_op: Union[None, Type[nn.Module], str] = LayerNorm2d,
        norm_op_kwargs: dict | None = None,
        dropout_op=None,
        dropout_op_kwargs: dict | None = None,
        nonlin: Union[None, Type[torch.nn.Module], str] = nn.ReLU,
        nonlin_kwargs: dict | None = None,
        conv_bias: bool = False,
    ):
        super().__init__()
        self.dinov3_adapter = dinov3_adapter
        self.target_channels = target_channels
        self.conv_op = conv_op
        self.norm_op = norm_op if norm_op is not None else nn.BatchNorm2d
        self.norm_op_kwargs = norm_op_kwargs if norm_op_kwargs is not None else {}
        self.nonlin = nonlin if nonlin is not None else nn.ReLU
        self.nonlin_kwargs = nonlin_kwargs if nonlin_kwargs is not None else {"inplace": True}
        self.conv_bias = conv_bias

        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs

        # in_ch = self.dinov3_adapter.backbone.embed_dim
        in_ch = int(self.dinov3_adapter.embed_dim)

        self.fapm = FAPM(
            in_ch,
            rank,
            target_channels,
            norm=self.norm_op,
            act=self.nonlin,
            norm_kwargs=self.norm_op_kwargs,
            act_kwargs=self.nonlin_kwargs,
            bias=conv_bias,
        )

        # Learnable upsampling for spatial alignment
        self.ups = nn.ModuleList()
        for oc in target_channels:
            self.ups.append(LearnableUpsampleBlock(oc))

        self.output_channels = target_channels
        self.strides = [[2, 2]] * len(target_channels)
        self.kernel_sizes = [[3, 3]] * len(target_channels)

    def forward(self, x: Float[Tensor, "b 3 h w"]):
        H, W = x.shape[-2:]
        feats = self.dinov3_adapter(x)

        others = None
        if isinstance(feats, tuple):
            # assert len(feats) == 2
            feats, others = feats

        keys = ["1", "2", "3", "4"]
        x_list = [feats[k] for k in keys]

        # Apply FAPM projection
        ys = self.fapm(x_list)

        # Apply learnable upsampling
        skips = []
        for i, y in enumerate(ys):
            target = (H // (2**i), W // (2**i))
            y = self.ups[i](y, target)
            skips.append(y)

        if others is None:
            return skips

        return skips, others

    def compute_conv_feature_map_size(self, input_size):
        return 0


class UNetDecoder(nn.Module):
    def __init__(
        self,
        encoder: DINOv3EncoderAdapter,
        num_classes: int,
        latent_width: int | None = None,
        n_conv_per_stage: int | list[int] = 2,
        depths_per_stage: int | list[int] = 2,
        nonlin_first: bool = False,
        norm_op: str = "layernorm2d",
        norm_op_kwargs: dict | None = None,
        dropout_op=None,
        dropout_op_kwargs: dict | None = None,
        nonlin: Union[Type[torch.nn.Module], None] = None,
        nonlin_kwargs: dict | None = None,
        conv_bias: Union[bool, None] = None,
        deep_supervision: bool = False,
        has_latent_condition: bool = False,
        block_types: list[str] = ["mbconv", "mbconv", "mbconv", "mbconv"],
        block_kwargs: dict = {},
    ):
        """
        This class needs the skips of the encoder as input in its forward.

        the encoder goes all the way to the bottleneck, so that's where the decoder picks up. stages in the decoder
        are sorted by order of computation, so the first stage has the lowest resolution and takes the bottleneck
        features and the lowest skip as inputs
        the decoder has two (three) parts in each stage:
        1) conv transpose to upsample the feature maps of the stage below it (or the bottleneck in case of the first stage)
        2) n_conv_per_stage conv blocks to let the two inputs get to know each other and merge
        3) (optional if deep_supervision=True) a segmentation output Todo: enable upsample logits?
        :param encoder:
        :param num_classes:
        :param n_conv_per_stage:
        :param deep_supervision:
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        self.num_classes = num_classes
        conv_op = encoder.conv_op
        n_stages_encoder = len(encoder.output_channels)

        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        if isinstance(depths_per_stage, int):
            depths_per_stage = [depths_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == len(depths_per_stage) == n_stages_encoder - 1, (
            "n_conv_per_stage must have as many entries as we have "
            "resolution stages - 1 (n_stages in encoder - 1), "
            "here: {}".format(n_stages_encoder)
        )

        # transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)
        conv_bias = encoder.conv_bias if conv_bias is None else conv_bias
        norm_op = encoder.norm_op if norm_op is None else norm_op
        norm_op_kwargs = encoder.norm_op_kwargs if norm_op_kwargs is None else norm_op_kwargs
        dropout_op = encoder.dropout_op if dropout_op is None else dropout_op
        dropout_op_kwargs = encoder.dropout_op_kwargs if dropout_op_kwargs is None else dropout_op_kwargs
        nonlin = encoder.nonlin if nonlin is None else nonlin
        if isinstance(nonlin, str):
            nonlin = get_act_layer(nonlin)

        # we start with the bottleneck and work out way up
        stages = []
        transpconvs = []
        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            block_type = block_types[s - 1]
            transpconvs.append(
                nn.ConvTranspose2d(
                    input_features_below,
                    input_features_skip,
                    stride_for_transpconv,
                    stride_for_transpconv,
                    bias=conv_bias,
                )
            )

            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            embed_dim = [2 * input_features_skip] * depths_per_stage[s - 1]
            depths = [n_conv_per_stage[s - 1]] * depths_per_stage[s - 1]

            ###### Build blocks ######
            if block_type == "mbconv":
                stage = MbConvSequentialCond(
                    in_chans=2 * input_features_skip,
                    cond_width=latent_width,
                    out_chans=input_features_skip,
                    # only 1 stage
                    embed_dim=embed_dim,
                    depths=depths,
                    norm_layer=norm_op,
                    act_layer=nonlin,
                    expand_ratio=1,
                )
            elif block_type == "nat":
                stage = Spatial2DNatStage(
                    in_chans=2 * input_features_skip,
                    embed_dim=embed_dim,
                    depths=depths,
                    cond_width=latent_width,
                    out_chans=input_features_skip,
                    norm_layer=norm_op,
                    drop_path=0.0,
                    **block_kwargs,
                )
            else:
                raise ValueError(f"Unsupported block_type: {block_type}")
            stages.append(stage)
            logger.debug(
                f"Build stage {s}: {block_type}, inp_chans={2 * input_features_skip},"
                f"out_chans={input_features_skip}, depths={depths}, embed_dim={embed_dim}"
            )

            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            if self.deep_supervision or s == (n_stages_encoder - 1):  # Zihan NOTE: add this
                # add segmentation layer each layer or only at the last layer
                seg_layers.append(
                    nn.Sequential(
                        create_norm_act_layer("layernorm2d", input_features_skip, "gelu"),
                        conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True),
                    )
                )
                logger.debug(f"Make segmentation layer at layer {s}")

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(
        self,
        skips: list[Float[Tensor, "b c h w"]],
        cond: Float[Tensor, "b latent_ch h w"] | None = None,
    ):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        lres_input = skips[-1]
        seg_outputs = []

        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s + 2)]), 1)
            x = self.stages[s](x, cond)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r
