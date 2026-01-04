"""
SemStereo: Semantic-Constrained Stereo Matching with Tokenizer Backbone.
Refatored to strictly follow the original SemStereo architecture with optimizations:
1. Pytorch SDPA for attention.
2. Standardized class names (CamelCase).
3. Dataclass configuration.

Original Paper: "SemStereo: Semantic-Constrained Stereo Matching Network for Remote Sensing"
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from jaxtyping import Float
from omegaconf import OmegaConf
from torch import Tensor
from torch.nn.modules.conv import _ConvNd
from timm.layers import get_norm_layer, get_act_layer, LayerNorm2d
import segmentation_models_pytorch as smp

from src.stage1.cosmos.cosmos_hybrid import CosmosHybridTokenizer
from src.stage2.segmentation.models.adapter import DINOv3EncoderAdapter
from src.stage2.segmentation.models.tokenizer_backbone_adapted import (
    TOKENIZER_INTERACTION_INDEXES,
    HybridTokenizerEncoderAdapter,
)
from src.utilities.config_utils import function_config_to_basic_types

from loguru import logger

logger = logger.bind(_name_="stereo_match_model_v2")

# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SemStereoConfig:
    # Model Architecture
    num_classes: int = 6

    # US3D is set to [−64, 64) and WHU is set to [0, 128)
    min_disp: int = -64
    max_disp: int = 64

    # Feature & Backbone
    adapter_out_channels: List[int] = field(default_factory=lambda: [512, 512, 512, 512])  # s4, s8, s16, s32
    semstereo_feature_channels: List[int] = field(
        default_factory=lambda: [128, 256, 512, 768, 512]
    )  # s2, s4, s8, s16, s32

    # Tokenizer settings (Sub-config could be nested, keeping flat for simplicity relative to current usage)
    tokenizer_pretrained_path: Optional[str] = None
    tokenizer_model_name: str = "cosmos_tokenizer"

    # Loss Weights
    alpha: float = 1.0  # Semantic loss weight (CE)
    beta: float = 0.5  # LRSC loss weight
    gamma: float = 0.5  # Dice loss weight for semantic segmentation

    # Flags
    seg_if: bool = True
    stereo_if: bool = True
    att_weights_only: bool = False
    debug: bool = False

    # Original Cfg Object (Legacy support for create function)
    legacy_cfg: Optional[dict] = None


# =============================================================================
# Helper Modules
# =============================================================================


class LayerNorm3d(nn.LayerNorm):
    """
    LayerNorm for 5D tensors in NCDHW format (Cost Volume normalization).

    In stereo matching, Cost Volume has shape [B, C, D, H, W] where:
    - B: Batch size
    - C: Feature channels
    - D: Disparity dimension (different disparity hypotheses)
    - H, W: Spatial dimensions

    This normalizes across the channel dimension for each sample independently,
    which is more stable than BatchNorm3d especially for small batch sizes
    and structured features like Cost Volumes.
    """

    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-6,
        affine: bool = True,
        **kwargs,
    ):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, C, D, H, W] input tensor
        Returns:
            Normalized tensor with same shape
        """
        # Permute to [B, D, H, W, C] for layer_norm
        x = x.permute(0, 2, 3, 4, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # Permute back to [B, C, D, H, W]
        x = x.permute(0, 4, 1, 2, 3)
        return x


from timm.layers import create_norm

create_norm._NORM_MAP["layernorm3d"] = LayerNorm3d  # type: ignore


class BasicConv(nn.Module):
    """Basic Convolution block with LayerNorm and ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        deconv: bool = False,
        is_3d: bool = False,
        bn: bool = True,
        relu: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.norm = LayerNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.norm = LayerNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.use_bn:
            x = self.norm(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x


class Conv2x(nn.Module):
    """
    Convolution block that optionally performs upsampling and concatenation/addition with a residual/skip connection.
    Used extensively in SPX module.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        deconv: bool = False,
        is_3d: bool = False,
        concat: bool = True,
        keep_concat: bool = True,
        bn: bool = True,
        relu: bool = True,
        keep_dispc: bool = False,
    ):
        super().__init__()
        self.concat = concat
        self.is_3d = is_3d

        kernel: Union[int, Tuple[int, int, int]] = 3
        stride: Union[int, Tuple[int, int, int]] = 2
        padding: Union[int, Tuple[int, int, int]] = 1

        if deconv:
            kernel = 4
            if is_3d:
                kernel = (4, 4, 4)
                if keep_dispc:
                    kernel = (1, 4, 4)
                    stride = (1, 2, 2)
                    padding = (0, 1, 1)

        self.conv1 = BasicConv(
            in_channels,
            out_channels,
            deconv,
            is_3d,
            bn=bn,
            relu=relu,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )

        if self.concat:
            mul = 2 if keep_concat else 1
            self.conv2 = BasicConv(
                out_channels * 2, out_channels * mul, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1
            )
        else:
            self.conv2 = BasicConv(
                out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x: Tensor, rem: Tensor) -> Tensor:
        x = self.conv1(x)
        if x.shape != rem.shape:
            # Handle dimension mismatch (usually upsampling)
            x = F.interpolate(x, size=rem.shape[-2:], mode="bilinear", align_corners=False)

        if self.concat:
            x = torch.cat((x, rem), 1)
        else:
            x = x + rem
        x = self.conv2(x)
        return x


class FeatUp(nn.Module):
    """
    Feature Upsampling Decoder with skip connections.

    Following the original SemStereo implementation, this module performs:
    1. 2x upsampling using ConvTranspose2d
    2. Skip connection concatenation with corresponding encoder features
    3. 3x3 Conv fusion to combine multi-scale information

    This is critical for semantic segmentation quality as it enables
    multi-scale feature fusion for both left and right images.

    Channel flow (with chans=[128, 128, 256, 512, 512]):
        - s32 (512ch) -> deconv32_16 -> 512ch, concat with s16 (512ch) -> 1024ch -> conv2 -> 1024ch
        - s16 (1024ch) -> deconv16_8 -> 256ch, concat with s8 (256ch) -> 512ch -> conv2 -> 512ch
        - s8 (512ch) -> deconv8_4 -> 128ch, concat with s4 (128ch) -> 256ch -> conv2 -> 256ch
        - s4 (256ch) -> deconv4_2 -> 128ch, concat with s2 (128ch) -> 256ch -> conv2 -> 256ch
    """

    def __init__(self, chans: list[int]):
        """
        Args:
            chans: Channel list at each scale [s2, s4, s8, s16, s32]
                   FAPM outputs [128, 256, 512, 768] at s4, s8, s16, s32
                   After projection: [128, 128, 256, 512, 512] at s2~s32
        """
        super().__init__()
        # Note: Conv2x(in_ch, out_ch) does:
        #   1. ConvTranspose2d(in_ch, out_ch) for 2x upsample
        #   2. Concat with rem (out_ch channels expected)
        #   3. Conv2d(out_ch*2, out_ch*2) if keep_concat=True

        # s32 -> s16: 512 -> 512, concat with s16 (512ch) -> 1024ch
        self.deconv32_16 = Conv2x(chans[4], chans[3], deconv=True, concat=True)
        # s16 -> s8: 1024 -> 256, concat with s8 (256ch) -> 512ch
        self.deconv16_8 = Conv2x(chans[3] * 2, chans[2], deconv=True, concat=True)
        # s8 -> s4: 512 -> 128, concat with s4 (128ch) -> 256ch
        self.deconv8_4 = Conv2x(chans[2] * 2, chans[1], deconv=True, concat=True)
        # s4 -> s2: 256 -> 128, concat with s2 (128ch) -> 256ch
        self.deconv4_2 = Conv2x(chans[1] * 2, chans[0], deconv=True, concat=True)

    def forward(
        self, feat_l: list[Tensor], feat_r: list[Tensor] | None = None
    ) -> tuple[list[Tensor], list[Tensor] | None]:
        """
        Upsample and fuse multi-scale features for left and right images.

        Args:
            feat_l: Left image features [f2, f4, f8, f16, f32] with channels [128, 128, 256, 512, 512]
            feat_r: Right image features [f2, f4, f8, f16, f32], optional

        Returns:
            Fused features for left (and right if provided)
            Output channels: [256, 256, 512, 1024, 512] at s2~s32
        """
        x2, x4, x8, x16, x32 = feat_l

        # Progressive upsampling with skip connections
        # Note: Conv2x output channels = out_ch * 2 due to concat + keep_concat
        x16_up = self.deconv32_16(x32, x16)  # 512 -> 512, concat 512 -> 1024ch
        x8_up = self.deconv16_8(x16_up, x8)  # 1024 -> 256, concat 256 -> 512ch
        x4_up = self.deconv8_4(x8_up, x4)  # 512 -> 128, concat 128 -> 256ch
        x2_up = self.deconv4_2(x4_up, x2)  # 256 -> 128, concat 128 -> 256ch

        feat_l_out = [x2_up, x4_up, x8_up, x16_up, x32]

        if feat_r is not None:
            y2, y4, y8, y16, y32 = feat_r
            y16_up = self.deconv32_16(y32, y16)
            y8_up = self.deconv16_8(y16_up, y8)
            y4_up = self.deconv8_4(y8_up, y4)
            y2_up = self.deconv4_2(y4_up, y2)
            feat_r_out = [y2_up, y4_up, y8_up, y16_up, y32]
            return feat_l_out, feat_r_out

        return feat_l_out, None


def convln_3d(in_planes: int, out_planes: int, kernel_size: int, stride: int, pad: int) -> nn.Module:
    """
    Helper for 3D Conv + LayerNorm.

    Why 3D convolutions?
    --------------------
    Cost Volume is a 5D tensor [B, C, D, H, W] where D represents different
    disparity candidates. Conv3D allows the network to:
    1. Aggregate features across different disparity hypotheses
    2. Learn spatial-disparity correlations (similar to spatial-temporal in videos)
    3. Refine the matching cost through multi-scale receptive fields

    Why LayerNorm instead of BatchNorm?
    ------------------------------------
    1. More stable with small batch sizes
    2. No running stats -> consistent train/test behavior
    3. Better suited for structured features like Cost Volumes
    4. Per-sample normalization avoids batch statistics contamination
    """
    return nn.Sequential(
        nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False),
        LayerNorm3d(out_planes),
    )


class AttentionBlock3D(nn.Module):
    """
    3D Window Attention Block optimized with Torch SDPA (Scaled Dot Product Attention).
    """

    def __init__(self, channels_3d: int, num_heads: int = 16, block_size: Tuple[int, int, int] = (4, 4, 4)):
        super().__init__()
        self.block_size = block_size
        self.dim_3d = channels_3d
        self.num_heads = num_heads

        assert channels_3d % num_heads == 0, "Channels must be divisible by num_heads"

        self.qkv_3d = nn.Linear(self.dim_3d, self.dim_3d * 3, bias=False)
        self.proj_3d = nn.Linear(self.dim_3d, self.dim_3d)

    def forward(self, x: Float[Tensor, "b c d h w"]) -> Float[Tensor, "b c d h w"]:
        """
        Args:
            x: [B, C, D, H, W] input tensor
        """
        B, C, D, H, W = x.shape
        bd, bh, bw = self.block_size

        # 1. Padding
        pad_d = (bd - D % bd) % bd
        pad_h = (bh - H % bh) % bh
        pad_w = (bw - W % bw) % bw

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x_padded = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
        else:
            x_padded = x

        D_p, H_p, W_p = x_padded.shape[2:]

        # 2. Window Partitioning -> [Num_windows, Window_Size, C]
        # Reshape to [B, C, D_blocks, block_d, H_blocks, block_h, W_blocks, block_w]
        x_windows = x_padded.view(B, C, D_p // bd, bd, H_p // bh, bh, W_p // bw, bw)
        # Permute to [B, D_blocks, H_blocks, W_blocks, block_d, block_h, block_w, C]
        x_windows = x_windows.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
        # Flatten to [Num_Windows, Window_Size_Flat, C]
        # Num_Windows = B * (D_p//bd) * (H_p//bh) * (W_p//bw)
        # Window_Size_Flat = bd * bh * bw
        x_windows_flat = x_windows.view(-1, bd * bh * bw, C)

        # 3. Attention (using SDPA)
        # qkv: [Num_Windows, Window_Size_Flat, 3 * C]
        qkv = self.qkv_3d(x_windows_flat)
        # Reshape to [Num_Windows, Window_Size_Flat, 3, Num_Heads, Head_Dim]
        qkv = qkv.reshape(x_windows_flat.shape[0], x_windows_flat.shape[1], 3, self.num_heads, C // self.num_heads)
        # Permute to [3, Num_Windows, Num_Heads, Window_Size_Flat, Head_Dim]
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Functional Scaled Dot Product Attention (FlashAttention enabled if available)
        attn_out = F.scaled_dot_product_attention(q, k, v)  # [Num_Windows, Num_Heads, Window_Size_Flat, Head_Dim]

        # 4. Projection & Reshape Back
        # Reshape back to [Num_Windows, Window_Size_Flat, C]
        attn_out = attn_out.transpose(1, 2).reshape(x_windows_flat.shape[0], x_windows_flat.shape[1], C)
        x_out = self.proj_3d(attn_out)

        # 5. Window Reverse
        x_out = x_out.view(B, D_p // bd, H_p // bh, W_p // bw, bd, bh, bw, C)
        # Permute back to [B, C, D_blocks, block_d, H_blocks, block_h, W_blocks, block_w]
        x_out = x_out.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        x_restored = x_out.view(B, C, D_p, H_p, W_p)

        # 6. Crop Padding
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x_restored = x_restored[:, :, :D, :H, :W]

        return x + x_restored


class Hourglass3D(nn.Module):
    """3D Hourglass Module for Cost Volume Aggregation."""

    def __init__(self, in_channels: int):
        super().__init__()

        self.conv1 = nn.Sequential(convln_3d(in_channels, in_channels * 2, 3, 2, 1), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(convln_3d(in_channels * 2, in_channels * 2, 3, 1, 1), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(convln_3d(in_channels * 2, in_channels * 4, 3, 2, 1), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(convln_3d(in_channels * 4, in_channels * 4, 3, 1, 1), nn.ReLU(inplace=True))

        self.attention_block = AttentionBlock3D(channels_3d=in_channels * 4, num_heads=16, block_size=(4, 4, 4))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            LayerNorm3d(in_channels * 2),
        )
        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            LayerNorm3d(in_channels),
        )

        self.redir1 = convln_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convln_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x: Tensor) -> Tensor:
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv4 = self.attention_block(conv4)
        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)
        return conv6


class ChannelAttention(nn.Module):
    """Applies channel attention to cost volume based on image features."""

    def __init__(self, cv_chan: int, im_chan: int):
        super().__init__()
        self.im_att = nn.Sequential(
            BasicConv(im_chan, im_chan // 2, kernel_size=1, stride=1, padding=0), nn.Conv2d(im_chan // 2, cv_chan, 1)
        )

    def forward(self, cv: Tensor, im: Tensor) -> Tensor:
        channel_att = self.im_att(im).unsqueeze(2)
        cv = torch.sigmoid(channel_att) * cv
        return cv


class SegmentationHead(nn.Module):
    """Semantic Segmentation Head."""

    def __init__(self, inplanes: int, interplanes: int, outplanes: int, scale_factor: Optional[float] = None):
        super().__init__()
        self.conv1 = BasicConv(inplanes, interplanes, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        out = self.conv2(x)
        if self.scale_factor is not None:
            # Using size instead of scale_factor for better determinism if needed, but scale_factor ok here
            height = int(x.shape[-2] * self.scale_factor)
            width = int(x.shape[-1] * self.scale_factor)
            out = F.interpolate(out, size=[height, width], mode="bilinear", align_corners=False)
        return out


class Propagation(nn.Module):
    """Disparity Propagation Module."""

    def __init__(self):
        super().__init__()
        self.replicationpad = nn.ReplicationPad2d(1)

    def forward(self, disparity_samples: Tensor) -> Tensor:
        kernel = torch.zeros(5, 1, 3, 3, device=disparity_samples.device).float()
        # Explicit coordinates
        kernel[0, 0, 0, 0] = 1.0  # Top-Left
        kernel[1, 0, 1, 1] = 1.0  # Center
        kernel[2, 0, 2, 2] = 1.0  # Bottom-Right
        kernel[3, 0, 2, 0] = 1.0  # Bottom-Left
        kernel[4, 0, 0, 2] = 1.0  # Top-Right

        disparity_samples = self.replicationpad(disparity_samples)
        aggregated = F.conv2d(disparity_samples, kernel, padding=0)
        return aggregated


class PropagationProb(nn.Module):
    """Probability Propagation Module (3D)."""

    def __init__(self):
        super().__init__()
        self.replicationpad = nn.ReplicationPad3d((1, 1, 1, 1, 0, 0))

    def forward(self, prob_volume: Tensor) -> Tensor:
        # Kernel: [OutCh=5, InCh=1, KD=1, KH=3, KW=3]
        kernel = torch.zeros(5, 1, 1, 3, 3, device=prob_volume.device).float()
        kernel[0, 0, 0, 0, 0] = 1.0
        kernel[1, 0, 0, 1, 1] = 1.0
        kernel[2, 0, 0, 2, 2] = 1.0
        kernel[3, 0, 0, 2, 0] = 1.0
        kernel[4, 0, 0, 0, 2] = 1.0

        if prob_volume.dim() == 4:
            prob_volume = prob_volume.unsqueeze(1)

        prob_volume = self.replicationpad(prob_volume)
        prob_volume_propa = F.conv3d(prob_volume, kernel, padding=0)
        return prob_volume_propa


class SSRUpsample(nn.Module):
    """Semantic Selective Refinement Upsampling."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.conv = nn.Sequential(
            LayerNorm2d(1), nn.Conv2d(1, num_classes, kernel_size=3, padding=1), LayerNorm2d(num_classes)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_classes, num_classes, kernel_size=1, padding=0), LayerNorm2d(num_classes)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_classes, num_classes, kernel_size=1, padding=0), LayerNorm2d(num_classes)
        )
        self.conv3 = nn.Conv2d(num_classes, 1, kernel_size=1, padding=0)

    def forward(self, depth_low: Tensor, spx_pred: Tensor, pred_label: Tensor) -> Tensor:
        b, c, h, w = depth_low.shape
        pred_label = F.softmax(pred_label, dim=1)

        depth_ = F.interpolate(depth_low, size=(h * 4, w * 4), mode="bilinear", align_corners=False)
        depth = self.conv(depth_)

        prob = torch.sigmoid(self.conv1(pred_label * spx_pred))
        prob = torch.sigmoid(self.conv2(prob * spx_pred))

        res = self.conv3(depth * prob)
        out = depth_ + res
        return out.squeeze(1)


# =============================================================================
# Functional Helpers
# =============================================================================


def groupwise_correlation_norm(fea1: Tensor, fea2: Tensor, num_groups: int) -> Tensor:
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    fea1 = fea1.view([B, num_groups, channels_per_group, H, W])
    fea2 = fea2.view([B, num_groups, channels_per_group, H, W])

    # Normalize with epsilon
    fea1_norm = fea1 / (torch.norm(fea1, 2, 2, True) + 1e-05)
    fea2_norm = fea2 / (torch.norm(fea2, 2, 2, True) + 1e-05)

    cost = (fea1_norm * fea2_norm).mean(dim=2)
    return cost


def build_gwc_volume_norm(refimg_fea: Tensor, targetimg_fea: Tensor, maxdisp: int, num_groups: int) -> Tensor:
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp * 2, H, W])

    for i in range(-maxdisp, maxdisp):
        if i < 0:
            volume[:, :, i + maxdisp, :, :i] = groupwise_correlation_norm(
                refimg_fea[:, :, :, :i], targetimg_fea[:, :, :, -i:], num_groups
            )
        elif i > 0:
            volume[:, :, i + maxdisp, :, i:] = groupwise_correlation_norm(
                refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i], num_groups
            )
        else:
            volume[:, :, i + maxdisp, :, :] = groupwise_correlation_norm(refimg_fea, targetimg_fea, num_groups)
    return volume.contiguous()


def disparity_regression(x: Tensor, maxdisp: int) -> Tensor:
    disp_values = torch.arange(-maxdisp, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp * 2, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=False)


def disparity_variance(x: Tensor, maxdisp: int, disparity: Tensor) -> Tensor:
    disp_values = torch.arange(-maxdisp, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp * 2, 1, 1)
    disp_values = (disp_values - disparity) ** 2
    return torch.sum(x * disp_values, 1, keepdim=True)


def spatial_transformer_grid(x: Tensor, y: Tensor, disp_range_samples: Tensor) -> Tuple[Tensor, Tensor]:
    bs, channels, height, width = y.size()
    ndisp = disp_range_samples.size(1)

    mh, mw = torch.meshgrid(
        [
            torch.arange(0, height, dtype=x.dtype, device=x.device),
            torch.arange(0, width, dtype=x.dtype, device=x.device),
        ],
        indexing="ij",
    )

    mh = mh.reshape(1, 1, height, width).repeat(bs, ndisp, 1, 1)
    mw = mw.reshape(1, 1, height, width).repeat(bs, ndisp, 1, 1)

    cur_disp_coords_x = mw - disp_range_samples
    cur_disp_coords_y = mh

    coords_x = cur_disp_coords_x / ((width - 1.0) / 2.0) - 1.0
    coords_y = cur_disp_coords_y / ((height - 1.0) / 2.0) - 1.0
    grid = torch.stack([coords_x, coords_y], dim=4)

    y_warped = F.grid_sample(
        y, grid.view(bs, ndisp * height, width, 2), mode="bilinear", padding_mode="zeros", align_corners=True
    ).view(bs, channels, ndisp, height, width)

    x_warped = x.unsqueeze(2).repeat(1, 1, ndisp, 1, 1)

    return y_warped, x_warped


def regression_topk(cost: Tensor, disparity_samples: Tensor, k: int) -> Tensor:
    _, ind = cost.sort(1, True)
    pool_ind = ind[:, :k]
    cost = torch.gather(cost, 1, pool_ind)
    prob = F.softmax(cost, 1)
    disparity_samples = torch.gather(disparity_samples, 1, pool_ind)
    pred = torch.sum(disparity_samples * prob, dim=1, keepdim=True)
    return pred


def warp_by_disparity(tensor: Tensor, disparity: Tensor) -> Tensor:
    b, c, h, w = tensor.shape
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, h, device=tensor.device),
        torch.linspace(-1, 1, w, device=tensor.device),
        indexing="ij",
    )
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
    disp_normalized = disparity.permute(0, 2, 3, 1) * 2 / w
    grid_warped = grid.clone()
    grid_warped[..., 0] = grid[..., 0] - disp_normalized.squeeze(-1)
    warped = F.grid_sample(tensor, grid_warped, mode="bilinear", padding_mode="border", align_corners=True)
    return warped


# =============================================================================
# Main Model
# =============================================================================


class SemStereo(nn.Module):
    def __init__(self, config: SemStereoConfig):
        super(SemStereo, self).__init__()
        self.config = config
        self.num_classes = config.num_classes
        self.min_disp = config.min_disp
        self.max_disp = config.max_disp
        self.maxdisp = config.max_disp  # Alias for compatibility with original code
        self.att_weights_only = config.att_weights_only
        self.seg_if = config.seg_if
        self.stereo_if = config.stereo_if

        # 1. Backbone
        self.encoder = self._create_tok_encoder()

        # Remove encoder.ups since we use our own upsampling layers
        # This saves memory and avoids confusion about unused parameters
        delattr(self.encoder, "ups")

        # 2. Adaptation Layer
        # FAPM outputs [128, 256, 512, 768] at s4, s8, s16, s32
        # We create s2 by upsampling s4, and project s32 to 512ch
        # Raw input channels to FeatUp: [128, 128, 256, 512, 512] @ s2~s32
        self.chans_raw = [128, 128, 256, 512, 512]  # Input to FeatUp
        self.chans2 = [64, 128, 256, 384, 256]  # Halved channels for stereo

        # Projection for stride 32 feature: 768ch -> 512ch
        self.s32_proj = BasicConv(768, 512, kernel_size=1)

        # Initial s2 projection from s4 (2x upsample + channel adjustment)
        self.s2_upsample = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False),
            LayerNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # FeatUp decoder for multi-scale feature fusion
        # Input channels: [128, 128, 256, 512, 512] @ s2~s32
        self.feat_up = FeatUp(self.chans_raw)

        # 3. Semantic Heads
        # FeatUp output at s2 is 128*2=256ch (due to concat in Conv2x)
        if self.seg_if:
            self.head_l = SegmentationHead(inplanes=256, interplanes=64, outplanes=self.num_classes, scale_factor=2)
            self.head_r = SegmentationHead(inplanes=256, interplanes=64, outplanes=self.num_classes, scale_factor=2)

        # 4. Stereo Components
        # FeatUp output channels: [256, 256, 512, 1024, 512] @ s2~s32
        self.chans_up = [256, 256, 512, 1024, 512]  # FeatUp output channels
        if self.stereo_if:
            self.gamma = nn.Parameter(torch.zeros(1))
            self.beta = nn.Parameter(2 * torch.ones(1))

            # SPX pathway (uses self.chans2 for stereo processing)
            self.spx2 = nn.Sequential(nn.ConvTranspose2d(self.chans2[0] * 2, 6, kernel_size=4, stride=2, padding=1))
            self.spx4_2 = Conv2x(self.chans2[1] * 2, self.chans2[0], True)
            self.spx8_4 = Conv2x(self.chans2[2] * 2, self.chans2[1], True)
            self.spx16_8 = Conv2x(self.chans2[3] * 2, self.chans2[2], True)
            self.spx32_16 = Conv2x(self.chans2[4], self.chans2[3], True)

            # Channel reduction from FeatUp outputs to chans2
            self.chal_0 = nn.Sequential(nn.Conv2d(self.chans_up[0], self.chans2[0], 1), LayerNorm2d(self.chans2[0]))
            self.chal_1 = nn.Sequential(nn.Conv2d(self.chans_up[1], self.chans2[1], 1), LayerNorm2d(self.chans2[1]))
            self.chal_2 = nn.Sequential(nn.Conv2d(self.chans_up[2], self.chans2[2], 1), LayerNorm2d(self.chans2[2]))
            self.chal_3 = nn.Sequential(nn.Conv2d(self.chans_up[3], self.chans2[3], 1), LayerNorm2d(self.chans2[3]))
            self.chal_4 = nn.Sequential(nn.Conv2d(self.chans_up[4], self.chans2[4], 1), LayerNorm2d(self.chans2[4]))

            # Cost Volume 1 (GWC)
            self.patch = nn.Conv3d(
                self.chans2[2] // 8,
                self.chans2[2] // 8,
                kernel_size=(1, 3, 3),
                stride=1,
                dilation=1,
                groups=self.chans2[2] // 8,
                padding=(0, 1, 1),
                bias=False,
            )
            self.corr_feature_att_8 = ChannelAttention(self.chans2[1] // 4, self.chans2[2])
            self.hourglass_att = Hourglass3D(32)
            self.classif_att_ = nn.Sequential(
                convln_3d(32, 32, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False),
            )

            self.propagation = Propagation()
            self.propagation_prob = PropagationProb()

            # Cost Volume 2 (Concat)
            self.concat_feature = nn.Sequential(
                BasicConv(self.chans2[1], self.chans2[1] // 2, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.chans2[1] // 2, self.chans2[1] // 4, 3, 1, 1, bias=False),
            )
            self.concat_stem = BasicConv(
                self.chans2[1] // 2, self.chans2[1] // 4, is_3d=True, kernel_size=3, stride=1, padding=1
            )
            self.concat_feature_att_4 = ChannelAttention(self.chans2[1] // 4, self.chans2[1])

            self.hourglass = Hourglass3D(32)
            self.classif = nn.Sequential(
                convln_3d(32, 32, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False),
            )

            self.ssr_upsample = SSRUpsample(self.num_classes)

        # Loss functions
        self.dice_loss_fn = smp.losses.DiceLoss(
            mode="multiclass",
            ignore_index=-100,
        )

        self._init_weights()

        # Disable gradients for unused parameters to avoid warnings
        self._disable_unused_params_gradients()

    def state_dict(self, *args, **kwargs):
        """
        Override state_dict to exclude frozen backbone parameters.

        This prevents saving large pretrained backbone weights in checkpoints,
        saving disk space and avoiding confusion about which weights to load.
        """
        # Get the full state dict
        state_dict = super().state_dict(*args, **kwargs)

        # Remove backbone parameters (they're frozen and loaded separately)
        keys_to_remove = [k for k in state_dict.keys() if k.startswith("encoder.dinov3_adapter.backbone.")]
        for key in keys_to_remove:
            del state_dict[key]

        logger.info(f"state_dict: Removed {len(keys_to_remove)} frozen backbone parameters")
        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        Override load_state_dict to skip frozen backbone parameters.

        This allows loading checkpoints without attempting to load backbone weights,
        which are loaded separately from the pretrained tokenizer checkpoint.
        """
        # Filter out backbone keys if present in the checkpoint
        filtered_state_dict = {
            k: v for k, v in state_dict.items() if not k.startswith("encoder.dinov3_adapter.backbone.")
        }

        removed_count = len(state_dict) - len(filtered_state_dict)
        if removed_count > 0:
            logger.info(f"load_state_dict: Skipped {removed_count} frozen backbone parameters")

        # Load the filtered state dict
        return super().load_state_dict(filtered_state_dict, *args, **kwargs)

    def _create_tok_encoder(self):
        # Fallback to legacy config access because adapter creation depends on it for now,
        # or assuming config.legacy_cfg is populated.
        cfg = self.config.legacy_cfg
        f_cfg = cfg.tokenizer_feature
        a_cfg = cfg.adapter
        t_cfg = cfg.tokenizer

        model_name = f_cfg.model_name
        interaction_indexes = TOKENIZER_INTERACTION_INDEXES[model_name]
        tok_backbone = CosmosHybridTokenizer.create_model(
            cnn_cfg=t_cfg.cnn_cfg,
            trans_enc_cfg=t_cfg.trans_enc_cfg,
            trans_dec_cfg=None,
            distillation_cfg=t_cfg.distillation_cfg,
            hybrid_tokenizer_cfg=t_cfg.hybrid_tokenizer_cfg,
        )
        if self.config.tokenizer_pretrained_path is not None:
            tok_backbone.load_pretrained(self.config.tokenizer_pretrained_path)
            for param in tok_backbone.parameters():
                param.requires_grad = False

        dinov3_adapter = HybridTokenizerEncoderAdapter(
            backbone=tok_backbone,
            in_channels=f_cfg.in_channels,
            interaction_indexes=interaction_indexes,
            pretrain_size=512,
            conv_inplane=f_cfg.conv_inplane,
            n_points=4,
            deform_num_heads=f_cfg.deform_num_heads,
            drop_path_rate=f_cfg.drop_path_rate,
            init_values=0.0,
            with_cffn=f_cfg.with_cffn,
            cffn_ratio=f_cfg.cffn_ratio,
            deform_ratio=f_cfg.deform_ratio,
            add_vit_feature=f_cfg.add_vit_feature,
            use_extra_extractor=f_cfg.use_extra_extractor,
            with_cp=f_cfg.with_cp,
        )
        encoder_adapter = DINOv3EncoderAdapter(
            dinov3_adapter=dinov3_adapter,
            target_channels=[128, 256, 512, 768],  # Direct SemStereo channel mapping
            keys=["1", "2", "3", "4"],  # Use all 4 DINOv3 scales
            conv_op=nn.Conv2d,
            norm_op=get_norm_layer(a_cfg.norm),
            nonlin=get_act_layer(a_cfg.act),
            dropout_op=a_cfg.drop,
            conv_bias=a_cfg.conv_bias,
        )
        return encoder_adapter

    def _init_weights(self):
        logger.info("=============== Init weights ================")
        for name, m in self.named_modules():
            if "encoder" in name and "backbone" in name:
                logger.trace(f"skip init weights: <green>{name}</green>")
                continue

            if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                # nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.trunc_normal_(m.weight, std=0.02)

            elif isinstance(m, (LayerNorm2d, LayerNorm3d, nn.LayerNorm, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        logger.info(f"============ Init weight done ==============")

    def _disable_unused_params_gradients(self):
        """
        Disable gradients for parameters that are not used in stereo matching forward pass.

        After using all 4 DINOv3 scales (keys=["1", "2", "3", "4"]):
        - encoder.dinov3_adapter.spm.fc1 is NOW USED (for f1/c1) ✓
        - encoder.dinov3_adapter.up is still unused (can be disabled)
        - encoder.dinov3_adapter.norm1 is NOW USED (for f1) ✓
        - head_r parameters (only used when seg_if=True and training with right GT)
        """
        unused_param_patterns = [
            "encoder.dinov3_adapter.up",  # Still unused
            # Removed "spm.fc1" and "norm1" as f1 is now used
        ]

        # If seg_if is False, also disable head_r parameters
        if not self.seg_if:
            unused_param_patterns.append("head_r")

        disabled_count = 0
        for name, param in self.named_parameters():
            for pattern in unused_param_patterns:
                if name.startswith(pattern):
                    param.requires_grad = False
                    disabled_count += 1
                    logger.debug(f"Disabled gradient for: {name}")
                    break

        logger.info(f"Disabled gradients for {disabled_count} unused parameters")

    def extract_features(self, img: Tensor) -> list[Tensor]:
        """
        Extract raw features from DINOv3 encoder.

        Returns features at 5 scales that will be fed into FeatUp:
            - f_s2: stride 2, 128ch (upsampled from s4)
            - f_s4: stride 4, 128ch (first FAPM output)
            - f_s8: stride 8, 256ch (second FAPM output)
            - f_s16: stride 16, 512ch (third FAPM output)
            - f_s32: stride 32, 512ch (fourth FAPM output projected)
        """
        # Get DINOv3 native features
        feats_dict = self.encoder.dinov3_adapter(img)
        if isinstance(feats_dict, tuple):
            feats_dict, _ = feats_dict

        # Extract features with keys ["1", "2", "3", "4"]
        f1_raw = feats_dict["1"]  # stride 4, embed_dim
        f2_raw = feats_dict["2"]  # stride 8, embed_dim
        f3_raw = feats_dict["3"]  # stride 16, embed_dim
        f4_raw = feats_dict["4"]  # stride 32, embed_dim

        # Apply FAPM projection to get correct channels
        # FAPM target_channels = [128, 256, 512, 768]
        f_list = [f1_raw, f2_raw, f3_raw, f4_raw]
        fapm_outputs = self.encoder.fapm(f_list)
        f_s4 = fapm_outputs[0]  # stride 4, 128ch
        f_s8 = fapm_outputs[1]  # stride 8, 256ch
        f_s16 = fapm_outputs[2]  # stride 16, 512ch
        f_s32_raw = fapm_outputs[3]  # stride 32, 768ch

        # Generate s2 feature by upsampling s4
        f_s2 = self.s2_upsample(f_s4)  # stride 4 -> stride 2, 128ch

        # Project s32 to 512ch
        f_s32 = self.s32_proj(f_s32_raw)  # stride 32, 768ch -> 512ch

        return [f_s2, f_s4, f_s8, f_s16, f_s32]

    def concat_volume_generator(self, left_input, right_input, disparity_samples):
        right_feature_map, left_feature_map = spatial_transformer_grid(left_input, right_input, disparity_samples)
        concat_volume = torch.cat((left_feature_map, right_feature_map), dim=1)
        return concat_volume

    def forward(self, left: Tensor, right: Tensor) -> dict[str, Any]:
        # Extract raw features
        features_left_raw = self.extract_features(left)
        features_right_raw = self.extract_features(right)

        # Apply FeatUp decoder for multi-scale fusion
        features_left, features_right = self.feat_up(features_left_raw, features_right_raw)

        # Semantic segmentation heads
        # FeatUp output at s2 is 256ch (128*2 due to concat)
        if self.seg_if:
            pred_label = self.head_l(features_left[0])
            pred_label_r = self.head_r(features_right[0])

        if self.stereo_if:
            features_left[0] = self.chal_0(features_left[0])
            features_left[1] = self.chal_1(features_left[1])
            features_left[2] = self.chal_2(features_left[2])
            features_left[3] = self.chal_3(features_left[3])
            features_left[4] = self.chal_4(features_left[4])

            features_right[1] = self.chal_1(features_right[1])
            features_right[2] = self.chal_2(features_right[2])

            xspx = self.spx32_16(features_left[4], features_left[3])
            xspx = self.spx16_8(xspx, features_left[2])
            xspx = self.spx8_4(xspx, features_left[1])
            xspx = self.spx4_2(xspx, features_left[0])
            spx_pred = self.spx2(xspx)

            # Stage 1: Coarse
            corr_volume = build_gwc_volume_norm(
                features_left[2], features_right[2], self.maxdisp // 8, self.chans2[2] // 8
            )
            corr_volume = self.patch(corr_volume)

            cost_att = self.corr_feature_att_8(corr_volume, features_left[2])
            cost_att = self.hourglass_att(cost_att)
            cost_att = self.classif_att_(cost_att)

            att_weights = F.interpolate(
                cost_att, [self.maxdisp // 4 * 2, left.size()[2] // 4, left.size()[3] // 4], mode="trilinear"
            )

            pred_att = torch.squeeze(att_weights, 1)
            pred_att_prob = F.softmax(pred_att, dim=1)
            pred_att_val = disparity_regression(pred_att_prob, self.maxdisp // 4)

            pred_variance = disparity_variance(pred_att_prob, self.maxdisp // 4, pred_att_val.unsqueeze(1))
            pred_variance = self.beta + self.gamma * pred_variance
            pred_variance = torch.sigmoid(pred_variance)
            pred_variance_samples = self.propagation(pred_variance)
            disparity_samples = self.propagation(pred_att_val.unsqueeze(1))

            right_feature_x4, left_feature_x4 = spatial_transformer_grid(
                features_left[1], features_right[1], disparity_samples
            )
            disparity_sample_strength = (left_feature_x4 * right_feature_x4).mean(dim=1)
            disparity_sample_strength = torch.softmax(disparity_sample_strength * pred_variance_samples, dim=1)

            att_weights = self.propagation_prob(att_weights)
            att_weights = att_weights * disparity_sample_strength.unsqueeze(2)
            att_weights = torch.sum(att_weights, dim=1, keepdim=True)
            att_weights_prob = F.softmax(att_weights, dim=2)
            _, ind = att_weights_prob.sort(2, True)

            # Stage 2: Refine
            k = 24
            ind_k = ind[:, :, :k]
            ind_k = ind_k.sort(2, False)[0]
            att_topk = torch.gather(att_weights_prob, 2, ind_k)
            disparity_sample_topk = ind_k.squeeze(1).float() - self.maxdisp // 4

            att_prob = torch.gather(att_weights, 2, ind_k).squeeze(1)
            att_prob = F.softmax(att_prob, dim=1)
            pred_att = att_prob * disparity_sample_topk
            pred_att = torch.sum(pred_att, dim=1)
            pred_att_up = self.ssr_upsample(pred_att.unsqueeze(1), spx_pred, pred_label)

            # Deep Refinement (only if not att_weights_only)
            if not self.att_weights_only:
                concat_features_left = self.concat_feature(features_left[1])
                concat_features_right = self.concat_feature(features_right[1])
                concat_volume = self.concat_volume_generator(
                    concat_features_left, concat_features_right, disparity_sample_topk
                )

                volume = att_topk * concat_volume
                volume = self.concat_stem(volume)
                volume = self.concat_feature_att_4(volume, features_left[1])
                cost = self.hourglass(volume)
                cost = self.classif(cost)
                pred = regression_topk(cost.squeeze(1), disparity_sample_topk, 2)
                pred_up = self.ssr_upsample(pred, spx_pred, pred_label)

        # Return outputs based on mode
        if self.seg_if and not self.stereo_if:
            return {"P_l": pred_label}

        if self.training:
            if self.att_weights_only:
                if not self.seg_if:
                    return {"d_final": pred_att_up * 4, "d_aux": [pred_att * 4]}
                return {
                    "d_final": pred_att_up * 4,
                    "d_aux": [pred_att * 4],
                    "P_l": pred_label,
                    "P_r": pred_label_r,
                }
            else:
                if not self.seg_if:
                    return {
                        "d_final": pred_up * 4,
                        "d_aux": [pred.squeeze(1) * 4, pred_att_up * 4, pred_att * 4],
                    }
                return {
                    "d_final": pred_up * 4,
                    "d_aux": [pred.squeeze(1) * 4, pred_att_up * 4, pred_att * 4],
                    "P_l": pred_label,
                    "P_r": pred_label_r,
                }
        else:
            # Inference
            if self.att_weights_only:
                if not self.seg_if:
                    return {"d_final": pred_att_up * 4}
                return {"d_final": pred_att_up * 4, "P_l": pred_label}
            else:
                if not self.seg_if:
                    return {"d_final": pred_up * 4}
                return {"d_final": pred_up * 4, "P_l": pred_label}

    def compute_losses(self, outputs, d_gt, seg_gt_l=None, seg_gt_r=None, alpha=None, beta=None, gamma=None):
        """
        Compute multi-task losses for SemStereo model (disparity + semantics).

        This implementation follows the original SemStereo paper's loss formulation:
        - Disparity loss: multi-scale supervision with weights [1.0, 0.6, 0.5, 0.3]
        - Semantic loss: (CE + Dice) * scale_factor
        - LRSC loss: left-right semantic consistency
        - Total loss: disp_loss + semantic_loss + lrsc_loss

        Args:
            outputs (Dict[str, Tensor]): Model forward output dictionary containing:
                - 'd_final': Final disparity prediction [B, H, W] in pixels
                - 'd_aux': Auxiliary disparity predictions list (for multi-scale supervision)
                - 'P_l': Left image semantic prediction [B, num_classes, H, W]
                - 'P_r': Right image semantic prediction [B, num_classes, H, W] (optional)

            d_gt (Tensor): Disparity ground truth [B, H, W] in pixels.
                Represents the horizontal offset of each pixel in the left image
                when matched in the right image.

            seg_gt_l (Tensor, optional): Left image semantic segmentation ground truth [B, H, W],
                where each pixel value is a class index (0 to num_classes-1).
                If None, semantic loss will not be computed.

            seg_gt_r (Tensor, optional): Right image semantic segmentation ground truth [B, H, W].
                Primarily used for Left-Right Semantic Consistency (LRSC) loss.
                If None, LRSC loss will not be computed.

            alpha, beta, gamma: Deprecated. Kept for API compatibility but not used.
                Original SemStereo uses fixed weights.

        Returns:
            Dict[str, Tensor]: Loss dictionary containing:
                - 'total': Weighted total loss
                - 'disp': Disparity loss (smooth L1)
                - 'seg': Semantic loss (CE + Dice combined)
                - 'dice': Dice loss component
                - 'lrsc': Left-right semantic consistency loss

        Notes:
            - Only disparity values within valid range participate in loss computation (min_disp < d_gt < max_disp)
            - LRSC loss warps left semantics to right view using predicted disparity, then compares with right GT
            - Multi-scale auxiliary losses use weights [0.6, 0.5, 0.3] for the 3 auxiliary outputs
        """
        # Get mask that shows valid disparity values
        # Note: Use >= for min_disp to include the boundary value (e.g., -64 is valid)
        mask = (d_gt < self.max_disp) & (d_gt >= self.min_disp)
        mask.detach_()

        # Handle shape mismatch: d_final is typically [B, H, W], while d_gt/mask are [B, 1, H, W]
        if d_gt.dim() == 4:
            d_gt_s = d_gt.squeeze(1)
            mask_s = mask.squeeze(1)
        else:
            d_gt_s = d_gt
            mask_s = mask

        # ========================================================================
        # Disparity Loss (following original SemStereo)
        # ========================================================================
        # Original weights: [1.0, 0.6, 0.5, 0.3] for [final, aux1, aux2, aux3]
        d_final = outputs["d_final"]
        if d_final.dim() == 4:
            d_final = d_final.squeeze(1)

        L_disp = F.smooth_l1_loss(d_final[mask_s], d_gt_s[mask_s], reduction="mean")

        # Add auxiliary losses with original weights
        if "d_aux" in outputs and outputs["d_aux"] is not None:
            aux_weights = [0.6, 0.5, 0.3]  # Original SemStereo weights
            for idx, d_est in enumerate(outputs["d_aux"]):
                if d_est is not None and idx < len(aux_weights):
                    if d_est.dim() == 4:
                        d_est = d_est.squeeze(1)
                    if d_est.shape[-2:] != d_gt_s.shape[-2:]:
                        d_est = F.interpolate(
                            d_est.unsqueeze(1).float(), size=d_gt_s.shape[-2:], mode="bilinear", align_corners=False
                        ).squeeze(1)
                    L_disp += aux_weights[idx] * F.smooth_l1_loss(
                        d_est[mask_s].to(d_gt_s.dtype), d_gt_s[mask_s], reduction="mean"
                    )

        # ========================================================================
        # Semantic Segmentation Loss (following original SemStereo)
        # ========================================================================
        # Original: (CE + Dice) * scale_factor
        # scale_factor = 1.6 if attention_weights_only else 2.4
        L_seg_ce = torch.tensor(0.0, device=d_gt.device)
        L_dice = torch.tensor(0.0, device=d_gt.device)

        if seg_gt_l is not None and "P_l" in outputs:
            # CE Loss
            L_seg_ce_l = F.cross_entropy(outputs["P_l"], seg_gt_l.long(), ignore_index=-100)
            L_dice_l = self.dice_loss_fn(outputs["P_l"], seg_gt_l)

            if seg_gt_r is not None and "P_r" in outputs:
                L_seg_ce_r = F.cross_entropy(outputs["P_r"], seg_gt_r.long(), ignore_index=-100)
                L_dice_r = self.dice_loss_fn(outputs["P_r"], seg_gt_r)
                L_seg_ce = (L_seg_ce_l + L_seg_ce_r) / 2.0
                L_dice = (L_dice_l + L_dice_r) / 2.0
            else:
                L_seg_ce = L_seg_ce_l
                L_dice = L_dice_l

            # Combine CE + Dice with scale factor (original SemStereo)
            semantic_scale = 1.6 if self.att_weights_only else 2.4
            L_seg_combined = (L_seg_ce + L_dice) * semantic_scale
        else:
            L_seg_combined = torch.tensor(0.0, device=d_gt.device)

        # ========================================================================
        # LRSC Loss (Left-Right Semantic Consistency)
        # ========================================================================
        L_LRSC = torch.tensor(0.0, device=d_gt.device)
        if "P_l" in outputs and "P_r" in outputs:
            if seg_gt_l is not None:
                # Warp left GT labels to right view using predicted disparity
                b, height, width = seg_gt_l.size()
                y_warped = torch.full((b, height, width), -100, device=d_gt.device, dtype=seg_gt_l.dtype)

                y_coords, x_coords = torch.meshgrid(
                    [torch.arange(0, height, device=d_gt.device), torch.arange(0, width, device=d_gt.device)],
                    indexing="ij",
                )
                x_coords = x_coords.unsqueeze(0).expand(b, -1, -1)
                x_disp = x_coords - d_final
                x_disp = torch.clamp(x_disp, min=0, max=(width - 1))

                # Gather warped labels
                seg_gt_l_expanded = seg_gt_l.unsqueeze(1)  # [B, 1, H, W]
                pixel_values = torch.gather(seg_gt_l_expanded, 3, x_disp.unsqueeze(1).long())
                y_warped = pixel_values.squeeze(1)

                # Compute LRSC loss
                L_LRSC = F.cross_entropy(outputs["P_r"], y_warped, ignore_index=-100)

        # ========================================================================
        # Total Loss (following original SemStereo)
        # ========================================================================
        loss = L_disp + L_seg_combined + L_LRSC

        return {
            "total": loss,
            "disp": L_disp,
            "seg": L_seg_combined,  # Combined (CE + Dice) * scale
            "dice": L_dice,  # Individual Dice component for logging
            "lrsc": L_LRSC,
        }

    @classmethod
    @function_config_to_basic_types
    def create_semstereo_model(
        cls,
        cfg: dict | None = None,
        num_classes: int = 6,
        **overrides,
    ):
        if cfg is None:
            from .tokenizer_hybrid_stero_matching import _create_default_cfg as create_backbone_cfg

            cfg = create_backbone_cfg()

        if overrides:
            cfg_omega = OmegaConf.create(cfg)
            cfg_omega.merge_with(OmegaConf.create(overrides))
            cfg = edict(OmegaConf.to_container(cfg_omega, resolve=True))

        # Bridge Legacy Edict Config -> Dataclass Config
        stereo_cfg = cfg.stereo
        config = SemStereoConfig(
            num_classes=num_classes,
            min_disp=stereo_cfg.min_disp,
            max_disp=stereo_cfg.max_disp,
            tokenizer_pretrained_path=cfg.tokenizer_pretrained_path,
            debug=getattr(cfg, "_debug", False),
            legacy_cfg=cfg,
        )

        return cls(config)


if __name__ == "__main__":
    """
    CUDA_VISIBLE_DEVICES=1 MODEL_COMPILED=0 python -m src.stage2.stereo_matching.models.tokenizer_hybrid_stero_matching_v2
    """
    from .tokenizer_hybrid_stero_matching import _create_default_cfg as create_backbone_cfg

    with logger.catch():
        cfg = create_backbone_cfg()
        cfg._debug = True
        cfg.tokenizer_pretrained_path = None

        config = SemStereoConfig(
            num_classes=6,
            max_disp=192,
            debug=True,
            legacy_cfg=cfg,
            tokenizer_pretrained_path="runs/stage1_cosmos_hybrid/2025-12-21_23-52-12_hybrid_cosmos_f16c64_ijepa_pretrained_sem_no_lejepa/ema/tokenizer/model.safetensors",
        )
        model = SemStereo(config).cuda()
        model.train()

        # size = 512
        # left = torch.randn(2, 3, size, size).cuda()
        # right = torch.randn(2, 3, size, size).cuda()

        from ..data import US3DStreamingDataset

        ds, _ = US3DStreamingDataset.create_dataloader(
            input_dir="data/Downstreams/US3D_Stereo_Matching/JAX/train",
            stream_ds_kwargs=dict(shuffle=False, is_cycled=False),
            loader_kwargs=dict(batch_size=1),
        )
        sample = ds[0]
        left, right = sample["left"][None].bfloat16().cuda(), sample["right"][None].bfloat16().cuda()

        print("=" * 80)
        print("Testing forward and backward to find parameters with None gradients")
        print("=" * 80)

        # Forward pass
        with torch.autocast("cuda", torch.bfloat16):
            out = model(left, right)

        # Create dummy loss
        loss = out["d_final"].mean()

        # Backward pass
        loss.backward()

        # Check all parameters
        print("\n" + "=" * 80)
        print("Parameters with None gradients:")
        print("=" * 80)

        none_grad_params = []
        for name, param in model.named_parameters():
            if param.grad is None:
                none_grad_params.append(name)
                print(f"  {name}: {param.shape}")

        print("\n" + "=" * 80)
        print(f"Total parameters with None gradients: {len(none_grad_params)}")
        print("=" * 80)

        from fvcore.nn import parameter_count_table

        print(parameter_count_table(model, 4))

        # Group by module
        # from collections import defaultdict

        # grouped = defaultdict(list)
        # for param_name in none_grad_params:
        #     parts = param_name.split(".")
        #     if len(parts) >= 3:
        #         module = ".".join(parts[:3])
        #         grouped[module].append(param_name)

        # print("\n" + "=" * 80)
        # print("Grouped by module:")
        # print("=" * 80)
        # for module, params in sorted(grouped.items()):
        #     print(f"\n{module}:")
        #     for p in params:
        #         print(f"  - {p}")

        # print("\n" + "=" * 80)
        # print("Test completed!")
        # print("=" * 80)
