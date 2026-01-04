"""
DPT (Dense Prediction Transformer) Head for Segmentation.

This module implements a DPT-style head that fuses multi-scale features
from both CNN and Transformer encoders for dense prediction tasks.

References:
    - Vision Transformers for Dense Prediction (DPT): https://arxiv.org/abs/2103.13413
    - UniMatch V2: https://github.com/liheyoung/unimatch-v2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from timm.layers import create_norm_layer
from natten import NeighborhoodAttention2D


class ResidualConvUnit(nn.Module):
    """Residual convolution module for feature refinement."""

    def __init__(
        self,
        features: int,
        activation: nn.Module,
        norm_layer: str | None = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)

        self.norm1 = create_norm_layer(norm_layer, features)
        self.norm2 = create_norm_layer(norm_layer, features)
        self.natten_norm = create_norm_layer(norm_layer, features)

        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(x)
        out = self.conv1(out)
        out = self.norm1(out)

        out = self.activation(out)
        out = self.conv2(out)
        out = self.norm2(out)

        return out + x


class FeatureFusionBlock(nn.Module):
    """Feature fusion block that combines features from different scales."""

    def __init__(
        self,
        features: int,
        activation: nn.Module,
        norm_layer: str | None = None,
        align_corners: bool = True,
        n_head: int = 16,
    ):
        super().__init__()
        self.align_corners = align_corners

        self.out_conv = nn.Conv2d(features, features, kernel_size=1, stride=1, padding=0, bias=True)
        self.res_conv_unit1 = ResidualConvUnit(features, activation, norm_layer)
        self.res_conv_unit2 = ResidualConvUnit(features, activation, norm_layer)
        self.attn = NeighborhoodAttention2D(
            embed_dim=features, kernel_size=7, stride=1, dilation=2, num_heads=max(n_head, features // 32)
        )
        self.attn_norm = create_norm_layer(norm_layer, features)

    def forward(self, *xs: torch.Tensor, size: tuple[int, int] | None = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            *xs: Variable number of input tensors. First is the main path,
                 second (if present) is added via residual connection.
            size: Target output size. If None, upsamples by 2x.

        Returns:
            Fused and upsampled feature tensor.
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.res_conv_unit1(xs[1])
            output = output + res

        output = self.res_conv_unit2(output)

        nat_inp = self.attn_norm(output).permute(0, 2, 3, 1)
        output = output + self.attn(nat_inp).permute(0, -1, 1, 2)

        if size is None:
            output = F.interpolate(output, scale_factor=2, mode="bilinear", align_corners=self.align_corners)
        else:
            output = F.interpolate(output, size=size, mode="bilinear", align_corners=self.align_corners)

        output = self.out_conv(output)
        return output


def _make_scratch(in_channels: list[int], out_features: int, groups: int = 1) -> nn.Module:
    """Create scratch conv layers for projecting features to common dimension."""
    scratch = nn.Module()

    for i, in_ch in enumerate(in_channels):
        conv = nn.Conv2d(in_ch, out_features, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
        setattr(scratch, f"layer{i + 1}_rn", conv)

    return scratch


class HybridDPTHead(nn.Module):
    """
    DPT Head for Hybrid CNN-Transformer backbone.

    This head fuses features from multiple scales, typically including:
    - 2 CNN features (low-level, high resolution)
    - 2 Transformer features (semantic, lower resolution)

    The features are projected to a common dimension, resized to compatible
    resolutions, and then fused using RefineNet-style blocks.

    Args:
        num_classes: Number of output segmentation classes.
        in_channels: List of input channel dimensions for each feature level.
                     E.g., [128, 256, 1152, 1152] for 2 CNN + 2 ViT features.
        feature_dim: Common feature dimension for fusion (default: 256).
        use_bn: Whether to use batch normalization in fusion blocks.
        align_corners: Alignment mode for interpolation.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: list[int],
        feature_dim: int = 256,
        norm_layer: str | None = None,
        align_corners: bool = True,
        n_head: int = 16,
        n_blocks: int = 1,
    ):
        super().__init__()

        self.num_levels = len(in_channels)
        self.feature_dim = feature_dim
        self.align_corners = align_corners
        self.n_blocks = n_blocks

        # Project all input features to common dimension
        self.projects = nn.ModuleList(
            [nn.Conv2d(in_ch, feature_dim, kernel_size=1, stride=1, padding=0) for in_ch in in_channels]
        )

        # Scratch layers for additional refinement
        self.scratch = _make_scratch([feature_dim] * self.num_levels, feature_dim)

        # RefineNet fusion blocks (from deepest to shallowest)
        # Each level has n_blocks FeatureFusionBlocks
        self.refinenets = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        FeatureFusionBlock(feature_dim, nn.ReLU(False), norm_layer, align_corners, n_head)
                        for _ in range(n_blocks)
                    ]
                )
                for _ in range(self.num_levels)
            ]
        )

        # Output head
        self.output_conv = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=align_corners),
            nn.Conv2d(feature_dim, num_classes, kernel_size=1, stride=1, padding=0),
        )

        logger.info(
            f"[HybridDPTHead] Created with {self.num_levels} levels, "
            f"in_channels={in_channels}, feature_dim={feature_dim}"
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: List of feature tensors from different scales.
                      Expected order: [cnn_feat1, cnn_feat2, vit_feat1, vit_feat2]
                      where cnn features are higher resolution.

        Returns:
            Segmentation logits tensor of shape (B, num_classes, H, W).
        """
        assert len(features) == self.num_levels, f"Expected {self.num_levels} features, got {len(features)}"

        # 1. Project all features to common dimension
        projected = []
        for i, (feat, proj) in enumerate(zip(features, self.projects)):
            projected.append(proj(feat))

        # 2. Apply scratch conv layers
        refined = []
        for i, feat in enumerate(projected):
            scratch_conv = getattr(self.scratch, f"layer{i + 1}_rn")
            refined.append(scratch_conv(feat))

        # 3. Fuse features from deep to shallow using refinenets
        # Start from the deepest (last) feature
        level_blocks = self.refinenets[-1]
        path = level_blocks[0](refined[-1], size=refined[-2].shape[2:])
        for block in level_blocks[1:]:
            path = block(path, size=path.shape[2:])

        # Progressively fuse with shallower features
        for i in range(self.num_levels - 2, 0, -1):
            target_size = refined[i - 1].shape[2:]
            level_blocks = self.refinenets[i]
            path = level_blocks[0](path, refined[i], size=target_size)
            for block in level_blocks[1:]:
                path = block(path, size=path.shape[2:])

        # Final fusion with shallowest feature
        level_blocks = self.refinenets[0]
        path = level_blocks[0](path, refined[0])
        for block in level_blocks[1:]:
            path = block(path)

        # 4. Output head
        out = self.output_conv(path)

        return out


class DPTSegmentationHead(nn.Module):
    """
    Simplified DPT segmentation head with configurable input feature sizes.

    This version handles features of different resolutions by first resizing
    them to a common resolution before fusion.

    Args:
        num_classes: Number of output segmentation classes.
        in_channels: List of input channel dimensions.
        feature_dim: Common feature dimension for fusion.
        target_scale: Scale factor relative to input resolution for output.
                      E.g., 0.25 means output is 1/4 of input size.
        use_bn: Whether to use batch normalization.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: list[int],
        feature_dim: int = 256,
        target_scale: float = 0.25,
        norm_layer: str | None = None,
    ):
        super().__init__()

        self.num_levels = len(in_channels)
        self.feature_dim = feature_dim
        self.target_scale = target_scale

        # Project and resize layers for each input
        self.adapters = nn.ModuleList()
        for in_ch in in_channels:
            adapter = nn.Sequential(
                nn.Conv2d(in_ch, feature_dim, kernel_size=1, bias=False),
                create_norm_layer(norm_layer, feature_dim),
                nn.ReLU(inplace=True),
            )
            self.adapters.append(adapter)

        # Fusion via concatenation followed by conv
        self.fusion = nn.Sequential(
            nn.Conv2d(feature_dim * self.num_levels, feature_dim, kernel_size=3, padding=1, bias=False),
            create_norm_layer(norm_layer, feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, bias=False),
            create_norm_layer(norm_layer, feature_dim),
            nn.ReLU(inplace=True),
        )

        # Segmentation head
        self.seg_head = nn.Conv2d(feature_dim, num_classes, kernel_size=1)

        logger.info(
            f"[DPTSegmentationHead] Created with {self.num_levels} levels, "
            f"in_channels={in_channels}, feature_dim={feature_dim}"
        )

    def forward(
        self,
        features: list[torch.Tensor],
        target_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: List of feature tensors from different scales.
            target_size: Optional target output size. If None, uses target_scale.

        Returns:
            Segmentation logits tensor.
        """
        # Determine target resolution
        if target_size is None:
            # Use the largest feature map size
            max_h = max(f.shape[2] for f in features)
            max_w = max(f.shape[3] for f in features)
            target_size = (max_h, max_w)

        # Adapt and resize all features
        adapted = []
        for feat, adapter in zip(features, self.adapters):
            x = adapter(feat)
            if x.shape[2:] != target_size:
                x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
            adapted.append(x)

        # Concatenate and fuse
        fused = torch.cat(adapted, dim=1)
        fused = self.fusion(fused)

        # Segment
        out = self.seg_head(fused)

        return out


if __name__ == "__main__":
    # Test HybridDPTHead
    batch_size = 2
    h, w = 64, 64

    # Simulating 4 features: 2 CNN (higher res) + 2 ViT (lower res)
    # CNN features: 256x64x64, 512x32x32
    # ViT features: 1152x16x16, 1152x16x16
    features = [
        torch.randn(batch_size, 256, h, w),
        torch.randn(batch_size, 512, h // 2, w // 2),
        torch.randn(batch_size, 1152, h // 4, w // 4),
        torch.randn(batch_size, 1152, h // 4, w // 4),
    ]

    in_channels = [256, 512, 1152, 1152]
    num_classes = 24

    head = HybridDPTHead(num_classes=num_classes, in_channels=in_channels, feature_dim=256, norm_layer="layernorm2d")
    out = head(features)
    print(f"[HybridDPTHead] Output shape: {out.shape}")  # Expected: (2, 24, 128, 128)

    # Test DPTSegmentationHead
    simple_head = DPTSegmentationHead(
        num_classes=num_classes, in_channels=in_channels, feature_dim=256, norm_layer="layernorm2d"
    )
    out2 = simple_head(features)
    print(f"[DPTSegmentationHead] Output shape: {out2.shape}")
