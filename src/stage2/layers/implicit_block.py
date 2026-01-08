from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import LayerNorm2d


class ImplicitQueryDecoder(nn.Module):
    def __init__(
        self,
        feature_dims: list[int] | tuple[int, ...] = (256, 512, 1024),
        embed_dim: int = 256,
        mlp_mid_dim: int = 128,
        num_fusion_blocks: int = 1,
        fusion_block_type: str = "default",
        fusion_block_kwargs: dict | None = None,
    ) -> None:
        super().__init__()

        if fusion_block_kwargs is None:
            fusion_block_kwargs = {}

        # 1) Reassemble blocks (simplified): project multi-level features to a shared dimension.
        self.projections = nn.ModuleList([nn.Conv2d(in_dim, embed_dim, kernel_size=1) for in_dim in feature_dims])

        # 2) Fusion blocks: each scale has `num_fusion_blocks` stacked blocks.
        self.fusion_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        FusionBlock(embed_dim, fusion_block_type, **fusion_block_kwargs)
                        for _ in range(num_fusion_blocks)
                    ]
                )
                for _ in range(len(feature_dims) - 1)
            ]
        )

        # 3) MLP head: decode fused tokens to a 1-channel prediction.
        self.mlp_head = nn.Sequential(
            nn.Conv2d(embed_dim, mlp_mid_dim, 1),
            nn.ReLU(),
            nn.Conv2d(mlp_mid_dim, 1, 1),
            nn.ReLU(),
        )

    @staticmethod
    def get_query_coords(batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Create normalized query grid in [-1, 1] with shape (B, H, W, 2)."""

        y_range = torch.linspace(-1.0, 1.0, height, device=device)
        x_range = torch.linspace(-1.0, 1.0, width, device=device)
        y, x = torch.meshgrid(y_range, x_range, indexing="ij")

        coords = torch.stack([x, y], dim=-1)
        return coords.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    def forward(self, features: list[torch.Tensor], target_h: int, target_w: int) -> torch.Tensor:
        """Query multi-scale features at arbitrary resolution via bilinear sampling."""

        batch_size = features[0].shape[0]
        device = features[0].device

        # A) Reassemble: project all features to `embed_dim`.
        proj_feats = [proj(feature) for proj, feature in zip(self.projections, features, strict=False)]

        # B) Infinite-resolution query: build coordinate grid.
        query_coords = self.get_query_coords(batch_size, target_h, target_w, device)

        # C) Feature query: sample each level with `grid_sample`.
        sampled_tokens: list[torch.Tensor] = []
        for feat in proj_feats:
            sampled = F.grid_sample(feat, query_coords, mode="bilinear", align_corners=False)
            sampled_tokens.append(sampled)

        # D) Hierarchical fusion: each scale may have multiple stacked fusion blocks.
        hidden = sampled_tokens[0]
        for fusion_stack, next_feat in zip(self.fusion_blocks, sampled_tokens[1:], strict=False):
            for block in fusion_stack:  # type: ignore
                hidden = block(hidden, next_feat)

        # E) Decode.
        depth = self.mlp_head(hidden)
        return depth


class LIIFBlock(nn.Module):
    """LIIF-style implicit decoder operating on a feature map.

    This module implements a practical variant of LIIF (Local Implicit Image Function):
    it queries a 2D feature map at arbitrary continuous coordinates, mixes in a
    relative-coordinate encoding, and predicts per-query outputs with an MLP.

    Expected usage:
    - Input feature: (B, C, H, W)
    - Query coords: normalized grid (B, H_out, W_out, 2) in [-1, 1]
    - Output: (B, out_dim, H_out, W_out)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 256,
        num_hidden_layers: int = 2,
        *,
        local_ensemble: bool = True,
        feat_unfold: bool = False,
        cell_decode: bool = True,
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        mlp_in_dim = in_dim
        if self.feat_unfold:
            mlp_in_dim *= 9

        # We concatenate: sampled feature + relative coordinate (dx, dy) [+ cell size].
        mlp_in_dim += 2
        if self.cell_decode:
            mlp_in_dim += 2

        self.imnet = self._build_mlp(mlp_in_dim, out_dim, hidden_dim, num_hidden_layers)

    @staticmethod
    def _build_mlp(in_dim: int, out_dim: int, hidden_dim: int, num_hidden_layers: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        current_dim = in_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.GELU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, out_dim))
        return nn.Sequential(*layers)

    @staticmethod
    def _make_base_grid(feature: torch.Tensor) -> torch.Tensor:
        """Return base coord grid for feature centers, shape (B, H, W, 2)."""

        batch_size, _, height, width = feature.shape
        device = feature.device

        y = torch.linspace(-1.0, 1.0, height, device=device)
        x = torch.linspace(-1.0, 1.0, width, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        base = torch.stack([xx, yy], dim=-1)
        return base.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    @staticmethod
    def _unfold_feature(feature: torch.Tensor) -> torch.Tensor:
        """Unfold 3x3 neighborhood so each location has 9*C channels."""

        batch_size, channels, height, width = feature.shape
        unfolded = F.unfold(feature, kernel_size=3, padding=1)
        return unfolded.view(batch_size, channels * 9, height, width)

    @staticmethod
    def _grid_sample(feature: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Sample feature at coords. Return (B, H_out, W_out, C).

        Note: LIIF typically uses 'nearest' sampling to verify specific grid features,
        but 'bilinear' can also work as a variant. Here we stick to nearest for standard LIIF behavior.
        """
        sampled = F.grid_sample(feature, coords, mode="nearest", align_corners=False)
        return sampled.permute(0, 2, 3, 1)

    @staticmethod
    def _grid_sample_coords(base_coords: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Sample base coordinates at query points. Return (B, H_out, W_out, 2)."""
        sampled = F.grid_sample(base_coords.permute(0, 3, 1, 2), coords, mode="nearest", align_corners=False)
        return sampled.permute(0, 2, 3, 1)

    @staticmethod
    def _expand_cell(cell: torch.Tensor | None, coords: torch.Tensor) -> torch.Tensor | None:
        if cell is None:
            return None
        if cell.ndim == 2:
            return cell[:, None, None, :].expand(coords.shape[0], coords.shape[1], coords.shape[2], 2)
        if cell.ndim == 4:
            return cell
        msg = f"Unsupported cell shape: {tuple(cell.shape)}"
        raise ValueError(msg)

    @staticmethod
    def make_coords(batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Generate normalized query grid for LIIF, shape (B, H, W, 2), range [-1, 1].

        Args:
            batch_size: Batch size.
            height: Output height.
            width: Output width.
            device: Tensor device.

        Returns:
            Coordinate grid of shape (B, H, W, 2) with values in [-1, 1].
        """

        y = torch.linspace(-1.0, 1.0, height, device=device)
        x = torch.linspace(-1.0, 1.0, width, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack([xx, yy], dim=-1)
        return coords.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    def _query_once(
        self,
        feat: torch.Tensor,
        base_coords: torch.Tensor,
        coords: torch.Tensor,
        cell: torch.Tensor | None,
        original_coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Query implicit function once (no local ensemble), return (B, H_out, W_out, out_dim)."""

        if original_coords is None:
            original_coords = coords

        # 1. Sample feature at 'coords' (which might be shifted in ensemble)
        feat_sample = self._grid_sample(feat, coords)

        # 2. Identify the nearest grid center for 'coords'
        q_base = self._grid_sample_coords(base_coords, coords)

        # 3. Calculate relative coordinate from the GRID CENTER to the ORIGINAL QUERY point
        # Critical Fix: Must use 'original_coords' here, not 'shifted coords'.
        rel_coord = original_coords - q_base

        inputs = [feat_sample, rel_coord]
        if self.cell_decode:
            expanded_cell = self._expand_cell(cell, coords)
            if expanded_cell is None:
                msg = "cell is required when cell_decode=True"
                raise ValueError(msg)
            inputs.append(expanded_cell)

        mlp_in = torch.cat(inputs, dim=-1)
        flat = mlp_in.view(-1, mlp_in.shape[-1])
        pred = self.imnet(flat).view(coords.shape[0], coords.shape[1], coords.shape[2], self.out_dim)
        return pred

    def forward(
        self,
        feature: torch.Tensor,
        target_hw: tuple[int, int] | None = None,
        *,
        coords: torch.Tensor | None = None,
        cell: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict values at query coordinates.

        Args:
            feature: Feature map (B, C, H, W).
            target_hw: Target output size (H, W). Used only when `coords` is None.
            coords: Query coords (B, H_out, W_out, 2) in [-1, 1]. If None, `target_hw` must be provided.
            cell: Optional cell size. Typical LIIF uses (B, 2) where values are
                normalized cell size in coord space.

        Returns:
            Prediction tensor (B, out_dim, H_out, W_out).

        Raises:
            ValueError: If both `coords` and `target_hw` are None, or if `target_hw` is None when `coords` is None.
        """

        if coords is None:
            if target_hw is None:
                msg = "Either `coords` or `target_hw` must be provided"
                raise ValueError(msg)
            batch_size = feature.shape[0]
            device = feature.device
            coords = self.make_coords(batch_size, target_hw[0], target_hw[1], device)

        assert coords is not None  # Type narrowing for mypy

        if self.feat_unfold:
            feature = self._unfold_feature(feature)

        base_coords = self._make_base_grid(feature)

        if not self.local_ensemble:
            out = self._query_once(feature, base_coords, coords, cell)
            return out.permute(0, 3, 1, 2)

        # Local ensemble: evaluate 4 neighbors with slight coordinate shifts.
        _, _, height, width = feature.shape
        rx = 1.0 / max(width, 1)
        ry = 1.0 / max(height, 1)

        vx = (-rx, rx)
        vy = (-ry, ry)

        preds: list[torch.Tensor] = []
        areas: list[torch.Tensor] = []

        for dx in vx:
            for dy in vy:
                shifted = coords.clone()
                shifted[..., 0] = (shifted[..., 0] + dx).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
                shifted[..., 1] = (shifted[..., 1] + dy).clamp(-1.0 + 1e-6, 1.0 - 1e-6)

                # Pass original 'coords' to correctly calculate rel_coord
                pred = self._query_once(feature, base_coords, shifted, cell, original_coords=coords)
                preds.append(pred)

                # Weighting area should also be based on distance from original coords to the center
                q_base = self._grid_sample_coords(base_coords, shifted)
                rel = coords - q_base  # Fix: use coords - q_base
                area = (rel[..., 0].abs() * rel[..., 1].abs()).clamp_min(1e-9)
                areas.append(area)

        total = torch.zeros_like(preds[0])
        area_sum = torch.zeros_like(areas[0])
        for pred, area in zip(preds, areas, strict=False):
            total = total + pred * area[..., None]
            area_sum = area_sum + area

        out = total / area_sum[..., None]
        return out.permute(0, 3, 1, 2)


class FusionBlock(nn.Module):
    """Residual gated fusion block.

    Implements: h_{k+1} = FFN(f_{k+1} + gate * Linear(h_k)).
    The FFN can be configured via `block_type` ("default", "mbconv", "nat").
    Additional kwargs are passed to the underlying block constructor.
    """

    def __init__(self, dim: int, block_type: str = "default", **block_kwargs) -> None:
        super().__init__()
        self.linear = nn.Conv2d(dim, dim, 1)
        self.gate = nn.Parameter(torch.zeros(1))
        self.norm = LayerNorm2d(dim)
        self.ffn = self._build_ffn(dim, block_type, **block_kwargs)
        self.block_type = block_type
        self.dim = dim

        # Optional: Conditional convolution
        # We check if 'cond_dim' is passed in block_kwargs to instantiate cond_conv
        cond_dim = block_kwargs.get("cond_dim", None)
        if cond_dim is not None:
            # cond is expected to be (B, C_cond, H, W), we use Conv2d 1x1 to project it to dim
            self.cond_proj = nn.Conv2d(cond_dim, dim, 1)

    @staticmethod
    def _build_ffn(dim: int, block_type: str, **block_kwargs) -> nn.Module:
        if block_type == "default":
            # Channel-first MLP = 1x1 Convs
            return nn.Sequential(nn.Conv2d(dim, dim, 1), nn.GELU(), nn.Conv2d(dim, dim, 1))
        if block_type == "mbconv":
            from .conv import FusedMBConv

            return FusedMBConv(
                dim,
                dim,
                kernel_size=block_kwargs.get("kernel_size", 3),
                stride=1,
                mid_channels=block_kwargs.get("mid_channels", None),
                expand_ratio=block_kwargs.get("expand_ratio", 4),
                use_bias=False,
                norm=block_kwargs.get("norm", ("layernorm2d", "layernorm2d")),
                act_func=block_kwargs.get("act_func", ("gelu", "silu")),
            )
        if block_type == "nat":
            from .blocks import Spatial2DNATBlock

            return Spatial2DNATBlock(
                dim=dim,
                k_size=block_kwargs.get("k_size", 8),
                stride=1,
                dilation=block_kwargs.get("dilation", 2),
                n_heads=block_kwargs.get("n_heads", 8),
                ffn_ratio=block_kwargs.get("ffn_ratio", 2.0),
                qkv_bias=block_kwargs.get("qkv_bias", True),
                qk_norm=block_kwargs.get("qk_norm", "layernorm2d"),
            )
        msg = f"Unsupported block_type: {block_type}"
        raise ValueError(msg)

    def forward(self, h_k: torch.Tensor, f_next: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        # h_k, f_next, cond are all (B, C, H, W)
        proj_h = self.linear(h_k)
        fused = f_next + self.gate * proj_h

        # Add condition if provided and projection layer exists
        if cond is not None and hasattr(self, "cond_proj"):
            fused = fused + self.cond_proj(cond)

        normed = self.norm(fused)
        out = self.ffn(normed)

        return out + fused
