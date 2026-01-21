"""
UViT - U-shaped Vision Transformer for Cloud Removal

This module implements a U-Net style Vision Transformer that supports:
- Multiple token mixers: Global Attention, NAT (Neighborhood Attention), ResBlock
- FiLM conditioning for time embeddings and image conditions
- 2D feature map processing (B, C, H, W)
- Optional time embedding support
- SwiGLU FFN
"""

from __future__ import annotations

import math
from typing import Any, Literal

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper
from timm.layers import create_norm_layer


# * --- Utilities --- #


def _pick_num_groups(num_channels: int, max_groups: int = 32) -> int:
    for groups in range(min(max_groups, num_channels), 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1


def get_2d_sincos_pos_embed(embed_dim: int, h: int, w: int) -> Tensor:
    """Generate 2D sinusoidal positional embeddings."""
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w = torch.arange(w, dtype=torch.float32)
    grid = torch.stack(torch.meshgrid(grid_h, grid_w, indexing="ij"), dim=0)  # (2, h, w)
    grid = grid.reshape(2, 1, h, w)

    pos_dim = embed_dim // 4
    omega = 1.0 / (10000 ** (torch.arange(0, pos_dim, dtype=torch.float32) / pos_dim))
    out_h = grid[0].flatten()[:, None] * omega[None, :]  # (h*w, pos_dim)
    out_w = grid[1].flatten()[:, None] * omega[None, :]  # (h*w, pos_dim)
    pos_embed = torch.cat(
        [torch.sin(out_h), torch.cos(out_h), torch.sin(out_w), torch.cos(out_w)], dim=-1
    )  # (h*w, embed_dim)
    pos_embed = pos_embed.reshape(h, w, embed_dim).permute(2, 0, 1)  # (embed_dim, h, w)
    return pos_embed


def timestep_embedding(timesteps: Tensor, dim: int, max_period: int = 10000) -> Tensor:
    """Create sinusoidal timestep embeddings. Output: (B, dim)"""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# * --- Token Mixers (2D) --- #


class GlobalAttention2D(nn.Module):
    """Global self-attention on 2D feature maps. Flattens to 1D, applies attention, reshapes back."""

    def __init__(
        self, dim: int, num_heads: int = 8, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        x_flat = einops.rearrange(x, "b c h w -> b (h w) c")

        qkv = self.qkv(x_flat)
        qkv = einops.rearrange(qkv, "b l (three nh hd) -> three b nh l hd", three=3, nh=self.num_heads)
        q, k, v = qkv.unbind(0)

        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        attn = einops.rearrange(attn, "b nh l hd -> b l (nh hd)")

        out = self.proj(attn)
        out = self.proj_drop(out)
        out = einops.rearrange(out, "b (h w) c -> b c h w", h=h, w=w)
        return out


class SwiGLU2D(nn.Module):
    """SwiGLU FFN for 2D feature maps (B, C, H, W)."""

    def __init__(
        self, in_features: int, hidden_features: int | None = None, out_features: int | None = None, drop: float = 0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.w2 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.w3 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class ResBlock2D(nn.Module):
    """Simple 2D Residual Block with two convolutions."""

    def __init__(self, dim: int, dropout: float = 0.0, norm_layer: str = "groupnorm"):
        super().__init__()
        self.norm1 = create_norm_layer(norm_layer, dim)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.norm2 = create_norm_layer(norm_layer, dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return x + h


# * --- FiLM Conditioning --- #


def _zero_init(module: nn.Linear | nn.Conv2d | None) -> None:
    if module is None:
        return
    if module.weight is not None:
        nn.init.zeros_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


class FiLMModulation(nn.Module):
    """FiLM-style modulation: scale and shift."""

    def __init__(self, cond_dim: int, out_dim: int, is_1d: bool = True):
        super().__init__()
        self.is_1d = is_1d
        if is_1d:
            self.proj = nn.Linear(cond_dim, out_dim * 2)
        else:
            self.proj = nn.Conv2d(cond_dim, out_dim * 2, kernel_size=1)
        _zero_init(self.proj)

    def forward(self, cond: Tensor) -> tuple[Tensor, Tensor]:
        params = self.proj(cond)
        if self.is_1d:
            scale, shift = params.chunk(2, dim=-1)
            scale = scale[:, None, None, :]  # (B, 1, 1, C)
            shift = shift[:, None, None, :]
            return scale.permute(0, 3, 1, 2), shift.permute(0, 3, 1, 2)  # (B, C, 1, 1)
        else:
            scale, shift = params.chunk(2, dim=1)
            return scale, shift


# * --- UViT Block (2D) --- #


MixerType = Literal["attention", "nat", "resblock"]


class UViTBlock(nn.Module):
    """
    UViT Block supporting 2D inputs with multiple mixer types.

    Args:
        dim: Number of channels
        mixer_type: Type of token mixer ('attention', 'nat', 'resblock')
        num_heads: Number of attention heads (for attention/nat)
        mlp_ratio: FFN hidden dim ratio
        dropout: Dropout rate
        time_embed_dim: Time embedding dimension (None to disable)
        cond_channels: Image condition channels (None to disable)
        skip: Whether to use skip connection from encoder
        nat_kernel_size: Kernel size for NAT
    """

    def __init__(
        self,
        dim: int,
        mixer_type: MixerType = "attention",
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        norm_layer: str = "groupnorm",
        time_embed_dim: int | None = None,
        cond_channels: int | None = None,
        skip: bool = False,
        nat_kernel_size: int = 7,
    ):
        super().__init__()
        self.dim = dim
        self.mixer_type = mixer_type

        # Normalization
        self.norm1 = create_norm_layer(norm_layer, dim)
        self.norm2 = create_norm_layer(norm_layer, dim)

        # Token mixer
        if mixer_type == "attention":
            self.mixer = GlobalAttention2D(dim, num_heads=num_heads)
        elif mixer_type == "nat":
            try:
                import natten

                self.mixer = natten.NeighborhoodAttention2D(dim, kernel_size=nat_kernel_size, num_heads=num_heads)
            except ImportError:
                raise ImportError("natten is required for NAT mixer. Install with: pip install natten")
        elif mixer_type == "resblock":
            self.mixer = ResBlock2D(dim, dropout=dropout, norm_layer=norm_layer)
        else:
            raise ValueError(f"Unknown mixer_type: {mixer_type}")

        # FFN (SwiGLU)
        hidden_dim = int(dim * mlp_ratio)
        self.ffn = SwiGLU2D(dim, hidden_dim, drop=dropout)

        # FiLM conditioning
        self.time_modulation = FiLMModulation(time_embed_dim, dim, is_1d=True) if time_embed_dim else None
        self.cond_modulation = FiLMModulation(cond_channels, dim, is_1d=False) if cond_channels else None

        # Skip connection (for decoder)
        self.skip_proj = nn.Conv2d(dim * 2, dim, kernel_size=1) if skip else None

    def _apply_modulation(self, x: Tensor, temb: Tensor | None, cond: Tensor | None) -> Tensor:
        """Apply FiLM modulation from time and condition."""
        scale = torch.ones_like(x)
        shift = torch.zeros_like(x)

        if self.time_modulation is not None and temb is not None:
            t_scale, t_shift = self.time_modulation(temb)
            scale = scale + t_scale
            shift = shift + t_shift

        if self.cond_modulation is not None and cond is not None:
            # Resize cond to match x
            if cond.shape[-2:] != x.shape[-2:]:
                cond = F.interpolate(cond, size=x.shape[-2:], mode="bilinear", align_corners=False)
            c_scale, c_shift = self.cond_modulation(cond)
            scale = scale + c_scale
            shift = shift + c_shift

        return x * scale + shift

    def forward(
        self, x: Tensor, temb: Tensor | None = None, cond: Tensor | None = None, skip: Tensor | None = None
    ) -> Tensor:
        # Skip connection from encoder
        if self.skip_proj is not None and skip is not None:
            x = self.skip_proj(torch.cat([x, skip], dim=1))

        # Apply modulation after norm1
        h = self.norm1(x)
        h = self._apply_modulation(h, temb, cond)

        # Token mixer
        if self.mixer_type == "nat":
            # NAT expects (B, H, W, C)
            h = einops.rearrange(h, "b c h w -> b h w c")
            h = self.mixer(h)
            h = einops.rearrange(h, "b h w c -> b c h w")
        else:
            h = self.mixer(h)
        x = x + h

        # FFN with modulation
        h = self.norm2(x)
        h = self._apply_modulation(h, temb, cond)
        h = self.ffn(h)
        x = x + h

        return x


# * --- PatchEmbed (2D output) --- #


class PatchEmbed2D(nn.Module):
    """Image to Patch Embedding. Outputs 2D feature map (B, C, H', W')."""

    def __init__(self, patch_size: int, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        assert h % self.patch_size == 0 and w % self.patch_size == 0, (
            f"Image size ({h}, {w}) must be divisible by patch_size ({self.patch_size})"
        )
        return self.proj(x)  # (B, embed_dim, H', W')


# * --- UViT --- #


class UViT(nn.Module):
    """
    U-shaped Vision Transformer for image-to-image tasks.

    Operates on 2D feature maps throughout, supports multiple mixer types,
    optional time conditioning, and FiLM-style image conditioning.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int | None = None,
        cond_channels: int | None = None,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        patch_size: int = 1,
        dropout: float = 0.0,
        norm_layer: str = "groupnorm",
        mixer_type: MixerType = "attention",
        use_time_embed: bool = True,
        act_checkpoint: bool = False,
        nat_kernel_size: int = 7,
        # NAT specific config
        nat_cfg: dict[str, Any] | None = None,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.use_time_embed = use_time_embed
        self.act_checkpoint = act_checkpoint

        nat_cfg = nat_cfg or {}
        nat_kernel_size = nat_cfg.get("k_size", nat_kernel_size)
        nat_num_heads = nat_cfg.get("num_heads", num_heads)

        # Patch embedding
        self.patch_embed = PatchEmbed2D(patch_size, in_channels, embed_dim)

        # Time embedding
        if use_time_embed:
            time_embed_dim_local = embed_dim * 4
            self.time_embed_dim: int | None = time_embed_dim_local
            self.time_embed: nn.Module | None = nn.Sequential(
                nn.Linear(embed_dim, time_embed_dim_local),
                nn.SiLU(),
                nn.Linear(time_embed_dim_local, time_embed_dim_local),
            )
        else:
            self.time_embed_dim = None
            self.time_embed = None

        # Condition projection for patched cond
        self._cond_channels_patched = cond_channels * (patch_size**2) if cond_channels else None

        # Encoder blocks (in_blocks)
        self.in_blocks = nn.ModuleList()
        for _ in range(depth // 2):
            block = UViTBlock(
                dim=embed_dim,
                mixer_type=mixer_type,
                num_heads=nat_num_heads if mixer_type == "nat" else num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                norm_layer=norm_layer,
                time_embed_dim=self.time_embed_dim,
                cond_channels=self._cond_channels_patched,
                skip=False,
                nat_kernel_size=nat_kernel_size,
            )
            if act_checkpoint:
                block = CheckpointWrapper(block)
            self.in_blocks.append(block)

        # Middle block
        self.mid_block = UViTBlock(
            dim=embed_dim,
            mixer_type=mixer_type,
            num_heads=nat_num_heads if mixer_type == "nat" else num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            norm_layer=norm_layer,
            time_embed_dim=self.time_embed_dim,
            cond_channels=self._cond_channels_patched,
            skip=False,
            nat_kernel_size=nat_kernel_size,
        )
        if act_checkpoint:
            self.mid_block = CheckpointWrapper(self.mid_block)

        # Decoder blocks (out_blocks)
        self.out_blocks = nn.ModuleList()
        for _ in range(depth // 2):
            block = UViTBlock(
                dim=embed_dim,
                mixer_type=mixer_type,
                num_heads=nat_num_heads if mixer_type == "nat" else num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                norm_layer=norm_layer,
                time_embed_dim=self.time_embed_dim,
                cond_channels=self._cond_channels_patched,
                skip=True,
                nat_kernel_size=nat_kernel_size,
            )
            if act_checkpoint:
                block = CheckpointWrapper(block)
            self.out_blocks.append(block)

        # Output
        self.norm_out = create_norm_layer(norm_layer, embed_dim)
        self.conv_out = nn.Conv2d(embed_dim, out_channels * (patch_size**2), kernel_size=3, padding=1)

        # Positional embedding (registered buffer, generated dynamically)
        self.register_buffer("pos_embed", None, persistent=False)
        self._cached_pos_embed_size: tuple[int, int] | None = None

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _get_pos_embed(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Get or generate positional embeddings for given size."""
        if self._cached_pos_embed_size != (h, w) or self.pos_embed is None:
            self.pos_embed = get_2d_sincos_pos_embed(self.embed_dim, h, w).to(device=device, dtype=dtype)
            self._cached_pos_embed_size = (h, w)
        return self.pos_embed

    def _patchify(self, x: Tensor) -> Tensor:
        if self.patch_size == 1:
            return x
        p = self.patch_size
        b, c, h, w = x.shape
        x = x.reshape(b, c, h // p, p, w // p, p)
        x = x.permute(0, 1, 3, 5, 2, 4).reshape(b, c * p * p, h // p, w // p)
        return x

    def _unpatchify(self, x: Tensor, out_channels: int) -> Tensor:
        if self.patch_size == 1:
            return x
        p = self.patch_size
        b, c, h, w = x.shape
        x = x.reshape(b, out_channels, p, p, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).reshape(b, out_channels, h * p, w * p)
        return x

    def forward(
        self,
        x: Tensor,
        t: Tensor | None = None,
        *,
        conditions: Tensor | list[Tensor] | tuple[Tensor, ...] | None = None,
        **kwargs: Any,
    ) -> tuple[Tensor, None, None]:
        """
        Args:
            x: Input tensor (B, C, H, W)
            t: Timesteps (B,) - can be None if use_time_embed=False
            conditions: Conditioning images (B, C', H, W) or list of them
        """
        _ = kwargs  # Ignore extra kwargs for compatibility

        # Time embedding
        temb: Tensor | None = None
        if self.use_time_embed and self.time_embed is not None:
            if t is None:
                raise ValueError("t must be provided when use_time_embed=True")
            temb = self.time_embed(timestep_embedding(t, self.embed_dim))

        # Process conditions
        cond: Tensor | None = None
        if conditions is not None:
            if isinstance(conditions, (list, tuple)):
                cond = torch.cat(list(conditions), dim=1)
            else:
                cond = conditions
            cond = self._patchify(cond)

        # Patchify and embed
        x = self.patch_embed(x)
        b, c, h, w = x.shape

        # Add positional embedding
        pos_embed = self._get_pos_embed(h, w, x.device, x.dtype)
        x = x + pos_embed.unsqueeze(0)

        # Encoder
        skips: list[Tensor] = []
        for blk in self.in_blocks:
            x = blk(x, temb, cond)
            skips.append(x)

        # Middle
        x = self.mid_block(x, temb, cond)

        # Decoder
        for blk in self.out_blocks:
            skip = skips.pop()
            x = blk(x, temb, cond, skip=skip)

        # Output
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        x = self._unpatchify(x, self.out_channels)

        return x, None, None


if __name__ == "__main__":
    # Test 1: Attention mixer with time embed
    print("Test 1: Attention mixer with time embed")
    model = UViT(
        in_channels=4,
        cond_channels=4,
        embed_dim=128,
        depth=4,
        num_heads=8,
        mixer_type="attention",
        use_time_embed=True,
    )
    x = torch.randn(1, 4, 64, 64)
    t = torch.tensor([0.5])
    cond = torch.randn(1, 4, 64, 64)
    out, _, _ = model(x, t, conditions=cond)
    print(f"Output shape: {out.shape}")

    # Test 2: ResBlock mixer without time embed
    print("\nTest 2: ResBlock mixer without time embed")
    model2 = UViT(
        in_channels=4,
        cond_channels=4,
        embed_dim=128,
        depth=4,
        mixer_type="resblock",
        use_time_embed=False,
    )
    out2, _, _ = model2(x, t=None, conditions=cond)
    print(f"Output shape: {out2.shape}")

    print("\nAll tests passed!")
