from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper
from timm.layers import create_norm_layer
from loguru import logger

from src.stage2.layers.blocks import build_spatial_block

logger = logger.bind(_name_="UNet")


def _timestep_embedding(timesteps: torch.Tensor, dim: int, *, max_period: int = 10000) -> torch.Tensor:
    if timesteps.ndim != 1:
        raise ValueError(f"expected timesteps with shape (B,), got {tuple(timesteps.shape)}")
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def _pick_num_groups(num_channels: int, max_groups: int) -> int:
    for groups in range(min(max_groups, num_channels), 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1


class TimeEmbedder(nn.Module):
    def __init__(self, time_in_dim: int, time_embed_dim: int) -> None:
        super().__init__()
        self.time_in_dim = time_in_dim
        self.mlp = nn.Sequential(
            nn.Linear(time_in_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        # timesteps = timesteps * 1000  # rescale to 1000
        emb = _timestep_embedding(timesteps, self.time_in_dim)
        return self.mlp(emb)


class AttentionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        num_heads: int,
        head_dim: int | None,
        num_groups: int,
    ) -> None:
        super().__init__()
        if head_dim is None:
            if channels % num_heads != 0:
                raise ValueError("channels must be divisible by num_heads when head_dim is None")
            head_dim = channels // num_heads
        else:
            if channels % head_dim != 0:
                raise ValueError("channels must be divisible by head_dim")
            num_heads = channels // head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.norm = create_norm_layer("layernorm2dfp32", channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        residual = x
        x = self.norm(x)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)  # b,nh,l,hd
        k = k.reshape(b, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)
        v = v.reshape(b, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)

        q = q.reshape(b * self.num_heads, h * w, self.head_dim)
        k = k.reshape(b * self.num_heads, h * w, self.head_dim)
        v = v.reshape(b * self.num_heads, h * w, self.head_dim)

        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        attn = attn.reshape(b, self.num_heads, h * w, self.head_dim).permute(0, 1, 3, 2)
        attn = attn.reshape(b, c, h, w)
        attn = self.proj(attn)
        return residual + attn


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear")
        return self.conv(x)


class DownStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int,
        time_embed_dim: int | None,
        cond_channels: int,
        dropout: float,
        num_groups: int,
        use_attention: bool,
        num_attention_heads: int,
        attention_head_dim: int | None,
        add_downsample: bool,
        block_type: str = "resblock",
        resblock_cfg: dict[str, Any] | None = None,
        nat_cfg: dict[str, Any] | None = None,
        convnext_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        for block_idx in range(num_res_blocks):
            blocks.append(
                build_spatial_block(
                    block_type=block_type,
                    in_channels=in_channels if block_idx == 0 else out_channels,
                    out_channels=out_channels,
                    cond_channels=cond_channels,
                    use_time_block=True,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    num_groups=num_groups,
                    resblock_cfg=resblock_cfg,
                    nat_cfg=nat_cfg,
                    convnext_cfg=convnext_cfg,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.attn = (
            AttentionBlock(
                out_channels,
                num_heads=num_attention_heads,
                head_dim=attention_head_dim,
                num_groups=num_groups,
            )
            if use_attention
            else None
        )
        self.downsample = Downsample(out_channels) if add_downsample else None

    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor | None,
        cond: torch.Tensor | None,
        skips: list[torch.Tensor],
    ) -> torch.Tensor:
        last_block_idx = len(self.blocks) - 1
        for block_idx, block in enumerate(self.blocks):
            x = block(x, temb, cond)
            if block_idx == last_block_idx and self.attn is not None:
                x = self.attn(x)
            skips.append(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class UpStage(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        num_res_blocks: int,
        time_embed_dim: int | None,
        cond_channels: int,
        dropout: float,
        num_groups: int,
        use_attention: bool,
        num_attention_heads: int,
        attention_head_dim: int | None,
        add_upsample: bool,
        block_type: str = "resblock",
        resblock_cfg: dict[str, Any] | None = None,
        nat_cfg: dict[str, Any] | None = None,
        convnext_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        current_channels = in_channels
        for _ in range(num_res_blocks):
            blocks.append(
                build_spatial_block(
                    block_type=block_type,
                    in_channels=current_channels + skip_channels,
                    out_channels=out_channels,
                    cond_channels=cond_channels,
                    use_time_block=True,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    num_groups=num_groups,
                    resblock_cfg=resblock_cfg,
                    nat_cfg=nat_cfg,
                    convnext_cfg=convnext_cfg,
                )
            )
            current_channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.attn = (
            AttentionBlock(
                out_channels,
                num_heads=num_attention_heads,
                head_dim=attention_head_dim,
                num_groups=num_groups,
            )
            if use_attention
            else None
        )
        self.upsample = Upsample(out_channels) if add_upsample else None

    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor | None,
        cond: torch.Tensor | None,
        skips: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        if len(skips) != len(self.blocks):
            raise ValueError(f"skip list length must match blocks, got skips={len(skips)} blocks={len(self.blocks)}")
        for block, skip in zip(self.blocks, reversed(skips), strict=True):
            x = torch.cat([x, skip], dim=1)
            x = block(x, temb, cond)
        if self.attn is not None:
            x = self.attn(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class ImageConditionUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        cond_channels: int,
        out_channels: int | None = None,
        base_channels: int = 64,
        patch_size: int = 1,
        channel_mults: tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        num_groups: int = 32,
        time_embed_dim: int | None = None,
        attention_stages: tuple[bool, ...] | None = None,
        num_attention_heads: int = 4,
        attention_head_dim: int | None = None,
        act_checkpoint: bool = False,
        use_time_embed: bool = True,
        block_type: str | list[str] = "resblock",
        resblock_cfg: dict[str, Any] | None = None,
        nat_cfg: dict[str, Any] | None = None,
        convnext_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if not channel_mults:
            raise ValueError("channel_mults must not be empty")
        if patch_size < 1:
            raise ValueError("patch_size must be >= 1")
        if attention_stages is None:
            attention_stages = (False,) * len(channel_mults)
        if len(attention_stages) != len(channel_mults):
            raise ValueError("attention_stages must match length of channel_mults")
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.patch_size = patch_size
        self._patch_area = patch_size * patch_size
        self._in_channels_patched = in_channels * self._patch_area
        self._cond_channels_patched = cond_channels * self._patch_area
        self._out_channels_patched = self.out_channels * self._patch_area
        self.time_embedder: TimeEmbedder | None
        if use_time_embed:
            time_embed_dim = time_embed_dim or base_channels * 4
            self.time_embedder = TimeEmbedder(base_channels, time_embed_dim)
        else:
            time_embed_dim = None
            self.time_embedder = None
        self.conv_in = nn.Conv2d(self._in_channels_patched, base_channels, kernel_size=3, padding=1)

        # Block type per stage
        if isinstance(block_type, str):
            block_types = [block_type] * len(channel_mults)
        else:
            if len(block_type) != len(channel_mults):
                raise ValueError("If block_type is a list, its length must match channel_mults")
            block_types = block_type
        self.block_types = block_types
        logger.info(
            "=" * 60 + f"Unet config: {self.block_types} "
            f"{channel_mults=}, {num_res_blocks=}, {attention_stages=} "
            f"{resblock_cfg=}, {nat_cfg=}, {convnext_cfg=}, {cond_channels=}, "
            f"{time_embed_dim=}, {dropout=}, {num_groups=}, {use_time_embed=} \n" + "=" * 60
        )

        # Down stages
        self.num_res_blocks = num_res_blocks
        down_stages: list[nn.Module] = []
        current_channels = base_channels
        for idx, mult in enumerate(channel_mults):
            stage_out_channels = base_channels * mult
            stage = DownStage(
                in_channels=current_channels,
                out_channels=stage_out_channels,
                num_res_blocks=num_res_blocks,
                time_embed_dim=time_embed_dim,
                cond_channels=self._cond_channels_patched,
                dropout=dropout,
                num_groups=num_groups,
                use_attention=attention_stages[idx],
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                add_downsample=idx < len(channel_mults) - 1,
                block_type=block_types[idx],
                resblock_cfg=resblock_cfg,
                nat_cfg=nat_cfg,
                convnext_cfg=convnext_cfg,
            )
            if act_checkpoint:
                stage = CheckpointWrapper(stage)
            down_stages.append(stage)
            current_channels = stage_out_channels
        self.down_stages = nn.ModuleList(down_stages)

        # Middle blocks - use the last stage's block type
        mid_block_type = block_types[-1]
        self.mid_block1 = build_spatial_block(
            block_type=mid_block_type,
            in_channels=current_channels,
            out_channels=current_channels,
            cond_channels=self._cond_channels_patched,
            use_time_block=True,
            time_embed_dim=time_embed_dim,
            dropout=dropout,
            num_groups=num_groups,
            resblock_cfg=resblock_cfg,
            nat_cfg=nat_cfg,
            convnext_cfg=convnext_cfg,
        )
        self.mid_attn = AttentionBlock(
            current_channels,
            num_heads=num_attention_heads,
            head_dim=attention_head_dim,
            num_groups=num_groups,
        )
        self.mid_block2 = build_spatial_block(
            block_type=mid_block_type,
            in_channels=current_channels,
            out_channels=current_channels,
            cond_channels=self._cond_channels_patched,
            use_time_block=True,
            time_embed_dim=time_embed_dim,
            dropout=dropout,
            num_groups=num_groups,
            resblock_cfg=resblock_cfg,
            nat_cfg=nat_cfg,
            convnext_cfg=convnext_cfg,
        )
        if act_checkpoint:
            self.mid_block1 = CheckpointWrapper(self.mid_block1)
            self.mid_attn = CheckpointWrapper(self.mid_attn)
            self.mid_block2 = CheckpointWrapper(self.mid_block2)

        # Up stages
        up_stages: list[nn.Module] = []
        for idx, mult in enumerate(reversed(channel_mults)):
            stage_channels = base_channels * mult
            stage_attention = attention_stages[len(channel_mults) - 1 - idx]
            stage_block_type = block_types[len(channel_mults) - 1 - idx]
            stage = UpStage(
                in_channels=current_channels,
                skip_channels=stage_channels,
                out_channels=stage_channels,
                num_res_blocks=num_res_blocks,
                time_embed_dim=time_embed_dim,
                cond_channels=self._cond_channels_patched,
                dropout=dropout,
                num_groups=num_groups,
                use_attention=stage_attention,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                add_upsample=idx < len(channel_mults) - 1,
                block_type=stage_block_type,
                resblock_cfg=resblock_cfg,
                nat_cfg=nat_cfg,
                convnext_cfg=convnext_cfg,
            )
            if act_checkpoint:
                stage = CheckpointWrapper(stage)
            up_stages.append(stage)
            current_channels = stage_channels
        self.up_stages = nn.ModuleList(up_stages)

        self.out = nn.Sequential(
            create_norm_layer("layernorm2dfp32", current_channels),
            nn.SiLU(),
            nn.Conv2d(current_channels, self._out_channels_patched, kernel_size=3, padding=1),
        )

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        if self.patch_size == 1:
            return x
        b, c, h, w = x.shape
        p = self.patch_size
        if h % p != 0 or w % p != 0:
            raise ValueError(f"expected H/W divisible by patch_size={p}, got {(h, w)}")
        x = x.reshape(b, c, h // p, p, w // p, p)
        x = x.permute(0, 1, 3, 5, 2, 4).reshape(b, c * p * p, h // p, w // p)
        return x

    def _unpatchify(self, x: torch.Tensor, *, out_channels: int) -> torch.Tensor:
        if self.patch_size == 1:
            return x
        b, c, h, w = x.shape
        p = self.patch_size
        expected = out_channels * p * p
        if c != expected:
            raise ValueError(f"expected channels={expected} after unpatchify, got {c}")
        x = x.reshape(b, out_channels, p, p, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).reshape(b, out_channels, h * p, w * p)
        return x

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None,
        *,
        return_logvar: bool = False,
        uncond: bool = False,
        conditions: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None, None]:
        _ = return_logvar, kwargs
        if x.ndim != 4:
            raise ValueError(f"expected x with shape (B,C,H,W), got {tuple(x.shape)}")
        if self.time_embedder is None:
            temb = None
        else:
            if t is None:
                raise ValueError("t must be provided when time embedding is enabled")
            if t.ndim != 1 or t.shape[0] != x.shape[0]:
                raise ValueError(f"expected t with shape (B,), got {tuple(t.shape)}")
            temb = self.time_embedder(t)

        cond: torch.Tensor | None = None
        if conditions is not None:
            if isinstance(conditions, (list, tuple)):
                if not conditions:
                    raise ValueError("conditions list/tuple must not be empty")
                for cond_item in conditions:
                    if cond_item.ndim != 4:
                        raise ValueError(f"expected conditions with shape (B,C,H,W), got {tuple(cond_item.shape)}")
                cond = torch.cat(list(conditions), dim=1)
            else:
                cond = conditions
            if cond.ndim != 4:
                raise ValueError(f"expected conditions with shape (B,C,H,W), got {tuple(cond.shape)}")
        if uncond:
            cond = None

        if cond is not None and cond.shape[-2:] != x.shape[-2:]:
            raise ValueError(f"conditions spatial size must match x, got {cond.shape[-2:]} vs {x.shape[-2:]}")

        x = self._patchify(x)
        if cond is not None:
            cond = self._patchify(cond)

        h = self.conv_in(x)

        skips: list[torch.Tensor] = []
        for stage in self.down_stages:
            h = stage(h, temb, cond, skips)

        h = self.mid_block1(h, temb, cond)
        h = self.mid_attn(h)
        h = self.mid_block2(h, temb, cond)

        for stage in self.up_stages:
            needed = self.num_res_blocks
            if len(skips) < needed:
                raise ValueError(f"skip list too short for stage: skips={len(skips)} needed={needed}")
            stage_skips = tuple(skips[-needed:])
            skips = skips[:-needed]
            h = stage(h, temb, cond, stage_skips)
        if skips:
            raise ValueError(f"unused skip tensors remain: {len(skips)}")

        h = self.out(h)
        h = self._unpatchify(h, out_channels=self.out_channels)

        return h, None, None


import pytest


@pytest.fixture
def init():
    x = torch.randn(1, 4, 64, 64)
    t = torch.tensor([0.5])
    cond = torch.randn(1, 4, 64, 64)
    return x, t, cond


def test_resblock(init):
    x, t, cond = init
    # Test 1: Default ResBlock with time embedding
    print("Testing ResBlock with time embed...")
    network = ImageConditionUNet(in_channels=4, cond_channels=4, base_channels=32, channel_mults=(1, 2))
    out, _, _ = network(x, t, conditions=cond)
    print(f"ResBlock Output shape: {out.shape}")


def test_natblock(init):
    x, t, cond = init
    # Test 2: NATBlock without time embedding
    print("\nTesting NATBlock without time embed...")
    network_nat = ImageConditionUNet(
        in_channels=4,
        cond_channels=4,
        base_channels=32,
        channel_mults=(1, 2),
        use_time_embed=False,
        block_type="natblock",
        nat_cfg={
            "k_size": 7,
            "num_heads": 4,
        },
    ).cuda()
    out_nat, _, _ = network_nat(x.cuda(), t=None, conditions=cond.cuda())
    print(f"NATBlock Output shape: {out_nat.shape}")


def test_convnext(init):
    x, t, cond = init
    # Test 3: ConvNext with channel attention
    print("\nTesting ConvNext with channel attention...")
    network_convnext = ImageConditionUNet(
        in_channels=4,
        cond_channels=4,
        base_channels=32,
        channel_mults=(1, 2),
        block_type="convnext",
        convnext_cfg={
            "use_channel_attention": True,
        },
    )
    out_convnext, _, _ = network_convnext(x, t, conditions=cond)
    print(f"ConvNext Output shape: {out_convnext.shape}")
