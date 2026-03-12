"""
Self-Flow Model.

This module contains the SelfFlowPerTokenDiT model, a Diffusion Transformer
with per-token timestep conditioning for Self-Flow training.
"""

import collections.abc
import math
from itertools import repeat

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from timm.models.vision_transformer import Attention, Mlp
from timm.layers.pos_embed import resample_abs_pos_embed

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper, CheckpointImpl


def wrap_module_checkpoint(model: nn.Module):
    """Wrap module in activation checkpoint"""
    return CheckpointWrapper(model, CheckpointImpl.NO_REENTRANT)


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


class PatchedPatchEmbed(nn.Module):
    """Simplified Sequence to Patch Embedding using Linear layer."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        bias: bool = True,
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        img_size = to_2tuple(img_size)

        self.grid_size = (
            img_size[0] // self.patch_size[0],
            img_size[1] // self.patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        patch_dim = self.patch_size[0] * self.patch_size[1] * in_chans
        self.proj = nn.Linear(patch_dim, embed_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=self.patch_size[0], p2=self.patch_size[1])
        x = self.proj(x)
        return x


def modulate(x, shift, scale):
    """Standard modulation with unsqueeze for (N, D) conditioning."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def modulate_per_token(x, shift, scale):
    """Per-token modulation for (N, T, D) conditioning."""
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations."""

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """Drops labels to enable classifier-free guidance."""
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class DiTBlock(nn.Module):
    """A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=nn.GELU,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class PerTokenDiTBlock(DiTBlock):
    """DiT block that handles per-token conditioning (N, T, D) instead of (N, D)."""

    def forward(self, x, c):
        """
        Args:
            x: (N, T, D) tokens
            c: (N, T, D) per-token conditioning
        """
        batch_size, seq_len, hidden_dim = c.shape
        c_flat = c.reshape(-1, hidden_dim)
        modulation_flat = self.adaLN_modulation(c_flat)
        modulation = modulation_flat.reshape(batch_size, seq_len, -1)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)

        x = x + gate_msa * self.attn(modulate_per_token(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate_per_token(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """The final layer of DiT."""

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class PerTokenFinalLayer(FinalLayer):
    """Final layer that handles per-token conditioning (N, T, D) instead of (N, D)."""

    def forward(self, x, c):
        """
        Args:
            x: (N, T, D) tokens
            c: (N, T, D) per-token conditioning
        """
        batch_size, seq_len, hidden_dim = c.shape
        c_flat = c.reshape(-1, hidden_dim)
        modulation_flat = self.adaLN_modulation(c_flat)
        modulation = modulation_flat.reshape(batch_size, seq_len, -1)

        shift, scale = modulation.chunk(2, dim=-1)
        x = modulate_per_token(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SimpleHead(nn.Module):
    """Simple projection head for self-distillation."""

    def __init__(self, in_dim, out_dim):
        super(SimpleHead, self).__init__()
        self.linear1 = nn.Linear(in_dim, in_dim + out_dim)
        self.linear2 = nn.Linear(in_dim + out_dim, out_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(self.act(x))
        return x


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """Generate 2D sinusoidal positional embeddings."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


class SelfFlowDiT(nn.Module):
    """
    Base Self-Flow DiT model.
    """

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        num_classes=1000,
        learn_sigma=False,
        has_condition: bool = False,
        compatibility_mode=False,
        distill_head=False,
        act_checkpoint=False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.has_condition = has_condition
        self.has_distill_head = distill_head
        self.compatibility_mode = compatibility_mode
        self.act_checkpoint = act_checkpoint

        self.x_embedder = PatchedPatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, dropout_prob=0.0) if has_condition else None

        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList()
        for _ in range(depth):
            block = DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            if act_checkpoint:
                block = wrap_module_checkpoint(block)
            self.blocks.append(block)

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        # Self-distillation projector
        self.projector = SimpleHead(hidden_size, hidden_size) if distill_head else None

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        if self.y_embedder is not None:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        t_mlp0 = self.t_embedder.mlp[0]
        t_mlp2 = self.t_embedder.mlp[2]
        assert isinstance(t_mlp0, nn.Linear) and isinstance(t_mlp2, nn.Linear)
        nn.init.normal_(t_mlp0.weight, std=0.02)
        nn.init.normal_(t_mlp2.weight, std=0.02)

        for block in self.blocks:
            mod_last = list(block.adaLN_modulation.children())[-1]
            assert isinstance(mod_last, nn.Linear)
            nn.init.constant_(mod_last.weight, 0)
            nn.init.constant_(mod_last.bias, 0)

        final_mod_last = self.final_layer.adaLN_modulation[-1]
        assert isinstance(final_mod_last, nn.Linear)
        nn.init.constant_(final_mod_last.weight, 0)
        nn.init.constant_(final_mod_last.bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def shufflechannel(self, x, hw):
        """Reorder channels/patches to match expected output format."""
        p = self.x_embedder.patch_size[0]
        # gh, gw = self.x_embedder.grid_size  # token grid
        gh, gw = hw[0] // p, hw[1] // p
        x = rearrange(
            x,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=gh,
            w=gw,
            p1=p,
            p2=p,
            c=self.out_channels,
        )
        if self.learn_sigma:
            x, _ = x.chunk(2, dim=1)
        return x

    def _resampled_pos_embed(self, hw: tuple[int, int]) -> torch.Tensor:
        """Return absolute positional embedding matching current token grid."""
        token_grid_h = hw[0] // self.patch_size
        token_grid_w = hw[1] // self.patch_size
        new_size = (token_grid_h, token_grid_w)
        old_size = self.x_embedder.grid_size

        if new_size == old_size:
            return self.pos_embed

        return resample_abs_pos_embed(
            self.pos_embed,
            new_size=new_size,
            old_size=old_size,
            num_prefix_tokens=0,
        )

    def _forward(self, x, t, y, return_features=False, return_raw_features=False):
        """forward pass."""
        assert not (return_raw_features and return_features)

        hw = x.shape[-2:]
        x = self.x_embedder(x)
        pos_embed = self._resampled_pos_embed(hw)
        x = x + pos_embed
        t = self.t_embedder(t)
        if self.has_condition:
            if y is None:
                raise ValueError("`y` (vector) must be provided when has_condition=True.")
            assert self.y_embedder is not None
            y = self.y_embedder(y, self.training)
            c = t + y
        else:
            c = t

        for i, block in enumerate(self.blocks):
            x = block(x, c)
            if (i + 1) == return_features:
                if self.has_distill_head:
                    assert self.projector is not None
                    zs = self.projector(x)
                else:
                    zs = x
            elif (i + 1) == return_raw_features:
                zs = x

        x = self.final_layer(x, c)

        if return_features or return_raw_features:
            return x, zs
        else:
            return x

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        vector: torch.Tensor | None = None,
        x_ids: torch.Tensor | None = None,
        return_features=False,
        return_raw_features=False,
    ):
        """Forward pass with compatibility mode handling."""
        hw = x.shape[-2:]
        if return_features or return_raw_features:
            x, zs = self._forward(
                x=x,
                t=timesteps,
                y=vector,
                return_features=return_features,
                return_raw_features=return_raw_features,
            )
            x = self.shufflechannel(x, hw=hw)
            return x, zs
        else:
            x = self._forward(x=x, t=timesteps, y=vector, return_features=return_features)
            x = self.shufflechannel(x, hw=hw)
            return x


class SelfFlowPerTokenDiT(SelfFlowDiT):
    """
    Self-Flow DiT with per-token timestep conditioning.

    This is the main model used for Self-Flow inference on ImageNet.
    Key feature: each token can have a different noise level during training.
    """

    def __init__(self, **kwargs):
        # Initialize parent class (creates standard DiTBlocks)
        super().__init__(**kwargs)

        hidden_size = kwargs["hidden_size"]
        num_heads = kwargs["num_heads"]
        mlp_ratio = kwargs["mlp_ratio"]
        patch_size = kwargs["patch_size"]
        in_channels = kwargs["in_channels"]
        learn_sigma = kwargs["learn_sigma"]

        out_channels = in_channels * 2 if learn_sigma else in_channels

        # Convert standard blocks to per-token versions, preserving weights
        self._convert_to_per_token_blocks(hidden_size, num_heads, mlp_ratio, patch_size, out_channels)

    def _convert_to_per_token_blocks(self, hidden_size, num_heads, mlp_ratio, patch_size, out_channels):
        """Convert DiTBlocks to PerTokenDiTBlocks while preserving weights."""
        new_blocks = nn.ModuleList()
        for original_block in self.blocks:
            new_block = PerTokenDiTBlock(hidden_size, num_heads, mlp_ratio)
            new_block.load_state_dict(original_block.state_dict())
            new_blocks.append(new_block)
        self.blocks = new_blocks

        original_final = self.final_layer
        new_final = PerTokenFinalLayer(hidden_size, patch_size, out_channels)
        new_final.load_state_dict(original_final.state_dict())
        self.final_layer = new_final

    def _forward(self, x, t, y, return_features=False, return_raw_features=False):
        """Forward with per-token timestep conditioning."""
        assert not (return_raw_features and return_features)

        hw = x.shape[-2:]
        x = self.x_embedder(x) + self._resampled_pos_embed(hw)
        batch_size, seq_len, hidden_dim = x.shape

        # Handle timestep embedding - per-token or broadcast
        if t.ndim == 1:
            t_emb = self.t_embedder(t).unsqueeze(1).expand(-1, seq_len, -1)
        elif t.ndim == 2:
            t_flat = t.reshape(-1)
            t_emb_flat = self.t_embedder(t_flat)
            t_emb = t_emb_flat.reshape(batch_size, seq_len, -1)
        else:
            raise ValueError(f"Timesteps must be 1D or 2D, got shape {t.shape}")

        if self.has_condition:
            if y is None:
                raise ValueError("`y` (vector) must be provided when has_condition=True.")
            assert self.y_embedder is not None
            y_emb = self.y_embedder(y, self.training).unsqueeze(1).expand(-1, seq_len, -1)
            c = t_emb + y_emb
        else:
            c = t_emb

        # Apply per-token blocks
        for i, block in enumerate(self.blocks):
            x = block(x, c)
            if (i + 1) == return_features:
                assert self.projector is not None
                zs = self.projector(x)
            elif (i + 1) == return_raw_features:
                zs = x

        # Apply per-token final layer
        x = self.final_layer(x, c)

        if return_features or return_raw_features:
            return x, zs
        else:
            return x
