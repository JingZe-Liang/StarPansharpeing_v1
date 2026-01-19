"""
Lightning DiT's codes are built from original DiT & SiT.
(https://github.com/facebookresearch/DiT; https://github.com/willisma/SiT)
It demonstrates that a advanced DiT together with advanced diffusion skills
could also achieve a very promising result with 1.35 FID on ImageNet 256 generation.

Enjoy everyone, DiT strikes back!

by Maple (Jingfeng Yao) from HUST-VL
"""

import math
from typing import cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed, Mlp
from timm.layers import SwiGLU as SwiGLUFFN
from timm.layers import RmsNorm as RMSNorm
from timm.layers import resample_abs_pos_embed

from .pos_embed import VisionRotaryEmbeddingFast


class LazyPatchEmbed2d(nn.Module):
    def __init__(self, *, patch_size: int, embed_dim: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.LazyConv2d(embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected condition image with shape (B,C,H,W), got {tuple(x.shape)}")
        if x.shape[-2] % self.patch_size != 0 or x.shape[-1] % self.patch_size != 0:
            raise ValueError(
                f"condition H/W must be divisible by patch_size={self.patch_size}, got H={x.shape[-2]} W={x.shape[-1]}"
            )
        x = self.proj(x)  # (B, D, H', W')
        return x.flatten(2).transpose(1, 2)  # (B, N, D)


class ApproxGELU(nn.GELU):
    def __init__(self) -> None:
        super().__init__(approximate="tanh")


def modulate(x: torch.Tensor, shift: torch.Tensor | None, scale: torch.Tensor) -> torch.Tensor:
    if scale.ndim == 2:
        scale = scale.unsqueeze(1)
    if shift is None:
        return x * (1 + scale)
    if shift.ndim == 2:
        shift = shift.unsqueeze(1)
    return x * (1 + scale) + shift


class Attention(nn.Module):
    """
    Attention module of LightningDiT.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        fused_attn: bool = True,
        use_rmsnorm: bool = False,
        use_v1_residual: bool = True,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        if use_rmsnorm:
            norm_layer = RMSNorm

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_v1_residual = use_v1_residual
        if self.use_v1_residual:
            self.v1_lambda = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        x: torch.Tensor,
        rope: VisionRotaryEmbeddingFast | None = None,
        rope_ids: torch.Tensor | None = None,
        v1: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        v_out = v
        q, k = self.q_norm(q), self.k_norm(k)

        if v1 is not None:
            if not self.use_v1_residual:
                raise ValueError("v1 residual is disabled for this Attention module")
            v = self.v1_lambda * v1 + (1.0 - self.v1_lambda) * v

        if rope is not None:
            q = rope(q, rope_ids)
            k = rope(k, rope_ids)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, v_out


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    Same as DiT.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        Args:
            t: A 1-D Tensor of N indices, one per batch element. These may be fractional.
            dim: The dimension of the output.
            max_period: Controls the minimum frequency of the embeddings.
        Returns:
            An (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )

        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LightningDiTBlock(nn.Module):
    """
    Lightning DiT Block. We add features including:
    - ROPE
    - QKNorm
    - RMSNorm
    - SwiGLU
    - No shift AdaLN.
    Not all of them are used in the final model, please refer to the paper for more details.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_qknorm=False,
        use_swiglu=False,
        use_rmsnorm=False,
        wo_shift=False,
        use_v1_residual: bool = True,
        **block_kwargs,
    ):
        super().__init__()

        # Initialize normalization layers
        if not use_rmsnorm:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)

        # Initialize attention layer
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            use_v1_residual=use_v1_residual,
            **block_kwargs,
        )

        # Initialize MLP layer
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if use_swiglu:
            # here we did not use SwiGLU from xformers because it is not compatible with torch.compile for now.
            self.mlp = SwiGLUFFN(hidden_size, int(2 / 3 * mlp_hidden_dim))
        else:
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=ApproxGELU, drop=0)

        # Initialize AdaLN modulation
        if wo_shift:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 4 * hidden_size, bias=True))
            self.cond_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))
        else:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
            self.cond_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 4 * hidden_size, bias=True))
        self.wo_shift = wo_shift

    def _interp_cond(self, x, cond_mod):
        L = x.shape[1]
        if cond_mod.shape[1] != L:
            h = w = int(math.sqrt(L))
            hc = wc = int(math.sqrt(cond_mod.shape[1]))
            cond_mod = rearrange(cond_mod, "b (h w) c -> b c h w", h=hc, c=wc)
            cond_mod = F.interpolate(cond_mod, size=(h, w), mode="bilinear", align_corners=False)
            cond_mod = rearrange(cond_mod, "b c h w -> b (h w) c")
        return cond_mod

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        cond_tokens: torch.Tensor | None = None,
        feat_rope: VisionRotaryEmbeddingFast | None = None,
        rope_ids: torch.Tensor | None = None,
        v1: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.wo_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=1)
            shift_msa = None
            shift_mlp = None
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        if cond_tokens is not None:
            if self.wo_shift:
                scale_msa_c, scale_mlp_c = self.cond_modulation(cond_tokens).chunk(2, dim=-1)
                scale_msa = scale_msa.unsqueeze(1) + scale_msa_c
                scale_mlp = scale_mlp.unsqueeze(1) + scale_mlp_c
            else:
                shift_msa_c, scale_msa_c, shift_mlp_c, scale_mlp_c = self.cond_modulation(cond_tokens).chunk(4, dim=-1)
                shift_msa = shift_msa.unsqueeze(1) + shift_msa_c
                scale_msa = scale_msa.unsqueeze(1) + scale_msa_c
                shift_mlp = shift_mlp.unsqueeze(1) + shift_mlp_c
                scale_mlp = scale_mlp.unsqueeze(1) + scale_mlp_c

        attn_out, v_current = self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            rope=feat_rope,
            rope_ids=rope_ids,
            v1=v1,
        )
        x = x + gate_msa.unsqueeze(1) * attn_out
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x, v_current


class FinalLayer(nn.Module):
    """
    The final layer of LightningDiT.
    """

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, use_rmsnorm: bool = False) -> None:
        super().__init__()
        if not use_rmsnorm:
            self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class LightningDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 32,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        learn_sigma: bool = False,
        use_qknorm: bool = True,
        qk_norm: bool | None = True,
        use_swiglu: bool = True,
        use_rope: bool = True,
        use_rmsnorm: bool = True,
        wo_shift: bool = False,
        cond_drop_prob: float = 0.0,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        if qk_norm is not None:
            use_qknorm = qk_norm
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels if not learn_sigma else in_channels * 2
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.use_rmsnorm = use_rmsnorm
        self.depth = depth
        self.hidden_size = hidden_size
        self.use_checkpoint = use_checkpoint
        self.input_size = input_size
        self.cond_drop_prob = cond_drop_prob

        self.x_embedder = PatchEmbed(
            input_size,
            patch_size,
            in_channels,
            hidden_size,
            bias=True,
            strict_img_size=False,
            output_fmt="NLC",
        )
        self.cond_embedder = LazyPatchEmbed2d(patch_size=patch_size, embed_dim=hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # use rotary position encoding, borrow from EVA
        self._rope_cache = nn.ModuleDict()
        if self.use_rope:
            head_dim = hidden_size // num_heads
            if head_dim % 2 != 0:
                raise ValueError("head_dim must be even for RoPE")
            self._rope_base_grid = input_size // patch_size

        self.blocks = nn.ModuleList(
            [
                LightningDiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    use_qknorm=use_qknorm,
                    use_swiglu=use_swiglu,
                    use_rmsnorm=use_rmsnorm,
                    wo_shift=wo_shift,
                    use_v1_residual=i > 0,
                )
                for i in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, use_rmsnorm=use_rmsnorm)
        self.initialize_weights()

    def _get_feat_rope(self, *, grid_size: int) -> VisionRotaryEmbeddingFast:
        key = str(grid_size)
        if key in self._rope_cache:
            return cast(VisionRotaryEmbeddingFast, self._rope_cache[key])
        head_dim = self.hidden_size // self.num_heads
        rope = VisionRotaryEmbeddingFast(dim=head_dim // 2, pt_seq_len=self._rope_base_grid, ft_seq_len=grid_size)
        self._rope_cache[key] = rope
        return rope

    def _maybe_drop_condition_tokens(self, cond_tokens: torch.Tensor, *, uncond: bool) -> torch.Tensor:
        if uncond or self.cond_drop_prob <= 0.0:
            return cond_tokens if not uncond else torch.zeros_like(cond_tokens)

        if not self.training:
            return cond_tokens

        drop_mask = (torch.rand(cond_tokens.shape[0], device=cond_tokens.device) < self.cond_drop_prob).to(
            dtype=cond_tokens.dtype
        )
        drop_mask = drop_mask.view(-1, 1, 1)
        return cond_tokens * (1.0 - drop_mask)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        if self.x_embedder.proj.bias is not None:
            nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        t_mlp0 = cast(nn.Linear, self.t_embedder.mlp[0])
        t_mlp2 = cast(nn.Linear, self.t_embedder.mlp[2])
        nn.init.normal_(t_mlp0.weight, std=0.02)
        nn.init.normal_(t_mlp2.weight, std=0.02)

        # Zero-out adaLN modulation layers in LightningDiT blocks:
        for block in self.blocks:
            block = cast(LightningDiTBlock, block)

            ada_mod = cast(nn.Sequential, block.adaLN_modulation)
            ada_last = cast(nn.Linear, ada_mod[-1])
            nn.init.constant_(ada_last.weight, 0)
            nn.init.constant_(ada_last.bias, 0)

            cond_mod = cast(nn.Sequential, block.cond_modulation)
            cond_last = cast(nn.Linear, cond_mod[-1])
            nn.init.constant_(cond_last.weight, 0)
            nn.init.constant_(cond_last.bias, 0)

        # Zero-out output layers:
        final_mod = cast(nn.Sequential, self.final_layer.adaLN_modulation)
        final_last = cast(nn.Linear, final_mod[-1])
        nn.init.constant_(final_last.weight, 0)
        nn.init.constant_(final_last.bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        *,
        return_logvar: bool = False,
        uncond: bool = False,
        conditions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None, None]:
        """
        Forward pass of LightningDiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        use_checkpoint: boolean to toggle checkpointing
        """
        _ = return_logvar
        use_checkpoint = self.use_checkpoint
        res = None
        if torch.is_tensor(conditions):
            res = conditions
        elif isinstance(conditions, (list, tuple)):
            res = conditions[0]

        H, W = x.shape[-2:]
        x = self.x_embedder(x)  # (B, T, D), where T = H*W / patch_size**2
        x = x + resample_abs_pos_embed(
            self.pos_embed,
            new_size=[H // self.patch_size, W // self.patch_size],
            old_size=[self.input_size // self.patch_size, self.input_size // self.patch_size],
            num_prefix_tokens=0,
        )

        cond_tokens: torch.Tensor | None = None
        if conditions is not None:
            cond_tokens = self.cond_embedder(conditions)  # (B, T, D)
            if cond_tokens.shape[1] != x.shape[1]:
                raise ValueError(
                    "condition and input must have the same patch-grid; "
                    f"got x_tokens={x.shape[1]} cond_tokens={cond_tokens.shape[1]}"
                )
            cond_tokens = self._maybe_drop_condition_tokens(cond_tokens, uncond=uncond)

        B, T_tokens, _ = x.shape
        grid_h = H // self.patch_size
        grid_w = W // self.patch_size
        rope_ids: torch.Tensor | None = None
        feat_rope: VisionRotaryEmbeddingFast | None = None
        if self.use_rope:
            if grid_h != grid_w:
                raise ValueError(f"RoPE currently requires square patch grid, got {grid_h=} {grid_w=}")
            feat_rope = self._get_feat_rope(grid_size=grid_h)
            rope_ids = torch.arange(T_tokens, device=x.device, dtype=torch.long).view(1, -1).expand(B, -1)

        c = self.t_embedder(t)  # (B, D)

        v1: torch.Tensor | None = None
        for i, block in enumerate(self.blocks):
            if use_checkpoint and self.training:
                x, v_current = torch.utils.checkpoint.checkpoint(
                    block, x, c, cond_tokens, feat_rope, rope_ids, v1, use_reentrant=False
                )
            else:
                x, v_current = block(x, c, cond_tokens, feat_rope, rope_ids, v1=v1)

            if i == 0:
                v1 = v_current

        x = self.final_layer(x, c)  # (B, T, patch_size**2*out_channels)
        x = self.unpatchify(x)  # (B, out_channels, H, W)

        if self.learn_sigma:
            x, _rest = x.chunk(2, dim=1)

        # add input
        if res is not None:
            x = x + res
        return x, None, None


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                             LightningDiT Configs                              #
#################################################################################


def LightningDiT_XL_1(**kwargs):
    return LightningDiT(depth=28, hidden_size=1152, patch_size=1, num_heads=16, **kwargs)


def LightningDiT_XL_2(**kwargs):
    return LightningDiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def LightningDiT_L_2(**kwargs):
    return LightningDiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def LightningDiT_B_1(**kwargs):
    return LightningDiT(depth=12, hidden_size=768, patch_size=1, num_heads=12, **kwargs)


def LightningDiT_B_2(**kwargs):
    return LightningDiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


def LightningDiT_1p0B_1(**kwargs):
    return LightningDiT(depth=24, hidden_size=1536, patch_size=1, num_heads=24, **kwargs)


def LightningDiT_1p0B_2(**kwargs):
    return LightningDiT(depth=24, hidden_size=1536, patch_size=2, num_heads=24, **kwargs)


def LightningDiT_1p6B_1(**kwargs):
    return LightningDiT(depth=28, hidden_size=1792, patch_size=1, num_heads=28, **kwargs)


def LightningDiT_1p6B_2(**kwargs):
    return LightningDiT(depth=28, hidden_size=1792, patch_size=2, num_heads=28, **kwargs)


LightningDiT_models = {
    "LightningDiT-B/1": LightningDiT_B_1,
    "LightningDiT-B/2": LightningDiT_B_2,
    "LightningDiT-L/2": LightningDiT_L_2,
    "LightningDiT-XL/1": LightningDiT_XL_1,
    "LightningDiT-XL/2": LightningDiT_XL_2,
    "LightningDiT-1p0B/1": LightningDiT_1p0B_1,
    "LightningDiT-1p0B/2": LightningDiT_1p0B_2,
    "LightningDiT-1p6B/1": LightningDiT_1p6B_1,
    "LightningDiT-1p6B/2": LightningDiT_1p6B_2,
}
