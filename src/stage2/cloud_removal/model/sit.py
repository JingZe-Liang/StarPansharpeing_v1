# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# SPRINT: SPRINT: Sparse-Dense Residual Fusion for Efficient Diffusion Transformers
# --------------------------------------------------------

from typing import Optional, cast
import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Mlp
from timm.layers import resample_abs_pos_embed
import torch.nn.functional as F
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, checkpoint_wrapper

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
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


class ApproxGELU(nn.GELU):
    def __init__(self) -> None:
        super().__init__(approximate="tanh")


def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
        nn.Linear(hidden_size, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, z_dim),
    )


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if shift.ndim == 2:
        shift = shift.unsqueeze(1)
    if scale.ndim == 2:
        scale = scale.unsqueeze(1)
    return x * (1 + scale) + shift


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def positional_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                  These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
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

    def forward(self, t):
        self.timestep_embedding = self.positional_embedding
        t_freq = self.timestep_embedding(t, dim=self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
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


#################################################################################
#                                 Core SiT Model                                #
#################################################################################


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: type[nn.Module] = nn.RMSNorm,
        use_v1_residual: bool = True,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if use_v1_residual:
            self.v1_lambda = nn.Parameter(torch.tensor(0.5))

        # for API compatibility with timm Attention
        self.fused_attn = False

    def forward(
        self,
        x: torch.Tensor,
        rope: Optional[VisionRotaryEmbeddingFast] = None,
        rope_ids: Optional[torch.Tensor] = None,
        v1: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Multi-head self-attention with optional 2D RoPE.

        x: (B, N, C)
        rope: VisionRotaryEmbeddingFast or None
        rope_ids: (B, N) or (N,) original (flattened) token indices for RoPE,
                  supporting routed / sparse subsets.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # (3, B, H, N, Dh)
        q, k, v = qkv.unbind(0)
        v_out = v  # Save v to return (for v1 residual), keep gradient
        q, k = self.q_norm(q), self.k_norm(k)

        if v1 is not None:
            v = self.v1_lambda * v1 + (1.0 - self.v1_lambda) * v

        if rope is not None:
            q = rope(q, rope_ids)
            k = rope(k, rope_ids)

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, v_out  # Return both output and v for v1 residual


class SiTBlock(nn.Module):
    """
    A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, use_v1_residual: bool = True, **block_kwargs):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=block_kwargs["qk_norm"],
            use_v1_residual=use_v1_residual,
        )
        if "fused_attn" in block_kwargs.keys():
            self.attn.fused_attn = block_kwargs["fused_attn"]
        self.norm2 = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=ApproxGELU, drop=0)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.cond_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 4 * hidden_size, bias=True))

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        cond_tokens: torch.Tensor | None = None,
        feat_rope: Optional[VisionRotaryEmbeddingFast] = None,
        rope_ids: Optional[torch.Tensor] = None,
        v1: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shift_msa_t, scale_msa_t, gate_msa, shift_mlp_t, scale_mlp_t, gate_mlp = self.adaLN_modulation(c).chunk(
            6, dim=-1
        )
        if cond_tokens is not None:
            shift_msa_c, scale_msa_c, shift_mlp_c, scale_mlp_c = self.cond_modulation(cond_tokens).chunk(4, dim=-1)
            shift_msa = shift_msa_t.unsqueeze(1) + shift_msa_c
            scale_msa = scale_msa_t.unsqueeze(1) + scale_msa_c
            shift_mlp = shift_mlp_t.unsqueeze(1) + shift_mlp_c
            scale_mlp = scale_mlp_t.unsqueeze(1) + scale_mlp_c
        else:
            shift_msa, scale_msa = shift_msa_t, scale_msa_t
            shift_mlp, scale_mlp = shift_mlp_t, scale_mlp_t
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
    The final layer of SiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class SiT(nn.Module):
    """
    Diffusion model with a Transformer backbone + SPRINT sparse-dense residual fusion.
    """

    def __init__(
        self,
        path_type="edm",
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        encoder_depth=8,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        z_dims=[768],
        projector_dim=2048,
        cond_drop_prob: float = 0.0,
        use_checkpoint: bool = False,
        use_repa: bool = False,
        **block_kwargs,  # fused_attn
    ):
        super().__init__()
        self.path_type = path_type
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.z_dims = z_dims
        self.encoder_depth = encoder_depth
        self.depth = depth
        self.cond_drop_prob = cond_drop_prob
        self.input_size = input_size
        self.use_checkpoint = use_checkpoint
        self.use_repa = use_repa

        # ----------------- SPRINT configuration -----------------
        # fθ / gθ / hθ split: default 2 / (D-4) / 2 as in the paper.
        self.num_f = 2
        self.num_h = 2
        self.num_g = self.depth - self.num_f - self.num_h
        assert self.num_g >= 0, "depth too small for SPRINT split"

        # Token drop ratio r (fraction of tokens to drop in sparse path)
        self.sprint_drop_ratio = 0.75

        # Path-drop learning probability p (drop whole sparse path during training)
        self.path_drop_prob = 0.05

        # [MASK] token for padding dropped positions
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

        # Fusion projection: concat(ft, g_pad) → fused hidden
        self.fusion_proj = nn.Linear(2 * hidden_size, hidden_size, bias=True)
        # --------------------------------------------------------

        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True, strict_img_size=False, output_fmt="NLC"
        )
        self.cond_embedder = LazyPatchEmbed2d(patch_size=patch_size, embed_dim=hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)  # timestep embedding type
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        blocks = []
        for i in range(depth):
            use_v1_residual = i > 0
            block = SiTBlock(
                hidden_size,
                num_heads,
                mlp_ratio=mlp_ratio,
                use_v1_residual=use_v1_residual,
                **block_kwargs,
            )
            # Wrap with checkpoint if enabled
            if use_checkpoint:
                block = checkpoint_wrapper(block, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)
        if self.use_repa:
            self.projectors = nn.ModuleList([build_mlp(hidden_size, projector_dim, z_dim) for z_dim in z_dims])
        else:
            self.projectors = nn.ModuleList()

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        # RoPE for spatial tokens
        head_dim = hidden_size // num_heads
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        hw_seq_len = input_size // patch_size
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=head_dim // 2,
            pt_seq_len=hw_seq_len,
        )

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
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

        # Zero-out adaLN modulation layers in SiT blocks:
        for block in self.blocks:
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

    def unpatchify(self, x, patch_size=None):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0] if patch_size is None else patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    # --------------------------- SPRINT helpers ---------------------------
    def _drop_tokens(self, x, drop_ratio):
        """
        Randomly drop a fraction of tokens (except we ensure at least one token kept).

        x: (B, T, C)
        drop_ratio: fraction of tokens to drop (0.0 ~ 1.0)
        Returns:
            x_keep: (B, T_keep, C)
            ids_keep: (B, T_keep) indices into original T, or None if no drop.
        """
        if drop_ratio <= 0.0:
            return x, None

        B, T, C = x.shape
        if T <= 1:
            return x, None

        num_keep = max(1, int(T * (1.0 - drop_ratio)))
        if num_keep >= T:
            return x, None

        device = x.device
        noise = torch.rand(B, T, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :num_keep]  # (B, T_keep)
        x_keep = x.gather(1, ids_keep.unsqueeze(-1).expand(-1, -1, C))
        return x_keep, ids_keep

    def _pad_with_mask(self, x_sparse, ids_keep, T_full):
        """
        x_sparse: (B, T_keep, C)
        ids_keep: (B, T_keep)
        T_full: full sequence length T
        Returns:
            x_pad: (B, T_full, C) with [MASK] at dropped positions.
        """
        if ids_keep is None:
            return x_sparse

        B, T_keep, C = x_sparse.shape
        assert T_full >= T_keep
        x_pad = self.mask_token.expand(B, T_full, C).clone()
        x_pad.scatter_(1, ids_keep.unsqueeze(-1).expand(-1, -1, C), x_sparse)
        return x_pad

    def _sprint_fuse(self, f_dense, g_full):
        """
        f_dense: (B, T, C) encoder output ft
        g_full: (B, T, C) padded sparse output g_pad
        Returns fused h: (B, T, C)
        """
        h = torch.cat([f_dense, g_full], dim=-1)  # (B, T, 2C)
        h = self.fusion_proj(h)
        return h

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

    # ---------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        return_logvar: bool = False,
        uncond: bool = False,
        conditions: torch.Tensor | None = None,
    ):
        """
        Forward pass of SiT with SPRINT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        conditions: image conditioning. Tensor (N,C,H,W). If you have multiple conditions (e.g. optical + SAR),
                    concatenate them on channel dim before passing in.
        """

        # Patch embedding
        H, W = x.shape[-2:]
        x = self.x_embedder(x)  # (B, P, D), where P = H * W / patch_size ** 2
        x = x + resample_abs_pos_embed(
            self.pos_embed,
            new_size=[H // self.patch_size, W // self.patch_size],
            old_size=[self.input_size // self.patch_size, self.input_size // self.patch_size],
            num_prefix_tokens=0,
        )

        cond_tokens: torch.Tensor | None = None
        if conditions is not None:
            cond_tokens = self.cond_embedder(conditions)  # (B, P, D)
            if cond_tokens.shape[1] != x.shape[1]:
                raise ValueError(
                    "condition and input must have the same patch-grid; "
                    f"got x_tokens={x.shape[1]} cond_tokens={cond_tokens.shape[1]}"
                )
            cond_tokens = self._maybe_drop_condition_tokens(cond_tokens, uncond=uncond)

        N, T, D = x.shape

        # RoPE position ids for full sequence (CLS + spatial tokens)
        hw = int(self.x_embedder.num_patches**0.5)
        # assert hw * hw == self.x_embedder.num_patches
        device = x.device
        flat_pos = torch.arange(hw * hw, device=device, dtype=torch.long).view(1, -1)
        flat_pos = flat_pos.expand(N, -1)  # (N, HW)
        rope_ids_full = flat_pos  # (N, T)

        # timestep embedding
        t_embed = self.t_embedder(t)  # (N, D)
        c = t_embed

        # ------------------------------------------------------------------
        # 1) Encoder fθ on all tokens (dense, shallow)
        # ------------------------------------------------------------------
        x_enc = x
        rope_ids_enc = rope_ids_full

        v1_full = None
        for i in range(self.num_f):
            x_enc, v_current = self.blocks[i](x_enc, c, cond_tokens, self.feat_rope, rope_ids_enc, v1=v1_full)
            if v1_full is None:
                v1_full = v_current

        assert v1_full is not None

        # Use encoder output for z-projections (REPA / SPRINT z_t)
        if self.use_repa:
            zs = [
                projector(x_enc.reshape(-1, D)).reshape(N, -1, z_dim)
                for projector, z_dim in zip(self.projectors, self.z_dims)
            ]
        else:
            zs = []

        # ------------------------------------------------------------------
        # 2) Drop tokens to build sparse input to gθ (SPRINT sparse path)
        # ------------------------------------------------------------------
        if self.training:
            x_sparse, ids_keep = self._drop_tokens(x_enc, self.sprint_drop_ratio)
        else:
            x_sparse = x_enc
            ids_keep = None

        if ids_keep is not None:
            rope_ids_sparse = rope_ids_full.gather(1, ids_keep)
        else:
            rope_ids_sparse = rope_ids_full

        if ids_keep is not None:
            v1_sparse_index = ids_keep[:, None, :, None].expand(-1, v1_full.shape[1], -1, v1_full.shape[-1])
            v1_sparse = torch.gather(v1_full, dim=2, index=v1_sparse_index)
        else:
            v1_sparse = v1_full

        cond_tokens_sparse: torch.Tensor | None = None
        if cond_tokens is not None:
            if ids_keep is None:
                cond_tokens_sparse = cond_tokens
            else:
                cond_tokens_sparse = cond_tokens.gather(1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

        # ------------------------------------------------------------------
        # 3) Middle blocks gθ on sparse tokens
        # ------------------------------------------------------------------
        x_mid = x_sparse
        for i in range(self.num_f, self.num_f + self.num_g):
            x_mid, _ = self.blocks[i](x_mid, c, cond_tokens_sparse, self.feat_rope, rope_ids_sparse, v1=v1_sparse)

        # ------------------------------------------------------------------
        # 4) Pad back to full length with [MASK] to get g_pad
        # ------------------------------------------------------------------
        g_pad = self._pad_with_mask(x_mid, ids_keep, T_full=T)

        # ------------------------------------------------------------------
        # 5) Path-drop learning
        #    - During training: same behavior as before (stochastic path drop).
        #    - During sampling: enable path drop only for unconditional flow
        #      via the `uncond` flag.
        # ------------------------------------------------------------------
        if self.training and self.path_drop_prob > 0.0:
            # Sync random decision across all ranks
            drop_path = torch.rand(1, device=x.device)
            if torch.distributed.is_initialized():
                torch.distributed.broadcast(drop_path, src=0)
            if drop_path.item() < self.path_drop_prob:
                # Keep gradient flow but zero out the contribution
                g_pad = g_pad * 0.0 + self.mask_token.expand_as(g_pad)
        elif uncond:  # Drop path for all samples
            g_pad = g_pad * 0.0 + self.mask_token.expand_as(g_pad)

        # ------------------------------------------------------------------
        # 6) Sparse–dense residual fusion: h_in = Fusion(ft, g_pad)
        # ------------------------------------------------------------------
        h_in = self._sprint_fuse(x_enc, g_pad)  # (N, T, D)

        # ------------------------------------------------------------------
        # 7) Decoder hθ on fused representation
        # ------------------------------------------------------------------
        x_dec = h_in
        for i in range(self.num_f + self.num_g, self.depth):
            x_dec, _ = self.blocks[i](x_dec, c, cond_tokens, self.feat_rope, rope_ids_full, v1=v1_full)

        x_out = self.final_layer(x_dec, c)
        x_out = self.unpatchify(x_out)

        return x_out, zs, None


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


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
#                                   SiT Configs                                  #
#################################################################################


def SiT_XL_1(**kwargs):
    return SiT(
        depth=28,
        hidden_size=1152,
        patch_size=1,
        num_heads=16,
        encoder_depth=8,
        **kwargs,
    )


def SiT_XL_2(**kwargs):
    return SiT(
        depth=28,
        hidden_size=1152,
        patch_size=2,
        num_heads=16,
        encoder_depth=8,
        **kwargs,
    )


def SiT_XL_4(**kwargs):
    return SiT(
        depth=28,
        hidden_size=1152,
        patch_size=4,
        num_heads=16,
        encoder_depth=8,
        **kwargs,
    )


def SiT_L_1(**kwargs):
    return SiT(
        depth=24,
        hidden_size=1024,
        patch_size=1,
        num_heads=16,
        encoder_depth=8,
        **kwargs,
    )


def SiT_L_2(**kwargs):
    return SiT(
        depth=24,
        hidden_size=1024,
        patch_size=2,
        num_heads=16,
        encoder_depth=8,
        **kwargs,
    )


def SiT_L_4(**kwargs):
    return SiT(
        depth=24,
        hidden_size=1024,
        patch_size=4,
        num_heads=16,
        encoder_depth=8,
        **kwargs,
    )


def SiT_B_1(**kwargs):
    return SiT(
        depth=12,
        hidden_size=768,
        patch_size=1,
        num_heads=12,
        encoder_depth=4,
        **kwargs,
    )


def SiT_B_2(**kwargs):
    return SiT(
        depth=12,
        hidden_size=768,
        patch_size=2,
        num_heads=12,
        encoder_depth=4,
        **kwargs,
    )


def SiT_B_4(**kwargs):
    return SiT(
        depth=12,
        hidden_size=768,
        patch_size=4,
        num_heads=12,
        encoder_depth=4,
        **kwargs,
    )


def SiT_S_1(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=1, num_heads=6, encoder_depth=4, **kwargs)


def SiT_S_2(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, encoder_depth=4, **kwargs)


def SiT_S_4(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, encoder_depth=4, **kwargs)


SiT_models = {
    "SiT-XL/1": SiT_XL_1,
    "SiT-XL/2": SiT_XL_2,
    "SiT-XL/4": SiT_XL_4,
    "SiT-L/1": SiT_L_1,
    "SiT-L/2": SiT_L_2,
    "SiT-L/4": SiT_L_4,
    "SiT-B/1": SiT_B_1,
    "SiT-B/2": SiT_B_2,
    "SiT-B/4": SiT_B_4,
    "SiT-S/1": SiT_S_1,
    "SiT-S/2": SiT_S_2,
    "SiT-S/4": SiT_S_4,
}
