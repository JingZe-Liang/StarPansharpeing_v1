import functools
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn.ops.rms_norm import RMSNorm as RMSNormFlash
from timm.layers.attention import Attention as Attention_
from timm.layers.drop import DropPath
from timm.layers.mlp import Mlp as Mlp_
from timm.layers.patch_embed import PatchEmbed

from src.stage2.pansharpening.models.rope import (
    RotaryPositionEmbeddingPytorchV2,
    get_2d_sincos_pos_embed,
)


class Attention(Attention_):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        qk_norm: nn.Module | None = None,
        **block_kwargs,
    ):
        super().__init__(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            **block_kwargs,
        )

        if qk_norm:
            self.q_norm = qk_norm(dim)
            self.k_norm = qk_norm(dim)

    def forward(self, x, mask=None, rope: Callable | None = None):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, C)
        q, k, v = qkv.unbind(2)
        dtype = q.dtype

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.reshape(B, N, self.num_heads, C // self.num_heads).to(dtype)
        k = k.reshape(B, N, self.num_heads, C // self.num_heads).to(dtype)
        v = v.reshape(B, N, self.num_heads, C // self.num_heads).to(dtype)

        # RoPE
        if rope is not None:
            q, k = rope(q, k)

        use_fp32_attention = getattr(
            self, "fp32_attention", False
        )  # necessary for NAN loss
        if use_fp32_attention:
            q, k, v = q.float(), k.float(), v.float()

        attn_bias = None
        if mask is not None:
            attn_bias = torch.zeros(
                [B * self.num_heads, q.shape[1], k.shape[1]],
                dtype=q.dtype,
                device=q.device,
            )
            attn_bias.masked_fill_(
                mask.squeeze(1).repeat(self.num_heads, 1, 1) == 0, float("-inf")
            )

        # sdpa
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if mask is not None and mask.ndim == 2:
            mask = (1 - mask.to(x.dtype)) * -10000.0
            mask = mask[:, None, None].repeat(1, self.num_heads, 1, 1)
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
        )
        x = x.transpose(1, 2)

        x = x.view(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if torch.get_autocast_gpu_dtype() == torch.float16:
            x = x.clip(-65504, 65504)

        return x


class ClipMlp(Mlp_):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=functools.partial(nn.GELU, approximate="tanh"),
        norm_layer=nn.LayerNorm,
        bias=True,
        drop=0.0,
        use_conv=False,
        clip_val: float = 8.0,
    ):
        super().__init__(
            in_features,
            hidden_features,
            out_features,
            act_layer,
            norm_layer,
            bias,
            drop,
            use_conv,
        )
        self.clip_val = clip_val

    def forward(self, x):
        return (
            super().forward(x).clip(-self.clip_val, self.clip_val)
        )  # taken from openai OSSA


class AttenionBlock(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        num_heads=8,
        qkv_bias=True,
        qk_norm: nn.Module | None = None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=functools.partial(nn.GELU, approximate="tanh"),
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.norm2 = norm_layer(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = ClipMlp(
            in_features=dim,
            hidden_features=hidden_dim,
            act_layer=act_layer,
            drop=drop,
            norm_layer=norm_layer,
        )

        self.drop_path = DropPath(drop_path)

    def forward(self, x, mask=None, pe=None):
        x = x + self.drop_path(self.attn(self.norm1(x), mask=mask, rope=pe))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        input_size=32,
        patch_size=2,
        out_channels=16,
        pos_embed_type="sincos",
        norm_layer=functools.partial(RMSNormFlash, eps=1e-6),
        act_layer=functools.partial(nn.GELU, approximate="tanh"),
    ):
        super().__init__()

        # patch embedding
        self.patch_size = patch_size
        self.input_size = input_size
        self.num_patches = (input_size // patch_size) ** 2
        self._n_modalities = 2
        self.patch_embed = PatchEmbed(
            img_size=input_size,
            patch_size=patch_size,
            in_chans=dim,
            embed_dim=dim // self._n_modalities,
            bias=True,
            strict_img_size=False,
        )
        self.fuse_stem = nn.Linear(dim, dim)
        self.base_size = input_size // self.patch_size
        self.pe_interpolation = 1.0
        self.out_channels = out_channels
        self.num_heads = num_heads

        # layers
        layers = []
        drop_path = [
            x.item() for x in torch.linspace(0, drop_path, depth)
        ]  # stochastic depth decay rule
        for i in range(depth):
            layers.append(
                AttenionBlock(
                    dim=dim,
                    mlp_ratio=mlp_ratio,
                    num_heads=num_heads,
                    qkv_bias=True,
                    qk_norm=norm_layer,
                    drop=drop,
                    attn_drop=drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
            )
        self.layers = nn.ModuleList(layers)

        # positional embedding
        self.pos_embed_type = pos_embed_type
        self.setup_pe(dim)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    def setup_pe(self, dim, rope_options: dict = {}):
        seq_len = self.num_patches

        if self.pos_embed_type == "sincos":
            self.register_buffer("pos_embed", torch.zeros(1, self.num_patches, dim))
            # sincos
            pos_embed = get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1],
                int(seq_len**0.5),
                pe_interpolation=self.pe_interpolation,
                base_size=self.base_size,
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        elif self.pos_embed_type == "rope":
            if len(rope_options) == 0:
                self.rope_options = {
                    "dim": dim // self.num_heads,
                    "rope_dim": "2D",
                    "beta_fast": rope_options.pop("beta_fast", 4),
                    "beta_slow": rope_options.pop("beta_slow", 1),
                    "rope_theta": rope_options.pop("rope_base", 10000),
                    "apply_yarn": rope_options.pop("apply_yarn", True),
                    "scale": rope_options.pop("rope_scale", 1.0),
                    "original_latent_shape": (self.base_size, self.base_size),
                }
            self.rope = RotaryPositionEmbeddingPytorchV2(
                seq_len=seq_len,
                latent_shape=(self.base_size, self.base_size),
                # does not changed
                **self.rope_options,
            )
        else:
            raise ValueError(
                f"Unsupported pos_embed_type: {self.pos_embed_type}. "
                "Supported types are 'sincos' and 'rope'."
            )

    def get_pe(self, img: torch.Tensor):
        if self.pos_embed_type == "sincos":
            if self.pos_embed.shape[1] != img.shape[-2] * img.shape[-1]:
                # re-init the pos_embed
                pos_embed = get_2d_sincos_pos_embed(
                    self.pos_embed.shape[-1],
                    (
                        img.shape[-2] // self.patch_size,
                        img.shape[-1] // self.patch_size,
                    ),
                    pe_interpolation=self.pe_interpolation,
                    base_size=self.base_size,
                )
                self.pos_embed = nn.Buffer(
                    data=torch.from_numpy(pos_embed).float()[None],
                    requires_grad=False,
                    persistent=False,
                )
            return self.pos_embed
        elif self.pos_embed_type == "rope":
            assert img.ndim == 4, "Image tensor must be 4D (N, C, H, W)"
            ph, pw = img.shape[-2] // self.patch_size, img.shape[-1] // self.patch_size
            seq_len_x = ph * pw
            pre_rope_seq_len = self.rope.cos_cached.shape[1]
            if seq_len_x > pre_rope_seq_len:
                # re-init the rope
                self.rope_options["latent_shape"] = (ph, pw)
                self.rope.__init__(seq_len_x, **self.rope_options)
            return self.rope
        else:
            raise ValueError(
                f"Unsupported pos_embed_type: {self.pos_embed_type}. "
                "Supported types are 'sincos' and 'rope'."
            )

    def unpatchify(self, x: torch.Tensor):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, latents: tuple):
        # patch embedding
        ys = []
        for latent in latents:
            y = self.patch_embed(latent)
            ys.append(y)

        # fuse all latent
        y = self.fuse_stem(torch.cat(ys, dim=-1))

        # pe
        pe = self.get_pe(latents[0])
        if self.pos_embed_type == "sincos":
            assert torch.is_tensor(pe), "Positional embedding must be a tensor."
            y = y + pe.to(y)
            pe = None

        # blocks
        for layer in self.layers:
            y = layer(y, mask=None, rope=pe)

        # unpatchify
        out = self.unpatchify(y)

        return out

    def init_weights(self):
        # Initialize transformer layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # patch embedding
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
