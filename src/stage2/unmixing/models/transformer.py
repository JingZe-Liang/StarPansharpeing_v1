from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Array, Float
from timm.layers import get_act_layer, get_norm_layer

# Attention, MLP
from timm.layers.patch_embed import PatchEmbed
from torch import Tensor

from ...layers import (
    AttentionBlock,
    RotaryPositionEmbeddingPytorchV2,
    get_2d_sincos_pos_embed,
)


@dataclass
class TransformerConfig:
    in_dim: int = 16
    out_channels: int = 256
    dim: int = 256
    depth: int = 8
    num_heads: int = 8
    with_raw_img: bool = False
    mlp_ratio: float = 4.0
    drop: float = 0.0
    drop_path: float = 0.0
    input_size: int = 32
    patch_size: int = 2
    raw_img_size: Any = None
    raw_img_chans: Any = None
    pos_embed_type: str = "sincos"
    norm_layer: Any = "layernorm"
    mlp_norm_layer: Any = "layernorm"
    act_layer: Any = "swiglu"
    feature_layer_ids: Any = None


class Transformer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        dim,
        depth,
        num_heads,
        with_raw_img=False,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        input_size: int = 32,
        patch_size=2,
        out_channels=16,
        raw_img_size=None,
        raw_img_chans=None,
        pos_embed_type="sincos",
        norm_layer="layernorm",
        mlp_norm_layer="layernorm",
        act_layer="swiglu",
        feature_layer_ids: list[int] | None = None,
    ):
        super().__init__()

        # patch embedding
        self.patch_size = patch_size
        self.input_size = input_size
        self.num_patches = (input_size // patch_size) ** 2
        self._n_modalities = 1
        self.with_raw_img = with_raw_img
        self.raw_img_size = raw_img_size
        self.raw_img_chans = raw_img_chans
        self.raw_patch_size = 16
        self.patch_embed = PatchEmbed(
            img_size=input_size,
            patch_size=patch_size,
            in_chans=in_dim,
            embed_dim=dim,
            bias=True,
            strict_img_size=False,
        )
        if self.with_raw_img:
            assert raw_img_size is not None
            assert raw_img_chans is not None
            assert self.pos_embed_type == "sincos"
            self.raw_img_patcher = PatchEmbed(
                img_size=raw_img_size,
                patch_size=self.raw_patch_size,
                in_chans=raw_img_chans,
                embed_dim=dim,
                bias=True,
                strict_img_size=False,
            )
        # self.fuse_stem = nn.Linear(dim // self._n_modalities * self._n_modalities, dim)
        self.base_size = input_size // self.patch_size
        self.pe_interpolation = 1.0
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.feature_layer_ids = feature_layer_ids
        if feature_layer_ids:
            assert max(feature_layer_ids) < depth, (
                "max feature_layer_id must be less than depth"
            )

        # layers
        layers = []
        drop_path = [
            x.item() for x in torch.linspace(0, drop_path, depth)
        ]  # stochastic depth decay rule
        norm_layer = get_norm_layer(norm_layer)
        mlp_norm_layer = get_norm_layer(mlp_norm_layer)
        act_layer = get_act_layer(act_layer)
        for i in range(depth):
            layers.append(
                AttentionBlock(
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
                    norm_layer=mlp_norm_layer,
                    act_layer=act_layer,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.head = nn.Sequential(
            norm_layer(dim),
            nn.Linear(dim, out_channels * patch_size**2, bias=True),
        )

        # positional embedding
        self.pos_embed_type = pos_embed_type
        self.setup_pe(dim)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    def setup_pe(self, dim, rope_options: dict | None = None):
        seq_len = self.num_patches
        if self.with_raw_img:
            raw_img_patches = self.raw_img_patcher.num_patches

        if self.pos_embed_type == "sincos":
            self.pos_embed_latent: nn.Buffer
            self.register_buffer(
                "pos_embed_latent", torch.zeros(1, self.num_patches, dim)
            )
            # sincos
            pos_embed = get_2d_sincos_pos_embed(
                self.pos_embed_latent.shape[-1],
                int(seq_len**0.5),
                pe_interpolation=self.pe_interpolation,
                base_size=self.base_size,
            )
            self.pos_embed_latent.data.copy_(
                torch.from_numpy(pos_embed).float().unsqueeze(0)
            )
            if self.with_raw_img:
                pos_embed_raw = get_2d_sincos_pos_embed(
                    dim,
                    int(raw_img_patches**0.5),
                    pe_interpolation=self.pe_interpolation,
                    base_size=self.raw_img_size // self.patch_size,
                )
                self.pos_embed_raw: nn.Buffer
                self.register_buffer(
                    "pos_embed_raw", torch.as_tensor(pos_embed_raw).float().unsqueeze(0)
                )

        elif self.pos_embed_type == "rope":
            if rope_options is None:
                self.rope_options = {
                    "dim": dim // self.num_heads,
                    "rope_dim": "2D",
                    "beta_fast": 4,
                    "beta_slow": 1,
                    "rope_theta": 10000,
                    "apply_yarn": True,
                    "scale": 1.0,
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

    def get_pe(self, hw: tuple | torch.Size, img_type=None):
        h, w = hw
        if self.pos_embed_type == "sincos":
            if img_type == "raw":
                assert self.with_raw_img
                pe = self.pos_embed_raw
                name = "pos_embed_raw"
                base_size = self.raw_img_size // self.raw_patch_size
                ps = self.raw_patch_size
            else:
                pe = self.pos_embed_latent
                name = "pos_embed_latent"
                base_size = self.base_size
                ps = self.patch_size

            if pe.shape[1] != h * w:
                # re-init the pos_embed
                pe = get_2d_sincos_pos_embed(
                    pe.shape[-1],
                    (h // ps, w // ps),
                    pe_interpolation=self.pe_interpolation,
                    base_size=base_size,
                )
                self.register_buffer(
                    name,
                    (pe := torch.from_numpy(pe).float().unsqueeze(0)),
                    persistent=False,
                )
            return pe
        elif self.pos_embed_type == "rope":
            # TODO: add multi-modal-rope
            # (modalities -> ids -> online RoPE class -> positional embedding -> kv rope fn)

            ph, pw = h // self.patch_size, w // self.patch_size
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

        x = rearrange(
            x, "bs (h w) (p1 p2 c) -> bs c (h p1) (w p2)", h=h, w=w, p1=p, p2=p, c=c
        )
        return x

    def forward(
        self,
        noisy_latent: Float[Tensor, "b c_latent h w"],
        hyper_img: Float[Tensor, "b c_hyper H W"] | None = None,
        *,
        feature_layer_ids: list[int] | None = None,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        # patch embedding
        y = self.patch_embed(noisy_latent)

        # positional embedding
        pe = self.get_pe(noisy_latent.shape[-2:])
        if self.pos_embed_type == "sincos":
            assert torch.is_tensor(pe), "Positional embedding must be a tensor."
            y = y + pe.to(y)
            pe = None

        if self.with_raw_img:
            assert hyper_img is not None
            y2 = self.raw_img_patcher(hyper_img)

            if self.pos_embed_type == "sincos":
                pe2 = self.get_pe(hyper_img.shape[-2:], img_type="raw")
                assert torch.is_tensor(pe2), "Positional embedding must be a tensor."
                y2 = y2 + pe2.to(y2)

            y = torch.cat([y, y2], dim=1)  # [bs, s1 + s2, c]

        # blocks
        feature_layer_ids_ = self.feature_layer_ids or feature_layer_ids
        features = []
        for i, layer in enumerate(self.layers):
            y = layer(y, mask=None, pe=pe)
            if feature_layer_ids_ is not None and i in feature_layer_ids_:
                features.append(y)

        # head
        y = y[:, : y.size(1)]
        y = self.head(y)

        # unpatchify
        out = self.unpatchify(y)

        if len(features) == 0:
            return out
        else:
            return out, features

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
