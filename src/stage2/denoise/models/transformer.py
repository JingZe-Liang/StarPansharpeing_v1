from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Array, Float
from loguru import logger
from timm.layers import get_act_layer, get_norm_layer

# Attention, MLP
from timm.layers.patch_embed import PatchEmbed
from timm.layers.pos_embed import resample_abs_pos_embed
from timm.layers.pos_embed_sincos import RotaryEmbeddingCat
from torch import Tensor

from src.utilities.config_utils import function_config_to_easy_dict

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
    raw_img_size: int | None = None
    raw_img_chans: int | None = None
    pos_embed_type: str = "sincos"
    norm_layer: str = "rmsnorm"
    mlp_norm_layer: str = "rmsnorm"
    act_layer: str = "silu"
    use_layerscale: bool = True
    feature_layer_ids: list[int] | None = None


class Transformer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()

        # Store config for reference
        self.cfg = cfg

        # patch embedding
        self.patch_size = cfg.patch_size
        self.input_size = cfg.input_size
        self.num_patches = (cfg.input_size // cfg.patch_size) ** 2
        self._n_modalities = 1
        self.with_raw_img = cfg.with_raw_img
        self.raw_img_size = cfg.raw_img_size
        self.raw_img_chans = cfg.raw_img_chans
        self.raw_patch_size = 16

        # Norm layers
        norm_layer = get_norm_layer(cfg.norm_layer)
        mlp_norm_layer = get_norm_layer(cfg.mlp_norm_layer)
        act_layer = get_act_layer(cfg.act_layer)

        self.patch_embed = PatchEmbed(
            img_size=cfg.input_size,
            patch_size=cfg.patch_size,
            in_chans=cfg.in_dim,
            embed_dim=cfg.dim,
            bias=True,
            strict_img_size=False,
            norm_layer=norm_layer,
        )
        if self.with_raw_img:
            assert cfg.raw_img_size is not None
            assert cfg.raw_img_chans is not None
            assert cfg.pos_embed_type == "sincos"
            self.raw_img_patcher = PatchEmbed(
                img_size=cfg.raw_img_size,
                patch_size=self.raw_patch_size,
                in_chans=cfg.raw_img_chans,
                embed_dim=cfg.dim,
                bias=True,
                strict_img_size=False,
            )
        # self.fuse_stem = nn.Linear(dim // self._n_modalities * self._n_modalities, dim)
        self.base_size = cfg.input_size // self.patch_size
        self.pe_interpolation = 1.0
        self.out_channels = cfg.out_channels
        self.num_heads = cfg.num_heads
        self.feature_layer_ids = cfg.feature_layer_ids
        if cfg.feature_layer_ids:
            assert max(cfg.feature_layer_ids) < cfg.depth, (
                "max feature_layer_id must be less than depth"
            )

        # layers
        layers = []
        drop_path_rates = [
            x.item() for x in torch.linspace(0, cfg.drop_path, cfg.depth)
        ]  # stochastic depth decay rule
        for i in range(cfg.depth):
            layers.append(
                AttentionBlock(
                    dim=cfg.dim,
                    num_heads=cfg.num_heads,
                    qkv_bias=True,
                    qk_norm=norm_layer,
                    norm_layer=mlp_norm_layer,
                    mlp_type="swiglu",
                    mlp_ratio=cfg.mlp_ratio,
                    drop=cfg.drop,
                    attn_drop=cfg.drop,
                    drop_path=(
                        drop_path_rates[i]
                        if isinstance(drop_path_rates, list)
                        else cfg.drop_path
                    ),
                    act_layer=act_layer,
                    use_layerscale=cfg.use_layerscale,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.head = nn.Sequential(
            norm_layer(cfg.dim),
            nn.Linear(cfg.dim, cfg.out_channels * cfg.patch_size**2, bias=True),
        )

        # positional embedding
        self.pos_embed_type = cfg.pos_embed_type
        self.setup_pe(cfg.dim)

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

        elif self.pos_embed_type == "rope_te":
            if rope_options is None:
                self.rope_options = {
                    "dim": dim // self.cfg.num_heads,
                    "rope_dim": "2D",
                    "beta_fast": 4,
                    "beta_slow": 1,
                    "rope_theta": 10000,
                    "apply_yarn": True,
                    "scale": 1.0,
                    "original_latent_shape": (self.base_size, self.base_size),
                }
            self.rope = RotaryPositionEmbeddingPytorchV2(  # ty: ignore error[missing-argument]
                seq_len=seq_len,
                latent_shape=(self.base_size, self.base_size),
                # does not changed
                **self.rope_options,
            )
        elif self.pos_embed_type == "rope":
            # rope implem from timm.
            self.rope_options = {
                "temperature": 10000.0,
                "in_pixels": False,
                "feat_shape": [self.base_size, self.base_size],
                "ref_feat_shape": [self.base_size, self.base_size],
            }
            # create the rope emb at forward
            self.rope = RotaryEmbeddingCat(
                dim=dim // self.num_heads, **self.rope_options
            )
        else:
            raise ValueError(
                f"Unsupported pos_embed_type: {self.pos_embed_type}. "
                "Supported types are 'sincos', 'rope_te' and 'rope'."
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
                # use resample_abs_pos_embed for interpolation like pansharpening model
                new_h, new_w = hw
                pos_embed = resample_abs_pos_embed(  # type: ignore
                    pe, new_size=(new_h, new_w), num_prefix_tokens=0
                )
                pe = torch.as_tensor(pos_embed).float()
            return pe

        elif self.pos_embed_type == "rope_te":
            # TODO: add multi-modal-rope
            # (modalities -> ids -> online RoPE class -> positional embedding -> kv rope fn)

            seq_len_x = h * w
            pre_rope_seq_len = self.rope.cos_cached.shape[1]
            if seq_len_x > pre_rope_seq_len:
                # re-init the rope
                self.rope_options["original_latent_shape"] = (h, w)
                self.rope.__init__(seq_len_x, **self.rope_options)
            return self.rope
        elif self.pos_embed_type == "rope":
            # TODO: add multi-modal-rope
            # (modalities -> ids -> online RoPE class -> positional embedding -> kv rope fn)
            ph, pw = h, w
            seq_len_x = ph * pw
            pe = self.rope.get_embed((ph, pw))
            return pe

        else:
            raise ValueError(
                f"Unsupported pos_embed_type: {self.pos_embed_type}. "
                "Supported types are 'sincos', 'rope_te' and 'rope'."
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
            x,
            "bs (h w) (p1 p2 c) -> bs c (h p1) (w p2)",
            h=h,
            w=w,
            p1=p,
            p2=p,
            c=c,
        )
        return x

    def forward(
        self,
        noisy_latent: Float[Tensor, "b c_latent h w"],  # noisy image latent
        hyper_img: Float[Tensor, "b c_hyper H W"] | None = None,  # raw image
        *,
        feature_layer_ids: list[int] | None = None,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        # patch embedding
        y = self.patch_embed(noisy_latent)

        # positional embedding
        hw = torch.tensor(noisy_latent.shape[-2:]) // self.patch_size
        pe = self.get_pe(tuple(hw))
        if self.pos_embed_type == "sincos":
            assert torch.is_tensor(pe), "Positional embedding must be a tensor."
            y = y + pe.to(y)
            pe = None

        if self.with_raw_img:
            assert hyper_img is not None
            y2 = self.raw_img_patcher(hyper_img)

            if self.pos_embed_type == "sincos":
                hw_raw = torch.tensor(hyper_img.shape[-2:]) // self.raw_patch_size
                pe2 = self.get_pe(tuple(hw_raw), img_type="raw")
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

    def init_weights(self, lin_init="trunc_normal"):
        # Initialize transformer layers
        def _basic_init(module):
            norms = [get_norm_layer(n) for n in ["layernorm", "simplenorm", "rmsnorm"]]
            if isinstance(module, nn.Linear):
                if lin_init == "trunc_normal":
                    nn.init.trunc_normal_(module.weight, std=0.02)
                elif lin_init == "xavier":
                    torch.nn.init.xavier_normal_(module.weight)
                else:
                    raise ValueError(f"Unsupported lin_init: {lin_init}")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, tuple(norms)):
                nn.init.constant_(module.weight, 1.0)
                if hasattr(module, "bias") and module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.layers.apply(_basic_init)

        # patch embedding
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.zeros_(self.patch_embed.proj.bias)

        # initialize the head properly
        norm, lin = self.head
        # For linear layer: use small random initialization instead of zeros
        torch.nn.init.trunc_normal_(lin.weight, std=0.02)
        torch.nn.init.zeros_(lin.bias)
        # For normalization layer: weight should be 1.0, not 0.0
        if hasattr(norm, "weight"):
            torch.nn.init.ones_(norm.weight)
        if hasattr(norm, "bias") and norm.bias is not None:
            torch.nn.init.zeros_(norm.bias)

        logger.info("[Transformer] Initializing model ...")

    @classmethod
    @function_config_to_easy_dict
    def create_model(cls, **kwargs):
        cfg = TransformerConfig(**kwargs)
        model = cls(cfg)
        return model
