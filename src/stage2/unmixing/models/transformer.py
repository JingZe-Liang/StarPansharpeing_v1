from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float
from loguru import logger
from timm.layers import get_act_layer, get_norm_layer

# Attention, MLP
from timm.layers.patch_embed import PatchEmbed
from timm.layers.pos_embed import resample_abs_pos_embed
from timm.layers.pos_embed_sincos import RotaryEmbeddingCat
from torch import Tensor

from src.utilities.config_utils import dataclass_from_dict

from ...layers import (
    AttentionBlock,
    LiteLA_GLUMB_Block,
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
    mlp_ratio: float = 4.0
    drop: float = 0.0
    drop_path: float = 0.0
    input_size: int = 32
    patch_size: int = 2
    with_raw_img: bool = False
    raw_img_size: int = 256
    raw_img_chans: int = 16
    raw_patch_size: int = 4
    pos_embed_type: str = "sincos"
    norm_layer: str = "flarmsnorm"
    mlp_norm_layer: str = "flarmsnorm"
    act_layer: str = "gelu"
    feature_layer_ids: Optional[list[int]] = None
    block_type: str = "AttentionBlock"  # or "LiteLA_GLUMB_Block"
    mlp_type: str = "mlp"  # or "glu_mb", "mlp", 'glumlp', 'swiglu'


class Transformer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()

        # Store config for reference
        self.cfg = cfg

        # patch embedding
        self.patch_size = cfg.patch_size
        self.input_size = cfg.input_size
        self.num_patches = (cfg.input_size // cfg.patch_size) ** 2
        self.patch_embed = PatchEmbed(
            img_size=cfg.input_size,
            patch_size=cfg.patch_size,
            in_chans=cfg.in_dim,
            embed_dim=cfg.dim,
            bias=True,
            strict_img_size=False,
        )
        self.with_raw_img = cfg.with_raw_img
        if self.with_raw_img:
            assert cfg.raw_img_size is not None
            assert cfg.raw_img_chans is not None
            assert cfg.pos_embed_type == "sincos"
            self.raw_img_patcher = PatchEmbed(
                img_size=cfg.raw_img_size,
                patch_size=cfg.raw_patch_size,
                in_chans=cfg.raw_img_chans,
                embed_dim=cfg.dim,
                bias=True,
                strict_img_size=False,
            )
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
        norm_layer = get_norm_layer(cfg.norm_layer)
        mlp_norm_layer = get_norm_layer(cfg.mlp_norm_layer)
        act_layer = get_act_layer(cfg.act_layer)
        for i in range(cfg.depth):
            if cfg.block_type == "LiteLA_GLUMB_Block":
                block = LiteLA_GLUMB_Block(
                    hidden_size=cfg.dim,
                    mlp_ratio=cfg.mlp_ratio,
                    drop_path=drop_path_rates[i],
                    qk_norm=False,
                    norm_type=cfg.norm_layer,
                    linear_head_dim=32,
                    mlp_type=cfg.mlp_type,
                    ffn_drop=cfg.drop,
                )
                if cfg.mlp_type == "glu_mb":
                    # fuse conv down to dim
                    self.fused_conv = nn.Linear(cfg.dim * 2, cfg.dim)
            elif cfg.block_type == "AttentionBlock":
                block = AttentionBlock(
                    dim=cfg.dim,
                    mlp_ratio=cfg.mlp_ratio,
                    num_heads=cfg.num_heads,
                    qkv_bias=True,
                    qk_norm=norm_layer,
                    drop=cfg.drop,
                    attn_drop=cfg.drop,
                    drop_path=(
                        drop_path_rates[i]
                        if isinstance(drop_path_rates, list)
                        else cfg.drop_path
                    ),
                    norm_layer=mlp_norm_layer,
                    act_layer=act_layer,
                    mlp_type=cfg.mlp_type,
                )
            else:
                raise ValueError(f"Unsupported block_type: {cfg.block_type}")

            layers.append(block)
        self.layers = nn.ModuleList(layers)
        self.head = nn.Sequential(
            norm_layer(cfg.dim),
            nn.Linear(cfg.dim, cfg.out_channels * cfg.patch_size**2, bias=True),
        )

        # positional embedding
        self.pos_embed_type = cfg.pos_embed_type
        self.setup_pe(cfg.dim)

        self.init_weights()

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    def setup_pe(self, dim, rope_options: dict | None = None):
        seq_len = self.num_patches

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
                torch.as_tensor(pos_embed).float().unsqueeze(0)
            )
            if self.with_raw_img:
                pos_embed_raw = get_2d_sincos_pos_embed(
                    dim,
                    int(self.cfg.raw_img_size // self.cfg.raw_patch_size),
                    pe_interpolation=self.pe_interpolation,
                    base_size=self.cfg.raw_img_size // self.cfg.raw_patch_size,
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
                "in_pixel": False,
                "feat_shape": [self.base_size, self.base_size],
            }
            self.rope = RotaryEmbeddingCat(
                dim=dim // self.num_heads, **self.rope_options
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
                base_size = self.cfg.raw_img_size // self.cfg.raw_patch_size
            else:
                pe = self.pos_embed_latent
                name = "pos_embed_latent"
                base_size = self.base_size
            # breakpoint()
            if pe.shape[1] != h * w:
                new_size = hw
                pe = resample_abs_pos_embed(  # type: ignore
                    pe,
                    new_size=new_size,
                    old_size=(base_size, base_size),
                    num_prefix_tokens=0,
                )
            return pe
        elif self.pos_embed_type == "rope":
            # TODO: add multi-modal-rope
            # (modalities -> ids -> online RoPE class -> positional embedding -> kv rope fn)

            seq_len_x = h * w
            pre_rope_seq_len = self.rope.cos_cached.shape[1]
            if seq_len_x > pre_rope_seq_len:
                # re-init the rope
                self.rope_options["latent_shape"] = (h, w)
                self.rope.__init__(seq_len_x, **self.rope_options)
            return self.rope
        elif self.pos_embed_type == "rope":
            ph, pw = h // self.patch_size, w // self.patch_size
            seq_len_x = ph * pw
            pe = self.rope.get_embed((ph, pw))
            return pe
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
        latent: Float[Tensor, "b c_latent h w"],
        hyper_img: Float[Tensor, "b c_hyper H W"] | None = None,
        *,
        feature_layer_ids: list[int] | None = None,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        # patch embedding
        y = self.patch_embed(latent)
        latent_s = y.shape[1]

        # positional embedding
        hw = torch.tensor(latent.shape[-2:]) // self.patch_size
        pe = self.get_pe(tuple(hw), img_type="latent")
        if self.pos_embed_type == "sincos":
            assert torch.is_tensor(pe), "Positional embedding must be a tensor."
            y = y + pe.to(y)
            pe = None

        # Hyper image pe
        if self.with_raw_img:
            assert hyper_img is not None
            y2 = self.raw_img_patcher(hyper_img)

            if self.pos_embed_type == "sincos":
                hw = torch.tensor(hyper_img.shape[-2:]) // self.cfg.raw_patch_size
                pe2 = self.get_pe(tuple(hw), img_type="raw")
                assert torch.is_tensor(pe2), "Positional embedding must be a tensor."
                y2 = y2 + pe2.to(y2)

            # cat
            if (
                self.cfg.block_type == "LiteLA_GLUMB_Block"
                and self.cfg.mlp_type == "glu_mb"
            ):
                assert y.shape[:-1] == y2.shape[:-1], (
                    f"{y.shape=} and {y2.shape=} to cat on channel dim."
                )
                y = self.fused_conv(torch.cat([y, y2], dim=-1))  # [bs, s, c]
            else:
                y = torch.cat([y, y2], dim=1)  # [bs, s1 + s2, c]

        # blocks
        feature_layer_ids_ = self.feature_layer_ids or feature_layer_ids
        features = []
        for i, layer in enumerate(self.layers):
            y = layer(y, mask=None, pe=pe, HW=hw)
            # logger.debug(f"Layer {i}: {y.norm()=}")
            if feature_layer_ids_ is not None and i in feature_layer_ids_:
                features.append(y)

        # head
        y = y[:, :latent_s]  # take the latent part
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
        if self.with_raw_img:
            w = self.raw_img_patcher.proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        logger.info(f"[Unmixing Transformer] Initialized weights")

    @classmethod
    def create_model(cls, **kwargs):
        cfg = dataclass_from_dict(TransformerConfig, kwargs)
        model = cls(cfg)
        return model


def test_model():
    model = Transformer(
        TransformerConfig(
            in_dim=16,
            out_channels=256,
            dim=256,
            depth=8,
            num_heads=8,
            with_raw_img=True,
            raw_img_size=256,
            raw_img_chans=16,
            raw_patch_size=8,
            input_size=32,
            patch_size=1,
            block_type="AttentionBlock",
            mlp_type="mlp",
        )
    ).cuda()
    latent = torch.randn(2, 16, 32, 32).cuda()
    hyper_img = torch.randn(2, 16, 256, 256).cuda()
    out = model(latent, hyper_img)
    print(out.shape)


if __name__ == "__main__":
    """
    python -m src.stage2.unmixing.models.transformer
    """
    test_model()
