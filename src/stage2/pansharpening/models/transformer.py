from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Array, Float
from timm.layers import get_act_layer, get_norm_layer
from timm.layers.patch_embed import PatchEmbed
from torch import Tensor

from src.utilities.logging import log

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
    act_layer: str = "swiglu"
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
        self._n_modalities = 2
        self.patch_embed = PatchEmbed(
            img_size=cfg.input_size,
            patch_size=cfg.patch_size,
            in_chans=cfg.in_dim,
            embed_dim=cfg.dim // self._n_modalities,
            bias=True,
            strict_img_size=False,
        )
        self.fuse_stem = nn.Linear(
            cfg.dim // self._n_modalities * self._n_modalities, cfg.dim
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
            layers.append(
                AttentionBlock(
                    dim=cfg.dim,
                    mlp_ratio=cfg.mlp_ratio,
                    num_heads=cfg.num_heads,
                    qkv_bias=True,
                    qk_norm=norm_layer,
                    drop=cfg.drop,
                    attn_drop=cfg.drop,
                    drop_path=drop_path_rates[i]
                    if isinstance(drop_path_rates, list)
                    else cfg.drop_path,
                    norm_layer=mlp_norm_layer,
                    act_layer=act_layer,
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
            self.pos_embed: nn.Buffer
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
        else:
            raise ValueError(
                f"Unsupported pos_embed_type: {self.pos_embed_type}. "
                "Supported types are 'sincos' and 'rope'."
            )

    def get_pe(self, hw: tuple | torch.Size, img_type=None):
        h, w = hw
        if self.pos_embed_type == "sincos":
            pe = self.pos_embed
            name = "pos_embed"
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
        ms_latent: Float[Tensor, "b c h w"],
        pan_latent: Float[Tensor, "b c h w"],
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        latents = (ms_latent, pan_latent)
        # patch embedding
        ys = []
        for latent in latents:
            y = self.patch_embed(latent)
            ys.append(y)

        # fuse all latent
        y = self.fuse_stem(torch.cat(ys, dim=-1))

        # pe
        pe = self.get_pe(latents[0].shape[-2:])
        if self.pos_embed_type == "sincos":
            assert torch.is_tensor(pe), "Positional embedding must be a tensor."
            y = y + pe.to(y)
            pe = None

        # blocks
        features = []
        for i, layer in enumerate(self.layers):
            y = layer(y, mask=None, pe=pe)
            if self.feature_layer_ids is not None and i in self.feature_layer_ids:
                features.append(y)

        # head
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
            norms = [get_norm_layer(n) for n in ["layernorm", "simplenorm", "rmsnorm"]]
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_normal_(module.weight)
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

        # zero-out the head
        norm, lin = self.head
        torch.nn.init.zeros_(lin.weight)
        torch.nn.init.zeros_(lin.bias)
        torch.nn.init.zeros_(norm.weight)
        if hasattr(norm, "bias") and norm.bias is not None:
            torch.nn.init.zeros_(norm.bias)

        log("[Transformer] Initializing model ...")


if __name__ == "__main__":
    device = "cpu"
    # torch.cuda.set_device(device)
    # x = torch.randn(1, 768, 128).to(device)

    # rmsnorm = RMSNorm(dim=128)
    # print(rmsnorm(x).shape)

    # attn = Attention(128, 8, qk_norm=RMSNormFlash).to(device)
    # print(attn(x).shape)

    # mlp = ClipMlp(
    #     in_features=128,
    #     hidden_features=256,
    #     out_features=128,
    #     norm_layer=RMSNormFlash,
    # ).to(device)
    # print(mlp(x).shape)

    from fvcore.nn import parameter_count_table

    # Create config for Transformer
    cfg = TransformerConfig(
        in_dim=16,
        dim=384,
        depth=8,
        num_heads=8,
        mlp_ratio=4,
        out_channels=16,
        pos_embed_type="sincos",
        norm_layer="rmsnorm",
        input_size=32,
    )
    model = Transformer(cfg).to(device)
    print(parameter_count_table(model))

    x = torch.randn(1, 16, 64, 64).to(device)
    out = model(x, x)
    print(out.shape)  # Expected shape: (1, 16, 32, 32)
    print(out)
