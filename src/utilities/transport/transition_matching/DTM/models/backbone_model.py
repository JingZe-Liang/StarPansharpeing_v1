# backbone_model.py
import torch
from torch import nn
import math
from timm.models.vision_transformer import Attention, Mlp


class TimestepEmbedder(nn.Module):
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
        t_freq = self.positional_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class ImageCondEncoder(nn.Module):
    def __init__(self, output_dim, cond_channels=5, image_cond_drop_prob=0.1):
        super().__init__()
        self.dropout_prob = image_cond_drop_prob

        # 轻量级CNN编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(cond_channels, 64, 3, 2, 1),
            nn.GroupNorm(16, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.GroupNorm(32, 256),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        encoder_output_dim = 256
        self.projection = nn.Linear(encoder_output_dim, output_dim)

        self.uncond_embedding = nn.Parameter(torch.randn(1, output_dim))

    def forward(self, image_cond, train=False):
        use_unconditional = image_cond is None or (
            train and self.dropout_prob > 0 and torch.rand(1) < self.dropout_prob
        )
        if use_unconditional:
            batch_size = 1
            if image_cond is not None:
                batch_size = image_cond.shape[0]
            return self.uncond_embedding.expand(batch_size, -1)

        features = self.encoder(image_cond)
        return self.projection(features)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SiTBlock(nn.Module):
    """SiT Transformer Block with adaLN-Zero conditioning."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class BackboneSiT(nn.Module):
    def __init__(
        self,
        num_patches=256,
        in_channels=4,
        patch_size=8,
        hidden_size=256,
        depth=6,
        num_heads=8,
        cond_channels=5,
        image_cond_drop_prob=0.1,
        **block_kwargs,
    ):
        super().__init__()

        patch_dim = patch_size * patch_size * in_channels
        self.patch_projector = nn.Linear(patch_dim, hidden_size)

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size))

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.cond_embedder = ImageCondEncoder(hidden_size, cond_channels, image_cond_drop_prob)

        self.blocks = nn.ModuleList(
            [SiTBlock(hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs) for _ in range(depth)]
        )

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def forward(self, x_seq, t, conditions=None, train=True):
        x = self.patch_projector(x_seq) + self.pos_embed

        t_embed = self.t_embedder(t)
        cond_embed = self.cond_embedder(conditions, train)
        if cond_embed.shape[0] != x.shape[0]:
            cond_embed = cond_embed.expand(x.shape[0], -1)

        c = t_embed + cond_embed

        for block in self.blocks:
            x = block(x, c)

        return x


def BackboneSiT_Pansharpening(image_size=64, in_channels=4, patch_size=8, **kwargs):
    num_patches = (image_size // patch_size) ** 2
    return BackboneSiT(
        num_patches=num_patches,
        in_channels=in_channels,
        patch_size=patch_size,
        depth=6,
        hidden_size=512,
        num_heads=8,
        **kwargs,
    )


Backbone_models = {
    "Backbone-Pansharpening": BackboneSiT_Pansharpening,
}
