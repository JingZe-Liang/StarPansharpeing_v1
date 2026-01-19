# head_model.py

import torch
from torch import nn
import math


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


class AdaLN_MLP_Block(nn.Module):
    def __init__(self, hidden_dim, mlp_ratio=4.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
        )

    def forward(self, x, scale, shift):
        x_norm = self.norm(x)
        x_modulated = x_norm * (1 + scale) + shift

        x_out = self.mlp(x_modulated)
        return x + x_out


class HeadMLP(nn.Module):
    def __init__(
        self,
        backbone_h_dim: int,
        patch_dim: int,
        mlp_hidden_dim: int = 1024,
        num_layers: int = 6,
        time_emb_dim: int = 256,
    ):
        super().__init__()
        self.num_layers = num_layers

        self.context_projector = nn.Linear(backbone_h_dim, mlp_hidden_dim)

        self.t_embedder = TimestepEmbedder(time_emb_dim)
        self.s_embedder = TimestepEmbedder(time_emb_dim)

        self.input_projector = nn.Linear(patch_dim, mlp_hidden_dim)

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, mlp_hidden_dim * 2 * num_layers))

        self.mlp_blocks = nn.ModuleList([AdaLN_MLP_Block(mlp_hidden_dim) for _ in range(num_layers)])

        self.output_layer = nn.Sequential(nn.LayerNorm(mlp_hidden_dim), nn.Linear(mlp_hidden_dim, patch_dim))

    def forward(self, h_t, t, Y_s, s):
        if s.dim() == 0:
            s = s.expand(h_t.shape[0])

        t_emb = self.t_embedder(t)
        s_emb = self.s_embedder(s)
        time_emb = t_emb + s_emb

        mod_params = self.adaLN_modulation(time_emb)
        all_scales_shifts = mod_params.chunk(2 * self.num_layers, dim=1)

        h_t_proj = self.context_projector(h_t)
        Y_s_proj = self.input_projector(Y_s)
        x = h_t_proj + Y_s_proj

        for i in range(self.num_layers):
            scale = all_scales_shifts[i * 2]
            shift = all_scales_shifts[i * 2 + 1]
            x = self.mlp_blocks[i](x, scale, shift)

        u = self.output_layer(x)

        return u
