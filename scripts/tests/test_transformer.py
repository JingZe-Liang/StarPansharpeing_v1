import torch

from src.stage2.pansharpening.models.transformer import Transformer, TransformerConfig

device = "cuda:1"
cfg = TransformerConfig(
    in_dim=16,
    dim=128,
    depth=4,
    num_heads=8,
    out_channels=16,
    pos_embed_type="sincos",
    norm_layer="layernorm",
    input_size=32,
)
model = Transformer(cfg).to(device)
x = torch.randn(1, 16, 64, 64).to(device)
out = model(x, x)
print(out.shape)  # Expected shape: (1, 16, 32, 32)
print(out)
