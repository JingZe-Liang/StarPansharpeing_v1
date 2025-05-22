# import torch

# gn = torch.nn.GroupNorm(num_groups=1, num_channels=3, eps=1e-6, affine=True)


# class LN2D(torch.nn.Module):
#     def __init__(self, shape: int | tuple, num_groups=1):
#         super().__init__()
#         self.norm = torch.nn.LayerNorm(
#             normalized_shape=shape,
#             eps=1e-6,
#             elementwise_affine=True,
#         )

#     def forward(self, x):
#         if len(self.norm.normalized_shape) == 1:
#             return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
#         else:
#             return self.norm(x)


# ln2d = LN2D((3, 32, 32), 1)

# x = torch.arange(3 * 32 * 32).float()
# x = x.reshape(3, 32, 32)[None]
# x_gn = gn(x)
# x_ln = ln2d(x)


# # assert close
# close = torch.allclose(x_gn, x_ln, atol=1e-6)
# print(f"GroupNorm and LayerNorm2D are close: {close}")  # True


# ln2d_only_channel = LN2D(3, 1)
# x_ln_c = ln2d_only_channel(x)
# close = torch.allclose(x_gn, x_ln_c, atol=1e-6)
# print(f"GroupNorm and LayerNorm2D only channel are close: {close}")  # False


from safetensors.torch import load_file, save_file


d = load_file(
    "/Data4/cao/ZiHanCao/exps/HyperspectralTokenizer/runs/stage1_cosmos/cosmos_f8c16p4_psnr_39/ema/tokenizer/model.safetensors"
)
d["decoder.quant_conv.weight"] = d["decoder.post_quant_conv.weight"]
d["decoder.quant_conv.bias"] = d["decoder.post_quant_conv.bias"]

del d["decoder.post_quant_conv.weight"]
del d["decoder.post_quant_conv.bias"]

save_file(
    d,
    "/Data4/cao/ZiHanCao/exps/HyperspectralTokenizer/runs/stage1_cosmos/cosmos_f8c16p4_psnr_39/ema/tokenizer/model2.safetensors",
)
print("done.")
