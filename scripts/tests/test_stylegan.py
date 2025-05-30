import torch

from src.stage1.utilities.losses.model.stylegan import StyleGANDiscriminator

disc = StyleGANDiscriminator(
    0,
    256,
    3,
    architecture="resnet",
    num_fp16_res=8,
    epilogue_kwargs={"mbstd_group_size": 3},
).cuda()

x = torch.randn(1, 3, 256, 256).cuda()

print(disc(x).shape)
