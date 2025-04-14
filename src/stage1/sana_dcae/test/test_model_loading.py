import sys

import accelerate
import ipdb
import PIL.Image
import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count_table
from tqdm import trange

sys.path.insert(0, __file__[: __file__.find("src")])
from src.stage1.sana_dcae.ae_model_zoo import (
    create_dc_ae_model_cfg,
)
from src.stage1.sana_dcae.models.efficientvit.dc_ae import (
    DCAE,
    dc_ae_f8c16,
    dc_ae_f32c32,
)
from src.utilities.optim.sana_came import CAME8BitWrapper, CAMEWrapper

# ae_cfg = create_dc_ae_model_cfg("dc-ae-f8c16")
# ae_cfg = dc_ae_f8c16("dc-ae-f8c16", None)

# sana dc-ae
ckpt_path = "/Data2/ZiHanCao/exps/hyperspectral-1d-tokenizer/src/stage1/sana_dcae/pretrained/dc-ae-f32c32-sana-1.0/model.safetensors"
ae_cfg = dc_ae_f32c32(
    "dc-ae-f32c32-sana-1.0",
    ckpt_path,
    extra={
        "in_channels": 3,
        "encoder": {"in_channels": 3},
        "decoder": {"in_channels": 3},
    },
)

# Verify config structure
assert hasattr(ae_cfg, "in_channels")
assert hasattr(ae_cfg, "latent_channels")
assert hasattr(ae_cfg, "encoder")
assert hasattr(ae_cfg, "decoder")
ae_cfg.encoder.act_checkpoint = True
ae_cfg.decoder.act_checkpoint = True
print(ae_cfg)

img_size = 512
bs = 8
dtype = torch.bfloat16
ae = DCAE(ae_cfg).cuda().to(torch.bfloat16)
print("create tokenizer done")

# ae = ae.float()

print("load ckpt ...")
import accelerate

# fake tensor
# x = torch.randn(bs, 3, img_size, img_size).cuda().to(dtype)
# real image
import numpy as np
import PIL

x = PIL.Image.open("/HardDisk/ZiHanCao/datasets/VIF-LLVIP/data/fused/train/010008.jpg")
x = np.array(x.resize((img_size, img_size))) / 255.0
# norm into (-1, 1)
x = x * 2 - 1
x = torch.as_tensor(x).permute(-1, 0, 1)[None].cuda().to(dtype)
# if repeat
x = x.repeat(bs, 1, 1, 1)
print(f"read image shaped as {x.shape}")

# pass into tokenizer
# ae = ae.eval()
# with torch.no_grad():
# with torch.autocast("cuda", torch.bfloat16):
#     z = ae.encode(x)
#     recon = ae.decode(z)
#     recon = recon.float()
#     print(f"===> z shaped as {z.shape}, reconstruction shaped as {recon.shape}")

# backward grad
# print("===> start backward")
# recon.mean().backward()

# for n, p in ae.named_parameters():
#     if p.requires_grad:
#         if p.grad is None:
#             print(n + " has no grad")
#         elif p.grad.isnan().any():
#             print(n + " has NaN grad")
#         else:
#             # print(n + " has grad")
#             p.grad = None

# PSNR
# from skimage.metrics import peak_signal_noise_ratio as psnr

# psnr_value = psnr(x.cpu().numpy(), recon.clone().detach().cpu().numpy())

# print(f"===> PSNR: {psnr_value}")


# # plot the reconstruction and original image
# import matplotlib.pyplot as plt

# # Move tensors to CPU and convert to numpy for plotting
# original_image = x[0].permute(1, 2, 0).cpu().numpy()  # Reshape to (H, W, C)
# reconstructed_image = recon[0].permute(1, 2, 0).cpu().numpy()  # Reshape to (H, W, C)

# # Clip values to ensure they are in the valid range for visualization
# original_image = np.clip(original_image, -1, 1)  # Ensure values are in [-1, 1]
# reconstructed_image = np.clip(
#     reconstructed_image, -1, 1
# )  # Ensure values are in [-1, 1]

# # Denormalize images back to [0, 1] for display
# original_image = (original_image + 1) / 2
# reconstructed_image = (reconstructed_image + 1) / 2

# # Plotting
# fig, axes = plt.subplots(
#     1, 2, figsize=(12, 6)
# )  # Create a subplot with 1 row and 2 columns

# # Original Image
# axes[0].imshow(original_image)
# axes[0].set_title("Original Image")
# axes[0].axis("off")  # Turn off axis labels

# # Reconstructed Image
# axes[1].imshow(reconstructed_image)
# axes[1].set_title("Reconstructed Image")
# axes[1].axis("off")  # Turn off axis labels

# # Show the plot
# plt.tight_layout()  # Adjust layout to prevent overlap

# # save
# plt.savefig("original and reconstructed image from DC_AE_f32_d32.png")

# * optimizers


print("=" * 30)
# import time

# time.sleep(2)

# # 创建优化器
# optimizer = torch.optim.AdamW(ae.parameters(), lr=1e-4, foreach=True)
# optimizer = CAMEWrapper(
#     ae.parameters(),
#     lr=1e-4,
#     eps=(1e-30, 1e-16),
#     clip_threshold=1.0,
#     betas=(0.9, 0.999, 0.9999),
#     weight_decay=0.0,
# )
optimizer = CAME8BitWrapper(
    ae.parameters(),
    lr=1e-4,
    eps=(1e-30, 1e-16),
    clip_threshold=1.0,
    betas=(0.9, 0.999, 0.9999),
    weight_decay=0.0,
    block_size=2048,  # Quantization block size
    min_8bit_size=int(64 * 64),  # Minimum parameter size to use 8-bit
)

for _ in trange(20):
    with torch.autocast("cuda", dtype):
        # 前向传播
        # recon = ae(x)
        z = ae.encode(x)
        recon = ae.decode(z)
        recon = recon.float()
        loss = recon.mean()

    # 计算损失并反向传播
    optimizer.zero_grad()
    loss.backward()
    _n = 0
    for n, p in ae.named_parameters():
        if p.requires_grad:
            if p.grad is None:
                _n += 1
                print(n + " has no grad")
            elif p.grad.isnan().any():
                _n += 1
                print(n + " has NaN grad")
            else:
                # print(n + " has grad")
                p.grad = None

    print(
        f"==> {_n} parameters has None or NaN grad, total {len(list(ae.parameters()))} parameters"
    )
    optimizer.step()


# print(flop_count_table(FlopCountAnalysis(ae, x)))

# print(parameter_count_table(ae))
