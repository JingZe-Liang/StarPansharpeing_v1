import sys

import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count_table
from tqdm import trange

sys.path.insert(0, "/Data4/cao/ZiHanCao/exps/HyperspectralTokenizer")
from src.stage1.sana_dcae.ae_model_zoo import (
    create_dc_ae_model_cfg,
)
from src.stage1.sana_dcae.models.efficientvit.dc_ae import DCAE, dc_ae_f8c16
from src.utilities.optim.sana_came import CAME8BitWrapper, CAMEWrapper

torch.cuda.set_device(1)
# ae_cfg = create_dc_ae_model_cfg("dc-ae-f8c16")
ae_cfg = dc_ae_f8c16("dc-ae-f8c16", None)

# Verify config structure
assert hasattr(ae_cfg, "in_channels")
assert hasattr(ae_cfg, "latent_channels")
assert hasattr(ae_cfg, "encoder")
assert hasattr(ae_cfg, "decoder")

print(ae_cfg)

img_size = 256
bs = 1
ae = DCAE(ae_cfg).cuda().to(torch.bfloat16)
x = torch.randn(bs, 3, img_size, img_size).cuda().to(torch.bfloat16)

# print(ae.get_last_layer().shape)


# # 创建优化器
optimizer = torch.optim.AdamW(ae.parameters(), lr=1e-4, foreach=True)
# optimizer = CAMEWrapper(
#     ae.parameters(),
#     lr=1e-4,
#     eps=(1e-30, 1e-16),
#     clip_threshold=1.0,
#     betas=(0.9, 0.999, 0.9999),
#     weight_decay=0.0,
# )
# optimizer = CAME8BitWrapper(
#     ae.parameters(),
#     lr=1e-4,
#     eps=(1e-30, 1e-16),
#     clip_threshold=1.0,
#     betas=(0.9, 0.999, 0.9999),
#     weight_decay=0.0,
#     block_size=2048,  # Quantization block size
#     min_8bit_size=int(64 * 64),  # Minimum parameter size to use 8-bit
# )

for _ in trange(100):
    with torch.autocast("cuda", torch.bfloat16):
        # 前向传播
        recon = ae(x)

        # 计算损失并反向传播
        loss = recon.mean()
        optimizer.zero_grad()
        loss.backward()
        for n, p in ae.named_parameters():
            if p.requires_grad and p.grad is None:
                print(n + " has no grad")
            elif p.grad.isnan().any():
                print(n + " has NaN grad")
            # else:
            #     print(n + " has grad")
        exit()
        optimizer.step()


# print(flop_count_table(FlopCountAnalysis(ae, x)))

# print(parameter_count_table(ae))
