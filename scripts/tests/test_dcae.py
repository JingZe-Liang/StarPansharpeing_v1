import sys

import torch
from omegaconf import OmegaConf

sys.path.append(__file__[: __file__.find("scripts")])
from src.stage1.sana_dcae.models.efficientvit.dc_ae import (
    DCAE,
    dc_ae_f16c16,
    dc_ae_f8c16_pure_conv,
    dc_ae_f16c16_pure_conv,
)
from src.utilities.optim import get_moun_optimizer

# model
extra_cfg = OmegaConf.create(
    {
        "embed_dim": 16,
        "beta": 0.0,
        "gamma0": 1.0,
        "gamma": 1.0,
        "zeta": 1.0,
        "inv_temperature": 1.0,
        "cb_entropy_compute": "group",
        "l2_norm": True,
        "input_format": "bchw",
        "persample_entropy_compute": "analytical",
        "group_size": 1,
    }
)
extra_cfg = OmegaConf.create(
    {
        "quant_cfg": extra_cfg,
        "quant_type": "bsq",
        "use_quant_conv": True,
        "repa_hidden_size": 512,
    }
)
cfg = dc_ae_f16c16_pure_conv(name="dc-ae-f16c16", extra=extra_cfg, pretrained_path=None)
model = DCAE(cfg).cuda(1)

opt = get_moun_optimizer(
    model.named_parameters(),
    lr=1e-3,
    weight_decay=1e-2,
    adamw_betas=(0.95, 0.99),
    use_cuda_kernel=False,
)
# opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2, foreach=True)

print(opt)

x = torch.randn(1, 3, 256, 256).cuda(1)

import time

t1 = time.time()
for _ in range(10):
    y = model(x)[0]
    opt.zero_grad
    # print(y)
    y.sum().backward()
    opt.step()
    print("optimizer step")

t2 = time.time()
print("time for optimizer: ", t2 - t1)

# from fvcore.nn import parameter_count_table

# print(parameter_count_table(model))
