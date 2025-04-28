import sys

import torch
from omegaconf import OmegaConf

sys.path.append(__file__[: __file__.find("scripts")])
from src.stage1.sana_dcae.models.efficientvit.dc_ae import DCAE, dc_ae_f16c16

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
cfg = dc_ae_f16c16(name="dc-ae-f16c16", extra=extra_cfg, pretrained_path=None)
model = DCAE(cfg).cuda(1)

x = torch.randn(1, 3, 512, 512).cuda(1)
y = model(x)
print(y)
