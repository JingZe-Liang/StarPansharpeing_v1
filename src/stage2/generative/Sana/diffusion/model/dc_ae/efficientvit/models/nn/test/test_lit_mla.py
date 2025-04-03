import sys

import torch

sys.path.insert(0, "/Data4/cao/ZiHanCao/exps/HyperspectralTokenizer")
from src.stage2.generative.Sana.diffusion.model.dc_ae.efficientvit.ae_model_zoo import (
    create_dc_ae_model_cfg,
)

ae = create_dc_ae_model_cfg("dc-ae-f32c32-sana-1.0")

import torchinfo

torchinfo.summary(ar, input_size=(1, 3, 256, 256))
