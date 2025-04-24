import sys

import torch

sys.path.insert(0, __file__[: __file__.find("src")])
from src.stage1.LeanVAE.LeanVAE.models.autoencoder import (
    LeanVAE,
    LeanVAE2D,
    LeanVAEConfig,
)


def main():
    # cfg = LeanVAEConfig()
    # LeanVAE_model = LeanVAE(cfg)

    # print(LeanVAE_model)

    # * vae 2d
    cfg = LeanVAEConfig()
    model_2d = LeanVAE2D(cfg)

    # * params
    # from fvcore.nn import parameter_count_table

    # print(parameter_count_table(model_2d))

    # print(model_2d)

    # test the 2d input
    x = torch.randn(1, 3, 256, 256)
    enc = model_2d.encode(x)
    print(enc.shape)

    y = model_2d(x)

    # x, recon = y
    # print(recon.shape)


if __name__ == "__main__":
    main()
