import hydra
import torch
from omegaconf import OmegaConf

from src.utilities.config_utils import clear_hydra_self_config


# @hydra.main(
#     config_path="../configs/tokenizer_gan/tokenizer",
#     config_name="comos_hybrid_f16c64",
#     version_base=None,
# )
def main():
    import os
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    # 清除现有Hydra实例
    GlobalHydra.instance().clear()

    # 获取绝对路径
    config_path = "../configs/tokenizer_gan"

    with initialize_config_dir(
        config_dir=config_path,  # initialize_config_dir需要绝对路径
        version_base=None,
        job_name="app",
    ):
        cfg = compose(
            config_name="unicosmos_gen_tokenizer_f8c16p1.yaml",  # 使用正确的配置文件名
            overrides=[
                "hydra.output_subdir=null",
                "hydra/hydra_logging=disabled",
                "hydra/job_logging=disabled",
            ],
        )
        print(cfg)

    model = hydra.utils.instantiate(cfg)
    print(model)

    x = torch.randn(1, 3, 256, 256).cuda()
    model = model.cuda()
    with torch.no_grad():
        encoded = model.encode(x)
        print(encoded)
        print("----")
        decoded = model.decode(encoded, (1, 3))
        print(decoded)


if __name__ == "__main__":
    main()
