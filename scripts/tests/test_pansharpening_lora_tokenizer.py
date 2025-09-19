import hydra
import torch


@hydra.main(
    config_path="../configs/pansharpening",
    config_name="cosmos_f8c16p4_fusionnet",
    version_base=None,
)
def main(cfg):
    model = hydra.utils.instantiate(cfg.pansharp_model)
    print(model)


if __name__ == "__main__":
    main()
