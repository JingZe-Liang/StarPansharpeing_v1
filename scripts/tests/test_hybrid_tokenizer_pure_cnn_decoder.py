import hydra
import torch
from omegaconf import OmegaConf


def load_config():
    config_path = "scripts/configs/tokenizer_gan/tokenizer/cosmos_hybrid_pure_cnn_decoder_f16c64.yaml"
    cfg = OmegaConf.load(config_path)
    return cfg


def main():
    # Load model configuration
    print("Loading model configuration...")
    cfg = load_config()

    # Create model
    print("Creating model...")
    model = hydra.utils.instantiate(cfg)
    print("Model created successfully.")

    # Print model info
    from fvcore.nn import FlopCountAnalysis, parameter_count_table

    print(parameter_count_table(model))

    # Test model with dummy input
    print("Testing model with dummy input...")
    x = torch.randn(1, 3, 256, 256).cuda() * 0.2
    model = model.cuda()
    model.eval()
    with torch.no_grad():
        with torch.autocast("cuda", torch.bfloat16):
            y = model(x)
            print(y.shape)


if __name__ == "__main__":
    main()
