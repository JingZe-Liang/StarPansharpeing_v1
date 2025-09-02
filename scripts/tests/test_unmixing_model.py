import torch

from src.stage2.unmixing.models.model import LatentUnmixingModel, UnmixingConfig


def main():
    config = dict(
        transformer=dict(
            out_channels=256,
        ),
        vitamin=dict(
            input_channel=8,
            output_channel=32,
            condition_channel=256,
        ),
        to_endmember=dict(
            num_endmember=4,
            channels=32,
            init_value=None,
            kernel=1,
        ),
        amotize_type="latent_to_pixel_fusion",
        learn_decoder=False,
        backward_decoder=False,
    )

    model = LatentUnmixingModel.from_config(config)
    print(model)

    x = torch.randn(1, 8, 256, 256)
    latent = torch.randn(1, 16, 32, 32)

    output = model(x, latent)
    print(output["abunds"].shape)
    print(output["recon"].shape)


if __name__ == "__main__":
    main()
