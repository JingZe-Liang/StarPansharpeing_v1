import torch

from src.stage2.unmixing.models.model import LatentUnmixingModel, UnmixingConfig


def main():
    """
    Urban dataset shapes:
        abundance: torch.Size([1, 4, 307, 307])
        endmember: torch.Size([1, 162, 4])
        img: torch.Size([1, 162, 307, 307])
        init_em: torch.Size([1, 162, 4])
    """

    config = dict(
        transformer=dict(
            out_channels=256,
        ),
        vitamin=dict(
            stem_width=64,
            depths=[1, 1, 1],
            use_residual=False,
            input_channel=162,
            output_channel=64,
            condition_channel=256,
            conv_cfg=dict(
                expand_ratio=2,
            ),
        ),
        to_endmember=dict(
            num_endmember=4,
            channels=64,
            init_value=None,
            kernel=1,
        ),
        amotize_type="latent_to_pixel_fusion",
        learn_decoder=False,
        backward_decoder=False,
    )

    model = LatentUnmixingModel.from_config(config)
    # print(model)

    x = torch.randn(1, 162, 307, 307)
    latent = torch.randn(1, 16, 307 // 8, 307 // 8)

    # output = model(x, latent)
    # print(output["abunds"].shape)
    # print(output["recon"].shape)

    from fvcore.nn import FlopCountAnalysis, flop_count_table

    flop_analysis = FlopCountAnalysis(model, (x, latent))
    print(flop_count_table(flop_analysis, max_depth=3))


if __name__ == "__main__":
    main()
