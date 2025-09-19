import hydra
import torch
import torch.nn as nn


@hydra.main(
    config_path="../configs/pansharpening",
    config_name="tokenizer_lora_vitamin",
    version_base=None,
)
def test_model(cfg):
    model = hydra.utils.instantiate(cfg.pansharp_model)

    # inputs
    bs = 2
    ms, pan = torch.randn(bs, 8, 512, 512).cuda(), torch.randn(bs, 1, 512, 512).cuda()
    ms_latent, pan_latent = (
        torch.randn(bs, 16, 64, 64).cuda(),
        torch.randn(bs, 16, 64, 64).cuda(),
    )
    decoder_fn = lambda x: x
    pixel_in = (ms, pan)
    latent_in = (ms_latent, pan_latent)
    model = model(decoder_fn=decoder_fn).cuda()
    model.set_checkpoint_mode()
    y = model(pixel_in, latent_in)["pixel_out"]
    print(y.shape)


if __name__ == "__main__":
    test_model()
