import torch

from src.stage1.cosmos.cosmos_tokenizer import (
    ContinuousImageTokenizer,
    ContinuousTokenizerConfig,
    EncoderDecoderConfig,
)


def _build_kl_tokenizer() -> ContinuousImageTokenizer:
    cfg = ContinuousTokenizerConfig(
        model=EncoderDecoderConfig(
            in_channels=3,
            out_channels=3,
            channels=16,
            channels_mult=[2, 2],
            num_res_blocks=1,
            attn_resolutions=[],
            dropout=0.0,
            resolution=16,
            z_channels=8,
            latent_channels=4,
            spatial_compression=4,
            patch_size=1,
            patch_method="haar",
            conv_in_module="conv",
            block_name="res_block",
            attn_type="none",
            padding_mode="zeros",
            norm_type="gn",
            norm_groups=8,
        ),
        quantizer_type="kl",
        use_channel_drop=False,
        latent_noise_prob=0.0,
        use_repa_loss=False,
        use_vf_loss=False,
        z_factor=1,
    )
    return ContinuousImageTokenizer(cfg)


def test_kl_latent_bypass_uses_latent_mode_shape() -> None:
    tokenizer = _build_kl_tokenizer()
    x = torch.randn(2, 3, 16, 16)

    encoded = tokenizer.encode(x, use_quantizer=False)

    assert encoded.latent.shape == (2, 4, 4, 4)
    assert encoded.q_loss is None
    assert encoded.latent_mean.shape == encoded.latent.shape
    assert encoded.latent_logvar.shape == encoded.latent.shape
    assert torch.allclose(encoded.latent, encoded.latent_mode)


def test_kl_latent_eval_is_deterministic_mode() -> None:
    tokenizer = _build_kl_tokenizer()
    x = torch.randn(2, 3, 16, 16)
    tokenizer.eval()

    encoded = tokenizer.encode(x, use_quantizer=True)

    assert encoded.q_loss is not None
    assert encoded.latent.shape == (2, 4, 4, 4)
    assert torch.allclose(encoded.latent, encoded.latent_mode)
