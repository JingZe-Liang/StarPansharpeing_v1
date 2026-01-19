import torch


def test_naflex_transformer_forward_intermediates_reduces_mhc_batch() -> None:
    from src.stage1.cosmos.modules.naflex import NaFlexVitCfg, Transformer

    cfg = NaFlexVitCfg(
        img_size=8,
        patch_size=2,
        embed_dim=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        in_chans=3,
        out_chans=8,
        out_2d_latent=True,
        unpatch_size=1,
        hc_streams=4,
        hc_implem="naive",
    )
    model = Transformer(cfg).eval()

    batch = 2
    x = torch.randn(batch, cfg.in_chans, cfg.img_size, cfg.img_size)

    out, intermediates = model.forward_intermediates(x, indices=[0], output_fmt="NCHW")  # type: ignore[assignment]
    assert isinstance(intermediates, list)
    assert out.shape[0] == batch
    assert intermediates[0].shape[0] == batch
