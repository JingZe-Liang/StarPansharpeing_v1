"""
Test file for DownstreamModelTokenizerWrapper with NAF pansharpening model and Cosmos tokenizer.
"""

import torch
import torch.nn as nn

from src.stage1.cosmos.cosmos_tokenizer import (
    ContinuousImageTokenizer,
    ContinuousTokenizerConfig,
    EncoderDecoderConfig,
)
from src.stage2.change_detection.models.dinov3_adapted import (
    AdapterConfig,
    DinoConfig,
    DinoUnetConfig,
    MultiscaleMBConvStageConfig,
)
from src.stage2.layers.wrapper import DownstreamModelTokenizerWrapper
from src.stage2.pansharpening.models.naf import (
    PansharpeningNAFNet,
    PansharpeningNAFNetConfig,
)


def test_tokenizer_pansharpening_model():
    """
    Test the integration of NAF pansharpening model with Cosmos tokenizer
    using DownstreamModelTokenizerWrapper with AmotizedModelMixin.
    """

    # Create Cosmos tokenizer configuration
    tokenizer_cfg = ContinuousTokenizerConfig(
        model=EncoderDecoderConfig(
            in_channels=8,
            out_channels=8,
            channels=128,
            channels_mult=[2, 4, 4],
            num_res_blocks=2,
            attn_resolutions=[],
            dropout=0.0,
            resolution=256,
            z_channels=16,
            latent_channels=16,
            spatial_compression=8,
            patch_size=1,
            patch_method="haar",
            conv_in_module="conv",
            block_name="res_block",
            attn_type="none",
            padding_mode="zeros",
            norm_type="gn",
            norm_groups=32,
        ),
        quantizer_type=None,  # Use autoencoder mode
        use_channel_drop=False,
        latent_noise_prob=0.0,
        use_repa_loss=False,
        use_vf_loss=False,
        z_factor=1,
    )

    # Create tokenizer
    tokenizer = ContinuousImageTokenizer(tokenizer_cfg)

    # Create Transformer configuration (amotized model)
    from src.stage2.pansharpening.models.transformer import (
        Transformer,
        TransformerConfig,
    )

    transformer_cfg = TransformerConfig(
        in_dim=16,  # Match tokenizer latent channels
        dim=384,
        depth=8,
        num_heads=8,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        patch_size=1,
        out_channels=256,
        norm_layer="rmsnorm",
        mlp_norm_layer="rmsnorm",
        act_layer="silu",
        pos_embed_type="sincos",
        input_size=32,
        with_raw_img=False,
        raw_img_size=None,
        raw_img_chans=None,
        feature_layer_ids=None,
    )

    # Create NAF pansharpening model configuration (pixel model)
    naf_cfg = PansharpeningNAFNetConfig(
        pan_channel=1,
        ms_channel=8,
        width=64,
        middle_blk_num=1,
        enc_blk_nums=[1, 1, 1],
        dec_blk_nums=[1, 1, 1],
        condition_channel=256,  # Comes from transformer output
        patch_size=1,
        dw_expand=1,
        ffn_expand=2,
        condition_on_decoder=True,
        residual_type="ms",
        is_neg_1_1=True,
        output_rescale=True,
    )

    # Create individual models
    transformer_model = Transformer(transformer_cfg)
    naf_model = PansharpeningNAFNet(naf_cfg)

    # Create a simple decoder function that matches AmotizedModelMixin expectations
    def simple_decoder(latent, shape):
        """Simple decoder function that extracts the first element from tokenizer.decode output."""
        decoded = tokenizer.decode(latent, shape)
        if isinstance(decoded, tuple):
            return decoded[0]  # Extract the tensor from tuple
        return decoded

    # Create AmotizedModelMixin to wrap both models
    from src.stage2.utilities.amotized.amotized_model_wrapper import AmotizedModelMixin

    amotized_model = AmotizedModelMixin(
        pixel_model=naf_model,
        amotized_model=transformer_model,
        decoder_fn=simple_decoder,
        amotize_type="latent_to_pixel_fusion",
        backward_decoder=False,
        learn_decoder=False,
    )

    # Wrap the models using DownstreamModelTokenizerWrapper
    encoder_processor = lambda ms, pan: [ms, pan.repeat(1, 8, 1, 1)]
    decoder_processor = lambda ms, pan: [ms, pan.mean(1, keepdim=True)]
    wrapped_model = DownstreamModelTokenizerWrapper(
        tokenizer=tokenizer,
        downstream_model=amotized_model,
        froze_tokenizer=True,
        n_img_encoded=2,  # MS and PAN images
        tokenizer_img_processor=encoder_processor,
        detokenizer_img_processor=decoder_processor,
    )

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapped_model = wrapped_model.to(device)

    # Create test data
    batch_size = 2
    ms_height, ms_width = 64, 64
    pan_height, pan_width = 64, 64  # Same resolution for simplicity

    # MS image: (batch_size, channels, height, width)
    ms_img = torch.randn(batch_size, 8, ms_height, ms_width, device=device)
    # PAN image: (batch_size, 1, height, width)
    pan_img = torch.randn(batch_size, 1, pan_height, pan_width, device=device)

    # Test forward pass
    wrapped_model.eval()
    with torch.no_grad():
        output = wrapped_model([ms_img, pan_img])

    print("✓ Test passed!")
    print(f"Input MS shape: {ms_img.shape}")
    print(f"Input PAN shape: {pan_img.shape}")
    print(f"Output shape: {output.keys()}")

    # Test training mode
    wrapped_model.train()
    output_train = wrapped_model((ms_img, pan_img))

    print("✓ Training mode test passed!")

    # Test gradient flow
    loss = output_train["pixel_out"].mean()
    loss.backward()

    # Check if gradients exist for downstream model parameters
    has_gradients = False
    for name, param in wrapped_model.downstream_model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_gradients = True
            break

    assert has_gradients, "No gradients found in downstream model"
    print("✓ Gradient flow test passed!")

    return wrapped_model


def test_cd_dino_adapted_model():
    """
    Test the integration of DinoV3 adapted model with Cosmos tokenizer
    using DownstreamModelTokenizerWrapper for change detection.
    """

    # Create Cosmos tokenizer configuration
    tokenizer_cfg = ContinuousTokenizerConfig(
        model=EncoderDecoderConfig(
            in_channels=8,
            out_channels=8,
            channels=128,
            channels_mult=[2, 4, 4],
            num_res_blocks=2,
            attn_resolutions=[],  # Add attention resolution to avoid AttnBlock removal error
            dropout=0.0,
            resolution=256,
            z_channels=16,
            latent_channels=16,
            spatial_compression=8,
            patch_size=1,
            patch_method="haar",
            conv_in_module="conv",
            block_name="res_block",
            attn_type="none",  # Enable attention
            padding_mode="zeros",
            norm_type="gn",
            norm_groups=32,
        ),
        quantizer_type=None,  # Use autoencoder mode
        use_channel_drop=False,
        latent_noise_prob=0.0,
        use_repa_loss=False,
        use_vf_loss=False,
        z_factor=1,
    )

    # Create tokenizer
    tokenizer = ContinuousImageTokenizer(tokenizer_cfg)

    # Create DinoUNet configuration

    dino_cfg = DinoUnetConfig(
        dino=DinoConfig(
            features_per_stage=(512, 512, 512, 512),
            pretrained_path=None,
            model_name="dinov3_vits16",
            pretrained_on="web",
        ),
        adapter=AdapterConfig(
            adapter_type="default",
            latent_width=16,  # Match tokenizer latent channels
            n_conv_per_stage=1,
            depth_per_stage=1,
            norm="layernorm2d",
            act="gelu",
            drop=0.0,
            act_first=False,
            conv_bias=False,
        ),
        cd_stage=MultiscaleMBConvStageConfig(
            channels=[512, 512, 512, 512],
            stride=1,
            kernel_size=3,
            norm_layer="layernorm2d",
            act_layer="gelu",
            expand_ratio=2.0,
            block_type="mbconv",
            depth=1,
        ),
        input_channels=3,
        num_classes=3,  # 0: unknown, 1: changed, 2: unchanged
        deep_supervision=False,
        n_stages=4,
        use_latent=True,
        ensure_rgb_type=[2, 1, 0],
        _debug=False,
    )

    # Create DinoUNet model (no amotized wrapper needed)
    from src.stage2.change_detection.model.dinov3_adapted import DinoUNet

    dino_model = DinoUNet(dino_cfg)

    # Wrap the models using DownstreamModelTokenizerWrapper
    # No image processor needed since DinoUNet handles RGB conversion internally
    wrapped_model = DownstreamModelTokenizerWrapper(
        tokenizer=tokenizer,
        downstream_model=dino_model,
        froze_tokenizer=True,
        n_img_encoded=2,  # Two temporal images for change detection
    )

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapped_model = wrapped_model.to(device)

    # Create test data for change detection
    batch_size = 1
    height, width = 256, 256

    # Two temporal images with multiple channels (e.g., hyperspectral)
    img1 = torch.randn(batch_size, 8, height, width, device=device)
    img2 = torch.randn(batch_size, 8, height, width, device=device)

    # Test forward pass
    wrapped_model.eval()
    with torch.no_grad():
        output = wrapped_model([img1, img2])

    # Verify output shape - should be change detection map
    expected_shape = (batch_size, dino_cfg.num_classes, height, width)
    assert output.shape == expected_shape, f"Expected output shape {expected_shape}, got {output.shape}"

    print("✓ DinoV3 adapted model test passed!")
    print(f"Input image 1 shape: {img1.shape}")
    print(f"Input image 2 shape: {img2.shape}")
    print(f"Output change detection map shape: {output.shape}")

    # Test training mode
    wrapped_model.train()
    output_train = wrapped_model([img1, img2])

    # Verify training output shape
    assert output_train.shape == expected_shape, (
        f"Expected training output shape {expected_shape}, got {output_train.shape}"
    )

    print("✓ Training mode test passed!")

    # Test gradient flow
    loss = output_train.mean()
    loss.backward()

    # Check if gradients exist for downstream model parameters
    has_gradients = False
    for name, param in wrapped_model.downstream_model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_gradients = True
            break

    assert has_gradients, "No gradients found in downstream model"
    print("✓ Gradient flow test passed!")

    # Test output values (change detection probabilities)
    # Output should be valid probabilities after softmax
    import torch.nn.functional as F

    if torch.cuda.is_available():
        output_probs = F.softmax(output, dim=1)
        assert torch.allclose(output_probs.sum(dim=1), torch.ones_like(output_probs.sum(dim=1)), atol=1e-6), (
            "Output probabilities should sum to 1"
        )
        print("✓ Output probability validation passed!")

    return wrapped_model


def test_init_from_cfg():
    from omegaconf import OmegaConf
    import hydra

    cfg = OmegaConf.load("scripts/configs/pansharpening/pansharp_model/pansharp_wrapper_nafnet_cosmos.yaml")
    print(cfg)

    print("Init model")
    model = hydra.utils.instantiate(cfg)
    print(model)

    print("Forward model")
    ms = torch.randn(1, 4, 256, 256)
    pan = torch.randn(1, 1, 256, 256)
    output = model.forward([ms, pan])["pixel_out"]
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    print("\nTesting DownstreamModelTokenizerWrapper with DinoV3 adapted model...")
    # test_cd_dino_adapted_model()
    test_init_from_cfg()

    print("\n✓ All tests passed!")
