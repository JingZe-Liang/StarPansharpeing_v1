import os
import sys

os.environ['MODEL_COMPILED'] = '0'

import hydra
import lovely_tensors as lt
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from src.utilities import config_utils  # register custom resolvers


def test_init_model():
    """
    Test model initialization using the cosmos tokenizer configuration.

    This test verifies that the CosmosFlowTokenizer can be properly initialized
    with the given configuration parameters.
    """
    # Load the configuration directly from the tokenizer config file
    with hydra.initialize(config_path="../configs/2d_cosmos_diff", version_base=None):
        cfg = hydra.compose(config_name="cosmos_flow_f8c16p1")

    logger.info("Tokenizer configuration loaded successfully")
    logger.info(f"Configuration keys: {list(cfg.keys())}")
    cfg2 = cfg.copy()

    # Print key configuration parameters
    cfg = cfg.tokenizer
    logger.info(f"Tokenizer target: {cfg._target_}")
    logger.info(f"Model latent channels: {cfg.tokenizer_cfg.model.latent_channels}")
    logger.info(f"Model patch size: {cfg.tokenizer_cfg.model.patch_size}")

    # Initialize the tokenizer
    logger.info("Initializing tokenizer...")
    tokenizer = hydra.utils.instantiate(cfg)
    tokenizer = tokenizer.cuda()

    logger.success(f"Tokenizer initialized successfully: {tokenizer.__class__.__name__}")

    # Print model information
    total_params = sum(p.numel() for p in tokenizer.parameters())
    trainable_params = sum(p.numel() for p in tokenizer.parameters() if p.requires_grad)

    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass with dummy input
    logger.info("Testing forward pass...")
    batch_size = 1  # Use batch size 1 for testing
    channels = 3
    height, width = 64, 64  # Small test size

    dummy_input = torch.randn(batch_size, channels, height, width).cuda()
    logger.info(f"Input shape: {dummy_input.shape}")

    # Set model to eval mode for testing
    tokenizer.eval()

    with torch.no_grad():
        # Try forward pass
        # Use the same forward logic as in the trainer
        other_kwargs = {
            "dec_mode": "step",
            "ema_model": None,
        }

        output = tokenizer(dummy_input, **other_kwargs)

        if isinstance(output, tuple) and len(output) == 3:
            recon, loss_dict, q_info = output
            logger.success(f"Forward pass successful")
            logger.info(f"Reconstruction shape: {recon.shape}")
            logger.info(f"Loss keys: {list(loss_dict.keys()) if isinstance(loss_dict, dict) else 'N/A'}")
            logger.info(f"Quantizer info: {q_info}")
        else:
            logger.success(f"Forward pass successful")
            logger.info(f"Output type: {type(output)}")
            if hasattr(output, 'shape'):
                logger.info(f"Output shape: {output.shape}")


    logger.success("Model initialization test completed successfully!")

if __name__ == "__main__":
    # Configure loguru to show test output
    lt.monkey_patch()
    with logger.catch():
        test_init_model()

