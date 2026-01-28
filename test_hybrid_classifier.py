"""Test CosmosTokenizerClassifier with Hybrid Tokenizer."""

import torch
from omegaconf import OmegaConf
from pathlib import Path

from src.stage2.classification.model.cosmos_tokenizer_classifier import CosmosTokenizerClassifier


def test_hybrid_classifier():
    """Test hybrid tokenizer classifier with intermediate features."""

    # Load tokenizer config
    tokenizer_cfg_path = Path(__file__).parent / "scripts/configs/classification/model/tokenizer/hybrid.yaml"
    with open(tokenizer_cfg_path) as f:
        import yaml

        tokenizer_cfg = yaml.safe_load(f)

    tokenizer_cfg = OmegaConf.create(tokenizer_cfg)

    # Model config
    num_classes = 21  # UCMerced has 21 classes
    tokenizer_pretrained_path = "runs/stage1_cosmos_hybrid/2025-12-21_23-52-12_hybrid_cosmos_f16c64_ijepa_pretrained_sem_no_lejepa/ema/tokenizer/model.safetensors"

    print("=" * 80)
    print("Testing Hybrid Tokenizer Classifier")
    print("=" * 80)
    print(f"Tokenizer config: {tokenizer_cfg_path}")
    print(f"Pretrained path: {tokenizer_pretrained_path}")
    print(f"Num classes: {num_classes}")
    print()

    # Create model with intermediate features
    print("Creating model with intermediate features...")
    n_last_blocks = 1
    use_avgpool = True

    model = CosmosTokenizerClassifier(
        tokenizer_cfg=tokenizer_cfg,
        num_classes=num_classes,
        tokenizer_pretrained_path=tokenizer_pretrained_path,
        freeze_tokenizer=True,
        use_intermediate_features=True,
        n_last_blocks=n_last_blocks,
        use_avgpool=use_avgpool,
        classifier={"classifier_type": "linear_probe"},
    )

    print(f"✓ Model created successfully")
    print(f"  - Tokenizer type: {type(model.tokenizer).__name__}")
    print(f"  - use_intermediate_features: {model.use_intermediate_features}")
    print(f"  - n_last_blocks: {model.n_last_blocks}")
    print(f"  - use_avgpool: {model.use_avgpool}")
    print(f"  - Classifier type: {type(model.classifier).__name__}")
    print()

    # Check feature dimensions
    feature_dim = model._get_feature_channels()
    embed_dim = tokenizer_cfg.trans_enc_cfg.embed_dim
    expected_dim = n_last_blocks * embed_dim + (embed_dim if use_avgpool else 0)
    print(f"Feature dimension: {feature_dim}")
    print(f"  - Embed dim: {embed_dim}")
    print(
        f"  - Expected: {n_last_blocks} blocks * {embed_dim} + avgpool({embed_dim if use_avgpool else 0}) = {expected_dim}"
    )
    assert feature_dim == expected_dim, f"Feature dim mismatch: {feature_dim} != {expected_dim}"
    print()

    # Test with random input
    batch_size = 2
    in_channels = 3  # RGB
    height, width = 256, 256

    x = torch.randn(batch_size, in_channels, height, width)

    print(f"Input shape: {x.shape}")
    print()

    # Forward pass
    print("Running forward pass...")
    model.eval()
    with torch.no_grad():
        output = model(x)

    logits = output["logits"]
    print(f"✓ Forward pass successful")
    print(f"  - Output logits shape: {logits.shape}")
    print(f"  - Expected shape: ({batch_size}, {num_classes})")

    # Verify output
    assert logits.shape == (batch_size, num_classes), f"Expected shape {(batch_size, num_classes)}, got {logits.shape}"

    # Check if logits are valid (not NaN or Inf)
    assert not torch.isnan(logits).any(), "Logits contain NaN values"
    assert not torch.isinf(logits).any(), "Logits contain Inf values"

    # Compute probabilities
    probs = torch.softmax(logits, dim=-1)
    print(f"  - Probabilities shape: {probs.shape}")
    print(f"  - Probabilities sum: {probs.sum(dim=-1)}")

    # Check probabilities sum to 1
    assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size)), "Probabilities should sum to 1"

    print()
    print("=" * 80)
    print("✅ All tests passed! Hybrid classifier is working correctly.")
    print("=" * 80)
    print(f"Sample logits: {logits[0][:5]}...  (first 5 of {num_classes})")
    print(f"Sample probabilities: {probs[0][:5]}...  (first 5 of {num_classes})")

    return model, output


if __name__ == "__main__":
    test_hybrid_classifier()
