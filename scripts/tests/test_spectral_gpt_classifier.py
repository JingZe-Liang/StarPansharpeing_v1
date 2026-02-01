"""Tests for SpectralGPTClassifier."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import Tensor

from src.stage2.classification.model.spectral_gpt_classifier import SpectralGPTClassifier


def test_spectral_gpt_classifier_init() -> None:
    """Test SpectralGPTClassifier initialization with default parameters."""
    model = SpectralGPTClassifier(
        model_type="tensor",
        model_name="vit_base_patch8_128",
        pretrained=False,
        freeze_backbone=True,
        num_classes=21,
    )

    assert model.model_type == "tensor"
    assert model.model_name == "vit_base_patch8_128"
    assert model.freeze_backbone is True
    assert model.img_is_neg_1_1 is False
    assert isinstance(model.backbone, torch.nn.Module)


def test_spectral_gpt_classifier_forward() -> None:
    """Test forward pass with random input."""
    num_classes = 21
    batch_size = 2

    model = SpectralGPTClassifier(
        model_type="tensor",
        model_name="vit_base_patch8_128",
        pretrained=False,
        freeze_backbone=True,
        num_classes=num_classes,
    )
    model.eval()

    # Input for vit_base_patch8_128: (B, 12, 128, 128)
    x = torch.randn(batch_size, 12, 128, 128)

    with torch.no_grad():
        output = model(x)

    assert "logits" in output
    logits = output["logits"]
    assert logits.shape == (batch_size, num_classes)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()


def test_spectral_gpt_classifier_forward_with_neg_1_1_input() -> None:
    """Test forward pass with [-1, 1] range input."""
    num_classes = 10
    batch_size = 2

    model = SpectralGPTClassifier(
        model_type="tensor",
        model_name="vit_base_patch8_128",
        pretrained=False,
        freeze_backbone=True,
        num_classes=num_classes,
        img_is_neg_1_1=True,
    )
    model.eval()

    # Input in [-1, 1] range
    x = torch.rand(batch_size, 12, 128, 128) * 2 - 1

    with torch.no_grad():
        output = model(x)

    assert output["logits"].shape == (batch_size, num_classes)


def test_spectral_gpt_classifier_tensor_model_type() -> None:
    """Test tensor model type with different configurations."""
    configs = [
        ("vit_base_patch8_128", 12, 128, 128),
        ("vit_base_patch8", 12, 96, 96),
        ("vit_base_patch16_128", 12, 128, 128),
    ]

    for model_name, channels, height, width in configs:
        model = SpectralGPTClassifier(
            model_type="tensor",
            model_name=model_name,
            pretrained=False,
            freeze_backbone=True,
            num_classes=21,
        )
        assert model.model_type == "tensor"
        assert model.model_name == model_name
        assert isinstance(model.backbone, torch.nn.Module)

        # Test forward pass
        x = torch.randn(1, channels, height, width)
        model.eval()
        with torch.no_grad():
            output = model(x)
        assert output["logits"].shape == (1, 21)


def test_spectral_gpt_classifier_unfreeze_backbone() -> None:
    """Test with unfreezed backbone."""
    model = SpectralGPTClassifier(
        model_type="tensor",
        model_name="vit_base_patch8_128",
        pretrained=False,
        freeze_backbone=False,
        num_classes=21,
    )

    # All parameters should be trainable
    for param in model.parameters():
        assert param.requires_grad


def test_spectral_gpt_classifier_state_dict_with_freeze() -> None:
    """Test state_dict when backbone is frozen."""
    model = SpectralGPTClassifier(
        model_type="tensor",
        model_name="vit_base_patch8_128",
        pretrained=False,
        freeze_backbone=True,
        num_classes=21,
    )

    state_dict = model.state_dict()

    # When frozen, only head parameters should be in state_dict
    for key in state_dict:
        assert key.startswith("backbone.head.")


def test_spectral_gpt_classifier_train_mode() -> None:
    """Test train mode behavior with frozen backbone."""
    model = SpectralGPTClassifier(
        model_type="tensor",
        model_name="vit_base_patch8_128",
        pretrained=False,
        freeze_backbone=True,
        num_classes=21,
    )

    model.train()

    # Backbone should be in eval mode, head in train mode
    for name, module in model.backbone.named_modules():
        if name == "head":
            assert module.training
        elif name != "":
            assert not module.training


def test_spectral_gpt_classifier_linear_probe_classifier() -> None:
    """Test with linear_probe classifier type."""
    model = SpectralGPTClassifier(
        model_type="tensor",
        model_name="vit_base_patch8_128",
        pretrained=False,
        freeze_backbone=True,
        num_classes=21,
        classifier_type="linear_probe",
    )

    batch_size = 2
    x = torch.randn(batch_size, 12, 128, 128)

    model.eval()
    with torch.no_grad():
        output = model(x)

    assert output["logits"].shape == (batch_size, 21)


def test_spectral_gpt_classifier_different_num_classes() -> None:
    """Test with different number of classes."""
    test_cases = [2, 10, 21, 100, 1000]

    for num_classes in test_cases:
        model = SpectralGPTClassifier(
            model_type="tensor",
            model_name="vit_base_patch8_128",
            pretrained=False,
            freeze_backbone=True,
            num_classes=num_classes,
        )

        x = torch.randn(1, 12, 128, 128)
        model.eval()
        with torch.no_grad():
            output = model(x)

        assert output["logits"].shape == (1, num_classes)


def test_spectral_gpt_classifier_load_checkpoint_with_sep_pos_embed() -> None:
    """Test loading checkpoint with sep_pos_embed enabled."""
    checkpoint_path = Path("/Data2/ZihanCao/Checkpoints/SpectralGPT/SpectralGPT.pth")
    if not checkpoint_path.exists():
        pytest.skip("SpectralGPT checkpoint not found; skipping load test.")

    model = SpectralGPTClassifier(
        model_type="tensor",
        model_name="vit_base_patch8",
        checkpoint_path=str(checkpoint_path),
        pretrained=True,
        freeze_backbone=True,
        num_classes=21,
        backbone_kwargs={"sep_pos_embed": True},
    )

    assert hasattr(model.backbone, "pos_embed_spatial")
    assert hasattr(model.backbone, "pos_embed_temporal")

    batch_size = 2
    x = torch.randn(batch_size, 12, 96, 96)
    model.eval()
    with torch.no_grad():
        output = model(x)

    assert output["logits"].shape == (batch_size, 21)


def test_spectral_gpt_classifier_different_patch_configs() -> None:
    """Test with different patch size configurations."""
    configs = [
        ("vit_base_patch8_128", 12, 128, 128),
        ("vit_base_patch8", 12, 96, 96),
        ("vit_base_patch16_128", 12, 128, 128),
        ("vit_base_patch16", 12, 96, 96),
    ]

    for model_name, channels, height, width in configs:
        model = SpectralGPTClassifier(
            model_type="tensor",
            model_name=model_name,
            pretrained=False,
            freeze_backbone=True,
            num_classes=21,
        )

        x = torch.randn(1, channels, height, width)
        model.eval()
        with torch.no_grad():
            output = model(x)

        assert output["logits"].shape == (1, 21), f"Failed for {model_name}"


def test_spectral_gpt_classifier_invalid_model_type() -> None:
    """Test that invalid model_type raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported model_type"):
        SpectralGPTClassifier(
            model_type="invalid_type",  # type: ignore[arg-type]
            pretrained=False,
            num_classes=21,
        )


def test_spectral_gpt_classifier_invalid_model_name() -> None:
    """Test that invalid model_name raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported model_name"):
        SpectralGPTClassifier(
            model_type="tensor",
            model_name="invalid_model",
            pretrained=False,
            num_classes=21,
        )


def test_spectral_gpt_classifier_backward_pass() -> None:
    """Test backward pass with loss computation."""
    model = SpectralGPTClassifier(
        model_type="tensor",
        model_name="vit_base_patch8_128",
        pretrained=False,
        freeze_backbone=True,
        num_classes=21,
    )

    model.train()
    x = torch.randn(2, 12, 128, 128)
    targets = torch.randint(0, 21, (2,))

    output = model(x)
    logits = output["logits"]

    loss = torch.nn.functional.cross_entropy(logits, targets)
    loss.backward()

    # Only head parameters should have gradients
    for name, param in model.named_parameters():
        assert param.grad is not None
        assert name.startswith("backbone.head.")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
