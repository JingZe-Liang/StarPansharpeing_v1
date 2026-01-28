"""Test loading HyperSIGMA pretrained weights."""

import torch

from src.stage2.classification.model.hypersigma_classifier import HyperSIGMAClassifier


def test_load_spatial_weights():
    """Test loading Spatial MAE weights."""
    print("=" * 80)
    print("Testing Spatial ViT-L weight loading...")
    print("=" * 80)

    weights_path = "/Data2/ZihanCao/Checkpoints/Hypersigma/spat-vit-large-ultra-checkpoint-1599.pth"

    # Create model
    model = HyperSIGMAClassifier(
        backbone_type="spat_vit_l",
        weights_path=weights_path,
        pretrained=True,
        freeze_backbone=True,
        num_classes=21,
        img_size=224,
    )

    print(f"\n✅ Model created successfully!")
    print(f"Backbone type: spat_vit_l")
    print(f"Embed dim: {model.backbone.embed_dim}")
    print(f"Depth: {len(model.backbone.blocks)}")

    # Test forward pass
    print("\n" + "=" * 80)
    print("Testing forward pass...")
    print("=" * 80)

    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)

    model.eval()
    with torch.no_grad():
        output = model(x)

    logits = output["logits"]
    print(f"Input shape: {x.shape}")
    print(f"Output logits shape: {logits.shape}")
    assert logits.shape == (batch_size, 21), f"Expected (2, 21), got {logits.shape}"

    # Check if backbone is frozen
    print("\n" + "=" * 80)
    print("Checking frozen parameters...")
    print("=" * 80)

    frozen_params = sum(1 for p in model.backbone.parameters() if not p.requires_grad)
    trainable_params = sum(1 for p in model.backbone.parameters() if p.requires_grad)
    classifier_params = sum(1 for p in model.backbone.classifier.parameters() if p.requires_grad)

    print(f"Frozen backbone parameters: {frozen_params}")
    print(f"Trainable parameters (classifier): {trainable_params}")
    print(f"Classifier parameters: {classifier_params}")

    assert trainable_params == classifier_params, "Only classifier should be trainable!"

    print("\n✅ Spatial ViT-L test passed!")
    return model


def test_checkpoint_structure():
    """Inspect checkpoint structure."""
    print("\n" + "=" * 80)
    print("Inspecting checkpoint structure...")
    print("=" * 80)

    spat_path = "/Data2/ZihanCao/Checkpoints/Hypersigma/spat-vit-large-ultra-checkpoint-1599.pth"
    spec_path = "/Data2/ZihanCao/Checkpoints/Hypersigma/spec-vit-large-ultra-checkpoint-1599.pth"

    # Load spatial checkpoint
    print("\n📦 Spatial MAE checkpoint:")
    spat_ckpt = torch.load(spat_path, map_location="cpu", weights_only=False)

    if isinstance(spat_ckpt, dict):
        print(f"Checkpoint keys: {list(spat_ckpt.keys())}")
        if "model" in spat_ckpt:
            state_dict = spat_ckpt["model"]
            print(f"Number of parameters: {len(state_dict)}")
            print(f"Sample keys (first 10):")
            for i, key in enumerate(list(state_dict.keys())[:10]):
                print(f"  {i + 1}. {key}: {state_dict[key].shape}")
        elif "state_dict" in spat_ckpt:
            state_dict = spat_ckpt["state_dict"]
            print(f"Number of parameters: {len(state_dict)}")
            print(f"Sample keys (first 10):")
            for i, key in enumerate(list(state_dict.keys())[:10]):
                print(f"  {i + 1}. {key}: {state_dict[key].shape}")
        else:
            print(f"Number of parameters: {len(spat_ckpt)}")
            print(f"Sample keys (first 10):")
            for i, key in enumerate(list(spat_ckpt.keys())[:10]):
                print(f"  {i + 1}. {key}: {spat_ckpt[key].shape}")
    else:
        print(f"Checkpoint type: {type(spat_ckpt)}")

    # Load spectral checkpoint
    print("\n📦 Spectral MAE checkpoint:")
    spec_ckpt = torch.load(spec_path, map_location="cpu", weights_only=False)

    if isinstance(spec_ckpt, dict):
        print(f"Checkpoint keys: {list(spec_ckpt.keys())}")
        if "model" in spec_ckpt:
            state_dict = spec_ckpt["model"]
            print(f"Number of parameters: {len(state_dict)}")
            print(f"Sample keys (first 10):")
            for i, key in enumerate(list(state_dict.keys())[:10]):
                print(f"  {i + 1}. {key}: {state_dict[key].shape}")
        elif "state_dict" in spec_ckpt:
            state_dict = spec_ckpt["state_dict"]
            print(f"Number of parameters: {len(state_dict)}")
            print(f"Sample keys (first 10):")
            for i, key in enumerate(list(state_dict.keys())[:10]):
                print(f"  {i + 1}. {key}: {state_dict[key].shape}")
        else:
            print(f"Number of parameters: {len(spec_ckpt)}")
            print(f"Sample keys (first 10):")
            for i, key in enumerate(list(spec_ckpt.keys())[:10]):
                print(f"  {i + 1}. {key}: {spec_ckpt[key].shape}")
    else:
        print(f"Checkpoint type: {type(spec_ckpt)}")


def test_cuda_inference():
    """Test inference on CUDA if available."""
    if not torch.cuda.is_available():
        print("\n⚠️  CUDA not available, skipping GPU test")
        return

    print("\n" + "=" * 80)
    print("Testing CUDA inference...")
    print("=" * 80)

    weights_path = "/Data2/ZihanCao/Checkpoints/Hypersigma/spat-vit-large-ultra-checkpoint-1599.pth"

    model = HyperSIGMAClassifier(
        backbone_type="spat_vit_l",
        weights_path=weights_path,
        pretrained=True,
        freeze_backbone=True,
        num_classes=21,
        img_size=224,
    )

    model = model.cuda()
    model.eval()

    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224).cuda()

    print(f"Input device: {x.device}")

    with torch.no_grad():
        output = model(x)

    logits = output["logits"]
    print(f"Output logits shape: {logits.shape}")
    print(f"Output device: {logits.device}")

    print("✅ CUDA inference test passed!")


if __name__ == "__main__":
    # Test checkpoint structure first
    test_checkpoint_structure()

    # Test loading spatial weights
    model = test_load_spatial_weights()

    # Test CUDA inference
    test_cuda_inference()

    print("\n" + "=" * 80)
    print("🎉 All tests passed!")
    print("=" * 80)
