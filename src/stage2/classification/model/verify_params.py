"""Check if parameters() correctly filters frozen weights."""

import torch

from src.stage2.classification.model.hypersigma_classifier import HyperSIGMAClassifier


def test_param_filtering():
    print("=" * 80)
    print("Testing Parameter Filtering")
    print("=" * 80)

    # 1. Test with freeze_backbone=True
    model = HyperSIGMAClassifier(
        backbone_type="spat_vit_l",
        pretrained=False,
        freeze_backbone=True,
        num_classes=21,
    )

    params = list(model.parameters())
    named_params = list(model.named_parameters())

    print(f"\n[freeze_backbone=True]")
    print(f"Total parameters returned: {len(params)}")
    print("Parameters list:")
    for name, p in named_params:
        print(f"  - {name}: {p.shape} (requires_grad={p.requires_grad})")

    # Verify only patch_embed and classifier are here
    for name, p in named_params:
        assert p.requires_grad, f"Parameter {name} should be trainable!"
        assert "patch_embed" in name or "classifier" in name, f"Unexpected parameter: {name}"

    # 2. Test with freeze_backbone=False
    model_all = HyperSIGMAClassifier(
        backbone_type="spat_vit_l",
        pretrained=False,
        freeze_backbone=False,
        num_classes=21,
    )

    params_all = list(model_all.parameters())
    print(f"\n[freeze_backbone=False]")
    print(f"Total parameters returned: {len(params_all)}")
    assert len(params_all) > 300, "Should return all parameters when not frozen!"

    print("\n✅ Parameter filtering test passed!")


if __name__ == "__main__":
    test_param_filtering()
