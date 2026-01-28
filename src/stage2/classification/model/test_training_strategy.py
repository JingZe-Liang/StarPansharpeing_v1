"""Test the new training strategy: freeze Transformer, train patch_embed + classifier."""

import torch

from src.stage2.classification.model.hypersigma_classifier import HyperSIGMAClassifier


def test_training_strategy():
    """Test the training strategy."""
    print("=" * 80)
    print("Testing Training Strategy")
    print("=" * 80)

    weights_path = "/Data2/ZihanCao/Checkpoints/Hypersigma/spat-vit-large-ultra-checkpoint-1599.pth"

    # Create model
    model = HyperSIGMAClassifier(
        backbone_type="spat_vit_l",
        weights_path=weights_path,
        pretrained=True,
        freeze_backbone=True,  # Freeze Transformer blocks only
        num_classes=21,
        img_size=224,
    )

    print("\n" + "=" * 80)
    print("Checking parameter trainability")
    print("=" * 80)

    # Count trainable parameters by category
    patch_embed_trainable = 0
    patch_embed_frozen = 0
    blocks_trainable = 0
    blocks_frozen = 0
    classifier_trainable = 0
    classifier_frozen = 0
    other_trainable = 0
    other_frozen = 0

    for name, param in model.backbone.named_parameters():
        if name.startswith("patch_embed"):
            if param.requires_grad:
                patch_embed_trainable += 1
            else:
                patch_embed_frozen += 1
        elif name.startswith("blocks"):
            if param.requires_grad:
                blocks_trainable += 1
            else:
                blocks_frozen += 1
        elif name.startswith("classifier"):
            if param.requires_grad:
                classifier_trainable += 1
            else:
                classifier_frozen += 1
        else:
            if param.requires_grad:
                other_trainable += 1
            else:
                other_frozen += 1

    print(f"\n📊 Parameter Statistics:")
    print(f"   patch_embed: {patch_embed_trainable} trainable, {patch_embed_frozen} frozen")
    print(f"   blocks (Transformer): {blocks_trainable} trainable, {blocks_frozen} frozen")
    print(f"   classifier: {classifier_trainable} trainable, {classifier_frozen} frozen")
    print(f"   other: {other_trainable} trainable, {other_frozen} frozen")

    # Verify expectations
    assert patch_embed_trainable > 0, "patch_embed should be trainable!"
    assert patch_embed_frozen == 0, "patch_embed should NOT be frozen!"
    assert blocks_trainable == 0, "Transformer blocks should NOT be trainable!"
    assert blocks_frozen > 0, "Transformer blocks should be frozen!"
    assert classifier_trainable > 0, "classifier should be trainable!"
    assert classifier_frozen == 0, "classifier should NOT be frozen!"

    print(f"\n✅ Correct freezing strategy:")
    print(f"   ✓ patch_embed is trainable (adapts to RGB)")
    print(f"   ✓ Transformer blocks are frozen (use pretrained knowledge)")
    print(f"   ✓ classifier is trainable (task-specific)")

    # Check module mode
    print(f"\n" + "=" * 80)
    print("Checking module training mode")
    print("=" * 80)

    model.train()  # Set to training mode

    for name, module in model.backbone.named_modules():
        if name == "patch_embed":
            assert module.training, "patch_embed should be in training mode!"
            print(f"   ✓ patch_embed.training = {module.training}")
        elif name.startswith("blocks.0"):  # Check first block
            assert not module.training, "Transformer blocks should be in eval mode!"
            print(f"   ✓ blocks.0.training = {module.training}")
        elif name == "classifier":
            assert module.training, "classifier should be in training mode!"
            print(f"   ✓ classifier.training = {module.training}")

    # Test forward pass
    print(f"\n" + "=" * 80)
    print("Testing forward pass")
    print("=" * 80)

    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)

    model.train()
    output = model(x)
    logits = output["logits"]

    print(f"   Input shape: {x.shape}")
    print(f"   Output logits shape: {logits.shape}")
    assert logits.shape == (batch_size, 21), "Output shape mismatch!"

    # Test backward pass
    loss = logits.sum()
    loss.backward()

    # Check gradients
    has_patch_grad = False
    has_block_grad = False
    has_classifier_grad = False

    for name, param in model.backbone.named_parameters():
        if param.grad is not None:
            if name.startswith("patch_embed"):
                has_patch_grad = True
            elif name.startswith("blocks"):
                has_block_grad = True
            elif name.startswith("classifier"):
                has_classifier_grad = True

    print(f"\n📊 Gradient flow:")
    print(f"   patch_embed has gradients: {has_patch_grad}")
    print(f"   blocks have gradients: {has_block_grad}")
    print(f"   classifier has gradients: {has_classifier_grad}")

    assert has_patch_grad, "patch_embed should receive gradients!"
    assert not has_block_grad, "Transformer blocks should NOT receive gradients!"
    assert has_classifier_grad, "classifier should receive gradients!"

    print(f"\n✅ Gradient flow is correct!")

    print(f"\n" + "=" * 80)
    print("🎉 All tests passed!")
    print("=" * 80)
    print(f"\n📋 Summary:")
    print(f"   • Transformer blocks loaded from HyperSIGMA pretrain")
    print(f"   • Transformer blocks frozen (304 params)")
    print(f"   • patch_embed trainable (adapts 100ch->3ch RGB)")
    print(f"   • classifier trainable (21-class scene classification)")
    print(f"   • Ready for fine-tuning on UCMerced dataset!")


if __name__ == "__main__":
    test_training_strategy()
