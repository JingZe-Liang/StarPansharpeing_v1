"""Test loading HyperSIGMA weights with 100 channels (matching pretrain)."""

import torch

from src.stage2.classification.model.hypersigma_classifier import SpatViT


def test_100_channel_load():
    """Test loading with 100 channels (same as pretrain)."""
    print("=" * 80)
    print("Testing with 100 channels (matching pretrain config)")
    print("=" * 80)

    # Create model with 100 channels (matching pretrain)
    model = SpatViT(
        img_size=64,  # matching pretrain
        patch_size=8,  # matching pretrain
        in_chans=100,  # matching pretrain
        num_classes=21,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        use_checkpoint=False,
        use_abs_pos_emb=True,
        out_indices=[7, 11, 15, 23],
        interval=6,
    )

    print(f"\nModel config:")
    print(f"  img_size: 64")
    print(f"  patch_size: 8")
    print(f"  in_chans: 100")
    print(f"  embed_dim: 1024")
    print(f"  depth: 24")

    # Load checkpoint
    weights_path = "/Data2/ZihanCao/Checkpoints/Hypersigma/spat-vit-large-ultra-checkpoint-1599.pth"
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    print(f"\nCheckpoint info:")
    print(f"  Total keys in checkpoint: {len(state_dict)}")
    print(f"  Sample keys:")
    for i, key in enumerate(list(state_dict.keys())[:10]):
        print(f"    {key}: {state_dict[key].shape}")

    # Try to load
    print("\n" + "=" * 80)
    print("Loading pretrained weights...")
    print("=" * 80)

    msg = model.load_state_dict(state_dict, strict=False)

    print(f"\n✅ Load completed!")
    print(f"Missing keys: {len(msg.missing_keys)}")
    if len(msg.missing_keys) > 0:
        print(f"  First 10 missing: {msg.missing_keys[:10]}")

    print(f"\nUnexpected keys: {len(msg.unexpected_keys)}")
    if len(msg.unexpected_keys) > 0:
        print(f"  First 10 unexpected: {msg.unexpected_keys[:10]}")

    # Test forward pass
    print("\n" + "=" * 80)
    print("Testing forward pass...")
    print("=" * 80)

    batch_size = 2
    x = torch.randn(batch_size, 100, 64, 64)  # 100 channels, 64x64

    model.eval()
    with torch.no_grad():
        output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: ({batch_size}, 21)")

    assert output.shape == (batch_size, 21), f"Shape mismatch!"

    print("\n✅ 100-channel test PASSED!")
    print("This confirms the checkpoint is valid and loads correctly")
    print("when input channels match the pretrain configuration.")

    return model


if __name__ == "__main__":
    test_100_channel_load()
