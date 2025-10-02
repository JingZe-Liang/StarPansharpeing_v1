"""
Test script for the build_block function in src/stage2/layers/blocks.py.

This script tests various block types supported by the build_block function
and verifies their functionality with different configurations.
"""

import torch
import torch.nn as nn

from src.stage2.layers.blocks import build_block


def test_block_forward(
    block: nn.Module,
    input_shape: tuple[int, ...],
    cond_shape: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """
    Test forward pass of a block with given input shape.

    Parameters
    ----------
    block : nn.Module
        The block to test
    input_shape : tuple[int, ...]
        Shape of input tensor (B, C, H, W) for 2D or (B, L, C) for 1D
    cond_shape : tuple[int, ...] | None
        Shape of condition tensor if required by the block

    Returns
    -------
    torch.Tensor
        Output tensor from the block
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    block = block.to(device)
    x = torch.randn(*input_shape, device=device, dtype=torch.float32)
    block.eval()
    with torch.no_grad():
        # Check if this is SoftmaxCrossAttention2D which needs cond parameter
        if "CrossAttention" in type(block).__name__:
            if cond_shape is None:
                # Default condition shape for cross attention
                cond = torch.randn(
                    x.shape[0],
                    32,
                    x.shape[2],
                    x.shape[3],
                    device=device,
                    dtype=torch.float32,
                )  # Default to 32 channels
            else:
                cond = torch.randn(*cond_shape, device=device, dtype=torch.float32)

            with torch.autocast(device_type="cuda", dtype=torch.float32):
                output = block(x, cond)
        else:
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                output = block(x)
    return output


def test_residual_blocks() -> None:
    """Test ResBlock and GLUResBlock configurations."""
    print("Testing Residual Blocks...")

    # Test ResBlock
    res_block = build_block("ResBlock", 64, 64, "layernorm2d", "gelu")
    print(f"ResBlock: {res_block}")

    # Test with 2D input
    output = test_block_forward(res_block, (2, 64, 32, 32))
    print(f"ResBlock output shape: {output.shape}")
    assert output.shape == (2, 64, 32, 32), "ResBlock output shape mismatch"

    # Test GLUResBlock with different gate kernel sizes
    for gate_size in [1, 3, 5]:
        glu_block = build_block(
            f"GLUResBlock@{gate_size}", 64, 64, "layernorm2d", "gelu"
        )
        print(f"GLUResBlock@{gate_size}: {glu_block}")

        output = test_block_forward(glu_block, (2, 64, 32, 32))
        print(f"GLUResBlock@{gate_size} output shape: {output.shape}")
        assert output.shape == (
            2,
            64,
            32,
            32,
        ), f"GLUResBlock@{gate_size} output shape mismatch"


def test_attention_blocks() -> None:
    """Test attention-based blocks."""
    print("\nTesting Attention Blocks...")

    # Test ChannelAttentionResBlock
    ca_block = build_block(
        "ChannelAttentionResBlock@SEModule@2", 64, 64, "layernorm2d", "gelu"
    )
    print(f"ChannelAttentionResBlock: {ca_block}")

    output = test_block_forward(ca_block, (2, 64, 32, 32))
    print(f"ChannelAttentionResBlock output shape: {output.shape}")
    assert output.shape == (
        2,
        64,
        32,
        32,
    ), "ChannelAttentionResBlock output shape mismatch"

    # Test SoftmaxCrossAttention (different input/output channels)
    ca_block = build_block("SoftmaxCrossAttention@32", 64, 128, "layernorm2d", "gelu")
    print(f"SoftmaxCrossAttention: {ca_block}")

    # Test SoftmaxCrossAttention manually since it needs special handling
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ca_block = ca_block.to(device)
    x = torch.randn(2, 64, 32, 32, device=device, dtype=torch.float32)
    # For SoftmaxCrossAttention@32, the cond should have 32 channels (kv_in_channels)
    cond = torch.randn(
        2, 256, 32, device=device, dtype=torch.float32
    )  # (batch, seq_len, kv_channels=32)
    ca_block.eval()
    with torch.no_grad():
        output = ca_block(x, cond)
    print(f"SoftmaxCrossAttention output shape: {output.shape}")
    assert output.shape == (
        2,
        128,
        32,
        32,
    ), "SoftmaxCrossAttention output shape mismatch"


def test_efficient_vit_blocks() -> None:
    """Test EfficientViT-based blocks."""
    print("\nTesting EfficientViT Blocks...")

    # Test GLUMBConv with different expand ratios
    for ratio in [2, 4, 6]:
        glu_mb = build_block(f"GLUMBConv@{ratio}", 64, 64, "layernorm2d", "gelu")
        print(f"GLUMBConv@{ratio}: {glu_mb}")

        output = test_block_forward(glu_mb, (2, 64, 32, 32))
        print(f"GLUMBConv@{ratio} output shape: {output.shape}")
        assert output.shape == (
            2,
            64,
            32,
            32,
        ), f"GLUMBConv@{ratio} output shape mismatch"

    # Test EViT variants
    evit_blocks = ["EViTGLU", "EViTNormQKGLU", "EViTS5GLU"]
    for block_name in evit_blocks:
        block = build_block(block_name, 64, 64, "layernorm2d", "gelu")
        print(f"{block_name}: {block}")

        output = test_block_forward(block, (2, 64, 32, 32))
        print(f"{block_name} output shape: {output.shape}")
        assert output.shape == (2, 64, 32, 32), f"{block_name} output shape mismatch"


def test_nat_blocks() -> None:
    """Test Neighborhood Attention blocks."""
    print("\nTesting NAT Blocks...")

    # Test default ViTNAT configuration
    nat_block = build_block("ViTNAT", 64, 64, "layernorm2d", "gelu")
    print(f"ViTNAT (default): {nat_block}")

    output = test_block_forward(nat_block, (2, 64, 32, 32))
    print(f"ViTNAT (default) output shape: {output.shape}")
    assert output.shape == (2, 64, 32, 32), "ViTNAT (default) output shape mismatch"

    # Test custom ViTNAT configuration
    # ViTNAT@8@1@1@8@4 (k_size, stride, dilation, n_heads, ffn_ratio)
    nat_block = build_block("ViTNAT@8@1@1@8@4", 64, 64, "layernorm2d", "gelu")
    print(f"ViTNAT (custom): {nat_block}")

    output = test_block_forward(nat_block, (2, 64, 32, 32))
    print(f"ViTNAT (custom) output shape: {output.shape}")
    assert output.shape == (2, 64, 32, 32), "ViTNAT (custom) output shape mismatch"


def test_different_norms_and_activations() -> None:
    """Test blocks with different normalization and activation functions."""
    print("\nTesting Different Norms and Activations...")

    norms = ["layernorm2d", "batchnorm2d", "groupnorm"]
    acts = ["gelu", "silu", "relu", "swish"]

    for norm in norms:
        for act in acts:
            try:
                block = build_block("ResBlock", 32, 32, norm, act)
                output = test_block_forward(block, (1, 32, 16, 16))
                print(f"ResBlock with {norm} + {act}: ✓")
            except Exception as e:
                print(f"ResBlock with {norm} + {act}: ✗ ({e})")


def test_error_cases() -> None:
    """Test error handling for invalid configurations."""
    print("\nTesting Error Cases...")

    # Test mismatched input/output channels for blocks that require same channels
    try:
        block = build_block("ResBlock", 32, 64, "layernorm2d", "gelu")
        assert False, "Should have raised AssertionError for channel mismatch"
    except AssertionError:
        print("ResBlock channel mismatch: ✓")

    try:
        block = build_block("GLUResBlock@3", 32, 64, "layernorm2d", "gelu")
        assert False, "Should have raised AssertionError for channel mismatch"
    except AssertionError:
        print("GLUResBlock channel mismatch: ✓")

    # Test unsupported block type
    try:
        block = build_block("UnsupportedBlock", 64, 64, "layernorm2d", "gelu")
        assert False, "Should have raised ValueError for unsupported block"
    except ValueError as e:
        print(f"Unsupported block error: ✓ ({e})")

    # Test invalid configuration parsing
    try:
        block = build_block("GLUResBlock@", 64, 64, "layernorm2d", "gelu")
        assert False, "Should have raised error for invalid configuration"
    except Exception as e:
        print(f"Invalid configuration error: ✓ ({e})")


def test_block_parameters() -> None:
    """Test that blocks have expected parameters and are trainable."""
    print("\nTesting Block Parameters...")

    block_types = [
        "ResBlock",
        "GLUResBlock@3",
        "GLUMBConv@4",
        "EViTGLU",
        "ChannelAttentionResBlock@SEModule@2",
    ]

    for block_type in block_types:
        block = build_block(block_type, 64, 64, "layernorm2d", "gelu")

        # Count parameters
        total_params = sum(p.numel() for p in block.parameters())
        trainable_params = sum(p.numel() for p in block.parameters() if p.requires_grad)

        print(f"{block_type}:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")

        # Test gradient flow
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        block = block.to(device)
        x = torch.randn(1, 64, 32, 32, device=device, requires_grad=True)
        output = block(x)
        loss = output.sum()
        loss.backward()

        # Check if gradients were computed
        has_grad = any(p.grad is not None for p in block.parameters())
        print(f"  Has gradients: {'✓' if has_grad else '✗'}")


def main() -> None:
    """Run all tests for the build_block function."""
    print("=" * 60)
    print("Testing build_block function")
    print("=" * 60)
    test_residual_blocks()
    test_attention_blocks()
    test_efficient_vit_blocks()
    test_nat_blocks()
    test_different_norms_and_activations()
    test_error_cases()
    test_block_parameters()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
