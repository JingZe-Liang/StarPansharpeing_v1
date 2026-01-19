"""Test script to find which component of SiT model cannot be deepcopied."""

import torch
from copy import deepcopy
import sys

sys.path.insert(0, "/home/user/zihancao/Project/hyperspectral-1d-tokenizer")

from src.stage2.cloud_removal.model.sit import SiT_B_2


def test_deepcopy_module(module, name="module"):
    """Test if a module can be deepcopied."""
    try:
        _ = deepcopy(module)
        print(f"✓ {name} can be deepcopied")
        return True
    except Exception as e:
        print(f"✗ {name} CANNOT be deepcopied: {e}")
        return False


def main():
    print("=" * 80)
    print("Testing SiT_B_2 deepcopy compatibility")
    print("=" * 80)

    # Create model
    print("\n1. Creating model...")
    model = SiT_B_2(input_size=64, in_channels=16, cond_drop_prob=0.0, qk_norm=True)

    # Initialize lazy modules
    print("\n2. Initializing lazy modules...")
    dummy_x = torch.randn(1, 16, 64, 64)
    dummy_t = torch.tensor([0.5])
    dummy_cond = torch.randn(1, 16, 64, 64)

    with torch.no_grad():
        _ = model(dummy_x, dummy_t, conditions=dummy_cond)
    print("   Lazy modules initialized")

    # Test full model
    print("\n3. Testing full model deepcopy...")
    if test_deepcopy_module(model, "Full model"):
        print("\n✓✓✓ Model can be deepcopied! No issues found.")
        return

    # If full model fails, test individual components
    print("\n4. Testing individual components...")

    # Test embedders
    test_deepcopy_module(model.x_embedder, "x_embedder")
    test_deepcopy_module(model.cond_embedder, "cond_embedder")
    test_deepcopy_module(model.t_embedder, "t_embedder")

    # Test pos_embed
    print(f"\n5. Checking pos_embed...")
    print(f"   pos_embed type: {type(model.pos_embed)}")
    print(f"   pos_embed requires_grad: {model.pos_embed.requires_grad}")
    print(f"   pos_embed is_leaf: {model.pos_embed.is_leaf}")
    test_deepcopy_module(model.pos_embed, "pos_embed")

    # Test blocks
    print(f"\n6. Testing {len(model.blocks)} transformer blocks...")
    for i, block in enumerate(model.blocks):
        if not test_deepcopy_module(block, f"block[{i}]"):
            # Test sub-components of failed block
            print(f"   Testing sub-components of block[{i}]:")
            test_deepcopy_module(block.norm1, f"  - block[{i}].norm1")
            test_deepcopy_module(block.attn, f"  - block[{i}].attn")
            test_deepcopy_module(block.norm2, f"  - block[{i}].norm2")
            test_deepcopy_module(block.mlp, f"  - block[{i}].mlp")
            test_deepcopy_module(block.adaLN_modulation, f"  - block[{i}].adaLN_modulation")
            test_deepcopy_module(block.cond_modulation, f"  - block[{i}].cond_modulation")

            # Check v_last
            if hasattr(block.attn, "v_last") and block.attn.v_last is not None:
                print(f"   block[{i}].attn.v_last exists!")
                print(f"   - type: {type(block.attn.v_last)}")
                print(f"   - requires_grad: {block.attn.v_last.requires_grad}")
                print(f"   - is_leaf: {block.attn.v_last.is_leaf}")

    # Test projectors
    print(f"\n7. Testing {len(model.projectors)} projectors...")
    for i, proj in enumerate(model.projectors):
        test_deepcopy_module(proj, f"projector[{i}]")

    # Test final layer
    test_deepcopy_module(model.final_layer, "final_layer")

    # Test RoPE
    test_deepcopy_module(model.feat_rope, "feat_rope")

    # Test mask_token
    print(f"\n8. Checking mask_token...")
    print(f"   mask_token type: {type(model.mask_token)}")
    print(f"   mask_token requires_grad: {model.mask_token.requires_grad}")
    print(f"   mask_token is_leaf: {model.mask_token.is_leaf}")

    # Test fusion_proj
    test_deepcopy_module(model.fusion_proj, "fusion_proj")

    print("\n" + "=" * 80)
    print("Testing complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
