"""Test deepcopy in training environment with accelerator."""

import torch
from copy import deepcopy
import sys
from pathlib import Path

sys.path.insert(0, "/home/user/zihancao/Project/hyperspectral-1d-tokenizer")

import hydra
from omegaconf import OmegaConf
from accelerate import Accelerator

from src.stage2.cloud_removal.model.sit import SiT_B_2


def test_scenario_1():
    """Test: Model after initialization"""
    print("\n" + "=" * 80)
    print("SCENARIO 1: Fresh model")
    print("=" * 80)

    model = SiT_B_2(input_size=64, in_channels=16, cond_drop_prob=0.0, qk_norm=True)

    try:
        _ = deepcopy(model)
        print("✓ Fresh model can be deepcopied")
        return True
    except Exception as e:
        print(f"✗ Fresh model CANNOT be deepcopied: {e}")
        return False


def test_scenario_2():
    """Test: Model after lazy init"""
    print("\n" + "=" * 80)
    print("SCENARIO 2: Model after lazy module initialization")
    print("=" * 80)

    model = SiT_B_2(input_size=64, in_channels=16, cond_drop_prob=0.0, qk_norm=True)

    # Initialize lazy modules
    dummy_x = torch.randn(1, 16, 64, 64)
    dummy_t = torch.tensor([0.5])
    dummy_cond = torch.randn(1, 16, 64, 64)

    with torch.no_grad():
        _ = model(dummy_x, dummy_t, conditions=dummy_cond)

    try:
        _ = deepcopy(model)
        print("✓ Model after lazy init can be deepcopied")
        return True
    except Exception as e:
        print(f"✗ Model after lazy init CANNOT be deepcopied: {e}")
        return False


def test_scenario_3():
    """Test: Model after accelerator.prepare()"""
    print("\n" + "=" * 80)
    print("SCENARIO 3: Model after accelerator.prepare()")
    print("=" * 80)

    model = SiT_B_2(input_size=64, in_channels=16, cond_drop_prob=0.0, qk_norm=True)

    # Initialize lazy modules first
    dummy_x = torch.randn(1, 16, 64, 64)
    dummy_t = torch.tensor([0.5])
    dummy_cond = torch.randn(1, 16, 64, 64)

    with torch.no_grad():
        _ = model(dummy_x, dummy_t, conditions=dummy_cond)

    # Prepare with accelerator
    accelerator = Accelerator(mixed_precision="bf16")
    model = accelerator.prepare(model)

    print(f"Model type after prepare: {type(model)}")

    try:
        _ = deepcopy(model)
        print("✓ Wrapped model can be deepcopied")
        return True
    except Exception as e:
        print(f"✗ Wrapped model CANNOT be deepcopied: {e}")

        # Try unwrapping
        print("\n  Trying with unwrapped model...")
        model_unwrapped = accelerator.unwrap_model(model)
        print(f"  Unwrapped model type: {type(model_unwrapped)}")

        try:
            _ = deepcopy(model_unwrapped)
            print("  ✓ Unwrapped model can be deepcopied")
            return False  # Wrapped failed, but unwrapped works
        except Exception as e2:
            print(f"  ✗ Unwrapped model also CANNOT be deepcopied: {e2}")
            return False


def test_scenario_4():
    """Test: Model after forward pass with gradient"""
    print("\n" + "=" * 80)
    print("SCENARIO 4: Model after forward pass with gradient")
    print("=" * 80)

    model = SiT_B_2(input_size=64, in_channels=16, cond_drop_prob=0.0, qk_norm=True)

    # Initialize lazy modules
    dummy_x = torch.randn(1, 16, 64, 64)
    dummy_t = torch.tensor([0.5])
    dummy_cond = torch.randn(1, 16, 64, 64)

    with torch.no_grad():
        _ = model(dummy_x, dummy_t, conditions=dummy_cond)

    # Now do a forward pass WITH gradient
    dummy_x2 = torch.randn(1, 16, 64, 64, requires_grad=True)
    dummy_t2 = torch.tensor([0.5])
    dummy_cond2 = torch.randn(1, 16, 64, 64)

    out, zs, _ = model(dummy_x2, dummy_t2, conditions=dummy_cond2)

    print(f"Model has gradients: {any(p.grad is not None for p in model.parameters())}")
    print(f"Attn v_last exists: {hasattr(model.blocks[0].attn, 'v_last') and model.blocks[0].attn.v_last is not None}")

    if hasattr(model.blocks[0].attn, "v_last") and model.blocks[0].attn.v_last is not None:
        v_last = model.blocks[0].attn.v_last
        print(f"  v_last.requires_grad: {v_last.requires_grad}")
        print(f"  v_last.is_leaf: {v_last.is_leaf}")
        print(f"  v_last.grad_fn: {v_last.grad_fn}")

    try:
        _ = deepcopy(model)
        print("✓ Model after forward pass can be deepcopied")
        return True
    except Exception as e:
        print(f"✗ Model after forward pass CANNOT be deepcopied: {e}")

        # Check if it's v_last
        if hasattr(model.blocks[0].attn, "v_last"):
            print("\n  Checking if v_last is the problem...")
            # Temporarily clear v_last
            for block in model.blocks:
                block.attn.v_last = None

            try:
                _ = deepcopy(model)
                print("  ✓ Model with v_last=None can be deepcopied!")
                print("  >>> v_last is the problem! <<<")
                return False
            except Exception as e2:
                print(f"  ✗ Still cannot deepcopy: {e2}")

        return False


def main():
    print("\n" + "🔍" * 40)
    print("Testing SiT Model Deepcopy in Different Scenarios")
    print("🔍" * 40)

    results = []

    results.append(("Scenario 1", test_scenario_1()))
    results.append(("Scenario 2", test_scenario_2()))
    results.append(("Scenario 3", test_scenario_3()))
    results.append(("Scenario 4", test_scenario_4()))

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
