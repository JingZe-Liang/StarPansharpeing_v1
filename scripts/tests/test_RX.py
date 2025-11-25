"""
Test script for RX algorithm implementations.

This script verifies the numerical consistency between NumPy and PyTorch
implementations of the RX algorithm.
"""

import torch
import numpy as np
import sys
import os

# Add the parent directory to the path to import RX
sys.path.append("src/stage2/detections/utils")
from RX import RX_numpy, RX_torch, RX


def test_numpy_torch_consistency():
    """Test that NumPy and PyTorch implementations produce the same results."""
    print("Testing NumPy vs PyTorch RX implementation consistency...")

    # Create a test image (H, W, C)
    np.random.seed(42)
    test_img_numpy = np.random.rand(256, 256, 10).astype(np.float32)

    # Convert to PyTorch format (B, C, H, W)
    test_img_torch = torch.from_numpy(test_img_numpy.transpose(2, 0, 1)).unsqueeze(0).float()

    # Run both implementations
    score_numpy = RX_numpy(test_img_numpy)
    score_torch = RX_torch(test_img_torch)[0].cpu().numpy()

    # Compare results
    diff = np.abs(score_numpy - score_torch)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"Max difference: {max_diff}")
    print(f"Mean difference: {mean_diff}")

    # Check if results are close (within numerical precision)
    assert np.allclose(score_numpy, score_torch, rtol=1e-4, atol=1e-4), (
        f"NumPy and PyTorch implementations differ by more than tolerance. Max diff: {max_diff}"
    )

    print("✓ NumPy and PyTorch implementations are consistent!")


def test_wrapper_function():
    """Test the wrapper function with different input types."""
    print("\nTesting wrapper function with different input types...")

    # Create test data
    np.random.seed(42)
    test_img_numpy = np.random.rand(64, 64, 5).astype(np.float32)
    test_img_torch_3d = torch.from_numpy(test_img_numpy.transpose(2, 0, 1)).float()
    test_img_torch_4d = test_img_torch_3d.unsqueeze(0)

    # Test all input types
    score_from_numpy = RX(test_img_numpy)
    score_from_torch_3d = RX(test_img_torch_3d)
    score_from_torch_4d = RX(test_img_torch_4d)

    # All should produce the same result (within numerical precision)
    assert np.allclose(score_from_numpy, score_from_torch_3d.cpu().numpy(), rtol=1e-4, atol=1e-4)
    assert np.allclose(score_from_numpy, score_from_torch_4d[0].cpu().numpy(), rtol=1e-4, atol=1e-4)

    print("✓ Wrapper function works correctly with all input types!")


def test_batch_processing():
    """Test batch processing capability."""
    print("\nTesting batch processing...")

    # Create batched test data
    batch_size = 4
    np.random.seed(42)

    # Create individual images and process them one by one
    individual_scores = []
    batch_data = []

    for i in range(batch_size):
        # Use consistent dimensions for all images in the batch
        img = np.random.rand(64, 64, 5).astype(np.float32)
        individual_scores.append(RX_numpy(img))
        # Convert from (H, W, C) to (C, H, W) format
        batch_data.append(torch.from_numpy(img.transpose(2, 0, 1)).float())

    # Process as batch
    batch_tensor = torch.stack(batch_data, dim=0)
    print(f"Batch tensor shape: {batch_tensor.shape}")

    # Try to run RX_torch with error handling
    try:
        batch_scores = RX_torch(batch_tensor)
    except Exception as e:
        print(f"Error in RX_torch: {e}")
        raise e

    # Compare results
    for i in range(batch_size):
        individual_score = individual_scores[i]
        batch_score = batch_scores[i].cpu().numpy()
        print(f"Sample {i}: Individual score shape: {individual_score.shape}, Batch score shape: {batch_score.shape}")

        assert np.allclose(individual_score, batch_score, rtol=1e-4, atol=1e-4), (
            f"Batch processing differs from individual processing for sample {i}"
        )

    print("✓ Batch processing works correctly!")


def test_edge_cases():
    """Test edge cases."""
    print("\nTesting edge cases...")

    # Test with constant image (all zeros)
    zero_img = np.zeros((64, 64, 3))
    score = RX(zero_img)
    # Should produce zero scores (or very close to zero)
    assert np.allclose(score, 0, atol=1e-6)

    # Test with constant image (all ones)
    ones_img = np.ones((64, 64, 3))
    score = RX(ones_img)
    # Should produce zero scores (or very close to zero)
    assert np.allclose(score, 0, atol=1e-6)

    print("✓ Edge cases handled correctly!")


def run_all_tests():
    """Run all tests."""
    print("Running RX algorithm tests...\n")

    try:
        test_numpy_torch_consistency()
        test_wrapper_function()
        test_batch_processing()
        test_edge_cases()

        print("\n🎉 All tests passed!")
        return True
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
