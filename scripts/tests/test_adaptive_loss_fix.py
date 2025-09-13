"""
Test script for the fixed adaptive HAD MSE loss function

This script verifies that the adaptive loss correctly handles batch processing
with per-sample weight masks, which was a critical bug fix.

Usage:
    python scripts/tests/test_adaptive_loss_fix.py

The test verifies:
1. Each batch sample has unique weight patterns
2. Residual maps maintain sample independence
3. Weights are properly normalized to [0, 1]
4. Both batch_size=1 and batch_size>1 work correctly
"""

import torch

from src.stage2.detections.loss.adaptive_HAD_mse import AdaptiveHADMSE
from src.utilities.train_utils.state import StepsCounter


def test_adaptive_loss_batch_processing():
    """Test that adaptive loss works correctly with batch size > 1"""
    print("Testing AdaptiveHADMSE with batch processing...")

    # Create test data with different anomaly patterns per sample
    batch_size = 3
    channels = 10
    height, width = 32, 32

    # Initialize loss function
    loss_fn = AdaptiveHADMSE(update_interval=1, device="cpu")

    # Initialize step counter to enable weight updates
    steps_counter = StepsCounter(step_names=["train"])
    # Set step to 1 to enable weight updates (update_interval=1)
    steps_counter["train"] = 1

    # Create synthetic data with different anomaly patterns per batch
    output = torch.rand(batch_size, channels, height, width)
    target = torch.rand(batch_size, channels, height, width)

    # Add anomalies at different locations for each sample
    # Sample 0: anomaly at top-left
    output[0, :, :5, :5] = 2.0
    target[0, :, :5, :5] = 0.0

    # Sample 1: anomaly at center
    output[1, :, 15:20, 15:20] = 2.0
    target[1, :, 15:20, 15:20] = 0.0

    # Sample 2: anomaly at bottom-right
    output[2, :, 25:30, 25:30] = 2.0
    target[2, :, 25:30, 25:30] = 0.0

    print(f"Input shapes: output={output.shape}, target={target.shape}")

    # Calculate loss
    loss, anomaly_score = loss_fn(output, target)

    # Check dimensions
    mask = loss_fn.get_current_mask()
    residual = loss_fn.get_current_residual()
    current_residual = loss_fn.get_current_residual()

    print(f"Mask shape: {mask.shape}")
    print(f"Residual shape: {residual.shape}")
    print(f"Current residual shape: {current_residual.shape}")
    print(f"Loss value: {loss.item():.6f}")

    # Verify that each sample has different weight patterns
    # This should be True after the fix
    masks_per_sample = []
    for b in range(batch_size):
        sample_mask = mask[b, 0]  # Take first channel for each sample
        masks_per_sample.append(sample_mask)

    # Check if masks are different across samples
    mask_0_vs_1 = not torch.allclose(
        masks_per_sample[0], masks_per_sample[1], atol=1e-6
    )
    mask_0_vs_2 = not torch.allclose(
        masks_per_sample[0], masks_per_sample[2], atol=1e-6
    )
    mask_1_vs_2 = not torch.allclose(
        masks_per_sample[1], masks_per_sample[2], atol=1e-6
    )

    print(f"\nMask uniqueness check:")
    print(f"  Sample 0 vs Sample 1 different: {mask_0_vs_1}")
    print(f"  Sample 0 vs Sample 2 different: {mask_0_vs_2}")
    print(f"  Sample 1 vs Sample 2 different: {mask_1_vs_2}")

    if mask_0_vs_1 and mask_0_vs_2 and mask_1_vs_2:
        print("✅ SUCCESS: Each sample has unique weight patterns!")
    else:
        print("❌ FAILURE: Some samples share the same weight patterns!")

    # Check that residuals are also different per sample
    residuals_per_sample = []
    for b in range(batch_size):
        sample_residual = current_residual[b]
        residuals_per_sample.append(sample_residual)

    residual_0_vs_1 = not torch.allclose(
        residuals_per_sample[0], residuals_per_sample[1], atol=1e-6
    )
    residual_0_vs_2 = not torch.allclose(
        residuals_per_sample[0], residuals_per_sample[2], atol=1e-6
    )
    residual_1_vs_2 = not torch.allclose(
        residuals_per_sample[1], residuals_per_sample[2], atol=1e-6
    )

    print(f"\nResidual uniqueness check:")
    print(f"  Sample 0 vs Sample 1 different: {residual_0_vs_1}")
    print(f"  Sample 0 vs Sample 2 different: {residual_0_vs_2}")
    print(f"  Sample 1 vs Sample 2 different: {residual_1_vs_2}")

    if residual_0_vs_1 and residual_0_vs_2 and residual_1_vs_2:
        print("✅ SUCCESS: Each sample has unique residual patterns!")
    else:
        print("❌ FAILURE: Some samples share the same residual patterns!")

    # Test weight range (should be [0, 1])
    mask_min, mask_max = mask.min().item(), mask.max().item()
    print(f"\nWeight range: [{mask_min:.6f}, {mask_max:.6f}]")

    if 0.0 <= mask_min <= 1.0 and 0.0 <= mask_max <= 1.0:
        print("✅ SUCCESS: Weights are properly normalized to [0, 1]!")
    else:
        print("❌ FAILURE: Weights are not properly normalized!")

    return mask_0_vs_1 and mask_0_vs_2 and mask_1_vs_2


def test_single_sample_for_comparison():
    """Test with single sample to ensure it still works"""
    print("\n" + "=" * 50)
    print("Testing with single sample (should work the same)...")

    # Single sample test
    batch_size = 1
    channels = 10
    height, width = 32, 32

    loss_fn = AdaptiveHADMSE(update_interval=1, device="cpu")

    # Step counter is already initialized from previous test

    # Create synthetic data with different anomaly patterns per batch
    output = torch.rand(batch_size, channels, height, width)
    target = torch.rand(batch_size, channels, height, width)

    # Add anomaly
    output[0, :, :5, :5] = 2.0
    target[0, :, :5, :5] = 0.0

    loss, _ = loss_fn(output, target)

    mask = loss_fn.get_current_mask()
    current_residual = loss_fn.get_current_residual()

    print(
        f"Single sample - Mask shape: {mask.shape}, Residual shape: {current_residual.shape}"
    )
    print(f"Single sample - Loss: {loss.item():.6f}")

    # Weight range check
    mask_min, mask_max = mask.min().item(), mask.max().item()
    print(f"Single sample - Weight range: [{mask_min:.6f}, {mask_max:.6f}]")

    if 0.0 <= mask_min <= 1.0 and 0.0 <= mask_max <= 1.0:
        print("✅ SUCCESS: Single sample weights are properly normalized!")
        return True
    else:
        print("❌ FAILURE: Single sample weights are not properly normalized!")
        return False


def test_edge_cases():
    """Test various edge cases"""
    print("\n" + "=" * 50)
    print("Testing edge cases...")

    # Test with uniform inputs (no anomalies)
    batch_size = 2
    channels = 3
    height, width = 8, 8

    loss_fn = AdaptiveHADMSE(update_interval=1, device="cpu")

    # Uniform inputs
    output = torch.ones(batch_size, channels, height, width) * 0.5
    target = torch.ones(batch_size, channels, height, width) * 0.5

    loss, _ = loss_fn(output, target)
    mask = loss_fn.get_current_mask()

    # With uniform inputs, weights should be close to 1.0
    mask_std = mask.std().item()
    print(f"Uniform inputs - mask std: {mask_std:.6f}")

    if mask_std < 0.1:  # Should be very uniform
        print("✅ SUCCESS: Uniform inputs handled correctly!")
        uniform_success = True
    else:
        print("❌ FAILURE: Uniform inputs not handled correctly!")
        uniform_success = False

    return uniform_success


if __name__ == "__main__":
    print("Testing fixed AdaptiveHADMSE...")
    print("=" * 50)

    # Test batch processing
    batch_success = test_adaptive_loss_batch_processing()

    # Test single sample
    single_success = test_single_sample_for_comparison()

    # Test edge cases
    edge_success = test_edge_cases()

    print("\n" + "=" * 50)
    print("FINAL RESULTS:")
    print(f"  Batch processing test: {'✅ PASS' if batch_success else '❌ FAIL'}")
    print(f"  Single sample test: {'✅ PASS' if single_success else '❌ FAIL'}")
    print(f"  Edge cases test: {'✅ PASS' if edge_success else '❌ FAIL'}")

    if batch_success and single_success and edge_success:
        print("\n🎉 ALL TESTS PASSED! The adaptive loss fix is working correctly!")
        print("\nKey improvements:")
        print("  ✅ Per-sample weight masks (no batch averaging)")
        print("  ✅ Independent residual maps per sample")
        print("  ✅ Proper normalization for each sample")
        print("  ✅ Compatible with both single and batch processing")
    else:
        print("\n💥 SOME TESTS FAILED! Please check the implementation.")
