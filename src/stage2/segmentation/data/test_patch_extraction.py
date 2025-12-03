"""
Independent test script to verify patch extraction correctness.

This script validates that:
1. Extracted patches are centered at the specified coordinates
2. Label patch centers match the original GT at those coordinates
3. Image and label patches are properly aligned
4. Padding is correctly applied at boundaries
5. Invalid boundary coordinates are properly skipped

Test Suite
----------
- Test 1: Patch Center Alignment
  Verifies that each extracted patch's center pixel matches the original GT
  at the specified coordinate.

- Test 2: Patch Dimensions
  Ensures all patches have the expected shape (patch_size x patch_size).

- Test 3: Boundary Padding
  Tests that valid boundary patches can be extracted with proper padding.

- Test 4: Invalid Boundary Coordinates
  Confirms that coordinates too close to edges are correctly skipped.

- Test 5: Unsampled Area Mask
  Validates the unsampled area mask has correct shape and dtype.

Run
---
python -m src.stage2.segmentation.data.test_patch_extraction

Expected Output
---------------
All 5 tests should PASS, confirming patch extraction is working correctly.
"""

import numpy as np
from loguru import logger

from .robust_sampler import RobustHyperspectralSampler, sample_img_with_gt_indices


def create_synthetic_test_data(height=128, width=128, num_bands=10, num_classes=5):
    """Create synthetic hyperspectral image and ground truth for testing."""
    img = np.random.rand(height, width, num_bands).astype(np.float32)

    total_pixels = height * width
    flat_gt = np.zeros(total_pixels, dtype=np.int32)

    # Create class distribution
    class_sizes = [int(0.15 * total_pixels) for _ in range(1, num_classes)]
    class_sizes.append(total_pixels - sum(class_sizes))  # remaining pixels

    start_idx = 0
    for class_id in range(1, num_classes + 1):
        end_idx = start_idx + class_sizes[class_id - 1]
        flat_gt[start_idx:end_idx] = class_id
        start_idx = end_idx

    # Shuffle and reshape
    np.random.shuffle(flat_gt)
    gt = flat_gt.reshape((height, width))

    return img, gt


def test_patch_center_alignment():
    """Test 1: Verify patch centers align with original GT coordinates."""
    logger.info("=" * 70)
    logger.info("Test 1: Patch Center Alignment")
    logger.info("=" * 70)

    height, width = 128, 128
    num_bands = 10
    patch_size = 32

    img, gt = create_synthetic_test_data(height, width, num_bands, num_classes=5)

    sampler = RobustHyperspectralSampler(gt, balance_strategy="equal_size", random_seed=42, skip_bg=0, verbose=False)

    sampled_indices = sampler.sample(samples_per_class=10)
    patches_dict, labels_dict, unsampled_area, used_coords = sample_img_with_gt_indices(
        img, gt, sampled_indices, patch_size=patch_size
    )

    total_patches = sum(len(v) for v in patches_dict.values())
    logger.info(f"Total patches extracted: {total_patches}")
    logger.info(f"Patch size: {patch_size}x{patch_size}")

    # Verify each patch
    mismatches = []
    center_offset = patch_size // 2

    for class_id, patch_list in patches_dict.items():
        for i, img_patch in enumerate(patch_list):
            label_patch = labels_dict[class_id][i]
            center_coord = used_coords[class_id][i]
            cx, cy = center_coord

            # Get center pixel from label patch
            label_center = label_patch[center_offset, center_offset]

            # Get original GT value at that coordinate
            gt_value = gt[cx, cy]

            # Verify they match
            if label_center != gt_value:
                mismatches.append(
                    {
                        "class_id": class_id,
                        "patch_idx": i,
                        "coord": (cx, cy),
                        "label_center": int(label_center),
                        "gt_value": int(gt_value),
                    }
                )

    if mismatches:
        logger.error(f"❌ Test FAILED: {len(mismatches)} mismatches found!")
        for mm in mismatches[:5]:
            logger.error(
                f"  Class {mm['class_id']}, Patch {mm['patch_idx']}: "
                f"coord={mm['coord']}, label_center={mm['label_center']}, "
                f"gt_value={mm['gt_value']}"
            )
        return False
    else:
        logger.success(f"✓ Test PASSED: All {total_patches} patches have correct center alignment!")
        return True


def test_patch_dimensions():
    """Test 2: Verify all patches have correct dimensions."""
    logger.info("\n" + "=" * 70)
    logger.info("Test 2: Patch Dimensions")
    logger.info("=" * 70)

    height, width = 64, 64
    num_bands = 8
    patch_size = 16

    img, gt = create_synthetic_test_data(height, width, num_bands, num_classes=4)

    sampler = RobustHyperspectralSampler(gt, balance_strategy="equal_size", random_seed=123, skip_bg=0, verbose=False)

    sampled_indices = sampler.sample(samples_per_class=5)
    patches_dict, labels_dict, unsampled_area, used_coords = sample_img_with_gt_indices(
        img, gt, sampled_indices, patch_size=patch_size
    )

    dimension_errors = []

    for class_id, patch_list in patches_dict.items():
        for i, img_patch in enumerate(patch_list):
            label_patch = labels_dict[class_id][i]

            expected_img_shape = (patch_size, patch_size, num_bands)
            expected_label_shape = (patch_size, patch_size)

            if img_patch.shape != expected_img_shape:
                dimension_errors.append(f"Image patch shape mismatch: {img_patch.shape} != {expected_img_shape}")

            if label_patch.shape != expected_label_shape:
                dimension_errors.append(f"Label patch shape mismatch: {label_patch.shape} != {expected_label_shape}")

    if dimension_errors:
        logger.error(f"❌ Test FAILED: {len(dimension_errors)} dimension errors!")
        for err in dimension_errors[:5]:
            logger.error(f"  {err}")
        return False
    else:
        total_patches = sum(len(v) for v in patches_dict.values())
        logger.success(f"✓ Test PASSED: All {total_patches} patches have correct dimensions!")
        return True


def test_boundary_padding():
    """Test 3: Verify padding at boundaries."""
    logger.info("\n" + "=" * 70)
    logger.info("Test 3: Boundary Padding")
    logger.info("=" * 70)

    height, width = 64, 64
    num_bands = 5
    patch_size = 20

    img, gt = create_synthetic_test_data(height, width, num_bands, num_classes=3)

    # Manually create coordinates near boundaries
    margin = patch_size // 2
    boundary_coords = {
        1: [
            (margin, margin),  # top-left corner (valid)
            (margin, width - margin - 1),  # top-right corner (valid)
            (height - margin - 1, margin),  # bottom-left corner (valid)
            (height - margin - 1, width - margin - 1),  # bottom-right corner (valid)
        ]
    }

    patches_dict, labels_dict, unsampled_area, used_coords = sample_img_with_gt_indices(
        img, gt, boundary_coords, patch_size=patch_size
    )

    total_extracted = sum(len(v) for v in patches_dict.values())
    logger.info(f"Attempted to extract {len(boundary_coords[1])} boundary patches")
    logger.info(f"Successfully extracted {total_extracted} patches")

    # All corner patches should be extracted since they meet the valid center criteria
    if total_extracted == len(boundary_coords[1]):
        logger.success("✓ Test PASSED: Boundary patches extracted correctly!")
        return True
    else:
        logger.error(f"❌ Test FAILED: Expected {len(boundary_coords[1])}, got {total_extracted}")
        return False


def test_invalid_boundary_coords():
    """Test 4: Verify that invalid boundary coordinates are skipped."""
    logger.info("\n" + "=" * 70)
    logger.info("Test 4: Invalid Boundary Coordinates Handling")
    logger.info("=" * 70)

    height, width = 64, 64
    num_bands = 5
    patch_size = 20
    margin = patch_size // 2

    img, gt = create_synthetic_test_data(height, width, num_bands, num_classes=3)

    # Create coordinates that are too close to borders (should be skipped)
    invalid_coords = {
        1: [
            (0, 0),  # too close to top-left
            (margin - 1, margin - 1),  # still too close
            (height - 1, width - 1),  # too close to bottom-right
            (height - margin, width - margin),  # beyond valid range
        ]
    }

    patches_dict, labels_dict, unsampled_area, used_coords = sample_img_with_gt_indices(
        img, gt, invalid_coords, patch_size=patch_size
    )

    total_extracted = sum(len(v) for v in patches_dict.values())
    logger.info(f"Attempted to extract {len(invalid_coords[1])} invalid boundary patches")
    logger.info(f"Extracted {total_extracted} patches (should be 0)")

    if total_extracted == 0:
        logger.success("✓ Test PASSED: Invalid boundary coordinates correctly skipped!")
        return True
    else:
        logger.error(f"❌ Test FAILED: Should skip all invalid coords, but extracted {total_extracted}")
        return False


def test_unsampled_area_mask():
    """Test 5: Verify unsampled area mask is correct."""
    logger.info("\n" + "=" * 70)
    logger.info("Test 5: Unsampled Area Mask")
    logger.info("=" * 70)

    height, width = 64, 64
    num_bands = 5
    patch_size = 16

    img, gt = create_synthetic_test_data(height, width, num_bands, num_classes=4)

    sampler = RobustHyperspectralSampler(gt, balance_strategy="equal_size", random_seed=999, skip_bg=0, verbose=False)

    sampled_indices = sampler.sample(samples_per_class=3)
    patches_dict, labels_dict, unsampled_area, used_coords = sample_img_with_gt_indices(
        img, gt, sampled_indices, patch_size=patch_size
    )

    # Check unsampled area shape
    if unsampled_area.shape != gt.shape:
        logger.error(f"❌ Test FAILED: Unsampled area shape {unsampled_area.shape} != GT shape {gt.shape}")
        return False

    # Check that unsampled area is boolean
    if unsampled_area.dtype != np.bool_:
        logger.error(f"❌ Test FAILED: Unsampled area dtype {unsampled_area.dtype} != np.bool_")
        return False

    sampled_pixels = np.sum(~unsampled_area)
    total_patches = sum(len(v) for v in patches_dict.values())
    expected_sampled_pixels = total_patches * patch_size * patch_size

    logger.info(f"Total patches: {total_patches}")
    logger.info(f"Sampled pixels in mask: {sampled_pixels}")
    logger.info(f"Expected sampled pixels: {expected_sampled_pixels}")

    # The sampled pixels should be approximately equal (may overlap)
    if sampled_pixels > 0 and sampled_pixels <= expected_sampled_pixels:
        logger.success("✓ Test PASSED: Unsampled area mask is valid!")
        return True
    # End of test_online_vs_offline_equivalence


def test_online_vs_offline_equivalence():
    """Test 6: Compare offline pre-extracted patches with online extraction for equivalence."""
    logger.info("\n" + "=" * 70)
    logger.info("Test 6: Online vs Offline Equivalence")
    logger.info("=" * 70)

    height, width = 64, 64
    num_bands = 5
    patch_size = 16
    img, gt = create_synthetic_test_data(height, width, num_bands, num_classes=4)

    sampler = RobustHyperspectralSampler(gt, balance_strategy="equal_size", random_seed=111, skip_bg=0, verbose=False)
    sampled_indices = sampler.sample(samples_per_class=6)

    # Offline: pre-extracted
    patches_dict, labels_dict, unsampled_area, used_coords = sample_img_with_gt_indices(
        img, gt, sampled_indices, patch_size=patch_size
    )

    # Online: for each used coord, extract patch using the same logic (pad -> slice)
    def extract_patch_online(img_a, gt_a, center, psize):
        margin = psize // 2
        padded_img = np.pad(img_a, ((margin, margin), (margin, margin), (0, 0)), mode="constant", constant_values=0)
        padded_gt = np.pad(gt_a, ((margin, margin), (margin, margin)), mode="constant", constant_values=255)
        px, py = int(center[0]), int(center[1])
        slices = slice(px, px + psize), slice(py, py + psize)
        return padded_img[slices[0], slices[1], :], padded_gt[slices[0], slices[1]]

    mismatches = []
    for class_id, patch_list in patches_dict.items():
        for i, offline_patch in enumerate(patch_list):
            center = used_coords[class_id][i]
            online_patch, online_label = extract_patch_online(img, gt, center, patch_size)
            # compare
            if not np.array_equal(online_patch, offline_patch):
                mismatches.append((class_id, i, "img"))
            if not np.array_equal(online_label, labels_dict[class_id][i]):
                mismatches.append((class_id, i, "label"))

    if mismatches:
        logger.error(f"❌ Test FAILED: {len(mismatches)} mismatches between online and offline extractions")
        for mm in mismatches[:10]:
            logger.error(f"  class {mm[0]} patch {mm[1]} type {mm[2]}")
        return False
    else:
        logger.success("✓ Test PASSED: Online extraction equals offline pre-extraction for all checked patches")
        return True
    # removed stray else


def run_all_tests():
    """Run all tests and report summary."""
    logger.info("\n" + "=" * 70)
    logger.info("PATCH EXTRACTION VALIDATION TEST SUITE")
    logger.info("=" * 70 + "\n")

    np.random.seed(2025)

    tests = [
        ("Patch Center Alignment", test_patch_center_alignment),
        ("Patch Dimensions", test_patch_dimensions),
        ("Boundary Padding", test_boundary_padding),
        ("Invalid Boundary Coords", test_invalid_boundary_coords),
        ("Unsampled Area Mask", test_unsampled_area_mask),
        ("Online vs Offline Equivalence", test_online_vs_offline_equivalence),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.exception(f"❌ Test '{test_name}' raised exception: {e}")
            results.append((test_name, False))

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {test_name}")

    logger.info("-" * 70)
    logger.info(f"Total: {passed}/{total} tests passed")

    if passed == total:
        logger.success("\n🎉 All tests PASSED! Patch extraction is correct.")
        return True
    else:
        logger.error(f"\n⚠️  {total - passed} test(s) FAILED. Please review the errors above.")
        return False


if __name__ == "__main__":
    """
    Run: python -m src.stage2.segmentation.data.test_patch_extraction
    """
    success = run_all_tests()
    exit(0 if success else 1)
