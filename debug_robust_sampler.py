"""Debug script to test the toy label map issue in robust_sampler.py"""

import numpy as np

# Test a corrected version with consistent shape
try:
    gt_2d_corrected = np.array(
        [
            [1, 1, 0, 2, 2, 1],
            [3, 0, 1, 3, 3, 0],
            [2, 2, 2, 0, 1, 0],
            [4, 4, 4, 4, 4, 0],  # small class (5 pixels)
            [5, 5, 5, 5, 5, 0],  # medium class (5 pixels)
            [6, 6, 6, 6, 0, 0],  # medium class (4 pixels)
            [7, 7, 7, 7, 0, 0],  # medium class (4 pixels)
            [8, 8, 8, 8, 0, 0],  # large class (4 pixels)
            [9, 9, 9, 9, 0, 0],  # large class (4 pixels)
            [0, 0, 0, 0, 0, 0],  # background (ignored)
        ]
    )
    print(f"Array shape: {gt_2d_corrected.shape}")
    print(f"Array dtype: {gt_2d_corrected.dtype}")
    print("SUCCESS: Corrected array created without errors")

    # Test class distribution
    unique, counts = np.unique(gt_2d_corrected, return_counts=True)
    print("\nClass distribution:")
    for cls, cnt in zip(unique, counts):
        if cls != 0:  # Skip background
            print(f"Class {cls}: {cnt} pixels")

except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("Testing with the RobustHyperspectralSampler class:")

try:
    from src.stage2.segmentation.data.robust_sampler import RobustHyperspectralSampler

    # Create sampler with corrected array
    print("\nCreating sampler with corrected array:")
    sampler = RobustHyperspectralSampler(gt_2d_corrected, verbose=True)

    # Test all sampling strategies
    strategies = [
        "equal_size",
        "proportional",
        "stratified_robust",
        "adaptive_inverse",
        "adaptive_sqrt",
        "adaptive_log",
        "combined_smart",
    ]

    for strategy in strategies:
        print(f"\n{'=' * 30} Testing strategy: {strategy} {'=' * 30}")
        sampler.strategy = strategy

        try:
            if strategy == "stratified_robust":
                result = sampler.sample(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, min_per_class=1)
                print(f"Return type: {type(result)}")
                print(f"Return structure: Tuple of (train_indices, val_indices, test_indices)")

                train_idx, val_idx, test_idx = result
                print(f"Train indices type: {type(train_idx)}")
                print(f"Val indices type: {type(val_idx)}")
                print(f"Test indices type: {type(test_idx)}")

                # Show detailed results
                for split_name, indices in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
                    total = sum(len(v) for v in indices.values())
                    print(f"\n{split_name.upper()} SPLIT:")
                    print(f"  Total pixels: {total}")
                    print(f"  Classes: {list(indices.keys())}")
                    for class_id, class_indices in indices.items():
                        print(f"    Class {class_id}: {len(class_indices)} pixels")

            elif strategy == "proportional":
                result = sampler.sample(total_samples=30, min_per_class=1)
                print(f"Return type: {type(result)}")
                print(f"Return structure: Dict[class_id, List[int]]")

                print(f"\nPROPORTIONAL SAMPLING RESULTS:")
                total = sum(len(v) for v in result.values())
                print(f"  Total pixels: {total}")
                print(f"  Classes: {list(result.keys())}")
                for class_id, indices in result.items():
                    print(f"    Class {class_id}: {len(indices)} pixels")

            elif strategy == "equal_size":
                result = sampler.sample(samples_per_class=3)
                print(f"Return type: {type(result)}")
                print(f"Return structure: Dict[class_id, List[int]]")

                print(f"\nEQUAL SIZE SAMPLING RESULTS:")
                total = sum(len(v) for v in result.values())
                print(f"  Total pixels: {total}")
                print(f"  Classes: {list(result.keys())}")
                for class_id, indices in result.items():
                    print(f"    Class {class_id}: {len(indices)} pixels")

            elif strategy.startswith("adaptive"):
                if strategy == "adaptive_inverse":
                    result = sampler.sample(target_size=25)
                elif strategy == "adaptive_sqrt":
                    result = sampler.sample(target_size=25)
                elif strategy == "adaptive_log":
                    result = sampler.sample(target_size=25)

                print(f"Return type: {type(result)}")
                print(f"Return structure: Dict[class_id, List[int]]")

                print(f"\n{strategy.upper()} SAMPLING RESULTS:")
                total = sum(len(v) for v in result.values())
                print(f"  Total pixels: {total}")
                print(f"  Classes: {list(result.keys())}")
                for class_id, indices in result.items():
                    print(f"    Class {class_id}: {len(indices)} pixels")

            elif strategy == "combined_smart":
                result = sampler.sample(target_samples_per_class=5)
                print(f"Return type: {type(result)}")
                print(f"Return structure: Dict[class_id, List[int]]")

                print(f"\nCOMBINED SMART SAMPLING RESULTS:")
                total = sum(len(v) for v in result.values())
                print(f"  Total pixels: {total}")
                print(f"  Classes: {list(result.keys())}")
                for class_id, indices in result.items():
                    print(f"    Class {class_id}: {len(indices)} pixels")

            print(f"\nStrategy '{strategy}' executed successfully!")

        except Exception as e:
            print(f"ERROR with strategy '{strategy}': {type(e).__name__}: {e}")

    # Show sampling summary
    print(f"\n{'=' * 30} SAMPLING SUMMARY {'=' * 30}")
    summary = sampler.get_sampling_summary()
    print("Summary statistics:")
    for key, value in summary.items():
        if key != "class_distribution":
            print(f"  {key}: {value}")
        else:
            print(f"  {key}:")
            for class_id, count in value.items():
                print(f"    Class {class_id}: {count} pixels")

except ImportError as e:
    print(f"Import error: {e}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
