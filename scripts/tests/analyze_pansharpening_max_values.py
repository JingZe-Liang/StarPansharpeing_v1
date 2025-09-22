import torch
import numpy as np
from src.stage2.pansharpening.data.pansharpening_loader import (
    get_pansharp_wds_dataloader,
)


def analyze_dataset_max_value(dataset_name: str, tar_path: str) -> dict:
    """
    Analyze the maximum values and 99th percentile of a pansharpening dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g., "WV2", "WV3", "WV4")
    tar_path : str
        Path to the tar file containing the dataset

    Returns
    ----------
    dict
        Dictionary containing maximum and 99th percentile values for different data types
    """
    print(f"\n=== Analyzing {dataset_name} Dataset ===")
    print(f"Tar path: {tar_path}")

    ds, dl = get_pansharp_wds_dataloader(
        tar_path,
        batch_size=1,
        shuffle_size=0,
        num_workers=0,
        add_satellite_name=False,
        to_neg_1_1=False,
        resample=False,
    )

    max_lrms = 0.0
    max_hrms = 0.0
    max_pan = 0.0
    sample_count = 0
    invalid_samples = 0

    # Collect all values for percentile calculation
    all_lrms = []
    all_hrms = []
    all_pan = []

    for sample in dl:
        # Check for invalid values (inf or nan)
        lrms_valid = torch.isfinite(sample["lrms"])
        hrms_valid = torch.isfinite(sample["hrms"])
        pan_valid = torch.isfinite(sample["pan"])

        if not (lrms_valid.all() and hrms_valid.all() and pan_valid.all()):
            invalid_samples += 1
            continue

        lrms_max = sample["lrms"].max().item()
        hrms_max = sample["hrms"].max().item()
        pan_max = sample["pan"].max().item()

        max_lrms = max(max_lrms, lrms_max)
        max_hrms = max(max_hrms, hrms_max)
        max_pan = max(max_pan, pan_max)

        # Collect all pixel values for percentile calculation
        all_lrms.extend(sample["lrms"].flatten().tolist())
        all_hrms.extend(sample["hrms"].flatten().tolist())
        all_pan.extend(sample["pan"].flatten().tolist())

        sample_count += 1

        if sample_count % 10 == 0:
            print(f"Processed {sample_count} valid samples...")
            print(f"Current max - LRMS: {max_lrms}, HRMS: {max_hrms}, PAN: {max_pan}")

    # Calculate 99th percentiles
    if all_lrms:
        lrms_99th = np.percentile(all_lrms, 99.5)
        hrms_99th = np.percentile(all_hrms, 99.5)
        pan_99th = np.percentile(all_pan, 99.5)
    else:
        lrms_99th = hrms_99th = pan_99th = 0.0

    result = {
        "dataset_name": dataset_name,
        "max_lrms": max_lrms,
        "max_hrms": max_hrms,
        "max_pan": max_pan,
        "lrms_99th": lrms_99th,
        "hrms_99th": hrms_99th,
        "pan_99th": pan_99th,
        "sample_count": sample_count,
        "invalid_samples": invalid_samples,
        "total_pixels_lrms": len(all_lrms),
        "total_pixels_hrms": len(all_hrms),
        "total_pixels_pan": len(all_pan),
    }

    print(f"\n=== Results for {dataset_name} ===")
    print(f"Total valid samples processed: {sample_count}")
    print(f"Invalid samples (with inf/nan values): {invalid_samples}")
    print(f"Total pixels analyzed:")
    print(f"  LRMS: {len(all_lrms):,}")
    print(f"  HRMS: {len(all_hrms):,}")
    print(f"  PAN: {len(all_pan):,}")
    print(f"\nMaximum values:")
    print(f"  LRMS: {max_lrms}")
    print(f"  HRMS: {max_hrms}")
    print(f"  PAN: {max_pan}")
    print(f"\n99th percentile values:")
    print(f"  LRMS: {lrms_99th:.2f}")
    print(f"  HRMS: {hrms_99th:.2f}")
    print(f"  PAN: {pan_99th:.2f}")
    print("=" * 60)

    return result


def main():
    """
    Main function to analyze all pansharpening datasets.
    """
    # Configuration based on the updated paths
    datasets = {
        "QB": "data/Downstreams/PanCollectionV2/QB/pansharpening_reduced/Pansharpening_QB_train.tar",
        "IKONOS": "data/Downstreams/PanCollectionV2/IKONOS/pansharpening_reduced/Pansharpening_IKONOS_train.tar",
        "WV2": "data/Downstreams/PanCollectionV2/WV2/pansharpening_reduced/Pansharpening_WV2_train.tar",
        "WV3": "data/Downstreams/PanCollectionV2/WV3/pansharpening_reduced/Pansharping_WV3_train.tar",
        "WV4": "data/Downstreams/PanCollectionV2/WV4/pansharpening_reduced/Pansharpening_WV4_train.tar",
    }

    results = []

    # Analyze each dataset
    for dataset_name, tar_path in datasets.items():
        try:
            result = analyze_dataset_max_value(dataset_name, tar_path)
            results.append(result)
        except Exception as e:
            print(f"Error analyzing {dataset_name}: {e}")
            results.append({"dataset_name": dataset_name, "error": str(e)})

    # Print summary
    print("\n" + "=" * 80)
    print("COMPLETE SUMMARY OF ALL PANSHARPENING DATASETS")
    print("=" * 80)

    for result in results:
        if "error" not in result:
            print(f"{result['dataset_name']}:")
            print(f"  Max values:")
            print(f"    LRMS: {result['max_lrms']}")
            print(f"    HRMS: {result['max_hrms']}")
            print(f"    PAN: {result['max_pan']}")
            print(f"  99th percentile values:")
            print(f"    LRMS: {result['lrms_99th']:.2f}")
            print(f"    HRMS: {result['hrms_99th']:.2f}")
            print(f"    PAN: {result['pan_99th']:.2f}")
            print(f"  Data summary:")
            print(f"    Valid samples: {result['sample_count']}")
            print(f"    Invalid samples: {result['invalid_samples']}")
            print(f"    Total pixels (LRMS): {result['total_pixels_lrms']:,}")
            print(f"    Total pixels (HRMS): {result['total_pixels_hrms']:,}")
            print(f"    Total pixels (PAN): {result['total_pixels_pan']:,}")
        else:
            print(f"{result['dataset_name']}: ERROR - {result['error']}")
        print()


if __name__ == "__main__":
    main()
