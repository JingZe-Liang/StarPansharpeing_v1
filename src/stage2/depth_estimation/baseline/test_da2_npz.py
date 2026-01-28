import argparse
import os
import sys
import glob
import torch
import numpy as np
from tqdm import tqdm

# Setup paths to allow imports from project root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# ../../../../ -> project root
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.stage2.depth_estimation.metrics.torchmetrics_depth import DepthEstimationMetrics


def get_args():
    parser = argparse.ArgumentParser(description="Compute metrics from saved NPZ files for Depth Anything V2")
    parser.add_argument(
        "--npz-dir", type=str, required=True, help="Directory containing .npz files saved by infer_da2_us3d.py"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=64.0,
        help="Scale factor to restore GT to metric units (dataset uses normalized depth)",
    )
    parser.add_argument(
        "--min-depth-eval",
        type=float,
        default=1.0,
        help="Minimum ground truth depth (in meters) to evaluate on. Helps avoid small value instability in AbsRel.",
    )
    return parser.parse_args()


def main():
    args = get_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Restoring GT scale by factor: {args.scale}")
    print(f"Evaluating only on GT >= {args.min_depth_eval} meters")

    # Initialize Metrics Calculator
    # Align to recovered metric GT
    metric_calc: DepthEstimationMetrics = DepthEstimationMetrics(align_mode="scale_shift")
    metric_calc.to(device)

    # Find all npz files
    npz_files = sorted(glob.glob(os.path.join(args.npz_dir, "*.npz")))
    if not npz_files:
        print(f"No .npz files found in {args.npz_dir}")
        sys.exit(1)

    print(f"Found {len(npz_files)} npz files. Starting evaluation...")

    for npz_path in tqdm(npz_files):
        try:
            data = np.load(npz_path)

            pred_np = data["pred"]
            gt_np = data["gt"]
            mask_np = data["mask"]

            # Convert to torch tensors
            pred_tensor = torch.from_numpy(pred_np).unsqueeze(0).unsqueeze(0).to(device)
            gt_tensor = torch.from_numpy(gt_np).unsqueeze(0).unsqueeze(0).to(device)
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)

            # 1. Restore Metric Scale
            gt_tensor = gt_tensor * args.scale

            # 2. Refine Mask
            # Original mask (valid pixels) AND GT >= min_eval (avoid division by near-zero)
            eval_mask = mask_tensor & (gt_tensor >= args.min_depth_eval)

            # Update metrics
            metric_calc.update(pred_tensor, gt_tensor, eval_mask)

        except Exception as e:
            print(f"Error processing {npz_path}: {e}")
            continue

        except Exception as e:
            print(f"Error processing {npz_path}: {e}")
            continue

    print("Computing final aggregated metrics...")
    metrics = metric_calc.compute()

    print("\n" + "=" * 30)
    print("      Evaluation Results      ")
    print("=" * 30)

    # Sort keys for cleaner output
    sorted_keys = sorted(metrics.keys())
    for k in sorted_keys:
        val = metrics[k].item()
        print(f"{k:<20}: {val:.6f}")

    print("=" * 30)

    # Save to file in the same directory
    result_path = os.path.join(args.npz_dir, "metrics_summary.txt")
    with open(result_path, "w") as f:
        f.write("Evaluation Results\n")
        f.write("==================\n")
        for k in sorted_keys:
            val = metrics[k].item()
            f.write(f"{k}: {val:.6f}\n")
    print(f"Results saved to {result_path}")


if __name__ == "__main__":
    """
    python src/stage2/depth_estimation/baseline/test_da2_npz.py \
    --npz-dir runs/stage2_depth_estimation_da2_baseline/us3d_oma_test/npz
    """
    main()
