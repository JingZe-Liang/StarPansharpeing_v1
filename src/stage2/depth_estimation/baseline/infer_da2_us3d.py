import argparse
import os
import sys
import torch
import cv2
import numpy as np
from tqdm import tqdm

# Setup paths to allow imports from project root
# Assuming this script is at src/stage2/depth_estimation/baseline/da2_us3d.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# ../../../../ -> project root
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Add DepthAnythingV2 to path
DA_PATH = os.path.join(PROJECT_ROOT, "src/stage2/depth_estimation/third_party/Depth_Anything_v2")
if DA_PATH not in sys.path:
    sys.path.append(DA_PATH)

try:
    from depth_anything_v2.dpt import DepthAnythingV2
except ImportError:
    print("Error: Could not import DepthAnythingV2. Make sure the submodule/repo is present.")
    sys.exit(1)

from src.stage2.depth_estimation.data.us3d_depth import US3DDepthStreamingDataset
from litdata import StreamingDataLoader

from src.stage2.depth_estimation.metrics.torchmetrics_depth import DepthEstimationMetrics


def get_args():
    parser = argparse.ArgumentParser(description="Run Depth Anything V2 inference on US3D dataset")
    parser.add_argument(
        "--encoder", type=str, default="vitl", choices=["vits", "vitb", "vitl", "vitg"], help="Model encoder size"
    )
    parser.add_argument("--outdir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--data-place", type=str, default="jax", choices=["jax", "oma"], help="US3D data location")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    return parser.parse_args()


def main():
    args = get_args()

    # Model Config
    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Model
    print(f"Loading Depth Anything V2 ({args.encoder})...")
    model = DepthAnythingV2(**model_configs[args.encoder])

    # Checkpoint path
    # Using the path where you soft-linked it: src/stage2/depth_estimation/third_party/Depth_Anything_v2/checkpoints
    ckpt_path = os.path.join(DA_PATH, f"checkpoints/depth_anything_v2_{args.encoder}.pth")

    if not os.path.exists(ckpt_path):
        # Fallback to absolute path just in case the link didn't work as expected or relative path is off
        # The user mentioned: /Data2/ZihanCao/Checkpoints/DepthAnythingV2
        fallback_path = f"/Data2/ZihanCao/Checkpoints/DepthAnythingV2/depth_anything_v2_{args.encoder}.pth"
        if os.path.exists(fallback_path):
            print(f"Checkpoint not found at {ckpt_path}, using fallback: {fallback_path}")
            ckpt_path = fallback_path
        else:
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path} or {fallback_path}. Please check paths.")

    print(f"Loading checkpoint from: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model = model.to(device).eval()

    # Dataset Config
    data_cfg = dict(
        jax=dict(
            train="data/Downstreams/US3D_Stereo_Matching/JAX/train",
            val="data/Downstreams/US3D_Stereo_Matching/JAX/vlal",
            test="data/Downstreams/US3D_Stereo_Matching/JAX/test",
        ),
        oma=dict(
            train="data/Downstreams/US3D_Stereo_Matching/OMA/train",
            val="data/Downstreams/US3D_Stereo_Matching/OMA/val",
            test="data/Downstreams/US3D_Stereo_Matching/OMA/test",
        ),
    )

    input_dir_rel = data_cfg[args.data_place][args.split]
    input_dir = os.path.join(PROJECT_ROOT, input_dir_rel)

    print(f"Loading dataset from {input_dir}...")
    if not os.path.exists(input_dir):
        print(f"Warning: Dataset path {input_dir} does not exist.")

    ds = US3DDepthStreamingDataset(
        input_dir=input_dir,
        target_key="agl",
        invalid_threshold=-500.0,
        clamp_range=(0, 64),
        scale=64,  # Matches training config
    )

    loader = StreamingDataLoader(ds, batch_size=1, num_workers=4, shuffle=False)

    # Initialize Metrics Calculator (Scale & Shift alignment for relative depth)
    metric_calc: DepthEstimationMetrics = DepthEstimationMetrics(align_mode="scale_shift")
    metric_calc.to(device)

    os.makedirs(args.outdir, exist_ok=True)
    vis_dir = os.path.join(args.outdir, "vis")
    npz_dir = os.path.join(args.outdir, "npz")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(npz_dir, exist_ok=True)

    print(f"Starting inference on {len(ds)} samples...")

    # Counter for valid saved samples
    saved_count = 0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):
            if args.limit and saved_count >= args.limit:
                break

            # Dataset returns img in [-1, 1] range, shape (B, C, H, W)
            img_tensor = batch["img"]  # (B, C, H, W)
            depth_gt = batch["depth"]  # (B, 1, H, W)
            valid_mask = batch["valid_mask"]  # (B, 1, H, W)

            # Process single image
            # Convert (-1, 1) tensor to (0, 255) numpy image for DA2
            img_np = img_tensor[0].cpu().numpy().transpose(1, 2, 0)  # (H, W, C) RGB
            img_np = (img_np + 1) / 2.0
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

            # DA2 expects BGR if mimicking cv2.imread
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # Run inference
            # infer_image returns a 2D numpy array (H, W)
            depth_pred = model.infer_image(img_bgr)

            # --- Metrics Update ---
            # Convert prediction to (B=1, C=1, H, W) tensor on device
            pred_tensor = torch.from_numpy(depth_pred).unsqueeze(0).unsqueeze(0).to(device)
            gt_tensor = depth_gt.to(device)
            mask_tensor = valid_mask.to(device)

            # Update metrics (accumulate stats)
            metric_calc.update(pred_tensor, gt_tensor, mask_tensor)
            # ----------------------

            # Save results
            save_name = f"{i:06d}"

            # Normalize and Colorize prediction for visualization
            pred_norm = (depth_pred - depth_pred.min()) / (depth_pred.max() - depth_pred.min() + 1e-8)
            pred_vis = (pred_norm * 255).astype(np.uint8)
            pred_vis = cv2.applyColorMap(pred_vis, cv2.COLORMAP_INFERNO)

            # Normalize and Colorize GT for visualization
            gt_val = depth_gt[0, 0].cpu().numpy()
            mask_val = valid_mask[0, 0].cpu().numpy().astype(bool)

            # Simple min-max for GT (ignoring invalid usually helps visualization)
            if mask_val.any():
                vmin, vmax = gt_val[mask_val].min(), gt_val[mask_val].max()
            else:
                vmin, vmax = 0, 1

            gt_norm = (gt_val - vmin) / (vmax - vmin + 1e-8)
            gt_norm = np.clip(gt_norm, 0, 1)
            gt_vis = (gt_norm * 255).astype(np.uint8)
            gt_vis = cv2.applyColorMap(gt_vis, cv2.COLORMAP_INFERNO)

            # Set invalid pixels to black in GT visualization
            gt_vis[~mask_val] = 0

            # Concatenate: Image | Pred | GT and save to vis folder
            combined_vis = np.hstack([img_bgr, pred_vis, gt_vis])
            cv2.imwrite(os.path.join(vis_dir, f"{save_name}_vis.png"), combined_vis)

            # Save all in one npz file in npz folder
            np.savez_compressed(
                os.path.join(npz_dir, f"{save_name}.npz"),
                pred=depth_pred.astype(np.float32),
                gt=depth_gt[0, 0].cpu().numpy().astype(np.float32),
                mask=valid_mask[0, 0].cpu().numpy().astype(bool),
            )

            saved_count += 1

    # --- Compute and Report Metrics ---
    print(f"\nInference complete on {saved_count} samples.")
    print("Computing metrics with Scale & Shift alignment...")
    metrics = metric_calc.compute()

    # Save metrics to text file
    metric_file = os.path.join(args.outdir, "metrics.txt")
    with open(metric_file, "w") as f:
        f.write(f"Evaluated on {saved_count} samples\n")
        f.write(f"Encoder: {args.encoder}\n")
        f.write("--- Metrics ---\n")

        # Sort keys for cleaner output
        sorted_keys = sorted(metrics.keys())
        for k in sorted_keys:
            val = metrics[k].item()
            line = f"{k}: {val:.6f}"
            print(line)
            f.write(line + "\n")

    print(f"Metrics saved to {metric_file}")
    print(f"Results visualization saved to {args.outdir}")


if __name__ == "__main__":
    """
    python src/stage2/depth_estimation/baseline/infer_da2_us3d.py  \
    --encoder vitl \
    --outdir runs/stage2_depth_estimation_da2_baseline/us3d_oma_test \
    --data-place oma --split test

    python src/stage2/depth_estimation/baseline/infer_da2_us3d.py \
    --encoder vitl \
    --outdir runs/stage2_depth_estimation_da2_baseline/us3d_jax_test \
    --data-place jax --split test
    """
    main()
