"""
Large image segmentation prediction script using Tokenizer Hybrid UNet.
Uses sliding window inference to handle arbitrary-sized images.
"""

import argparse
from pathlib import Path

import hydra
import numpy as np
import torch
import tifffile
from omegaconf import DictConfig, OmegaConf

from src.data.window_slider import model_predict_patcher
from src.stage2.segmentation.metrics.basic import HyperSegmentationScore
from src.stage2.segmentation.metrics.metric_from_dpa import DPASegmentationScore
from src.utilities.train_utils.visualization import visualize_segmentation_map


def parse_args():
    parser = argparse.ArgumentParser(description="Large image segmentation prediction")
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/Downstreams/5Billion-ChinaCity-Segmentation/TrainSet_For_Baseline/Image_16bit_RGBNir/GF2_PMS1__L1A0000647768-MSS1.tiff",
        help="Path to input image or directory",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="tmp/",
        help="Path to save output results",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="scripts/configs/segmentation/model/tokenizer_hybrid_decoder.yaml",
        help="Path to model config yaml",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="scripts/configs/segmentation/dataset/five_billion.yaml",
        help="Path to dataset config yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="runs/stage2_classification/2025-12-31_23-42-37_FiveBillionChina_tokenizer_hybrid_adaptor/ema/seg_ema_model/model.safetensors",
        help="Path to model checkpoint (safetensors format)",
    )
    parser.add_argument("--patch_size", type=int, default=512, help="Patch size for sliding window")
    parser.add_argument("--stride", type=int, default=256, help="Stride for sliding window")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"], help="Data type")
    parser.add_argument(
        "--norm_const",
        type=float,
        default=1024.0,
        help="Normalization constant: 255 for 8-bit, 1024 for 16-bit images",
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        default="data/Downstreams/5Billion-ChinaCity-Segmentation/TrainSet_For_Baseline/Annotation_Index/GF2_PMS1__L1A0000647768-MSS1_24label.png",
        help="Path to ground truth label image (optional). If provided, metrics will be computed.",
    )
    parser.add_argument(
        "--gt_ignore_index",
        type=int,
        default=0,
        help="Ignore label value in GT. Use -1 to disable ignore handling.",
    )
    return parser.parse_args()


def load_config(model_config_path: str, dataset_config_path: str) -> DictConfig:
    """Load and merge model and dataset configurations."""
    model_cfg = OmegaConf.load(model_config_path)
    dataset_cfg = OmegaConf.load(dataset_config_path)

    # Create a combined config with 'dataset' key for proper resolution
    combined = OmegaConf.create({"segment_model": model_cfg, "dataset": dataset_cfg})
    OmegaConf.resolve(combined)

    return combined


def load_model(cfg: DictConfig, checkpoint_path: str, device: str):
    """Instantiate model and load weights from checkpoint (.pth or .safetensors)."""
    model = hydra.utils.instantiate(cfg.segment_model)

    # Load checkpoint - supports both .pth and .safetensors
    checkpoint_path_obj = Path(checkpoint_path)
    if checkpoint_path_obj.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(checkpoint_path)
    else:
        # Load .pth or .pt files
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        # Handle different checkpoint formats
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

    # Handle potential 'module.' prefix from DDP training
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    incompatible = model.load_state_dict(new_state_dict, strict=False)
    print(f"Model loaded. Incompatible keys: {incompatible}")

    model = model.to(device)
    model.eval()
    return model


def preprocess_image(
    img_path: str,
    to_neg_1_1: bool = True,
    norm_const: float = 255.0,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Load and preprocess image for inference.

    Args:
        img_path: Path to input image.
        to_neg_1_1: If True, normalize to [-1, 1] range.
        norm_const: Normalization constant (255 for 8-bit, 1024 for 16-bit).
        device: Device to use.
        dtype: Data type.
    """
    img = tifffile.imread(img_path)
    x = np.array(img)

    # HWC -> NCHW
    x = torch.as_tensor(x).permute(2, 0, 1).unsqueeze(0).to(device)
    x = x.float() / norm_const
    x = x.clamp(0.0, 1.0)

    if to_neg_1_1:
        x = x * 2.0 - 1.0

    x = x.to(dtype)
    return x


def run_inference(
    model: torch.nn.Module,
    img: torch.Tensor,
    patch_size: int,
    stride: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Run sliding window inference on the image."""

    def forward_closure(batch: dict) -> dict:
        x = batch["img"]
        with torch.autocast("cuda", dtype):
            with torch.no_grad():
                output = model(x)
        return {"pred_logits": output}

    outputs = model_predict_patcher(
        forward_closure,
        {"img": img},
        patch_keys=["img"],
        merge_keys=["pred_logits"],
        patch_size=patch_size,
        stride=stride,
        use_tqdm=True,
    )
    return outputs["pred_logits"]


def load_ground_truth(gt_path: str, device: str = "cuda") -> torch.Tensor:
    """Load ground truth label image.

    Args:
        gt_path: Path to ground truth label image.
        device: Device to use.

    Returns:
        Ground truth labels as tensor (1, H, W).
    """
    from PIL import Image

    gt = np.array(Image.open(gt_path).convert("L"))
    gt_tensor = torch.as_tensor(gt).unsqueeze(0).to(device)
    return gt_tensor.long()


def _normalize_gt_for_metrics(
    gt: torch.Tensor,
    *,
    num_classes: int,
    gt_ignore_index: int | None,
) -> tuple[torch.Tensor, int | None]:
    gt = gt.long()

    if gt_ignore_index is None:
        return gt, None

    if gt_ignore_index == 0:
        # FiveBillion: 0=未标注(忽略)，其余标签为 1-based 类别编号（即使某张图没出现所有类别也正常）
        ignore_index = 255
        if num_classes >= ignore_index:
            raise ValueError(f"{num_classes=} must be < {ignore_index=} to remap ignore safely.")

        gt_mapped = torch.where(gt == 0, torch.full_like(gt, ignore_index), gt - 1)
        invalid = (gt_mapped != ignore_index) & ((gt_mapped < 0) | (gt_mapped >= num_classes))
        if invalid.any():
            gt_min = int(gt.min().item())
            gt_max = int(gt.max().item())
            mapped_min = (
                int(gt_mapped[gt_mapped != ignore_index].min().item()) if (gt_mapped != ignore_index).any() else -1
            )
            mapped_max = (
                int(gt_mapped[gt_mapped != ignore_index].max().item()) if (gt_mapped != ignore_index).any() else -1
            )
            raise ValueError(
                "GT 标签映射后出现越界值；请检查 num_classes/gt_ignore_index/标签编码是否一致。"
                f" raw_range=[{gt_min},{gt_max}] mapped_range=[{mapped_min},{mapped_max}] {num_classes=}"
            )

        return gt_mapped, ignore_index

    return gt, gt_ignore_index


def _normalize_pred_gt_for_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    *,
    num_classes: int,
    gt_ignore_index: int | None,
) -> tuple[torch.Tensor, torch.Tensor, int | None]:
    pred = pred.long()
    gt, ignore_index = _normalize_gt_for_metrics(gt, num_classes=num_classes, gt_ignore_index=gt_ignore_index)
    return pred, gt, ignore_index


def _visualize_segmentation_map_with_ignore(
    label_map: torch.Tensor,
    *,
    n_class: int,
    colors: str,
    ignore_index: int | None,
):
    from PIL import Image

    label_map = label_map.detach().cpu().long()
    if label_map.ndim == 3:
        label_map_2d = label_map[0]
    else:
        label_map_2d = label_map

    ignore_mask = None
    if ignore_index is not None:
        ignore_mask_tensor = label_map_2d == ignore_index
        ignore_mask = ignore_mask_tensor.cpu().numpy()
        label_map_2d = torch.where(ignore_mask_tensor, torch.zeros_like(label_map_2d), label_map_2d)

    vis = visualize_segmentation_map(
        label_map_2d.unsqueeze(0),
        n_class=n_class,
        colors=colors,
        to_pil=True,
        to_rgba=False,
        bg_black=False,
    )

    if ignore_mask is None:
        return vis.convert("RGB")

    rgb = np.array(vis)
    rgb[ignore_mask, :3] = 0
    return Image.fromarray(rgb, mode="RGB")


def compute_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    num_classes: int,
    *,
    gt_ignore_index: int | None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Compute metrics using both basic and DPA implementations.

    Args:
        pred: Predicted labels (1, H, W).
        gt: Ground truth labels (1, H, W).
        num_classes: Number of classes.

    Returns:
        Dictionary containing metrics from both implementations.
    """
    if pred.device != gt.device:
        gt = gt.to(device=pred.device)

    pred, gt, ignore_index = _normalize_pred_gt_for_metrics(
        pred, gt, num_classes=num_classes, gt_ignore_index=gt_ignore_index
    )

    # Initialize metrics calculators
    basic_metrics = HyperSegmentationScore(
        n_classes=num_classes,
        ignore_index=ignore_index,
        task="multiclass",
        reduction="micro",
        per_class=False,
        cal_metrics=["accuracy", "precision", "recall", "kappa", "f1", "dice", "miou"],
    ).to(pred.device)

    dpa_metrics = DPASegmentationScore(
        n_classes=num_classes,
        ignore_index=ignore_index,
        per_class=False,
    ).to(pred.device)

    # Compute basic metrics
    basic_results = basic_metrics(pred, gt)

    # Compute DPA metrics
    dpa_metrics.update(pred, gt)
    dpa_results = dpa_metrics.compute()

    return {"basic": basic_results, "dpa": dpa_results}


def print_metrics_comparison(metrics: dict[str, dict[str, torch.Tensor]]) -> None:
    """Print metrics comparison between basic and DPA implementations.

    Args:
        metrics: Dictionary containing metrics from both implementations.
    """
    print("\n" + "=" * 80)
    print("METRICS COMPARISON: Basic vs DPA")
    print("=" * 80)

    basic = metrics["basic"]
    dpa = metrics["dpa"]

    # Define metric mappings for comparison
    metric_mappings = [
        ("Accuracy", "accuracy", "oa"),
        ("F1 Score", "f1_score", "mf1"),
        ("Mean IoU", "mean_iou", "miou"),
        ("Precision", "precision", "ua"),
        ("Recall", "recall", "pa"),
        ("Kappa", "kappa", "kappa"),
    ]

    for name, basic_key, dpa_key in metric_mappings:
        basic_val = basic.get(basic_key)
        dpa_val = dpa.get(dpa_key)

        if basic_val is not None and dpa_val is not None:
            # Convert tensors to floats for printing
            if isinstance(basic_val, torch.Tensor):
                basic_val = basic_val.item()
            if isinstance(dpa_val, torch.Tensor):
                dpa_val = dpa_val.item()

            diff = abs(basic_val - dpa_val)
            print(f"{name:15s}: Basic={basic_val:.4f}, DPA={dpa_val:.4f}, Diff={diff:.4f}")

    # Note: Dice is only in basic
    if "dice" in basic:
        dice_val = basic["dice"]
        if isinstance(dice_val, torch.Tensor):
            dice_val = dice_val.item()
        print(f"{'Dice':15s}: Basic={dice_val:.4f}, DPA=N/A")

    print("=" * 80 + "\n")


def main():
    args = parse_args()

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    print(f"Loading config from {args.model_config} and {args.dataset_config}")
    cfg = load_config(args.model_config, args.dataset_config)
    to_neg_1_1 = cfg.dataset.get("to_neg_1_1", True)
    num_classes = cfg.dataset.get("num_classes", 24)

    print(f"Loading model from {args.checkpoint}")
    model = load_model(cfg, args.checkpoint, args.device)

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load ground truth if provided
    gt_tensor = None
    gt_ignore_index: int | None = None
    if args.gt_path is not None:
        print(f"Loading ground truth from {args.gt_path}")
        gt_tensor = load_ground_truth(args.gt_path, device=args.device)
        print(f"  Ground truth shape: {gt_tensor.shape}")
        gt_ignore_index = None if args.gt_ignore_index == -1 else args.gt_ignore_index

    input_path = Path(args.input_path)
    if input_path.is_dir():
        image_paths = list(input_path.glob("*.tif")) + list(input_path.glob("*.png"))
    else:
        image_paths = [input_path]

    print(f"Found {len(image_paths)} images to process")

    for img_path in image_paths:
        print(f"Processing: {img_path.name}")

        img = preprocess_image(
            str(img_path),
            to_neg_1_1=to_neg_1_1,
            norm_const=args.norm_const,
            device=args.device,
            dtype=dtype,
        )
        print(f"  Image shape: {img.shape}")

        pred_logits = run_inference(model, img, args.patch_size, args.stride, dtype)
        print(f"  Prediction shape: {pred_logits.shape}")

        pred_class = pred_logits.argmax(dim=1)

        # Compute metrics if ground truth is provided
        if gt_tensor is not None:
            # Check if shapes match
            pred_cpu = pred_class.cpu()
            gt_cpu = gt_tensor.cpu()

            if pred_cpu.shape == gt_cpu.shape:
                gt_for_vis, gt_ignore_index_for_vis = _normalize_gt_for_metrics(
                    gt_cpu,
                    num_classes=num_classes,
                    gt_ignore_index=gt_ignore_index,
                )
                gt_vis = _visualize_segmentation_map_with_ignore(
                    gt_for_vis,
                    n_class=num_classes,
                    colors="gid",
                    ignore_index=gt_ignore_index_for_vis,
                )
                gt_out_name = img_path.stem + "_gt.png"
                gt_vis.save(output_path / gt_out_name)
                print(f"  Saved: {gt_out_name}")

                print("  Computing metrics...")
                metrics = compute_metrics(pred_cpu, gt_cpu, num_classes, gt_ignore_index=gt_ignore_index)
                print_metrics_comparison(metrics)
            else:
                print(f"  Warning: Shape mismatch between prediction {pred_cpu.shape} and GT {gt_cpu.shape}")
                print("  Skipping metrics computation.")

        vis_img = visualize_segmentation_map(
            pred_class.cpu(),
            n_class=num_classes,
            colors="gid",
            to_pil=True,
            bg_black=False,
        )

        out_name = img_path.stem + "_pred.png"
        vis_img.save(output_path / out_name)
        print(f"  Saved: {out_name}")

    print("Done!")


if __name__ == "__main__":
    main()
