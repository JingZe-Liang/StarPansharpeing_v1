from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image


def start_points(size: int, split_size: int, overlap: float = 0.0) -> list[int]:
    if size <= split_size:
        return [0]
    points = [0]
    stride = max(1, int(split_size * (1 - overlap)))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            points.append(size - split_size)
            break
        points.append(pt)
        counter += 1
    return points


def list_sites(input_root: Path) -> list[Path]:
    return sorted([p for p in input_root.iterdir() if p.is_dir()])


def load_label(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"))


def is_positive_patch(label_patch: np.ndarray, min_pos_ratio: float) -> bool:
    nonzero = np.count_nonzero(label_patch)
    if nonzero == 0:
        return False
    ratio = nonzero / label_patch.size
    return ratio >= min_pos_ratio


def count_pos_neg(
    sites: list[Path],
    patch_size: int,
    overlap: float,
    min_pos_ratio: float,
) -> tuple[int, int]:
    pos_count = 0
    neg_count = 0
    for site in sites:
        label = load_label(site / "ref.png")
        height, width = label.shape[:2]
        y_points = start_points(height, patch_size, overlap)
        x_points = start_points(width, patch_size, overlap)
        for y in y_points:
            for x in x_points:
                label_patch = label[y : y + patch_size, x : x + patch_size]
                if is_positive_patch(label_patch, min_pos_ratio):
                    pos_count += 1
                else:
                    neg_count += 1
    return pos_count, neg_count


def clear_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_file():
            item.unlink()


def save_patch(
    out_a: Path,
    out_b: Path,
    out_label: Path,
    index: int,
    img1: np.ndarray,
    img2: np.ndarray,
    label: np.ndarray,
    img_ext: str,
    label_ext: str,
    jpg_quality: int,
) -> None:
    img1_path = out_a / f"{index}.{img_ext}"
    img2_path = out_b / f"{index}.{img_ext}"
    label_path = out_label / f"{index}.{label_ext}"
    Image.fromarray(img1).save(img1_path, quality=jpg_quality)
    Image.fromarray(img2).save(img2_path, quality=jpg_quality)
    Image.fromarray(label).save(label_path)


def sample_and_save(
    sites: list[Path],
    output_root: Path,
    patch_size: int,
    overlap: float,
    min_pos_ratio: float,
    neg_to_pos: float,
    seed: int,
    img_ext: str,
    label_ext: str,
    jpg_quality: int,
    clear_output: bool,
) -> None:
    out_a = output_root / "A"
    out_b = output_root / "B"
    out_label = output_root / "label"
    if clear_output:
        clear_dir(out_a)
        clear_dir(out_b)
        clear_dir(out_label)
    else:
        out_a.mkdir(parents=True, exist_ok=True)
        out_b.mkdir(parents=True, exist_ok=True)
        out_label.mkdir(parents=True, exist_ok=True)

    pos_total, neg_total = count_pos_neg(sites, patch_size, overlap, min_pos_ratio)
    target_neg = min(neg_total, int(round(pos_total * neg_to_pos)))
    rng = random.Random(seed)

    neg_seen = 0
    neg_kept = 0
    saved = 0

    for site in sites:
        img1 = np.asarray(Image.open(site / "im1.png"))
        img2 = np.asarray(Image.open(site / "im2.png"))
        label = load_label(site / "ref.png")
        height, width = label.shape[:2]
        y_points = start_points(height, patch_size, overlap)
        x_points = start_points(width, patch_size, overlap)
        for y in y_points:
            for x in x_points:
                label_patch = label[y : y + patch_size, x : x + patch_size]
                if is_positive_patch(label_patch, min_pos_ratio):
                    save_patch(
                        out_a,
                        out_b,
                        out_label,
                        saved,
                        img1[y : y + patch_size, x : x + patch_size],
                        img2[y : y + patch_size, x : x + patch_size],
                        label_patch,
                        img_ext,
                        label_ext,
                        jpg_quality,
                    )
                    saved += 1
                    continue

                if target_neg == 0:
                    neg_seen += 1
                    continue

                remaining = neg_total - neg_seen
                need = target_neg - neg_kept
                keep_prob = need / remaining
                if rng.random() < keep_prob:
                    save_patch(
                        out_a,
                        out_b,
                        out_label,
                        saved,
                        img1[y : y + patch_size, x : x + patch_size],
                        img2[y : y + patch_size, x : x + patch_size],
                        label_patch,
                        img_ext,
                        label_ext,
                        jpg_quality,
                    )
                    saved += 1
                    neg_kept += 1
                neg_seen += 1

    print(
        "Done. pos_total={}, neg_total={}, target_neg={}, neg_kept={}, saved={}".format(
            pos_total,
            neg_total,
            target_neg,
            neg_kept,
            saved,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clip GVLM patches with a mixed strategy (all positive + sampled negatives)."
    )
    parser.add_argument("--input-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--overlap", type=float, default=0.0)
    parser.add_argument("--min-pos-ratio", type=float, default=0.0)
    parser.add_argument("--neg-to-pos", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img-ext", type=str, default="jpg")
    parser.add_argument("--label-ext", type=str, default="png")
    parser.add_argument("--jpg-quality", type=int, default=95)
    parser.add_argument("--clear-output", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not (0.0 <= args.overlap < 1.0):
        raise ValueError("overlap must be in [0.0, 1.0).")
    if args.patch_size <= 0:
        raise ValueError("patch_size must be positive.")
    if args.min_pos_ratio < 0.0 or args.min_pos_ratio > 1.0:
        raise ValueError("min_pos_ratio must be in [0.0, 1.0].")
    if args.neg_to_pos < 0.0:
        raise ValueError("neg_to_pos must be >= 0.")
    if args.jpg_quality < 1 or args.jpg_quality > 100:
        raise ValueError("jpg_quality must be in [1, 100].")


def main() -> None:
    args = parse_args()
    validate_args(args)
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    sites = list_sites(input_root)
    if not sites:
        raise RuntimeError(f"No site folders found under: {input_root}")

    sample_and_save(
        sites=sites,
        output_root=output_root,
        patch_size=args.patch_size,
        overlap=args.overlap,
        min_pos_ratio=args.min_pos_ratio,
        neg_to_pos=args.neg_to_pos,
        seed=args.seed,
        img_ext=args.img_ext,
        label_ext=args.label_ext,
        jpg_quality=args.jpg_quality,
        clear_output=args.clear_output,
    )


if __name__ == "__main__":
    main()
