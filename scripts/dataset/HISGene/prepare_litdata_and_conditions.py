from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Literal


def _derive_seed(seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def collect_hsi_paths(dataset_dir: Path, recursive: bool = False) -> list[Path]:
    patterns = ("*.tif", "*.tiff", "*.TIF", "*.TIFF")
    paths: list[Path] = []
    for pattern in patterns:
        iterator = dataset_dir.rglob(pattern) if recursive else dataset_dir.glob(pattern)
        paths.extend(p for p in iterator if p.is_file())
    return sorted(set(paths), key=lambda p: p.as_posix())


def collect_grouped_hsi_paths(input_root: Path, recursive: bool = False) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    subdirs = sorted([p for p in input_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not subdirs:
        raise RuntimeError(f"No subdirectories found under {input_root}")

    for dataset_dir in subdirs:
        paths = collect_hsi_paths(dataset_dir, recursive=recursive)
        if len(paths) == 0:
            print(f"[skip] {dataset_dir}: no .tif/.tiff files found")
            continue
        grouped[dataset_dir.name] = paths
    return grouped


def split_paths(
    paths: list[Path],
    test_ratio: float,
    seed: int,
    split_key: str,
) -> tuple[list[Path], list[Path]]:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError(f"`test_ratio` must be in (0, 1), got {test_ratio}")

    if len(paths) <= 1:
        return list(paths), []

    ordered = sorted(paths, key=lambda p: p.as_posix())
    rng = random.Random(_derive_seed(seed, split_key))
    rng.shuffle(ordered)

    test_count = max(1, int(round(len(ordered) * test_ratio)))
    test_count = min(test_count, len(ordered) - 1)
    test_paths = ordered[:test_count]
    train_paths = ordered[test_count:]
    return train_paths, test_paths


def stratified_split_groups(
    group_to_paths: dict[str, list[Path]],
    test_ratio: float,
    seed: int,
) -> tuple[list[Path], list[Path], dict[str, dict[str, int]]]:
    train_all: list[Path] = []
    test_all: list[Path] = []
    stats: dict[str, dict[str, int]] = {}
    for group_name in sorted(group_to_paths.keys()):
        train_paths, test_paths = split_paths(
            group_to_paths[group_name],
            test_ratio=test_ratio,
            seed=seed,
            split_key=group_name,
        )
        train_all.extend(train_paths)
        test_all.extend(test_paths)
        stats[group_name] = {
            "total": len(group_to_paths[group_name]),
            "train": len(train_paths),
            "test": len(test_paths),
        }
    return train_all, test_all, stats


def build_records(paths: list[Path], dataset_root: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    for path in paths:
        rel = path.relative_to(dataset_root)
        key = rel.with_suffix("").as_posix().replace("/", "__")
        records.append((str(path), key))
    return records


def optimize_record(record: tuple[str, str]) -> dict[str, str | bytes]:
    import numpy as np
    import tifffile

    from src.data.codecs import tiff_codec_io

    path_str, key = record
    img = np.asarray(tifffile.imread(path_str))
    encoded = tiff_codec_io(img, compression="jpeg2000", compression_args={"reversible": True})
    return {"__key__": key, "img": encoded}


def optimize_to_litdata(
    records: list[tuple[str, str]],
    output_dir: Path,
    num_workers: int,
    chunk_bytes: str,
    mode: Literal["append", "overwrite"],
    start_method: str,
) -> None:
    from litdata import optimize

    output_dir.mkdir(parents=True, exist_ok=True)
    optimize(
        optimize_record,
        records,
        str(output_dir),
        num_workers=num_workers,
        chunk_bytes=chunk_bytes,
        mode=mode,
        start_method=start_method,
    )


def repair_litdata_index_img_format(
    index_path: Path,
    from_format: str = "bytes",
    to_format: str = "tifffile",
) -> bool:
    if not index_path.exists():
        return False

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    config = payload.get("config", {})
    data_format = config.get("data_format")
    data_spec = config.get("data_spec")
    if not isinstance(data_format, list):
        return False

    img_index: int | None = None
    if isinstance(data_spec, str):
        try:
            spec_obj = json.loads(data_spec)
            if (
                isinstance(spec_obj, list)
                and len(spec_obj) > 1
                and isinstance(spec_obj[1], dict)
                and isinstance(spec_obj[1].get("context"), str)
            ):
                keys = json.loads(spec_obj[1]["context"])
                if isinstance(keys, list) and "img" in keys:
                    img_index = keys.index("img")
        except Exception:
            img_index = None

    changed = False
    if img_index is not None and img_index < len(data_format):
        if data_format[img_index] == from_format:
            data_format[img_index] = to_format
            changed = True
    else:
        bytes_indices = [i for i, fmt in enumerate(data_format) if fmt == from_format]
        if len(bytes_indices) == 1:
            data_format[bytes_indices[0]] = to_format
            changed = True

    if changed:
        index_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return changed


def repair_litdata_index_caption_format(index_path: Path) -> bool:
    if not index_path.exists():
        return False

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    config = payload.get("config", {})
    data_format = config.get("data_format")
    data_spec = config.get("data_spec")
    if not isinstance(data_format, list) or not isinstance(data_spec, str):
        return False

    changed = False
    try:
        spec_obj = json.loads(data_spec)
        if (
            isinstance(spec_obj, list)
            and len(spec_obj) > 1
            and isinstance(spec_obj[1], dict)
            and isinstance(spec_obj[1].get("context"), str)
        ):
            keys = json.loads(spec_obj[1]["context"])
            if isinstance(keys, list):
                for old_key in ("caption.json", "caption.txt"):
                    if old_key in keys:
                        cap_idx = int(keys.index(old_key))
                        keys[cap_idx] = "caption"
                        spec_obj[1]["context"] = json.dumps(keys, ensure_ascii=False)
                        config["data_spec"] = json.dumps(spec_obj, ensure_ascii=False)
                        changed = True

                        if cap_idx < len(data_format):
                            tgt_format = "json" if old_key.endswith(".json") else "str"
                            if data_format[cap_idx] != tgt_format:
                                data_format[cap_idx] = tgt_format
                                changed = True
                        break
    except Exception:
        return False

    if changed:
        index_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return changed


def configure_caption_backend_env(args: argparse.Namespace) -> None:
    os.environ["CAPTION_BACKEND"] = str(args.caption_backend).strip().lower()
    print(f"[caption] backend: {os.environ['CAPTION_BACKEND']}")
    if args.caption_ckpt:
        os.environ["QWEN25VL_CKPT"] = str(args.caption_ckpt)
        print(f"[caption] using local qwen ckpt: {args.caption_ckpt}")
    if args.caption_local_files_only:
        os.environ["QWEN25VL_LOCAL_ONLY"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        print("[caption] enabled local-files-only/offline mode")
    if args.hf_cache_dir:
        cache_dir = str(args.hf_cache_dir)
        os.environ["HF_HOME"] = cache_dir
        os.environ["TRANSFORMERS_CACHE"] = cache_dir
        os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
        print(f"[caption] set HF cache dir: {cache_dir}")


def generate_conditions_litdata(
    image_litdata_dir: Path,
    condition_litdata_dir: Path,
    conditions: list[str],
    rgb_channels: list[int],
    device: str,
    condition_save_format: str,
    caption_save_format: str,
    use_linstretch: bool,
    resume_from: int = 0,
) -> int:
    from scripts.data_prepare.generative_condition_prepare import webdataset_conditions_prepare
    from src.data.litdata_hyperloader import ImageStreamingDataset

    condition_litdata_dir.mkdir(parents=True, exist_ok=True)
    ds = ImageStreamingDataset(
        input_dir=str(image_litdata_dir),
        to_neg_1_1=False,
        force_to_rgb=True,
    )
    total = webdataset_conditions_prepare(
        datasets=[ds],
        base_dir=str(condition_litdata_dir),
        tar_name="conditions",
        nums=None,
        conditions=conditions,
        rgb_channels=rgb_channels,
        device=device,
        to_pil=True,
        save_original_rgb=False,
        condition_save_format=condition_save_format,  # type: ignore[arg-type]
        caption_save_format=caption_save_format,  # type: ignore[arg-type]
        resume_from=resume_from,
        relative_data_dir="conditions",
        save_attn_mask=False,
        use_linstretch=use_linstretch,
        dataset_type="litdata",
    )
    return int(total)


def run_caption_only(args: argparse.Namespace) -> None:
    root = Path(args.input_root)
    image_base_dir = root / args.image_litdata_name
    caption_base_dir = root / args.caption_litdata_name
    if not image_base_dir.exists():
        raise FileNotFoundError(f"Image litdata dir does not exist: {image_base_dir}")

    rgb_channels = parse_rgb_channels(args.rgb_channels)
    configure_caption_backend_env(args)
    split_found = False
    for split in ("train", "test"):
        image_out = image_base_dir / split
        if not image_out.exists():
            print(f"[skip] image split not found: {image_out}")
            continue
        split_found = True

        caption_out = caption_base_dir / split
        print(f"[caption] generating captions for {split} -> {caption_out}")
        count = generate_conditions_litdata(
            image_litdata_dir=image_out,
            condition_litdata_dir=caption_out,
            conditions=["caption"],
            rgb_channels=rgb_channels,
            device=args.device,
            condition_save_format=args.condition_save_format,
            caption_save_format=args.caption_save_format,
            use_linstretch=args.use_linstretch,
            resume_from=0,
        )
        idx = caption_out / "index.json"
        if repair_litdata_index_caption_format(idx):
            print(f"[fix] patched caption index format: {idx}")
        print(f"[caption] done {split}: {count} samples")

    if not split_found:
        raise RuntimeError(f"No train/test split found under {image_base_dir}")


def parse_conditions(raw: str) -> list[str]:
    items = [s.strip() for s in raw.split(",") if s.strip()]
    if not items:
        raise ValueError("`conditions` cannot be empty.")
    return items


def parse_rgb_channels(raw: str) -> list[int]:
    channels = [int(s.strip()) for s in raw.split(",") if s.strip()]
    if len(channels) != 3:
        raise ValueError("`rgb_channels` must be three comma-separated integers, like '0,1,2'.")
    return channels


def parse_mode(raw: str) -> Literal["append", "overwrite"]:
    if raw == "append":
        return "append"
    if raw == "overwrite":
        return "overwrite"
    raise ValueError(f"Unsupported mode: {raw}")


def write_split_manifest(
    output_path: Path,
    subdir_stats: dict[str, dict[str, int]],
    seed: int,
    test_ratio: float,
    train_count: int,
    test_count: int,
) -> None:
    manifest = {
        "seed": seed,
        "test_ratio": test_ratio,
        "train_count": train_count,
        "test_count": test_count,
        "total_count": train_count + test_count,
        "groups": subdir_stats,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    if args.captions_only:
        run_caption_only(args)
        return

    root = Path(args.input_root)
    if not root.exists():
        raise FileNotFoundError(f"Input root does not exist: {root}")

    conditions = parse_conditions(args.conditions)
    rgb_channels = parse_rgb_channels(args.rgb_channels)
    mode = parse_mode(args.mode)
    if "caption" in conditions:
        configure_caption_backend_env(args)

    group_to_paths = collect_grouped_hsi_paths(root, recursive=args.recursive)
    if len(group_to_paths) == 0:
        raise RuntimeError(f"No valid tif/tiff files found under subdirectories of {root}")

    train_paths, test_paths, subdir_stats = stratified_split_groups(
        group_to_paths=group_to_paths,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    for group_name in sorted(subdir_stats.keys()):
        stat = subdir_stats[group_name]
        print(f"[split] {group_name}: total={stat['total']} train={stat['train']} test={stat['test']}")

    write_split_manifest(
        output_path=root / "split_manifest.json",
        subdir_stats=subdir_stats,
        seed=args.seed,
        test_ratio=args.test_ratio,
        train_count=len(train_paths),
        test_count=len(test_paths),
    )
    print(f"[split] all: total={len(train_paths) + len(test_paths)} train={len(train_paths)} test={len(test_paths)}")

    image_base_dir = root / args.image_litdata_name
    condition_base_dir = root / args.condition_litdata_name
    for split, paths in (("train", train_paths), ("test", test_paths)):
        if len(paths) == 0:
            print(f"[skip] {split}: empty split")
            continue

        image_out = image_base_dir / split
        condition_out = condition_base_dir / split
        records = build_records(paths, dataset_root=root)

        print(f"[litdata] building images for {split} -> {image_out}")
        optimize_to_litdata(
            records=records,
            output_dir=image_out,
            num_workers=args.num_workers,
            chunk_bytes=args.chunk_bytes,
            mode=mode,
            start_method=args.start_method,
        )
        index_path = image_out / "index.json"
        if repair_litdata_index_img_format(index_path):
            print(f"[fix] patched index serializer for img: {index_path}")

        if args.skip_conditions:
            continue

        print(f"[cond] generating conditions for {split} -> {condition_out}")
        count = generate_conditions_litdata(
            image_litdata_dir=image_out,
            condition_litdata_dir=condition_out,
            conditions=conditions,
            rgb_channels=rgb_channels,
            device=args.device,
            condition_save_format=args.condition_save_format,
            caption_save_format=args.caption_save_format,
            use_linstretch=args.use_linstretch,
            resume_from=0,
        )
        print(f"[cond] done {split}: {count} samples")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare HISGene train/test LitData and condition LitData datasets.")
    parser.add_argument("--input-root", type=str, default="data2/HSIGene_dataset")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--image-litdata-name", type=str, default="LitData_hyper_images")
    parser.add_argument("--condition-litdata-name", type=str, default="LitData_conditions")
    parser.add_argument("--caption-litdata-name", type=str, default="LitData_image_captions")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--chunk-bytes", type=str, default="512Mb")
    parser.add_argument("--mode", type=str, choices=["append", "overwrite"], default="overwrite")
    parser.add_argument("--start-method", type=str, choices=["spawn", "fork"], default="spawn")
    parser.add_argument("--skip-conditions", action="store_true")
    parser.add_argument("--captions-only", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--conditions", type=str, default="hed,segmentation,sketch,mlsd")
    parser.add_argument("--rgb-channels", type=str, default="20,12,5")
    parser.add_argument("--condition-save-format", type=str, choices=["png", "jpg"], default="jpg")
    parser.add_argument("--caption-save-format", type=str, choices=["txt", "json"], default="json")
    parser.add_argument("--caption-backend", type=str, choices=["qwen25vl", "internvl35"], default="internvl35")
    parser.add_argument("--caption-ckpt", type=str, default="")
    parser.add_argument("--hf-cache-dir", type=str, default="")
    parser.add_argument("--caption-local-files-only", action="store_true")
    parser.add_argument("--use-linstretch", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    """
    RGB bands: [20, 12, 5]
    python scripts/dataset/HISGene/prepare_litdata_and_conditions.py \
    --input-root data2/HSIGene_dataset \
    --device cuda:0 \
    --num-workers 0 \
    --chunk-bytes 512Mb

    python scripts/dataset/HISGene/prepare_litdata_and_conditions.py \
    --input-root data2/HSIGene_dataset \
    --captions-only \
    --caption-backend internvl35 \
    --caption-ckpt /Data/ZiHanCao/checkpoints/models--OpenGVLab--InternVL3_5-8B \
    --hf-cache-dir /Data/ZiHanCao/checkpoints \
    --caption-local-files-only \
    --device cuda:0 \
    --rgb-channels 20,12,5 \
    --caption-save-format json
    """
    main()
