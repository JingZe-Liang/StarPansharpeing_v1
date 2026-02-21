#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

import litdata
import src.data.codecs  # noqa: F401


def safe_get(ds: Any, idx: int) -> tuple[bool, Any | None, str | None]:
    try:
        return True, ds[idx], None
    except Exception as e:
        return False, None, f"{type(e).__name__}: {e}"


def get_sample_key(sample: Any) -> str | None:
    if not isinstance(sample, dict):
        return None
    key = sample.get("__key__")
    if isinstance(key, list) and key:
        return str(key[0])
    if key is None:
        return None
    return str(key)


def is_array_like(x: Any) -> bool:
    return torch.is_tensor(x) or isinstance(x, np.ndarray)


def validate_img_sample(sample: Any) -> str | None:
    if not isinstance(sample, dict):
        return f"img sample must be dict, got {type(sample)}"
    if "img" not in sample:
        return "img sample missing key: img"
    img = sample["img"]
    if img is None:
        return "img is None"
    if isinstance(img, bytes):
        return "img is undecoded bytes"
    if not is_array_like(img):
        return f"img type unsupported: {type(img)}"
    return None


def validate_cond_sample(sample: Any) -> str | None:
    expected = ["hed", "segmentation", "sketch", "mlsd"]
    if not isinstance(sample, dict):
        return f"condition sample must be dict, got {type(sample)}"
    for key in expected:
        if key not in sample:
            return f"condition sample missing key: {key}"
        val = sample[key]
        if val is None:
            return f"condition {key} is None"
        if isinstance(val, bytes):
            return f"condition {key} is undecoded bytes"
        if not is_array_like(val):
            return f"condition {key} type unsupported: {type(val)}"
    return None


def validate_caption_sample(sample: Any) -> str | None:
    if not isinstance(sample, dict):
        return f"caption sample must be dict, got {type(sample)}"
    if "caption" not in sample:
        return "caption sample missing key: caption"
    caption = sample["caption"]
    if isinstance(caption, dict):
        caption = caption.get("caption")
    if not isinstance(caption, str):
        return f"caption is not str, got {type(caption)}"
    return None


def scan_integrity(
    img_dir: str,
    cond_dir: str,
    caption_dir: str,
    max_samples: int = -1,
) -> dict[str, Any]:
    img_ds = litdata.StreamingDataset(input_dir=img_dir, shuffle=False)
    cond_ds = litdata.StreamingDataset(input_dir=cond_dir, shuffle=False)
    caption_ds = litdata.StreamingDataset(input_dir=caption_dir, shuffle=False)

    len_img = len(img_ds)
    len_cond = len(cond_ds)
    len_caption = len(caption_ds)
    total = min(len_img, len_cond, len_caption)
    if max_samples > 0:
        total = min(total, max_samples)

    failures: list[dict[str, Any]] = []
    ok_count = 0

    for idx in tqdm(range(total), desc="Scanning"):
        record: dict[str, Any] = {"idx": idx, "ok": True}

        ok_img, sample_img, err_img = safe_get(img_ds, idx)
        ok_cond, sample_cond, err_cond = safe_get(cond_ds, idx)
        ok_caption, sample_caption, err_caption = safe_get(caption_ds, idx)

        if not ok_img:
            record["ok"] = False
            record["img_error"] = err_img
        if not ok_cond:
            record["ok"] = False
            record["cond_error"] = err_cond
        if not ok_caption:
            record["ok"] = False
            record["caption_error"] = err_caption

        if ok_img:
            err = validate_img_sample(sample_img)
            if err is not None:
                record["ok"] = False
                record["img_error"] = err
        if ok_cond:
            err = validate_cond_sample(sample_cond)
            if err is not None:
                record["ok"] = False
                record["cond_error"] = err
        if ok_caption:
            err = validate_caption_sample(sample_caption)
            if err is not None:
                record["ok"] = False
                record["caption_error"] = err

        key_img = get_sample_key(sample_img) if ok_img else None
        key_cond = get_sample_key(sample_cond) if ok_cond else None
        key_caption = get_sample_key(sample_caption) if ok_caption else None
        keys = [k for k in [key_img, key_cond, key_caption] if k is not None]
        if len(set(keys)) > 1:
            record["ok"] = False
            record["key_error"] = f"key mismatch img={key_img}, cond={key_cond}, caption={key_caption}"

        if record["ok"]:
            ok_count += 1
        else:
            failures.append(record)

    return {
        "img_dir": img_dir,
        "cond_dir": cond_dir,
        "caption_dir": caption_dir,
        "lengths": {"img": len_img, "cond": len_cond, "caption": len_caption},
        "scanned": total,
        "ok": ok_count,
        "failed": len(failures),
        "failures": failures,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan litdata image/condition/caption datasets and report bad samples."
    )
    parser.add_argument(
        "--img-dir", type=str, required=True, help="LitData image directory (e.g. .../LitData_hyper_images/train)"
    )
    parser.add_argument(
        "--cond-dir", type=str, required=True, help="LitData condition directory (e.g. .../LitData_conditions/train)"
    )
    parser.add_argument(
        "--caption-dir",
        type=str,
        required=True,
        help="LitData caption directory (e.g. .../LitData_image_captions/train)",
    )
    parser.add_argument("--max-samples", type=int, default=-1, help="Max samples to scan, -1 means all.")
    parser.add_argument("--output-json", type=str, default="", help="Optional output json path.")
    parser.add_argument("--show-first-n", type=int, default=20, help="Print first N failure records.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = scan_integrity(
        img_dir=args.img_dir,
        cond_dir=args.cond_dir,
        caption_dir=args.caption_dir,
        max_samples=args.max_samples,
    )

    print("[scan] done")
    print(f"[scan] lengths: {result['lengths']}")
    print(f"[scan] scanned={result['scanned']} ok={result['ok']} failed={result['failed']}")

    show_n = max(0, int(args.show_first_n))
    if result["failed"] > 0 and show_n > 0:
        print(f"[scan] first {min(show_n, result['failed'])} failures:")
        for row in result["failures"][:show_n]:
            print(json.dumps(row, ensure_ascii=False))

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[scan] saved report to: {out_path}")


if __name__ == "__main__":
    main()
