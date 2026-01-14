from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from litdata.streaming.cache import Cache
from tqdm import tqdm

from src.data.codecs import tiff_codec_io

SPLIT_NAME_MAP = {1: "train", 2: "val", 3: "test"}


@dataclass(frozen=True)
class SplitRow:
    split_id: int
    roi: str
    sample: str


@dataclass(frozen=True)
class SamplePaths:
    key: str
    season: str
    scene_id: int
    patch_id: int
    s1_path: Path
    s2_path: Path
    s2_cloudy_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert SEN12MS-CR to litdata using Cache.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("data/SEN12MS-CR"),
        help="SEN12MS-CR root directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output root for litdata and meta csv. Defaults to base-dir.",
    )
    parser.add_argument(
        "--splits-csv",
        type=Path,
        default=None,
        help="Splits csv path. Defaults to <base-dir>/splits/splits.csv.",
    )
    parser.add_argument(
        "--chunk-bytes",
        type=str,
        default="512Mb",
        help="Chunk size for litdata cache.",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default=None,
        help="Optional compression name for litdata cache.",
    )
    return parser.parse_args()


def load_splits_csv(csv_path: Path) -> list[SplitRow]:
    rows: list[SplitRow] = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for line_no, row in enumerate(reader, start=2):
            if not row:
                continue
            if "split" not in row or "roi" not in row or "sample" not in row:
                raise ValueError(f"Missing columns in splits csv at line {line_no}: {row}")
            rows.append(
                SplitRow(
                    split_id=int(row["split"]),
                    roi=row["roi"],
                    sample=row["sample"],
                )
            )
    rows.sort(key=lambda item: (item.split_id, item.roi, item.sample))
    return rows


def group_by_split(rows: list[SplitRow]) -> dict[int, list[SplitRow]]:
    grouped: dict[int, list[SplitRow]] = {}
    for row in rows:
        grouped.setdefault(row.split_id, []).append(row)
    return grouped


def build_paths(base_dir: Path, row: SplitRow) -> SamplePaths:
    roi_root = row.roi.split("/")[0]
    season = roi_root.removesuffix("_s1")
    roi_name = Path(row.roi).name
    scene_id = int(roi_name.split("_", 1)[1])
    patch_id = int(row.sample.rsplit("_p", 1)[1].split(".")[0])
    key = f"{season}_{scene_id}_p{patch_id}"

    s1_path = base_dir / season / f"{season}_s1" / f"s1_{scene_id}" / row.sample
    s2_name = row.sample.replace("_s1_", "_s2_")
    s2_path = base_dir / season / f"{season}_s2" / f"s2_{scene_id}" / s2_name
    s2_cloudy_name = s2_name.replace("_s2_", "_s2_cloudy_")
    s2_cloudy_path = base_dir / season / f"{season}_s2_cloudy" / f"s2_cloudy_{scene_id}" / s2_cloudy_name

    return SamplePaths(
        key=key,
        season=season,
        scene_id=scene_id,
        patch_id=patch_id,
        s1_path=s1_path,
        s2_path=s2_path,
        s2_cloudy_path=s2_cloudy_path,
    )


def read_tiff(path: Path) -> np.ndarray:
    with rasterio.open(path) as dataset:
        return dataset.read()


def write_meta_csv(meta_rows: list[dict[str, str | int]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not meta_rows:
        raise ValueError(f"No meta rows to write for {output_path}.")
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(meta_rows[0].keys()))
        writer.writeheader()
        writer.writerows(meta_rows)


def write_split(
    *,
    rows: list[SplitRow],
    base_dir: Path,
    output_root: Path,
    split_id: int,
    split_name: str,
    chunk_bytes: str,
    compression: str | None,
) -> None:
    out_s1 = output_root / "litdata_s1" / split_name
    out_s2 = output_root / "litdata_s2_clean" / split_name
    out_s2_cloudy = output_root / "litdata_s2_cloudy" / split_name
    for out_dir in (out_s1, out_s2, out_s2_cloudy):
        out_dir.mkdir(parents=True, exist_ok=True)

    cache_s1 = Cache(str(out_s1), chunk_bytes=chunk_bytes, compression=compression)
    cache_s2 = Cache(str(out_s2), chunk_bytes=chunk_bytes, compression=compression)
    cache_s2_cloudy = Cache(str(out_s2_cloudy), chunk_bytes=chunk_bytes, compression=compression)

    meta_rows: list[dict[str, str | int]] = []
    for index, row in enumerate(tqdm(rows, desc=f"Split {split_name}", unit="sample")):
        paths = build_paths(base_dir, row)
        if not paths.s1_path.exists():
            raise FileNotFoundError(paths.s1_path)
        if not paths.s2_path.exists():
            raise FileNotFoundError(paths.s2_path)
        if not paths.s2_cloudy_path.exists():
            raise FileNotFoundError(paths.s2_cloudy_path)

        s1 = read_tiff(paths.s1_path).astype("float32")
        s2 = read_tiff(paths.s2_path)
        s2_cloudy = read_tiff(paths.s2_cloudy_path)

        cache_s1[index] = {"__key__": paths.key, "img": tiff_codec_io(s1, compression_args={"level": 9})}
        cache_s2[index] = {"__key__": paths.key, "img": tiff_codec_io(s2, compression_args={"level": 9})}
        cache_s2_cloudy[index] = {"__key__": paths.key, "img": tiff_codec_io(s2_cloudy, compression_args={"level": 9})}

        meta_rows.append(
            {
                "index": index,
                "__key__": paths.key,
                "split_id": split_id,
                "split": split_name,
                "season": paths.season,
                "scene_id": paths.scene_id,
                "patch_id": paths.patch_id,
                "s1_path": str(paths.s1_path.relative_to(base_dir)),
                "s2_path": str(paths.s2_path.relative_to(base_dir)),
                "s2_cloudy_path": str(paths.s2_cloudy_path.relative_to(base_dir)),
            }
        )

    for cache in (cache_s1, cache_s2, cache_s2_cloudy):
        cache.done()
        cache.merge()

    write_meta_csv(
        meta_rows,
        output_root / "litdata_meta" / f"{split_name}.csv",
    )


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir
    output_root = args.output_root or base_dir
    splits_csv = args.splits_csv or (base_dir / "splits" / "splits.csv")

    rows = load_splits_csv(splits_csv)
    grouped = group_by_split(rows)

    for split_id in tqdm(sorted(grouped.keys()), desc="All splits", unit="split"):
        if split_id not in SPLIT_NAME_MAP:
            raise ValueError(f"Unknown split id {split_id}. Update SPLIT_NAME_MAP if needed.")
        write_split(
            rows=grouped[split_id],
            base_dir=base_dir,
            output_root=output_root,
            split_id=split_id,
            split_name=SPLIT_NAME_MAP[split_id],
            chunk_bytes=args.chunk_bytes,
            compression=args.compression,
        )


if __name__ == "__main__":
    """
    python scripts/dataset/SEN12MS_CR/make_litdata_cache.py \
    --base-dir /Data2/ZihanCao/dataset/SEN12MS-CR \
    --output-root /Data2/ZihanCao/dataset/SEN12MS-CR \
    --chunk-bytes 512Mb
    """
    main()
