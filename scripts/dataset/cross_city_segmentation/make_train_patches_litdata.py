from __future__ import annotations

import argparse
import io
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Any, Literal

import h5py
import litdata as ld
import numpy as np
import scipy.io as sio
import tifffile
import torch
import torch.nn.functional as F
from litdata.processing.data_processor import ALL_DONE
from safetensors.numpy import load_file
from tqdm import tqdm

from src.stage2.segmentation.data.cross_city_multimodal import generate_patch_coords


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export CrossCity train patches to LitData (streaming queue mode)")
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/Downstreams/CrossCitySegmentation",
        help="Root directory of CrossCitySegmentation",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        choices=["augsburg", "beijing"],
        default="augsburg",
        help="CrossCity source domain",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "val"],
        default="train",
        help="Dataset split to export. val+beijing will export wuhan as validation set.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="LitData output directory. Empty means auto path under data_root/litdata_train",
    )
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--patch-resize-to", type=int, default=0, help="0 means no resize")
    parser.add_argument("--upsample-hsi-to-msi", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--queue-size", type=int, default=1024)
    parser.add_argument("--write-mode", type=str, choices=["overwrite", "append"], default="overwrite")
    return parser.parse_args()


def _resolve_output_dir(args: argparse.Namespace) -> str:
    if args.output_dir:
        return args.output_dir
    resize_tag = "none" if int(args.patch_resize_to) <= 0 else str(args.patch_resize_to)
    suffix = f"{args.dataset_name}_ps{args.patch_size}_rs{resize_tag}_s{args.stride}_split_modal"
    return str(Path(args.data_root) / "litdata_train" / suffix / args.mode)


def _read_h5_cube(h5_file: h5py.File, key: str) -> np.ndarray:
    arr = np.array(h5_file[key])
    if arr.ndim == 3 and arr.shape[0] < arr.shape[-1]:
        print(f">>>>>> warning: arr is dim-3, shaped as {arr.shape}")
        arr = np.transpose(arr, (1, 2, 0))
    return arr


def _upsample_hsi(hsi: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    if hsi.shape[:2] == target_hw:
        return hsi
    t = torch.from_numpy(hsi).permute(2, 0, 1).unsqueeze(0).float()
    up = F.interpolate(t, size=target_hw, mode="nearest")
    return up.squeeze(0).permute(1, 2, 0).numpy().astype(np.float32, copy=False)


def _resolve_beijing_files(data_root: Path, mode: Literal["train", "val"]) -> tuple[Path, Path]:
    data_dir = data_root / "data2"
    if mode == "train":
        return data_dir / "beijing.safetensors", data_dir / "beijing_label.mat"

    wuhan_mat_file = data_dir / "wuhan.mat"
    if wuhan_mat_file.exists():
        return wuhan_mat_file, data_dir / "wuhan_label.mat"

    wuhan_safe_file = data_dir / "wuhan.safetensors"
    if wuhan_safe_file.exists():
        return wuhan_safe_file, data_dir / "wuhan_label.mat"
    raise FileNotFoundError(f"Cannot find wuhan data file under {data_dir}")


def _load_modalities(
    data_root: Path,
    dataset_name: Literal["augsburg", "beijing"],
    mode: Literal["train", "val"],
    upsample_hsi_to_msi: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if dataset_name == "augsburg":
        file_name = "augsburg_multimodal.mat" if mode == "train" else "berlin_multimodal.mat"
        train_file = data_root / "data1" / file_name
        print(f"File {train_file} has size {Path(train_file).stat().st_size / (1024**3)} Gb")
        data = sio.loadmat(train_file)
        hsi = data["HSI"]
        msi = data["MSI"]
        sar = data["SAR"]
        gt = data["label"]
    elif dataset_name == "beijing":
        train_file, label_file = _resolve_beijing_files(data_root, mode)
        print(f"File {train_file} has size {Path(train_file).stat().st_size / (1024**3)} Gb")
        if train_file.suffix == ".safetensors":
            f = load_file(train_file)
            hsi = f["HSI"].transpose(1, 2, 0)
            msi = f["MSI"].transpose(1, 2, 0)
            sar = f["SAR"].transpose(1, 2, 0)
        elif train_file.suffix == ".mat":
            with h5py.File(train_file, "r") as f:
                hsi = _read_h5_cube(f, "HSI")
                msi = _read_h5_cube(f, "MSI")
                sar = _read_h5_cube(f, "SAR")
        else:
            raise ValueError(f"Unsupported file suffix for beijing/wuhan: {train_file.suffix}")

        if label_file.exists():
            with h5py.File(label_file, "r") as f_label:
                gt = _read_h5_cube(f_label, "label")
        else:
            raise
        if gt is None:
            raise ValueError("Cannot find training label for beijing")
        if upsample_hsi_to_msi and hsi.shape[:2] != msi.shape[:2]:
            hsi = _upsample_hsi(hsi.astype(np.float32, copy=False), msi.shape[:2])
    else:
        raise ValueError(f"Unsupported dataset_name={dataset_name}")
    print("load train file/label done.")

    if gt.ndim == 3:
        gt = np.squeeze(gt)

    hsi = hsi.astype(np.float32, copy=False)
    msi = msi.astype(np.float32, copy=False)
    sar = sar.astype(np.float32, copy=False)
    gt = gt.astype(np.int16, copy=False)

    if hsi.shape[:2] != msi.shape[:2]:
        scale_y = msi.shape[0] / hsi.shape[0]
        scale_x = msi.shape[1] / hsi.shape[1]
        if np.isclose(scale_y, round(scale_y)) and np.isclose(scale_x, round(scale_x)):
            print(
                "Auto upsample HSI to MSI resolution due to spatial mismatch: "
                f"hsi={hsi.shape[:2]} -> msi={msi.shape[:2]}"
            )
            hsi = _upsample_hsi(hsi, msi.shape[:2])
        else:
            raise ValueError(
                "HSI/MSI spatial mismatch cannot be safely auto-aligned: "
                f"hsi={hsi.shape[:2]}, msi={msi.shape[:2]}. "
                "Please check source files or run with --upsample-hsi-to-msi."
            )

    if not (hsi.shape[:2] == msi.shape[:2] == sar.shape[:2] == gt.shape[:2]):
        raise ValueError(
            "Spatial mismatch among modalities/gt: "
            f"hsi={hsi.shape[:2]}, msi={msi.shape[:2]}, sar={sar.shape[:2]}, gt={gt.shape[:2]}."
        )

    return hsi, msi, sar, gt


def _encode_tif_zlib_bytes(img_hwc: np.ndarray) -> bytes:
    with io.BytesIO() as buffer:
        tifffile.imwrite(buffer, img_hwc, compression="zlib", compressionargs={"level": 7})
        return buffer.getvalue()


def _extract_patch(arr: np.ndarray, y: int, x: int, patch_size: int) -> np.ndarray:
    return arr[y : y + patch_size, x : x + patch_size, ...]


def _maybe_resize_patch(
    hsi: np.ndarray,
    msi: np.ndarray,
    sar: np.ndarray,
    gt: np.ndarray,
    patch_resize_to: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if patch_resize_to <= 0:
        return hsi, msi, sar, gt

    def _resize_img(x: np.ndarray) -> np.ndarray:
        x_t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()
        x_t = F.interpolate(x_t, size=(patch_resize_to, patch_resize_to), mode="bilinear", align_corners=False)
        return x_t.squeeze(0).permute(1, 2, 0).numpy().astype(np.float32, copy=False)

    def _resize_gt(x: np.ndarray) -> np.ndarray:
        x_t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float()
        x_t = F.interpolate(x_t, size=(patch_resize_to, patch_resize_to), mode="nearest")
        return x_t.squeeze(0).squeeze(0).numpy().astype(np.int16, copy=False)

    return _resize_img(hsi), _resize_img(msi), _resize_img(sar), _resize_gt(gt)


def _producer(
    q: Queue,
    hsi: np.ndarray,
    msi: np.ndarray,
    sar: np.ndarray,
    gt: np.ndarray,
    dataset_name: str,
    patch_size: int,
    stride: int,
    patch_resize_to: int,
) -> None:
    h, w = gt.shape
    coords = generate_patch_coords(h, w, patch_size, stride)
    print(f"Will generate {len(coords)} patches")

    for i, (y, x) in tqdm(enumerate(coords)):
        hsi_p = _extract_patch(hsi, y, x, patch_size)
        msi_p = _extract_patch(msi, y, x, patch_size)
        sar_p = _extract_patch(sar, y, x, patch_size)
        gt_p = _extract_patch(gt, y, x, patch_size)

        hsi_p, msi_p, sar_p, gt_p = _maybe_resize_patch(hsi_p, msi_p, sar_p, gt_p, patch_resize_to)

        sample = {
            "__key__": f"{dataset_name}_{i:04d}",
            "img_hsi": _encode_tif_zlib_bytes(hsi_p),
            "img_msi": _encode_tif_zlib_bytes(msi_p),
            "img_sar": _encode_tif_zlib_bytes(sar_p),
            "gt": gt_p,
        }
        q.put(sample)

    q.put(ALL_DONE)


def _identity(x: dict[str, Any]) -> dict[str, Any]:
    return x


def main() -> None:
    args = parse_args()
    output_dir = _resolve_output_dir(args)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root)
    print("loading full data.")
    hsi, msi, sar, gt = _load_modalities(
        data_root=data_root,
        dataset_name=args.dataset_name,
        mode=args.mode,
        upsample_hsi_to_msi=args.upsample_hsi_to_msi,
    )
    print("start preparing litdata.")

    q: Queue = Queue(maxsize=args.queue_size)
    worker = Process(
        target=_producer,
        args=(
            q,
            hsi,
            msi,
            sar,
            gt,
            f"{args.dataset_name}_{args.mode}",
            args.patch_size,
            args.stride,
            int(args.patch_resize_to),
        ),
    )
    worker.start()

    ld.optimize(
        fn=_identity,
        queue=q,
        output_dir=output_dir,
        num_workers=args.num_workers,
        chunk_bytes="512Mb",
        mode=args.write_mode,
        start_method="fork",
    )
    worker.join()

    print(f"LitData exported to: {output_dir}")


if __name__ == "__main__":
    main()
