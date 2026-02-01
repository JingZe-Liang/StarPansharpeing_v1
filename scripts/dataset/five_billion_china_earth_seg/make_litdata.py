import os
from pathlib import Path
from multiprocessing import Queue, Process
from litdata import optimize
import torch
import numpy as np
import tifffile
from tqdm import tqdm
import natsort
from typing import Literal
import argparse

# Import necessary functions from demo.py
from scripts.dataset.five_billion_china_earth_seg.demo import (
    read_image,
    sliding_window,
    postprocess_img,
    to_suitable_dtype_img,
    img_saver_backend_compact_with_wds,
    _litdate_identity,
    ALL_DONE,
)
from src.utilities.logging.print import logger, catch_any

# Define the root directory of the dataset
DATASET_ROOT = "data/Downstreams/5Billion-ChinaCity-Segmentation"
OUTPUT_DIR = "data/Downstreams/5Billion-ChinaCity-Segmentation/litdata"

# Define parameters
IMG_CLIP_SIZE = (1024, 1024)
IMG_STRIDE = (1024, 1024)
SAVE_KWARGS = {
    "tiff_compression_type": "jpeg2000",
    "tiff_jpg_irreversible": True,
    "jpeg_quality": 90,
}


def process_and_save_to_queue(
    img_path: Path,
    label_path: Path,
    sink: Queue,
    img_clip_size: tuple[int, int],
    img_stride: tuple[int, int],
    save_kwargs: dict,
    total_img_saved_count: int,
):
    try:
        # Read image
        img = read_image(img_path)
        if img is None:
            return 0

        # Read label
        # Label is usually a grayscale image or indexed image
        label = read_image(label_path, tiff_read_mode="array")
        if label is None:
            logger.warning(f"Failed to read label from {label_path}, skipping.")
            return 0

        # Ensure label has channel dim for standardized processing
        if label.ndim == 2:
            label = label[..., None]

        # Use sliding window to crop both image and label
        # Image sliding window
        img_patches_gen = sliding_window(
            img,
            img_clip_size,
            img_stride,
            is_yield=True,
            is_hwc=True,
        )

        # Label sliding window
        label_patches_gen = sliding_window(
            label,
            img_clip_size,
            img_stride,
            is_yield=True,
            is_hwc=True,
        )

        n_patches = 0

        # Iterate over patches
        for (img_patch, img_coord), (label_patch, label_coord) in zip(img_patches_gen, label_patches_gen):
            if img_patch is None or label_patch is None:
                continue

            # Verify coordinates match
            assert img_coord == label_coord, f"Coordinates mismatch: {img_coord} != {label_coord}"

            n_patches += 1
            total_img_saved_count += 1

            # Process Image Patch
            img_patch_proc, min_max = postprocess_img(
                img_patch,
                to_tensor=True,
                normalize=False,  # Do not normalize for raw saving if we want original values, or set to True if needed
                transpose=True,  # [C, H, W]
                rescale="clamp",
            )
            # Find suitable dtype and convert back to numpy [H, W, C] if needed by saver
            img_patch_proc, _ = to_suitable_dtype_img(img_patch_proc, *min_max, is_normed=False)

            # Process Label Patch
            # Label usually doesn't need normalization or rescaling in the same way, but needs to be handled
            # Assuming label is uint8 or similar integer type
            label_patch_proc = label_patch.cpu().numpy() if torch.is_tensor(label_patch) else label_patch

            # Construct dictionary item
            img_name = img_path.stem
            patch_idx = n_patches - 1
            saved_name = f"{total_img_saved_count}_{img_name}_patch-{patch_idx}"

            # Compress image patch
            img_bytes = img_saver_backend_compact_with_wds(img_patch_proc, extension="tiff", **save_kwargs)

            # Compress label patch (use png for exact values, or tiff with LZW/None)
            # Using 'tiff' with 'zlib' or 'lzw' for lossless label compression is safer than jpeg2000 if not configured carefully
            label_bytes = img_saver_backend_compact_with_wds(
                label_patch_proc,
                extension="tiff",
                tiff_compression_type="zlib",  # Lossless
                tiff_jpg_irreversible=False,
            )

            data_item = {"__key__": saved_name, "img": img_bytes, "label": label_bytes}

            sink.put(data_item)

        return n_patches

    except Exception as e:
        logger.error(f"Error processing {img_path}: {e}")
        return 0


TEST_FILES = {
    "GF2_PMS1__L1A0001064454-MSS1.tiff",
    "GF2_PMS1__L1A0001118839-MSS1.tiff",
    "GF2_PMS1__L1A0001344822-MSS1.tiff",
    "GF2_PMS1__L1A0001348919-MSS1.tiff",
    "GF2_PMS1__L1A0001366278-MSS1.tiff",
    "GF2_PMS1__L1A0001366284-MSS1.tiff",
    "GF2_PMS1__L1A0001395956-MSS1.tiff",
    "GF2_PMS1__L1A0001432972-MSS1.tiff",
    "GF2_PMS1__L1A0001670888-MSS1.tiff",
    "GF2_PMS1__L1A0001680857-MSS1.tiff",
    "GF2_PMS1__L1A0001680858-MSS1.tiff",
    "GF2_PMS1__L1A0001757429-MSS1.tiff",
    "GF2_PMS1__L1A0001765574-MSS1.tiff",
    "GF2_PMS2__L1A0000607677-MSS2.tiff",
    "GF2_PMS2__L1A0000607681-MSS2.tiff",
    "GF2_PMS2__L1A0000718813-MSS2.tiff",
    "GF2_PMS2__L1A0001038935-MSS2.tiff",
    "GF2_PMS2__L1A0001038936-MSS2.tiff",
    "GF2_PMS2__L1A0001119060-MSS2.tiff",
    "GF2_PMS2__L1A0001367840-MSS2.tiff",
    "GF2_PMS2__L1A0001378491-MSS2.tiff",
    "GF2_PMS2__L1A0001378501-MSS2.tiff",
    "GF2_PMS2__L1A0001396036-MSS2.tiff",
    "GF2_PMS2__L1A0001396037-MSS2.tiff",
    "GF2_PMS2__L1A0001416129-MSS2.tiff",
    "GF2_PMS2__L1A0001471436-MSS2.tiff",
    "GF2_PMS2__L1A0001517494-MSS2.tiff",
    "GF2_PMS2__L1A0001591676-MSS2.tiff",
    "GF2_PMS2__L1A0001787564-MSS2.tiff",
    "GF2_PMS2__L1A0001821754-MSS2.tiff",
}

# Also support .tif extension in check
TEST_FILES_STEMS = {f.split(".")[0] for f in TEST_FILES}


def runner(split: Literal["train", "test", "all"], sink: Queue):
    """
    Runner function for processing train/test split.

    Parameters
    ----------
    split : Literal["train", "test", "all"]
        Which split to process. If "all", both train and test will be processed
        and two sinks are required.
    sink : Queue
        Queue for putting processed data items.
    """
    img_dir = Path(DATASET_ROOT) / "Image_16bit_RGBNir"
    label_dir = Path(DATASET_ROOT) / "Annotation_Index"

    # List all image files
    img_files = list(img_dir.glob("*.tiff")) + list(img_dir.glob("*.tif"))
    img_files = natsort.natsorted(img_files)

    logger.info(f"Found {len(img_files)} images.")
    logger.info(f"Processing split: {split}")

    total_count = 0

    for img_path in tqdm(img_files):
        # Derive label path
        label_name = f"{img_path.stem}_24label.png"
        label_path = label_dir / label_name

        if not label_path.exists():
            logger.warning(f"Label file not found for {img_path}, skipping.")
            continue

        # Determine if it's train or test
        is_test = img_path.name in TEST_FILES or img_path.stem in TEST_FILES_STEMS

        # Skip files that don't match the split
        if split == "train" and is_test:
            continue
        if split == "test" and not is_test:
            continue

        count = process_and_save_to_queue(
            img_path,
            label_path,
            sink,
            IMG_CLIP_SIZE,
            IMG_STRIDE,
            SAVE_KWARGS,
            total_count,
        )

        total_count += count

    logger.info(f"All files processed. Total patches: {total_count}.")
    sink.put(ALL_DONE)


def runner_all(train_sink: Queue, test_sink: Queue):
    img_dir = Path(DATASET_ROOT) / "Image_16bit_RGBNir"
    label_dir = Path(DATASET_ROOT) / "Annotation_Index"

    # List all image files
    img_files = list(img_dir.glob("*.tiff")) + list(img_dir.glob("*.tif"))
    img_files = natsort.natsorted(img_files)

    logger.info(f"Found {len(img_files)} images.")

    total_train_count = 0
    total_test_count = 0

    for img_path in tqdm(img_files):
        # Derive label path
        label_name = f"{img_path.stem}_24label.png"
        label_path = label_dir / label_name

        if not label_path.exists():
            logger.warning(f"Label file not found for {img_path}, skipping.")
            continue

        # Determine if it's train or test
        is_test = img_path.name in TEST_FILES or img_path.stem in TEST_FILES_STEMS

        sink = test_sink if is_test else train_sink
        current_count = total_test_count if is_test else total_train_count

        count = process_and_save_to_queue(
            img_path,
            label_path,
            sink,
            IMG_CLIP_SIZE,
            IMG_STRIDE,
            SAVE_KWARGS,
            current_count,
        )

        if is_test:
            total_test_count += count
        else:
            total_train_count += count

    logger.info(f"All files processed. Train patches: {total_train_count}, Test patches: {total_test_count}.")
    train_sink.put(ALL_DONE)
    test_sink.put(ALL_DONE)


@catch_any()
def main(split: Literal["train", "test", "all"]):
    """
    Main function to process dataset.

    Parameters
    ----------
    split : Literal["train", "test", "all"]
        Which split to process.
    """
    if not OUTPUT_DIR:
        raise ValueError("OUTPUT_DIR is not set")

    base_output_dir = Path(OUTPUT_DIR)

    # Process based on split
    if split == "all":
        train_output_dir = base_output_dir / "train"
        test_output_dir = base_output_dir / "test"

        train_output_dir.mkdir(parents=True, exist_ok=True)
        test_output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Queues
        train_q: Queue = Queue(maxsize=10)
        test_q: Queue = Queue(maxsize=10)

        # Start Producer Process
        producer = Process(target=runner_all, args=(train_q, test_q), daemon=True)
        producer.start()

        # Run optimize in main process for train
        optimize(
            fn=_litdate_identity,
            queue=train_q,
            output_dir=str(train_output_dir),
            num_workers=0,
            chunk_bytes="256Mb",
            mode="overwrite",
            start_method="fork",
        )

        # Run optimize in main process for test
        optimize(
            fn=_litdate_identity,
            queue=test_q,
            output_dir=str(test_output_dir),
            num_workers=0,
            chunk_bytes="256Mb",
            mode="overwrite",
            start_method="fork",
        )

        # Wait for producer to finish
        producer.join()

    elif split == "train":
        train_output_dir = base_output_dir / "train"
        train_output_dir.mkdir(parents=True, exist_ok=True)

        train_q: Queue = Queue(maxsize=10)  # type: ignore[no-redef]

        producer = Process(target=runner, args=("train", train_q), daemon=True)
        producer.start()

        optimize(
            fn=_litdate_identity,
            queue=train_q,
            output_dir=str(train_output_dir),
            num_workers=0,
            chunk_bytes="256Mb",
            mode="overwrite",
            start_method="fork",
        )

        producer.join()

    elif split == "test":
        test_output_dir = base_output_dir / "test"
        test_output_dir.mkdir(parents=True, exist_ok=True)

        test_q: Queue = Queue(maxsize=10)  # type: ignore[no-redef]

        producer = Process(target=runner, args=("test", test_q), daemon=True)
        producer.start()

        optimize(
            fn=_litdate_identity,
            queue=test_q,
            output_dir=str(test_output_dir),
            num_workers=0,
            chunk_bytes="256Mb",
            mode="overwrite",
            start_method="fork",
        )

        producer.join()

    logger.success(f"Dataset creation completed for split: {split}.")


if __name__ == "__main__":
    from src.utilities.logging import set_logger_file

    parser = argparse.ArgumentParser(description="Process 5Billion China City Segmentation dataset")
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test", "all"],
        default="all",
        help="Which split to process: train, test, or all (default: all)",
    )

    args = parser.parse_args()

    set_logger_file(
        "data/Downstreams/5Billion-ChinaCity-Segmentation/make_litdata_5billion.log", "debug", add_time=False
    )
    main(args.split)
