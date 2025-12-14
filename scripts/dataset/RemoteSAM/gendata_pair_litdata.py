import os
from pathlib import Path

import litdata as ld
import numpy as np
from loguru import logger
from PIL import Image
from tqdm import tqdm

from src.data.codecs import rgb_codec_io
from src.utilities.logging import set_logger_file


def optimize_fn_caption(
    caption_dir: str = "data/RemoteSAM270k/RemoteSAM-270K/captions/JPEGImages",
    name_file: str = "data2/RemoteSAM270k/name_from_litdata.txt",
    output_dir: str = "data2/RemoteSAM270k/LitData_image_captions",
):
    set_logger_file("tmp/sam270k_caption_litdata.log", add_time=False)
    f = open(name_file, mode="r")
    names = f.readlines()
    f.close()
    logger.info(f"Total {len(names)} samples to process.")

    def _get_caption_data(name: str) -> bytes:
        caption_path = f"{caption_dir}/{name}.jsonl"
        if not os.path.exists(caption_path):
            logger.warning(f"Missing caption {caption_path} for {name}, skipping", tqdm=True)
            return b"NotFound"
        return open(caption_path, "rb").read()

    def to_bytes(name: str):
        name = name.strip()
        caption_data = _get_caption_data(name)
        return {"__key__": name, "caption": caption_data}

    ld.optimize(
        to_bytes,
        names,
        output_dir,
        # chunk_bytes="512Mb",
        chunk_size=2800,
        num_workers=0,
        mode="overwrite",
        start_method="fork",
    )
    logger.success("Done.")


def optimize_fn_conditions(
    name_file: str = "data2/RemoteSAM270k/name_from_litdata.txt",
    cond_dir: str = "data/RemoteSAM270k/RemoteSAM-270K/tmp_conditions",
    output_dir: str = "data2/RemoteSAM270k/LitData_image_conditions",
):
    set_logger_file("tmp/sam270k_condition_litdata.log", add_time=False)
    f = open(name_file, mode="r")
    # writer = BinaryWriter(output_dir, chunk_bytes="512Mb")
    cond_types: list[str] = ["hed", "segmentation", "sketch", "mlsd"]

    def _get_cond_paths(stem):
        cond_paths = {c: f"{cond_dir}/{stem}.{c}.jpg" for c in cond_types}
        for cp in cond_paths.values():
            if not os.path.exists(cp):
                raise ValueError(f"Missing condition {cp} for {stem}, skipping")

        return cond_paths

    names = f.readlines()
    logger.info(f"Total {len(names)} samples to process.")
    f.close()

    def to_bytes(inp):
        name, index = inp
        name = name.strip()
        try:
            cond_paths = _get_cond_paths(name)
            cond_data = {c: open(cp, "rb").read() for c, cp in cond_paths.items()}
        except ValueError as e:
            logger.warning(str(e), tqdm=True)
            cond_data = {c: b"NotFound" for c in cond_types}

        if index % 1000 == 0:
            logger.info(f"Processed {index} samples.", tqdm=True)

        return {"__key__": name, **cond_data}

    ld.optimize(
        to_bytes,
        list(zip(names, range(len(names)))),
        output_dir,
        chunk_bytes="512Mb",
        num_workers=0,
        mode="overwrite",
        start_method="fork",
    )
    logger.success("Successfully optimized contions")


if __name__ == "__main__":
    optimize_fn_conditions()
    # optimize_fn_caption()
