import os
from pathlib import Path

import litdata as ld
import numpy as np
from loguru import logger
from PIL import Image
from tqdm import tqdm

from src.data.codecs import rgb_codec_io


def optimize_fn(
    name_file: str = "data2/RemoteSAM270k/name_from_litdata.txt",
    cond_dir: str = "data/RemoteSAM270k/RemoteSAM-270K/tmp_conditions",
    output_dir: str = "data2/RemoteSAM270k/LitData_image_conditions",
):
    f = open(name_file, mode="r")
    writer = ld.streaming.writer.BinaryWriter(output_dir, chunk_bytes="512Mb")
    cond_types: list[str] = ["hed", "segmentation", "sketch", "mlsd"]

    def _get_cond_paths(stem):
        cond_paths = {c: f"{cond_dir}/{stem}.{c}.jpg" for c in cond_types}
        for cp in cond_paths:
            if not os.path.exists(cp):
                raise ValueError(f"Missing condition {cp} for {stem}, skipping")

        return cond_paths

    names = f.readlines()
    i = 0
    for name in tqdm(names):
        name = name.strip()
        try:
            cond_paths = _get_cond_paths(name)
        except ValueError as e:
            logger.warning(str(e), tqdm=True)
            continue

        cond_data = {c: open(cp, "rb").read() for c, cp in cond_paths.items()}

        data = {"__key__": name, **cond_data}
        writer.add_item(i, data)
        i += 1
    f.close()


if __name__ == "__main__":
    optimize_fn()
