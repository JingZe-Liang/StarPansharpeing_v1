import multiprocessing as mp

mp.set_start_method("spawn", force=True)
from pathlib import Path

import numpy as np
import PIL.Image as Image
import tifffile
from litdata import optimize
from litdata.processing.data_processor import ALL_DONE
from litdata.streaming.writer import BinaryWriter
from loguru import logger
from tqdm import tqdm

from src.data.codecs import rgb_codec_io, tiff_codec_io

IMAGE_DIR = "data2/RemoteSAM270k/RawData"
OUTPUT_DIR = "data2/RemoteSAM270k/LitData_hyper_images2"
BASE_DIR = "data2/RemoteSAM270k"


def optimize_fn(path: str):
    if path.endswith(".tiff") or path.endswith(".tif"):
        img = tifffile.imread(path)
        enc_bytes = tiff_codec_io(
            img,
            compression="jpeg2000",
            compression_args={
                "reversible": False,
                "level": 98,
            },
        )
    else:
        img = np.array(Image.open(path))
        enc_bytes = rgb_codec_io(img, format="jpeg", quality=98)

    data = {"__key__": Path(path).stem, "img": enc_bytes}
    return data


def queue_data_producer(q, paths: list[str]):
    for path in paths:
        try:
            data = optimize_fn(path)
            q.put(data)
        except Exception as e:
            logger.error(f"Error processing {path}: {e}")

    q.put(ALL_DONE)


def fn(index):
    return index


def main(image_dir: str, output_dir: str, is_mp: bool = False):
    assert Path(image_dir).exists(), f"Image directory {image_dir} does not exist."
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # extensions
    extensions = [".jpg", ".jpeg", ".png", ".tiff", ".tif"]
    image_paths = []
    # walk through the directory
    for dirpath, dirnames, filenames in tqdm(Path(image_dir).walk(), desc="Collecting image paths"):
        for filename in filenames:
            file_path = Path(dirpath) / filename
            if file_path.suffix.lower() in extensions and file_path.is_file():
                image_paths.append(str(file_path))

    logger.info(f"Found {len(image_paths)} images in {image_dir}.")

    # optimize(optimize_fn, image_paths, output_dir, num_workers=0, mode="overwrite", start_method="spawn")
    if not is_mp:
        writer = BinaryWriter(output_dir, chunk_bytes="512Mb")
        name_file = Path(BASE_DIR) / "name.txt"
        file = open(name_file, "w")
        idx = 0
        for path in tqdm(image_paths, desc="Writing images", total=len(image_paths)):
            try:
                data = optimize_fn(path)
                writer.add_item(idx, data)
                name = Path(path).stem
                file.write(f"{name}\n")
                idx += 1  # 只有成功写入才递增
                if idx % 1000 == 0:
                    writer.write_chunks_index()
                    logger.info(f"Write index file.")
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
        writer.done()
        file.close()

    else:
        q = mp.Queue(100)
        p = mp.Process(target=queue_data_producer, args=(q, image_paths))
        p.start()
        logger.info("Starting queue data producer.")

        optimize(
            fn=fn,
            queue=q,
            output_dir=output_dir,
            chunk_bytes="512Mb",
            num_workers=4,
            mode="overwrite",
            start_method="spawn",
        )
        p.join()

    logger.success(f"Optimized images saved to {output_dir}.")


if __name__ == "__main__":
    main(image_dir=IMAGE_DIR, output_dir=OUTPUT_DIR, is_mp=False)
