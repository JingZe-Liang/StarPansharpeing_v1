import functools
import io
import os
import shutil
import tarfile
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Generator

import duckdb
import natsort
import numpy as np
import pyarrow.parquet as pq
import tifffile
import webdataset as wds
from datasets import Dataset, load_dataset
from loguru import logger
from PIL import Image
from pyarrow import Table as pa_Table
from pyarrow.parquet import ParquetFile

# rich tqdm
from rich.progress import track

from src.data.codecs import py_obj_to_jsonl, rgb_codec_io, tiff_codec_io
from src.utilities.logging.print import _console


def get_dir_size(path):
    """Return total size of files in the directory (in bytes)."""
    path = Path(path)
    if not path.exists():
        return 0
    elif path.is_file():
        return path.stat().st_size

    total = 0
    for root, dirs, files in path.walk():
        for file in files:
            file_path = root / file
            total += file_path.stat().st_size
    return total


# read parquet file into a Dataset
# def parquet_to_webdataset(parquet_file):
#     dataset = load_dataset(
#         "parquet", data_files=parquet_file
#     )  # this will cache the dataset into /tmp
#     logger.info(f"Loaded dataset from {parquet_file}")
#     return dataset["train"]


def parquet_reader(parquet_file):
    table = pq.read_table(parquet_file, use_threads=True, memory_map=True)
    return Dataset(table)


def parquet_reader_duckdb(parquet_file):
    """使用DuckDB进行超快速读取"""
    with duckdb.connect() as conn:
        arrow_table = conn.execute(f"SELECT * FROM '{parquet_file}'").arrow()
    # arrow_table is already a pyarrow.Table, pass it directly to Dataset
    return Dataset(arrow_table)


def parquet_reader_pyarrow(parquet_file):
    table = pq.read_table(parquet_file, use_threads=True, memory_map=True)

    def inner():
        nonlocal table

        for tile_i in range(table.num_rows):
            pyd = table.slice(tile_i, 1).to_pydict()
            # keys: rgb, 1m
            pyd["1m"] = np.array(pyd["1m"][0])
            pyd["rgb"] = np.array(pyd["rgb"][0])
            pyd["metadata"] = pyd["metadata"][0]

            yield pyd

    return table.num_rows, inner()


def reading_parquet_file_neon(parquet_file):
    # ds = parquet_to_webdataset(parquet_file)  #! this will cache the dataset into /tmp, Why?
    ds = parquet_reader(parquet_file)
    # ds = parquet_reader_duckdb(parquet_file)

    name = Path(parquet_file).stem
    name = name.replace(".", "-")
    tile_n = len(ds)
    for tile_i, sample in enumerate(ds):
        rgb = np.array(sample["rgb"]).astype(np.uint8)
        hsi = np.array(sample["1m"]).astype(np.uint8)  # 255. max, also to uint8
        metadata = sample["metadata"]

        # transpose
        rgb = rgb.transpose(0, 2, 3, 1)
        hsi = hsi.transpose(0, 2, 3, 1)

        for i, (rgb_i, hsi_i) in enumerate(zip(rgb, hsi)):
            logger.info(f"Processing tile [{tile_i}/{tile_n}], sample {i} in {name}")
            # create sample
            sample = {
                "__key__": name + f"-tile-{tile_i}" + f"-seq-{i}",
                "rgb.jpg": rgb_codec_io(rgb_i, quality=80),
                "hsi.tiff": tiff_codec_io(
                    hsi_i,
                    compression="jpeg2000",
                    compression_args={"level": 80},
                ),
                "metadata.jsonl": py_obj_to_jsonl(metadata),
            }

            yield sample


def reading_parquet_file_satellogic(parquet_file):
    tile_n = None

    # > reading parquet file
    # ds = parquet_to_webdataset(parquet_file)  #! this will cache the dataset into /tmp, Why?
    # ds = parquet_reader(parquet_file)
    # ds = parquet_reader_duckdb(parquet_file)
    tile_n, ds = parquet_reader_pyarrow(parquet_file)

    # > loops
    name = Path(parquet_file).stem
    name = name.replace(".", "-")
    tile_n = len(ds) if tile_n is None else tile_n  # type: ignore
    for tile_i, sample in enumerate(ds):
        rgb = np.asarray(sample["rgb"]).astype(np.uint8)
        nir = np.asarray(sample["1m"]).astype(np.uint8)  # 255. max, also to uint8
        # cat together
        img = np.concatenate([rgb, nir], axis=1)
        metadata = sample["metadata"]

        # transpose
        img = img.transpose(0, 2, 3, 1)

        for i, img_i in enumerate(img):
            logger.info(f"Processing tile [{tile_i}/{tile_n}], sample {i} in {name}")
            # create sample
            sample = {
                "__key__": name + f"-tile-{tile_i}" + f"-seq-{i}",
                "img.tiff": tiff_codec_io(
                    img_i,
                    compression="jpeg2000",
                    compression_args={"level": 80},
                ),
                "metadata.jsonl": py_obj_to_jsonl(metadata),
            }

            yield sample


def parquet_to_webdataset_writter(
    shard_writer: wds.writer.ShardWriter,
    sample_geneator: Generator,
):
    for sample in sample_geneator:
        shard_writer.write(sample)


def move_file_to_tmp(file, tmp_dir, clear_dir_lock):
    try:
        shutil.move(file, tmp_dir / file.name)
        logger.info(f"Processed {file.name} and moved to {tmp_dir / file.name}")
        # Dir size check and cleanup need to be thread-safe
        if (dir_size := get_dir_size(tmp_dir)) > 10 * 1024 * 1024 * 1024:
            with clear_dir_lock:
                if get_dir_size(tmp_dir) > 10 * 1024 * 1024 * 1024:
                    shutil.rmtree(tmp_dir)
                    tmp_dir.mkdir(parents=True, exist_ok=True)
                    logger.info("clear tmp dir {} GB".format(dir_size / (1024**3)))
                    return 1
        return 0
    except Exception as e:
        logger.error(f"Error in collect_results: {e}")
        return -1


# * --- one process, multi-threads, multi-processes-threads processing --- #


def parquet_to_webdataset(
    parquet_dir: str | list[str],
    output_pattern: str = "neon-%04d.tar",
    remove_file: bool = False,
    async_remove: bool = False,
    dataset_name: str = "neon",
):
    tmp_dir = Path("/Data4/cao/ZiHanCao/exps/HyperspectralTokenizer/tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    Path(output_pattern).parent.mkdir(parents=True, exist_ok=True)

    if remove_file and async_remove:
        clear_dir_lock = Lock()
        executor = ThreadPoolExecutor(max_workers=4)

        def move_async(file: Path, tmp_dir: Path):
            return executor.submit(
                move_file_to_tmp, Path(file), Path(tmp_dir), clear_dir_lock
            )

        future_to_file = {}
    else:
        future_to_file = None

    reading_fn = reading_fns.get(dataset_name, None)
    assert reading_fn is not None, f"Unsupported dataset name: {dataset_name}"

    shard_writter = wds.writer.ShardWriter(
        output_pattern, maxsize=4 * 1024 * 1024 * 1024, start_shard=8
    )  # 4GB max size
    if isinstance(parquet_dir, str):
        files = list(Path(parquet_dir).glob("*.parquet"))
        files = natsort.natsorted(files, key=lambda x: x.name)
        logger.info(f"Found {len(files)} parquet files in {parquet_dir}")
    elif isinstance(parquet_dir, list):
        files = [Path(f) for f in parquet_dir]
        for f in files:
            assert Path(f).exists(), f"File {f} does not exist"
        files = natsort.natsorted(files, key=lambda x: x.name)
        logger.info(f"Found {len(files)} parquet files in {parquet_dir}")
    else:
        raise ValueError("parquet_dir must be a string or a list of strings")

    for file in track(
        files,
        description="Processing parquet files",
        transient=True,
        total=len(files),
        console=_console,
        disable=False,
    ):
        # sample_generator = reading_parquet_file_neon(file.as_posix())
        sample_generator = reading_fn(file.as_posix())

        parquet_to_webdataset_writter(
            shard_writer=shard_writter,
            sample_geneator=sample_generator,
        )
        if remove_file:
            if async_remove:
                future = move_async(file, tmp_dir)
                future_to_file[future] = file
            else:
                move_file_to_tmp(file, tmp_dir, clear_dir_lock)

    if async_remove:
        for future in as_completed(future_to_file):
            try:
                res = future.result()
                file = future_to_file[future]
                if res == -1:
                    logger.error(f"Error moving file {file} to tmp dir")
                elif res == 1:
                    logger.info(f"Temp dir is larger than 10GB, clear {tmp_dir}")
                else:
                    logger.info(f"Moved file {file} to tmp dir")
            except Exception as e:
                logger.error(f"Exception during move: {e}")

    shard_writter.close()
    logger.info(f"WebDataset written to {output_pattern}")


def parquet_to_webdataset_async(
    parquet_dir: str | list[str],
    output_root: str | Path = "data/EarthView/hyper_images2",
    to_wds=False,
    n_threads: int = 4,
    dataset_name: str = "neon",
    remove_file: bool = False,
):
    """
    多线程并行把 parquet 转成 WebDataset。
    每个线程独占一个子目录 thread_{idx}/，内部顺序写 shard。
    remove_file=True 时，主线程在所有任务完成后阻塞删除源文件。
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # 1. 展开 / 收集文件列表
    if isinstance(parquet_dir, str):
        files = list(Path(parquet_dir).glob("*.parquet"))
    else:
        files = [Path(p) for p in parquet_dir]
    files = natsort.natsorted(files, key=lambda x: x.name)
    logger.info(f"Found {len(files)} parquet files")

    # 2. 按线程数切片
    files_per_thread = [files[i::n_threads] for i in range(n_threads)]

    # 3. 线程任务
    reading_fn = reading_fns[dataset_name]

    def worker(thread_idx: int, file_list):
        _len = len(file_list)
        logger.info(f"thread {thread_idx} writing {_len} files")

        try:
            # > all to webdataset shards
            if to_wds:
                out_dir = output_root / f"thread_{thread_idx}"
                out_dir.mkdir(parents=True, exist_ok=True)
                pattern = str(out_dir / f"{dataset_name}-%04d.tar")

                with wds.writer.ShardWriter(
                    pattern, maxsize=8 * 1024**3, start_shard=0
                ) as sink:
                    for f in file_list:
                        for sample in reading_fn(str(f)):
                            sink.write(sample)
            # > all to dir
            else:
                Path(output_root).mkdir(parents=True, exist_ok=True)

                for f in file_list:
                    for sample in reading_fn(str(f)):
                        pair_name = Path(output_root, sample["__key__"])
                        img_name = pair_name.with_suffix(".img.tiff")
                        metadata_name = pair_name.with_suffix(".metadata.jsonl")
                        with open(img_name, "wb") as img_file:
                            img_file.write(sample["img.tiff"])
                        with open(metadata_name, "wb") as meta_file:
                            meta_file.write(sample["metadata.jsonl"])

            logger.info(
                f"Thread {thread_idx} finished, processed {len(file_list)} files"
            )
        except Exception as e:
            logger.error(f"Thread {thread_idx} failed: {e}")

    # 4. 启动线程池
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = [pool.submit(worker, i, fl) for i, fl in enumerate(files_per_thread)]
        for f in as_completed(futures):
            f.result()  # 如有异常会在这里抛出

    # 5. 阻塞删除源文件（如需）
    if remove_file:
        for f in files:
            f.unlink(missing_ok=True)
        logger.info(f"Removed {len(files)} source parquet files")


def _worker(
    thread_idx: int,
    file_list: list[Path],
    output_root: Path,
    dataset_name: str,
) -> None:
    """
    子进程任务：把若干 parquet 文件写成 WebDataset shard。
    必须放在顶层模块，否则 ProcessPoolExecutor 无法序列化。
    """
    out_dir = output_root / f"thread_{thread_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / f"{dataset_name}-%04d.tar")

    reading_fn = reading_fns[dataset_name]

    with wds.writer.ShardWriter(pattern, maxsize=4 * 1024**3, start_shard=0) as sink:
        for f in file_list:
            for sample in reading_fn(str(f)):
                sink.write(sample)

    logger.info(f"Process {os.getpid()} finished, processed {len(file_list)} files")


def parquet_to_webdataset_async_mp_tp(
    parquet_dir: str | list[str] | Path,
    output_root: str | Path = "data/EarthView/hyper_images2",
    n_workers: int = 4,
    dataset_name: str = "neon",
    remove_file: bool = False,
) -> None:
    """
    多进程并行把 parquet 转成 WebDataset。
    每个进程独占一个子目录 thread_{idx}/，内部顺序写 shard。
    remove_file=True 时，主进程在所有任务完成后阻塞删除源文件。
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # 1. 收集文件
    if isinstance(parquet_dir, (str, Path)):
        files = list(Path(parquet_dir).glob("*.parquet"))
    else:
        files = [Path(p) for p in parquet_dir]
    files = natsort.natsorted(files, key=lambda x: x.name)
    logger.info(f"Found {len(files)} parquet files")

    # 2. 按进程数切片
    files_per_worker = [files[i::n_workers] for i in range(n_workers)]

    # 3. 启动进程池
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [
            pool.submit(_worker, i, fl, output_root, dataset_name)
            for i, fl in enumerate(files_per_worker)
        ]
        for fut in as_completed(futures):
            fut.result()  # 如有异常会在这里抛出

    # 4. 删除源文件
    if remove_file:
        for f in files:
            f.unlink(missing_ok=True)
        logger.info(f"Removed {len(files)} source parquet files")


def shard_to_one_dir_sorted(file_dir: str, name=None):
    files = list(Path(file_dir).glob("**/*.tar"))
    files = natsort.natsorted(files)

    if name is None:
        name = "shard"

    for i, file in enumerate(files):
        new_name = f"{file_dir}/{name}-{i:04d}.tar"
        shutil.move(file, new_name)
        logger.info(f"move {file} to {new_name}")


def shards_tar_merged(
    tar_files=None,
    tar_dir=None,
    output_file=None,
):
    assert output_file is not None, "output_file must be specified"

    if tar_dir is not None:
        tar_dir = Path(tar_dir)
        tar_files = list(tar_dir.glob("**/*.tar"))

    assert tar_files and len(tar_files) > 0, "No .tar files provided or found"
    tar_files = [Path(f) for f in tar_files]

    # sort
    tar_files = natsort.natsorted(tar_files, key=lambda f: f.stem)
    logger.info(f"Merged list are {tar_files}")

    assert tar_files is not None and len(tar_files) > 0, (
        "No .tar files provided or found"
    )

    tar_files = [Path(f) for f in tar_files if Path(f).stem != "merged"]

    with tarfile.open(output_file, "w") as out_tar:
        for tar_path in tar_files:
            print(f"Merging {tar_path}...")
            # breakpoint()
            with tarfile.open(tar_path, "r") as tar:
                # num = sum(1 for m in tar if m.isfile())
                # logger.info(f"{tar_path} contains {num} files")
                # for member in track(tar, total=num, description="Extracting"):
                for member in tar:
                    try:
                        extract = tar.extractfile(member)
                        out_tar.addfile(member, extract)
                    except Exception as e:
                        logger.error(
                            f"Error extracting {member.name} from {tar_path}: {e}"
                        )


if __name__ == "__main__":
    # path = "/Data4/cao/ZiHanCao/exps/HyperspectralTokenizer/data/EarthView/data/train-00000-of-00607.parquet"
    # dataset = parquet_to_webdataset(path)

    # for i, sample in enumerate(dataset):
    #     # print(f"Sample {i}: {sample.keys()}")
    #     # print(f'RGB shape: {sample["rgb"].shape}')
    #     print(i)

    # shards_tar_merged(
    #     tar_dir="/Data4/cao/ZiHanCao/exps/HyperspectralTokenizer/data/EarthView/hyper_images/satellogic",
    #     output_file="data/EarthView/hyper_images/satellogic/merged.tar",
    # )
    # exit(0)

    # list_untared_grouped(
    #     "/Data4/cao/ZiHanCao/exps/HyperspectralTokenizer/data/BigEarthNet_S2/conditions/tmp"
    # )
    # exit(0)

    import os

    import braceexpand

    # path = "data/EarthView/data/train-{00349..00606}-of-00607.parquet"
    path = "data/EarthView/satellogic/train-{02135..04000}-of-07863.parquet"
    paths = list(braceexpand.braceexpand(path))

    reading_fns = {
        "neon": reading_parquet_file_neon,
        "satellogic": reading_parquet_file_satellogic,
    }

    # parquet_to_webdataset(
    #     paths,
    #     "data/EarthView/hyper_images2/neon-%04d.tar",
    #     remove_file=False,
    #     dataset_name="neon",
    # )

    parquet_to_webdataset_async(
        paths,
        output_root="data/EarthView/hyper_images/satellogic/shard2",
        n_threads=3,  # type: ignore
        dataset_name="satellogic",
        to_wds=False,
    )

    # parquet_to_webdataset_async_mp_tp(
    #     paths,
    #     output_root="data/EarthView/hyper_images/statelogic",
    #     n_workers=4,
    # )
