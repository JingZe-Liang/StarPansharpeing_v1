import uuid
from typing import Any

import webdataset as wds
import wids
import wids.wids_decode
from PIL import Image

from src.data.codecs import img_decode_io, string_decode_io, tiff_decode_io
from src.utilities.config_utils import log_print


def _add_random_id(sample):
    sample["random_id"] = str(uuid.uuid4())
    return sample


def wids_image_decode(sample: dict[str, Any], read_caption=False, read_name=False):
    if ".img.tiff" in sample or ".img.tif" in sample:
        try:
            sample["img"] = tiff_decode_io(sample[".img.tiff"].getvalue())
            del sample[".img.tiff"]  # Remove the original BytesIO object
        except KeyError:
            sample["img"] = tiff_decode_io(sample[".img.tif"].getvalue())
            del sample[".img.tif"]  # Remove the original BytesIO object
        finally:
            log_print(f"Decode image error")
    if ".img_content" in sample:
        sample["img"] = img_decode_io(sample[".img_content"].getvalue())
        del sample[".img_content"]
    if ".caption" in sample:
        if read_caption:
            sample["caption"] = string_decode_io(sample[".caption"].getvalue())
        del sample[".caption"]
    if ".img_name" in sample:
        if read_name:
            sample["img_name"] = string_decode_io(sample[".img_name"].getvalue())
        del sample[".img_name"]

    # remove None key/values
    _key_to_del = []
    for k, v in sample.items():
        if v is None:
            _key_to_del.append(k)
    for k in _key_to_del:
        del sample[k]

    return sample


def test_fn(img):
    return img


def make_loader(path):
    ds = wids.ShardListDataset(
        path,
        cache_dir=None,
        transformations=wids_image_decode,
        localname=lambda fname: fname,
    )
    sampler = wids.DistributedChunkedSampler(ds, num_replicas=1, rank=0, shuffle=False)
    loader = wds.WebLoader(ds, sampler=sampler, batch_size=1, num_workers=0)
    loader = loader.map_dict(img=test_fn)

    return ds, loader


def nsample_in_tar(tar_file: str):
    from wids.wids_mmtar import MMIndexedTar

    mmtar = MMIndexedTar(tar_file)
    print(mmtar.names(), len(mmtar.names()))
    print(len(mmtar))


if __name__ == "__main__":
    dataset, loader = make_loader("shardindex.json")

    for sample in loader:
        pass

    exit(0)

    from pathlib import Path

    import pandas as pd
    from tqdm import tqdm

    print("ready to read the image shapes")

    parquet_file = Path("image_shapes.parquet")
    batch_size = 1000
    image_shapes_data = []

    for sample in (tbar := tqdm(loader, total=len(dataset))):
        shape = sample["img"].shape[1:]
        sample_d = {
            "height": shape[0],
            "width": shape[1],
            "channels": shape[2] if len(shape) > 2 else 1,
            "index": sample["__index__"].item(),
            "shard_index": sample["__shardindex__"].item(),
        }

        image_shapes_data.append(sample_d)

        # 每处理batch_size个样本就追加写入文件
        if len(image_shapes_data) >= batch_size:
            df = pd.DataFrame(image_shapes_data)

            # 如果文件不存在，创建新文件；否则追加
            if not parquet_file.exists():
                df.to_parquet(parquet_file, index=False, compression="snappy")
            else:
                # 读取现有数据并追加
                existing_df = pd.read_parquet(parquet_file)
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                combined_df.to_parquet(parquet_file, index=False, compression="snappy")

            image_shapes_data = []  # 清空缓存

    # 处理剩余的数据
    if image_shapes_data:
        df = pd.DataFrame(image_shapes_data)
        if not parquet_file.exists():
            df.to_parquet(parquet_file, index=False, compression="snappy")
        else:
            existing_df = pd.read_parquet(parquet_file)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df.to_parquet(parquet_file, index=False, compression="snappy")

    print(f"Data saved to {parquet_file}")

    # import random

    # index = list(range(2000))

    # print("indexing sequentially")
    # import time

    # start = time.perf_counter()
    # for i in tqdm(index):
    #     # print(dataset[i]["img"].shape[:2])
    #     dataset[i]
    # end = time.perf_counter()
    # print(f"Time taken for sequential indexing: {end - start:.2f} seconds")

    # random.shuffle(index)
    # print("indexing randomly")
    # start = time.perf_counter()
    # for i in tqdm(index):
    #     # print(dataset[i]["img"].shape[:2])
    #     dataset[i]
    # end = time.perf_counter()
    # print(f"Time taken for random indexing: {end - start:.2f} seconds")
