import os
from typing import Dict, List

import pyarrow as pa
import pyarrow.parquet as pq
import tifffile
from tqdm import tqdm


def create_metadata_only_parquet(
    folder_path: str, output_file: str, extra_metadata: Dict[str, List] = None
) -> None:
    """仅存储TIFF文件元信息到Parquet

    Args:
        folder_path: 包含.tiff文件的文件夹
        output_file: 输出Parquet路径
        extra_metadata: 额外需要存储的元数据 {列名: 值列表}
    """
    tiff_files = [
        f for f in os.listdir(folder_path) if f.lower().endswith((".tiff", ".tif"))
    ]

    # 基础元数据结构
    data = {
        "file_name": [],
        "file_path": [],
        "shape": [],  # 存储形状元组 (h,w,c)
        "dtype": [],  # 存储数据类型
        "img": [],  # 存储图像数据
    }

    # 添加额外元数据列
    if extra_metadata:
        data.update(extra_metadata)

    # 填充数据
    for filename in tqdm(tiff_files[:20]):
        tiff_img = tifffile.imread(os.path.join(folder_path, filename))
        full_path = os.path.join(folder_path, filename)
        data["file_name"].append(filename)
        data["file_path"].append(full_path)
        data["shape"].append(None)  # 延迟加载时获取
        data["dtype"].append(None)  # 延迟加载时获取
        data["img"].append(tiff_img.tolist())  # 将numpy数组转换为列表

    # 定义Schema
    schema = pa.schema(
        [
            ("file_name", pa.string()),
            ("file_path", pa.string()),
            ("shape", pa.list_(pa.int32())),
            ("dtype", pa.string()),
            ("img", pa.list_(pa.list_(pa.list_(pa.uint16())))),
        ]
    )

    # 写入Parquet
    table = pa.Table.from_pydict(data, schema=schema)
    pq.write_table(table, output_file, compression="ZSTD")


# 使用示例
create_metadata_only_parquet(
    folder_path="/HardDisk/ZiHanCao/datasets/Multispectral-DFC2019/Track2/MSI",
    output_file="/HardDisk/ZiHanCao/datasets/Multispectral-DFC2019/DFC_2019.parquet",
)

import numpy as np
import tifffile
from datasets import Array3D, Dataset, Features, Sequence, Value

# 或者如果图像尺寸完全动态，可以改为使用Sequence
features = Features(
    {
        "file_name": Value("string"),
        "file_path": Value("string"),
        "shape": Sequence(Value("int32")),
        "dtype": Value("string"),
        "image": Sequence(Sequence(Sequence(Value("float32")))),  # 完全动态的3D数组
    }
)


# 从Parquet创建Dataset（不立即加载图像）
dataset = Dataset.from_parquet(
    "/HardDisk/ZiHanCao/datasets/Multispectral-DFC2019/DFC_2019.parquet",
    features=features,
    streaming=True,  # 启用流式加载
)


# 自定义加载函数
def load_image_on_demand(example):
    """按需加载TIFF文件"""
    with tifffile.TiffFile(example["file_path"]) as tif:
        arr = tif.asarray()
        if arr.ndim == 3 and arr.shape[0] < arr.shape[-1]:  # (c,h,w) -> (h,w,c)
            arr = np.moveaxis(arr, 0, -1)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "image": arr.astype("float32"),  # 自动转换为float32
    }


# 应用延迟加载（使用时才会真正读取文件）
# dataset.set_transform(load_image_on_demand)
# dataset.cast_column("image", Array3D())  # 将图像列转换为Array3D类型
dataset = dataset.map(
    load_image_on_demand,
    batched=False,
)
# dataset = dataset.batch(
#     batch_size=10,
# )
dataset = dataset.with_format("torch")

# import torch.utils.data as data

# data.DataLoader(
#     dataset,
#     batch_size=10,
#     num_workers=4,
#     pin_memory=True,
# )

# 现在可以正常访问数据
# print(dataset[0]["image"].shape)  # 输出 (height, width, channels)
# print(next(iter(dataset)))

for d in dataset.iter(batch_size=10):
    print(d["image"].shape)  # 输出 (batch_size, height, width, channels)
