import io
import tarfile
from bz2 import compress
from pathlib import Path

import braceexpand
import numpy as np
import tifffile
import webdataset as wds
from tqdm import tqdm

from src.data.codecs import tiff_codec_io

# --- 新增：波段选择和归一化配置 ---
# 要移除的无效波段 (0-indexed)
invalid_channels = [
    126,
    127,
    128,
    129,
    130,
    131,
    132,
    133,
    134,
    135,
    136,
    137,
    138,
    139,
    140,
    160,
    161,
    162,
    163,
    164,
    165,
    166,
]
# 创建有效波段的索引列表 (0-indexed)
valid_channels_indices = [c for c in range(224) if c not in invalid_channels]

# 裁剪和归一化的值
minimum_value = 0
maximum_value = 10000
# --- 配置结束 ---


classes = [
    "QL_PIXELMASK.TIF",
    "QL_QUALITY_CIRRUS.TIF",
    "QL_QUALITY_CLASSES.TIF",
    "QL_QUALITY_CLOUDSHADOW.TIF",
    "QL_QUALITY_CLOUD.TIF",
    "QL_QUALITY_HAZE.TIF",
    "QL_QUALITY_SNOW.TIF",
    "QL_QUALITY_TESTFLAGS.TIF",
    "QL_SWIR.TIF",
    "QL_VNIR.TIF",
    "SPECTRAL_IMAGE.TIF",
    "THUMBNAIL.jpg",
]


def targz_file_to_tarfile(targz_file_path: str, tar_file_writer: wds.ShardWriter, use_tbar=False):
    file = tarfile.open(targz_file_path, mode="r:gz")
    if use_tbar:
        total = len(file.getmembers())
        tbar = tqdm(file, total=total)
        print(f"Processing {targz_file_path} with {total} members...")
    else:
        tbar = file

    # Iterate through each member in the tar.gz file
    for member in tbar:
        name = member.name
        stem = Path(name).stem.replace("-SPECTRAL_IMAGE", "")
        is_spectral_image = name.endswith("SPECTRAL_IMAGE.TIF")
        if member.isfile() and (is_spectral_image):
            s = f"Processing {member.name}"
            if use_tbar:
                tbar.set_description(s)
            else:
                print(s)

            # Extract the file content
            file_content = file.extractfile(member)
            if file_content is None:
                print(f"Failed to extract {member.name}")
                continue
            file_content = file_content.read()

            buf = io.BytesIO(file_content)
            img = tifffile.imread(buf)
            assert img.ndim == 3, f"Expected 3D image, got {img.ndim}D"
            assert img.shape[0] == 224, f"Expected 224 channels, got {img.shape[0]}"

            # --- 新增处理步骤 ---
            # 1. 选择有效波段
            img = img[valid_channels_indices, :, :]

            # 2. 裁剪数据以消除不确定性
            img = np.clip(img, a_min=minimum_value, a_max=maximum_value)

            # # 3. Min-max 归一化
            # img = (img - minimum_value) / (maximum_value - minimum_value)
            # img = img.astype(np.float32)

            # --- 处理步骤结束 ---

            # compress
            img = img.transpose(1, 2, 0)  # Change to HWC format
            compressed_buf = tiff_codec_io(
                img,
                compression="jpeg2000",
                compression_args={"level": 80, "reversible": False},
            )

            # Write to the shard writer
            tar_file_writer.write(
                {
                    "__key__": stem,
                    "img.tiff": compressed_buf,
                }
            )

    file.close()
    print(f"Completed processing {targz_file_path}\n")
    print("-" * 60)


def main():
    # Create a shard writer
    num_valid_channels = len(valid_channels_indices)
    output_path = f"data/hyspecnet11k/hyper_images/hyspecnet11k_{num_valid_channels}_bands_px_128_%04d.tar"
    # mkdir
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tar_file_writer = wds.ShardWriter(output_path, maxcount=1000)

    path = "/HardDisk/ZiHanCao/datasets/Multispectral-HyperSepcNet11k/tmp/hyspecnet-11k-{01..10}.tar.gz"
    path = list(braceexpand.braceexpand(path))
    print(f"Found {len(path)} tar.gz files to process: {path}")

    for path in path:
        # Process the tar.gz file
        targz_file_to_tarfile(path, tar_file_writer)


if __name__ == "__main__":
    main()
    print("All files processed successfully.")
