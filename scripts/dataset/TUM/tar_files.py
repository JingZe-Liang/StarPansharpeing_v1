import os
import tarfile
from pathlib import Path

import tifffile


def read_img(path: str):
    img = tifffile.imread(path)

    return img


def tar_files(input_paths, output_file="tarfile_%04d.tar", max_size: int = 4 * 1024 * 1024 * 1024):
    """
    Create tar files from all .tif files in the input paths.

    Args:
        input_paths (str or list): Path(s) to directories or files containing .tif files to be tarred.
        output_file (str): Output tar file pattern with %d placeholder for numbering. Default is "tarfile_%04d.tar".
        max_size (int): Maximum size of each tar file in bytes. Default is 4GB.
    """
    if isinstance(input_paths, str):
        input_paths = [Path(input_paths)]

    tar_index = 0
    current_tar = None
    current_size = 0

    try:
        for tif_file in input_paths:
            tif_file: Path
            file_size = tif_file.stat().st_size

            # 检查是否需要创建新的 tar 文件
            if current_tar is None or (current_size + file_size > max_size):
                # 关闭当前 tar 文件
                if current_tar is not None:
                    current_tar.close()
                    print(f"Completed {output_file % tar_index} (size: {current_size / (1024 * 1024 * 1024):.2f} GB)")

                # 创建新的 tar 文件
                tar_index += 1
                current_tar_name = output_file % (tar_index - 1)
                current_tar = tarfile.open(current_tar_name, "w")
                current_size = 0
                print(f"Creating {current_tar_name}...")

            # 添加文件到当前 tar
            arcname = str(tif_file.relative_to(tif_file.parent.parent)).replace("/", "_")

            # 在扩展名前添加 .img
            if arcname.endswith((".tif", ".tiff")):
                name_parts = arcname.rsplit(".", 1)
                arcname = f"{name_parts[0]}.img.{name_parts[1]}"

            current_tar.add(tif_file, arcname=arcname)
            current_size += file_size

            print(
                f"Added {arcname} ({file_size / (1024 * 1024):.2f} MB), tar file size: {current_size / (1024 * 1024):.2f} MB"
            )

    finally:
        # 关闭最后一个 tar 文件
        if current_tar is not None:
            current_tar.close()
            print(f"Completed {output_file % (tar_index - 1)} (size: {current_size / (1024 * 1024 * 1024):.2f} GB)")

    print(f"Archive process completed. Created {tar_index} tar file(s).")


if __name__ == "__main__":
    from itertools import chain

    tif_files1 = Path("/HardDisk/ZiHanCao/datasets/Multispectral-SegMunich_Change_Detection/train/img").glob("*.tif")
    tif_files2 = Path("/HardDisk/ZiHanCao/datasets/Multispectral-SegMunich_Change_Detection/val/label").glob("*.tif")
    tif_files = chain(tif_files1, tif_files2)

    # 示例用法
    tar_files(
        input_paths=tif_files,
        output_file="/Data4/cao/ZiHanCao/exps/HyperspectralTokenizer/data/TUM_128/hyper_images/TUM_128_data_%04d.tar",
        max_size=2 * 1024 * 1024 * 1024,  # 2GB
    )

    # for tif_file in tif_files:
    #     img = read_img(tif_file)
    #     print(f"Read image {tif_file} with shape {img.shape} and dtype {img.dtype}")
