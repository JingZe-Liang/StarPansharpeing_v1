import os
import tarfile
from pathlib import Path


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

    # 收集所有 .tif 文件
    # tif_files = []
    # for input_path in input_paths:
    #     input_path = pathlib.Path(input_path)
    #     if input_path.is_file() and input_path.suffix.lower() == ".tif":
    #         tif_files.append(input_path)
    #     elif input_path.is_dir():
    #         # 递归查找所有 .tif 文件
    #         tif_files.extend(input_path.rglob("*.tif"))
    #         tif_files.extend(input_path.rglob("*.TIF"))

    # tif_files = list(input_paths)

    # if not tif_files:
    #     print("No .tif files found in the specified paths.")
    #     return

    # print(f"Found {len(tif_files)} .tif files to archive.")

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

            current_tar.add(tif_file, arcname=arcname)
            current_size += file_size

            print(f"Added {tif_file.name} ({file_size / (1024 * 1024):.2f} MB)")

    finally:
        # 关闭最后一个 tar 文件
        if current_tar is not None:
            current_tar.close()
            print(f"Completed {output_file % (tar_index - 1)} (size: {current_size / (1024 * 1024 * 1024):.2f} GB)")

    print(f"Archive process completed. Created {tar_index} tar file(s).")


if __name__ == "__main__":
    path = "/HardDisk/ZiHanCao/datasets/Multispectral-fMoW-Sentinel/fmow-sentinel/"
    img_paths = Path(path).rglob("**/*.tif")
    # 示例用法
    tar_files(
        input_paths=img_paths,
        output_file="fMoW_data_%04d.tar",
        max_size=2 * 1024 * 1024 * 1024,  # 2GB
    )
