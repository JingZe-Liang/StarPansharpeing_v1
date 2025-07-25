# unzip mat file into tiff file into tar file
import io
import os
import tarfile
import zipfile
from pathlib import Path

import webdataset as wds
from natsort import natsorted
from scipy.io import loadmat
from tqdm import tqdm

from src.data.codecs import tiff_codec_io


def targz_file_to_tarfile(
    tar_gz_file_path: str, tar_file_writer: wds.ShardWriter, use_tbar=True
):
    file = tarfile.open(tar_gz_file_path, "r:gz")
    if use_tbar:
        tbar = tqdm(file)
    else:
        tbar = file

    for member in tbar:
        if (
            member.isfile()
            and member.name.endswith(".tiff")
            or member.name.endswith(".tif")
        ):
            string = f"Processing {member.name}, "
            if use_tbar:
                tbar.set_description(string)
            else:
                print(string)

            # Extract the file content
            file_content = file.extractfile(member).read()

            # Write to the shard writer
            # tar_file_writer.write(
            #     {
            #         "__key__": "-".join(os.path.split(member.name)).replace(".tiff", "").replace(".tif", ""),
            #         "img.tiff": file_content,
            #     }
            # )


def zipfile_to_tarfile(zip_file_path, tar_file_writer: wds.ShardWriter, use_tbar=True):
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        if use_tbar:
            tbar = tqdm(zip_ref.infolist())
        else:
            tbar = zip_ref.infolist()

        for info in tbar:
            file = info.filename
            string = f"Processing {file}, "
            if file.endswith(".mat"):
                mat_file_bytes = zip_ref.read(file)
                try:
                    mat_data = loadmat(io.BytesIO(mat_file_bytes))
                except ImportError:
                    print(f"Raw bytes of {info.filename}: {len(mat_file_bytes)} bytes")
                    continue

                assert "img" in mat_data, f"Key 'img' not found in {file}"
                img = mat_data["img"]
                string += str(img.shape)

                if use_tbar:
                    tbar.set_description(string)
                else:
                    print(string)

                # save into io buffer
                # img: [c, h, w]
                img = img.transpose(1, 2, 0)
                # print(img.shape)
                tar_file_writer.write(
                    {
                        "__key__": "-".join(os.path.split(file)).replace(".mat", ""),
                        "img.tiff": tiff_codec_io(  # compression using jpeg2000
                            img,  # [h, w, c]
                            compression="jpeg2000",
                            compression_args={"reversible": False, "level": 85},
                        ),
                    }
                )


def move_file():
    import os
    import shutil
    from pathlib import Path

    path = Path("data/HyperGlobal/hyper_images/hypers")
    for file in path.iterdir():
        assert file.is_file(), f"{file} is not a file"
        if file.name.startswith("EO1"):
            new_path = path.parent / "EO1" / file.name
        elif file.name.startswith("GF5"):
            new_path = path.parent / "GF5" / file.name
        else:
            raise ValueError(f"Unknown file type: {file}")

        shutil.move(file, new_path)
        print(f"Moved {file} to {new_path}")


# if __name__ == "__main__":
#     # move_file()

#     from pathlib import Path

#     from braceexpand import braceexpand

#     shard_pattern = (
#         "data/HyperGlobal/hyper_images2/HyperGlobal-EO1-xx_bands-px_64_%04d.tar"
#     )

#     Path(shard_pattern).parent.mkdir(parents=True, exist_ok=True)
#     sink = wds.ShardWriter(
#         shard_pattern,
#         maxsize=8 * 1024 * 1024 * 1024,  # 8G
#     )
#     _zipfiles = [
#         "/HardDisk/ZiHanCao/datasets/HyperGlobal-450k/EO1-part{1..6}.zip",
#         # "/HardDisk/ZiHanCao/datasets/HyperGlobal-450k/GF5-part{1..5}.zip",
#     ]
#     zipfiles = []
#     for zipf in _zipfiles:
#         zipfiles.extend(braceexpand(zipf))

#     for zip_file_path in zipfiles:
#         print(f"Processing {zip_file_path}")
#         zipfile_to_tarfile(zip_file_path, sink)

#     sink.close()


import concurrent.futures
import threading

from rich.console import Console
from rich.progress import Progress, TaskID


def process_mat_file(
    zip_ref: zipfile.ZipFile,
    info,
    tar_file_writer: wds.ShardWriter,
    progress: Progress,
    task_id: TaskID,
):
    """处理单个 mat 文件"""
    file = info.filename

    if not file.endswith(".mat"):
        return

    try:
        mat_file_bytes = zip_ref.read(file)
        try:
            mat_data = loadmat(io.BytesIO(mat_file_bytes))
        except ImportError:
            progress.console.print(
                f"Raw bytes of {info.filename}: {len(mat_file_bytes)} bytes"
            )
            return

        assert "img" in mat_data, f"Key 'img' not found in {file}"
        img = mat_data["img"]

        # 更新进度条描述
        progress.update(
            task_id,
            description=f"Processing file: {zip_ref.filename}, name: {os.path.basename(file)}, shape: {tuple(img.shape)}",
        )

        # 处理图像数据
        img = img.transpose(1, 2, 0)  # [c, h, w] -> [h, w, c]

        # 编码图像
        encoded_img = tiff_codec_io(
            img,
            compression="jpeg2000",
            compression_args={"reversible": False, "level": 85},
        )

        # 直接写入（无需锁，因为每个zip对应独立的tar）
        tar_file_writer.write(
            {
                "__key__": "-".join(os.path.split(file)).replace(".mat", ""),
                "img.tiff": encoded_img,
            }
        )

        # 更新进度
        progress.advance(task_id)

    except Exception as e:
        progress.console.print(f"Error processing {file}: {e}")


def process_single_zip(
    zip_file_path: str, shard_pattern: str, progress: Progress, max_workers=4
):
    """处理单个zip文件，创建独立的tar文件"""

    # 为每个zip文件创建独立的ShardWriter
    zip_name = os.path.splitext(os.path.basename(zip_file_path))[0]
    zip_shard_pattern = shard_pattern.replace("xx", zip_name)

    Path(zip_shard_pattern).parent.mkdir(parents=True, exist_ok=True)
    sink = wds.ShardWriter(
        zip_shard_pattern,
        maxsize=8 * 1024 * 1024 * 1024,  # 8G
    )

    try:
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            # 获取所有 mat 文件
            mat_files = [
                info for info in zip_ref.infolist() if info.filename.endswith(".mat")
            ]

            if not mat_files:
                return

            # 为当前 zip 文件创建一个进度条
            task_id = progress.add_task(
                f"[cyan]{os.path.basename(zip_file_path)}",
                total=len(mat_files),
            )

            # 使用线程池处理文件
            # with concurrent.futures.ThreadPoolExecutor(
            #     max_workers=max_workers
            # ) as executor:
            #     # 提交所有任务
            #     futures = []
            #     for info in mat_files:
            #         future = executor.submit(
            #             process_mat_file,
            #             zip_ref,
            #             info,
            #             sink,
            #             progress,
            #             task_id,
            #         )
            #         futures.append(future)

            #     # 等待所有任务完成
            #     concurrent.futures.wait(futures)

            for info in mat_files:
                process_mat_file(zip_ref, info, sink, progress, task_id)

            progress.update(
                task_id,
                description=f"[green]✓ {os.path.basename(zip_file_path)}",
            )
    finally:
        sink.close()


def rename_shards(
    path,
    base_name="data/HyperGlobal/hyper_images2/HyperGlobal-GF5-bands-px_64_%04d.tar",
):
    shards = list(Path(path).glob("*.tar"))
    shards = natsorted(shards)
    index = 0
    for shard in shards:
        new_name = base_name % index
        shard.rename(new_name)
        index += 1
        print(f"Renamed {shard} to {new_name}")


# 更新主函数以支持多个 zip 文件的并行处理
if __name__ == "__main__":
    # rename_shards(
    #     "data/HyperGlobal/hyper_images2",
    # )
    # exit(0)

    from pathlib import Path

    from braceexpand import braceexpand

    shard_pattern = "data/HyperGlobal/hyper_images2/HyperGlobal-xx-bands-px_64_%04d.tar"

    _zipfiles = [
        # "/HardDisk/ZiHanCao/datasets/HyperGlobal-450k/EO1-part{1..6}.zip",
        "/HardDisk/ZiHanCao/datasets/HyperGlobal-450k/GF5-part{1..5}.zip",
    ]
    zipfiles = []
    for zipf in _zipfiles:
        zipfiles.extend(braceexpand(zipf))

    # 创建全局进度显示
    with Progress(console=Console()) as progress:
        # 总体进度条
        main_task = progress.add_task(
            "[bold blue]Overall Progress", total=len(zipfiles)
        )

        # 使用线程池并行处理所有zip文件
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(zipfiles)
        ) as executor:
            # 提交所有zip文件处理任务
            futures = []
            for zip_file_path in zipfiles:
                future = executor.submit(
                    process_single_zip,
                    zip_file_path,
                    shard_pattern,
                    progress,
                    max_workers=6,  # 每个zip内部的并发数
                )
                futures.append(future)

            # 等待所有任务完成，并更新总进度
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                    progress.advance(main_task)
                except Exception as e:
                    progress.console.print(f"Error processing zip file: {e}")
                    progress.advance(main_task)
