"""
Tar caption files in a directory into a single tar file.
Filter the old caption tar file to only keep caption files.

Author: Zihan Cao
Date: 2025/10/07
Copyright: (c) 2025 Zihan Cao, UESTC, Mathematical school. All Rights Reserved.
"""

from pathlib import Path
from tarfile import TarFile

from loguru import logger
from tqdm import tqdm

from src.data.tar_utils import (
    get_content_from_member,
    get_tar_member_iter,
    write_tar_file,
)


def tar_captions():
    img_ds_path: str = "data/RemoteSAM270k/RemoteSAM-270K/RemoteSAM270K.tar"
    caption_dir: str = "data/RemoteSAM270k/RemoteSAM-270K/captions"
    caption_tar_save_path: str = "data/RemoteSAM270k/RemoteSAM-270K/captions.tar"
    caption_ext: str = ".caption.jsonl"

    Path(caption_tar_save_path).parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving captions to {caption_tar_save_path}")

    caption_tar_writer = TarFile(name=caption_tar_save_path, mode="w")

    with TarFile(img_ds_path, mode="r") as ds_reader:
        for member in tqdm(
            get_tar_member_iter(ds_reader, sort=False), desc="Tar-ing captions ..."
        ):
            img_name = member.name
            caption_file = (Path(caption_dir) / img_name).with_suffix(caption_ext)
            if not caption_file.exists():
                logger.warning(f"Caption file {caption_file} does not exist")
                continue

            write_tar_file(
                tar_writer=caption_tar_writer,
                file=caption_file,
            )

    caption_tar_writer.close()
    logger.success(f"Caption tar file saved to {caption_tar_save_path}")


def filter_only_captions():
    caption_tar_path: str = (
        "data/LoveDA/condition_captions/LoveDA-3_bands-px_1024-0000.tar"
    )
    new_caption_tar_path: str = (
        "data/LoveDA/condition_captions/LoveDA-3_bands-px_1024-0000_new.tar"
    )
    caption_ext: str = ".caption.json"

    caption_tar_reader = TarFile(name=caption_tar_path, mode="r")
    new_caption_tar_writer = TarFile(name=new_caption_tar_path, mode="w")

    is_caption_n = 0
    try:
        for member in (
            tbar := tqdm(
                get_tar_member_iter(caption_tar_reader, sort=False),
                desc="Filtering captions ...",
            )
        ):
            if member.name.endswith(caption_ext):
                write_tar_file(
                    tar_writer=new_caption_tar_writer,
                    file=member,
                    content=get_content_from_member(caption_tar_reader, member),
                )
                is_caption_n += 1
            tbar.set_postfix({"is_caption_n": is_caption_n})
    except Exception as e:
        logger.error(f"Error occurred: {e}")
    finally:
        caption_tar_reader.close()
        new_caption_tar_writer.close()

    logger.success(f"New caption tar file saved to {new_caption_tar_path}")


if __name__ == "__main__":
    # tar_captions()
    # filter_only_captions()

    captions_with_embeds = [
        # 'data/QuickBird/condition_captions/QuickBird-4_bands-px_256-MSI-0000.tar',
        # 'data/WorldView2/condition_captions/WorldView2-8_bands-px_256-MSI-0000.tar',
        "data/BigEarthNet_S2/condition_captions/BigEarthNet_data_0000.tar",
        "data/BigEarthNet_S2/condition_captions/BigEarthNet_data_0001.tar",
    ]

    new_captions = [p.replace(".tar", "_new.tar") for p in captions_with_embeds]
    filter_multiple_captions(captions_with_embeds, new_captions)
