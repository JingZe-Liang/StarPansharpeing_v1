import os
import tarfile
import time
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Callable

from natsort import natsorted


def extract_member_from_tar(
    tar_reader: tarfile.TarFile,
    member: tarfile.TarInfo,
    extract_path: str | Path | None = None,
):
    """Extract a specific member from tar archive.

    Args:
        tar_reader: TarFile object to extract from
        member_name: Name of the member to extract
        extract_path: Path where the member should be extracted

    Returns:
        bool: True if extraction was successful, False otherwise
    """

    if member:
        if extract_path is not None:
            extract_path = Path(extract_path)
            extract_path.mkdir(parents=True, exist_ok=True)
            tar_reader.extract(member, path=extract_path)
            print(f"Extracted {member.name} to {extract_path}")
        else:
            return get_content_from_member(tar_reader, member)
    else:
        print(f"Member {member_name} not found in tar file")
        return False


def get_content_from_member(tar_reader: tarfile.TarFile, member: tarfile.TarInfo):
    if member.isfile():
        content = tar_reader.extractfile(member)
        if content:
            data = content.read()
            content.close()
            return data
        else:
            print(f"Could not read content of {member.name}")
            return None
    else:
        print(f"Member {member.name} is not a file")
        return None


def get_tar_member_iter(tar_reader: tarfile.TarFile, sort=True):
    if not sort:
        while True:
            member = tar_reader.next()
            if member is None:
                break
            if member.isfile():
                yield member
    else:
        members = tar_reader.getmembers()
        members = [m for m in members if m.isfile()]
        members = natsorted(members, key=lambda x: x.name)
        for member in members:
            yield member


def get_member_content(tar_reader: tarfile.TarFile, member_name: str) -> bytes | None:
    """Get the content of a specific member from tar archive.

    Args:
        tar_reader: TarFile object to read from
        member_name: Name of the member to read

    Returns:
        bytes: Content of the member if found, None otherwise
    """
    member_content = tar_reader.extractfile(member_name)
    if member_content:
        content = member_content.read()
        member_content.close()
        return content
    else:
        print(f"Member {member_name} not found in tar file")
        return None


def create_tar_info(member_name: str, member_content: bytes, other_kwargs: dict = {}):
    tarinfo = tarfile.TarInfo(name=member_name)
    tarinfo.size = len(member_content)
    tarinfo.mtime = other_kwargs.pop("mtime", time.time())
    tarinfo.mode = other_kwargs.pop("mode", 0o644)
    tarinfo.uid = other_kwargs.pop("uid", os.getuid())
    tarinfo.gid = other_kwargs.pop("gid", os.getgid())
    tarinfo.type = other_kwargs.pop("type", tarfile.REGTYPE)
    return tarinfo


def write_tar_file(
    tar_writer: tarfile.TarFile,
    file: tarfile.TarInfo | str | Path,
    content: bytes,
):
    if isinstance(file, tarfile.TarInfo):
        if content is not None:
            tar_writer.addfile(file, BytesIO(content))
    elif isinstance(file, (str, Path)):
        tar_writer.add(file)
    else:
        raise ValueError(f"Invalid file type: {type(file)}")


def extract_tar_files_into(
    tar_file: str | Path,
    dest: str | Path | tarfile.TarFile,
    member_filters: Callable[[tarfile.TarInfo], bool] | None = None,
):
    tar_file = Path(tar_file)
    if isinstance(dest, (str, Path)):
        dest = Path(dest)
        if dest.is_dir() and not dest.exists():
            print(f"Dest {dest} is a directory, extract all file in tar file into it")
        elif dest.with_suffix(".tar"):
            print(f"Dest {dest} is a tar file, create it and copy files into it")
            dest = tarfile.TarFile(dest, "w")

    tar_reader = tarfile.open(tar_file, "r")
    try:
        for member in get_tar_member_iter(tar_reader):
            extract_path = None if isinstance(dest, tarfile.TarFile) else dest
            content = extract_member_from_tar(tar_reader, member, extract_path)
            assert isinstance(content, bytes) or content is None, "Content should be bytes or None"

            if content is None:
                continue
            if member_filters is not None and not member_filters(member):
                print(f"Skip member {member.name} due to filter")
                continue

            if isinstance(dest, tarfile.TarFile):
                new_member = create_tar_info(member.name, content)
                write_tar_file(dest, new_member, content)
                print(f"Written member {member.name} to tar file {dest.name}")
            else:
                print(f"Extracted member {member.name} to directory {dest}")

    except Exception as e:
        print(f"Error occurred: {e}")
        tar_reader.close()
        if isinstance(dest, tarfile.TarFile):
            dest.close()


# Filter functions
def filter_by_index(member: tarfile.TarInfo):
    stem = Path(member.name).stem
    index = stem.split(".")[0]
    assert index.isdigit(), f"Index {index} is not a digit"
    if int(index) >= 60:
        return True
    return False


def filter_in_index(member: tarfile.TarInfo, index_need: list):
    stem = Path(member.name).stem
    index = stem.split(".")[0]
    assert index.isdigit(), f"Index {index} is not a digit"
    if int(index) in index_need:
        return True
    return False


if __name__ == "__main__":
    import numpy as np

    length = 80  # WV4: 500, QB: 250, WV2: 250, IKONOS: 100, WV3: 80
    full_list = list(range(length))
    perm = np.random.permutation(length)
    train_indices = perm[int(length * 0.2) :].tolist()
    test_indices = perm[: int(length * 0.2)].tolist()
    print("train_indices", train_indices)
    print("test_indices", test_indices)
    extract_tar_files_into(
        "data/Downstreams/PanCollectionV2/WV3/pansharpening_reduced/Pansharpening_WV3.tar",
        dest="data/Downstreams/PanCollectionV2/WV3/pansharpening_reduced/Pansharping_WV3_train.tar",
        member_filters=partial(filter_in_index, index_need=train_indices),
    )
    extract_tar_files_into(
        "data/Downstreams/PanCollectionV2/WV3/pansharpening_reduced/Pansharpening_WV3.tar",
        dest="data/Downstreams/PanCollectionV2/WV3/pansharpening_reduced/Pansharping_WV3_val.tar",
        member_filters=partial(filter_in_index, index_need=test_indices),
    )

    # extract_tar_files_into(
    #     "data/Downstreams/PanCollectionV2/WV4/pansharpening_reduced/Pansharpening_WV4.tar",
    #     dest="data/Downstreams/PanCollectionV2/WV4/pansharpening_reduced/Pansharpening_WV4_train.tar",
    #     member_filters=partial(filter_in_index, index_need=train_indices),
    # )
    # extract_tar_files_into(
    #     "data/Downstreams/PanCollectionV2/WV4/pansharpening_reduced/Pansharpening_WV4.tar",
    #     dest="data/Downstreams/PanCollectionV2/WV4/pansharpening_reduced/Pansharpening_WV4_val.tar",
    #     member_filters=partial(filter_in_index, index_need=test_indices),
    # )

    # extract_tar_files_into(
    #     "data/Downstreams/PanCollectionV2/QB/pansharpening_reduced/Pansharpening_QB.tar",
    #     dest="data/Downstreams/PanCollectionV2/QB/pansharpening_reduced/Pansharpening_QB_train.tar",
    #     member_filters=partial(filter_in_index, index_need=train_indices),
    # )
    # extract_tar_files_into(
    #     "data/Downstreams/PanCollectionV2/QB/pansharpening_reduced/Pansharpening_QB.tar",
    #     dest="data/Downstreams/PanCollectionV2/QB/pansharpening_reduced/Pansharpening_QB_val.tar",
    #     member_filters=partial(filter_in_index, index_need=test_indices),
    # )

    # extract_tar_files_into(
    #     "data/Downstreams/PanCollectionV2/WV2/pansharpening_reduced/Pansharpening_WV2.tar",
    #     dest="data/Downstreams/PanCollectionV2/WV2/pansharpening_reduced/Pansharpening_WV2_train.tar",
    #     member_filters=partial(filter_in_index, index_need=train_indices),
    # )
    # extract_tar_files_into(
    #     "data/Downstreams/PanCollectionV2/WV2/pansharpening_reduced/Pansharpening_WV2.tar",
    #     dest="data/Downstreams/PanCollectionV2/WV2/pansharpening_reduced/Pansharpening_WV2_val.tar",
    #     member_filters=partial(filter_in_index, index_need=test_indices),
    # )

    # extract_tar_files_into(
    #     "data/Downstreams/PanCollectionV2/IKONOS/pansharpening_reduced/Pansharpening_IKONOS.tar",
    #     dest="data/Downstreams/PanCollectionV2/IKONOS/pansharpening_reduced/Pansharpening_IKONOS_train.tar",
    #     member_filters=partial(filter_in_index, index_need=train_indices),
    # )
    # extract_tar_files_into(
    #     "data/Downstreams/PanCollectionV2/IKONOS/pansharpening_reduced/Pansharpening_IKONOS.tar",
    #     dest="data/Downstreams/PanCollectionV2/IKONOS/pansharpening_reduced/Pansharpening_IKONOS_val.tar",
    #     member_filters=partial(filter_in_index, index_need=test_indices),
    # )
