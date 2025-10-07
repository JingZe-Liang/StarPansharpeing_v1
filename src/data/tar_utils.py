import io
import os
import re
import tarfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

import numpy as np
import PIL
import PIL.Image
import tifffile
import webdataset as wds
from natsort import natsorted
from tqdm import tqdm

from src.utilities.logging import log_print


@contextmanager
def tar_sink_manager():
    """
    Context manager for writing to a tar file.
    """

    total_sinks = {}

    def get_sink(name, tar_path):
        if name not in total_sinks:
            total_sinks[name] = wds.TarWriter(tar_path)
            log_print(f"Created new tar sink for {tar_path}")
        return total_sinks[tar_path]

    try:
        yield get_sink

    finally:
        for sink in total_sinks.values():
            sink.close()
            log_print(f"Closed tar sink for {sink.name}")


class TarSinkManager:
    total_sinks = {}

    def __init__(self, base_dir: str):
        self.base_dir = base_dir

    def get_sink(self, name: str, tar_rel_path: str):
        tar_path = os.path.join(self.base_dir, tar_rel_path)
        os.makedirs(os.path.dirname(tar_path), exist_ok=True)

        if name not in self.total_sinks:
            self.total_sinks[name] = wds.TarWriter(tar_path)
            log_print(f"Created new tar sink for {tar_path}")
        return self.total_sinks[name]

    def close_all(self):
        for name, sink in self.total_sinks.items():
            sink.close()
            log_print(f"Closed tar sink for {name}")
        self.total_sinks.clear()
        log_print("Closed all tar sinks.")


def remove_key_in_tar(tar_path: str, key: str):
    """
    Remove specific key(s) (supporting regex) from a tar file.
    The original tar will be replaced after filtering.
    """
    from tqdm import tqdm

    assert tar_path.endswith(".tar"), "The tar_path must end with .tar"

    # Compile the regex pattern
    pattern = re.compile(key)
    tmp_path = tar_path + ".tmp"

    with tarfile.open(tar_path, "r") as tar, tarfile.open(tmp_path, "w") as out_tar:
        for member in (tbar := tqdm(tar, desc="Filtering tar members")):
            if not pattern.search(member.name):
                out_tar.addfile(
                    member, tar.extractfile(member) if member.isfile() else None
                )
            else:
                tbar.set_description(f"Skipping {member.name}")
    # Replace original tar file

    ans = input("Press Enter to replace original tar file [y/n]: ")
    if ans.lower() != "y":
        print("Replacement cancelled.")
        return
    else:
        os.replace(tmp_path, tar_path)


def read_tar_filenames_safe(
    tar_path: str | None = None,
    tar: tarfile.TarFile | None = None,
    close_tar=False,
    safe_read_names=True,
    progress=True,
    check_file=True,
) -> list[str]:
    # if provide the tar, the pointer of tar will be moved to the end

    if tar_path is not None:
        if not os.path.exists(tar_path):
            return []
        close_tar = True  # force to close
        tar = tarfile.open(tar_path, "r")
    elif tar is None:
        raise ValueError("Either tar_path or tar must be provided.")

    successful_files = []

    not_safe_load_raised = False
    if not safe_read_names:
        try:
            successful_files = tar.getnames()
        except tarfile.TarError as e:
            not_safe_load_raised = True
            log_print(f"Failed to get names from tar: {e}", "warning")

    try:
        if not_safe_load_raised:
            assert tar_path is not None, "tar_path must be provided if tar is None."
            tar.close()
            tar = tarfile.open(tar_path, "r")  # seek the pointer to the start
        if progress:
            tbar = tqdm(unit="samples", desc="Reading tar files ...", leave=False)

        while True:
            try:
                member = tar.next()
                if member is None:
                    log_print("Finished extracting tar file or reached end of tar.")
                    break

                if member.isfile():
                    try:
                        file_data = tar.extractfile(member)
                        if file_data is not None:
                            if check_file:
                                _ = file_data.read(1)  # may raise error
                            successful_files.append(member.name)
                            tbar.update(1)
                    except Exception as e:
                        if progress:
                            tbar.clear()
                        log_print(
                            f"Failed to read file {member.name} in tar: {e}",
                            "warning",
                        )
                        if progress:
                            tbar.refresh()
                        continue

            except tarfile.TarError as e:
                if progress:
                    tbar.clear()
                log_print(
                    f"Failed to next file (head, index, or format error) from tar: "
                    f"{e}\n Break reading.",
                    "warning",
                )
                break
    except:
        pass
    finally:
        if progress:
            tbar.close()

    if close_tar:
        tar.close()
        log_print(f"Closed tar file: {tar_path}")

    return successful_files


def extract_tar_files_safe(tar_path=None, tar=None, close_tar=True):
    if tar is None:
        assert tar_path is not None
        assert os.path.exists(tar_path), "Tar file does not exist."
        log_print(f"Opening tar file: {tar_path}")
        tar = tarfile.open(tar_path, "r")
        close_tar = True
    else:
        assert tar is not None, "Tar file is required."

    try:
        while True:
            try:
                member = tar.next()
                if member is None:
                    break

                if member.isfile():
                    try:
                        file_data = tar.extractfile(member)
                        yield member, file_data
                    except Exception as e:
                        log_print(
                            f"Failed to read file {member.name} in tar: {e}",
                            "warning",
                        )
                        continue

            except tarfile.TarError as e:
                log_print(
                    f"Failed to next file (head, index, or format error) from tar: {e}. Beaking reading.",
                    "warning",
                )
                break
    except:
        pass
    finally:
        if close_tar:
            tar.close()
            log_print(f"Closed tar file: {tar_path if tar_path else 'provided tar'}")


def flatten_dir_into_one_tar(
    tar_path: str,
    file_dir: str | list | None = None,
    files: list | None = None,
    save_arch_name=False,
    img_ext: str | list[str] = "tif",
    file_type="img",
    compress=True,
    img_call_fn=lambda x: x,  # default no-op function
    is_rgb=False,
):
    from src.data.codecs import rgb_codec_io, tiff_codec_io

    if file_dir:
        if isinstance(file_dir, (str, Path)):
            file_dir = [file_dir]
        if isinstance(img_ext, str):
            img_ext = [img_ext]

        for file_d in file_dir:
            assert Path(file_d).is_dir(), f"{file_dir} is not a directory."
        assert tar_path.endswith(".tar"), "The tar_path must end with .tar"

        files = []
        for file_d in file_dir:
            for ext in img_ext:
                files.extend(list(Path(file_d).rglob(f"*{ext}")))
    elif files:
        assert len(files) > 0, "Length of files should > 0"
    else:
        raise ValueError(f"files or file dir should be provided.")

    log_print(f"Found {len(files)} files")
    tar_file_writer = tarfile.TarFile(tar_path, "w")

    try:
        for idx, file in enumerate(
            tbar := tqdm(
                files,
                desc="Flattening directory into tar",
                unit="file",
                total=len(files),
            )
        ):
            file: Path
            if not file.is_file():
                continue

            # Generate a numbered name for all files
            original_name = file.name
            numbered_name = f"{idx:06d}-{original_name}"

            if not save_arch_name:
                name = numbered_name
            else:
                # relative_path = file.relative_to(file_dir)
                # prev_name = file.name
                name = file.with_name(f"{idx:06d}-{file.name}").as_posix()

            if not compress:
                # add img_type
                parts = name.split(".")
                name = "_".join(parts[:-1]) + ".img" + "." + parts[-1]
                tar_file_writer.add(file, arcname=name)

                tbar.set_description(f"Add {name}")
            else:
                # read in
                name = name.replace(f".{img_ext}", f".{file_type}.tiff")

                if is_rgb:
                    PIL.Image.MAX_IMAGE_PIXELS = None  # Disable PIL's image size limit
                    img = np.array(PIL.Image.open(file))
                else:
                    assert file.as_posix().endswith(("tif", "tiff"))
                    img = tifffile.imread(file)  # [h, w, c]

                if img.shape[-1] > img.shape[0]:  # [c, h, w]
                    img = img.transpose(1, 2, 0)

                img = img_call_fn(img)

                if "tif" in img_ext:
                    buf = tiff_codec_io(
                        img,
                        compression="jpeg2000",
                        compression_args={"reversible": False, "level": 85},
                    )
                elif "rgb" in img_ext:
                    buf = rgb_codec_io(img, format="jpeg", quality=85)
                else:
                    raise ValueError(
                        f"Unsupported image extension: {img_ext}. Supported: tif, rgb."
                    )

                tarinfo = tarfile.TarInfo(name=name)
                tarinfo.size = len(buf)
                tar_file_writer.addfile(tarinfo, fileobj=io.BytesIO(buf))

                tbar.set_description(f"Add {name}, shaped {tuple(img.shape)}")

    except Exception as e:
        log_print(
            f"Error occurred while writing to tar file: {e} handling file {file}",
            "error",
        )
    finally:
        tar_file_writer.close()

    log_print(f"Created tar file: {tar_path}")


def concate_tars(*src_tars, output_tar: str, repeat_find=True):
    n_total = 0
    if repeat_find:
        s = set()
    with tarfile.TarFile(output_tar, "w") as out_tar:
        for tar_file in src_tars:
            assert Path(tar_file).exists(), f"Tar file {tar_file} does not exist."
            tar = tarfile.TarFile(tar_file, "r")
            log_print("merged tar:{}".format(tar.name))

            tar_files = read_tar_filenames_safe(tar_path=tar_file, close_tar=True)

            for i, (member, file_data) in (
                tbar := tqdm(
                    enumerate(extract_tar_files_safe(tar=tar, close_tar=False)),
                    total=len(tar_files),
                    desc="Merging tar members",
                )
            ):
                if repeat_find:
                    if member.name in s:
                        log_print(
                            f"Skipping duplicate member {member.name} in tar {tar.name}",
                            "warning",
                        )
                        continue
                    else:
                        s.add(member.name)

                try:
                    out_tar.addfile(member, file_data)
                    n_total += 1
                    tbar.set_description(f"Extract {member.name}")
                except Exception as e:
                    log_print(
                        f"Failed to add {member.name} to output tar: {e}", "error"
                    )
                    continue
            tar.close()

    log_print(
        f"Concatenated {len(src_tars)} tar files into {output_tar}, total {n_total} files."
    )


# * --- Basic TAR utilities --- #


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


def get_tar_member_iter(tar_reader: tarfile.TarFile, sort=False):
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
    file: tarfile.TarInfo | str | Path | None = None,
    content: bytes | None = None,
    arcname: str | None = None,
):
    assert file is not None or content is not None, (
        "Either file or content must be provided."
    )
    if isinstance(file, tarfile.TarInfo):
        if arcname is not None:
            file.name = arcname
        if content is not None:
            tar_writer.addfile(file, io.BytesIO(content))
    elif isinstance(file, (str, Path)):
        tar_writer.add(file, arcname=arcname)
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
            assert isinstance(content, bytes) or content is None, (
                "Content should be bytes or None"
            )

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


if __name__ == "__main__":
    # Example usage
    # tar_path = "/Data4/cao/ZiHanCao/exps/HyperspectralTokenizer/data/LoveDA/conditions/conditions/LoveDA-3_bands-px_1024-0000.tar"
    # key_to_remove = r".*\.rgb.png"  # Remove all .rgb files
    # remove_key_in_tar(tar_path, key_to_remove)
    # print(f"Removed keys matching '{key_to_remove}' from {tar_path}.")

    # > SEN12MS
    # flatten_dir_into_one_tar(
    #     file_dir="/HardDisk/ZiHanCao/datasets/Multispectral-SEN12MS/ROIs2017_winter_s2",
    #     tar_path="/HardDisk/ZiHanCao/datasets/Multispectral-SEN12MS/ROIs2017_winter_s2.tar",
    #     save_arch_name=False,
    #     img_ext="tif",
    #     file_type="img",
    #     compress=True,
    # )

    # exit(0)

    # > SN1 buildings
    # p_8bands = [
    #     "/HardDisk/ZiHanCao/datasets/Multispectral-Spacenet-series/SN1_buildings/train/8band",
    #     "/HardDisk/ZiHanCao/datasets/Multispectral-Spacenet-series/SN1_buildings/test_public/8band",
    #     list(
    #         Path(
    #             "/HardDisk/ZiHanCao/datasets/Multispectral-Spacenet-series/SN2_buildings/"
    #         ).glob("**/PS-MS")
    #     ),
    #     list(
    #         Path(
    #             "/HardDisk/ZiHanCao/datasets/Multispectral-Spacenet-series/SN3_roads/"
    #         ).glob("**/PS-MS")
    #     ),
    #     list(
    #         Path(
    #             "/HardDisk/ZiHanCao/datasets/Multispectral-Spacenet-series/SN4_buildings/"
    #         ).glob("**/MS")
    #     ),
    #     list(
    #         Path(
    #             "/HardDisk/ZiHanCao/datasets/Multispectral-Spacenet-series/SN5_roads"
    #         ).glob("**/PS-MS")
    #     ),
    # ]
    # p_4bands = [
    #     list(
    #         Path(
    #             "/HardDisk/ZiHanCao/datasets/Multispectral-Spacenet-series/SN4_buildings/"
    #         ).glob("**/PS-RGBNIR")
    #     ),
    #     list(
    #         Path(
    #             "/HardDisk/ZiHanCao/datasets/Multispectral-Spacenet-series/SN6_buildings/"
    #         ).glob("**/PS-RGBNIR")
    #     ),
    #     list(
    #         Path(
    #             "/HardDisk/ZiHanCao/datasets/Multispectral-Spacenet-series/SN7_buildings/train"
    #         ).glob("**/images")
    #     ),
    # ]
    # p_3bands = [
    #     "/HardDisk/ZiHanCao/datasets/Multispectral-Spacenet-series/SN1_buildings/train/3band",
    #     "/HardDisk/ZiHanCao/datasets/Multispectral-Spacenet-series/SN1_buildings/test_public/3band",
    #     list(
    #         Path(
    #             "/HardDisk/ZiHanCao/datasets/Multispectral-Spacenet-series/SN2_buildings/"
    #         ).glob("**/PS-RGB")
    #     ),
    #     list(
    #         Path(
    #             "/HardDisk/ZiHanCao/datasets/Multispectral-Spacenet-series/SN2_buildings/"
    #         ).glob("**/PS-RGB")
    #     ),
    #     list(
    #         Path(
    #             "/HardDisk/ZiHanCao/datasets/Multispectral-Spacenet-series/SN5_roads"
    #         ).glob("**/PS-RGB")
    #     ),
    #     list(
    #         Path(
    #             "/HardDisk/ZiHanCao/datasets/Multispectral-Spacenet-series/SN6_buildings/"
    #         ).glob("**/PS-RGB")
    #     ),
    #     list(
    #         Path(
    #             "/HardDisk/ZiHanCao/datasets/Multispectral-Spacenet-series/SN8_floods"
    #         ).glob("**/PRE-event")
    #     ),
    #     list(
    #         Path(
    #             "/HardDisk/ZiHanCao/datasets/Multispectral-Spacenet-series/SN8_floods"
    #         ).glob("**/POST-event")
    #     ),
    # ]
    # p_lst = [
    #     list(
    #         Path("data/Multispectral-Spacenet-series/SN6_buildings").glob(
    #             "**/SAR-Intensity"
    #         )
    #     )
    # ]

    path = "data/SkyDiffusion"
    img_paths = []
    for ext in ("jpg", "jpeg", "png"):
        img_paths.extend(list(Path(path).glob(f"**/*.{ext}")))
    # p_lst = [img_paths]

    # from skimage.filters import median
    import cv2
    import torch
    from kornia.filters import median_blur
    from scipy.ndimage import median_filter
    from skimage.restoration import denoise_tv_chambolle

    def lee_filter(img, size=5):
        img = img.astype(np.float32)
        mean = cv2.boxFilter(img, -1, (size, size))
        mean_sqr = cv2.boxFilter(img**2, -1, (size, size))
        var = mean_sqr - mean**2
        var_noise = np.mean(var)
        weights = var / (var + var_noise + 1e-8)
        return mean + weights * (img - mean)

    def img_call_fn(x):
        # Normalize the image to [0, 255] and convert to uint8
        x = (x - x.min()) / (x.max() - x.min())
        # Apply filter
        # x = denoise_tv_chambolle(x, channel_axis=-1, weight=0.1)
        # x = lee_filter(x, size=5)
        x = median_blur(torch.as_tensor(x).permute(-1, 0, 1)[None], 5)
        x = x.squeeze(0).permute(1, 2, 0).numpy()
        x = (x * 255).astype(np.uint8)

        return x

    # img_call_fn = lambda x: median_filter(((x - x.min()) / (x.max() - x.min()) * 255).astype(np.uint8), size=5)

    # > FMoW dataset
    # p_lst = ["/HardDisk/ZiHanCao/datasets/Multispectral-fMoW-Data-multispectral-RGB"]

    log_print("------------------- Start ! ----------------------")

    # for p_lst in [p_8bands, p_4bands, p_3bands]:
    # for p in p_lst:
    #     print(p)
    # print("-" * 30)

    base_dir = Path("data/SkyDiffusion")
    tar_path = base_dir / "0000.tar"
    flatten_dir_into_one_tar(
        tar_path=tar_path,
        files=img_paths,
        compress=False,
        save_arch_name=False,
    )

    exit(0)

    for p_list in [p_lst]:
        for i, p in enumerate(p_lst):
            print("-" * 30)
            print(f"Working on paths: {p}")
            print("-" * 30)

            # import time

            # time.sleep(10)

            pp = Path(p) if isinstance(p, (str, Path)) else Path(p[0])
            base_dir = Path("data/Multispectral-Spacenet-series")
            # base_dir = Path(
            #     "/HardDisk/ZiHanCao/datasets/Multispectral-fMoW-Data-multispectral-RGB"
            # )
            rel_name1 = pp.relative_to(base_dir)
            if len(rel_name1.parts) > 0:
                rel_name1 = rel_name1.parts[0]
            else:
                rel_name1 = "default"

            last_name = pp.stem
            tar_path = base_dir / f"{str(i).zfill(2)}_{rel_name1}_{last_name}.tar"
            # if Path(tar_path).exists():
            #     log_print(f"Tar file {tar_path} already exists. Skipping.", "warning")
            #     continue
            flatten_dir_into_one_tar(
                file_dir=p,
                tar_path=tar_path.as_posix(),
                save_arch_name=False,
                img_ext="tif",  # "_rgb.jpg",  # "tif",
                file_type="img",
                compress=True,
                # img_call_fn=img_call_fn,
            )

    # path = "data/BigEarthNet_S2/conditions_resumed/BigEarthNet_data_0000.tar"
    # try:
    #     tar = tarfile.TarFile(path)
    #     names = tar.getnames()
    #     print(f"Tar file {path} contains {len(names)} files.")

    # except Exception as e:
    #     print(f"Error opening tar file {path}: {e}")

    #     names = read_tar_filenames_safe(path)
    #     print(f"Successfully read {len(names)} files from the corrupted tar file.")
