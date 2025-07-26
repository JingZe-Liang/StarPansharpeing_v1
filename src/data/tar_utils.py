import io
import os
import re
import tarfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import PIL
import PIL.Image
import tifffile
import webdataset as wds
from skimage.io import imread
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

    # breakpoint()
    with tarfile.open(tar_path, "r") as tar, tarfile.open(tmp_path, "w") as out_tar:
        for member in (tbar := tqdm(tar, desc="Filtering tar members")):
            if not pattern.search(member.name):
                out_tar.addfile(
                    member, tar.extractfile(member) if member.isfile() else None
                )
            else:
                tbar.set_description(f"Skipping {member.name}")
    # Replace original tar file
    import os

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
    file_dir: str | list,
    tar_path: str,
    save_arch_name=False,
    img_ext: str = ".tiff",
    file_type="img",
    compress=True,
):
    if isinstance(file_dir, (str, Path)):
        file_dir = [file_dir]

    for file_d in file_dir:
        assert Path(file_d).is_dir(), f"{file_dir} is not a directory."
    assert tar_path.endswith(".tar"), "The tar_path must end with .tar"

    files = []
    for file_d in file_dir:
        files.extend(list(Path(file_d).rglob(f"*{img_ext}")))
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

            name = name.replace(f".{img_ext}", f".{file_type}.{img_ext}")
            if not compress:
                tar_file_writer.add(file, arcname=name)
            else:
                # read in
                from src.data.codecs import rgb_codec_io, tiff_codec_io

                if "rgb" in img_ext:
                    PIL.Image.MAX_IMAGE_PIXELS = None  # Disable PIL's image size limit
                    img = np.array(PIL.Image.open(file))
                    # img = imread(file)
                else:
                    img = tifffile.imread(file)  # [h, w, c]
                if img.shape[-1] > img.shape[0]:  # [c, h, w]
                    img = img.transpose(1, 2, 0)

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

    # > FMoW dataset
    p_lst = ["/HardDisk/ZiHanCao/datasets/Multispectral-fMoW-Data-multispectral-RGB"]

    log_print("------------------- Start ! ----------------------")

    # for p_lst in [p_8bands, p_4bands, p_3bands]:
    # for p in p_lst:
    #     print(p)
    # print("-" * 30)

    for p_list in [p_lst]:
        for i, p in enumerate(p_lst):
            print("-" * 30)
            print(f"Working on paths: {p}")
            print("-" * 30)

            # import time

            # time.sleep(10)

            pp = Path(p) if isinstance(p, (str, Path)) else Path(p[0])
            # base_dir = Path("/HardDisk/ZiHanCao/datasets/Multispectral-Spacenet-series")
            base_dir = Path(
                "/HardDisk/ZiHanCao/datasets/Multispectral-fMoW-Data-multispectral-RGB"
            )
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
                img_ext="_rgb.jpg",  # "tif",
                file_type="img",
                compress=True,
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
