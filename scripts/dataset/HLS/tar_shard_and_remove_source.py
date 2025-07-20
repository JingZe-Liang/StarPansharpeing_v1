import tarfile
from genericpath import isfile
from pathlib import Path

from rich.progress import track
from tqdm import tqdm

from src.utilities.logging.print import _console, log_print


def tar_dir_and_remove(num_files_per_tar: int, dir_path: str | Path):
    tar_i = 0
    dir_path = Path(dir_path)
    tar = tarfile.open(dir_path / f"{str(tar_i).zfill(4)}.tar", "w")
    n_sample = 0

    _total_tbar = len(itered := list(dir_path.iterdir()))
    for path in track(
        itered, description="Tarring files", total=_total_tbar, console=_console
    ):
        if (
            path.is_file()
            # and path.stat().st_size > 5 * 1024 * 1024
            and path.name.endswith((".tif", ".tiff"))
        ):  # > 5M is valid image
            if n_sample >= num_files_per_tar:
                tar.close()
                tar_i += 1
                tar = tarfile.open(dir_path / f"{str(tar_i).zfill(4)}.tar", "w")
                log_print(
                    f"Created new tar file: {dir_path / f'{str(tar_i).zfill(4)}.tar'}",
                    level="info",
                )
                n_sample = 0

            tar.add(path, arcname=path.name)
            path.unlink()  # Delete file after adding
            n_sample += 1
        # elif path.is_file() and path.name.endswith((".tif", ".tiff")):
        #     msg = f"Skipping {path} because it is too small ({path.stat().st_size / 5*1024 * 1024}MB < 5MB)"
        #     log_print(msg, level="warning")
        #     path.unlink()

    tar.close()  # Don't forget to close the last tar


if __name__ == "__main__":
    tar_dir_and_remove(1000, "/Data4/cao/ZiHanCao/exps/HyperspectralTokenizer/data/HLS")
