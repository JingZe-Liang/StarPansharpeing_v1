import grp
import tarfile
from pathlib import Path

import h5py
import numpy as np
from natsort import natsorted
from tqdm import tqdm

from src.data.codecs import tiff_decode_io
from src.data.tar_utils import (
    extract_member_from_tar,
    get_content_from_member,
    get_tar_member_iter,
)
from src.utilities.io import read_image


def save_to_h5(dir: str | Path, h5file: h5py.File):
    dir = Path(dir)
    files = list(dir.glob("*"))
    indices = np.array([int(f.stem.split(".")[0]) for f in files])
    indices = np.unique(indices)
    grps = {}
    for name in (names := ["hrms", "lrms", "pan"]):
        g = h5file.create_group(name)
        grps[name] = g
    for i in tqdm(indices):
        group_files = [f"{i}.hrms.tiff", f"{i}.lrms.tiff", f"{i}.pan.tiff"]
        for n, gf in zip(names, group_files):
            img = read_image(dir / gf)
            grps[n].create_dataset(
                str(i), data=img.transpose([-1, 0, 1]), compression="gzip"
            )


def save_to_h5_from_tar(tar_path: str, h5file: h5py.File):
    tar = tarfile.open(tar_path, "r")
    h5_names = ["hrms", "lrms", "pan"]
    grps = {name: h5file.create_group(name) for name in h5_names}
    for member in tqdm(tar.getmembers()):
        member_name = member.name
        content = get_content_from_member(tar, member)
        # decode
        img = tiff_decode_io(content)
        # name
        n, grp_name = Path(member_name).stem.split(".")
        grps[grp_name].create_dataset(
            n, data=img.transpose([-1, 0, 1]), compression="gzip"
        )


if __name__ == "__main__":
    # path = "data/WorldView2/pansharpening_reduced/tmp"
    # tar_path = "data/WorldView3/pansharpening_reduced/Pansharping_WV3_train.tar"
    # h5_path = "data/WorldView3/pansharpening_reduced/Pansharping_WV3_train.h5"
    # h5file = h5py.File(h5_path, "w")
    # # save_to_h5(path, h5file)
    # save_to_h5_from_tar(tar_path, h5file)

    paths = [
        # "data/WorldView3/pansharpening_reduced/Pansharping_WV3_train.tar",
        # "data/WorldView3/pansharpening_reduced/Pansharping_WV3_val.tar",
        # "data/WorldView2/pansharpening_reduced/Pansharpening_WV2_train.tar",
        "data/WorldView2/pansharpening_reduced/Pansharpening_WV2_val.tar",
        "data/WorldView4/pansharpening_reduced/Pansharpening_WV4_train.tar",
        "data/WorldView4/pansharpening_reduced/Pansharpening_WV4_val.tar",
        "data/IKONOS/pansharpening_reduced/Pansharpening_IKONOS_train.tar",
        "data/IKONOS/pansharpening_reduced/Pansharpening_IKONOS_val.tar",
        "data/QuickBird/pansharpening_reduced/Pansharpening_QB_train.tar",
        "data/QuickBird/pansharpening_reduced/Pansharpening_QB_val.tar",
    ]

    for tar_path in paths:
        h5_path = Path(tar_path).with_suffix(".h5").as_posix()
        print(f"Processing {tar_path} to {h5_path}")
        h5file = h5py.File(h5_path, "w")
        save_to_h5_from_tar(tar_path, h5file)
        h5file.close()
        print(f"Saved to {h5_path}")
