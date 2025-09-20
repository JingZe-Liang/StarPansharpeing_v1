import grp
import tarfile
from pathlib import Path

import h5py
import numpy as np
from natsort import natsorted
from tqdm import tqdm
from webdataset import TarWriter

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
        if content is None:
            continue
        # decode
        img = tiff_decode_io(content)
        # name
        n, grp_name = Path(member_name).stem.split(".")
        grps[grp_name].create_dataset(
            n, data=img.transpose([-1, 0, 1]), compression="gzip"
        )


def save_to_npz_tar_from_tar(tar_path: str, save_tar_path: str):
    """
    Convert TIFF files in tar archive to NPZ format using webdataset loader.

    This function uses the existing webdataset loader to read pansharpening data
    (hrms, lrms, pan) from tar files and saves them as NPZ files in a new tar archive.

    Args:
        tar_path: Path to the source tar file containing TIFF files
        save_tar_path: Path to save the output NPZ tar file
    """
    import io

    from webdataset import TarWriter

    from src.stage2.pansharpening.data.pansharpening_loader import (
        get_pansharp_wds_dataloader,
    )

    # Use the existing webdataset loader to read data
    _, dataloader = get_pansharp_wds_dataloader(
        wds_paths=tar_path,
        batch_size=1,
        num_workers=0,
        shuffle_size=0,
        to_neg_1_1=False,  # Keep original values
        resample=False,
        add_satellite_name=False,
    )

    print(f"Processing tar file: {tar_path}")

    Path(save_tar_path).parent.mkdir(parents=True, exist_ok=True)
    tar_writter = TarWriter(save_tar_path)

    # Process each sample
    for i, sample in enumerate(tqdm(dataloader, desc="Converting to NPZ ...")):
        try:
            # Extract image data from sample
            hrms = sample["hrms"][0].numpy()  # Remove batch dimension
            lrms = sample["lrms"][0].numpy()
            pan = sample["pan"][0].numpy()

            # Create NPZ file with all three images
            npz_buffer = io.BytesIO()
            np.savez_compressed(npz_buffer, hrms=hrms, lrms=lrms, pan=pan)

            # Get the bytes from the buffer
            npz_content = npz_buffer.getvalue()

            # Create sample name - use index if __key__ is not available
            if "__key__" in sample and len(sample["__key__"]) > 0:
                sample_name = Path(sample["__key__"][0]).stem
            else:
                sample_name = f"sample_{i:04d}"

            # Write to tar
            file_in = {
                "__key__": sample_name,
                "pair.npz": npz_content,
            }
            tar_writter.write(file_in)

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue

    print(f"Completed processing {tar_path}")
    tar_writter.close()


def save_to_npz_tar_from_tar_full_resolution(
    tar_path: dict[str, str], save_tar_path: str
):
    """
    Convert TIFF files in tar archive to NPZ format using webdataset loader.

    This function uses the existing webdataset loader to read pansharpening data
    (hrms, lrms, pan) from tar files and saves them as NPZ files in a new tar archive.

    Args:
        tar_path: Path to the source tar file containing TIFF files
        save_tar_path: Path to save the output NPZ tar file
    """
    import io

    from src.stage2.pansharpening.data.pansharpening_loader import (
        get_wids_mat_full_resolution_dataloder,
    )

    # Use the existing webdataset loader to read data
    _, dataloader = get_wids_mat_full_resolution_dataloder(
        wds_paths=tar_path,
        batch_size=1,
        num_workers=0,
        shuffle_size=0,
        to_neg_1_1=False,  # Keep original values
        add_satellite_name=False,
    )

    print(f"Processing tar file: {tar_path}")

    Path(save_tar_path).parent.mkdir(parents=True, exist_ok=True)
    tar_writter = TarWriter(save_tar_path)

    # Process each sample
    for i, sample in enumerate(tqdm(dataloader, desc="Converting to NPZ ...")):
        try:
            # Extract image data from sample
            lrms = sample["lrms"][0].numpy()
            pan = sample["pan"][0].numpy()

            # Create NPZ file with all three images
            npz_buffer = io.BytesIO()
            np.savez_compressed(npz_buffer, lrms=lrms, pan=pan)

            # Get the bytes from the buffer
            npz_content = npz_buffer.getvalue()

            # Create sample name - use index if __key__ is not available
            if "__key__" in sample and len(sample["__key__"]) > 0:
                sample_name = Path(sample["__key__"][0]).stem
            else:
                sample_name = f"sample_{i:04d}"

            # Write to tar
            file_in = {
                "__key__": sample_name,
                "pair.npz": npz_content,
            }
            tar_writter.write(file_in)

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue

    print(f"Completed processing {tar_path}")
    tar_writter.close()


if __name__ == "__main__":
    # path = "data/WorldView2/pansharpening_reduced/tmp"
    # tar_path = "data/WorldView3/pansharpening_reduced/Pansharping_WV3_train.tar"
    # h5_path = "data/WorldView3/pansharpening_reduced/Pansharping_WV3_train.h5"
    # h5file = h5py.File(h5_path, "w")
    # # save_to_h5(path, h5file)
    # save_to_h5_from_tar(tar_path, h5file)

    paths = [
        "data/WorldView3/pansharpening_reduced/Pansharping_WV3_train.tar",
        "data/WorldView3/pansharpening_reduced/Pansharping_WV3_val.tar",
        "data/WorldView2/pansharpening_reduced/Pansharpening_WV2_train.tar",
        "data/WorldView2/pansharpening_reduced/Pansharpening_WV2_val.tar",
        "data/WorldView4/pansharpening_reduced/Pansharpening_WV4_train.tar",
        "data/WorldView4/pansharpening_reduced/Pansharpening_WV4_val.tar",
        "data/IKONOS/pansharpening_reduced/Pansharpening_IKONOS_train.tar",
        "data/IKONOS/pansharpening_reduced/Pansharpening_IKONOS_val.tar",
        "data/QuickBird/pansharpening_reduced/Pansharpening_QB_train.tar",
        "data/QuickBird/pansharpening_reduced/Pansharpening_QB_val.tar",
    ]
    full_paths = [
        dict(
            lrms="data/WorldView2/pansharpening_full/MS_shardindex.json",
            pan="data/WorldView2/pansharpening_full/PAN_shardindex.json",
        ),
        dict(
            lrms="data/WorldView3/pansharpening_full/MS_shardindex.json",
            pan="data/WorldView3/pansharpening_full/PAN_shardindex.json",
        ),
        dict(
            lrms="data/WorldView4/pansharpening_full/MS_shardindex.json",
            pan="data/WorldView4/pansharpening_full/PAN_shardindex.json",
        ),
        dict(
            lrms="data/IKONOS/pansharpening_full/MS_shardindex.json",
            pan="data/IKONOS/pansharpening_full/PAN_shardindex.json",
        ),
        dict(
            lrms="data/QuickBird/pansharpening_full/MS_shardindex.json",
            pan="data/QuickBird/pansharpening_full/PAN_shardindex.json",
        ),
    ]

    # for tar_path in paths:
    #     # H5 File converter
    #     # h5_path = Path(tar_path).with_suffix(".h5").as_posix()
    #     # print(f"Processing {tar_path} to {h5_path}")
    #     # h5file = h5py.File(h5_path, "w")
    #     # save_to_h5_from_tar(tar_path, h5file)
    #     # h5file.close()
    #     # print(f"Saved to {h5_path}")

    #     # NPZ Tar converter
    #     tar_save_path = Path(tar_path).with_suffix(".npz.tar").as_posix()
    #     save_to_npy_tar_from_tar(tar_path, tar_save_path)
    #     print(f"Saved to {tar_save_path}")

    # full
    for wids_dict_path in full_paths:
        lrms_path = wids_dict_path["lrms"]
        save_path = Path(lrms_path).parent / "Pansharpening_FullResolution_val.npz.tar"
        save_to_npz_tar_from_tar_full_resolution(wids_dict_path, save_path.as_posix())
