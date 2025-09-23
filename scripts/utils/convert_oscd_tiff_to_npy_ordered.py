#!/usr/bin/env python3
"""
Convert TIFF files in OSCD dataset to NPY format and repack as tar file.

This script processes OSCD (Onera Satellite Change Detection) dataset files,
extracting TIFF images and converting them to NumPy format for better performance
in deep learning pipelines.

This version maintains file order and processes files in memory without
extracting to disk.
"""

import argparse
import io
import os
import tarfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile
from tqdm import tqdm

from src.utilities.io import read_image

SUPPORT_TAR_IMAGE_EXTS = [
    ".tif",
    ".tiff",
    ".npy",
    ".mat",
    ".img",
    ".png",
    ".jpg",
    ".jpeg",
]


def convert_tiff_bytes_to_npy(tiff_bytes: bytes) -> bytes:
    """
    Convert TIFF file bytes to NPY format bytes.

    Parameters
    ----------
    tiff_bytes : bytes
        Raw TIFF file data as bytes

    Returns
    -------
    bytes
        NPY file data as bytes
    """
    # Read TIFF from bytes
    img = tifffile.imread(io.BytesIO(tiff_bytes))

    # Convert to NPY bytes
    npy_buffer = io.BytesIO()
    np.save(npy_buffer, img)
    npy_buffer.seek(0)
    return npy_buffer.read()


def process_oscd_dataset_in_memory(input_tar_path: str, output_tar_path: str) -> None:
    """
    Process OSCD dataset in memory: convert TIFF files to NPY while maintaining order.

    Parameters
    ----------
    input_tar_path : str
        Path to the input tar file containing TIFF files
    output_tar_path : str
        Path for the output tar file containing NPY files
    """
    print(f"Processing {input_tar_path}...")

    # Open input and output tar files
    with (
        tarfile.open(input_tar_path, "r") as input_tar,
        tarfile.open(output_tar_path, "w") as output_tar,
    ):
        # Get all members in original order
        members = input_tar.getmembers()
        tiff_count = sum(
            1 for m in members if m.name.lower().endswith((".tif", ".tiff"))
        )
        npy_count = sum(1 for m in members if m.name.endswith(".npy"))

        print(f"Found {tiff_count} TIFF files and {npy_count} NPY files to process")

        # Process each member in order
        for member in tqdm(members, desc="Converting files"):
            if member.isfile():
                try:
                    # Extract file content
                    file_data = input_tar.extractfile(member)
                    if file_data is None:
                        print(f"Warning: Could not extract {member.name}")
                        continue

                    content = file_data.read()
                    file_data.close()

                    # Determine new name and content
                    if member.name.lower().endswith((".tif", ".tiff")):
                        # Convert TIFF to NPY
                        new_name = member.name.replace(".tif", ".npy").replace(
                            ".tiff", ".npy"
                        )
                        new_content = convert_tiff_bytes_to_npy(content)
                        file_type = "TIFF->NPY"
                    elif member.name.endswith(".npy"):
                        # Keep NPY files as-is
                        new_name = member.name
                        new_content = content
                        file_type = "NPY"
                    else:
                        # Skip other files
                        continue

                    # Create new tar info with same metadata but updated size
                    new_info = tarfile.TarInfo(name=new_name)
                    new_info.size = len(new_content)
                    new_info.mtime = member.mtime
                    new_info.mode = member.mode
                    new_info.uid = member.uid
                    new_info.gid = member.gid
                    new_info.type = member.type

                    # Add to output tar
                    output_tar.addfile(new_info, io.BytesIO(new_content))

                except Exception as e:
                    print(f"Error processing {member.name}: {e}")
                    continue

    print(f"Conversion complete! Output saved to {output_tar_path}")


def verify_output_tar(output_tar_path: str) -> None:
    """
    Verify the output tar file contains correct files.

    Parameters
    ----------
    output_tar_path : str
        Path to the output tar file to verify
    """
    print(f"Verifying {output_tar_path}...")

    with tarfile.open(output_tar_path, "r") as tar:
        members = tar.getnames()
        print(f"Output tar contains {len(members)} files")

        # Test loading a few NPY files
        for i, member_name in enumerate(members[:3]):  # Test first 3 files
            if member_name.endswith(".npy"):
                try:
                    file_data = tar.extractfile(member_name)
                    if file_data:
                        content = file_data.read()
                        file_data.close()

                        # Try to load as numpy array
                        arr = np.load(io.BytesIO(content))
                        print(
                            f"  ✓ {member_name}: shape={arr.shape}, dtype={arr.dtype}"
                        )
                    else:
                        print(f"  ✗ {member_name}: Could not extract")
                except Exception as e:
                    print(f"  ✗ {member_name}: Error loading - {e}")


def main():
    """
    Main function to handle command line arguments and execute conversion.
    """
    parser = argparse.ArgumentParser(
        description="Convert TIFF files in OSCD dataset to NPY format (maintains order)"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input tar file (e.g., OSCD_13bands_train.tar)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path for output tar file (e.g., OSCD_13bands_train.npy.tar)",
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify output tar file after conversion"
    )

    args = parser.parse_args()

    # Validate input file exists
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process the dataset
    process_oscd_dataset_in_memory(args.input, args.output)

    # Verify output if requested
    if args.verify:
        verify_output_tar(args.output)


if __name__ == "__main__":
    main()
