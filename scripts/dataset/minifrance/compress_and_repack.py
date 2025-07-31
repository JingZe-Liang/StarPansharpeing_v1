#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to read JPG files from tar archives, compress them with quality 80,
and repack them into new tar files.
"""

import io
import os
import tarfile
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def process_tar_file(input_tar_path, output_tar_path, quality=80):
    """
    Process a tar file containing JPG images, compress them, and save to a new tar file.

    Args:
        input_tar_path (str): Path to the input tar file
        output_tar_path (str): Path to the output tar file
        quality (int): JPEG compression quality (default: 80)
    """
    # Open the input tar file
    with tarfile.open(input_tar_path, "r") as tar:
        print(f"Processing {input_tar_path}")

        # Extract all jpg files
        jpg_members = [
            member for member in tar.getmembers() if member.name.endswith(".jpg")
        ]
        print(f"Found {len(jpg_members)} JPG files")

        # Create output tar file
        with tarfile.open(output_tar_path, "w") as out_tar:
            for i, member in tqdm(enumerate(jpg_members), total=len(jpg_members)):
                # if i % 100 == 0:
                #     print(f"Processed {i}/{len(jpg_members)} files")

                # Extract the image to memory
                f = tar.extractfile(member)
                if f is not None:
                    # Open image with PIL
                    img = Image.open(f)

                    # Compress image in memory
                    buf = io.BytesIO()
                    img.save(buf, "JPEG", quality=quality, optimize=True)
                    buf.seek(0)  # Reset buffer position to beginning

                    # Create TarInfo with correct size
                    tarinfo = tarfile.TarInfo(name=member.name)
                    value = buf.getvalue()
                    tarinfo.size = len(value)

                    # Add compressed image to output tar directly from memory
                    out_tar.addfile(tarinfo, io.BytesIO(value))

            print(f"Finished processing {input_tar_path}")
            print(f"Output saved to {output_tar_path}")


def main():
    """Main function to process all tar files in the directory."""
    # Define paths
    input_dir = Path(
        "/Data4/cao/ZiHanCao/exps/HyperspectralTokenizer/data/miniFrance/hyper_images"
    )
    output_dir = Path(
        "/Data4/cao/ZiHanCao/exps/HyperspectralTokenizer/data/miniFrance/compressed_images"
    )

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process all tar files
    tar_files = list(input_dir.glob("*.tar"))
    print(f"Found {len(tar_files)} tar files to process")

    for tar_file in tar_files:
        output_tar_path = output_dir / f"compressed_{tar_file.name}"
        # if not output_tar_path.exists():
        process_tar_file(str(tar_file), str(output_tar_path))
        # else:
        #     print(f"Skipping {tar_file.name} as compressed version already exists")


if __name__ == "__main__":
    main()
