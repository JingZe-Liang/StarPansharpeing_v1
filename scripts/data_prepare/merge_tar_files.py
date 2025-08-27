import tarfile
import os
from loguru import logger
from tqdm import tqdm
import sys

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    colorize=True,
    level="DEBUG",
)


def merge_tar_files(tar_files, output_path):
    """
    Merges multiple tar files into a single tar file, handling incomplete source files.

    Args:
        tar_files (list): A list of paths to the source tar files.
        output_path (str): The path for the output merged tar file.
    """
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    with tarfile.open(output_path, "w") as out_tar:
        for tar_file_path in tqdm(tar_files, desc="Merging tar files"):
            logger.info(f"tar file: {tar_file_path}")
            try:
                # Open each source tar file
                with tarfile.open(tar_file_path, "r") as tar1:
                    # --- Modified part: Iterate manually and handle ReadError ---
                    logger.info(f"Attempting to read members from: {tar_file_path}")
                    member_count = 0
                    while True:
                        try:
                            member = tar1.next()  # Get the next member info
                            if member is None:
                                # End of a valid archive marker found
                                logger.info(
                                    "Reached end of archive markers for this file."
                                )
                                break  # Exit the while loop for this tar file

                            # If member is successfully read, add it to the output tar
                            # Use tar1.extractfile(member) to get the file-like object for content
                            if member.isfile():  # Only process regular files
                                f = tar1.extractfile(member)
                                if f is not None:
                                    out_tar.addfile(
                                        member, f
                                    )  # Add member to output tar``
                                    f.close()
                                    member_count += 1
                                    # Optional: log progress or member name
                                    logger.debug(f"  Added member: {member.name}")
                            elif member.isdir():
                                # Optionally handle directories if needed, adding just the header
                                out_tar.addfile(member)
                                member_count += 1
                                logger.debug(f"  Added directory header: {member.name}")

                        except tarfile.ReadError as e:
                            # Caught the error indicating incomplete file
                            logger.warning(
                                f"Caught tarfile.ReadError for {tar_file_path}: {e}"
                            )
                            logger.warning(
                                f"The tar file appears to be incomplete or corrupted. Processed {member_count} members from this file before the error."
                            )
                            break  # Stop processing this specific tar file

                        except Exception as e:
                            # Catch other potential errors during iteration/processing of a member
                            logger.error(
                                f"Caught an unexpected error while processing a member in {tar_file_path}: {e}"
                            )
                            # Decide if you want to continue or stop on other errors
                            break  # Stop processing this specific tar file on other errors

                    # --- End of modified part ---

            except FileNotFoundError:
                logger.error(f"Source tar file not found: {tar_file_path}")
            except tarfile.TarError as e:
                logger.error(
                    f"An error occurred while opening or processing tar file {tar_file_path}: {e}"
                )
            except Exception as e:
                logger.error(
                    f"An unexpected error occurred with file {tar_file_path}: {e}"
                )


if __name__ == "__main__":
    tar_files = [
        "data/pansharpening/MMSeg_YREB/dataset_0000.tar",
        "data/pansharpening/MMSeg_YREB/MMSeg_YREB_train_part-12_bands-MSI-0000.tar",
    ]
    output_path = "data/pansharpening/MMSeg_YREB/cat.tar"

    merge_tar_files(tar_files, output_path)
