"""
Download utilities for loading files from URLs.
This module provides functions for downloading files from remote URLs with support for
various protocols including http/https, local file paths, and progress tracking.
"""

import hashlib
import os
import urllib.request
from pathlib import Path
from typing import Optional, Union

import requests
from tqdm import tqdm


def load_file_from_url(
    url: str,
    model_dir: str,
    progress: bool = True,
    file_name: Optional[str] = None,
    check_hash: bool = False,
    hash_prefix: Optional[str] = None,
) -> str:
    """
    Load file from a URL or local path. Similar to basicsr.utils.download_util.load_file_from_url.

    Args:
        url (str): URL or local file path to download from
        model_dir (str): Directory to save the downloaded file
        progress (bool): Whether to show download progress bar
        file_name (str, optional): Name for the downloaded file. If None, infer from URL
        check_hash (bool): Whether to check file hash
        hash_prefix (str, optional): Expected hash prefix for verification

    Returns:
        str: Path to the downloaded/located file

    Raises:
        RuntimeError: If download fails or hash verification fails
    """
    os.makedirs(model_dir, exist_ok=True)

    # Handle local file paths
    if os.path.exists(url):
        if os.path.isfile(url):
            # Return the local file path directly
            return url
        else:
            raise RuntimeError(f"Path exists but is not a file: {url}")

    # Determine the filename
    if file_name is None:
        file_name = os.path.basename(url)
        if not file_name:
            file_name = "downloaded_file"

    # Full path for the downloaded file
    cached_file = os.path.join(model_dir, file_name)

    # Check if file already exists
    if os.path.exists(cached_file):
        if check_hash and hash_prefix:
            if verify_hash(cached_file, hash_prefix):
                print(f"File already exists and hash verified: {cached_file}")
                return cached_file
            else:
                print(
                    f"File exists but hash verification failed, re-downloading: {cached_file}"
                )
                os.remove(cached_file)
        else:
            print(f"File already exists: {cached_file}")
            return cached_file

    print(f"Downloading {url} to {cached_file}")

    try:
        # Download the file
        if url.startswith(("http://", "https://")):
            download_url_to_file(url, cached_file, progress=progress)
        else:
            raise ValueError(f"Unsupported URL scheme: {url}")

        # Verify hash if requested
        if check_hash and hash_prefix:
            if not verify_hash(cached_file, hash_prefix):
                os.remove(cached_file)
                raise RuntimeError(f"Hash verification failed for {cached_file}")

        print(f"Successfully downloaded: {cached_file}")
        return cached_file

    except Exception as e:
        # Clean up partial download
        if os.path.exists(cached_file):
            os.remove(cached_file)
        raise RuntimeError(f"Failed to download {url}: {str(e)}")


def download_url_to_file(
    url: str, dst: str, progress: bool = True, chunk_size: int = 8192
) -> None:
    """
    Download a file from a URL to a destination path with progress bar.

    Args:
        url (str): URL to download from
        dst (str): Destination file path
        progress (bool): Whether to show progress bar
        chunk_size (int): Size of chunks to download at a time
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    # Use requests for better handling
    try:
        with requests.get(url, stream=True, timeout=30) as response:
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            with open(dst, "wb") as f:
                if progress and total_size > 0:
                    with tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        desc=os.path.basename(dst),
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                else:
                    # No progress bar or unknown size
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)

    except requests.exceptions.RequestException as e:
        # Fallback to urllib if requests fails
        print(f"Requests failed, falling back to urllib: {e}")
        download_with_urllib(url, dst, progress)


def download_with_urllib(url: str, dst: str, progress: bool = True) -> None:
    """
    Fallback download method using urllib.

    Args:
        url (str): URL to download from
        dst (str): Destination file path
        progress (bool): Whether to show progress bar
    """

    def show_progress(block_num, block_size, total_size):
        if progress and total_size > 0:
            downloaded = block_num * block_size
            percent = min(100, (downloaded * 100) // total_size)
            print(f"\rDownloading... {percent}%", end="", flush=True)

    try:
        if progress:
            urllib.request.urlretrieve(url, dst, reporthook=show_progress)
            print()  # New line after progress
        else:
            urllib.request.urlretrieve(url, dst)
    except Exception as e:
        raise RuntimeError(f"urllib download failed: {e}")


def verify_hash(file_path: str, expected_prefix: str) -> bool:
    """
    Verify file hash against expected prefix.

    Args:
        file_path (str): Path to file to verify
        expected_prefix (str): Expected hash prefix

    Returns:
        bool: True if hash matches, False otherwise
    """
    try:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)

        actual_hash = sha256_hash.hexdigest()
        return actual_hash.startswith(expected_prefix)

    except Exception as e:
        print(f"Hash verification failed: {e}")
        return False


def get_file_size(file_path: str) -> int:
    """
    Get file size in bytes.

    Args:
        file_path (str): Path to file

    Returns:
        int: File size in bytes
    """
    return os.path.getsize(file_path) if os.path.exists(file_path) else 0


def calculate_md5(file_path: str) -> str:
    """
    Calculate MD5 hash of a file.

    Args:
        file_path (str): Path to file

    Returns:
        str: MD5 hash as hex string
    """
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def calculate_sha256(file_path: str) -> str:
    """
    Calculate SHA256 hash of a file.

    Args:
        file_path (str): Path to file

    Returns:
        str: SHA256 hash as hex string
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()
