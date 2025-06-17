import io
import warnings
from typing import Any, Dict

import numpy as np
import scipy.io
import tifffile
import torch
from PIL import Image
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

from src.utilities.logging import log_print

from .utils import to_n_tuple

# --- Encoding Functions ---


def tiff_codec_io(
    img: np.ndarray,
    planarconfig: str | tifffile.PLANARCONFIG | None = None,
    photometric: str | tifffile.PHOTOMETRIC | None = None,
    compression: str = "zlib",
) -> bytes:
    """Encodes a NumPy array into TIFF formatted bytes.

    Args:
        img (np.ndarray): Input image array to be encoded.
        planarconfig (str | tifffile.PLANARCONFIG | None, optional): TIFF planar configuration. Defaults to None.
            Can be 'CONTIG' (contiguous) or 'SEPARATE' (planar).
        photometric (str | tifffile.PHOTOMETRIC | None, optional): TIFF photometric interpretation. Defaults to None.
            Common values include 'RGB', 'MINISBLACK', etc.
        compression (str, optional): Compression method to use. Defaults to "zlib".
            Other options include 'lzw', 'jpeg', 'packbits', 'deflate', etc.

    Returns:
        bytes: TIFF formatted bytes of the encoded image.

    Note:
        The function uses tifffile library to encode the image. The planarconfig and photometric
        parameters should match the image data structure to ensure correct encoding.
    """
    with io.BytesIO() as buffer:
        tifffile.imwrite(
            buffer,
            img,
            planarconfig=planarconfig,
            photometric=photometric,
            compression=compression,
        )

        return buffer.getvalue()


def npz_codec_io(
    data_dict: Dict[str, np.ndarray], do_compression: bool = True
) -> bytes:
    """Encodes a dictionary of NumPy arrays into NPZ file formatted bytes."""
    with io.BytesIO() as buffer:
        # Saves the dictionary. Keys become variable names in the NPZ file.
        if do_compression:
            np.savez_compressed(buffer, **data_dict, allow_pickle=True)
        else:
            np.savez(buffer, **data_dict, allow_pickle=True)

        return buffer.getvalue()


def mat_codec_io(
    data_dict: Dict[str, np.ndarray], do_compression: bool = True
) -> bytes:
    """Encodes a dictionary of NumPy arrays into MAT file formatted bytes."""
    with io.BytesIO() as buffer:
        # Saves the dictionary. Keys become variable names in the MAT file.
        scipy.io.savemat(buffer, data_dict, do_compression=do_compression)
        return buffer.getvalue()


def safetensors_codec_io(data_dict: Dict[str, torch.Tensor]) -> bytes:
    """Encodes a dictionary of PyTorch tensors into safetensors formatted bytes."""
    # Directly returns the bytes for the saved tensors.
    # Ensure tensors are on CPU if that's the desired storage format.
    return save_safetensors(data_dict)


# --- Decoding Functions ---

# Image.MAX_IMAGE_PIXELS = None  # Disable the decompression bomb protection in PIL


def img_decode_io(img_bytes: bytes):
    with warnings.catch_warnings(record=True) as w:
        try:
            # import ipdb; ipdb.set_trace()  # noqa: T201
            img = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))

            for warning in w:
                _w = Image.DecompressionBombWarning
                if issubclass(warning.category, _w):
                    # Handle decompression bomb warnings specifically
                    warnings.warn(
                        f"Image decompression bomb warning: {warning.message}, image size: {img.shape}",
                        category=UserWarning,
                    )
                    break
                else:
                    # Handle other image decoding warnings
                    warnings.warn(
                        f"Image decoding warning: {warning.message}",
                        category=UserWarning,
                    )
            return img
        except Image.DecompressionBombError as e:
            log_print(f"Too large image: {e}", "warning", warn_once=True)
            return None
        except Exception as e:
            log_print(f"Error decoding image: {e}", "error")
            return None


def string_decode_io(string_bytes: bytes) -> str:
    """Decodes bytes into a string."""
    return string_bytes.decode("utf-8")  # Assuming UTF-8 encoding for the string


def tiff_decode_io(
    tiff_bytes: bytes,
    use_out_param: bool = True,  # Flag to control using 'out' parameter
) -> np.ndarray:
    """
    Decodes TIFF formatted bytes into a NumPy array.
    Optionally reads metadata first to pre-allocate memory for the 'out' parameter.
    """
    __predefined_buf_size = 16 * 1024 * 1024
    __max_workers = 4

    if use_out_param:
        try:
            # Step 1: Read metadata using TiffFile to get shape and dtype from the first series
            expected_shape: tuple[int, ...]
            expected_dtype: np.dtype
            with io.BytesIO(tiff_bytes) as metadata_buffer:
                with tifffile.TiffFile(metadata_buffer) as tif:
                    if not tif.series or not tif.series[0]:
                        raise ValueError(
                            "TIFF file contains no series or series[0] is invalid."
                        )
                    current_series: tifffile.TiffPageSeries = tif.series[0]
                    expected_shape = current_series.shape
                    expected_dtype = np.dtype(
                        current_series.dtype
                    )  # Ensure it's a numpy.dtype

            # Step 2: Pre-allocate the output array
            output_array: np.ndarray = np.empty(expected_shape, dtype=expected_dtype)

            # Step 3: Read image data into the pre-allocated array
            # Use a new BytesIO object for imread to ensure the stream is at the beginning
            with io.BytesIO(tiff_bytes) as data_buffer:
                tifffile.imread(
                    data_buffer,
                    out=output_array,
                    buffersize=__predefined_buf_size,  # Example buffer size
                    maxworkers=__max_workers,  # Example max workers
                )
            return output_array
        except Exception as e:
            # Fallback to the simpler method if any error occurs during the 'out' optimization path
            # You might want to log the error 'e' for debugging purposes
            # print(f"Warning: Failed to use 'out' parameter optimization: {e}. Falling back.")
            with io.BytesIO(tiff_bytes) as buffer:
                img: np.ndarray = tifffile.imread(
                    buffer,
                    buffersize=__predefined_buf_size,
                    maxworkers=__max_workers,
                )
            return img
    else:
        # Original behavior if use_out_param is False
        with io.BytesIO(tiff_bytes) as buffer:
            img: np.ndarray = tifffile.imread(
                buffer,
                buffersize=__predefined_buf_size,
                maxworkers=__max_workers,
            )
        return img


def mat_decode_io(mat_bytes: bytes) -> Dict[str, np.ndarray]:
    """Decodes MAT file formatted bytes into a dictionary of NumPy arrays."""
    with io.BytesIO(mat_bytes) as buffer:
        # loadmat returns a dictionary, potentially including header info
        mat_dict: Dict[str, Any] = scipy.io.loadmat(buffer)
    # Filter out MATLAB specific keys
    data_dict: Dict[str, np.ndarray] = {
        k: v
        for k, v in mat_dict.items()
        if not k.startswith(
            "__"
        )  # Remove keys like __header__, __version__, __globals__
    }
    return data_dict


def safetensors_decode_io(safetensors_bytes: bytes, to_device=False) -> torch.Tensor:
    """Decodes safetensors formatted bytes into a dictionary of PyTorch tensors."""
    # Directly loads the tensors from bytes.
    # Tensors will be loaded onto the device they were saved from,
    # or potentially CPU depending on safetensors implementation details.
    # Usually, it's best practice to move tensors to the desired device after loading.
    data_dict: Dict[str, torch.Tensor] = load_safetensors(safetensors_bytes)
    if to_device:
        for key, tensor in data_dict.items():
            data_dict[key] = tensor.to(torch.device("cuda"))
    return data_dict[list(data_dict.keys())[0]]  # assume only one key


def npz_decode_io(npz_bytes: bytes) -> Dict[str, np.ndarray]:
    """Decodes NPZ file formatted bytes into a dictionary of NumPy arrays."""
    with io.BytesIO(npz_bytes) as buffer:
        # load returns a dictionary, potentially including header info
        npz_dict: Dict[str, Any] = np.load(buffer, allow_pickle=True)

        data_dict = {k: v.copy() for k, v in npz_dict.items() if not k.startswith("__")}
    return data_dict


def npy_codec_io(img: np.ndarray, compress: bool = False) -> bytes:
    """Encodes a NumPy array into NPY file formatted bytes."""
    with io.BytesIO() as buffer:
        np.savez(
            buffer,
            img,
        )
        return buffer.getvalue()


# * --- wids codecs --- #


def wids_remove_none_keys(sample: dict[str, Any]) -> dict[str, Any]:
    """Remove keys with None values from the sample dictionary."""
    # remove None key/values
    _key_to_del = []
    for k, v in sample.items():
        if v is None:
            _key_to_del.append(k)
    for k in _key_to_del:
        del sample[k]

    return sample


def is_tiff_file(file_path: str) -> bool:
    """Check if the file is a TIFF file based on its extension."""
    return file_path.lower().endswith((".tiff", ".tif"))


def is_rgb_file(file_path: str) -> bool:
    """Check if the file is an RGB image based on its extension."""
    return file_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))


def wids_image_decode(
    sample: dict[str, Any],
    read_caption=False,
    read_name=False,
    norm=True,
    permute=True,
    resize: int | tuple[int, int] | None = None,
):
    _keys_to_try = [".img.tiff", ".img.tif"]
    for k in _keys_to_try:
        if k in sample:
            if is_tiff_file(k):
                sample["img"] = tiff_decode_io(sample[k].getvalue())
                del sample[k]
            elif is_rgb_file(k):
                sample["img"] = img_decode_io(sample[k].getvalue())
                del sample[k]
            else:
                raise ValueError(f"Unsupported file type for key {k}: {sample[k]}")
    if ".img_content" in sample:
        sample["img"] = img_decode_io(sample[".img_content"].getvalue())
        del sample[".img_content"]
    if ".caption" in sample:
        if read_caption:
            sample["caption"] = string_decode_io(sample[".caption"].getvalue())
        del sample[".caption"]
    if ".img_name" in sample:
        if read_name:
            sample["img_name"] = string_decode_io(sample[".img_name"].getvalue())
        del sample[".img_name"]
    if "img" not in sample:
        return sample
    else:
        if sample["img"] is None:
            raise ValueError("Decoded image is None, possibly due to decoding error.")

        img = sample["img"].astype(np.float32) / 255.0
        if norm:
            sample["img"] = img * 2 - 1  # to [-1, 1]
        if permute:
            # Permute the image dimensions from HWC to CHW
            sample["img"] = np.transpose(sample["img"], (2, 0, 1))
        if resize is not None:
            if permute:
                img = torch.as_tensor(sample["img"])[None].float()
            else:
                img = torch.as_tensor(np.transpose(sample["img"], (1, 2, 0)))[
                    None
                ].float()

            sz = resize
            if isinstance(resize, int):
                sz = to_n_tuple(sz, 2)
            img = torch.nn.functional.interpolate(
                img, size=sz, mode="bilinear", align_corners=False
            )[0]
            if not permute:
                img = img.permute(1, 2, 0)

        sample = wids_remove_none_keys(sample)

        return sample


def wids_latent_decode(sample: dict[str, Any]):
    if ".latents.npz" in sample:
        sample["latents"] = npz_decode_io(sample[".latents.npz"].getvalue())
        del sample[".latents.npz"]
    if ".latents.safetensors" in sample:
        sample["latents"] = safetensors_decode_io(
            sample[".latents.safetensors"].getvalue()
        )
        del sample[".latents.safetensors"]
    if ".latents.npy" in sample:
        sample["latents"] = npy_codec_io(sample[".latents.npy"].getvalue())
        del sample[".latents.npy"]

    sample = wids_remove_none_keys(sample)

    return sample
