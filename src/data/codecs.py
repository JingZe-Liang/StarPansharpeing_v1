import io
from typing import Dict, Any
import numpy as np
import scipy.io
import torch
import tifffile
from safetensors.torch import (
    save as save_safetensors,
    load as load_safetensors,
)  # Ensure safetensors is installed


# --- Encoding Functions ---


def tiff_codec_io(
    img: np.ndarray,
    planarconfig: str = None,
    photometric: str = None,
    compression: str = "zlib",
) -> bytes:
    """Encodes a NumPy array into TIFF formatted bytes.

    Args:
        img (np.ndarray): Input image array to be encoded.
        planarconfig (str, optional): TIFF planar configuration. Defaults to None.
            Can be 'CONTIG' (contiguous) or 'SEPARATE' (planar).
        photometric (str, optional): TIFF photometric interpretation. Defaults to None.
            Common values include 'RGB', 'MINISBLACK', etc.

    Returns:
        bytes: TIFF formatted bytes of the encoded image.
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


def tiff_decode_io(tiff_bytes: bytes) -> np.ndarray:
    """Decodes TIFF formatted bytes into a NumPy array."""
    with io.BytesIO(tiff_bytes) as buffer:
        img: np.ndarray = tifffile.imread(buffer)
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


def safetensors_decode_io(safetensors_bytes: bytes) -> Dict[str, torch.Tensor]:
    """Decodes safetensors formatted bytes into a dictionary of PyTorch tensors."""
    # Directly loads the tensors from bytes.
    # Tensors will be loaded onto the device they were saved from,
    # or potentially CPU depending on safetensors implementation details.
    # Usually, it's best practice to move tensors to the desired device after loading.
    data_dict: Dict[str, torch.Tensor] = load_safetensors(safetensors_bytes)
    return data_dict


def npz_decode_io(npz_bytes: bytes) -> Dict[str, np.ndarray]:
    """Decodes NPZ file formatted bytes into a dictionary of NumPy arrays."""
    with io.BytesIO(npz_bytes) as buffer:
        # load returns a dictionary, potentially including header info
        npz_dict: Dict[str, Any] = np.load(buffer, allow_pickle=True)

        data_dict = {k: v.copy() for k, v in npz_dict.items() if not k.startswith("__")}
    return data_dict
