import io
import json
import os
import pickle
import warnings
from contextlib import suppress
from functools import partial
from io import BytesIO
from typing import Any, Dict, Literal

import numpy as np
import scipy.io
import tifffile
import torch
import yaml
from loguru import logger
from PIL import Image
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors
from timm.layers.helpers import to_2tuple

from src.utilities.logging import log_print

from .utils import normalize_image_

NV_IMAGE_DECODE_ENABLED = False
try:
    from nvidia import nvimgcodec

    NV_IMAGE_DECODE_ENABLED = True
    # if the loader has prefetch_factor > 0
    # gpu decode will take more gpu memory
    # decoder
    nvimg_gpu_dec_backends = dict(
        backends=[
            nvimgcodec.Backend(nvimgcodec.GPU_ONLY, load_hint=0.5),
            nvimgcodec.Backend(nvimgcodec.HYBRID_CPU_GPU),
        ]
    )
    nvimg_cpu_dec_backends = dict(backend_kinds=[nvimgcodec.CPU_ONLY])
    # encoder
    # logger.debug("Nvidia Image Codec is enabled.")
except ImportError:
    pass


# *==============================================================
# * Litdata Serializers
# *==============================================================

# litdata serializers

import jsonlines as jsonl
from litdata.streaming import serializers


def _check_valid_obj(b: bytes):
    if b[:8] == b"NotFound":
        return None, False
    return b, True


class JsonlSerializer(serializers.Serializer):
    """Serializer for JSONL (JSON Lines) format."""

    def serialize(self, obj: dict) -> tuple[bytes, str]:
        """Serialize a dictionary to JSONL bytes format."""
        buf = io.BytesIO()
        writer = jsonl.Writer(buf)
        writer.write(obj)
        writer.close()
        return buf.getvalue(), "jsonl"

    def deserialize(self, byte_data: bytes) -> dict | list[dict] | None:
        """Deserialize JSONL bytes back to a dictionary or list of dictionaries."""
        byte_data, valid = _check_valid_obj(byte_data)
        if not valid:
            return None

        text_data = byte_data.decode("utf-8").strip()

        if not text_data:
            return {}

        lines = text_data.split("\n")
        objects = [json.loads(line) for line in lines if line.strip()]

        if len(objects) == 1:
            return objects[0]
        else:
            return objects

    def can_serialize(self, data: Any) -> bool:
        """Check if the data can be serialized as JSONL."""
        return isinstance(data, dict)


class TiffFileSerializer(serializers.TIFFSerializer):
    def deserialize(self, bytes_data: bytes) -> torch.Tensor | None:
        """Deserialize bytes into an object."""
        bytes_data, valid = _check_valid_obj(bytes_data)
        if not valid:
            return None

        arr = super().deserialize(bytes_data)
        # additional transport like in JPEGSerializer
        if arr.ndim == 3:
            arr = arr.transpose([-1, 0, 1])
        return torch.from_numpy(arr)


class JPEGGeneralSerializer(serializers.JPEGSerializer):
    def _force_to_rgb(self, arr: np.ndarray | torch.Tensor):
        if arr.shape[0] == 4:
            # png with alpha channel
            logger.debug(f"Warning: 4 channels, shape {arr.shape} -> change to 3 RGB channels.")
            arr = arr[:3]
        return arr

    def deserialize(self, data: bytes) -> torch.Tensor | None:
        """Deserialize bytes into an object."""
        data, valid = _check_valid_obj(data)
        if not valid:
            return None

        # filtering and libpng C lib warnings
        saved_stdout = os.dup(1)
        saved_stderr = os.dup(2)

        try:
            devnull = os.open("/dev/null", os.O_WRONLY)
            os.dup2(devnull, 1)  # stdout
            os.dup2(devnull, 2)  # stderr
            os.close(devnull)

            with suppress(RuntimeError):
                # torchvision decode failed
                arr = super().deserialize(data)
                arr = self._force_to_rgb(arr)
                return arr

            # Use general PIL decoder
            arr = img_decode_io(data)

            if arr is None:
                return None
            elif arr.ndim == 3:
                arr = arr.transpose([-1, 0, 1])

            arr = self._force_to_rgb(arr)

            return torch.from_numpy(arr)
        finally:
            # Restore stdout and stderr
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)


def _load_list_img_bytes(img_bytes_list: bytes, decode_type: str = "tiff"):
    # list of bytes of images
    # pickle load into a list of image bytes
    img_lst = pickle.loads(img_bytes_list)
    ret_img_lst = []
    for img_b in img_lst:
        if decode_type == "tiff":
            img = tiff_decode_io(img_b)
        elif decode_type == "img":
            img = img_decode_io(img_b)
        else:
            raise ValueError(f"Unsupported decode type: {decode_type}")
        ret_img_lst.append(img)
    return ret_img_lst


class TiffSequenceFileSerializer(serializers.Serializer):
    def serialize(self, obj: list[np.ndarray]) -> tuple[bytes, str]:
        """
        Serialize a list of numpy arrays to a pickle file.
        """
        bytes_lst = []
        for arr in obj:
            assert isinstance(arr, np.ndarray), f"All elements must be numpy arrays, but got {type(arr)}"
            arr_bytes = tiff_codec_io(arr)  # zlib compression
            bytes_lst.append(arr_bytes)
        bytes_all = pickle.dumps(bytes_lst, protocol=5)
        return bytes_all, "tifffile_seq"

    def deserialize(self, byte_data: bytes) -> list[np.ndarray]:
        return _load_list_img_bytes(byte_data, "tiff")

    def can_serialize(self, obj: list[np.ndarray]) -> bool:
        return isinstance(obj, list) and all(isinstance(x, np.ndarray) for x in obj)


class JPEGSequenceGeneralSerializer(serializers.Serializer):
    def serialize(self, obj: list[np.ndarray]) -> tuple[bytes, str]:
        """
        Serialize a list of numpy arrays to a pickle file.
        """
        bytes_lst = []
        for arr in obj:
            assert isinstance(arr, np.ndarray), f"All elements must be numpy arrays, but got {type(arr)}"
            arr_bytes = rgb_codec_io(arr, format="jpeg", quality=95)
            bytes_lst.append(arr_bytes)
        bytes_all = pickle.dumps(bytes_lst, protocol=5)
        return bytes_all, "jpeg_seq"

    def _force_to_rgb(self, arr: np.ndarray | torch.Tensor):
        if arr.shape[0] == 4:
            # png with alpha channel
            logger.debug(f"Warning: 4 channels, shape {arr.shape} -> change to 3 RGB channels.")
            arr = arr[:3]
        return arr

    def deserialize(self, byte_data: bytes):
        byte_data, valid = _check_valid_obj(byte_data)
        if not valid:
            return None

        img_lst = _load_list_img_bytes(byte_data, "img")
        for i in range(len(img_lst)):
            img_lst[i] = self._force_to_rgb(img_lst[i])
        return img_lst

    def can_serialize(self, obj: Any) -> bool:
        return isinstance(obj, list) and all(isinstance(item, np.ndarray) for item in obj)


serializers._SERIALIZERS["jsonl"] = JsonlSerializer()  # type: ignore
serializers._SERIALIZERS["tifffile"] = TiffFileSerializer()
serializers._SERIALIZERS["jpeg"] = JPEGGeneralSerializer()
serializers._SERIALIZERS["tifffile_seq"] = TiffSequenceFileSerializer()
serializers._SERIALIZERS["jpeg_seq"] = JPEGSequenceGeneralSerializer()

# logger.debug("Registered JsonlSerializer for litdata")
# logger.debug("Modified TiffFileSerializer for litdata")
# logger.debug("Modified JPEGGeneralSerializer for litdata")


# *==============================================================
# * Utilities
# *==============================================================


def py_obj_to_jsonl(metadata: dict):
    buffer = io.BytesIO()
    buffer.write(json.dumps(metadata).encode("utf-8") + b"\n")
    return buffer.getvalue()


def rgb_codec_io(img: np.ndarray, format="jpeg", **kwargs):
    buffer = io.BytesIO()
    assert img.ndim in (2, 3), "Image must be 2D (grayscale) or 3D (color) array"
    if img.ndim == 3:
        assert img.shape[-1] in (1, 3), "Image must be RGB or gray"
    assert img.dtype == np.uint8, "Image must be of type uint8"
    Image.fromarray(img).save(buffer, format=format, **kwargs)

    return buffer.getvalue()


def tiff_codec_io(
    img: np.ndarray,
    planarconfig: str | tifffile.PLANARCONFIG | None = None,
    photometric: str | tifffile.PHOTOMETRIC | None = None,
    compression: str = "zlib",
    compression_args: dict[str, Any] | None = None,
) -> bytes:
    """Encodes a NumPy array into TIFF formatted bytes.

    Args:
        img (np.ndarray): Input image array to be encoded. Can be 2D (grayscale) or 3D (color) array.
        planarconfig (str | tifffile.PLANARCONFIG | None, optional): TIFF planar configuration. Defaults to None.
            Can be 'CONTIG' (contiguous, interleaved channels) or 'SEPARATE' (planar, separate planes).
        photometric (str | tifffile.PHOTOMETRIC | None, optional): TIFF photometric interpretation. Defaults to None.
            Common values include 'MINISBLACK' (grayscale), 'RGB', 'PALETTE', 'MASK', etc.
        compression (str, optional): Compression method to use. Defaults to "zlib".
            Other options include 'lzw', 'jpeg', 'packbits', 'deflate', 'none', etc.
        compression_args (dict[str, Any] | None, optional): Additional arguments for the compression method.
            For example, {'level': 5} for zlib compression level.

    Returns:
        bytes: TIFF formatted bytes of the encoded image.

    Note:
        The function uses tifffile library to encode the image. The planarconfig and photometric
        parameters should match the image data structure to ensure correct encoding.
        For example, for an RGB image, photometric should be 'RGB' and planarconfig can be 'CONTIG'.
    """
    with io.BytesIO() as buffer:
        tifffile.imwrite(
            buffer,
            img,
            planarconfig=planarconfig,
            photometric=photometric,
            compression=compression,
            compressionargs=compression_args,
        )

        return buffer.getvalue()


def npz_codec_io(data_dict: Dict[str, np.ndarray], do_compression: bool = True) -> bytes:
    """Encodes a dictionary of NumPy arrays into NPZ file formatted bytes."""
    with io.BytesIO() as buffer:
        # Saves the dictionary. Keys become variable names in the NPZ file.
        if do_compression:
            np.savez_compressed(buffer, **data_dict, allow_pickle=True)
        else:
            np.savez(buffer, **data_dict, allow_pickle=True)

        return buffer.getvalue()


def mat_codec_io(data_dict: Dict[str, np.ndarray], do_compression: bool = True) -> bytes:
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


def npy_codec_io(img: np.ndarray, compress: bool = False) -> bytes:
    """Encodes a NumPy array into NPY file formatted bytes."""
    with io.BytesIO() as buffer:
        np.savez(buffer, img)
        return buffer.getvalue()


# --- Decoding Functions ---

# Image.MAX_IMAGE_PIXELS = None  # Disable the decompression bomb protection in PIL


def img_decode_io(img_bytes: bytes):
    with warnings.catch_warnings(record=True) as w:
        try:
            with io.BytesIO(img_bytes) as f:
                img = Image.open(f)
                # remove icc profile to avoid libpng print info
                # see https://github.com/ultralytics/ultralytics/issues/339#issuecomment-1691086802
                img.info.pop("icc_profile", None)
                try:
                    img = np.array(img.convert("RGB"))
                except Exception as e:
                    log_print(f"Error converting image to RGB: {e}", "error")
                    return None

            for warning in w:
                if issubclass(warning.category, Image.DecompressionBombWarning):
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


def string_decode_io(string_bytes: bytes) -> str:
    """Decodes bytes into a string."""
    return string_bytes.decode("utf-8")  # Assuming UTF-8 encoding for the string


def json_decode_io(json_bytes: bytes) -> dict | None:
    """Decodes bytes into a JSON dictionary."""
    try:
        return json.loads(json_bytes)
    except json.JSONDecodeError as e:
        log_print(f"Error decoding JSON: {e}", "error")
        return None


def yaml_decode_io(yaml_bytes: bytes) -> dict | None:
    """Decodes bytes into a YAML dictionary."""
    try:
        return yaml.load(yaml_bytes, Loader=yaml.FullLoader)
    except yaml.YAMLError as e:
        log_print(f"Error decoding YAML: {e}", "error")
        return None


def tiff_decode_io(
    tiff_bytes: bytes,
    use_out_param: bool = True,  # Flag to control using 'out' parameter
    backend: str = "tifffile",  # Backend to use for TIFF decoding
) -> np.ndarray:
    """
    Decodes TIFF formatted bytes into a NumPy array.
    Optionally reads metadata first to pre-allocate memory for the 'out' parameter.
    """
    if backend == "tifffile":
        __predefined_buf_size = 16 * 1024 * 1024
        __max_workers = 8

        if use_out_param:
            try:
                # Step 1: Read metadata using TiffFile to get shape and dtype from the first series
                expected_shape: tuple[int, ...]
                expected_dtype: np.dtype
                with io.BytesIO(tiff_bytes) as metadata_buffer:
                    with tifffile.TiffFile(metadata_buffer) as tif:
                        if not tif.series or not tif.series[0]:
                            raise ValueError("TIFF file contains no series or series[0] is invalid.")
                        current_series: tifffile.TiffPageSeries = tif.series[0]
                        expected_shape = current_series.shape
                        expected_dtype = np.dtype(current_series.dtype)  # Ensure it's a numpy.dtype

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
            except Exception:
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
    else:
        # use nvimgcodec backend

        # from nvidia.nvimgcodec import Decoder, DecodeSource
        # decoder = Decoder()
        # data = DecodeSource(tiff_bytes)
        # img = decoder.decode(data)

        raise NotImplementedError(
            f"TIFF decoding with backend '{backend}' is not implemented. Please use 'tifffile' backend."
        )


def mat_decode_io(mat_bytes: bytes) -> Dict[str, np.ndarray]:
    """Decodes MAT file formatted bytes into a dictionary of NumPy arrays."""
    with io.BytesIO(mat_bytes) as buffer:
        # loadmat returns a dictionary, potentially including header info
        mat_dict: Dict[str, Any] = scipy.io.loadmat(buffer)
    # Filter out MATLAB specific keys
    data_dict: Dict[str, np.ndarray] = {
        k: v
        for k, v in mat_dict.items()
        if not k.startswith("__")  # Remove keys like __header__, __version__, __globals__
    }
    return data_dict


def single_img_mat_decode_io(mat_bytes: bytes) -> np.ndarray:
    data: dict = mat_decode_io(mat_bytes)
    return data[list(data.keys())[0]]  # assume only one key


def safetensors_decode_io(
    safetensors_bytes: bytes,
    to_device=False,
    return_dict=False,
    ret_keys: list[str] | str | bool | None = None,
) -> dict[str, torch.Tensor] | torch.Tensor:
    """Decodes safetensors formatted bytes into a dictionary of PyTorch tensors."""
    # Directly loads the tensors from bytes.
    # Tensors will be loaded onto the device they were saved from,
    # or potentially CPU depending on safetensors implementation details.
    # Usually, it's best practice to move tensors to the desired device after loading.
    data_dict: Dict[str, torch.Tensor] = load_safetensors(safetensors_bytes)
    if to_device:
        for key, tensor in data_dict.items():
            data_dict[key] = tensor.to(torch.device("cuda"))

    if not return_dict:
        # deprecated
        return data_dict[list(data_dict.keys())[0]]  # assume only one key
    return extract_keys_from_data(data_dict, ret_keys=return_dict)


def npz_decode_io(
    npz_bytes: bytes,
    *,
    ret_keys: list[str] | str | bool | None = None,
) -> Dict[str, np.ndarray]:
    """Decodes NPZ file formatted bytes into a dictionary of NumPy arrays."""
    with io.BytesIO(npz_bytes) as buffer:
        # load returns a dictionary, potentially including header info
        npz_dict: Dict[str, Any] = np.load(buffer, allow_pickle=True)
        data_dict = {k: v.copy() for k, v in npz_dict.items() if not k.startswith("__")}

    return extract_keys_from_data(data_dict, ret_keys=ret_keys)


def npy_decode_io(npy_bytes: bytes) -> np.ndarray:
    return np.load(io.BytesIO(npy_bytes))


def extract_keys_from_data(data_dict: dict, ret_keys: list[str] | str | bool | None = None):
    if isinstance(ret_keys, str):
        return data_dict[ret_keys]
    elif isinstance(ret_keys, list):
        return {k: data_dict[k] for k in ret_keys}
    elif ret_keys is False:
        # false mean return the first one value
        k = list(data_dict.keys())[0]
        return data_dict[k]
    else:  # true or none
        return data_dict


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


def is_mat_file(file_path: str) -> bool:
    """Check if the file is a MATLAB file based on its extension."""
    return file_path.lower().endswith((".mat"))


def is_rgb_file(file_path: str) -> bool:
    """Check if the file is an RGB image based on its extension."""
    return file_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".img_content"))


def is_text_file(file_path: str) -> bool:
    return file_path.lower().endswith((".caption", ".txt", ".img_name", ".caption.json"))


def is_encoded_file(file_path: str) -> bool:
    return file_path.lower().endswith((".safetensors", ".npy", ".npz"))


def is_config_file(file_path: str) -> bool:
    return file_path.lower().endswith((".yaml", ".yml", ".json", "jsonl"))


def is_img_file(file_path: str) -> bool:
    return is_rgb_file(file_path) or is_tiff_file(file_path) or is_mat_file(file_path)


def wids_caption_decode(sample: dict):
    _keys = list(sample.keys())
    for k in _keys:
        if is_text_file(k):
            sample[k] = wids_caption_decode(sample.pop(k).getvalue())

    return sample


def wids_config_decode(sample: dict):
    _keys = list(sample.keys())
    for k in _keys:
        if is_config_file(k):
            sample[k] = wids_config_decode(sample.pop(k).getvalue())

    return sample


def wids_image_decode(
    sample: dict[str, Any],
    to_neg_1_1: bool = True,
    permute: bool | Literal["auto"] = True,
    resize_fn: str = "interpolate",
    interp_mode: str = "bilinear",
    resize: int | tuple[int, int] | None = None,
    process_img_keys: str | list[str] | Literal["ALL"] | None = "img",
    mat_is_single_img: bool = True,
    per_channel: bool = False,
    norm_type: str = "clip_zero_div",
    quantile_clip=1.0,
    mannual_img_min_max=None,
    _disable_norm=False,
) -> Dict[str, Any]:
    def img_process_fn(img, key: str, resize=resize):
        img = img.astype(np.float32)

        # check if is image
        if img.ndim not in (2, 3) or not isinstance(img, (np.ndarray, torch.Tensor)):
            log_print(
                f"[Wids Decoder]: key {key} is not an image, shaped as {img.shape}, typed as {img.dtype}, skipping.",
                warn_once=True,
            )
            return img

        if permute is True and img.ndim == 3:
            # Permute the image dimensions from HWC to CHW
            img = np.transpose(img, (2, 0, 1))  # (c, h, w)
        elif permute == "auto":
            h, w, c = img.shape
            # assume c is the smallest dimension
            if c < h and c < w:  # is (h, w, c) shape
                img = np.transpose(img, (2, 0, 1))  # to CHW
            log_print(
                f"[Wids Decoder]: auto permute is enabled, the image shape is ({h, w, c}) -> ({tuple(img.shape)})",
                once=True,
            )

        # to Tensor
        img = torch.as_tensor(img)

        # Normalize image
        if not _disable_norm:
            img = img.type(torch.float32)  # ensure float32
            img, *_ = normalize_image_(
                img,
                norm_type=norm_type,
                per_channel=per_channel,
                to_neg_1_1=to_neg_1_1,
                q_clip=quantile_clip,
                mannual_img_min_max=mannual_img_min_max,
            )

        # Resize
        if resize is not None:
            resize = to_2tuple(resize)
            img = img.unsqueeze(0)  # add batch dim
            if resize_fn == "interpolate":
                img = torch.nn.functional.interpolate(
                    img,
                    size=resize,
                    mode=interp_mode,
                    align_corners=False if interp_mode == "bilinear" else None,
                ).squeeze(0)
            else:
                raise NotImplementedError(f"not implmented function {resize_fn}")

        return img

    def get_decode_fn(k: str):
        if is_rgb_file(k):
            return img_decode_io
        elif is_tiff_file(k):
            return tiff_decode_io
        elif is_mat_file(k):
            return single_img_mat_decode_io if mat_is_single_img else mat_decode_io
        else:
            raise ValueError(f"Unsupported image file type: {k}")

    ## Entry

    # 1. Decode image files
    img_in_sample = False
    keys = list(sample.keys())
    for key in keys:
        if key.startswith("__"):
            continue
        if is_img_file(key):
            img_in_sample = True
            decode_fn = get_decode_fn(key)
            img = decode_fn(sample.pop(key).getvalue())
            sample[key] = img

    # 2. Process image files
    if img_in_sample and process_img_keys is not None:
        if isinstance(process_img_keys, str):
            if process_img_keys == "ALL":
                process_img_keys = [k for k in sample.keys() if is_img_file(k)]
            else:
                process_img_keys = [process_img_keys]
        for key in process_img_keys:
            img = sample[key]
            sample[key] = img_process_fn(img, key)
    else:
        assert process_img_keys is None, "process_img_keys should be None when there is no image modality"

    return sample


def wids_latent_decode(sample: dict[str, Any], return_dict=False):
    # three types of latents: npz, safetensors, and npy

    call_fns = {
        "npz": partial(npz_decode_io, ret_keys=return_dict),
        "safetensors": partial(safetensors_decode_io, return_dict=return_dict),
        "npy": npy_decode_io,
    }

    keys = list(sample.keys())
    for k in keys:
        if is_encoded_file(k):
            name_ck = k.split(".")
            assert len(name_ck) == 2, f"Invalid key format: {k}, should be name.extension"
            name, ck = name_ck
            latent = call_fns[ck](sample.pop(k).getvalue())  # may raise KeyError
            sample[name] = latent

    return sample


def wids_caption_embed_decode(sample: dict[str, Any], max_length=300) -> dict[str, torch.Tensor | str]:
    # decode caption (str), valid_length (int), and caption_feature (torch.Tensor), and
    # attention_mask (torch.Tensor, optional)

    _keys = list(sample.keys())

    # has captions and caption features
    caption_json_key: str | None = None
    features_key: str | None = None
    for k in _keys:
        if "caption" in k and k.endswith((".json", ".jsonl", ".txt")):
            caption_json_key = k
        elif "features" in k and k.endswith(".safetensors"):
            features_key = k

    assert caption_json_key is not None or features_key is not None, (
        "Sample must contain either caption JSON or features key."
    )

    # caption text
    if caption_json_key is not None:
        caption = json_decode_io(sample.pop(caption_json_key).getvalue())
        assert caption is not None, "Caption JSON decoding failed."
        sample["caption"] = caption["caption"]
        sample["valid_length"] = int(caption.get("valid_length", len(sample["caption"])))
    else:
        raise ValueError("Captions not found.")

    # features
    embeds: dict[str, torch.Tensor] = {}
    if features_key is not None:
        embeds = safetensors_decode_io(sample.pop(features_key).getvalue(), return_dict=True)
        # pad right
        cap_f = embeds["caption_feature"].squeeze(0)  # [n, d]
        if cap_f.shape[0] < max_length:
            cap_f_pad = torch.nn.functional.pad(cap_f, (0, 0, 0, max_length - cap_f.shape[0]), value=0.0)
        else:
            cap_f_pad = cap_f[:, :max_length]
        sample["caption_feature"] = cap_f_pad  # bfloat16
    else:
        sample["caption_feature"] = None

    # text mask
    if "attention_mask" in embeds:
        sample["attention_mask"] = embeds["attention_mask"].float()  # [max_length,]
    else:
        vl = sample["valid_length"]
        attn_mask = torch.zeros((max_length,), dtype=torch.float32)
        attn_mask[:vl] = 1.0  # effective token is 1 in mask
        sample["attention_mask"] = attn_mask

    return sample
