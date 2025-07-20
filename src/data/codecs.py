import io
import json
import warnings
from typing import Any, Dict, Literal, cast

import numpy as np
import scipy.io
import tifffile
import torch
import yaml
from PIL import Image
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

from src.utilities.logging import log_print, once

from .utils import to_n_tuple

# --- Encoding Functions ---


def py_obj_to_jsonl(metadata: dict):
    buffer = io.BytesIO()
    buffer.write(json.dumps(metadata).encode("utf-8") + b"\n")

    return buffer.getvalue()


def rgb_codec_io(img, format="jpeg", **kwargs):
    buffer = io.BytesIO()
    assert img.ndim == 3 and img.shape[-1] == 3, "Image must be RGB"
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
            compressionargs=compression_args,
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
    # import time
    # t1 = time.perf_counter()

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
                            raise ValueError(
                                "TIFF file contains no series or series[0] is invalid."
                            )
                        current_series: tifffile.TiffPageSeries = tif.series[0]
                        expected_shape = current_series.shape
                        expected_dtype = np.dtype(
                            current_series.dtype
                        )  # Ensure it's a numpy.dtype

                # Step 2: Pre-allocate the output array
                output_array: np.ndarray = np.empty(
                    expected_shape, dtype=expected_dtype
                )

                # Step 3: Read image data into the pre-allocated array
                # Use a new BytesIO object for imread to ensure the stream is at the beginning
                with io.BytesIO(tiff_bytes) as data_buffer:
                    tifffile.imread(
                        data_buffer,
                        out=output_array,
                        buffersize=__predefined_buf_size,  # Example buffer size
                        maxworkers=__max_workers,  # Example max workers
                    )
                # t2 = time.perf_counter()
                # log_print(
                #     "tifffile decode time: {:.2f} seconds".format(t2 - t1), "debug"
                # )
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
                # t2 = time.perf_counter()
                # log_print(
                #     "tifffile decode time: {:.2f} seconds".format(t2 - t1), "debug"
                # )
                return img
        else:
            # Original behavior if use_out_param is False
            with io.BytesIO(tiff_bytes) as buffer:
                img: np.ndarray = tifffile.imread(
                    buffer,
                    buffersize=__predefined_buf_size,
                    maxworkers=__max_workers,
                )
            # t2 = time.perf_counter()
            # log_print("tifffile decode time: {:.2f} seconds".format(t2 - t1), "debug")
            return img
    else:
        # use nvimgcodec backend

        # from nvidia.nvimgcodec import Decoder, DecodeSource
        # decoder = Decoder()
        # data = DecodeSource(tiff_bytes)
        # img = decoder.decode(data)

        raise NotImplementedError(
            f"TIFF decoding with backend '{backend}' is not implemented. "
            "Please use 'tifffile' backend."
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
        if not k.startswith(
            "__"
        )  # Remove keys like __header__, __version__, __globals__
    }
    return data_dict


def safetensors_decode_io(
    safetensors_bytes: bytes,
    to_device=False,
    return_dict=False,
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
        return data_dict[list(data_dict.keys())[0]]  # assume only one key
    return data_dict


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
        np.savez(buffer, img)
        return buffer.getvalue()


def npy_decode_io(npy_bytes: bytes) -> np.ndarray:
    return np.load(io.BytesIO(npy_bytes))


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
    return file_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".img_content"))


def is_text_file(file_path: str) -> bool:
    return file_path.lower().endswith((".caption", ".txt", ".img_name"))


def is_config_file(file_path: str) -> bool:
    return file_path.lower().endswith((".yaml", ".yml", ".json", "jsonl"))


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
    to_neg_1_1=True,
    permute: bool | Literal["auto"] = True,
    resize_fn: str = "interpolate",
    resize: int | tuple[int, int] | None = None,
    process_img_keys: str | list[str] | Literal["ALL"] | None = "img",
):
    def _image_modality_post_process(sample: dict[str, Any], key: str = "img"):
        img = sample[key].astype(np.float32)
        # check if is image
        if img.ndim not in (2, 3) or not isinstance(img, np.ndarray):
            log_print(
                f"[Wids Decoder]: key {key} is not an image, shaped as {img.shape}, typed as {img.dtype}, skipping.",
                warn_once=True,
            )
            return sample

        _img_max = img.max()
        if _img_max < 1e-4:
            img = np.zeros_like(img) if not to_neg_1_1 else np.ones_like(img) / 2
        else:
            img = img / (_img_max + 1e-6)

        if to_neg_1_1:
            img = (img * 2 - 1).clip(-1.0, 1.0)  # to [-1, 1]

        if permute is True:
            # Permute the image dimensions from HWC to CHW
            img = np.transpose(img, (2, 0, 1))
        elif permute == "auto":
            h, w, c = img.shape
            # assume c is the smallest dimension
            if c < h and c < w:  # is (h, w, c) shape
                img = np.transpose(img, (2, 0, 1))  # to CHW
            once(log_print)(
                f"[Wids Decoder]: auto permute is enabled, the image shape is ({h, w, c}) -> ({tuple(img.shape)})",
            )

        if resize is not None:
            if permute:
                img = torch.as_tensor(img)[None].float()
            else:
                img = torch.as_tensor(np.transpose(img, (1, 2, 0)))[None].float()

            sz = resize
            if isinstance(resize, int):
                sz = to_n_tuple(sz, 2)

            if resize_fn == "interpolate":
                img = torch.nn.functional.interpolate(
                    img, size=sz, mode="bilinear", align_corners=False
                )[0]
            else:
                raise NotImplementedError(f"not implmented function {resize_fn}")

            if not permute:
                img = img.permute(1, 2, 0)

        sample[key] = img

        return sample

    _img_flag = False

    _keys = list(sample.keys())
    for key in _keys:
        if key.startswith("__"):
            continue

        if is_tiff_file(key):
            sample[key] = tiff_decode_io(sample.pop(key).getvalue())
            _img_flag = True
        elif is_rgb_file(key):
            sample[key] = img_decode_io(sample.pop(key).getvalue())
            _img_flag = True
        elif is_text_file(key):
            sample[key] = string_decode_io(sample.pop(key).getvalue())
        else:
            raise ValueError(f"Unsupported file type for {key}")

    if _img_flag and process_img_keys is not None:
        if isinstance(process_img_keys, str):
            if process_img_keys == "ALL":
                process_img_keys = [
                    k for k in sample.keys() if is_rgb_file(k) or is_tiff_file(k)
                ]
            else:
                process_img_keys = [process_img_keys]

        for key in process_img_keys:
            sample = _image_modality_post_process(sample, key)
    else:
        assert process_img_keys is None, (
            "process_img_keys should be None when there is no image modality"
        )

    return sample


def wids_latent_decode(sample: dict[str, Any]):
    # three types of latents: npz, safetensors, and npy

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

    return sample


def wids_caption_embed_decode(
    sample: dict[str, Any], max_length=300
) -> dict[str, torch.Tensor | str]:
    # decode caption (str), valid_length (int), and caption_feature (torch.Tensor), and
    # attention_mask (torch.Tensor, optional)

    if ".caption.json" in sample:
        caption = json_decode_io(sample.pop(".caption.json").getvalue())
        assert caption is not None, "Caption JSON decoding failed."

        sample["caption"] = caption["caption"]
        sample["valid_length"] = int(caption["valid_length"])

    if ".features.safetensors" in sample:
        name = ".features.safetensors"
    elif ".features_and_mask.safetensors" in sample:
        name = ".features_and_mask.safetensors"
    else:
        name = None

    if name is not None:
        embeds = safetensors_decode_io(sample.pop(name).getvalue(), return_dict=True)
        embeds = cast(Dict[str, torch.Tensor], embeds)
        sample["caption_feature"] = embeds["caption_feature"]  # bfloat16
        if "attention_mask" in embeds:
            sample["attention_mask"] = embeds["attention_mask"].float()
        else:
            vl = sample["valid_length"]
            attn_mask = torch.zeros((max_length,), dtype=torch.float32)
            attn_mask[:vl] = 1.0
            sample["attention_mask"] = attn_mask

    return sample
