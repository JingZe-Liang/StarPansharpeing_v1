import glob
import io
import math
import sys
from pathlib import Path
from typing import Iterable, Literal, cast

import einops
import natsort
import numpy as np
import safetensors
import safetensors.torch
import tifffile
import torch
import webdataset as wds
from PIL import Image
from scipy.io import loadmat
from torchvision.io import read_video
from tqdm import tqdm

from src.utilities.logging.print import configure_logger, logger

Image.MAX_IMAGE_PIXELS = None  # disable the warning for large images

# logger.remove()
# logger.add(
#     sink=sys.stdout,
#     level="DEBUG",
#     format="<green>[{time:MM-DD HH:mm:ss}]</green> <cyan>{name}</cyan> <level>[{level}]</level> - <level>{message}</level>",
#     backtrace=True,
#     diagnose=True,
#     colorize=True,
# )

configure_logger(level="info", auto=False)


_numpy_dtype_to_tensor = {
    "uint8": torch.uint8,
    "uint16": torch.uint16,
    "uint32": torch.uint32,
    "uint64": torch.uint64,
}

_dtype_max_dtypes = [np.uint8, np.uint16, np.uint32, np.uint64]
_dtype_max_vals = [
    np.iinfo(np.uint8).max,
    np.iinfo(np.uint16).max,
    np.iinfo(np.uint32).max,
    np.iinfo(np.uint64).max,
]

BANDS_NAME = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B10",
    "B11",
    "B12",
]

# global index that saved images count
total_img_saved_count = 0


def background_is_all_zero(
    img: np.ndarray | torch.Tensor,
    norm=True,
    ratio: float = 0.7,
    thresh: int = 30,  # 5 in uint8, 0 .. 255
    is_hwc=True,
):
    img = img.astype(np.float32) if isinstance(img, np.ndarray) else img.float()
    if norm:
        img = img.clip(0, None)
        img = img / img.max() * 255.0

    # [h, w, c]
    if img.ndim == 2:
        img_gray = img
    elif img.ndim == 3:
        dim = -1 if is_hwc else 0
        img_gray = img.max(dim).values
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    pix_total = img.shape[0] * img.shape[1]
    pix_zero = (img_gray < thresh).sum().item()  # 0 .. 255

    zero_ratio = pix_zero / pix_total
    # logger.debug(f"Background ratio: {zero_ratio:.4f}")
    return zero_ratio > ratio


def to_batched(img: torch.Tensor, is_hwc=False):
    img = torch.as_tensor(img)

    if img.ndim == 2:
        img = img.unsqueeze(0).unsqueeze(0)
        inverse = lambda x: x[0, 0]
    elif img.ndim == 3:
        if is_hwc:
            img = img.permute(2, 0, 1)  # c x h x w
            img = img.unsqueeze(0)
            inverse = lambda x: x[0].permute(1, 2, 0)  # h x w x c
        else:
            # c x h x w
            img = img.unsqueeze(0)
            inverse = lambda x: x[0]
    else:
        assert img.ndim == 4, "img must be 4-dimension"
        inverse = lambda x: x

    return img, inverse


def get_einops_pattern(tensor_dim, reduce_dims):
    # tensor_dim: int or sequence; reduce_dims: tuple of dims to reduce
    dims = tensor_dim if isinstance(tensor_dim, int) else len(tensor_dim)
    # normalize negative dims and sort
    rd = sorted(d if d >= 0 else dims + d for d in reduce_dims)
    # generate a name for each axis, e.g. 'a','b','c',...
    axes = [chr(ord("a") + i) for i in range(dims)]
    # in the output pattern, replace reduced axes with '1'
    out = [("1" if i in rd else axes[i]) for i in range(dims)]
    return f"{' '.join(axes)} -> {' '.join(out)}"


def per_channel_normalize(img: torch.Tensor | np.ndarray, c_dim: tuple = (-2, -1)):
    if isinstance(img, (torch.Tensor, np.ndarray)):
        dims = img.dim() if torch.is_tensor(img) else img.ndim
        pattern = get_einops_pattern(dims, c_dim)

        min_c = einops.reduce(img, pattern=pattern, reduction="min")
        max_c = einops.reduce(img, pattern=pattern, reduction="max")

        img = (img - min_c) / (
            max_c - min_c + 1e-8
        )  # Add small epsilon to avoid division by zero

        # Use torch.clamp for PyTorch tensors and np.clip for NumPy arrays to ensure compatibility
        if torch.is_tensor(img):
            img = img.clamp(0, 1)
        else:
            img = np.clip(img, 0, 1)
        raise ValueError(f"Unsupported image type: {type(img)}, shape: {img.shape}")

    return img


def per_channel_add_min(img: torch.Tensor | np.ndarray, hw_dim: tuple = (-2, -1)):
    if isinstance(img, (torch.Tensor, np.ndarray)):
        dims = img.dim() if torch.is_tensor(img) else img.ndim
        pattern = get_einops_pattern(dims, hw_dim)

        min_c = einops.reduce(img, pattern=pattern, reduction="min")
        img = img - min_c
    else:
        raise ValueError(f"Unsupported image type: {type(img)}, shape: {img.shape}")

    return img


def sliding_window(
    image,
    patch_size,
    stride,
    is_yield=True,
    pad_type: str | None = "resize",
    is_hwc=True,
    check_background=False,
):
    """
    image: [h, w, c] or [c, h, w]
    output: list of patches, each patch is [h, w, c] or [c, h, w]
    """
    patches = []
    coords = []
    shape = image.shape
    if not is_hwc:
        # [c, h, w]
        h, w = shape[1:]
    else:
        # [h, w, c]
        h, w = shape[:-1]

    # 计算 padding 大小
    pad_h = max(0, patch_size[0] - (h % stride[0])) if (h % stride[0]) != 0 else 0
    pad_w = max(0, patch_size[1] - (w % stride[1])) if (w % stride[1]) != 0 else 0

    if pad_type is not None and isinstance(image, np.memmap):
        logger.warning("Image is a memmap, padding will not be applied.")

    elif pad_type == "pad":
        # 如果pad大小大于h,w的1/4，那么不要多出来的部分；小于1/4就正常pad
        if pad_h > h / 4:
            pad_h = 0
            logger.warning(
                f"Pad height {pad_h} is greater than half of image height {h}, skipping padding"
            )

        if pad_w > w / 4:
            pad_w = 0
            logger.warning(
                f"Pad width {pad_w} is greater than half of image width {w}, skipping padding"
            )

        # 对图像进行 padding
        if pad_h > 0 or pad_w > 0:
            image = torch.as_tensor(image) if isinstance(image, np.ndarray) else image
            image, shape_inverse = to_batched(image, is_hwc)
            orig_dtype = image.dtype
            image = torch.nn.functional.pad(
                image.float(), (0, pad_w, 0, pad_h), mode="constant", value=0.0
            ).to(orig_dtype)
            image = shape_inverse(image)
    elif pad_type == "resize":
        if pad_h < h / 2:
            target_h = ((h + stride[0] - 1) // stride[0]) * stride[0]
        else:
            target_h = h
        if pad_w < w / 2:
            target_w = ((w + stride[1] - 1) // stride[1]) * stride[1]
        else:
            target_w = w

        if target_h != h or target_w != w:
            logger.debug(f"Resizing image from ({h}, {w}) to ({target_h}, {target_w})")
            image = torch.as_tensor(image) if isinstance(image, np.ndarray) else image
            image, shape_inverse = to_batched(image, is_hwc)
            orig_dtype = image.dtype

            image = torch.nn.functional.interpolate(
                image.float(),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).to(orig_dtype)
            image = shape_inverse(image)

    shape = image.shape
    logger.debug(f"img shape: {shape}")
    h_padded, w_padded = shape[:-1] if is_hwc else shape[-2:]
    logger.debug(
        f"clip patch with shape: {(h_padded, w_padded)}, patch_size: {patch_size}, stride: {stride}",
    )
    rows = list(range(0, h_padded, stride[0]))
    cols = list(range(0, w_padded, stride[1]))
    logger.debug(f"Got {len(rows) * len(cols)} patches")

    # 确保最后一个 patch 覆盖到边界
    for i in rows:
        for j in cols:
            # 如果超出边界，则调整 patch 的起始位置
            i_end = min(i + patch_size[0], h_padded)
            j_end = min(j + patch_size[1], w_padded)
            i_start = i_end - patch_size[0]
            j_start = j_end - patch_size[1]

            if not is_hwc:
                img_slide = image[..., i_start:i_end, j_start:j_end]
            else:
                img_slide = image[i_start:i_end, j_start:j_end, ...]

            if isinstance(img_slide, np.memmap):
                # img_slide = np.array(img_slide)
                logger.debug(
                    f"Copying memmap to numpy array at patch ({i_start}:{i_end}, {j_start}:{j_end})"
                )
                img_slide = img_slide.copy()  # Ensure we have a copy, not a view

            img_slide = torch.as_tensor(img_slide)
            # Check the patch is all zero
            if (
                background_is_all_zero(img_slide.cuda(), norm=True, is_hwc=is_hwc)
                and check_background
            ):  # on device check
                logger.warning(
                    f"Background is all zero at patch ({i_start}:{i_end}, {j_start}:{j_end})"
                )
                if is_yield:
                    yield None, None
                else:
                    patches.append(None)
                    coords.append(None)
                continue

            if is_yield:
                yield img_slide, (i_start, j_start)
            else:
                patches.append(img_slide)
                coords.append((i_start, j_start))

    return patches, coords


def resize_img(
    img: np.ndarray | torch.Tensor,
    size: int | tuple,
    is_hwc: bool = True,
):
    assert not isinstance(img, np.memmap), "np.memmap is not supported for resizing"

    # to batch
    img = torch.as_tensor(img) if isinstance(img, np.ndarray) else img
    img, shape_inverse = to_batched(img, is_hwc=is_hwc)
    if background_is_all_zero(img[0], norm=True, is_hwc=is_hwc):
        logger.warning("Image is all zero, skipping resizing.")
        return [(None, None)]

    dtype = img.dtype

    shape = img.shape
    logger.debug(f"resize image from {shape} to {size}")

    # resize
    img = torch.nn.functional.interpolate(
        img.float(),
        size=(size, size) if isinstance(size, int) else size,
        mode="bilinear",
        align_corners=False,
    ).to(dtype)
    img = shape_inverse(img)

    dummy_out = None
    return [(torch.as_tensor(img), dummy_out)]


def resize_or_clip_img(
    img: np.ndarray | torch.Tensor,
    resize_size,
    patch_size,
    stride,
    pad_type="resize",
    is_hwc: bool = True,
):
    if is_hwc:
        h, w = img.shape[:2]
    else:
        h, w = img.shape[-2:]

    h_max = w_max = 1024
    # if h > h_max and w > w_max:
    #     ratio = w / h
    #     if 3 / 4 < ratio < 4 / 3:
    #         # resize
    #         yield from resize_img(img, resize_size, is_hwc)
    #     else:
    #         # clip
    #         yield from sliding_window(
    #             img, patch_size, stride, pad_type=pad_type, is_hwc=is_hwc
    #         )
    if h > h_max or w > w_max:
        # clip
        yield from sliding_window(
            img, patch_size, stride, pad_type=pad_type, is_hwc=is_hwc
        )
    else:
        # yield (torch.as_tensor(img), None)

        h = math.ceil(h / 16) * 16
        w = math.ceil(w / 16) * 16

        yield from resize_img(img, (h, w), is_hwc=is_hwc)


def merge_patches(patches, coords, original_shape, patch_size, stride):
    """
    简化版 merge patches，移除 padding
    """
    h, w = original_shape[-2:]

    # 计算 padding 大小
    pad_h = max(0, patch_size[0] - (h % stride[0])) if (h % stride[0]) != 0 else 0
    pad_w = max(0, patch_size[1] - (w % stride[1])) if (w % stride[1]) != 0 else 0

    h_padded = h + pad_h
    w_padded = w + pad_w

    # 初始化合并图像和计数图
    device = patches[0].device
    dtype = patches[0].dtype
    merged_image = torch.zeros(
        (original_shape[0], original_shape[1], h_padded, w_padded),
        device=device,
        dtype=dtype,
    )
    count_map = torch.zeros(
        (original_shape[0], original_shape[1], h_padded, w_padded),
        device=device,
        dtype=dtype,
    )

    for patch, (i, j) in zip(patches, coords):
        merged_image[..., i : i + patch_size[0], j : j + patch_size[1]] += patch
        count_map[..., i : i + patch_size[0], j : j + patch_size[1]] += 1

    # 避免除以零的情况
    merged_image = torch.where(
        count_map > 0, merged_image / count_map, torch.tensor(0.0)
    )

    # 移除 padding
    if pad_h > 0 or pad_w > 0:
        merged_image = merged_image[..., :h, :w]

    return merged_image


def read_image(
    img_path: str | Path,
    *,
    mat_load_key="I",
    verbose=True,
    tiff_read_mode="array",
    tiff_bands_seperated: bool = False,
    force_to_dtype: np.dtype | None = None,
):
    if isinstance(img_path, str):
        img_path = Path(img_path)

    if verbose:
        logger.info("reading image from: {}", img_path.as_posix())

    if img_path.suffix == ".mat":
        try:
            d = loadmat(img_path)
            key_ = list(d.keys())[-1]
            img = d[key_]
        except NotImplementedError as e:
            logger.warning(
                f"Mat file is not supported by scipy.io.loadmat reading: {e}. Try to "
                "read using h5py."
            )
            import h5py

            with h5py.File(img_path, "r") as f:
                key_ = list(f.keys())[-1]
                img = f[key_][:]
    elif img_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
        try:
            img = np.array(Image.open(img_path))
            # may be mask
            if img.ndim == 2 and img_path.suffix.lower() == ".png":
                img = img[..., None]
        except Exception as e:
            logger.warning(f"Failed to load image from: {img_path.as_posix()}. {e}")
            return None
    elif img_path.suffix == ".npy":
        img = np.load(img_path)
    elif img_path.suffix.lower() in [".tif", ".tiff"] and tiff_bands_seperated:
        # assume img_path endswith *B01.tif
        basic_name = (
            img_path.name
        )  # data/HLS/HLS.L30.T58UEF.2025152T235555.v2.0.B01.tif
        uni_tif_paths = []

        for band_name in BANDS_NAME:
            parts = basic_name.split(".")  # replace B01 with band_name
            name = ".".join(parts[:-2]) + "." + band_name + ".tif"
            band_path = img_path.parent / name
            if not band_path.exists():
                logger.warning(f"band {band_name} not found in: {img_path}")
                return None
            uni_tif_paths.append(band_path)
        # read all bands
        bands_imgs = []
        for p in uni_tif_paths:
            try:
                img = tifffile.imread(p)
            except Exception as e:
                logger.warning(f"failed to load image from: {p}. {e}")
                return None
            img = np.clip(img, 0, None)
            bands_imgs.append(img)
        img = np.stack(bands_imgs, axis=-1)  # [h, w, c]
    elif img_path.suffix.lower() in [".tif", ".tiff"]:
        # memmap
        tif = tifffile.TiffFile(img_path)
        try:
            img = (
                tif.asarray(out="memmap")
                if tiff_read_mode == "memmap"
                else tif.asarray()
            )
        except Exception as e:
            logger.warning(f"failed to load image from: {img_path.as_posix()}. {e}")
            return None
    elif img_path.suffix.lower() in [".mp4"]:
        from torchvision.io import video_reader

        # read video frame
        reader = video_reader.VideoReader(img_path.as_posix(), num_threads=2)
        frames = []
        for frame in reader:
            pts = frame["pts"]
            if pts % 0.5 <= 0.001:
                data = (
                    frame["data"].numpy().transpose(1, 2, 0)
                )  # [c, h, w] -> [h, w, c]
                frames.append(data)

            if pts > 60:
                logger.warning("the frames are too many, stop reading")
        return frames
    else:
        raise ValueError(f"Unsupported image format: {img_path.suffix}")

    if force_to_dtype is not None:
        try:
            assert isinstance(img, np.ndarray), "img must be a numpy array"
            img = img.astype(force_to_dtype)
        except Exception as e:
            logger.warning(
                f"Failed to convert image to {force_to_dtype} dtype: {e}. "
                "Please check the image data type and the target dtype."
            )

    return img


def postprocess_img(
    img: np.ndarray | torch.Tensor,
    normalize: bool = True,
    rescale: Literal["clamp", "min_max", "add_min"] = "clamp",
    to_tensor=True,
    transpose: bool = True,
):
    if to_tensor:
        # [H, W, C] -> [1, C, H, W]
        img = torch.as_tensor(img)
        if img.ndim == 2:
            # is pan image
            img = img[..., None]
        if transpose:
            img = img.permute(2, 0, 1)
        img = img.unsqueeze(0)

    if normalize:
        assert torch.is_tensor(img), "img must be a tensor"

        img = img.to(torch.float32)
        img_max = img.max()
        if (img_min := img.min()) < 0:
            logger.warning(f"Image min value is {img_min}, {rescale} to 0")

        if rescale == "clamp":
            img = img.clamp(min=0)
            img = img / img_max
        elif rescale == "min_max":
            img = per_channel_normalize(
                img, c_dim=(-2, -1) if img.ndim == 4 else (0, 1)
            )
        else:
            raise ValueError(f"Unsupported rescale method: {rescale} for {normalize=}")
    else:
        # NOTE: To be honest, I am not sure if directly clamping is suitable for hyperspectral image
        # because some sensors may output the negtive value

        img = img.float() if torch.is_tensor(img) else img.astype(np.float32)

        if rescale == "clamp":
            img = img.clamp(min=0)
        elif rescale == "add_min":
            img = per_channel_add_min(img, hw_dim=(-2, -1) if img.ndim == 4 else (0, 1))
        else:
            raise ValueError(f"Unsupported rescale method: {rescale} for {normalize=}")

        img_max = img.to(torch.int32).max()

    return img, img_max


def to_suitable_dtype_img(
    img: torch.Tensor | np.ndarray,
    img_max: torch.Tensor,
    dtype: np.dtype | None = None,
    is_normed: bool = True,
):
    def determine_fit_dtype(img_max, dtype):
        if dtype is not None:
            return dtype

        global _dtype_max_dtypes, _dtype_max_vals, _numpy_dtype_to_tensor

        for dt, max_val in zip(_dtype_max_dtypes, _dtype_max_vals):
            if img_max <= max_val:
                # logger.info(f"dtype fit determined as {dt}")
                return dt

        raise ValueError(f"Image max value {img_max} is too large to fit any dtype")

    if dtype not in [np.float32, np.float16]:
        dtype = determine_fit_dtype(img_max, dtype)

    if is_normed:
        img = img * img_max

    if torch.is_tensor(img):
        assert img.shape[0] == 1, "batch size must be 1"
        img = img.cpu().permute(0, 2, 3, 1)[0].numpy().astype(dtype)
    elif isinstance(img, np.ndarray):
        img = img.astype(dtype)
    else:
        raise ValueError(f"Unsupported image type: {type(img)}")

    return img, dtype


def img_saver_backend_compact_with_wds(
    img: np.ndarray,
    extension: str,
    tiff_compression_type: Literal["zlib", "lzw", "jpeg", "jpeg2000", "none"] = "zlib",
    tiff_jpg_irreversible: bool = False,
    jpeg_quality: int = 90,
):
    assert extension in [
        "npy",
        "tiff",
        "jpeg",
        "png",
        "safetensors",
        "webdataset",
    ], "only support npy/tif/jpg/jpeg/png format or letting webdataset to encode"
    byte_io = io.BytesIO()

    assert isinstance(img, np.ndarray), "img must be a numpy array"
    assert img.ndim in (2, 3), "img must be 2D or 3D"

    if extension == "npy":
        np.save(byte_io, img)
    elif extension == "tiff":
        # see imagecodecs.jpeg2k_encode signature
        if tiff_compression_type == "jpeg2000":
            compression_args = {
                "reversible": not tiff_jpg_irreversible,
                "level": jpeg_quality,
                # "codecformat": "jp2",
            }
        else:
            compression_args = None

        # tiff write in memory
        tifffile.imwrite(
            byte_io,
            img,
            shape=img.shape,
            compression=tiff_compression_type,
            compressionargs=compression_args,
        )
    elif extension in ["jpeg", "png"]:
        assert img.dtype == np.uint8, "img must be uint8"
        options = {"format": extension}
        if extension == "jpeg":
            options["quality"] = jpeg_quality
        try:
            # only save RGB file for jpeg encoding
            # if want to save mask/label file, use png encoding
            Image.fromarray(img.squeeze()).convert("RGB").save(byte_io, **options)
        except Exception as e:
            print(f"Error saving {extension} img with shape {img.shape}: {e}")
            raise RuntimeError(
                f"Failed to save image with shape {img.shape} and dtype {img.dtype} as {extension}"
            ) from e
    elif extension == "safetensors":
        byte = safetensors.torch.save(
            {"img": torch.as_tensor(img, dtype=_numpy_dtype_to_tensor[img.dtype.name])}
        )
        return byte
    elif extension == "webdataset":
        # left for webdataset to encode
        return img
    else:
        raise ValueError(f"Unsupported image format: {extension}")

    return byte_io.getvalue()


def tiff_decoder(key, x):
    if key.endswith(".tiff"):
        return tifffile.imread(io.BytesIO(x))
    else:
        return x


def clip_img_to_webdataset(
    sink,
    img_path: str | Path,
    img_clip_size: tuple[int, int] = (512, 512),
    img_stride: tuple[int, int] = (512, 512),
    img_resize: int = 512,
    save_backend: str = "tiff",
    transpose: bool = True,
    read_fn_kwargs: dict = {},
    save_kwargs: dict = {"jpg_quality": 90},
    use_yield: bool = True,
    process_img_type: str = "clip",
    rescale: str = "clamp",
    force_save_dtype: str | np.dtype = "auto",
    add_global_index: bool = False,
    assert_channel_n: int | None = None,
):
    def slide_image_and_save(img, img_name):
        if process_img_type == "clip":
            slide_g = sliding_window(
                img,
                img_clip_size,
                img_stride,
                is_yield=use_yield,
                is_hwc=transpose,
            )
        elif process_img_type == "resize":
            slide_g = resize_img(
                img,
                size=img_resize,
                is_hwc=transpose,
            )
        elif process_img_type == "clip_resize":
            slide_g = resize_or_clip_img(
                img,
                img_resize,
                img_clip_size,
                img_stride,
                pad_type="resize",
                is_hwc=transpose,
            )

        elif process_img_type is None:
            # keep the image size unchanged (no clip or resize)
            slide_g = [torch.as_tensor(img), (None, None)]
            slide_g = [slide_g]
        else:
            raise ValueError(f"Unsupported process_img_type: {process_img_type}")

        if force_save_dtype is not None and force_save_dtype != "auto":
            save_dtype = (
                np.dtype(force_save_dtype)
                if isinstance(force_save_dtype, str)
                else force_save_dtype
            )
        else:
            save_dtype = None

        n_patches = 0
        patch = None
        for patch_idx, (patch, coord) in enumerate(slide_g):
            # logger.debug(f"patch_idx: {patch_idx}, patch: {patch}, coord: {coord}")
            if patch is None:
                # the background is all zero, skip
                continue

            global total_img_saved_count
            total_img_saved_count += 1

            patch, img_max = postprocess_img(
                patch,
                to_tensor=True,
                normalize=False,
                transpose=transpose,
                rescale=rescale,
            )

            patch, dtype = to_suitable_dtype_img(
                patch, img_max, dtype=save_dtype, is_normed=False
            )
            save_dtype = dtype

            # write to webdataset
            # img_name = img_name.replace(
            #     ".jp2", "-jp2"
            # )  # ensure the webdataset extrace 'img' as the key

            img_name = img_name.replace(
                ".", "-"
            )  # ensure the webdataset extrace 'img' as the key
            saved_name = (
                f"{total_img_saved_count}_{img_name}_patch-{patch_idx}"
                if add_global_index
                else f"{img_name}_patch-{patch_idx}"
            )
            saved_ext = save_backend if save_backend not in ("jpeg", "JPEG") else "jpg"

            sink.write(
                {
                    "__key__": saved_name,
                    f"img.{saved_ext}": img_saver_backend_compact_with_wds(
                        patch,
                        save_backend,
                        **save_kwargs,
                    ),
                }
            )

            n_patches += 1

        if patch is None:
            logger.warning(f"Image {img_name} has no valid patches, skipping saving.")
            return 0, img.shape, (None, None)
        return n_patches, img.shape, patch.shape

    img_name = Path(img_path).stem
    img = read_image(
        img_path,
        verbose=False,
        **read_fn_kwargs,
    )
    if img is None:
        logger.warning(f"Failed to read image from {img_path}, skipping.")
        return None, None, None
    if assert_channel_n is not None:
        if img.shape[-1] != assert_channel_n:
            logger.warning(
                f"Image {img_path} has {img.shape[-1]} channels, expected {assert_channel_n} channels."
            )
            return None, None, None

    if Path(img_path).suffix.lower() in [".mp4"] and isinstance(img, list):
        # is video
        for i, img_f in enumerate(img):
            img_name_frame = f"{img_name}_frame-{i}"
            n_patch, shape, p_shape = slide_image_and_save(img_f, img_name_frame)
    else:  # is image
        n_patch, shape, p_shape = slide_image_and_save(img, img_name)

    return n_patch, shape, p_shape


@logger.catch(reraise=True)
def loop_dataset_tif_MSI_images_to_webdataset(
    webdataset_pattern: str,
    dataset_root: str | Path | list[str | Path] | None = None,
    msi_files: list[str | Path] | Iterable | None = None,
    img_clip_size: tuple[int, int] = (512, 512),
    img_stride: tuple[int, int] = (512, 512),
    img_resize: int = 512,
    process_img_type: Literal["clip", "resize", "clip_resize", None] = "clip",
    save_backend: Literal[
        "tiff", "jpeg", "png", "npy", "safetensors", "webdataset"
    ] = "tiff",
    max_size: int = 4 * 1024 * 1024 * 1024,
    force_save_dtype: str | np.dtype = "auto",
    save_kwargs: dict = {"jpg_quality": 90},
    glob_pattern: str | list[str] = ["*.tif", "*.tiff"],
    read_transpose: bool = True,  # [c, h, w] needs transpose
    read_fn_kwargs: dict = {},
    tqdm_or_not: bool = True,
    delete_file: bool = False,
    channel_n: int | None = None,  # assert the channel number of the image
):
    if dataset_root is not None:
        if isinstance(dataset_root, (str, Path)):
            dataset_root = [dataset_root]

        msi_files = []
        for p in dataset_root:
            p = Path(p)
            for glob_p in glob_pattern:
                msi_files.extend(p.glob(glob_p))
    elif msi_files is not None:
        if isinstance(msi_files, (list, Iterable)):
            msi_files = [Path(msi_f) for msi_f in msi_files]
        else:
            raise ValueError("msi_files must be a list or iterable of file paths")
    else:
        raise ValueError("Either dataset_root or msi_files must be provided")

    assert len(msi_files) > 0, "no MSI files found"
    msi_files = natsort.natsorted(msi_files)
    logger.info(f"found {len(msi_files)} MSI files")

    # make output dir
    Path(webdataset_pattern).parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"make output dir {Path(webdataset_pattern).parent.as_posix()}")

    # write to webdataset
    with wds.ShardWriter(webdataset_pattern, maxsize=max_size) as sink:
        for msi_file in (tbar := tqdm(msi_files, disable=not tqdm_or_not)):
            msi_file = cast(Path, msi_file)  # ensure msi_file is Path type
            n_patches, shape, patch_shape = clip_img_to_webdataset(
                sink,
                img_path=msi_file,
                img_clip_size=img_clip_size,
                img_stride=img_stride,
                img_resize=img_resize,
                process_img_type=process_img_type,
                save_backend=save_backend,
                transpose=read_transpose,
                read_fn_kwargs=read_fn_kwargs,
                save_kwargs=save_kwargs,
                rescale="clamp",  # default rescale method
                force_save_dtype=force_save_dtype,
                assert_channel_n=channel_n,
            )
            if delete_file:
                Path(msi_file).unlink(missing_ok=True)

            if tqdm_or_not:
                tbar.set_description(
                    f"writing {msi_file.name}, {n_patches=}, {shape=}, {patch_shape=}, {delete_file=}"
                )

    logger.info(f"webdataset written to {webdataset_pattern}")


def test_webdatasets(tar_file_paths: list[str]):
    dataset = wds.WebDataset(tar_file_paths, resampled=False, shardshuffle=False)
    dataset = dataset.decode(tiff_decoder)

    def to_tensor(sample):
        img = torch.as_tensor(sample["img.tiff"]).float()
        img = img / img.max()
        img = img * 2 - 1
        img = img.permute(-1, 0, 1)

        return {"img": img}

    dataset = dataset.map(to_tensor)
    dataloader = wds.WebLoader(dataset, batch_size=1, num_workers=0)

    from tqdm import tqdm

    for i, sample in enumerate(tqdm(dataloader)):  # total=200):
        if i != 0:
            print(sample["img"].shape)


def SAR_img_to_gray(sar_img: np.ndarray | str):
    if isinstance(sar_img, str):
        sar_img = tifffile.imread(sar_img)

    assert sar_img.ndim == 3, "SAR image must be 3D"
    assert sar_img.shape[-1] == 2, "SAR image must be 2 channels at last dimension"

    real_part = sar_img[..., 0]
    imag_part = sar_img[..., 1]

    amp = np.sqrt(real_part**2 + imag_part**2)
    phase = np.arctan2(imag_part, real_part)
    phase_norm = (phase + np.pi) / (2 * np.pi)

    log_amp = np.log(amp + 1)
    # norm
    log_amp = (log_amp - log_amp.min()) / (log_amp.max() - log_amp.min())

    return dict(
        amp=amp,
        log_amp=log_amp,
        phase=phase_norm,
    )


def mp_tarfile_rearrange(tarfile_dir: str | Path):
    import shutil

    parent_dir = Path(tarfile_dir)

    # rearrange the output files
    tar_files = [str(p) for p in parent_dir.glob("**/*.tar")]
    tar_files = natsort.natsorted(tar_files)

    for total_i, tarf in enumerate(tar_files):
        name = Path(tarf).name
        prefix_name = "-".join(name.split("-")[:-1])
        name = f"{prefix_name}-{str(total_i).zfill(4)}.tar"
        shutil.move(str(tarf), str(parent_dir / name))
        logger.info(f"renamed {tarf} to {parent_dir / name}")


if __name__ == "__main__":
    # log file
    from src.utilities.logging import set_logger_file

    set_logger_file(
        "data/Multispectral-FMow-full/not_4bands.warning.log", "warning", add_time=False
    )

    _mp = False

    func_kwargs = {
        "process_img_type": "clip_resize",
        "img_clip_size": (1024, 1024),
        "img_stride": (1024, 1024),
        "img_resize": 1024,
        "save_kwargs": {
            "tiff_compression_type": "jpeg2000",
            "tiff_jpg_irreversible": True,
            "jpeg_quality": 80,
        },
        "read_fn_kwargs": {
            "tiff_bands_seperated": False,
        },
    }

    # mp_tarfile_rearrange("data/Disaterm3/hyper_images")
    # exit(0)

    if not _mp:
        # all_msi_files = list(Path("data/HLS").glob("*B01.tif"))
        # all_msi_files = list(Path("data/DIOR_RSVG_Dataset/JPEGImages").glob("*.jpg"))
        # all_msi_files = list(
        #     Path("data/YuZhongDataset/OpenEarthMap/OpenEarthMap_wo_xBD").glob(
        #         "*/images/*.tif"
        #     )
        # )
        # all_msi_files = list(Path("data/RefSegRS/RefSegRS/images").glob("*.tif"))
        # all_msi_files = list(Path("data/VDD/VDD").glob("*/gt/*.png"))
        # all_msi_files = list(
        #     Path("data/YuZhongDataset/LoveDA").glob("**/images_png/*.png")
        # )
        # all_msi_files = list(
        #     Path("data/ERA_UAV_Video_Dataset/SingleFrames").glob("**/*.png")
        # )

        # RSCaption task collections
        # base_path = Path("data/RSCaptions")
        # exts = ["tif", "jpg"]
        # all_msi_files_s = []
        # webdataset_pts = []
        # for subdir in base_path.iterdir():
        #     if subdir.is_dir():
        #         all_file_in_sub_dir = []
        #         for ext in exts:
        #             ext_files = list(subdir.glob(f"**/*.{ext}"))
        #             all_file_in_sub_dir.extend(ext_files)
        #         logger.info(
        #             f"in subdir {subdir.name}, found {len(all_file_in_sub_dir)} files"
        #         )
        #         if len(all_file_in_sub_dir) == 0:
        #             logger.warning(
        #                 f"subdir {subdir.name} has no files, please check the path"
        #             )
        #         else:
        #             all_msi_files_s.append(all_file_in_sub_dir)
        #             webdataset_pts.append(
        #                 f"data/RSCaptions/hyper_images/RSCaptionCollection-{subdir.name}-%04d.tar"
        #             )

        # logger.info(f"Found {sum([len(l) for l in all_msi_files_s])} files")

        # LoveDA dataset
        # path = "data/YuZhongDataset/LoveDA"
        # all_msi_files_s = list(Path(path).glob("**/images_png/*.png"))
        # webdataset_pts = "data/LoveDA/hyper_images/LoveDA-3_bands-px_1024-%04d.tar"

        # OpenEarthMap dataset
        # path = "data/YuZhongDataset/OpenEarthMap/OpenEarthMap_wo_xBD"
        # all_msi_files_s = list(Path(path).glob("*/images/*.tif"))
        # webdataset_pts = (
        #     "data/OpenEarthMap/hyper_images/OpenEarthMap-3_bands-px_1024-%04d.tar"
        # )

        # TEOChatlas
        # path = 'data/TEOChatlas'

        # disaterm3
        # all_msi_files_s = [
        #     list(Path("data/Disaterm3/train_images/train_images").glob("**/*.png"))
        #     + list(Path("data/Disaterm3/train_images/train_images").glob("**/*.jpg"))
        #     + list(Path("data/Disaterm3/test/test/rgb_images").glob("**/*.png"))
        #     + list(Path("data/Disaterm3/test/test/rgb_images").glob("**/*.jpg"))
        # ]
        # webdataset_pts = (
        #     "data/Disaterm3/hyper_images/Disaterm3-3_bands-px_1024-RGB-jp2k-90-%04d.tar"
        # )

        # CityBench
        # all_msi_files_s = list(Path("data/CityBench-CityData").glob("**/*.jpg"))
        # all_msi_files_s += list(Path("data/CityBench-CityData").glob("**/*.png"))
        # webdataset_pts = (
        #     "data/CityBench-CityData/hyper_images/CityBench-CityData-%04d.tar"
        # )

        # Hyperspectral-collections
        # all_msi_files_s = list(Path("data/HyperSpectral-Collections").glob("*.mat"))
        # webdataset_pts = "data/HyperSpectral-Collections/hyper_images/hyperspectral_collections_%04d.tar"
        # func_kwargs.update(
        #     dict(
        #         process_img_type="clip",
        #         img_clip_size=(256, 256),
        #         img_stride=(256, 256),
        #         save_backend="tiff",
        #         save_kwargs=dict(
        #             tiff_compression_type="jpeg2000",
        #         ),
        #     )
        # )

        # AerialGV
        # all_msi_files_s = list(glob.glob("data/AerialVG/images/*"))
        # webdataset_pts = "data/AerialVG/hyper_images/AerialVG-3_bands-RGB-%04d.tar"

        # FMoW rgb
        # all_msi_files_s = list(
        #     Path(
        #         "/HardDisk/ZiHanCao/datasets/Multispectral-fMoW-Data-multispectral-RGB"
        #     ).glob("**/*_rgb.jpg")
        # )
        # webdataset_pts = "data/Fmow_rgb/hyper_images/FMoW-3_bands-RGB-%04d.tar"

        # Fmow full
        all_msi_files_s = list(
            Path("data/Multispectral-FMow-full/train").rglob("**/*.tif")
        )
        webdataset_pts = "data/Multispectral-FMow-full/hyper_images_8bands/FMoW-8_bands-px_1024-RGB-jp2k-80-%04d.tar"
        func_kwargs.update(
            {
                "channel_n": 8,
                "save_backend": "tiff",
                "process_img_type": "clip_resize",
                "img_clip_size": (1024, 1024),
                "img_stride": (1024, 1024),
            }
        )

        # Inria Aeiral images
        # path = "data/YuZhongDataset/InriaAerialLabelingDataset/AerialImageDataset"
        # all_msi_files_s = list(Path(path).glob("**/images/*.tif"))
        # webdataset_pts = "data/InriaAerialLabelingDataset/hyper_images/InriaAerialLabelingDataset-3_bands-px_512-RGB-jp2k-80-%04d.tar"

        # AIRS contest MARS data
        # path = "data/AIRS-Universe-Melas-Chasma/"
        # all_msi_files_s = [
        #     "data/AIRS-Universe-Melas-Chasma/CopratesChasma.mat",
        #     "data/AIRS-Universe-Melas-Chasma/GaleCrater.mat",
        #     "data/AIRS-Universe-Melas-Chasma/MelasChasma.mat",
        # ]
        # webdataset_pts = "data/AIRS-Universe-Melas-Chasma/hyper_images/AIS_contest_data_372_bands-px_80-MSI-%04d.tar"
        # func_kwargs.update(
        #     {
        #         "process_img_type": "clip",
        #         "img_clip_size": (80, 80),
        #         "img_stride": (80, 80),
        #         "save_backend": "tiff",
        #         "force_save_dtype": np.float32,
        #         "save_kwargs": {
        #             "tiff_compression_type": "zlib",
        #             "jpeg_quality": 100,
        #         },
        #     }
        # )

        # DOTA v1 dataset
        # path = "data/YuZhongDataset/DOTA_v1.0"
        # all_msi_files_s = list(Path(path).glob("**/images/*.png"))
        # webdataset_pts = "data/DOTA_v1/hyper_images/DOTA_v1-3_bands-px_1024-%04d.tar"
        # func_kwargs.update(
        #     {
        #         "process_img_type": "clip",
        #         "img_clip_size": (1024, 1024),
        #         "img_stride": (1024, 1024),
        #     }
        # )

        # UDD dataset
        # path = "data/YuZhongDataset/UDD"
        # all_msi_files_s = list(Path(path).glob("**/src/*.JPG"))
        # webdataset_pts = "data/UDD/hyper_images/UDD-3_bands-px_512-RGB-jp2k-95-%04d.tar"

        # xView2 dataset
        # path = "data/xView2"
        # all_msi_files_s = list(Path(path).glob("**/images/*.png"))
        # webdataset_pts = (
        #     "data/xView2/hyper_images/xView2-3_bands-px_1024-RGB-jp2k-80-%04d.tar"
        # )

        # VDD dataset
        # path = "data/VDD/VDD"
        # all_msi_files_s = list(Path(path).glob("**/src/*.JPG"))
        # webdataset_pts = "data/VDD/hyper_images/VDD_3_bands-px_2k-RGB-%04d.tar"

        # uavid dataset
        # path = "data/YuZhongDataset/uavid_image"
        # all_msi_files_s = list(Path(path).glob("**/Images/*.png"))
        # webdataset_pts = (
        #     "data/uavid/hyper_images/uavid-3_bands-px_2k-RGB-jp2k-80-%04d.tar"
        # )

        if not isinstance(all_msi_files_s[0], list):
            all_msi_files_s = [all_msi_files_s]
        if not isinstance(webdataset_pts, list):
            webdataset_pts = [webdataset_pts]

        assert len(all_msi_files_s) == len(webdataset_pts), (
            f"all_msi_files_s and webdataset_pts must have the same length, "
            f"but got {len(all_msi_files_s)} and {len(webdataset_pts)}"
        )
        for wds_pt, all_msi_files in zip(webdataset_pts, all_msi_files_s):
            loop_dataset_tif_MSI_images_to_webdataset(
                # webdataset_pattern="data/DIOR_RSVG_Dataset/hyper_images/DIOR_RSVG_3_bands-px_800-RGB-jp2k-80-%04d.tar",
                # webdataset_pattern="data/OpenEarthMap/hyper_images/OpenEarthMap_3_bands-px_1000-RGB-jp2k-80-%04d.tar",
                # webdataset_pattern='data/RefSegRS/hyper_images/RefSegRS_3_bands-px_512-RGB-jp2k-80-%04d.tar',
                # webdataset_pattern="data/VDD/segmentation_label/VDD_3_bands-px_512-RGB-jp2k-80-%04d.tar",
                # webdataset_pattern="data/LoveDA/hyper_images/LoveDA-3_bands-px_1024-%04d.tar",
                # webdataset_pattern="data/ERA_UAV_Video_Dataset/hyper_images/ERA_UAV_Video_Dataset_key_frames_3_bands-px_512-RGB-jp2k-80-%04d.tar",
                webdataset_pattern=wds_pt,
                msi_files=all_msi_files,
                save_backend=func_kwargs.pop("save_backend", "jpeg"),
                **func_kwargs,
            )

    else:
        # If multiprocessing is enabled, we will use the main process to run the function
        import multiprocessing as mp

        def split_list(lst, n):
            """将列表 lst 分成 n 份"""
            k, m = divmod(len(lst), n)
            return (
                lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)
            )

        # 获取所有 msi_files
        all_msi_files = list(
            Path("data/Disaterm3/train_images/train_images").glob("**/*.png")
        )
        all_msi_files += list(
            Path("data/Disaterm3/train_images/train_images").glob("**/*.jpg")
        )
        all_msi_files += list(
            Path("data/Disaterm3/test/test/rgb_images").glob("**/*.png")
        )
        all_msi_files += list(
            Path("data/Disaterm3/test/test/rgb_images").glob("**/*.jpg")
        )

        # 按进程数分割 msi_files
        num_processes = 8  # mp.cpu_count()
        msi_files_split = list(split_list(all_msi_files, num_processes))

        # 为每个进程指定输出文件夹
        output_folders = [
            f"data/Disaterm3/hyper_images/part_{i}" for i in range(num_processes)
        ]
        for folder in output_folders:
            Path(folder).mkdir(parents=True, exist_ok=True)

        # 定义每个进程的任务
        def process_files(msi_files, output_folder, is_main_process):
            loop_dataset_tif_MSI_images_to_webdataset(
                webdataset_pattern=f"{output_folder}/Disaterm3_3_bands-px_1024-RGB-jp2k-90-%04d.tar",
                msi_files=msi_files,
                process_img_type=func_kwargs["process_img_type"],
                img_clip_size=func_kwargs["img_clip_size"],
                img_stride=func_kwargs["img_stride"],
                save_kwargs=func_kwargs["save_kwargs"],
                read_fn_kwargs=func_kwargs["read_fn_kwargs"],
                tqdm_or_not=is_main_process,
                save_backend=func_kwargs.get("save_backend", "jpeg"),
            )

        # 在调用时传递进程索引
        with mp.Pool(num_processes) as pool:
            results = pool.starmap(
                process_files,
                # `i == 0` 表示第一个进程才显示tqdm
                [
                    (files, folder, i == 0)
                    for i, (files, folder) in enumerate(
                        zip(msi_files_split, output_folders)
                    )
                ],
            )

        logger.info("All processes completed successfully.")
