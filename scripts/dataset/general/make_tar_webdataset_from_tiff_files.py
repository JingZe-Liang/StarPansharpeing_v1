import io
import sys
from pathlib import Path
from typing import Literal

import einops
import natsort
import numpy as np
import safetensors
import safetensors.torch
import tifffile
import torch
import webdataset as wds
from loguru import logger
from PIL import Image
from scipy.io import loadmat
from tqdm import tqdm

logger.remove()
logger.add(
    sink=sys.stdout,
    level="DEBUG",
    format="<green>[{time:MM-DD HH:mm:ss}]</green> <cyan>{name}</cyan> <level>[{level}]</level> - <level>{message}</level>",
    backtrace=True,
    diagnose=True,
    colorize=True,
)


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


def to_batched(img: torch.Tensor):
    if img.ndim == 2:
        img = img.unsqueeze(0).unsqueeze(0)
        inverse = lambda x: x[0, 0]
    elif img.ndim == 3:
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

        img = (img - min_c) / (max_c - min_c + 1e-8)  # Add small epsilon to avoid division by zero

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


def sliding_window(image, patch_size, stride, is_yield=True, pad_if_necessary=False):
    """
    简化版 sliding window，确保覆盖，添加 padding
    image: [h, w, c]
    """
    patches = []
    coords = []
    shape = image.shape
    h, w = shape[:-1]

    if pad_if_necessary and isinstance(image, np.memmap):
        logger.warning("Image is a memmap, padding will not be applied.")

    if pad_if_necessary and not isinstance(image, np.memmap):
        # 计算 padding 大小
        pad_h = max(0, patch_size[0] - (h % stride[0])) if (h % stride[0]) != 0 else 0
        pad_w = max(0, patch_size[1] - (w % stride[1])) if (w % stride[1]) != 0 else 0

        # 对图像进行 padding
        if pad_h > 0 or pad_w > 0:
            image, inverse = to_batched(image)
            image = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
            image = inverse(image)

    logger.info(f"img shape: {shape}")
    h_padded, w_padded = shape[:-1]
    logger.info(f"clip patch with shape: {(h_padded, w_padded)}, patch_size: {patch_size}, stride: {stride}")
    rows = list(range(0, h_padded, stride[0]))
    cols = list(range(0, w_padded, stride[1]))
    logger.info(f"Got {len(rows) * len(cols)} patches")

    # 确保最后一个 patch 覆盖到边界
    for i in rows:
        for j in cols:
            # 如果超出边界，则调整 patch 的起始位置
            i_end = min(i + patch_size[0], h_padded)
            j_end = min(j + patch_size[1], w_padded)
            i_start = i_end - patch_size[0]
            j_start = j_end - patch_size[1]

            # img_slide = image[..., i_start:i_end, j_start:j_end]
            img_slide = image[i_start:i_end, j_start:j_end, ...]  # Assuming image is [h, w, c]
            if isinstance(img_slide, np.memmap):
                # img_slide = np.array(img_slide)
                logger.debug(f"Copying memmap to numpy array at patch ({i_start}:{i_end}, {j_start}:{j_end})")
                img_slide = img_slide.copy()  # Ensure we have a copy, not a view

            if is_yield:
                yield img_slide, (i_start, j_start)
            else:
                patches.append(img_slide)
                coords.append((i_start, j_start))

    return patches, coords


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
    merged_image = torch.where(count_map > 0, merged_image / count_map, torch.tensor(0.0))

    # 移除 padding
    if pad_h > 0 or pad_w > 0:
        merged_image = merged_image[..., :h, :w]

    return merged_image


def read_image(
    img_path: str | Path,
    *,
    mat_load_key="I",
    verbose=True,
):
    if isinstance(img_path, str):
        img_path = Path(img_path)

    if verbose:
        logger.info("reading image from: {}", img_path.as_posix())

    if img_path.suffix == ".mat":
        d = loadmat(img_path)
        key_ = list(d.keys())[-1]
        img = d[key_]
    elif img_path.suffix.lower() in [".tif", ".tiff"]:
        # memmap
        # try:
        #     img = tifffile.memmap(img_path, mode="r")
        # except Exception as e:
        #     logger.warning("failed to load image from: {}", img_path.as_posix())
        #     logger.warning("trying to load image as numpy array")
        #     try:
        tif = tifffile.TiffFile(img_path)
        img = tif.asarray(out="memmap")
        # except Exception as e:
        #     logger.error("failed to load image as numpy array: {}", e)
        #     raise e
    elif img_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
        img = np.array(Image.open(img_path))
    elif img_path.suffix == ".npy":
        img = np.load(img_path)
    else:
        raise ValueError(f"Unsupported image format: {img_path.suffix}")

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
            img = per_channel_normalize(img, c_dim=(-2, -1) if img.ndim == 4 else (0, 1))
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


def to_int_dtype_img(
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

    dtype = determine_fit_dtype(img_max, dtype)
    if is_normed:
        img = img * img_max
    if torch.is_tensor(img):
        assert img.shape[0] == 1, "batch size must be 1"
        img = (img).cpu().permute(0, 2, 3, 1)[0].numpy().astype(dtype)
    elif isinstance(img, np.ndarray):
        img = img.astype(dtype)
    else:
        raise ValueError(f"Unsupported image type: {type(img)}")

    return img, dtype


def img_saver_backend_compact_with_wds(
    img: np.ndarray,
    extension: str,
):
    assert extension in [
        "npy",
        "tiff",
        "jpg",
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
        tifffile.imwrite(byte_io, img, shape=img.shape, compression="zlib")
    elif extension in ["jpg", "jpeg", "png"]:
        assert img.dtype == np.uint8, "img must be uint8"
        Image.fromarray(img).save(byte_io, format=extension)
    elif extension == "safetensors":
        byte = safetensors.torch.save({"img": torch.as_tensor(img, dtype=_numpy_dtype_to_tensor[img.dtype.name])})
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
    img_path: str,
    img_clip_size: tuple[int, int] = (512, 512),
    img_stride: tuple[int, int] = (512, 512),
    save_backend: str = "tiff",
    transpose: bool = True,
    read_fn_kwargs: dict = {},
    slide_use_yield: bool = True,
):
    img_name = Path(img_path).stem

    img_tensor = read_image(
        img_path,
        verbose=False,
        **read_fn_kwargs,
    )
    slide_g = sliding_window(img_tensor, img_clip_size, img_stride, is_yield=slide_use_yield)

    # to tensor after patching when using memmap
    # img_patches = [torch.as_tensor(patch) for patch in img_patches]
    # img_max = -torch.inf
    # for patch in img_patches:
    #     patch, img_max_p = postprocess_img(
    #         patch,
    #         to_tensor=True,
    #         normalize=False,
    #         transpose=transpose,
    #     )
    #     img_max = torch.maximum(img_max, img_max_p)

    save_dtype = None
    for patch_idx, (patch, coord) in enumerate(slide_g):
        patch, img_max = postprocess_img(
            patch,
            to_tensor=True,
            normalize=False,
            transpose=transpose,
        )
        # img_max = patch.max()
        patch, dtype = to_int_dtype_img(patch, img_max, dtype=save_dtype, is_normed=False)
        save_dtype = dtype

        # write to webdataset
        sink.write(
            {
                "__key__": f"{img_name}_patch-{patch_idx}",
                f"img.{save_backend}": img_saver_backend_compact_with_wds(
                    patch,
                    save_backend,
                ),
            }
        )


@logger.catch(reraise=True)
def loop_dataset_tif_MSI_images_to_webdataset(
    webdataset_pattern: str,
    dataset_root: str | Path | list[str | Path] | None = None,
    msi_files: list[str | Path] | None = None,
    img_clip_size: tuple[int, int] = (512, 512),
    img_stride: tuple[int, int] = (512, 512),
    save_backend: str = "tiff",
    max_size: int = 4 * 1024 * 1024 * 1024,
    glob_pattern: str | list[str] = ["*.tif", "*.tiff"],
    read_transpose: bool = True,
    read_fn_kwargs: dict = {},
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
        msi_files = [Path(msi_f) for msi_f in msi_files]
        for msi_f in msi_files:
            assert Path(msi_f).exists(), f"MSI file {msi_f} does not exist"
    else:
        raise ValueError("Either dataset_root or msi_files must be provided")

    # dataset_root = Path(dataset_root)
    # assert dataset_root.exists(), f"dataset root {dataset_root} does not exist"

    # # find MSI files
    # msi_files = []
    # for p in glob_pattern:
    #     msi_files.extend(list(dataset_root.glob(p)))

    assert len(msi_files) > 0, "no MSI files found"
    msi_files = natsort.natsorted(msi_files)
    logger.info(f"found {len(msi_files)} MSI files")

    # make output dir
    Path(webdataset_pattern).parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"make output dir {Path(webdataset_pattern).parent.as_posix()}")

    # write to webdataset
    with wds.ShardWriter(webdataset_pattern, maxsize=max_size) as sink:
        for msi_file in (tbar := tqdm(msi_files)):
            tbar.set_description(f"writing {msi_file.name}")
            clip_img_to_webdataset(
                sink,
                img_path=msi_file,
                img_clip_size=img_clip_size,
                img_stride=img_stride,
                save_backend=save_backend,
                transpose=read_transpose,
                read_fn_kwargs=read_fn_kwargs,
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


if __name__ == "__main__":
    # loop_dataset_tif_MSI_images_to_webdataset(
    #     dataset_root="/HardDisk/ZiHanCao/datasets/Multispectral-Houston/2018IEEE_Contest/Phase2/FullHSIDataset",
    #     webdataset_pattern="Multispectral_webdatasets/Houston-50_bands-px_512-MSI-%04d.tar",
    #     img_clip_size=(512, 512),
    #     img_stride=(512, 512),
    #     save_backend="tiff",
    #     glob_pattern=["*.mat"],
    # )

    # loop_dataset_tif_MSI_images_to_webdataset(
    #     dataset_root="/HardDisk/ZiHanCao/datasets/NBU_RS_Data/Sat_Dataset/Dataset/1 IKONOS/1 IKONOS/MS_256",
    #     webdataset_pattern="Multispectral_webdatasets/IKONOS-4_bands-px_256-MSI-%04d.tar",
    #     img_clip_size=(256, 256),
    #     img_stride=(256, 256),
    #     save_backend="tiff",
    #     glob_pattern=["*.mat"],
    #     read_fn_kwargs={"mat_load_key": "imgMS"},
    # )

    # loop_dataset_tif_MSI_images_to_webdataset(
    #     dataset_root="/HardDisk/ZiHanCao/datasets/NBU_RS_Data/Sat_Dataset/Dataset/2 QuickBird/2 QuickBird/MS_256",
    #     webdataset_pattern="Multispectral_webdatasets/QuickBird-4_bands-px_256-MSI-%04d.tar",
    #     img_clip_size=(256, 256),
    #     img_stride=(256, 256),
    #     save_backend="tiff",
    #     glob_pattern=["*.mat"],
    #     read_fn_kwargs={"mat_load_key": "imgMS"},
    # )

    # loop_dataset_tif_MSI_images_to_webdataset(
    #     # dataset_root="/HardDisk/ZiHanCao/datasets/NBU_RS_Data/Sat_Dataset/Dataset/6 WorldView-3/6 WorldView-3/MS_256",
    #     dataset_root="/HardDisk/ZiHanCao/datasets/NBU_RS_Data/Sat_Dataset/Dataset/6 WorldView-3/6 WorldView-3/PAN_1024",
    #     webdataset_pattern="Multispectral_webdatasets/WorldView3-PAN-1_bands-px_512-MSI-%04d.tar",
    #     img_clip_size=(512, 512),
    #     img_stride=(512, 512),
    #     save_backend="tiff",
    #     glob_pattern=["*.mat"],
    #     read_fn_kwargs={"mat_load_key": ["imgMS", "I_MS"]},
    # )

    # loop_dataset_tif_MSI_images_to_webdataset(
    #     dataset_root="NBU_RS_Data/Sat_Dataset/Dataset/4 WorldView-4/4 WorldView-4/PAN_1024",
    #     webdataset_pattern="Multispectral_webdatasets/WorldView4-PAN-1_bands-px_512-MSI-%04d.tar",
    #     img_clip_size=(512, 512),
    #     img_stride=(512, 512),
    #     save_backend="tiff",
    #     glob_pattern=["*.mat"],
    #     read_fn_kwargs={"mat_load_key": ["imgMS", "I_MS"]},
    # )

    # loop_dataset_tif_MSI_images_to_webdataset(
    #     dataset_root="NBU_RS_Data/Sat_Dataset/Dataset/1 IKONOS/1 IKONOS/PAN_1024",
    #     webdataset_pattern="Multispectral_webdatasets/IKONOS-PAN-1_bands-px_512-MSI-%04d.tar",
    #     img_clip_size=(512, 512),
    #     img_stride=(512, 512),
    #     save_backend="tiff",
    #     glob_pattern=["*.mat"],
    #     read_fn_kwargs={"mat_load_key": ["imgMS", "I_MS"]},
    # )

    # loop_dataset_tif_MSI_images_to_webdataset(
    #     dataset_root="/HardDisk/ZiHanCao/datasets/Multispectral-GID/GID/img_dir/test",
    #     webdataset_pattern="Multispectral_webdatasets/GID-GF2-test-4_bands-px_512-MSI-%04d.tar",
    #     img_clip_size=(512, 512),
    #     img_stride=(512, 512),
    #     save_backend="tiff",
    #     glob_pattern=["*.tif"],
    # )

    # loop_dataset_tif_MSI_images_to_webdataset(
    #     dataset_root="/HardDisk/ZiHanCao/datasets/Multispectral-MUESLI/Hyper_2m_Tiles",
    #     webdataset_pattern="Multispectral_webdatasets/MUSLI_safetensors/MUSLI-438_bands-px_512-MSI-%04d.tar",
    #     img_clip_size=(512, 512),
    #     img_stride=(512, 512),
    #     save_backend="safetensors",
    #     glob_pattern=["*.tif"],
    # )

    # loop_dataset_tif_MSI_images_to_webdataset(
    #     dataset_root="/HardDisk/ZiHanCao/datasets/Multispectral-MMSegWhisper/MMSeg-YREB/train/MSI",
    #     webdataset_pattern="Multispectral_webdatasets/MMSeg_YREB_train_part-12_bands-MSI-%04d.tar",
    #     img_clip_size=(256, 256),
    #     img_stride=(256, 256),
    #     save_backend="tiff",
    #     glob_pattern=["*.tif"],
    # )

    # loop_dataset_tif_MSI_images_to_webdataset(
    #     dataset_root="/HardDisk/ZiHanCao/datasets/Multispectral-DryadHyper/01/hyspecnet-11k",
    #     webdataset_pattern="Multispectral_webdatasets/DryadHyper-224_bands-px_128-MSI-%04d.tar",
    #     img_clip_size=(128, 128),
    #     img_stride=(128, 128),
    #     save_backend="tiff",
    #     glob_pattern=["**/*SPECTRAL_*.TIF"],
    #     read_transpose=False,
    # )

    # loop_dataset_tif_MSI_images_to_webdataset(
    #     dataset_root="/HardDisk/ZiHanCao/datasets/Multispectral-DFC2020/DFC_Public_Dataset",
    #     webdataset_pattern="Multispectral_webdatasets/DFC_2020_public-13_bands-px_256-MSI-%04d.tar",
    #     img_clip_size=(256, 256),
    #     img_stride=(256, 256),
    #     save_backend="tiff",
    #     glob_pattern=["**/s2*/*.tif"],
    # )

    # loop_dataset_tif_MSI_images_to_webdataset(
    #     dataset_root="/HardDisk/ZiHanCao/datasets/Hyperspectral-WHU-OHS/tr/image",
    #     img_clip_size=(512, 512),
    #     img_stride=(512, 512),
    #     glob_pattern=["*.tif"],
    #     webdataset_pattern="Multispectral_webdatasets/OHS-32_bands-px_512-MSI-%04d.tar",
    #     read_fn_kwargs={
    #         "rescale": "add_min"
    #     },  # add min since the image is already normalized
    # )

    loop_dataset_tif_MSI_images_to_webdataset(
        webdataset_pattern="/HardDisk/ZiHanCao/datasets/Multispectral_webdatasets/MDAS-HySpex/MDAS-HySpex-368_bands-px_256-MSI-%04d.tar",
        msi_files=[
            "/HardDisk/ZiHanCao/datasets/Multispectral-MDAS/MDAS_dataset/Augsburg_data_4_publication/entire_city/HySpex.tif",  # (4036, 6232, 368)
            # "/HardDisk/ZiHanCao/datasets/Multispectral-MDAS/MDAS_dataset/Augsburg_data_4_publication/sub_area_1/HySpex_sub_area1.tif",  # (1364, 1636, 368)
            # "/HardDisk/ZiHanCao/datasets/Multispectral-MDAS/MDAS_dataset/Augsburg_data_4_publication/sub_area_2/HySpex_sub_area2.tif",  # (1364, 1636, 368)
            # "/HardDisk/ZiHanCao/datasets/Multispectral-MDAS/MDAS_dataset/Augsburg_data_4_publication/sub_area_3/HySpex_sub_area3.tif",  # (1364, 1636, 368)
        ],
        img_clip_size=(256, 256),
        img_stride=(256, 256),
    )

    # loop_dataset_tif_MSI_images_to_webdataset(
    #     webdataset_pattern="/HardDisk/ZiHanCao/datasets/Multispectral_webdatasets/MDAS-EeteS/MDAS-EeteS-242_bands-px_256-MSI-%04d.tar",
    #     msi_files=[
    #         "/HardDisk/ZiHanCao/datasets/Multispectral-MDAS/MDAS_dataset/Augsburg_data_4_publication/sub_area_1/EeteS_EnMAP_2dot2m_sub_area1.tif",  # (1364, 1636, 242)
    #     ],
    #     img_clip_size=(256, 256),
    #     img_stride=(256, 256),
    # )

    # loop_dataset_tif_MSI_images_to_webdataset(
    #     webdataset_pattern="/HardDisk/ZiHanCao/datasets/Multispectral_webdatasets/MDAS-Optical/MDAS-Optical-4_bands-px_512-MSI-%04d.tar",
    #     msi_files=[
    #         "/HardDisk/ZiHanCao/datasets/Multispectral-MDAS/MDAS_dataset/Augsburg_data_4_publication/sub_area_1/3K_RGB_sub_area1.tif",  # (15000, 18000, 4)
    #         "/HardDisk/ZiHanCao/datasets/Multispectral-MDAS/MDAS_dataset/Augsburg_data_4_publication/sub_area_2/3K_RGB_sub_area2.tif",  # (15000, 18000, 4)
    #         "/HardDisk/ZiHanCao/datasets/Multispectral-MDAS/MDAS_dataset/Augsburg_data_4_publication/sub_area_3/3K_RGB_sub_area3.tif",  # (15000, 18000, 4)
    #     ],
    #     img_stride=(512, 512),
    #     img_clip_size=(512, 512),
    # )

    # test_webdatasets(
    #     [
    #         "/HardDisk/ZiHanCao/datasets/Multispectral_webdatasets/MUSLI-438_bands-px_369-MSI-0000.tar"
    #     ]
    # )

    # SAR_img_to_gray(
    #     "/HardDisk/ZiHanCao/datasets/Multispectral-DFC2020/s1_0/ROIs0000_test_s1_0_p19.tif"
    # )
