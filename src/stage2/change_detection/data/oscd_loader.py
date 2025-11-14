import random
from collections.abc import Generator
from functools import partial

import accelerate
import numpy as np
import torch
import webdataset as wds
from kornia.augmentation import (
    AugmentationSequential,
    ImageSequential,
    RandomAffine,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomVerticalFlip,
)
from kornia.constants import DataKey
from timm.layers.helpers import to_2tuple

from src.data.codecs import tiff_decode_io
from src.data.utils import norm_img, not_dunder_keys, remove_extension, remove_meta_data
from src.data.window_slider import WindowSlider, create_windowed_dataloader
from src.stage2.change_detection.data.label_centric_patcher import (
    label_centrical_patcher,
)
from src.utilities.logging import log

OSCD_UNCHANGED_LABEL = 0
OSCD_CHANGED_LABEL = 1


def gt_to_int(gt: np.ndarray, gt_changes: dict | None = None):
    if gt.ndim == 3:  # (h, w, c)
        # 255. is changed; 0. is not changed
        gt = gt[..., 0]
    elif gt.ndim == 2:  # (h, w)
        pass
    else:
        raise ValueError(f"Invalid gt shape: {gt.shape}")

    gt = (gt > 0).astype("int")  # 0 or 1

    # If gt labels needs to remapped
    if gt_changes is not None:  # is default mapping
        gt_unchanged = gt == OSCD_UNCHANGED_LABEL
        gt_changed = gt == OSCD_CHANGED_LABEL

        # remapped
        remapped_unchanged = gt_changes["unchanged"]
        remapped_changed = gt_changes["changed"]

        gt[gt_unchanged] = remapped_unchanged
        gt[gt_changed] = remapped_changed

    return gt


def random_img_crop(crop_size=(256, 256)):
    crop = AugmentationSequential(
        RandomResizedCrop(crop_size, scale=(0.2, 1.0), ratio=(0.75, 1.33), p=1.0),
        data_keys=[DataKey.INPUT, DataKey.INPUT, DataKey.MASK],
        same_on_batch=False,
        keepdim=True,
    )

    def _crop_closure(sample: dict):
        img1, img2, gt = (
            sample["img1"],
            sample["img2"],
            sample["gt"],
        )
        img1, img2, gt = crop(img1, img2, gt)
        sample["img1"], sample["img2"], sample["gt"] = img1, img2, gt.type(torch.long)
        return sample

    return _crop_closure


def as_tensor(x, pin_memory: bool = False):
    return torch.as_tensor(x, device="cuda" if pin_memory else "cpu")


def random_img_augment(prob: float = 0.5):
    pipe = AugmentationSequential(
        RandomHorizontalFlip(p=prob),
        RandomVerticalFlip(p=prob),
        # RandomAffine(degrees=30, translate=(0.0, 0.1), scale=(0.8, 1.2), p=prob),
        data_keys=["input", "input", "mask"],
        same_on_batch=True,
        keepdim=True,
        random_apply=1,
    )

    def _augment_closure(sample: dict):
        img1, img2, gt = sample["img1"], sample["img2"], sample["gt"]
        img1, img2, gt = pipe(img1, img2, gt)
        sample["img1"], sample["img2"], sample["gt"] = img1, img2, gt.type(torch.long)
        return sample

    return _augment_closure


def shared_times_norm_img(
    sample: dict,
    norm_keys: str | list[str] | None = ["img1", "img2"],
    to_neg_1_1: bool = True,
    permute: bool = True,
    check_nan: bool = False,
    on_device: bool = False,
    clip_zero: bool = True,
    per_channel: bool = True,
    quantile_clip: float = 0.99,
):
    """
    Normalize image(s) in a sample to [0, 1] and optionally to [-1, 1].

    This function processes image data in various formats (dict, np.ndarray, torch.Tensor)
    and applies normalization, permutation, and other preprocessing operations.

    Args:
        sample: Sample dictionary containing image data to normalize
        norm_keys: Keys in the sample to normalize. Can be str, list[str], or None  (default: ["img1", "img2"])
        to_neg_1_1: Whether to normalize images to [-1, 1] range (default: True)
        permute: Whether to permute image dimensions from (H, W, C) to
                    (C, H, W) or (B, H, W, C) to (B, C, H, W) (default: True)
        check_nan: Whether to check and replace NaN values with 0 (default: False)
        on_device: Whether to move tensors to CUDA device (default: False)
        clip_zero: Whether to clip negative values to zero before normalization (default: True)
        per_channel: Whether to normalize per channel (default: True)

    NOTE:
    Normalization per-channel and shared with different times is important.
    But for tokenization, we need to normalize per-sample, because some applications
    rely on the value-relation between bands.
    """
    if norm_keys is None:
        norm_keys = not_dunder_keys(sample)
    elif isinstance(norm_keys, str):
        norm_keys = [norm_keys]

    assert len(norm_keys) > 1, (
        f"need at least 2 keys to do shared-times normalization for CD task, but got {norm_keys}"
    )

    for key in norm_keys:
        if key not in sample:
            log(f"{key} not in sample", level="warning", warn_once=True)
            continue

        if isinstance(sample[key], dict):
            _sample_dict: dict = sample[key]  # type: ignore
            _img = _sample_dict.get("img")
        elif isinstance(sample[key], (np.ndarray, torch.Tensor)):
            _img = sample[key]
        else:
            raise ValueError(
                f"Unsupported type for {key}: {type(sample[key])}. Expected dict, np.ndarray, or torch.Tensor."
            )

        img = torch.as_tensor(_img, dtype=torch.float32)
        if on_device:
            img = img.to(torch.device("cuda"), non_blocking=True)
        if check_nan:
            img = torch.nan_to_num(img, nan=0.0, posinf=1, neginf=0.0)
        if permute:
            if img.ndim == 3:
                # (H, W, C) -> (C, H, W)
                img = img.permute(-1, 0, 1)
            elif img.ndim == 4:
                #  # (B, C, H, W) -> (B, C, H, W)
                img = img.permute(0, -1, 1, 2)
            else:
                log(
                    f"found img dim with {img.ndim} dimensions, expected 3 or 4",
                    level="warning",
                    warn_once=True,
                )
                return None  # None for webdataset means drop this sample
        sample[key] = img

    # Time-shard normalization here
    # Different from tokenization per-sample normalization
    imgs_normed = [sample[k] for k in norm_keys]
    img_t = torch.stack(imgs_normed, dim=-1)  # (C, H, W, T) or (H, W, T)

    # To avoid too large values in some channels, clip them to the quantile value
    if quantile_clip < 1.0:
        # (C, 1) or (1,)
        q_max = img_t.flatten(-3).quantile(quantile_clip, dim=-1, keepdim=True)
        q_max = q_max[..., None, None]  # (C, 1, 1) or (1, 1, 1)
        img_t.clamp_(max=q_max)

    # shard min and max
    # (C, 1, 1, 1) or (1, 1, 1)
    img_shard_min = img_t.amin((-3, -2, -1), keepdim=True).squeeze(-1)
    img_shard_max = img_t.amax((-3, -2, -1), keepdim=True).squeeze(-1)

    # normalize each time image
    for i, (img_idx, key) in enumerate(zip(range(img_t.shape[-1]), norm_keys)):
        # img = sample[key]
        img = img_t[..., img_idx]
        img.sub_(img_shard_min)
        if img_shard_max.max().item() < 1e-4:
            img = torch.zeros_like(img) if not to_neg_1_1 else torch.ones_like(img) / 2
        else:
            img.div_(img_shard_max + 1e-6)

        if to_neg_1_1:
            img = (img * 2 - 1).clip(-1.0, 1.0)
        sample[key] = img

    return sample


def create_oscd_loader(
    wds_paths: list[str],
    batch_size: int,
    num_workers: int,
    shuffle_size: int = 0,
    to_neg_1_1: bool = False,  # downstream task does not need to norm image to (-1, 1)
    resample: bool = True,
    pin_memory: bool = False,
    shuffle: bool = False,
    remove_meta: bool = False,
    norm_img_keys: list[str] = ["img1", "img2"],
    crop_size: tuple[int, int] | int | None = None,
    random_flip_prob: float = 0.0,
    shared_norm: bool = True,
    prefetch_factor: int | None = None,
    remapped_gt: dict | None = None,
):
    """
    Create a dataloader for OSCD (Onera Satellite Change Detection) dataset.

    This function creates a webdataset-based dataloader for loading and preprocessing
    OSCD data, which typically includes pairs of satellite images and change detection ground truth.

    Args:
        wds_paths: List of webdataset paths containing OSCD data
        batch_size: Batch size for the dataloader
        num_workers: Number of worker processes for data loading
        shuffle_size: Size of the shuffle buffer for webdataset (default: 100)
        to_neg_1_1: Whether to normalize images to [-1, 1] range (default: False)
        resample: Whether to resample the dataset indefinitely (default: True)
        pin_memory: Whether to pin memory in CUDA tensors (default: True)
        shuffle: Whether to shuffle the data (default: True)
        remove_meta: Whether to remove metadata from samples (default: False)
        norm_img_keys: List of image keys to normalize (default: ["img1", "img2", "gt"])
        prefetch_factor: Number of batches to prefetch per worker (default: 2)

    Returns:
        tuple: A tuple containing:
            - dataset: WebDataset instance
            - dataloader: WebLoader instance for batched data loading

    Example:
        >>> (
        ...     dataset,
        ...     dataloader,
        ... ) = create_oscd_loader(
        ...     wds_paths=[
        ...         "path/to/oscd.tar"
        ...     ],
        ...     batch_size=32,
        ...     num_workers=4,
        ... )
        >>> for (
        ...     batch
        ... ) in dataloader:
        ...     # Process OSCD data with img1, img2, and gt keys
        ...     pass
    """
    # 1. create dataset
    dataset = wds.WebDataset(
        wds_paths,
        resampled=resample,  # no need `iter(dataloader)` for `next` function
        shardshuffle=False,
        nodesplitter=wds.shardlists.single_node_only
        if not accelerate.state.PartialState().use_distributed
        else wds.shardlists.split_by_node,  # split_by_node if is multi-node training
        workersplitter=wds.shardlists.split_by_worker,
        seed=2025,
        verbose=True,
        detshuffle=True if shuffle_size > 0 else False,
        empty_check=False,
        handler=wds.warn_and_continue,
    )
    dataset = dataset.decode(
        *[wds.handle_extension("tif tiff", tiff_decode_io), "torch"],
        handler=wds.reraise_exception,
    )

    dataset = dataset.map(remove_extension)
    if remove_meta:
        dataset = dataset.map(remove_meta_data)

    # 1.1 norm image
    dataset = dataset.map(
        partial(
            shared_times_norm_img if shared_norm else norm_img,
            to_neg_1_1=to_neg_1_1,
            norm_keys=norm_img_keys,
            per_channel=True,
            quantile_clip=0.99,
        )
    )

    # 1.2 map gt to int
    if remapped_gt is not None:
        assert "changed" in remapped_gt and "unchanged" in remapped_gt, (
            f"remapped_gt must contain 'changed' and 'unchanged' keys, got {remapped_gt}"
        )
    dataset = dataset.map_dict(gt=partial(gt_to_int, gt_changes=remapped_gt))

    # 1.3 all to tensor
    as_tensor_fn = partial(as_tensor, pin_memory=False)
    dataset = dataset.map_dict(**{key: as_tensor_fn for key in ["img1", "img2", "gt"]})

    # 1.4 crop images
    if crop_size is not None:
        dataset = dataset.map(random_img_crop(to_2tuple(crop_size)))

    # 2. create dataloader
    if shuffle and shuffle_size > 0:
        dataset = dataset.shuffle(shuffle_size)
    dataloader = wds.WebLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=pin_memory,
        drop_last=False,
        shuffle=False,
        # multiprocessing_context="spawn" if pin_memory else None,
    )

    # 2.1 augment images
    if random_flip_prob > 0:
        augmentation = random_img_augment(random_flip_prob)
        dataloader = dataloader.map(augmentation)

    return dataset, dataloader


def create_window_slider_oscd_dataloader(
    wds_paths: list[str],
    num_workers: int,
    window_size: int = 64,
    stride: int | None = None,
    overlap: float | None = None,
    slide_keys: list[str] = ["img1", "img2", "gt"],
    shuffle_size: int = 0,
    to_neg_1_1: bool = False,
    resample: bool = True,
    pin_memory: bool = False,
    shuffle: bool = False,
    remove_meta: bool = False,
    norm_img_keys: list[str] = ["img1", "img2"],
) -> tuple[
    wds.WebDataset, Generator[dict[str, torch.Tensor | np.ndarray | str], None, None]
]:
    """
    Create a window sliding dataloader for OSCD (Onera Satellite Change Detection) dataset.

    This function creates a dataloader that processes large satellite images by extracting
    smaller windows using sliding window technique. This is useful for processing large images
    that cannot be processed in their entirety due to memory constraints.

    Args:
        wds_paths: List of webdataset paths containing OSCD data
        num_workers: Number of worker processes for data loading
        window_size: Size of the square window to extract (default: 64)
        stride: Stride between windows. If None, uses window_size (no overlap)
        overlap: Overlap ratio between windows (0 to 1). If specified, stride is calculated automatically
        slide_keys: List of keys in the sample that should be processed with window sliding
        shuffle_size: Size of the shuffle buffer for webdataset
        to_neg_1_1: Whether to normalize images to [-1, 1] range
        resample: Whether to resample the dataset indefinitely
        prefetch_factor: Number of batches to prefetch per worker
        pin_memory: Whether to pin memory in CUDA tensors
        shuffle: Whether to shuffle the data
        remove_meta: Whether to remove metadata from samples
        norm_img_keys: List of image keys to normalize

    Returns:
        Generator that yields windowed samples with change detection data

    Example:
        >>> dataloader = create_window_slider_oscd_dataloader(
        ...     wds_paths=[
        ...         "path/to/oscd.tar"
        ...     ],
        ...     num_workers=4,
        ...     window_size=128,
        ...     overlap=0.5,
        ... )
        >>> for (
        ...     batch
        ... ) in dataloader:
        ...     # Process windowed change detection data
        ...     pass
    """
    # 1. Create base dataloader using create_oscd_loader with batch_size=1 for window sliding
    dataset, base_dataloader = create_oscd_loader(
        wds_paths=wds_paths,
        batch_size=1,  # Use batch_size=1 for window sliding
        num_workers=num_workers,
        shuffle_size=shuffle_size,
        to_neg_1_1=to_neg_1_1,
        resample=resample,
        pin_memory=pin_memory,
        shuffle=shuffle,
        remove_meta=remove_meta,
        norm_img_keys=norm_img_keys,
    )

    # 2. Create windowed dataloader generator
    windowed_dataloader = create_windowed_dataloader(
        dataloader=base_dataloader,
        slide_keys=slide_keys,
        window_size=window_size,
        stride=stride,
        overlap=overlap,
    )

    return dataset, windowed_dataloader


# * --- Test --- #


def test_oscd_wind_loader():
    ds, dl = create_window_slider_oscd_dataloader(
        wds_paths=["data/Downstreams/ChangeDetection/OSCD/OSCD_13bands_train.tar"],
        num_workers=0,
        window_size=128,
        overlap=0.5,
        to_neg_1_1=False,
    )
    for batch in dl:
        img1 = batch["img1"]
        img2 = batch["img2"]
        label = batch["gt"]
        print(img1.shape, img2.shape, label.shape, "label unique", torch.unique(label))


def test_oscd_loader(mode=None):
    from tqdm import tqdm

    gt_remapped = {
        "changed": 1,
        "unchanged": 2,
    }
    # path = ["data/Downstreams/ChangeDetection/OSCD/OSCD_13bands_train.npy.tar"]
    path = ["data/Downstreams/ChangeDetection/OSCD/OSCD_13bands_test.npy.tar"]
    _, dl = create_oscd_loader(
        path,
        batch_size=2,
        num_workers=8,
        to_neg_1_1=False,
        shuffle=True,
        shuffle_size=10,
        shared_norm=True,
        remapped_gt=None,
        crop_size=256,
        random_flip_prob=0.8,
        pin_memory=True,
        prefetch_factor=4,
    )
    for batch in tqdm(dl):
        img1 = batch["img1"]
        img2 = batch["img2"]
        label = batch["gt"]
        label = torch.zeros_like(label)
        if mode == "label_centrical":
            for img1, img2, label in label_centrical_patcher(
                img1,
                img2,
                label,
                micro_batch_size=4,
                patch_size=128,
                label_mode="seg",
                changed_label=1,
                unchanged_label=0,
            ):
                print(
                    img1.shape,
                    img2.shape,
                    label.shape,
                    "label unique",
                    torch.unique(label),
                )
            print("next batch")
        elif mode == "window_slider":
            window_slider = WindowSlider(["img1", "img2", "gt"], 128, 128)
            model_out_lst = []
            for batch_s in window_slider.slide_windows(batch):
                img1, img2, label = batch_s["img1"], batch_s["img2"], batch_s["gt"]
                print(
                    img1.shape,
                    img2.shape,
                    label.shape,
                    "label unique",
                    torch.unique(label),
                )
                # dummy model
                label += 1
                win_info = batch_s["window_info"]
                model_out_lst.append({"gt": label, "window_info": win_info})
            # merge
            gt_merged = window_slider.merge_windows(model_out_lst, merged_keys=["gt"])[
                "gt"
            ]
            print("merged gt shape", gt_merged, "unique", torch.unique(gt_merged))
        else:
            # do nothing
            print(
                img1.shape,
                img2.shape,
                label.shape,
                "label unique",
                torch.unique(label),
            )


if __name__ == "__main__":
    test_oscd_loader(mode="label_centrical")
    # test_oscd_wind_loader()
