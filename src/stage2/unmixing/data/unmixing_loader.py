import math
from functools import partial
from typing import Any

import accelerate
import numpy as np
import torch
import webdataset as wds
from einops import rearrange
from torch import Tensor

from src.data.codecs import tiff_decode_io
from src.data.utils import remove_extension
from src.data.window_slider import WindowSlider, create_windowed_dataloader
from src.utilities.logging import catch_any, log


def permute_img(img: Tensor | np.ndarray) -> Tensor | np.ndarray:
    img = rearrange(img, "... h w c ->  ... c h w")
    return img


def img_divisible_by(img: Tensor | np.ndarray, divisor: int = 8) -> Tensor:
    *_, h, w = img.shape
    if not torch.is_tensor(img):
        img = torch.as_tensor(img)

    hp = math.ceil(h / divisor) * divisor
    wp = math.ceil(w / divisor) * divisor
    if hp == h and wp == w:
        return torch.as_tensor(img)

    log(
        f"[Unmixing Dataset]: Resizing image from ({h}, {w}) to ({hp}, {wp}) to be divisible by {divisor}",
        warn_once_pattern=r"\[Unmixing Dataset\]: Resizing image",
        once=True,
        level="debug",
    )
    if img.ndim == 3:
        img = img[None]  # add batch dim

    img = torch.nn.functional.interpolate(
        img, size=(hp, wp), mode="bicubic", align_corners=True
    )
    img = img.squeeze(0)

    return img  # type: ignore[return-value]


def get_unmixing_dataloader(
    wds_paths: str | list[str],
    batch_size: int,
    num_workers: int,
    shuffle_size: int = 0,
    to_neg_1_1: bool = False,
    divisible_by: int = 8,
    resample: bool = True,
    pin_memory=True,
):
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
        handler=wds.reraise_exception,
    )
    dataset = dataset.decode(
        *[wds.handle_extension("tif tiff", tiff_decode_io), "torch"],
        handler=wds.reraise_exception,
    )
    dataset = dataset.map(remove_extension)
    dataset = dataset.map_dict(img=permute_img, abunds=permute_img)
    if divisible_by > 0 and isinstance(divisible_by, int):
        div_fn = partial(img_divisible_by, divisor=divisible_by)
        dataset = dataset.map_dict(img=div_fn, abunds=div_fn)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        shuffle=False,
    )

    return dataset, dataloader


# * --- Test --- * #


def test_loader() -> None:
    path = "data/Downstreams/UrbanUnmixing/splits/DC.tar"
    # ds, dl = get_unmixing_dataloader(path, 1, 0, resample=True)

    # print("=== Testing Full Image DataLoader ===")
    # for i, sample in enumerate(dl):
    #     if i >= 1:  # Only test first sample
    #         break
    #     print(f'vca indices: {sample["init_vca_indices"][0]}')
    #     for k, v in sample.items():
    #         if not k.startswith("__"):
    #             print(f"{k}: {v.shape}")
    #     print("-" * 60)

    # print("\n=== Testing Window Slider ===")
    # # Reset dataloader
    # ds, dl = get_unmixing_dataloader(path, 1, 0, resample=True)

    # # Create window generator - only slide 'img' and 'abunds' keys
    # window_gen = create_windowed_dataloader(
    #     dl, slide_keys=["img", "abunds"], window_size=64, stride=32
    # )

    # # Collect some windows for testing merge
    # window_results = {}
    # for i, window_sample in enumerate(window_gen):
    #     if i >= 9:  # Collect first 9 windows
    #         break

    #     print(f"Window {i}:")
    #     for k, v in window_sample.items():
    #         if not k.startswith("__") and k != "window_info":
    #             print(f"  {k}: {v.shape}")

    #     # Store for merge test
    #     window_info = window_sample["window_info"]
    #     pos = (window_info["batch_idx"], window_info["i"], window_info["j"])
    #     window_results[pos] = window_sample

    # print("\n=== Testing Merge ===")
    # # Test merging
    # merged = WindowSlider.merge_windows(window_results, merge_method="average")
    # print("Merged shapes:")
    # for k, v in merged.items():
    #     if not k.startswith("__"):
    #         print(f"  {k}: {v.shape}")

    # Test data consistency (no overlap case)
    print("\n=== Testing Data Consistency (No Overlap) ===")
    # Reset dataloader to get original sample
    ds, dl = get_unmixing_dataloader(
        path, batch_size=1, num_workers=0, resample=False, divisible_by=64
    )  # Use divisible_by=64 to avoid resizing
    original_sample = next(iter(dl))

    # Create window generator with no overlap (stride = window_size)
    window_gen_no_overlap = create_windowed_dataloader(
        dl,
        slide_keys=["img", "abunds"],
        window_size=64,
        stride=32,  # No overlap
    )

    # Collect all windows for complete merge
    window_results_no_overlap = {}
    for i, window_sample in enumerate(window_gen_no_overlap):
        window_info = window_sample["window_info"]
        pos = (window_info["batch_idx"], window_info["i"], window_info["j"])
        window_results_no_overlap[pos] = window_sample
        if i % 10 == 0:
            print(f"Collected {i} windows, shaped as {window_sample['img'].shape}")

    # Test merging with no overlap
    merged_no_overlap = WindowSlider.merge_windows(
        window_results_no_overlap, merge_method="average"
    )

    # Compare merged data with original data for slide_keys
    slide_keys = ["img", "abunds"]
    for key in slide_keys:
        if key in merged_no_overlap and key in original_sample:
            original_data = original_sample[key]
            merged_data = merged_no_overlap[key]

            # Convert to numpy for comparison
            if torch.is_tensor(original_data):
                original_np = original_data.cpu().numpy()
            else:
                original_np = original_data

            if torch.is_tensor(merged_data):
                merged_np = merged_data.cpu().numpy()
            else:
                merged_np = merged_data

            # Calculate difference
            diff = np.abs(original_np - merged_np)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)

            print(f"{key} consistency check:")
            print(f"  Original shape: {original_np.shape}")
            print(f"  Merged shape: {merged_np.shape}")
            print(f"  Max difference: {max_diff}")
            print(f"  Mean difference: {mean_diff}")
            print(f"  Data identical: {max_diff < 1e-6}")

            if max_diff > 1e-6:
                print(f"  WARNING: Data mismatch detected for {key}!")

                # Debug: Check if shapes match
                if original_np.shape != merged_np.shape:
                    print(
                        f"  Shape mismatch: original {original_np.shape} vs merged {merged_np.shape}"
                    )
            else:
                print(f"  ✓ Data consistency verified for {key}")


if __name__ == "__main__":
    test_loader()
