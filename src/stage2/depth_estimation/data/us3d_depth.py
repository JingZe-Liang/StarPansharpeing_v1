from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from torch import Tensor

from src.stage2.depth_estimation.utils.masking import apply_clamp_and_scale, fill_invalid, make_valid_mask
from src.stage2.stereo_matching.data.US3D import US3DStreamingDataset


class US3DDepthStreamingDataset(US3DStreamingDataset):
    def __init__(
        self,
        *,
        target_key: Literal["agl", "dsp"] = "agl",
        invalid_threshold: float = -500.0,
        clamp_range: tuple[float, float] | None = None,
        scale: float | None = None,
        norm_depth: bool = False,
        ignore_mask: bool = False,
        **kwargs,
    ):
        self.target_key = target_key
        self.invalid_threshold = float(invalid_threshold)
        self.clamp_range = [-10, 64] if clamp_range is None else clamp_range
        self.scale = scale
        self.norm_depth = norm_depth
        self.ignore_mask = ignore_mask
        super().__init__(**kwargs)

    def _get_min_max_mask(self, depth):
        min_v, max_v = self.clamp_range
        min_valid_mask = depth > min_v
        max_valid_mask = depth < max_v
        # large than 64 is valid, but smaller than -10 is not
        depth = depth.clamp(min=min_v, max=max_v)
        return min_valid_mask, depth

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        sample = super().__getitem__(idx)
        img: Tensor = sample["left"]
        depth: Tensor = sample[self.target_key].float()
        min_v, max_v = self.clamp_range

        nan_mask = torch.isnan(depth)
        if nan_mask.any():
            depth = torch.where(nan_mask, torch.ones_like(depth) * min_v, depth)

        if torch.isnan(img).any():
            img = torch.where(torch.isnan(img), torch.zeros_like(img), img)

        min_max_mask, depth = self._get_min_max_mask(depth)
        valid_mask = ~nan_mask & min_max_mask

        s = self.scale or 64
        depth = depth / float(s)

        if self.norm_depth:
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        if self.ignore_mask:
            hw = depth.shape[-2:]
            valid_mask = torch.ones(hw, dtype=torch.bool)[None]  # 1,H,W
        return {
            "img": img,
            "depth": depth,
            "valid_mask": valid_mask,
        }


def test_us3d_loader():
    data_place: str = "jax"  # [jax, oma]
    mode: str = "test"  # [train, val, test]

    data_cfg = dict(
        jax=dict(
            train="data/Downstreams/US3D_Stereo_Matching/JAX/train",
            val="data/Downstreams/US3D_Stereo_Matching/JAX/vlal",
            test="data/Downstreams/US3D_Stereo_Matching/JAX/test",
        ),
        oma=dict(
            train="data/Downstreams/US3D_Stereo_Matching/OMA/train",
            val="data/Downstreams/US3D_Stereo_Matching/OMA/val",
            test="data/Downstreams/US3D_Stereo_Matching/OMA/test",
        ),
    )

    input_dir = data_cfg[data_place][mode]
    ds = US3DDepthStreamingDataset(
        input_dir=input_dir,
        target_key="agl",
        invalid_threshold=-500.0,
        clamp_range=(0, 64),
        scale=64,
    )

    from litdata import StreamingDataLoader
    from tqdm import tqdm, trange

    loader = StreamingDataLoader(ds, batch_size=4, num_workers=1)
    for batch in tqdm(loader, desc="Loading"):
        pass


def test_us3d_depth_dataset():
    data_place: str = "jax"  # [jax, oma]
    mode: str = "train"  # [train, val, test]

    data_cfg = dict(
        jax=dict(
            train="data/Downstreams/US3D_Stereo_Matching/JAX/train",
            val="data/Downstreams/US3D_Stereo_Matching/JAX/vlal",
            test="data/Downstreams/US3D_Stereo_Matching/JAX/test",
        ),
        oma=dict(
            train="data/Downstreams/US3D_Stereo_Matching/OMA/train",
            val="data/Downstreams/US3D_Stereo_Matching/OMA/val",
            test="data/Downstreams/US3D_Stereo_Matching/OMA/test",
        ),
    )

    input_dir = data_cfg[data_place][mode]
    ds = US3DDepthStreamingDataset(
        input_dir=input_dir,
        target_key="agl",
        invalid_threshold=-500.0,
        clamp_range=(0, 64),
        scale=64,
    )
    print(ds)

    sample = ds[20]
    img = sample["img"]
    depth = sample["depth"]
    valid_mask = sample["valid_mask"]
    img = (img + 1) / 2
    print(f"img.shape: {img.shape}")

    # plot all
    import matplotlib.pyplot as plt

    fig, axes = plt.subplot_mosaic(
        [
            ["img", "depth"],
            ["valid_mask", "depth_filled"],
        ],
        figsize=(10, 8),
    )

    # Display image
    img_np = img.numpy().transpose(1, 2, 0)
    axes["img"].imshow(img_np)
    axes["img"].set_title("Image")
    axes["img"].axis("off")

    # Display depth (AGL)
    # valid mask info
    sum_valid = valid_mask.sum()
    print(f"valid ratio: {sum_valid}/{valid_mask.numel():.4f}")
    depth_np = depth.squeeze(0).numpy()
    depth_masked = np.where(valid_mask.squeeze(0).numpy(), depth_np, np.nan)
    im_depth = axes["depth"].imshow(depth_masked, cmap="viridis")
    axes["depth"].set_title("Depth (AGL)")
    axes["depth"].axis("off")
    plt.colorbar(im_depth, ax=axes["depth"], fraction=0.046, pad=0.04)

    axes["depth_filled"].imshow(depth_np, cmap="viridis")
    axes["depth_filled"].set_title("Depth (Filled Invalids)")
    plt.colorbar(im_depth, ax=axes["depth_filled"], fraction=0.046, pad=0.04)

    # Display valid mask
    valid_mask_np = valid_mask.squeeze(0).numpy().astype(np.float32)
    axes["valid_mask"].imshow(valid_mask_np, cmap="gray")
    axes["valid_mask"].set_title("Valid Mask")
    axes["valid_mask"].axis("off")

    plt.tight_layout()
    plt.savefig("us3d_depth_test_vis.png", dpi=150, bbox_inches="tight")
    print("Visualization saved to us3d_depth_test_vis.png")
    plt.close()
