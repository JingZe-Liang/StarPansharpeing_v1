from typing import Callable, List

import torch
from torch import Tensor
from litdata import StreamingDataLoader, StreamingDataset
from kornia.augmentation import (
    AugmentationSequential,
    RandomResizedCrop,
    RandomHorizontalFlip,
    RandomVerticalFlip,
)
from kornia.constants import Resample
from contextlib import contextmanager
from functools import partial
from loguru import logger

from src.data import _BaseStreamingDataset
from src.data.utils import normalize_image_


def stereo_matching_default_transforms(prob: float = 0.5):
    transform = AugmentationSequential(
        RandomHorizontalFlip(p=prob),
        RandomVerticalFlip(p=prob),
        data_keys=["input"],
        keepdim=True,
    )
    transform._disable_features = True  # disable the auto numpy / PIL transformation
    return transform


@contextmanager
def augmentation_resample_nearest_context(aug):
    is_seq_pipe = isinstance(aug, AugmentationSequential)
    nearest_flags = dict(resample=Resample.NEAREST, align_corners=None)
    if is_seq_pipe:
        try:
            _flags = [p.flags.copy() for p in aug]  # keep for recovering
            for p in aug:
                p.flags.update(nearest_flags)
            yield
        finally:
            for p, flag in zip(aug, _flags):
                p.flags = flag
    else:
        _flags = aug.flags.copy()  # keep for recovering
        try:
            aug.flags.update(nearest_flags)
            yield
        finally:
            aug.flags = _flags


class US3DStreamingDataset(_BaseStreamingDataset):
    def __init__(
        self,
        transforms: AugmentationSequential | None = None,
        output_size: int = 512,
        augmentation_prob: float = 0.5,
        to_neg_1_1=True,
        dsp_clamp_kwargs: dict = {"clamp_range": (None, 64), "norm": None},
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.to_neg_1_1 = to_neg_1_1
        self.transforms = transforms if transforms is not None else self._get_default_transforms(augmentation_prob)
        self.resize = RandomResizedCrop(
            size=(output_size, output_size), scale=(0.8, 1.0), ratio=(3 / 4, 4 / 3), p=1, keepdim=True
        )
        self.dsp_clamp_kwargs = dsp_clamp_kwargs

    def _get_default_transforms(self, p):
        return stereo_matching_default_transforms(prob=p)

    def _apply_transforms(self, left, right, dsp, agl, seg_label):
        dsp_dt, agl_dt, seg_label_dt = dsp.dtype, agl.dtype, seg_label.dtype
        dsp, agl, seg_label = dsp.float(), agl.float(), seg_label.float()

        if self.resize is not None:
            left = self.resize(left)
            right = self.resize(left, params=self.resize._params)
            with augmentation_resample_nearest_context(self.resize):
                dsp, agl, seg_label = map(partial(self.resize, params=self.resize._params), [dsp, agl, seg_label])

        if self.transforms is not None:
            left = self.transforms(left)
            right = self.transforms(right, params=self.transforms._params)
            with augmentation_resample_nearest_context(self.transforms):
                dsp, agl, seg_label = map(
                    partial(self.transforms, params=self.transforms._params), [dsp, agl, seg_label]
                )

        dsp, agl, seg_label = dsp.type(dsp_dt), agl.type(agl_dt), seg_label.type(seg_label_dt)
        return left, right, dsp, agl, seg_label

    def _apply_dsp_clamp(self, dsp: Tensor):
        if self.dsp_clamp_kwargs.get("clamp_range"):
            dsp = dsp.clip(*self.dsp_clamp_kwargs["clamp_range"])
        if self.dsp_clamp_kwargs.get("norm"):
            logger.warning(
                f"Norm={self.dsp_clamp_kwargs['norm']} is not supported. Will cause the stereo matching network train unexpectly."
            )
            _invalid_thresh = -500
            mask = dsp < _invalid_thresh
            dsp[mask] /= self.dsp_clamp_kwargs["norm"]
            dsp[~mask] = -999
        return dsp

    def _preprocess_left_right(self, left: Tensor, right: Tensor):
        # Shared min-max normalization across both images and all channels (global)
        # This removes quantizer and per-channel normalization logic as requested
        imgs = torch.stack([left, right])

        # Global min/max across all dimensions
        min_val = imgs.min()
        max_val = imgs.max()

        diff = max_val - min_val
        if diff < 1e-6:
            diff = torch.tensor(1e-6, device=left.device, dtype=left.dtype)

        left = (left - min_val) / diff
        right = (right - min_val) / diff

        left = left.clamp(0.0, 1.0)
        right = right.clamp(0.0, 1.0)

        if self.to_neg_1_1:
            left = left * 2.0 - 1.0
            right = right * 2.0 - 1.0

        return left, right

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        left, right, dsp, agl, seg_label = sample["left"], sample["right"], sample["dsp"], sample["agl"], sample["cls"]

        # Resize
        left, right = left.float(), right.float()
        left, right = self._preprocess_left_right(left, right)

        # US3D keeps -999 is invalid, accorrding to papers
        dsp = self._apply_dsp_clamp(dsp)

        # Label Mapping for US3D (DFC2019)
        # Original: 2 (Ground), 5 (Trees), 6 (Vegetation), 9 (Water), 17 (Buildings), 65 (Roads)
        # target: 0, 1, 2, 3, 4, 5
        mapping = {2: 0, 5: 1, 6: 2, 9: 3, 17: 4, 65: 5}
        mapped_seg = torch.full_like(seg_label, -100, dtype=torch.long)
        for k, v in mapping.items():
            mapped_seg[seg_label == k] = v
        seg_label = mapped_seg

        # Resize and transforms
        left, right, dsp, agl, seg_label = self._apply_transforms(left, right, dsp, agl, seg_label)

        return {
            "left": left,
            "right": right,
            "dsp": dsp[None],
            "agl": agl[None],
            "cls": seg_label,
        }

    @classmethod
    def create_dataloader(
        cls,
        input_dir: str | list[str],
        stream_ds_kwargs: dict | None = None,
        combined_kwargs: dict | None = None,
        loader_kwargs: dict | None = None,
    ):
        """Create dataset and dataloader

        Args:
            input_dir: Path or list of paths to the dataset
            stream_ds_kwargs: Arguments passed to the dataset constructor
            combined_kwargs: Arguments passed to CombinedStreamingDataset (used with multiple datasets)
            loader_kwargs: Arguments passed to StreamingDataLoader

        Returns:
            (dataset, dataloader) tuple
        """
        stream_ds_kwargs = stream_ds_kwargs or {}
        combined_kwargs = combined_kwargs or {"batching_method": "per_stream"}
        loader_kwargs = loader_kwargs or {}

        ds = cls.create_dataset(input_dir=input_dir, combined_kwargs=combined_kwargs, **stream_ds_kwargs)
        dl = StreamingDataLoader(ds, **loader_kwargs)
        return ds, dl


def __test_us3d_dataset():
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
    ds, dl = US3DStreamingDataset.create_dataloader(input_dir)

    # Print shapes
    print("Testing dataset and dataloader...")
    print(f"Dataset size: {len(ds)}")

    # Get one sample to check shapes and ranges
    # sample = next(iter(dl))
    sample = ds[2]
    print("\nSample shapes and ranges:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={list(value.shape)}, range=[{value.min().item():.4f}, {value.max().item():.4f}]")
        else:
            print(f"  {key}: {type(value)}")

    # Visualization Test - Compare old vs new
    from src.stage2.stereo_matching.utils.vis import visualize_stereo
    from src.stage2.stereo_matching.utils.vis_original import visualize_stereo_sample

    print("\nGenerating visualization...")

    # Extract first sample from batch
    left = sample["left"].cpu().numpy()  # (C, H, W) -> (H, W, C)
    right = sample["right"].cpu().numpy()
    dsp = sample["dsp"][0].cpu().numpy()  # (H, W)
    agl = sample["agl"][0].cpu().numpy()  # (H, W)
    seg = sample["cls"].cpu().numpy()  # (H, W)

    # Transpose RGB images from (C, H, W) to (H, W, C)
    left = left.transpose(1, 2, 0)
    right = right.transpose(1, 2, 0)

    # Method 1: Original visualization (vis_original.py)
    # print("1. Generating with original function (vis_original.py)...")
    # vis_original = visualize_stereo_sample(
    #     left_rgb=left,
    #     right_rgb=right,
    #     dsp_gt=dsp,
    #     agl=agl,
    #     left_seg=seg,
    #     title=f"US3D Test (Original) - {data_place} {mode}",
    # )
    # output_path_original = "test_us3d_vis_original.png"
    # vis_original.save(output_path_original)
    # print(f"   Original visualization saved to {output_path_original}")

    # Method 2: New visualization (vis.py)
    print("2. Generating with new function (vis.py)...")
    vis_new_list = visualize_stereo(
        left_rgb=left,
        right_rgb=right,
        dsp_gt=dsp,
        agl=agl,
        left_seg=seg,
        title=f"US3D Test (New) - {data_place} {mode}",
        invalid_thres=-500,
        img_normalization="neg1_1",
    )
    output_path_new = "test_us3d_vis_new.png"
    vis_new_list[0].save(output_path_new)
    print(f"   New visualization saved to {output_path_new}")


def __test_augmentation_seq():
    from kornia.constants import Resample

    # aug = RandomResizedCrop(size=(64, 64), ratio=(3 / 4, 4 / 3), resample=Resample.BILINEAR)
    x = torch.randn(1, 3, 256, 256)
    y = torch.randint(-64, 64, (1, 1, 64, 64)).float()

    # version 1.
    # aug = stereo_matching_default_transforms(["input"], 1.0)
    # x_aug = aug(x)
    # params = aug._params
    # for pipe in aug:
    #     pipe.flags |= {"resample": Resample.NEAREST, "align_corners": None}
    # y_aug = aug(y, params=params)
    # print(x_aug, y_aug)

    # version 2.
    aug = stereo_matching_default_transforms(["input", "mask"], 1.0)
    x_aug, y_aug = aug(x, y)
    print(x_aug, y_aug)


if __name__ == "__main__":
    __test_us3d_dataset()
    # __test_augmentation_seq()
