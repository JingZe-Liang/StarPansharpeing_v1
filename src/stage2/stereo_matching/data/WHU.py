from typing import Callable
from functools import partial
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
import torch
from torch.utils.data import default_collate

from src.data import _BaseStreamingDataset
from src.data.utils import normalize_image_


def stereo_matching_default_transforms(prob: float = 0.5):
    """Create default augmentation pipeline for stereo matching

    Args:
        prob: Probability of applying each flip augmentation

    Returns:
        AugmentationSequential with horizontal and vertical flips
    """
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
    """Context manager to temporarily set augmentation resample mode to NEAREST

    This is used when applying augmentations to masks (dsp, agl, seg_label)
    to avoid interpolation artifacts.

    Args:
        aug: AugmentationSequential or single augmentation instance

    Yields:
        None
    """
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


def whu_collate_fn(batch):
    """Specialized collate function for WHU dataset, automatically adding agl and cls fields

    Args:
        batch: List of batch data items

    Returns:
        Collated batch dictionary, containing left, right, dsp, agl, cls
    """
    # Process the batch using torch's default collate
    collated = default_collate(batch)

    # Add fields missing in WHU dataset to maintain compatibility with the trainer
    collated["agl"] = None
    collated["cls"] = None

    return collated


class WHUStreamingDataset(_BaseStreamingDataset):
    def __init__(
        self,
        transforms: Callable | None = None,
        output_size: int = 512,
        augmentation_prob: float = 0.5,
        to_neg_1_1=True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.to_neg_1_1 = to_neg_1_1
        self.transforms = transforms if transforms is not None else self._get_default_transforms(augmentation_prob)
        self.resize = RandomResizedCrop(
            size=(output_size, output_size), scale=(0.8, 1.0), ratio=(3 / 4, 4 / 3), p=1, keepdim=True
        )

    def _get_default_transforms(self, p):
        return stereo_matching_default_transforms(prob=p)

    def _apply_transforms(self, left, right, dsp):
        """Apply augmentation transforms to left, right, and disparity map

        Args:
            left: Left image tensor
            right: Right image tensor
            dsp: Disparity map tensor

        Returns:
            Transformed left, right, and dsp tensors
        """
        dsp_dt = dsp.dtype
        dsp = dsp.float()

        if self.transforms is not None:
            left = self.transforms(left)
            right = self.transforms(right, params=self.transforms._params)
            with augmentation_resample_nearest_context(self.transforms):
                dsp = partial(self.transforms, params=self.transforms._params)(dsp)

        dsp = dsp.type(dsp_dt)
        return left, right, dsp

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
        left, right, dsp = sample["left"], sample["right"], sample["disp"]

        # Resize
        left, right = left.float(), right.float()
        left, right = self._preprocess_left_right(left, right)

        # Apply resize with proper resample mode for dsp
        dsp_dt = dsp.dtype
        dsp = dsp.float()

        if self.resize is not None:
            left = self.resize(left)
            right = self.resize(right, params=self.resize._params)
            with augmentation_resample_nearest_context(self.resize):
                dsp = partial(self.resize, params=self.resize._params)(dsp)

        dsp = dsp.type(dsp_dt)

        # Transform
        if self.transforms is not None:
            left, right, dsp = self._apply_transforms(left, right, dsp)

        return {
            "left": left,
            "right": right,
            "dsp": dsp,
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

        # Use whu_collate_fn by default unless another collate_fn is specified by the user
        if "collate_fn" not in loader_kwargs:
            loader_kwargs["collate_fn"] = whu_collate_fn

        ds = cls.create_dataset(input_dir=input_dir, combined_kwargs=combined_kwargs, **stream_ds_kwargs)
        dl = StreamingDataLoader(ds, **loader_kwargs)
        return ds, dl


def __test_dataset():
    path = "data/Downstreams/WHU_Stereo_Matching/litdata/train"

    # Create dataset directly
    print("Creating WHU dataset...")
    ds = WHUStreamingDataset(input_dir=path)
    sample = ds[2]
    print(f"Dataset size: {len(ds)}")

    # Print sample info
    left = sample["left"]
    right = sample["right"]
    dsp = sample["dsp"]
    print("\nSample shapes and ranges:")
    print(f"  left: shape={list(left.shape)}, range=[{left.min().item():.4f}, {left.max().item():.4f}]")
    print(f"  right: shape={list(right.shape)}, range=[{right.min().item():.4f}, {right.max().item():.4f}]")
    print(f"  dsp: shape={list(dsp.shape)}, range=[{dsp.min().item():.4f}, {dsp.max().item():.4f}]")

    # Visualization
    print("\nGenerating visualization...")
    from src.stage2.stereo_matching.utils.vis import visualize_stereo

    # Convert to numpy for visualization
    left_np = left.cpu().numpy()
    right_np = right.cpu().numpy()
    dsp_np = dsp.cpu().numpy()

    # Transpose RGB images from (C, H, W) to (H, W, C)
    left_np = left_np.transpose(1, 2, 0)
    right_np = right_np.transpose(1, 2, 0)

    # WHU doesn't have agl and seg_label, pass None
    # WHU images are normalized to [-1, 1] range, so specify img_normalization="neg1_1"
    # WHU uses -999.0 as invalid disparity value marker
    vis_new_list = visualize_stereo(
        left_rgb=left_np,
        right_rgb=right_np,
        dsp_gt=dsp_np,
        agl=None,
        left_seg=None,
        title="WHU Test Sample",
        invalid_thres=-500,  # WHU uses -999 for invalid disparity values
        img_normalization="neg1_1",  # Input is in [-1, 1] range
    )
    output_path = "test_whu_vis.png"
    vis_new_list[0].save(output_path)
    print(f"Visualization saved to {output_path}")


if __name__ == "__main__":
    """
    python -m src.stage2.stereo_matching.data.WHU
    """
    __test_dataset()
