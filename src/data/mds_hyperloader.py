from io import BytesIO
from typing import Callable

import numpy as np
import PIL.Image as Image
import tifffile
import torch
from streaming import Stream, StreamingDataLoader, StreamingDataset
from streaming.base.format.mds import encodings
from torch import Tensor
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import to_tensor
from typing_extensions import Annotated

from src.stage2.denoise.utils.noise_adder import (
    UniHSINoiseAdderKornia,
    get_tokenizer_trainer_noise_adder,
)

from .augmentations import hyper_transform
from .codecs import NVImageEncoding, SkipLargeJpegImageEncoding, TiffEncoding
from .utils import large_image_resizer_clipper, norm_img

encodings._encodings["tiff"] = TiffEncoding  # type: ignore
encodings._encodings["nvimage"] = NVImageEncoding  # type: ignore
encodings._encodings["jpeg"] = SkipLargeJpegImageEncoding  # type: ignore


def to_tensor_img(
    img: Image.Image | np.ndarray | Tensor, is_permuted=False
) -> torch.Tensor:
    if torch.is_tensor(img):
        if not is_permuted:
            img = img.permute(2, 0, 1)  # hwc -> chw
        return img
    else:
        if isinstance(img, Image.Image):
            img = img.convert("RGB")  # gray to rgb

        img = to_tensor(img)  # hwc -> chw
        if is_permuted:
            # img is chw orignally, need to permute to back
            # chw -> whc in to_tensor
            img = img.permute(2, 1, 0)
        return img


class ImageStreamDataset(StreamingDataset):
    def __init__(
        self,
        permute=False,
        resize_before_transform: int | None = None,
        hyper_transforms_lst: tuple[str, ...] | None = (
            "grayscale",
            "channel_shuffle",
            "rotation",
            "cutmix",
            "horizontal_flip",
            "vertical_flip",
        ),
        hyper_degradation_lst: str | list[str] | None = None,
        transform_prob: float = 0.0,
        degradation_prob: float = 0.0,
        random_apply: int | tuple[int, int] = 1,
        constraint_filtering_size: int | tuple[int, int] | None = None,
        to_neg_1_1: bool = True,
        norm_options: dict = dict(
            quantile_clip=1.0,
            sigma_clip=0.0,
            norm_info_add_in_sample=False,
            norm_type="clip_zero_div",
            per_channel=False,
        ),
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.permute = permute
        self.constraint_filtering_size = constraint_filtering_size
        self.norm_options = norm_options
        self.to_neg_1_1 = to_neg_1_1
        self._img_key = "img"

        # Resize and crop
        self.resize_before_transform = resize_before_transform
        self.use_resize_clip = resize_before_transform is not None
        if self.use_resize_clip:
            self.`resize_clip_fn` = large_image_resizer_clipper(
                img_key=self._img_key,
                tgt_size=resize_before_transform,
                op_for_large="clip",
            )

        # Augmentations
        self.use_aug = (
            hyper_transforms_lst is not None
            and len(hyper_transforms_lst) > 0
            and transform_prob > 0.0
        )
        if self.use_aug:
            self.augmentation: Callable = hyper_transform(
                op_list=hyper_transforms_lst,
                probs=transform_prob,
                random_apply=random_apply,
            )

        # Degradations
        self.degradation_prob = degradation_prob
        self.use_deg = degradation_prob > 0.0 and hyper_degradation_lst is not None
        if self.use_deg:
            if hyper_degradation_lst == "all":
                deg_pipe = get_tokenizer_trainer_noise_adder(p=degradation_prob)
            elif hyper_degradation_lst is not None:
                deg_pipe = UniHSINoiseAdderKornia(
                    noise_type=hyper_degradation_lst,
                    p=degradation_prob,
                    same_on_batch=True,
                    is_neg_1_1=True,
                )
            else:
                raise ValueError(
                    f"hyper_degradation_lst: {hyper_degradation_lst} is not supported"
                )
            self.deg_pipe = deg_pipe

    def _pil_to_tensor(self, sample):
        """Convert a PIL Image to a tensor."""
        sample[self._img_key] = to_tensor_img(
            sample[self._img_key], is_permuted=not self.permute
        )
        return sample

    def _augment(self, sample):
        """Apply augmentations to the sample."""
        if self.use_aug:
            sample[self._img_key] = self.augmentation(sample[self._img_key])
        return sample

    def _degrade(self, sample):
        """Apply degradations to the sample."""
        if self.use_deg:
            sample[self._img_key] = self.deg_pipe(sample[self._img_key])
        return sample

    def _crop_resize(self, sample):
        """Crop and resize the image in the sample if required."""
        if self.use_resize_clip:
            # force to float
            sample[self._img_key] = sample[self._img_key].float()
            sample = self.resize_clip_fn(sample)
        return sample

    def _norm_img(self, sample):
        """Normalize the image in the sample."""
        sample = norm_img(
            sample,
            permute=False,
            to_neg_1_1=self.to_neg_1_1,
            **self.norm_options,
        )
        return sample

    def __getitem__(self, idx):
        d = super().__getitem__(idx)

        # skip the none or undecoded value
        if d[self._img_key] is None or isinstance(d[self._img_key], bytes):
            return None

        # Image transformation and augmentation
        d = self._pil_to_tensor(d)
        d = self._crop_resize(d)
        d = self._norm_img(d)
        d = self._augment(d)
        d = self._degrade(d)

        return d


def collate_fn_skip_none(batch):
    batch = [d for d in batch if d is not None]
    return default_collate(batch)


if __name__ == "__main__":
    """
    python -m src.data.mds_hyperloader
    """
    path = "data/BigEarthNet_S2/MDS_hyper_images"
    stream = Stream(local=path)
    img_ds = ImageStreamDataset(
        permute=False,
        to_neg_1_1=True,
        resize_before_transform=None,
        transform_prob=0.4,
        # local=path,
        streams=[stream],
        batch_size=16,
        shuffle=True,
    )
    print(len(img_ds))

    dl = StreamingDataLoader(
        img_ds,
        batch_size=16,
        num_workers=6,
        persistent_workers=False,
        prefetch_factor=None,
        collate_fn=collate_fn_skip_none,
    )
    import time

    from tqdm import tqdm

    t1 = time.time()
    for i, sample in tqdm(enumerate(dl)):
        if i == 0:
            print(f"Batch {i}: keys={sample.keys()}, img_shape={sample['img'].shape}")
        if i > 500:
            print(f"Done. Time: {time.time() - t1}s")
            break
