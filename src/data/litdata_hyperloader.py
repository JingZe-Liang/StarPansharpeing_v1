import random
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import PIL.Image as Image
import torch
from kornia.augmentation import AugmentationSequential, RandomResizedCrop, Resize
from litdata import (
    CombinedStreamingDataset,
    ParallelStreamingDataset,
    StreamingDataLoader,
    StreamingDataset,
)
from litdata.utilities import dataset_utilities as litdata_utils
from loguru import logger
from torch import Tensor
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import to_tensor
from typing_extensions import Any, Optional, Union, cast

from src.stage2.denoise.utils.noise_adder import (
    UniHSINoiseAdderKornia,
    get_tokenizer_trainer_noise_adder,
)
from src.utilities.config_utils import function_config_to_basic_types

from .augmentations import hyper_transform
from .utils import large_image_resizer_clipper, norm_img


def to_tensor_img(
    img: Image.Image | np.ndarray | Tensor,
    is_permuted: bool = True,
    repeat_gray_n: int = 3,
    force_to_rgb=False,
) -> torch.Tensor:
    is_arr = False
    if torch.is_tensor(img) or (is_arr := isinstance(img, np.ndarray)):
        if is_arr:
            img = torch.from_numpy(img)

        img = cast(torch.Tensor, img)
        if img.ndim == 2:
            img = img.unsqueeze(0)

        if not is_permuted:
            # img = img.permute(2, 0, 1)  # hwc -> chw
            # in serilizer, img is chw orignally, need to permute to back
            # chw (dataset) -> wch (serializer) -> permute(1, -1, 0)
            img = img.permute(1, -1, 0)

        if img.shape[0] == 1 and repeat_gray_n > 1:
            # use expand instead of repeat
            img = img.expand(repeat_gray_n, -1, -1)
        if img.shape[0] == 4 and force_to_rgb:
            img = img[:3]

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


class IndexedCombinedStreamingDataset(CombinedStreamingDataset):
    """A CombinedStreamingDataset that also returns the index of each sample."""

    def __init__(self, combined_is_cycled=False, *args, **kwargs) -> None:
        kwargs.setdefault("seed", 2025)
        super().__init__(*args, **kwargs)
        self.combined_is_cycled = combined_is_cycled

        if not combined_is_cycled:
            self.__check_can_be_indexed()
            self.accum_lens = np.cumsum([0] + [len(ds) for ds in self._datasets])

        # set the epoch to 0
        self.current_epoch = 0

    def __check_can_be_indexed(self) -> None:
        """Check if the dataset can be indexed."""
        if not self._iterate_over_all:
            raise ValueError(
                "IndexedCombinedStreamingDataset only supports iterate_over_all=True"
            )

    def _check_datasets(self, datasets: list[StreamingDataset]) -> None:
        # if any(not isinstance(d, StreamingDataset) for d in datasets):
        #     raise RuntimeError("The provided datasets should be instances of the StreamingDataset.")
        return

    def set_weights(self, weights: list[float]):
        self._weights = weights
        logger.debug(f"Set weights to {weights}", not_rank0_print=True)

    def __len__(self) -> int | float:
        """Return the total number of samples across all datasets."""
        if self.combined_is_cycled:
            return float("inf")
        return self.accum_lens[-1]

    def __getitem__(self, idx: int) -> dict:
        if self.combined_is_cycled:
            logger.error(
                "The combined dataset is cycled, not supported for indexing. "
                "Return using __iter__ method to sample."
            )
            raise IndexError(f"Index {idx} not supported for cycled combined dataset.")

        # Check bounds
        total_length = self.accum_lens[-1]
        if idx < 0 or idx >= total_length:
            raise IndexError(
                f"Index {idx} out of range for dataset of length {total_length}"
            )

        # find the dataset containing this index
        ds_idx = int(np.searchsorted(self.accum_lens, idx, side="right") - 1)
        ds_idx = int(max(0, min(int(ds_idx), len(self._datasets) - 1)))
        sample_idx = idx - int(self.accum_lens[ds_idx])

        # Ensure sample_idx is within bounds
        if sample_idx < 0 or sample_idx >= len(self._datasets[ds_idx]):
            raise IndexError(
                f"Index {idx} resulted in invalid sample_idx {sample_idx} for dataset {ds_idx}"
            )

        return self._datasets[ds_idx][sample_idx]


class SingleCycleStreamingDataset(ParallelStreamingDataset):
    def __init__(
        self,
        dataset: StreamingDataset,  # must be one dataset to cycle
        length: int | float = float("inf"),
        *args,
        **kwargs,
    ):
        kwargs.setdefault("seed", 2025)
        super().__init__(
            [dataset], length=length, transform=self.transform, *args, **kwargs
        )

    def _check_datasets(self, dataset):
        # do nothing check
        return None

    def transform(self, zipped_samples: zip):
        (sample_d,) = zipped_samples
        return sample_d


class _BaseStreamingDataset(StreamingDataset):
    """
    Fixed index file name support.
    """

    def __init__(self, *args, **kwargs) -> None:
        input_dir = Path(kwargs.pop("input_dir"))
        index_file_name = None
        if input_dir.is_dir():
            pass
        elif Path(input_dir).is_file() and Path(input_dir).suffix == ".json":
            index_file_name = input_dir.stem
            input_dir = input_dir.parent
        else:
            raise ValueError(
                f"input_dir must be a directory or a index json file, got: {input_dir}"
            )

        # FIXME: I don't know why the 'index_path' args does not work
        index_file_name = kwargs.pop("index_file_name", index_file_name)
        if index_file_name is not None:
            litdata_utils._INDEX_FILENAME = f"{index_file_name}.json"

        kwargs.setdefault("seed", 2025)
        super().__init__(input_dir=input_dir, *args, **kwargs)

        # change back
        if index_file_name is not None:
            litdata_utils._INDEX_FILENAME = "index.json"

    def _check_datasets(self, datasets: list[StreamingDataset]) -> None:
        # override the class

        # if any(not isinstance(d, StreamingDataset) for d in datasets):
        #     raise RuntimeError("The provided datasets should be instances of the StreamingDataset.")
        return

    @classmethod
    def create_dataset(
        cls,
        input_dir: str | list[str],
        other_ds: StreamingDataLoader | list[StreamingDataLoader] | None = None,
        combined_kwargs: dict = {"batching_method": "stratified"},
        is_cycled: bool = False,
        *args,
        **kwargs,
    ):
        """
        combined_kwargs: dict
            weights: Optional[Sequence[float]] = None,
            iterate_over_all: bool = True,
            batching_method: BatchingMethodType = "stratified",
            force_override_state_dict: bool = False,
        """
        if isinstance(input_dir, str):
            ds = cls(input_dir=input_dir, *args, **kwargs)
        elif isinstance(input_dir, list) and len(input_dir) == 1:
            ds = cls(input_dir=input_dir[0], *args, **kwargs)
        else:
            streams = [cls(input_dir=d, *args, **kwargs) for d in input_dir]
            ds = IndexedCombinedStreamingDataset(datasets=streams, **combined_kwargs)

        if other_ds is not None:
            ds = IndexedCombinedStreamingDataset(
                datasets=[ds]
                + ([other_ds] if not isinstance(other_ds, list) else other_ds),
                **combined_kwargs,
            )

        if is_cycled:
            ds = SingleCycleStreamingDataset(dataset=ds)
            # logger.debug(f"Create a cycled dataset for input dir: {input_dir}")

        return ds

    @classmethod
    def create_dataloader(
        cls,
        input_dir: str | list[str],
        stream_ds_kwargs: dict = {},
        combined_kwargs: dict = {"batching_method": "per_stream"},
        loader_kwargs: dict = {},
    ):
        ds = cls.create_dataset(
            input_dir=input_dir, combined_kwargs=combined_kwargs, **stream_ds_kwargs
        )
        dl = StreamingDataLoader(ds, **loader_kwargs)
        return ds, dl


class ImageStreamingDataset(_BaseStreamingDataset):
    def __init__(
        self,
        is_hwc: bool = True,
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
        force_to_rgb: bool = False,
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
        index_file_name: str | None = None,
        check_chans: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(index_file_name=index_file_name, *args, **kwargs)

        self.is_hwc = is_hwc
        self.constraint_filtering_size = constraint_filtering_size
        self.norm_options = norm_options
        self.to_neg_1_1 = to_neg_1_1
        self.force_to_rgb = force_to_rgb
        self._img_key = "img"
        self._check_chans= check_chans

        # Resize and crop
        self.resize_before_transform = resize_before_transform
        self.use_resize_clip = resize_before_transform is not None
        if self.use_resize_clip:
            self.resize_clip_fn = large_image_resizer_clipper(
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
        # hwc -> chw in serializer
        sample[self._img_key] = to_tensor_img(
            sample[self._img_key],
            is_permuted=self.is_hwc,
            force_to_rgb=self.force_to_rgb,
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
            sample, permute=False, to_neg_1_1=self.to_neg_1_1, **self.norm_options
        )
        return sample

    def _filter_only_img(self, sample):
        kept_keys = ["__key__", "img"]
        d = {k: v for k, v in sample.items() if k in kept_keys}
        # assert len(d) == len(kept_keys), d.keys()
        return d

    def _conditions_select_random_one(self, sample):
        if "hed" in sample:
            new_sample = {}
            # hed, segmentation, sketch, mlsd
            cond_keys = ["hed", "segmentation", "sketch", "mlsd"]
            c = random.choice(cond_keys)
            img = sample[c]
            logger.debug(f"Selected condition: {c}", once=True)
            new_sample[self._img_key] = img
            new_sample["__key__"] = sample["__key__"]
            return new_sample
        return sample

    def __check_chans_for_hyper_images(self, d, orig_img_shape: torch.Size, idx: int):
        """Check the number of channels for hyper images"""
        assert d["img"].shape[0] in [
            3,
            4,
            8,
            10,
            12,
            13,
            32,
            50,
            150,
            175,
            202,
            224,
            242,
            368,
            369,
        ], f"{d['img'].shape=}, {orig_img_shape=}, {d['__key__']=}, {idx=}"

    def _skip_undecode(self, d):
        # skip the none or undecoded value
        if self._img_key not in d:
            logger.error(
                f"Image key {self._img_key} not found in sample: {d['__key__']} "
                f"existing keys are {d.keys()}"
            )
            return None
        if d[self._img_key] is None or isinstance(d[self._img_key], bytes):
            logger.warning(f"Skip the {type(d[self._img_key])} value: {d['__key__']}")
            return None

        return d

    def __getitem__(self, idx):
        d = super().__getitem__(idx)

        d = self._conditions_select_random_one(d)
        d = self._skip_undecode(d)

        _orig_img_shape = d[self._img_key].shape

        # Image transformation and augmentation
        d = self._pil_to_tensor(d)
        d = self._crop_resize(d)
        d = self._norm_img(d)
        d = self._augment(d)
        d = self._degrade(d)
        # if not 'per_stream' shuffle, make sure each stream has the same keys
        d = self._filter_only_img(d)

        if self._check_chans:
            self.__check_chans_for_hyper_images(d, _orig_img_shape, idx)

        return d


class ConditionsStreamingDataset(_BaseStreamingDataset):
    def __init__(self, to_neg_1_1=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._condition_keys = ["hed", "segmentation", "sketch", "mlsd"]
        self.to_neg_1_1 = to_neg_1_1

    def __getitem__(self, idx):
        d = super().__getitem__(idx)
        assert len(d) == len(self._condition_keys) + 1, (
            f"Condition keys missing in the data: {d.keys()}"
        )

        for k in self._condition_keys:
            d[k] = to_tensor_img(d[k], is_permuted=True)

        d = norm_img(
            d, norm_keys=self._condition_keys, permute=False, to_neg_1_1=self.to_neg_1_1
        )

        return d


class CaptionStreamingDataset(_BaseStreamingDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        # {'caption': dict, 'valid_legth': int}
        d = super().__getitem__(idx)

        if isinstance(d["caption"], dict):
            caption = d["caption"]["caption"]
            assert isinstance(caption, str), (
                f"Caption must be a string, got {type(caption)}"
            )
            d["caption"] = caption
        return d  # {caption: str, __key__: str}


class GenerativeStreamingDataset(ParallelStreamingDataset):
    _modality_ds_keys = ["condition_ds", "caption_ds", "img_ds"]

    def __init__(
        self,
        img_ds: StreamingDataset,
        condition_ds: StreamingDataset,
        caption_ds: StreamingDataset,
        *,
        resize: int,
        post_process_type: str = "none",
    ) -> None:
        self._condition_keys = ["hed", "segmentation", "sketch", "mlsd"]
        self._caption_key = "caption"
        self._img_key = "img"
        self._post_process_type = post_process_type

        # patch __init__
        super().__init__(
            [img_ds, condition_ds, caption_ds],
            transform=self.transform,
        )

        # datasets
        self.img_ds = img_ds
        self.condition_ds = condition_ds
        self.caption_ds = caption_ds

        # check if the same length
        assert len(img_ds) == len(condition_ds) == len(caption_ds), (
            f"Datasets must have the same length: "
            f"img_ds={len(img_ds)}, condition_ds={len(condition_ds)}, caption_ds={len(caption_ds)}"
        )

        # crop_resize
        self.resize = resize
        self.crop_resize_fn = AugmentationSequential(
            # first resize to 1024 for better cropping
            Resize(size=1024, p=1.0),
            # then the img, conditions are crop at the same place
            RandomResizedCrop(
                size=(self.resize, self.resize)
                if isinstance(self.resize, int)
                else self.resize,
                p=1.0,
                scale=(0.6, 1.0),
                ratio=(0.75, 1.33),
                keepdim=True,
                cropping_mode="resample",
            ),
            # conditions and image
            data_keys=["input"] * (len(self._condition_keys) + 1),
            keepdim=True,
        )

    def _crop_resize(self, d):
        cond_imgs = [d[k] for k in self._condition_keys]
        img = d[self._img_key]
        proc_imgs = self.crop_resize_fn(*cond_imgs, img)

        # Save
        for i, k in enumerate(self._condition_keys + [self._img_key]):
            d[k] = proc_imgs[i]

        return d

    def __check_is_paired(self, *samples):
        check_by_key = "__key__"
        _prev_val = None
        for d in samples:
            if _prev_val is None:
                _prev_val = d[check_by_key]
            else:
                if _prev_val != d[check_by_key]:
                    return False

        return True

    @staticmethod
    def __sana_post_process(
        d: dict, condition_keys: list[str]
    ) -> tuple[Tensor, Tensor | float, Tensor | float, dict[str, str | list[Tensor]]]:
        y_lc = torch.nan  # caption feature if has
        y_text: str = d["caption"]["caption"]
        y_mask_l = torch.nan  # mask
        data_info = {
            "y_text": y_text,
            "valid_length": len(y_text),
            "inp_shape": d["img"].shape[0],  # channel
        }
        img_chw: Tensor = d["img"]
        cond_chw: list[Tensor] = [d[c] for c in condition_keys]
        data_info["control_signal"] = cond_chw
        return img_chw, y_lc, y_mask_l, data_info

    def _post_process(self, d: dict):
        if self._post_process_type == "sana":
            d = self.__sana_post_process(d, self._condition_keys)
        elif self._post_process_type in ("none", None):
            pass
        else:
            raise ValueError(f"Unknown post process type: {self._post_process_type}")

        return d

    def transform(self, samples: tuple[dict, ...], rng) -> dict:
        """Static method to transform a dict sample."""
        img_d, cond_d, caption_d = samples
        # Check is paired
        if not self.__check_is_paired(img_d, cond_d, caption_d):
            raise ValueError(
                f"Not paired: {img_d['__key__']}, {cond_d['__key__']}, {caption_d['__key__']}"
            )

        # img, cond_imgs, cond_texts
        sample_d = {
            "__key__": img_d["__key__"],
            "img": img_d[self._img_key],
            **{k: cond_d[k] for k in self._condition_keys},
            "caption": caption_d[self._caption_key],
        }

        # Image transformation and augmentation
        sample_d = self._crop_resize(sample_d)

        # Post process to reorganize the dict
        sampled_d = self._post_process(sample_d)

        return sample_d

    @classmethod
    def create_dataset(
        cls,
        img_input_dir: str | list[str],
        condition_input_dir: str | list[str],
        caption_input_dir: str | list[str],
        img_kwargs: dict = {},
        cond_kwargs: dict = {},
        caption_kwargs: dict = {},
        gen_kwargs: int = 512,
    ):
        # Pairs: img, conditions, captions
        img_ds = ImageStreamingDataset.create_dataset(
            input_dir=img_input_dir, **img_kwargs
        )
        cond_ds = ConditionsStreamingDataset.create_dataset(
            input_dir=condition_input_dir, **cond_kwargs
        )
        caption_ds = CaptionStreamingDataset.create_dataset(
            input_dir=caption_input_dir, **caption_kwargs
        )

        return cls(
            img_ds=img_ds, condition_ds=cond_ds, caption_ds=caption_ds, **gen_kwargs
        )

    @classmethod
    def create_dataloader(
        cls,
        img_input_dir: str | list[str],
        condition_input_dir: str | list[str],
        caption_input_dir: str | list[str],
        img_kwargs: dict = {},
        cond_kwargs: dict = {},
        caption_kwargs: dict = {},
        gen_kwargs: dict = {},
        loader_kwargs: dict = {},
    ):
        ds = cls.create_dataset(
            img_input_dir=img_input_dir,
            condition_input_dir=condition_input_dir,
            caption_input_dir=caption_input_dir,
            img_kwargs=img_kwargs,
            cond_kwargs=cond_kwargs,
            caption_kwargs=caption_kwargs,
            gen_kwargs=gen_kwargs,
        )
        dl = StreamingDataLoader(ds, **loader_kwargs)
        return ds, dl

    # def load_state_dict(self, obj: dict[str, Any]):
    #     """Load state dict for the dataset and its components."""

    #     for mk in self._modality_ds_keys:
    #         if mk not in obj:
    #             raise ValueError(f"Modality dataset key {mk} not found in state dict.")
    #         getattr(self, mk).load_state_dict(obj[mk])
    #         logger.debug(f"Loaded state dict for {mk}")

    # def state_dict(
    #     self, num_samples_yielded: int, num_workers: int, batch_size: int
    # ) -> dict[str, Any]:
    #     """Return state dict for the dataset and its components."""
    #     kwargs = {
    #         "num_samples_yielded": num_samples_yielded,
    #         "num_workers": num_workers,
    #         "batch_size": batch_size,
    #     }

    #     sd = {}
    #     for mk in self._modality_ds_keys:
    #         sd[mk] = getattr(self, mk).state_dict(**kwargs)
    #     return sd


class SizeBasedBatchsizeStreamingDataloader(StreamingDataLoader):
    def __init__(
        self,
        dataset,
        size_based_batch_sizes: dict[int, int],  # size: batch_size
        cache_minor=False,
        *args: Any,
        batch_size: int = 1,
        num_workers: int = 0,
        profile_batches: Union[bool, int] = False,
        profile_skip_batches: int = 0,
        profile_dir: Optional[str] = None,
        prefetch_factor: Optional[int] = None,
        shuffle: Optional[bool] = None,
        drop_last: Optional[bool] = None,
        collate_fn: Optional[Callable] = None,
        **kwargs: Any,
    ):
        super().__init__(
            dataset,
            *args,
            batch_size=batch_size,
            num_workers=num_workers,
            profile_batches=profile_batches,
            profile_skip_batches=profile_skip_batches,
            profile_dir=profile_dir,
            prefetch_factor=prefetch_factor,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=collate_fn,
            **kwargs,
        )
        self.size_bs_s = size_based_batch_sizes
        self.cache_minor = cache_minor

        if self.cache_minor:
            self._size_caches: dict[tuple[int, int], list] = defaultdict(list)

    def _yield_from_cache(self, channel: int, size: int):
        cache_key = (channel, size)
        if cache_key not in self._size_caches:
            return

        target_bs = self.size_bs_s[size]
        cache = self._size_caches[cache_key]

        total_samples = sum(batch["img"].shape[0] for batch in cache)
        if total_samples > 64:
            logger.warning(
                f"Cache for {cache_key} is too large, total samples: {total_samples}"
            )

        while total_samples >= target_bs:
            collected_samples = []
            collected_keys = None
            samples_collected = 0

            while cache and samples_collected < target_bs:
                batch = cache[0]
                batch_samples = batch["img"].shape[0]

                if collected_keys is None:
                    collected_keys = list(batch.keys())
                    collected_samples = {k: [] for k in collected_keys}

                needed = target_bs - samples_collected
                take = min(needed, batch_samples)

                for k in collected_keys:
                    collected_samples[k].append(batch[k][:take])

                samples_collected += take

                if take == batch_samples:
                    cache.pop(0)
                else:
                    for k in collected_keys:
                        batch[k] = batch[k][take:]
                    break

            if samples_collected == target_bs:
                final_batch = {}
                for k in collected_keys:
                    if torch.is_tensor(collected_samples[k][0]):
                        final_batch[k] = torch.cat(collected_samples[k], dim=0)
                    else:
                        final_batch[k] = [
                            item for sublist in collected_samples[k] for item in sublist
                        ]

                yield final_batch

                total_samples = sum(batch["img"].shape[0] for batch in cache)

    def _add_to_cache(self, batch: dict, channel: int, size: int):
        cache_key = (channel, size)
        self._size_caches[cache_key].append(batch)

    def _yield_remaining_cache(self):
        for (channel, size), batches in self._size_caches.items():
            if batches:
                final_batch = {}
                _keys = batches[0].keys()
                for key in _keys:
                    if torch.is_tensor(batches[0][key]):
                        final_batch[key] = torch.cat([b[key] for b in batches], dim=0)
                        logger.debug(
                            f"Yielding batch of size {final_batch['img'].shape}"
                        )
                    else:
                        final_batch[key] = [
                            item for sublist in batches for item in sublist[key]
                        ]

                yield final_batch

        if len(self._size_caches) > 0:
            logger.warning("Still has some cache in, may leak some samples.")
        self._size_caches.clear()

    def __iter__(self):
        for batch in super().__iter__():
            img = batch["img"]
            size = img.shape[-1]  # bs, c, h, w
            channel = img.shape[1]  # bs, c, h, w

            if self.cache_minor:
                yield from self._yield_from_cache(channel, size)

            if size in self.size_bs_s:
                target_bs = self.size_bs_s[size]
                cur_bs = img.shape[0]

                if cur_bs >= target_bs:
                    for i in range(0, cur_bs, target_bs):
                        yield {k: v[i : i + target_bs] for k, v in batch.items()}
                else:
                    if self.cache_minor:
                        # logger.warning(
                        #     f"[Size based DL]: Current batch size {cur_bs} (channel {channel}) is smaller than "
                        #     f"target batch size {target_bs} for size {size}. Added to cache.",
                        #     once_pattern=r"\[Size based DL]\: Current batch size .*",
                        # )
                        self._add_to_cache(batch, channel, size)
                    else:
                        # logger.warning(
                        #     f"[Size based DL]: Current batch size {cur_bs} (channel {channel}) is smaller than "
                        #     f"target batch size {target_bs} for size {size}. Added to cache.",
                        #     once_pattern=r"\[Size based DL]\: .* is smaller than .*",
                        # )
                        yield batch
            else:
                yield batch

        if self.cache_minor:
            yield from self._yield_remaining_cache()


def collate_fn_skip_none(
    check_only_first_n: int = 2000,
    size_bs_s: dict[int, int] | None = None,
    raise_if_shape_mismatch=False,
):
    """
    Collate function that skips None values and optionally splits batches based on size.

    Parameters
    ----------
    check_only_first_n : int
        Number of batches to check for channel size consistency
    size_bs_s : dict[int, int] | None
        Size to batch size mapping for dynamic batching

    Returns
    -------
    Callable
        Collate function
    """
    checked_n = 0

    def inner(batch: list):
        nonlocal checked_n

        # Check if is None
        _orig_len = len(batch)
        batch = [d for d in batch if d is not None]
        if len(batch) == 0:
            return None
        elif len(batch) < _orig_len:
            logger.warning(
                f"Skip {(_orig_len - len(batch))} samples in the batch due to None values."
            )

        # Check shapes are matched
        if checked_n < check_only_first_n:
            _prev_chans = batch[0]["img"].shape[0]
            _shape_mismatch = False

            for i, b in enumerate(batch):
                if b["img"].shape[0] != _prev_chans:
                    logger.error(
                        f"Skip a sample in the batch due to different channel size: "
                        f"expected {_prev_chans}, got {b['img'].shape[0]}."
                    )
                    _shape_mismatch = True
                    if not raise_if_shape_mismatch:
                        # not raise, just remove the sample
                        batch.pop(i)

                if _shape_mismatch:
                    logger.error(
                        f"sample {i}: shape is {b['img'].shape}, key: {b['__key__']}"
                    )

            if _shape_mismatch and raise_if_shape_mismatch:
                # Total batch shapes
                for b in batch:
                    logger.error(
                        f"sample {i}: shape is {b['img'].shape}, key: {b['__key__']}"
                    )
                raise ValueError("Different channel size in the batch.")
            checked_n += 1

        if len(batch) > 0:
            batch = default_collate(batch)
        else:
            batch = None

        return batch

    return inner


@function_config_to_basic_types
def create_hyper_image_litdata_flatten_paths_loader(
    paths: dict[str, dict],
    weights: list[float | int] | None = None,
    stream_ds_kwargs: dict = {
        "transform_prob": 0.0,
        "resize_before_transform": 256,
        "shuffle": False,
        "is_cycled": True,
        "is_hwc": True,
    },
    loader_kwargs: dict = {
        "batch_size": 8,
        "num_workers": 16,
        "persistent_workers": True,
        "prefetch_factor": None,
        "shuffle": False,
    },
    macro_sampled_batch_size: dict[int, int] = {
        128: 16,
        256: 12,
        512: 6,
    },
):
    paths_dict = {}
    # Flatten all files
    for name, path_dict in paths.items():
        for sub_name, path_kwgs in path_dict.items():
            paths_dict[sub_name] = path_kwgs

    dataset = []
    for name, (paths, kwargs) in paths_dict.items():
        stream_ds_kwargs_ = stream_ds_kwargs.copy()
        stream_ds_kwargs_.update(kwargs)
        ds = ImageStreamingDataset.create_dataset(
            input_dir=paths,
            combined_kwargs={"batching_method": "per_stream"},
            **stream_ds_kwargs_,
        )
        dataset.append(ds)
        logger.info(f"Create dataset for {name} with paths: {paths}")

    # composite
    ds_total = IndexedCombinedStreamingDataset(
        combined_is_cycled=True,
        datasets=dataset,
        weights=[1.0] * len(dataset) if weights is None else weights,
        iterate_over_all=False,
        seed=2025,
        batching_method="per_stream",
    )

    loader_kwargs["collate_fn"] = collate_fn_skip_none(1000)
    dataloader = SizeBasedBatchsizeStreamingDataloader(
        ds_total,
        size_based_batch_sizes=macro_sampled_batch_size,
        cache_minor=True,
        **loader_kwargs,
    )

    # statistics
    # from tqdm import tqdm

    # from src.utilities.logging import configure_logger

    # configure_logger(_auto_=False, add_tqdm_filter=True)

    # bands_info_n = {}
    # print("Start testing...")
    # for i, sample in tqdm(  # type: ignore
    #     enumerate(dl),
    #     # total=len(ds_total) // dl.batch_size,
    # ):
    #     if i == 0 and "__key__" not in sample:
    #         print(sample.keys())

    #     chan = sample["img"].shape[1]
    #     if chan not in bands_info_n:
    #         bands_info_n[chan] = 0
    #     bands_info_n[chan] += sample["img"].shape[0]

    #     # logger.debug(f"Batch {i}: shape {sample['img'].shape=}")
    #     if i % 10 == 0:
    #         logger.info(
    #             f"channel samples: {', '.join(f'{k}: {v}' for k, v in bands_info_n.items())}",
    #             tqdm=True,
    # )

    return dataset, dataloader


def __test_create_hyper_image_litdata_loader():
    from .path_consts import HYPERSPECTRAL_PATHS, MULTISPECTRAL_PATHS, RGB_PATHS

    stream_ds_kwargs = {
        "transform_prob": 0.0,
        "resize_before_transform": 256,
        "is_cycled": True,
    }

    loader_kwargs = {
        "batch_size": 18,
        "num_workers": 16,
        "persistent_workers": False,
        "prefetch_factor": None,
        "collate_fn": collate_fn_skip_none,
        "shuffle": False,
    }

    # RGB dataset
    stream_ds_kwargs_ = stream_ds_kwargs.copy()
    rgb_paths, kwargs = list(RGB_PATHS.values())[0]
    stream_ds_kwargs_.update(kwargs)
    rgb_ds = ImageStreamingDataset.create_dataset(
        input_dir=rgb_paths,
        combined_kwargs={"batching_method": "stratified"},
        **stream_ds_kwargs_,
    )

    # Multispectral dataset
    multi_ds = []
    for name, (paths, kwargs) in MULTISPECTRAL_PATHS.items():
        stream_ds_kwargs_ = stream_ds_kwargs.copy()
        stream_ds_kwargs_.update(kwargs)
        ds = ImageStreamingDataset.create_dataset(
            input_dir=paths,
            combined_kwargs={"batching_method": "stratified"},
            **stream_ds_kwargs_,
        )
        multi_ds.append(ds)

    # Hyperspectral dataset
    hyper_ds = []
    for name, (paths, kwargs) in HYPERSPECTRAL_PATHS.items():
        stream_ds_kwargs_ = stream_ds_kwargs.copy()
        stream_ds_kwargs_.update(kwargs)
        ds = ImageStreamingDataset.create_dataset(
            input_dir=paths,
            combined_kwargs={"batching_method": "stratified"},
            **stream_ds_kwargs_,
        )
        hyper_ds.append(ds)

    # * Composite the dataset ===============
    total_ds_lst = [rgb_ds, *multi_ds, *hyper_ds]
    # total_ds_lst = hyper_ds
    ds = IndexedCombinedStreamingDataset(
        combined_is_cycled=True,
        datasets=total_ds_lst,
        weights=[1.0] * len(total_ds_lst),
        iterate_over_all=False,
        batching_method="per_stream",
    )
    dl = SizeBasedBatchsizeStreamingDataloader(
        ds, size_based_batch_sizes={128: 16, 256: 12, 512: 6}, **loader_kwargs
    )

    # from src.utilities.logging import configure_logger

    # configure_logger(add_tqdm_filter=True, _auto_=False)

    # # statistics
    # bands_info_n = {}

    # print("Start testing...")
    # for i, sample in tqdm(  # type: ignore
    #     enumerate(dl),
    #     # total=len(ds) // dl.batch_size,
    # ):
    #     if i == 0 and "__key__" not in sample:
    #         print(sample.keys())

    #     chan = sample["img"].shape[1]
    #     if chan not in bands_info_n:
    #         bands_info_n[chan] = 0
    #     bands_info_n[chan] += sample["img"].shape[0]
    #     if i % 10 == 0:
    #         logger.info(
    #             f"channel samples: {', '.join(f'{k}: {v}' for k, v in bands_info_n.items())}",
    #             tqdm=True,
    #         )
    #         # print(f"channel samples: {str(bands_info_n)}")

    #     # if i > 500:
    #     #     break

    return ds, dl


def __test_index_file_litdata_loader():
    index_file = "data/RS5M/LitData_images_val/val_index.json"
    input_dir = "data/RS5M/LitData_images_val/"

    stream_ds_kwargs = {
        "transform_prob": 0.0,
        "resize_before_transform": 256,
        "is_cycled": False,
        "index_file_name": "val_index",
    }
    ds = ImageStreamingDataset.create_dataset(
        input_dir=input_dir,
        combined_kwargs={"batching_method": "stratified"},
        index_path=index_file,
        **stream_ds_kwargs,
    )


def __test_normal_image_loader():
    from litdata.streaming.serializers import BytesSerializer, StringSerializer

    path = "data2/HyperspectralEarth/LitData_hyper_images"
    stream_ds_kwargs = {
        "transform_prob": 0.0,
        "resize_before_transform": 128,
        "is_cycled": False,
    }
    # serializers = {
    #     "__key__": StringSerializer(),
    #     "img": BytesSerializer(),
    # }
    ds = ImageStreamingDataset.create_dataset(
        input_dir=path,
        combined_kwargs={"batching_method": "per_stream"},
        # serializers=serializers,
        **stream_ds_kwargs,
    )
    loader_kwargs = {
        "batch_size": 4,
        "num_workers": 2,
    }

    dl = StreamingDataLoader(ds, **loader_kwargs)
    for sample in dl:
        print(sample["img"].shape)


def __test_get_item_key():
    path = "data2/RemoteSAM270k/LitData_hyper_images"
    stream_ds_kwargs = {
        "transform_prob": 0.0,
        "resize_before_transform": 128,
        "is_cycled": False,
    }
    ds = ImageStreamingDataset.create_dataset(
        input_dir=path,
        combined_kwargs={"batching_method": "per_stream"},
        # serializers=serializers,
        **stream_ds_kwargs,
    )

    for i in range(len(ds)):
        print(ds[i]["__key__"])


if __name__ == "__main__":
    """
    python -m src.data.litdata_hyperloader
    """

    # create_hyper_image_litdata_loader()
    # create_hyper_image_litdata_flatten_paths_loader()
    # test_index_file_litdata_loader()
    # __test_normal_image_loader()
    # __test_get_item_key()

    from omegaconf import OmegaConf

    cfg = OmegaConf.load(
        "scripts/configs/tokenizer_gan/dataset/litdata_one_loader.yaml"
        # "scripts/configs/tokenizer_gan/dataset/litdata_hyperspectral.yaml"
    )

    logger.info(cfg.train_loader.paths)
    create_hyper_image_litdata_flatten_paths_loader(
        paths=cfg.train_loader.paths,
        weights=cfg.train_loader.weights,
        loader_kwargs=cfg.train_loader.loader_kwargs,
    )
