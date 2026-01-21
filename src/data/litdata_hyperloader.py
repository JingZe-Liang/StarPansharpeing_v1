import random
from collections import defaultdict
from pathlib import Path
from collections.abc import Iterable, Iterator
from typing import Callable, Literal

import numpy as np
import PIL.Image as Image
import torch
from easydict import EasyDict as edict
from kornia.augmentation import AugmentationSequential, RandomResizedCrop, Resize
from litdata import (
    CombinedStreamingDataset,
    ParallelStreamingDataset,
    StreamingDataLoader,
    StreamingDataset,
)
import litdata as ld
from loguru import logger
from torch import Tensor
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import to_tensor
from typing_extensions import Any, Optional, Union, cast

import src.data.codecs  # register serializers and deserializers
from src.stage2.denoise.utils.noise_adder import (
    UniHSINoiseAdderKornia,
    get_tokenizer_trainer_noise_adder,
)
from src.utilities.config_utils import function_config_to_basic_types, set_defaults
from src.utilities.train_utils import time_recorder, TimeRecorder

from .augmentations import hyper_transform
from .cycled_dataset import CycledDataset
from .utils import large_image_resizer_clipper, norm_img


def _get_index_file_state() -> dict[str, str]:
    return {
        "constants": ld.constants._INDEX_FILENAME,
        "dataset_utilities": ld.utilities.dataset_utilities._INDEX_FILENAME,
        "streaming_dataset": ld.streaming.dataset._INDEX_FILENAME,
        "streaming_cache": ld.streaming.cache._INDEX_FILENAME,
        "streaming_reader": ld.streaming.reader._INDEX_FILENAME,
        "streaming_config": ld.streaming.config._INDEX_FILENAME,
    }


def _apply_index_file_state(state: dict[str, str]) -> None:
    targets = {
        "constants": ld.constants,
        "dataset_utilities": ld.utilities.dataset_utilities,
        "streaming_dataset": ld.streaming.dataset,
        "streaming_cache": ld.streaming.cache,
        "streaming_reader": ld.streaming.reader,
        "streaming_config": ld.streaming.config,
    }
    for key, module in targets.items():
        setattr(module, "_INDEX_FILENAME", state[key])


def _set_index_file(file_name: str = "index.json") -> None:
    _apply_index_file_state(
        {
            "constants": file_name,
            "dataset_utilities": file_name,
            "streaming_dataset": file_name,
            "streaming_cache": file_name,
            "streaming_reader": file_name,
            "streaming_config": file_name,
        }
    )


def _reset_index_file():
    ld.constants._INDEX_FILENAME = "index.json"
    ld.utilities.dataset_utilities._INDEX_FILENAME = "index.json"
    ld.streaming.dataset._INDEX_FILENAME = "index.json"
    ld.streaming.cache._INDEX_FILENAME = "index.json"
    ld.streaming.reader._INDEX_FILENAME = "index.json"
    ld.streaming.config._INDEX_FILENAME = "index.json"


def _as_index_filename(index_file_name: str | None) -> str | None:
    if index_file_name is None:
        return None
    if index_file_name.endswith(".json"):
        return index_file_name
    return f"{index_file_name}.json"


class _IndexFileOverride:
    def __init__(self, index_file_name: str | None) -> None:
        self.index_file_name = _as_index_filename(index_file_name)
        self._prev: dict[str, str] | None = None

    def __enter__(self) -> "_IndexFileOverride":
        if self.index_file_name is None:
            return self
        self._prev = _get_index_file_state()
        _set_index_file(self.index_file_name)
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        if self._prev is not None:
            _apply_index_file_state(self._prev)


def to_tensor_img(
    img: Image.Image | np.ndarray | Tensor,
    is_permuted: bool = True,
    repeat_gray_n: int = 3,
    force_to_rgb=False,
) -> torch.Tensor | None:
    if img is None:
        return None

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
            raise ValueError("IndexedCombinedStreamingDataset only supports iterate_over_all=True")

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
                "The combined dataset is cycled, not supported for indexing. Return using __iter__ method to sample."
            )
            raise IndexError(f"Index {idx} not supported for cycled combined dataset.")

        # Check bounds
        total_length = self.accum_lens[-1]
        if idx < 0 or idx >= total_length:
            raise IndexError(f"Index {idx} out of range for dataset of length {total_length}")

        # find the dataset containing this index
        ds_idx = int(np.searchsorted(self.accum_lens, idx, side="right") - 1)
        ds_idx = int(max(0, min(int(ds_idx), len(self._datasets) - 1)))
        sample_idx = idx - int(self.accum_lens[ds_idx])

        # Ensure sample_idx is within bounds
        if sample_idx < 0 or sample_idx >= len(self._datasets[ds_idx]):
            raise IndexError(f"Index {idx} resulted in invalid sample_idx {sample_idx} for dataset {ds_idx}")

        return self._datasets[ds_idx][sample_idx]

    def _split_num_samples_yielded(self, total: int) -> list[int]:
        if total <= 0 or len(self._datasets) == 0:
            return [0 for _ in range(len(self._datasets))]

        weights = list(self._weights) if self._weights is not None else []
        clean_weights = [float(w) if w is not None and float(w) > 0 else 0.0 for w in weights]

        if len(clean_weights) != len(self._datasets):
            clean_weights = [0.0 for _ in range(len(self._datasets))]

        weight_sum = sum(clean_weights)
        if weight_sum <= 0:
            lens = [max(int(get_dataset_len(ds)), 0) for ds in self._datasets]
            len_sum = sum(lens)
            if len_sum <= 0:
                per = total // max(1, len(self._datasets))
                out = [per for _ in range(len(self._datasets))]
                out[0] += total - sum(out)
                return out
            clean_weights = [l / len_sum for l in lens]
        else:
            clean_weights = [w / weight_sum for w in clean_weights]

        raw = [total * w for w in clean_weights]
        counts = [int(v) for v in raw]
        remainder = total - sum(counts)
        if remainder > 0:
            frac = [rv - int(rv) for rv in raw]
            for idx in sorted(range(len(frac)), key=lambda i: frac[i], reverse=True)[:remainder]:
                counts[idx] += 1
        return counts

    def state_dict(
        self,
        num_workers: int,
        batch_size: int,
        num_samples_yielded: int | list[int] | None = None,
    ) -> dict[str, Any]:
        if isinstance(num_samples_yielded, int):
            num_samples_yielded = self._split_num_samples_yielded(num_samples_yielded)
        return super().state_dict(num_workers, batch_size, num_samples_yielded)

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not state_dict:
            return
        if "dataset" in state_dict:
            super().load_state_dict(state_dict)
            return

        for dataset_idx, dataset in enumerate(self._datasets):
            key = str(dataset_idx)
            if key in state_dict:
                dataset.load_state_dict(state_dict[key])


class SingleCycleStreamingDataset(ParallelStreamingDataset):
    def __init__(
        self,
        dataset: Any,  # can be a wrapper dataset as well
        *,
        length: int | float | None = float("inf"),
        seed: int = 2025,
        resume: bool = True,
        reset_rngs: bool = False,
        force_override_state_dict: bool = False,
    ) -> None:
        super().__init__(
            cast(list[StreamingDataset], [dataset]),
            length=length,
            force_override_state_dict=force_override_state_dict,
            transform=cast(Any, self._transform),
            seed=seed,
            resume=resume,
            reset_rngs=reset_rngs,
        )

    def _check_datasets(self, datasets: list[StreamingDataset]) -> None:
        # do nothing check
        return

    def _transform(self, samples: tuple[Any, ...], rngs: Any = None) -> Any:
        (sample_d,) = samples
        return sample_d

    def __getitem__(self, index: int) -> Any:
        return self._datasets[0][index]

    def __iter__(self) -> Iterator[Any]:
        while True:
            iterator = super().__iter__()
            yielded_any = False
            while True:
                try:
                    item = next(iterator)
                except StopIteration:
                    break
                yielded_any = True
                yield item

            if not yielded_any:
                yield None

    def state_dict(
        self,
        num_workers: int,
        batch_size: int,
        num_samples_yielded: int | list[int] | None = None,
    ) -> dict[str, Any]:
        inner = self._datasets[0]
        if isinstance(num_samples_yielded, list):
            total_yielded = int(num_samples_yielded[0]) if len(num_samples_yielded) > 0 else 0
        elif isinstance(num_samples_yielded, int):
            total_yielded = int(num_samples_yielded)
        else:
            total_yielded = 0

        if hasattr(inner, "state_dict"):
            cycle_length = int(len(inner)) if hasattr(inner, "__len__") else 0
            in_cycle = 0 if cycle_length <= 0 else (total_yielded % cycle_length)
            return {
                "0": inner.state_dict(
                    num_samples_yielded=in_cycle,
                    num_workers=num_workers,
                    batch_size=batch_size,
                )
            }

        return {
            "0": {"num_samples_yielded": total_yielded, "num_workers": int(num_workers), "batch_size": int(batch_size)}
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        inner = self._datasets[0]
        if not state_dict:
            return

        if "dataset" in state_dict:
            super().load_state_dict(state_dict)
            return

        if "0" in state_dict and hasattr(inner, "load_state_dict"):
            inner.load_state_dict(state_dict["0"])
            return

        if hasattr(inner, "load_state_dict"):
            inner.load_state_dict(state_dict)


class _BaseStreamingDataset(StreamingDataset):
    """
    Fixed index file name support.
    """

    def __init__(self, *, input_dir: str, **kwargs: Any) -> None:
        input_dir, index_file_name = self._change_litdata_index_file(input_dir, kwargs)
        kwargs.setdefault("seed", 2025)
        super().__init__(input_dir=str(input_dir), **kwargs)

        self.index_file_name = index_file_name

        # change back
        # hacky: call __len__ to cache first and then reset the original index file name
        if self.__len__() > 0:
            try:
                _ = self[0]
            except Exception as e:
                logger.warning(f"Failed to pre-load the first item for {input_dir}: {e}")

        if index_file_name is not None:
            _reset_index_file()

    def _change_litdata_index_file(self, path: str | Path, kwargs: dict[str, Any]):
        """
        Change the index file name of litdata.
        """
        path = Path(path)
        index_file_name = None
        if path.is_dir():
            pass
        elif path.is_file() and path.suffix == ".json":
            index_file_name = path.stem
            path = path.parent
        else:
            raise ValueError(f"input_dir must be a directory or a index json file, got: {path}")

        index_file_name = kwargs.pop("index_file_name", None) or index_file_name
        if index_file_name is not None:
            index_filename = _as_index_filename(index_file_name)
            _set_index_file(cast(str, index_filename))
            assert (path / cast(str, index_filename)).exists(), (
                f"Index file not found: {path / cast(str, index_filename)}"
            )
            logger.debug(f"Set litdata index file name to: {index_filename}")

        return path, index_file_name

    def _check_datasets(self, datasets: list[StreamingDataset]) -> None:
        """override the original method
        do nothing check.
        """

        # if any(not isinstance(d, StreamingDaaset) for d in datasets):
        #     raise RuntimeError("The provided datasets should be instances of the StreamingDataset.")
        return

    def _create_cache(self, worker_env: Any):
        with _IndexFileOverride(self.index_file_name):
            return super()._create_cache(worker_env=worker_env)

    @classmethod
    def create_dataset(
        cls,
        input_dir: str | list[str],
        other_ds: StreamingDataLoader | list[StreamingDataLoader] | None = None,
        combined_kwargs: dict = {"batching_method": "stratified"},
        is_cycled: bool = False,
        **kwargs: Any,
    ):
        """
        combined_kwargs: dict
            weights: Optional[Sequence[float]] = None,
            iterate_over_all: bool = True,
            batching_method: BatchingMethodType = "stratified",
            force_override_state_dict: bool = False,
        """
        if isinstance(input_dir, str):
            ds = cls(input_dir=input_dir, **kwargs)
        elif isinstance(input_dir, list) and len(input_dir) == 1:
            ds = cls(input_dir=input_dir[0], **kwargs)
        else:
            streams = [cls(input_dir=d, **kwargs) for d in input_dir]
            ds = IndexedCombinedStreamingDataset(datasets=streams, **combined_kwargs)

        if other_ds is not None:
            ds = IndexedCombinedStreamingDataset(
                datasets=[ds] + ([other_ds] if not isinstance(other_ds, list) else other_ds),
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
        ds = cls.create_dataset(input_dir=input_dir, combined_kwargs=combined_kwargs, **stream_ds_kwargs)
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
        **kwargs: Any,
    ) -> None:
        super().__init__(
            input_dir=cast(str, kwargs.pop("input_dir")),
            index_file_name=index_file_name,
            **kwargs,
        )

        self.is_hwc = is_hwc
        self.constraint_filtering_size = constraint_filtering_size
        self.norm_options = norm_options
        self.to_neg_1_1 = to_neg_1_1
        self.force_to_rgb = force_to_rgb
        self._img_key = "img"
        self._check_chans = check_chans

        # Resize and crop
        self.resize_before_transform = resize_before_transform
        self.use_resize_clip = resize_before_transform is not None
        if self.use_resize_clip:
            tgt_size = cast(int, resize_before_transform)
            self.resize_clip_fn = large_image_resizer_clipper(
                img_key=self._img_key,
                tgt_size=tgt_size,
                op_for_large="clip",
            )

        # Augmentations
        self.use_aug = hyper_transforms_lst is not None and len(hyper_transforms_lst) > 0 and transform_prob > 0.0
        if self.use_aug:
            self.augmentation: Callable = hyper_transform(
                op_list=cast(tuple[str, ...], hyper_transforms_lst),
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
                raise ValueError(f"hyper_degradation_lst: {hyper_degradation_lst} is not supported")
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
        sample = norm_img(sample, permute=False, to_neg_1_1=self.to_neg_1_1, **self.norm_options)
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
        # fmt: off
        assert d["img"].shape[0] in [ 3, 4, 8, 10, 12, 13, 32, 50, 150, 175, 202, 224, 242, 368, 369], (
            f"{d['img'].shape=}, {orig_img_shape=}, {d['__key__']=}, {idx=}"
        )
        # fmt: on

    def _skip_undecode(self, d):
        # skip the none or undecoded value
        if self._img_key not in d:
            logger.error(f"Image key {self._img_key} not found in sample: {d['__key__']} existing keys are {d.keys()}")
            return None
        if d[self._img_key] is None or isinstance(d[self._img_key], bytes):
            logger.warning(f"Skip the {type(d[self._img_key])} value: {d['__key__']}")
            return None

        return d

    def _ensure_dict_sample(self, sample):
        if not isinstance(sample, dict):
            # Assume the sample is the img tensor/array
            assert isinstance(sample, (torch.Tensor, np.ndarray)), (
                f"sample must be dict or tensor/array, got {type(sample)}"
            )
            sample = {"img": sample, "__key__": "N/A"}  # N/A means no key saved in litdata dataset.
        return sample

    def __getitem__(self, idx):
        d = super().__getitem__(idx)
        if d is None:
            logger.warning(f"Caught Exception in __getitem__. Returning None.")
            return None

        d = self._ensure_dict_sample(d)
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

        return edict(d)


class ConditionsStreamingDataset(_BaseStreamingDataset):
    def __init__(self, to_neg_1_1: bool = True, **kwargs: Any) -> None:
        super().__init__(input_dir=cast(str, kwargs.pop("input_dir")), **kwargs)
        self._condition_keys = ["hed", "segmentation", "sketch", "mlsd"]
        self.to_neg_1_1 = to_neg_1_1

    def __getitem__(self, idx):
        d = super().__getitem__(idx)

        assert len(d) == len(self._condition_keys) + 1, f"Condition keys missing in the data: {d.keys()}"

        for k in self._condition_keys:
            if d[k] is None:
                # Check if any can not be decoded, or return all the condition as None
                return None
            d[k] = to_tensor_img(d[k], is_permuted=True)

        d = norm_img(d, norm_keys=self._condition_keys, permute=False, to_neg_1_1=self.to_neg_1_1)

        return edict(d)


class CaptionStreamingDataset(_BaseStreamingDataset):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(input_dir=cast(str, kwargs.pop("input_dir")), **kwargs)

    def __getitem__(self, idx):
        # {'caption': dict, 'valid_legth': int}
        d = super().__getitem__(idx)
        if d is None:
            return None

        if isinstance(d["caption"], dict):
            caption = d["caption"]["caption"]
            assert isinstance(caption, str), f"Caption must be a string, got {type(caption)}"
            d["caption"] = caption
        return edict(d)


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
        super().__init__([img_ds, condition_ds, caption_ds], transform=cast(Any, self.transform))

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
                size=(self.resize, self.resize) if isinstance(self.resize, int) else self.resize,
                p=1.0,
                scale=(0.8, 1.0),
                ratio=(0.75, 1.33),
                keepdim=True,
                cropping_mode="resample",
            ),
            # conditions and image
            data_keys=["input"] * (len(self._condition_keys) + 1),
            keepdim=True,
        )

    def __len__(self) -> int:
        return len(self.img_ds)

    def __getitem__(self, idx: int) -> Any:
        samples = (self.img_ds[idx], self.condition_ds[idx], self.caption_ds[idx])
        if any(s is None for s in samples):
            return None
        return self.transform(samples, rngs=None)

    def _crop_resize(self, d):
        cond_imgs = [d[k] for k in self._condition_keys]
        img = d[self._img_key]
        proc_imgs = self.crop_resize_fn(*cond_imgs, img)

        # Save
        _prev_sz = None
        for i, k in enumerate(self._condition_keys + [self._img_key]):
            img = proc_imgs[i]

            # Ensure RGB channels
            if img.ndim == 2:
                img = img[None].repeat_interleave(3, dim=0)
            elif img.ndim == 3 and img.shape[0] == 1:
                img = img.repeat_interleave(3, dim=0)

            if _prev_sz is None:
                _prev_sz = img.shape[-2:]
            else:
                assert _prev_sz == img.shape[-2:], (
                    f"Size mismatch after crop_resize for key {k}: {_prev_sz} vs {img.shape[-2:]}"
                )

            d[k] = img

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
    def __sana_post_process(d: dict[str, Any], condition_keys: list[str]) -> tuple[Tensor, Any, Any, dict[str, Any]]:
        y_lc = torch.nan  # caption feature if has
        y_text: str = d["caption"]
        y_mask_l = torch.nan  # mask
        data_info: dict[str, Any] = {
            "y_text": y_text,
            "valid_length": len(y_text),
            "inp_shape": d["img"].shape[0],  # channel
            "control_images": {k: d[k] for k in condition_keys},
        }
        img_chw: Tensor = d["img"]
        return img_chw, y_lc, y_mask_l, data_info

    def _post_process(self, d: dict[str, Any]) -> Any:
        if self._post_process_type == "sana":
            return self.__sana_post_process(d, self._condition_keys)
        elif self._post_process_type in ("none", None):
            return d
        else:
            raise ValueError(f"Unknown post process type: {self._post_process_type}")
        return d

    def __check_fully_decoded(self, item: dict[str, Any]):
        for k, v in item.items():
            if v is None:
                logger.warning(f"item {k} is undecoded.")
                return False
        return True

    def transform(self, samples: tuple[Any, ...], rngs: Any = None) -> Any:
        """Static method to transform a dict sample."""
        img_d, cond_d, caption_d = cast(tuple[dict[str, Any], dict[str, Any], dict[str, Any]], samples)

        # Check if all fully decoded
        if not all(list(map(self.__check_fully_decoded, [img_d, cond_d, caption_d]))):
            logger.warning(f"[Gen Streaming Dataset]: found one smaple undecoded. Skip this sample.")
            return None

        # Check is paired
        if not self.__check_is_paired(img_d, cond_d, caption_d):
            raise ValueError(f"Not paired: {img_d['__key__']}, {cond_d['__key__']}, {caption_d['__key__']}")

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

        return sampled_d

    @classmethod
    def create_dataset(
        cls,
        img_input_dir: str | list[str],
        condition_input_dir: str | list[str],
        caption_input_dir: str | list[str],
        img_kwargs: dict = {},
        cond_kwargs: dict = {},
        caption_kwargs: dict = {},
        gen_kwargs: dict = {"resize": 512},
    ):
        # Pairs: img, conditions, captions
        img_ds = ImageStreamingDataset.create_dataset(input_dir=img_input_dir, **img_kwargs)
        cond_ds = ConditionsStreamingDataset.create_dataset(input_dir=condition_input_dir, **cond_kwargs)
        caption_ds = CaptionStreamingDataset.create_dataset(input_dir=caption_input_dir, **caption_kwargs)

        return cls(img_ds=img_ds, condition_ds=cond_ds, caption_ds=caption_ds, **gen_kwargs)

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


class _TimedLoaderIterator:
    def __init__(
        self, dataloader: Iterable[Any], recorder: TimeRecorder | None = time_recorder, name: str = "loader_iter"
    ) -> None:
        self._iter = iter(dataloader)
        self._recorder = recorder
        self._name = name

    def __iter__(self) -> "_TimedLoaderIterator":
        return self

    def __next__(self) -> Any:
        if self._recorder is None:
            return next(self._iter)
        with self._recorder.record(self._name):
            return next(self._iter)


class SizeBasedBatchsizeStreamingDataloader(StreamingDataLoader):
    def __init__(
        self,
        dataset,
        size_based_batch_sizes: dict[int, int | None],  # size: batch_size
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
        assert target_bs is not None, f"Batch size cannot be None for size {size} in cache"
        cache = self._size_caches[cache_key]

        total_samples = sum(batch["img"].shape[0] for batch in cache)
        if total_samples > 64:
            logger.warning(f"Cache for {cache_key} is too large, total samples: {total_samples}")

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
                assert collected_keys is not None

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

            if samples_collected == target_bs:
                final_batch = {}
                assert collected_keys is not None
                for k in collected_keys:
                    if torch.is_tensor(collected_samples[k][0]):
                        final_batch[k] = torch.cat(collected_samples[k], dim=0)
                    else:
                        final_batch[k] = [item for sublist in collected_samples[k] for item in sublist]

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
                        logger.debug(f"Yielding batch of size {final_batch['img'].shape}")
                    else:
                        final_batch[key] = [item for sublist in batches for item in sublist[key]]

                yield final_batch

        if len(self._size_caches) > 0:
            logger.warning("Still has some cache in, may leak some samples.")
        self._size_caches.clear()

    def __iter__(self):
        for batch in super().__iter__():
            if batch is None:
                continue
            img = batch["img"]
            size = img.shape[-1]  # bs, c, h, w
            channel = img.shape[1]  # bs, c, h, w

            if size not in self.size_bs_s or self.size_bs_s[size] is None:
                yield batch
                continue

            if self.cache_minor:
                yield from self._yield_from_cache(channel, size)

            if size in self.size_bs_s:
                target_bs = self.size_bs_s[size]
                assert target_bs is not None, f"Batch size cannot be None for size {size}"
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


def get_dataset_len(ds):
    if isinstance(ds, ParallelStreamingDataset):
        # If it's a ParallelStreamingDataset, return the length of the first dataset
        return len(ds._datasets[0])
    else:
        return len(ds)


def collate_fn_skip_none(
    check_only_first_n: int | None = None,
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

    def inner(batch: list[Any]) -> Any:
        nonlocal checked_n

        # Check if is None
        _orig_len = len(batch)
        batch = [d for d in batch if d is not None]  # skip None
        if len(batch) == 0:
            return None
        elif len(batch) < _orig_len:
            logger.warning(f"Skip {(_orig_len - len(batch))} samples in the batch due to None values.")

        # Check shapes are matched
        if check_only_first_n is None or checked_n < check_only_first_n:
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
                    logger.error(f"sample {i}: shape is {b['img'].shape}, key: {b['__key__']}")

            if _shape_mismatch and raise_if_shape_mismatch:
                # Total batch shapes
                for b in batch:
                    logger.error(f"sample {i}: shape is {b['img'].shape}, key: {b['__key__']}")
                raise ValueError("Different channel size in the batch.")
            checked_n += 1

        if len(batch) == 0:
            return None
        return default_collate(batch)

    return inner


@function_config_to_basic_types
def create_hyper_image_litdata_flatten_paths_loader(
    paths: Any,
    weights: list[float | int] | None = None,
    stream_ds_kwargs: dict = {
        "transform_prob": 0.0,
        "resize_before_transform": 256,
        "shuffle": False,  # BUG: shuffle is True will cause the loader return only least ds samples.
        "is_cycled": True,
        "is_hwc": True,
    },
    loader_kwargs: dict = {
        "batch_size": 8,
        "num_workers": 8,
        "persistent_workers": True,
        "prefetch_factor": None,
        "shuffle": False,
    },
    macro_sampled_batch_size: dict[int, int | None] = {
        128: 16,
        256: 12,
        512: 6,
    },
    use_itered_cycle: bool = True,
    _collect_stats: bool = False,
) -> tuple[list[StreamingDataset], StreamingDataLoader]:
    def _flatten_paths(d: dict[str, Any]) -> dict[str, list]:
        res: dict[str, list] = {}
        for k, v in d.items():
            if isinstance(v, dict):
                res.update(_flatten_paths(v))
            else:
                res[k] = v
        return res

    # Flatten all files
    paths_dict: dict[str, list]
    if isinstance(paths, dict):
        paths_dict = _flatten_paths(paths)
    else:
        # If it's already a flat list or other type, wrap it
        paths_dict = {"default": [paths, {}]}

    def _as_paths_list(ds_paths: Any) -> list[str]:
        if isinstance(ds_paths, str):
            return [ds_paths]
        if isinstance(ds_paths, list):
            if not all(isinstance(p, str) for p in ds_paths):
                raise TypeError(f"`ds_paths` must be list[str], got: {ds_paths}")
            return ds_paths
        raise TypeError(f"`ds_paths` must be str or list[str], got: {type(ds_paths)}")

    expanded_datasets: list[Any] = []
    expanded_weights: list[float] = []

    if weights is not None and len(weights) != len(paths_dict):
        raise ValueError(f"`weights` length must match number of datasets: {len(weights)} vs {len(paths_dict)}")

    for group_idx, (name, (ds_paths, ds_kwargs)) in enumerate(paths_dict.items()):
        ds_kwargs = set_defaults(ds_kwargs, stream_ds_kwargs)
        is_cycled = bool(ds_kwargs.pop("is_cycled", False))
        ds_paths_list = _as_paths_list(ds_paths)

        group_weight = 1.0 if weights is None else float(weights[group_idx])
        logger.debug(f"Dataset group: {name}, weight: {group_weight}")

        if use_itered_cycle:
            #  use itered cycle dataset
            group_ds = ImageStreamingDataset.create_dataset(
                input_dir=ds_paths_list,
                combined_kwargs={"batching_method": "per_stream"},
                is_cycled=is_cycled,
                **ds_kwargs,
            )

            expanded_datasets.append(group_ds)
            expanded_weights.append(group_weight)
            group_len = int(get_dataset_len(group_ds))
        else:
            # use flatten dataset and then combined dataset
            group_leaf: list[Any] = []
            for p in ds_paths_list:
                leaf = ImageStreamingDataset.create_dataset(
                    input_dir=p,
                    combined_kwargs={"batching_method": "per_stream"},
                    is_cycled=False,
                    **ds_kwargs,
                )
                group_leaf.append(CycledDataset(leaf) if is_cycled else leaf)

            expanded_datasets.extend(group_leaf)

            leaf_lens = [float(get_dataset_len(ds)) for ds in group_leaf]
            total_leaf_len = sum(leaf_lens)
            if total_leaf_len > 0:
                expanded_weights.extend([group_weight * (l / total_leaf_len) for l in leaf_lens])
            else:
                per_leaf_weight = group_weight / max(1, len(group_leaf))
                expanded_weights.extend([per_leaf_weight] * len(group_leaf))

            group_len = sum(get_dataset_len(ds) for ds in group_leaf)

        logger.info(f"Create dataset for {name} with paths: {ds_paths_list}")
        logger.info(f"Dataset has {group_len} samples.")
        logger.info("---------" * 6 + "\n")

    ds_total_len = sum(get_dataset_len(ds) for ds in expanded_datasets)
    logger.info(f"Total number of samples: {ds_total_len}")

    # logger.info(f"Expanded datasets: {expanded_datasets}")
    logger.info(f"Expanded weights: {expanded_weights}")
    ds_total = IndexedCombinedStreamingDataset(
        combined_is_cycled=True,
        datasets=expanded_datasets,
        weights=expanded_weights,
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
    if _collect_stats:
        from tqdm import tqdm

        logger.warning("Collecting statistics...")

        bands_info_n = {}
        for i, sample in tqdm(  # type: ignore
            enumerate(dataloader),
            # total=len(ds_total) // dl.batch_size,
        ):
            if i == 0 and "__key__" not in sample:
                print(sample.keys())

            chan = sample["img"].shape[1]
            if chan not in bands_info_n:
                bands_info_n[chan] = 0
            bands_info_n[chan] += sample["img"].shape[0]

            # logger.debug(f"Batch {i}: shape {sample['img'].shape=}")
            if i % 10 == 0:
                logger.info(
                    f"channel samples: {', '.join(f'{k}: {v}' for k, v in bands_info_n.items())}",
                    tqdm=True,
                )

    return expanded_datasets, dataloader


def get_fast_test_hyper_litdata_load(
    data_type: Literal[
        "DCF",
        "MMSeg",
        "Houston",
        "OHS",
        "WV3",
        "QB",
        "WV2",
        "WV4",
        "IKONOS",
        "RS5M",
        "BigEarthNetS2",
        "fmow_RGB",
        "fmow_MS",
        "SAM270k",
    ] = "RS5M",
    batch_size: int = 8,
    stream_ds_kwargs: dict[str, Any] | None = None,
    loader_kwargs: dict[str, Any] | None = None,
) -> tuple[list[ImageStreamingDataset], StreamingDataLoader]:
    """
    Build a quick litdata loader for model/module/function testing.

    The dataset paths are hard-coded here on purpose to avoid relying on yaml.
    """

    candidates: dict[str, dict[str, Any]] = {
        "DCF": {
            "paths": ["data2/DCF_2019/LitData_hyper_images"],
            "overrides": {"resize_before_transform": 128, "is_hwc": True},
        },
        "MMSeg": {
            "paths": ["data/MMSeg_YREB/LitData_hyper_images"],
            "overrides": {"resize_before_transform": 128, "is_hwc": True},
        },
        "Houston": {
            "paths": ["data/Houston/LitData_hyper_images"],
            "overrides": {"resize_before_transform": 512, "is_hwc": True},
        },
        "SAM270k": {
            "paths": ["data2/RemoteSAM270k/LitData_hyper_images"],
            "overrides": {"resize_before_transform": 512, "is_hwc": True},
        },
        "OHS": {
            "paths": ["data/OHS/LitData_hyper_images"],
            "overrides": {"resize_before_transform": 512, "is_hwc": True},
        },
        "WV3": {
            "paths": ["data/WorldView3/LitData_hyper_images"],
            "overrides": {"resize_before_transform": 512, "is_hwc": True},
        },
        "QB": {
            "paths": ["data/QuickBird/LitData_hyper_images"],
            "overrides": {"resize_before_transform": 512, "is_hwc": True},
        },
        "WV2": {
            "paths": ["data/WorldView2/LitData_hyper_images"],
            "overrides": {"resize_before_transform": 512, "is_hwc": True},
        },
        "WV4": {
            "paths": ["data/WorldView4/LitData_hyper_images"],
            "overrides": {"resize_before_transform": 512, "is_hwc": True},
        },
        "IKONOS": {
            "paths": ["data/IKONOS/LitData_hyper_images"],
            "overrides": {"resize_before_transform": 512, "is_hwc": True},
        },
        "RS5M": {
            "paths": ["data/RS5M/LitData_images_train"],
            "overrides": {"resize_before_transform": 512, "is_hwc": True, "force_to_rgb": True},
        },
        "BigEarthNetS2": {
            "paths": ["data2/BigEarthNet_S2/LitData_hyper_images"],
            "overrides": {"resize_before_transform": 128, "is_hwc": False},
        },
        "fmow_MS": {
            "paths": ["data2/Multispectral-FMow-full/LitData_hyper_images_8bands"],
            "overrides": {"resize_before_transform": 512, "is_hwc": True},
        },
        "fmow_RGB": {
            "paths": ["data/Fmow_rgb/LitData_hyper_images"],
            "overrides": {"resize_before_transform": 512, "is_hwc": True},
        },
    }

    if data_type not in candidates:
        raise ValueError(f"Unsupported data_type: {data_type}")

    candidate = candidates[data_type]
    dataset_path = candidate["paths"][0]

    final_stream_kwargs: dict[str, Any] = {
        "transform_prob": 0.0,
        "resize_before_transform": 256,
        "shuffle": False,
        "is_cycled": True,
        "is_hwc": True,
    }
    final_stream_kwargs.update(candidate["overrides"])
    if stream_ds_kwargs is not None:
        final_stream_kwargs.update(stream_ds_kwargs)

    final_loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": 2,
        "persistent_workers": False,
        "prefetch_factor": None,
        "shuffle": False,
    }
    if loader_kwargs is not None:
        final_loader_kwargs.update(loader_kwargs)

    dataset = ImageStreamingDataset.create_dataset(
        input_dir=dataset_path,
        combined_kwargs={"batching_method": "per_stream"},
        **final_stream_kwargs,
    )
    dataloader = StreamingDataLoader(dataset, **final_loader_kwargs)
    return [dataset], dataloader


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
        "collate_fn": collate_fn_skip_none(),
        "shuffle": False,
    }

    # RGB dataset
    stream_ds_kwargs_ = stream_ds_kwargs.copy()
    rgb_paths, kwargs = list(RGB_PATHS.values())[0]
    stream_ds_kwargs_.update(kwargs)  # ty: ignore[no-matching-overload]
    rgb_ds = ImageStreamingDataset.create_dataset(  # ty: ignore[invalid-argument-type]
        input_dir=rgb_paths,  # ty: ignore[invalid-argument-type]
        combined_kwargs={"batching_method": "stratified"},
        **stream_ds_kwargs_,  # ty: ignore[invalid-argument-type]
    )

    # Multispectral dataset
    multi_ds = []
    for name, (paths, kwargs) in MULTISPECTRAL_PATHS.items():
        stream_ds_kwargs_ = stream_ds_kwargs.copy()
        stream_ds_kwargs_.update(kwargs)  # ty: ignore[no-matching-overload]
        ds = ImageStreamingDataset.create_dataset(  # ty: ignore[invalid-argument-type]
            input_dir=paths,  # ty: ignore[invalid-argument-type]
            combined_kwargs={"batching_method": "stratified"},
            **stream_ds_kwargs_,  # ty: ignore[invalid-argument-type]
        )
        multi_ds.append(ds)

    # Hyperspectral dataset
    hyper_ds = []
    for name, (paths, kwargs) in HYPERSPECTRAL_PATHS.items():
        stream_ds_kwargs_ = stream_ds_kwargs.copy()
        stream_ds_kwargs_.update(kwargs)  # ty: ignore[no-matching-overload]
        ds = ImageStreamingDataset.create_dataset(  # ty: ignore[invalid-argument-type]
            input_dir=paths,  # ty: ignore[invalid-argument-type]
            combined_kwargs={"batching_method": "stratified"},
            **stream_ds_kwargs_,  # ty: ignore[invalid-argument-type]
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
    dl = SizeBasedBatchsizeStreamingDataloader(  # ty: ignore[invalid-argument-type]
        ds,
        size_based_batch_sizes={128: 16, 256: 12, 512: 6},
        **loader_kwargs,  # ty: ignore[invalid-argument-type]
    )

    return ds, dl


def __test_index_file_litdata_loader():
    index_file = "data/RS5M/LitData_images_val/val_index.json"
    input_dir = "data/RS5M/LitData_images_val/"

    ds = ImageStreamingDataset.create_dataset(
        input_dir=input_dir,
        combined_kwargs={"batching_method": "stratified"},
        index_path=index_file,
        transform_prob=0.0,
        resize_before_transform=256,
        is_cycled=False,
        index_file_name="val_index",
    )


def __test_normal_image_loader():
    from omegaconf import OmegaConf
    import hydra
    from litdata.streaming.serializers import BytesSerializer, StringSerializer

    # path = "data2/RemoteSAM270k/LitData_hyper_images2"
    # stream_ds_kwargs = {
    #     "transform_prob": 0.0,
    #     "resize_before_transform": 512,
    #     "is_cycled": False,
    # }
    # serializers = {
    #     "__key__": StringSerializer(),
    #     "img": BytesSerializer(),
    # }
    # ds = ImageStreamingDataset.create_dataset(
    #     input_dir=path,
    #     combined_kwargs={"batching_method": "per_stream"},
    #     # serializers=serializers,
    #     **stream_ds_kwargs,
    # )
    # loader_kwargs = {
    #     "batch_size": 4,
    #     "num_workers": 2,
    # }

    # dl = StreamingDataLoader(ds, **loader_kwargs)

    cfg = OmegaConf.load(
        "/home/user/zihancao/Project/hyperspectral-1d-tokenizer/scripts/configs/tokenizer_gan/dataset/litdata_one_loader.yaml"
    )
    ds, dl = hydra.utils.instantiate(cfg.train_loader)

    print("try to get one sample")

    from tqdm import tqdm

    _break_i = 1000
    c_d = {}
    for i, sample in enumerate(dl):
        # print(sample["img"].shape)
        b, c = sample["img"].shape[:2]

        c_d[c] = c_d.get(c, 0) + b
        print(f"{c_d}")
        if i >= _break_i:
            break

    print("test get state dict and load state dict")
    with logger.catch():
        sd = dl.state_dict()
        dl.load_state_dict(sd)
        print("can load state dict.")

    del ds, dl
    exit(0)


def __test_gen_loader():
    import lovely_tensors as lt

    lt.monkey_patch()

    img_path = "data2/RemoteSAM270k/LitData_hyper_images2"
    caption_path = "data2/RemoteSAM270k/LitData_image_captions"
    condition_path = "data2/RemoteSAM270k/LitData_image_conditions"

    ds = GenerativeStreamingDataset.create_dataset(img_path, condition_path, caption_path)
    from tqdm import trange

    # for i in trange(196100, 220000):
    #     sample = ds[i]
    #     if sample is None or any(v is None for v in sample.values()):
    #         print(f"[Warning]: found sample undecoded - for index {i} - sample is {sample}")

    dl = StreamingDataLoader(ds, batch_size=8, num_workers=4, shuffle=False)
    print(dl.state_dict())
    for sample in dl:
        # print(sample["img"].shape)
        if sample is None:
            print("Found None sample.")


def __test_get_item_key():
    path = "data2/RemoteSAM270k/LitData_hyper_images"
    ds = ImageStreamingDataset.create_dataset(
        input_dir=path,
        combined_kwargs={"batching_method": "per_stream"},
        # serializers=serializers,
        transform_prob=0.0,
        resize_before_transform=128,
        is_cycled=False,
    )

    for i in range(len(ds)):
        print(ds[i]["__key__"])


def __test_ds_len():
    path = "data2/RemoteSAM270k/LitData_image_conditions"

    ds = ConditionsStreamingDataset.create_dataset(
        input_dir=path,
        combined_kwargs={"batching_method": "per_stream"},
        # serializers=serializers,
        # stream_ds_kwargs={"transform_prob": 0.0, "resize_before_transform": 128, "is_cycled": False},
    )
    print(f"Dataset length: {len(ds)}")
    print(ds[0])


def __test_get_mars_data():
    from tqdm import tqdm

    mars_data_dir: list[str] = [
        "data2/MarsBench/mb-boulder_det/",
        "data2/MarsBench/mb-boulder_seg/",
        "data2/MarsBench/mb-conequest_seg/",
        "data2/MarsBench/mb-crater_multi_seg/",
        "data2/MarsBench/mb-domars16k/",
        "data2/MarsBench/mb-dust_devil_det/",
        "data2/MarsBench/mb-landmark_cls/",
        "data2/MarsBench/mb-mars_seg_mer/",
        # "data2/MarsBench/mb-mmls/",
        "data2/MarsBench/mb-surface_cls/",
        "data2/MarsBench/mb-surface_multi_label_cls/",
    ]
    # add suffix: litdata
    mars_data_dir = [f + "litdata" for f in mars_data_dir]
    print(mars_data_dir)
    # path = mars_data_dir[0]
    # ds, dl= ImageStreamingDataset.create_dataloader(input_dir=path, loader_kwargs={'batch_size': 3, 'num_workers': 1})

    # Get the dataloader
    # paths should be a dict for _flatten_paths, but if it's a list, we wrap it
    if isinstance(mars_data_dir, list):
        mars_paths = {"mars_data": (mars_data_dir, {"is_cycled": False})}
    else:
        mars_paths = mars_data_dir

    ds, dl = create_hyper_image_litdata_flatten_paths_loader(
        mars_paths, loader_kwargs={"batch_size": 8, "num_workers": 4}
    )
    total_ds = sum(get_dataset_len(ds) for ds in ds)
    print(f"Total data: {total_ds}")
    print(f"Total iters: {total_ds // 8}")
    with logger.catch():
        for sample in tqdm(dl, total=total_ds // 8):
            # print(sample.keys())
            pass


if __name__ == "__main__":
    """
    python -m src.data.litdata_hyperloader
    """

    # create_hyper_image_litdata_loader()
    # create_hyper_image_litdata_flatten_paths_loader()
    # test_index_file_litdata_loader()
    __test_normal_image_loader()
    # __test_gen_loader()
    # __test_ds_len()
    # __test_get_item_key()
    # __test_get_mars_data()

    # from omegaconf import OmegaConf

    # cfg = OmegaConf.load(
    #     "scripts/configs/tokenizer_gan/dataset/litdata_one_loader.yaml"
    #     # "scripts/configs/tokenizer_gan/dataset/litdata_hyperspectral.yaml"
    # )

    # lcfg = cfg.val_loader
    # # logger.info(lcfg.paths)
    # ds, dl = create_hyper_image_litdata_flatten_paths_loader(
    #     paths=lcfg.paths,
    #     weights=lcfg.weights,
    #     loader_kwargs=lcfg.loader_kwargs,
    #     stream_ds_kwargs={
    #         "transform_prob": 0.0,
    #         "resize_before_transform": 256,
    #         "shuffle": True,
    #         "is_cycled": True,
    #         "is_hwc": True,
    #     },
    # )

    # for sample in dl:
    #     print(sample["img"].shape)
