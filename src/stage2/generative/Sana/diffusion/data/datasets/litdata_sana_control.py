# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import torch

from src.stage2.generative.Sana.diffusion.data.builder import DATASETS


ControlSignalType = Literal["hed", "segmentation", "sketch", "mlsd", "random"]


@dataclass(frozen=True)
class _LitdataControlPaths:
    img_dir: str
    condition_dir: str
    caption_dir: str


def _parse_control_paths(data_dir: Any) -> _LitdataControlPaths:
    if not isinstance(data_dir, list) or len(data_dir) != 3:
        raise ValueError(
            "For litdata controlnet, `data_dir` must be a list of 3 paths: [img_dir, condition_dir, caption_dir]."
        )
    img_dir, condition_dir, caption_dir = data_dir
    if not isinstance(img_dir, str) or not isinstance(condition_dir, str) or not isinstance(caption_dir, str):
        raise ValueError("For litdata controlnet, `data_dir` must contain 3 strings.")
    return _LitdataControlPaths(img_dir=img_dir, condition_dir=condition_dir, caption_dir=caption_dir)


def _normalize_control_signal_type(control_signal_type: str) -> ControlSignalType:
    if control_signal_type == "scribble":
        control_signal_type = "sketch"
    if control_signal_type in {"hed", "segmentation", "sketch", "mlsd", "random"}:
        return cast(ControlSignalType, control_signal_type)
    raise ValueError(
        f"Unsupported control_signal_type={control_signal_type!r}. Expected one of: hed/segmentation/sketch/mlsd/scribble/random."
    )


def _maybe_get_extra_dict(extra: Any) -> dict[str, Any]:
    return extra if isinstance(extra, dict) else {}


@DATASETS.register_module()
class SanaLitdataGenerativeControlDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: list[str],
        *,
        resolution: int = 512,
        max_length: int = 300,
        config: Any | None = None,
        extra: Any = None,
        **_: Any,
    ) -> None:
        paths = _parse_control_paths(data_dir)
        extra_dict = _maybe_get_extra_dict(extra)

        img_kwargs = extra_dict.get("img_kwargs", {})
        cond_kwargs = extra_dict.get("cond_kwargs", {})
        caption_kwargs = extra_dict.get("caption_kwargs", {})
        gen_kwargs = extra_dict.get("gen_kwargs", {})

        if (
            not isinstance(img_kwargs, dict)
            or not isinstance(cond_kwargs, dict)
            or not isinstance(caption_kwargs, dict)
        ):
            raise ValueError("`extra.img_kwargs/cond_kwargs/caption_kwargs` must be dict if provided.")
        if not isinstance(gen_kwargs, dict):
            raise ValueError("`extra.gen_kwargs` must be dict if provided.")

        img_kwargs = {"shuffle": False, "is_cycled": False, **img_kwargs}
        cond_kwargs = {"shuffle": False, "is_cycled": False, **cond_kwargs}
        caption_kwargs = {"shuffle": False, "is_cycled": False, **caption_kwargs}
        gen_kwargs = {"resize": resolution, **gen_kwargs}

        from src.data.litdata_hyperloader import GenerativeStreamingDataset

        self._dataset = GenerativeStreamingDataset.create_dataset(
            img_input_dir=paths.img_dir,
            condition_input_dir=paths.condition_dir,
            caption_input_dir=paths.caption_dir,
            img_kwargs=img_kwargs,
            cond_kwargs=cond_kwargs,
            caption_kwargs=caption_kwargs,
            gen_kwargs=gen_kwargs,
        )

        self.ori_imgs_nums = len(self._dataset)
        self.max_length = max_length
        self.resolution = resolution
        self._caption_type = "prompt"
        self._clipscore = "0.0"

        control_signal_type = "random"
        control_signal_types: list[str] = []
        if config is not None and getattr(config, "controlnet", None) is not None:
            control_signal_type = getattr(config.controlnet, "control_signal_type", "random")
            control_signal_types_raw = getattr(config.controlnet, "control_signal_types", [])
            if isinstance(control_signal_types_raw, list):
                control_signal_types = [str(v) for v in control_signal_types_raw]
        self.control_signal_type: ControlSignalType = _normalize_control_signal_type(str(control_signal_type))
        self.control_signal_types = control_signal_types

    def __len__(self) -> int:
        return len(self._dataset)

    def _select_control_image(self, sample: dict[str, Any]) -> tuple[torch.Tensor, str]:
        if self.control_signal_type == "random":
            keys = ["hed", "segmentation", "sketch", "mlsd"]
            key = keys[int(torch.randint(0, len(keys), ()).item())]
            return sample[key], key
        key = str(self.control_signal_type)
        return sample[key], key

    def _select_control_images(self, sample: dict[str, Any]) -> tuple[list[torch.Tensor], list[str]]:
        if self.control_signal_types:
            keys = [str(k) for k in self.control_signal_types]
            keys = [("sketch" if k == "scribble" else k) for k in keys]
        elif self.control_signal_type == "random":
            keys = ["hed", "segmentation", "sketch", "mlsd"]
        else:
            keys = [str(self.control_signal_type)]

        images: list[torch.Tensor] = []
        for key in keys:
            if key not in sample:
                raise ValueError(f"Missing control key {key!r} in sample.")
            images.append(sample[key])
        return images, keys

    def __getitem__(self, idx: int):
        sample = self._dataset[idx]
        if not isinstance(sample, dict):
            raise ValueError("Litdata GenerativeStreamingDataset must return a dict sample.")

        img = sample["img"]
        caption = sample["caption"]
        if not torch.is_tensor(img):
            raise ValueError("Expected `img` to be a torch.Tensor.")
        if not isinstance(caption, str):
            raise ValueError("Expected `caption` to be a string.")

        control_images, control_types = self._select_control_images(sample)
        for control_image in control_images:
            if not torch.is_tensor(control_image):
                raise ValueError("Expected control image to be a torch.Tensor.")

        attention_mask = torch.ones(1, 1, self.max_length, dtype=torch.int16)
        data_info = {
            "img_hw": torch.tensor([float(self.resolution), float(self.resolution)], dtype=torch.float32),
            "aspect_ratio": torch.tensor(1.0, dtype=torch.float32),
            "control_image": control_images,
            "control_types": control_types,
        }
        dataindex_info = {"index": idx, "shard": "litdata", "shardindex": idx, "key": sample.get("__key__", "")}

        return (
            img,
            caption,
            attention_mask,
            data_info,
            idx,
            self._caption_type,
            dataindex_info,
            self._clipscore,
        )
