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

import os
import time

from mmengine.registry import Registry, build_from_cfg
from termcolor import colored
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from src.stage2.generative.Sana.diffusion.data.transforms import get_transform
from src.stage2.generative.Sana.diffusion.utils.logger import get_root_logger

DATASETS = Registry("datasets")

DATA_ROOT = "data"


def set_data_root(data_root):
    global DATA_ROOT
    DATA_ROOT = data_root


def get_data_path(data_dir):
    if os.path.isabs(data_dir):
        return data_dir
    global DATA_ROOT
    return os.path.join(DATA_ROOT, data_dir)


def get_data_root_and_path(data_dir):
    if os.path.isabs(data_dir):
        return data_dir
    global DATA_ROOT
    return DATA_ROOT, os.path.join(DATA_ROOT, data_dir)


def collate_fn_sana():
    """
    Sana专用的collate函数,处理元组格式的样本并过滤None值。

    Sana dataset返回格式:
    (img, caption, attention_mask, data_info, idx, caption_type, dataindex_info, clipscore)
    """
    def inner(batch: list):
        # 过滤None样本
        orig_len = len(batch)
        batch = [sample for sample in batch if sample is not None]

        if len(batch) == 0:
            return None
        elif len(batch) < orig_len:
            logger = get_root_logger()
            logger.warning(
                f"Skipped {orig_len - len(batch)} None samples in batch. "
                f"Remaining: {len(batch)}/{orig_len}"
            )

        # 使用default_collate处理元组格式
        return default_collate(batch)

    return inner


def build_dataset(cfg, resolution=224, **kwargs):
    logger = get_root_logger()

    dataset_type = cfg.get("type")
    from src.stage2.generative.Sana.diffusion.data import datasets as _datasets  # noqa: F401

    if dataset_type == "SanaLitdataGenerativeControlDataset":
        from src.stage2.generative.Sana.diffusion.data.datasets import litdata_sana_control as _litdata  # noqa: F401
    logger.info(f"Constructing dataset {dataset_type}...")
    t = time.time()
    transform = cfg.pop("transform", "default_train")
    transform = get_transform(transform, resolution)
    dataset = build_from_cfg(
        cfg,
        DATASETS,
        default_args=dict(transform=transform, resolution=resolution, **kwargs),
    )
    logger.info(
        f"{colored(f'Dataset {dataset_type} constructed: ', 'green', attrs=['bold'])}"
        f"time: {(time.time() - t):.2f} s, length (use/ori): {len(dataset)}/{dataset.ori_imgs_nums}"
    )
    return dataset


def build_dataloader(dataset, batch_size=256, num_workers=4, shuffle=True, **kwargs):
    if "collate_fn" not in kwargs:
        kwargs["collate_fn"] = collate_fn_sana()

    if "batch_sampler" in kwargs:
        dataloader = DataLoader(
            dataset,
            batch_sampler=kwargs["batch_sampler"],
            collate_fn=kwargs.get("collate_fn"),
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            **kwargs,
        )
    return dataloader
