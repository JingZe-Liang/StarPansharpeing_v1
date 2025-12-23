# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math
import random

import numpy as np
import torch


def generate_ibot_masks(
    mask_generator,
    batch_size,
    n_tokens,
    mask_probability=0.5,
    mask_ratio_min_max=(0.1, 0.5),
    device=None,
):
    n_samples_masked = int(batch_size * mask_probability)
    probs = torch.linspace(*mask_ratio_min_max, n_samples_masked + 1)
    masks_list = []

    for i in range(0, n_samples_masked):
        prob_max = probs[i + 1]
        mask = torch.BoolTensor(mask_generator(int(n_tokens * prob_max)))
        masks_list.append(mask)

    for _ in range(n_samples_masked, batch_size):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)

    masks = torch.stack(masks_list).flatten(1)
    if device is not None:
        masks = masks.to(device)

    mask_indices = masks.flatten().nonzero().flatten()

    masks_weight = (1 / masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(masks)[masks]

    n_masked_patches_tensor = torch.full(
        (1,),
        fill_value=mask_indices.shape[0],
        dtype=torch.long,
        device=device if device is not None else masks.device,
    )

    return masks, mask_indices, masks_weight, n_masked_patches_tensor


class MaskingGenerator:
    def __init__(
        self,
        input_size,
        num_masking_patches=None,
        min_num_patches=4,
        max_num_patches=None,
        min_aspect=0.3,
        max_aspect=None,
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height,
            self.width,
            self.min_num_patches,
            self.max_num_patches,
            self.num_masking_patches,
            self.log_aspect_ratio[0],
            self.log_aspect_ratio[1],
        )
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for _ in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top : top + h, left : left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self, num_masking_patches=0):
        mask = np.zeros(shape=self.get_shape(), dtype=bool)
        mask_count = 0
        while mask_count < num_masking_patches:
            max_mask_patches = num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return self.complete_mask_randomly(mask, num_masking_patches)

    def complete_mask_randomly(self, mask, num_masking_patches):
        shape = mask.shape
        m2 = mask.flatten()
        to_add = np.random.choice(np.where(~m2)[0], size=num_masking_patches - m2.sum(), replace=False)
        m2[to_add] = True
        return m2.reshape(shape)
