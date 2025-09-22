from pathlib import Path

import numpy as np
import PIL
import torch

from src.stage2.pansharpening.data.pansharpening_loader import (
    get_pansharp_wds_dataloader,
)
from src.utilities.train_utils.visualization import visualize_batch_comparisons_imgs

# IKONOS: 1193.00
# QB: 634.00
# WV3: 1293.00
# WV2: 1208.00
# WV4: 1516.00


def main(tar_file, save_dir="tmp/pansharpening_ds_show"):
    _, dataloader = get_pansharp_wds_dataloader(
        tar_file,
        1,
        shuffle_size=0,
        resample=False,
        num_workers=0,
        to_neg_1_1=False,
        norm=2047.0,
    )
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    for i, batch in enumerate(dataloader):
        lr, hr, pan = batch["lrms"], batch["hrms"], batch["pan"]
        print(lr.shape, hr.shape, pan.shape)
        # save
        grid = visualize_batch_comparisons_imgs(
            lr,
            pan,
            hr,
            rgb_channels=[2, 1, 0],
            norm=False,
            to_grid=True,
            to_uint8=True,
            use_linstretch=True,
        )
        PIL.Image.fromarray(grid).save(f"{save_dir}/sample_{i}.png")
        if i >= 10:
            break


if __name__ == "__main__":
    tar_file = "data/Downstreams/PanCollectionV2/IKONOS/pansharpening_reduced/Pansharpening_IKONOS.tar"
    main(tar_file)
