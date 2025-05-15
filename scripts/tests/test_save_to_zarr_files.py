from pathlib import Path
from safetensors.torch import load_file
import numpy as np
import sys
from tqdm import tqdm

from src.utilities.saver.zarr_saver import zarr_saver


sft_files = Path("data/MUSLI_safetensors/safetensors").glob("*.safetensors")


def arr_generator(files):
    for f in tqdm(files):
        img = np.array(load_file(f)["img"])
        yield f.stem, img


if __name__ == "__main__":
    zarr_saver(arr_generator(sft_files), "./test.zarr")
