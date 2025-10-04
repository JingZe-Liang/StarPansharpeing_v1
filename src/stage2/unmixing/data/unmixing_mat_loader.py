"""
Data infos:
apex - GT: (4, 285)
apex - S_GT: (110, 110, 4)
apex - Y: (285, 12100)
apex - cols: (1, 1)
apex - lines: (1, 1)
apex - vca: (285, 4)
-------------------
DC - GT: (5, 191)
DC - S_GT: (290, 290, 5)
DC - Y: (191, 84100)
DC - cols: (1, 1)
DC - lines: (1, 1)
-------------------
houston - GT: (4, 144)
houston - S_GT: (105, 105, 4)
houston - Y: (144, 11025)
houston - cols: (1, 1)
houston - lines: (1, 1)
-------------------
houston_170_dataset - DSM: (170, 170)
houston_170_dataset - M: (144, 4)
houston_170_dataset - M1: (144, 4)
houston_170_dataset - MPN: (170, 170, 5)
houston_170_dataset - Y: (170, 170, 144)
houston_170_dataset - label: (170, 170, 4)
-------------------
Jasper - A: (4, 10000)
Jasper - GT: (4, 198)
Jasper - S_GT: (100, 100, 4)
Jasper - Y: (198, 10000)
Jasper - cols: (1, 1)
Jasper - lines: (1, 1)
-------------------
moffett - GT: (3, 184)
moffett - S_GT: (50, 50, 3)
moffett - Y: (184, 2500)
moffett - cols: (1, 1)
moffett - lines: (1, 1)
-------------------
Samson - GT: (3, 156)
Samson - S_GT: (95, 95, 3)
Samson - Y: (156, 9025)
Samson - cols: (1, 1)
Samson - lines: (1, 1)
-------------------
Urban4 - GT: (4, 162)
Urban4 - S_GT: (307, 307, 4)
Urban4 - Y: (162, 94249)
Urban4 - cols: (1, 1)
Urban4 - lines: (1, 1)
-------------------
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from loguru import logger
from scipy.io import loadmat
from skimage.metrics import peak_signal_noise_ratio
from torch.utils.data import DataLoader, Dataset

from ..traditional.pipe import vca_fclsu_nnls_solver

KEYS_MAPPING = {
    "GT": "endmembers",
    "S_GT": "abunds",
    "Y": "img",
    "lines": "H",
    "cols": "W",
    "vca": "init_endmembers",  # vca traidtional method
    "A": "abunds",
    "DSM": "DSM",
    "M": "endmembers",
    "M1": "init_endmembers",
    "MPN": "lidar",
    "label": "abunds",
}


class UnmixingMatDataset(Dataset):
    def __init__(self, dataset_dir: str, dataset_name: str):
        self.ds_dir = dataset_dir
        self.ds_name = (
            dataset_name if dataset_name.endswith(".mat") else dataset_name + ".mat"
        )

        self.mat_d = self._load_mat_dict(Path(self.ds_dir) / self.ds_name)

        # pipeline init
        logger.debug(
            f"Image {self.ds_name} loaded with shape {self.mat_d['img'].shape}."
        )
        self.n_endmembers = self.mat_d["endmembers"].shape[0]
        edm, abunds = self._vca_solve(self.mat_d["img"], self.n_endmembers)
        self.mat_d["init_vca_endmembers"] = edm
        self.mat_d["init_vca_abunds"] = abunds

    @staticmethod
    def _vca_solve(img: np.ndarray, n_endmembers: int, algo="sivm"):
        # algo: vca, vca_custom, sisal, sivm
        endmembers, abunds = vca_fclsu_nnls_solver(img, n_endmembers, algo=algo)
        logger.info(f"VCA+FCLSU done, extract {endmembers.shape[1]} endmembers.")
        if endmembers.shape[0] > endmembers.shape[1]:
            endmembers = endmembers.T  # (n_endmember, channels)

        # check for the recon quality
        # [em, c] @ [em, h, w] = [c, h, w]
        recon = endmembers.T @ abunds.cpu().numpy().reshape(n_endmembers, -1)
        recon = recon.reshape(img.shape)
        psnr = peak_signal_noise_ratio(img, recon, data_range=1.0)
        mse_error = np.mean((img - recon) ** 2)
        logger.info(
            f"VCA+FCLSU reconstruction PSNR: {psnr:.2f}dB, MSE: {mse_error:.5f}"
        )

        return endmembers, abunds

    @staticmethod
    def _remap_keys(d: Dict[str, Any]) -> Dict[str, Any]:
        new_d = {}
        for k, v in d.items():
            if k in KEYS_MAPPING:
                new_d[KEYS_MAPPING[k]] = v
            elif k not in ["__header__", "__version__", "__globals__"]:
                new_d[k] = v
        return new_d

    @staticmethod
    def _reshape_img(d: Dict):
        for k in list(d.keys()):
            v = d[k]
            if k == "img":
                if v.ndim == 2:  # [c, h*w]
                    d[k] = v.reshape(-1, d["H"], d["W"])
                elif v.ndim == 3:  # [h, w, c]
                    d[k] = v.transpose(-1, 0, 1)
                    d["H"], d["W"] = v.shape[:-1]
                else:
                    raise ValueError(f"Unsupported shape {v.shape}.")
            elif k == "abunds":
                d[k] = v.transpose(-1, 0, 1)

        return d

    @staticmethod
    def _take_shape_item(d: Dict):
        for k, v in d.items():
            if k in ["H", "W"]:
                d[k] = v.item()
        return d

    def _load_mat_dict(self, file_path: str | Path):
        d = loadmat(file_path)
        d = self._remap_keys(d)
        d = self._take_shape_item(d)
        d = self._reshape_img(d)
        return d

    def __len__(self):
        return 1

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.mat_d


def get_unmixing_mat_loader(
    ds_dir: str,
    ds_name: str,
    batch_size: int,
):
    ds = UnmixingMatDataset(ds_dir, ds_name)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return ds, dl


if __name__ == "__main__":
    """
    python -m src.stage2.unmixing.data.unmixing_mat_loader
    """

    ds_names = [
        "apex",
        "DC",
        "houston",
        # "houston_170_dataset",
        "Jasper",
        "moffett",
        "Samson",
        "Urban4",
    ]

    for dn in ds_names:
        ds = UnmixingMatDataset(
            "src/stage2/unmixing/traditional/Hyperspectral-Unmixing-Models/Datasets",
            dn,
        )

        # print(ds[0].keys())
        for k, v in ds[0].items():
            print(k, v.shape if hasattr(v, "shape") else v)
        print("-" * 60)
