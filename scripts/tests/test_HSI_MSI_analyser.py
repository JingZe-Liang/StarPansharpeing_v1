import numpy as np
import torch

from src.stage2.pansharpening.metrics.metric_pansharpening import AnalysisPanAcc


# mat file reader
def read_any_mat(file_path):
    import scipy.io

    mat = scipy.io.loadmat(file_path)
    d = {}
    for k, v in mat.items():
        if isinstance(v, np.ndarray):
            d[k] = torch.as_tensor(v, dtype=torch.float32).permute(2, 0, 1)[None]
    return d


analysis = AnalysisPanAcc(ref=False, ratio=4, default_max_value=2047)
d = read_any_mat("matlab.mat")
for k, v in d.items():
    print(k, f"min={v.min().item()}, max={v.max().item()}")

# c = d['HSI'].shape[1]
# for ci in range(c):
#     hsi_i = d['HSI'][0, ci].mean()
#     fused_i = d['Z_CNMF'][0, ci].mean()
#     print(f'Band {ci}: HSI={hsi_i.item():.4f}, Fused={fused_i.item():.4f}')

const = 2047.0
analysis(
    d["Z_CNMF"],
    d["HSI"],
    torch.nn.functional.interpolate(
        d["HSI"], scale_factor=4, mode="bilinear", align_corners=False
    ),
    d["MSI"],
)

# HSI 0 - 2**n
# HSI / 2**n

# 0-1
# 1 * const (2**n)


print(analysis.print_str())
