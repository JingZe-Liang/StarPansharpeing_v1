from typing import Literal, TypedDict

import numpy as np
import torch
from beartype import beartype
from torch import Tensor
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    SpectralAngleMapper,
    StructuralSimilarityIndexMeasure,
)
from torchmetrics.image.sam import SpectralAngleMapper
from torchmetrics.regression import MeanSquaredError

DenoisingMetricsOutput = TypedDict(
    "DenoisingMetricsOutput",
    {
        "psnr": float,
        "ssim": float | Tensor,
        "sam": float,
        "rmse": float,
    },
)


@beartype
class DenoisingMetrics(torch.nn.Module):
    def __init__(
        self,
        data_range: float = 1.0,
        sigma: float = 1.5,
        kernel_size: int = 11,
        reduction: Literal[
            "elementwise_mean", "sum", "none", None
        ] = "elementwise_mean",
    ):
        super().__init__()
        self.psnr = PeakSignalNoiseRatio(data_range=data_range, reduction=reduction)
        self.ssim = StructuralSimilarityIndexMeasure(
            data_range=data_range,
            sigma=sigma,
            kernel_size=kernel_size,
            reduction=reduction,
        )
        self.mse = MeanSquaredError()
        self.sam = SpectralAngleMapper(reduction=reduction)

    def update(self, denoised: Tensor, clean: Tensor):
        self.psnr.update(denoised, clean)
        self.ssim.update(denoised, clean)
        self.mse.update(denoised, clean)
        self.sam.update(denoised, clean)

    def forward(self, denoised: Tensor, clean: Tensor):
        self.update(denoised, clean)
        return self.compute()

    def _may_to_float(self, x: Tensor) -> float | Tensor:
        if x.numel() == 1:
            return x.item()
        return x

    def compute(self) -> DenoisingMetricsOutput:
        mse = self.mse.compute()
        rmse = torch.sqrt(mse)

        psnr = self.psnr.compute()
        ssim = self.ssim.compute()  # type: ignore[arg-type]
        sam = self.sam.compute()

        psnr, ssim, rmse, sam = map(self._may_to_float, [psnr, ssim, rmse, sam])
        return dict(psnr=psnr, ssim=ssim, rmse=rmse, sam=sam)

    def reset(self):
        self.psnr.reset()
        self.ssim.reset()
        self.mse.reset()
        self.sam.reset()


# * --- Test --- #


def test_metrics():
    import torch

    gen = torch.manual_seed(42)
    preds = torch.rand([16, 3, 16, 16], generator=gen)
    target = torch.rand([16, 3, 16, 16], generator=gen)

    metrics = DenoisingMetrics()
    results = metrics(preds, target)
    print(results)
    assert isinstance(results, dict)
    assert all(k in results for k in ["psnr", "ssim", "sam", "rmse"])
    assert all(isinstance(v, float) for v in results.values())


if __name__ == "__main__":
    test_metrics()
