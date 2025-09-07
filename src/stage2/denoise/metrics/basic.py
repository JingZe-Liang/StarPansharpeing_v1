import numpy as np
import torch
from beartype import beartype
from torch import Tensor
from torchmetrics.aggregation import MeanMetric
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    SpectralAngleMapper,
    StructuralSimilarityIndexMeasure,
)
from torchmetrics.image.sam import SpectralAngleMapper
from torchmetrics.regression import MeanSquaredError


@beartype
class DenosingMetrics:
    def __init__(
        self,
        data_range: float = 1.0,
        sigma: float = 1.5,
        kernel_size: int = 11,
        reduction="elementwise_mean",
    ):
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

    def __call__(self, denoised: Tensor, clean: Tensor):
        self.update(denoised, clean)
        return self.compute()

    def compute(self):
        mse = self.mse.compute()
        rmse = torch.sqrt(mse)

        return {
            "psnr": self.psnr.compute().item(),
            "ssim": self.ssim.compute().item(),
            "sam": self.sam.compute().item(),
            "rmse": rmse.item(),
        }

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

    metrics = DenosingMetrics()
    results = metrics(preds, target)
    print(results)
    assert isinstance(results, dict)
    assert all(k in results for k in ["psnr", "ssim", "sam", "rmse"])
    assert all(isinstance(v, float) for v in results.values())


if __name__ == "__main__":
    test_metrics()
