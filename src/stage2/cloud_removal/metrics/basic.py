# Match the baseline metric
from __future__ import annotations

from typing import Any

import lpips
import numpy as np
import torch
import torchmetrics
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


class CRMetrics(torchmetrics.Metric):
    def __init__(self, *, lpips_device: torch.device | str | None = None, interp_to: int | None = None) -> None:
        super().__init__()
        self.add_state("psnr_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("ssim_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("lpips_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("rmse_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self._lpips_fn = lpips.LPIPS(net="alex", version="0.1")
        self._lpips_device: torch.device | None = None
        if lpips_device is not None:
            self._lpips_device = torch.device(lpips_device)
            self._lpips_fn.to(self._lpips_device)
        self._lpips_on_device = self._lpips_device is not None
        self.interp_to = interp_to
        if interp_to not in {None, 256}:
            raise ValueError(f"interp_to must be None or 256 to match baseline, got {interp_to=}")

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        if preds.ndim != 4 or targets.ndim != 4:
            raise ValueError(f"Expected BCHW tensors, got {preds.shape=} and {targets.shape=}")
        if preds.shape != targets.shape:
            raise ValueError(f"Shape mismatch: {preds.shape=} vs {targets.shape=}")
        batch = preds.shape[0]
        for i in range(batch):
            pred = preds[i]
            target = targets[i]
            metrics = self._img_metrics(target=target, pred=pred)
            self.psnr_sum += torch.tensor(metrics["PSNR"], device=self.psnr_sum.device)
            self.ssim_sum += torch.tensor(metrics["SSIM"], device=self.ssim_sum.device)
            self.lpips_sum += torch.tensor(metrics["LPIPS"], device=self.lpips_sum.device)
            self.rmse_sum += torch.tensor(metrics["RMSE"], device=self.rmse_sum.device)
            self.count += 1

    def compute(self) -> dict[str, torch.Tensor]:
        if self.count.item() == 0:
            return {
                "PSNR": torch.tensor(float("nan")),
                "SSIM": torch.tensor(float("nan")),
                "LPIPS": torch.tensor(float("nan")),
                "RMSE": torch.tensor(float("nan")),
            }
        denom = self.count.to(self.psnr_sum.dtype)
        return {
            "PSNR": self.psnr_sum / denom,
            "SSIM": self.ssim_sum / denom,
            "LPIPS": self.lpips_sum / denom,
            "RMSE": self.rmse_sum / denom,
        }

    def _img_metrics(self, *, target: torch.Tensor, pred: torch.Tensor) -> dict[str, float]:
        """Match the image metric from baseline that use [0, 255] value range"""
        # Interpolate to target size if specified (to match EMRDM baseline with 256x256)
        if self.interp_to is not None:
            target = torch.nn.functional.interpolate(
                target.unsqueeze(0),
                size=(self.interp_to, self.interp_to),
                mode="bicubic",
                align_corners=False,
            ).squeeze(0)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(0),
                size=(self.interp_to, self.interp_to),
                mode="bicubic",
                align_corners=False,
            ).squeeze(0)

        rmse = torch.sqrt(torch.mean((target - pred) ** 2)).item()
        img_a = (pred * 255.0).clamp(0, 255).to(torch.uint8)
        img_b = (target * 255.0).clamp(0, 255).to(torch.uint8)
        psnr_val = self._psnr(img_a, img_b)
        ssim_val, lpips_val = self._ssim_lpips(img_a, img_b)
        return {"PSNR": psnr_val, "SSIM": ssim_val, "LPIPS": lpips_val, "RMSE": rmse}

    def _psnr(self, img_a: torch.Tensor, img_b: torch.Tensor) -> float:
        a = img_a.cpu().numpy().transpose(1, 2, 0)
        b = img_b.cpu().numpy().transpose(1, 2, 0)
        return float(psnr(a, b, data_range=255))

    def _ssim_lpips(self, img_a: torch.Tensor, img_b: torch.Tensor) -> tuple[float, float]:
        if img_a.shape[0] == 4:
            ssim_total = 0.0
            lpips_total = 0.0
            for i in range(img_a.shape[0]):
                a = img_a[i].expand(3, img_a.shape[1], img_a.shape[2])
                b = img_b[i].expand(3, img_b.shape[1], img_b.shape[2])
                ssim_total += self._ssim(a, b)
                lpips_total += self._lpips(a, b)
            n = float(img_a.shape[0])
            return ssim_total / n, lpips_total / n
        return self._ssim(img_a, img_b), self._lpips(img_a, img_b)

    def _ssim(self, img_a: torch.Tensor, img_b: torch.Tensor) -> float:
        # logic from baseline code
        a = np.tensordot(img_a.cpu().numpy().transpose(1, 2, 0), [0.298912, 0.586611, 0.114478], axes=1)
        b = np.tensordot(img_b.cpu().numpy().transpose(1, 2, 0), [0.298912, 0.586611, 0.114478], axes=1)
        return float(ssim(a, b, data_range=255))

    def _lpips(self, img_a: torch.Tensor, img_b: torch.Tensor) -> float:
        im1 = img_a.float().unsqueeze(0)
        im2 = img_b.float().unsqueeze(0)
        if self._lpips_device != im1.device:
            self._lpips_fn.to(im1.device)
            self._lpips_device = im1.device
            self._lpips_on_device = True
        return float(self._lpips_fn(im1, im2).item())
