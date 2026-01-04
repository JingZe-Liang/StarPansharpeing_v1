import torch
from torch import Tensor
from torchmetrics import Metric


def compute_epe(
    est: Tensor,
    gt: Tensor,
    min_disp: float,
    max_disp: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Compute End-Point-Error (EPE).
    Functional impl for stateless usage.

    Args:
        est: Predicted disparity map [B, H, W] or [B, 1, H, W]
        gt: Ground truth disparity map [B, H, W] or [B, 1, H, W]
        min_disp: Minimum disparity value (used for mask generation)
        max_disp: Maximum disparity value (used for mask generation)

    Returns:
        tuple: (total_error, num_valid_pixels, average_epe)
            - total_error: Sum of absolute errors for all valid pixels
            - num_valid_pixels: Number of valid pixels
            - average_epe: Average EPE
    """
    # Ensure input is tensor
    if not isinstance(est, Tensor):
        est = torch.tensor(est)
    if not isinstance(gt, Tensor):
        gt = torch.tensor(gt)

    # Unify shape [B, H, W]
    if est.dim() == 4:
        est = est.squeeze(1)
    if gt.dim() == 4:
        gt = gt.squeeze(1)

    # Generate mask
    # Corresponding reference: mask1 = gt >= max_disp -> 0; mask2 = gt < min_disp -> 0
    # So valid range is [min_disp, max_disp)
    mask = (gt >= min_disp) & (gt < max_disp)

    # Compute error
    error_map = torch.abs(est - gt)
    masked_error = error_map[mask]

    # Convert to float for accumulation
    total_error = masked_error.sum()
    num_valid_pixels = mask.sum()

    if num_valid_pixels > 0:
        average_epe = total_error / num_valid_pixels
    else:
        average_epe = torch.tensor(0.0, device=est.device)

    return total_error, num_valid_pixels, average_epe


def compute_d1(
    est: Tensor,
    gt: Tensor,
    min_disp: float,
    max_disp: float,
    threshold: float = 3.0,
    relative_threshold: float = 0.05,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Compute D1 Error (Percentage of bad pixels).
    Default threshold is 3px and 5% relative error (D1-3px).
    Functional impl for stateless usage.

    A pixel is considered "bad" if BOTH conditions are met:
    1. Absolute error > threshold (default 3.0)
    2. Relative error > relative_threshold (default 0.05, i.e., 5%)

    Args:
        est: Predicted disparity map [B, H, W] or [B, 1, H, W]
        gt: Ground truth disparity map [B, H, W] or [B, 1, H, W]
        min_disp: Minimum disparity value
        max_disp: Maximum disparity value
        threshold: Absolute error threshold, default 3.0
        relative_threshold: Relative error threshold, default 0.05 (5%)

    Returns:
        tuple: (num_error_pixels, num_valid_pixels, d1_score)
            - num_error_pixels: Number of pixels with error > threshold
            - num_valid_pixels: Number of valid pixels
            - d1_score: D1 score (bad pixel ratio)
    """
    if not isinstance(est, Tensor):
        est = torch.tensor(est)
    if not isinstance(gt, Tensor):
        gt = torch.tensor(gt)

    if est.dim() == 4:
        est = est.squeeze(1)
    if gt.dim() == 4:
        gt = gt.squeeze(1)

    # Generate valid pixel mask
    mask = (gt >= min_disp) & (gt < max_disp)

    # Compute absolute error
    error_map = torch.abs(est - gt)

    # Consider error only within valid regions
    # Check where error exceeds BOTH absolute threshold AND relative threshold
    # This matches the SemStereo implementation: (E > 3) & (E / D_gt.abs() > 0.05)
    bad_pixels_map = (error_map > threshold) & (error_map / (gt.abs() + 1e-8) > relative_threshold)

    # Only bad pixels within valid mask count
    valid_bad_pixels = bad_pixels_map & mask

    num_error_pixels = valid_bad_pixels.sum()
    num_valid_pixels = mask.sum()

    if num_valid_pixels > 0:
        d1_score = num_error_pixels / num_valid_pixels
    else:
        d1_score = torch.tensor(0.0, device=est.device)

    return num_error_pixels, num_valid_pixels, d1_score


def compute_threshold_error(
    est: Tensor,
    gt: Tensor,
    min_disp: float,
    max_disp: float,
    threshold: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Compute threshold-based error metric (percentage of pixels with error > threshold).
    Commonly used thresholds: 1.0px, 2.0px, 3.0px.
    Functional impl for stateless usage.

    Args:
        est: Predicted disparity map [B, H, W] or [B, 1, H, W]
        gt: Ground truth disparity map [B, H, W] or [B, 1, H, W]
        min_disp: Minimum disparity value
        max_disp: Maximum disparity value
        threshold: Error threshold (e.g., 1.0, 2.0, 3.0)

    Returns:
        tuple: (num_error_pixels, num_valid_pixels, error_ratio)
            - num_error_pixels: Number of pixels with error > threshold
            - num_valid_pixels: Number of valid pixels
            - error_ratio: Ratio of error pixels to valid pixels
    """
    if not isinstance(est, Tensor):
        est = torch.tensor(est)
    if not isinstance(gt, Tensor):
        gt = torch.tensor(gt)

    if est.dim() == 4:
        est = est.squeeze(1)
    if gt.dim() == 4:
        gt = gt.squeeze(1)

    # Generate valid pixel mask
    mask = (gt >= min_disp) & (gt < max_disp)

    # Compute absolute error
    error_map = torch.abs(est - gt)

    # Check where error exceeds threshold
    bad_pixels_map = error_map > threshold

    # Only bad pixels within valid mask count
    valid_bad_pixels = bad_pixels_map & mask

    num_error_pixels = valid_bad_pixels.sum()
    num_valid_pixels = mask.sum()

    if num_valid_pixels > 0:
        error_ratio = num_error_pixels / num_valid_pixels
    else:
        error_ratio = torch.tensor(0.0, device=est.device)

    return num_error_pixels, num_valid_pixels, error_ratio


class EndPointError(Metric):
    """
    Compute End-Point-Error (EPE) for stereo matching.
    Calculates the average absolute error between predicted and ground truth disparities.
    """

    # state variables
    sum_error: Tensor
    total_valid_pixels: Tensor

    def __init__(self, min_disp: float = -64, max_disp: float = 64, **kwargs):
        super().__init__(**kwargs)
        self.min_disp = min_disp
        self.max_disp = max_disp

        # Register state, dist_reduce_fx='sum' means auto accumulate in DDP
        self.add_state("sum_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_valid_pixels", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        Update Metric state.

        Args:
            preds: Predicted disparity map [B, H, W] or [B, 1, H, W]
            target: Ground truth disparity map [B, H, W] or [B, 1, H, W]
        """
        # Call functional implementation
        total_error, num_valid, _ = compute_epe(preds, target, self.min_disp, self.max_disp)

        self.sum_error += total_error
        self.total_valid_pixels += num_valid

    def compute(self) -> Tensor:
        """
        Compute global EPE.
        """
        if self.total_valid_pixels > 0:
            return self.sum_error / self.total_valid_pixels
        return torch.tensor(0.0, device=self.sum_error.device)


class D1Error(Metric):
    """
    Compute D1 Error (Percentage of bad pixels) for stereo matching.
    A pixel is considered "bad" if BOTH conditions are met:
    1. Absolute error > threshold (default 3.0)
    2. Relative error > relative_threshold (default 0.05, i.e., 5%)
    """

    # state variables
    bad_pixels: Tensor
    total_valid_pixels: Tensor

    def __init__(
        self,
        min_disp: float = -64,
        max_disp: float = 64,
        threshold: float = 3.0,
        relative_threshold: float = 0.05,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.min_disp = min_disp
        self.max_disp = max_disp
        self.threshold = threshold
        self.relative_threshold = relative_threshold

        # Register state
        self.add_state("bad_pixels", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_valid_pixels", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        Update Metric state.

        Args:
            preds: Predicted disparity map [B, H, W] or [B, 1, H, W]
            target: Ground truth disparity map [B, H, W] or [B, 1, H, W]
        """
        # Call functional implementation
        num_error, num_valid, _ = compute_d1(
            preds, target, self.min_disp, self.max_disp, self.threshold, self.relative_threshold
        )

        self.bad_pixels += num_error
        self.total_valid_pixels += num_valid

    def compute(self) -> Tensor:
        """
        Compute global D1 Error.
        """
        if self.total_valid_pixels > 0:
            return self.bad_pixels / self.total_valid_pixels
        return torch.tensor(0.0, device=self.bad_pixels.device)


class ThresholdError(Metric):
    """
    Compute threshold-based error metric for stereo matching.
    Calculates the ratio of pixels where absolute error exceeds the threshold.
    Commonly used thresholds: 1.0px, 2.0px, 3.0px.
    """

    # state variables
    bad_pixels: Tensor
    total_valid_pixels: Tensor

    def __init__(
        self,
        threshold: float,
        min_disp: float = -64,
        max_disp: float = 64,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.min_disp = min_disp
        self.max_disp = max_disp

        # Register state
        self.add_state("bad_pixels", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_valid_pixels", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        Update Metric state.

        Args:
            preds: Predicted disparity map [B, H, W] or [B, 1, H, W]
            target: Ground truth disparity map [B, H, W] or [B, 1, H, W]
        """
        # Call functional implementation
        num_error, num_valid, _ = compute_threshold_error(preds, target, self.min_disp, self.max_disp, self.threshold)

        self.bad_pixels += num_error
        self.total_valid_pixels += num_valid

    def compute(self) -> Tensor:
        """
        Compute global threshold error.
        """
        if self.total_valid_pixels > 0:
            return self.bad_pixels / self.total_valid_pixels
        return torch.tensor(0.0, device=self.bad_pixels.device)
