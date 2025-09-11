"""
Adaptive Hyperspectral Anomaly Detection MSE Loss

This module implements an adaptive MSE loss function for hyperspectral anomaly detection
based on the Auto-AD method. The loss function automatically updates spatial weights
based on reconstruction errors to focus more on background regions and enhance anomaly detection.

The core idea is:
1. Background data has regular patterns and can be well reconstructed
2. Anomalies break data patterns and are difficult to reconstruct
3. Adaptive weighting focuses training on background reconstruction
4. Large reconstruction errors indicate potential anomalies

Reference:
    Auto-AD: Automatically Detecting Anomalies in Hyperspectral Images
    https://github.com/RSIDEA-WHU2020/Auto-AD
"""

# from typing import NoReturn

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utilities.train_utils.state import StepsCounter


class AdaptiveHADMSE(nn.Module):
    """
    Adaptive MSE Loss for Hyperspectral Anomaly Detection

    This loss function implements the adaptive weighting mechanism from Auto-AD,
    which automatically adjusts spatial weights based on reconstruction errors
    to enhance anomaly detection performance.

    Key features:
    - Adaptive spatial weighting based on reconstruction errors
    - Automatic weight updates at specified intervals
    - Residual calculation for anomaly scoring
    - Compatible with PyTorch training loops

    Args:
        update_interval (int): Number of steps between weight updates (default: 100)
        step_name (str): Name of the step counter to use (default: "train")
        init_mask_value (float): Initial mask value (default: 1.0)
        init_residual_value (float): Initial residual value (default: 1.0)
        epsilon (float): Small value to prevent division by zero (default: 1e-6)
        device (str): Device to use for computations (default: "cuda")

    Example:
        >>> loss_fn = AdaptiveHADMSE(update_interval=100, step_name="train")
        >>> output = model(input)
        >>> loss, anomaly_score = loss_fn(output, target)
        >>> loss.backward()
    """

    def __init__(
        self,
        update_interval: int = 100,
        step_name: str = "train",
        init_mask_value: float = 1.0,
        init_residual_value: float = 1.0,
        epsilon: float = 1e-6,
        device: str = "cuda",
    ):
        super().__init__()
        self.update_interval = update_interval
        self.step_name = step_name
        self.epsilon = epsilon
        self.device = device

        # Initialize step counter
        self.step_counter = StepsCounter(step_names=[step_name])

        # Initialize mask and residual
        self.register_buffer("mask", None)
        self.register_buffer("residual", None)
        self.init_mask_value = init_mask_value
        self.init_residual_value = init_residual_value

        # For storing current reconstruction error
        self.current_residual = None

    def _initialize_buffers(self, shape: tuple[int, ...]):
        """Initialize mask and residual buffers based on input shape"""
        if self.mask is None:
            # shape: (batch, channels, height, width)
            self.mask = torch.ones(shape, device=self.device) * self.init_mask_value

        if self.residual is None:
            # residual is batch-wise spatial map: (batch, height, width)
            batch, channels, height, width = shape
            self.residual = (
                torch.ones((batch, height, width), device=self.device)
                * self.init_residual_value
            )

    def _update_weights(self, output: torch.Tensor, target: torch.Tensor):
        """
        Update spatial weights based on reconstruction errors

        Args:
            output: Network output [B, C, H, W]
            target: Target image [B, C, H, W]
        """
        batch, channels, height, width = output.shape

        # Calculate reconstruction error for each pixel
        error = (output - target) ** 2  # [B, C, H, W]

        # Sum errors across channels to get spatial residual map
        residual_img = error.sum(dim=1)  # [B, H, W]

        # Store current residual for anomaly detection (keep batch dimension)
        self.current_residual = residual_img.detach()

        # Average across batch dimension for weight calculation
        # This creates a unified weight mask based on overall reconstruction patterns
        avg_residual = residual_img.mean(dim=0)  # [H, W]

        # Convert averaged residuals to weights (inverse relationship)
        # Higher error -> lower weight
        r_max = avg_residual.max()
        residual_weights = r_max - avg_residual  # Invert
        r_min = residual_weights.min()
        r_max = residual_weights.max()

        # Normalize to [0, 1]
        if r_max > r_min:
            residual_weights = (residual_weights - r_min) / (
                r_max - r_min + self.epsilon
            )
        else:
            residual_weights = torch.ones_like(residual_weights)

        # Update mask: all channels use the same weight pattern
        for b in range(batch):
            for c in range(channels):
                self.mask[b, c, :] = residual_weights

        # Update residual for output
        self.residual = residual_img

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        return_anomaly_score: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Calculate adaptive MSE loss

        Args:
            output: Network output tensor [B, C, H, W]
            target: Target tensor [B, C, H, W]
            return_anomaly_score: Whether to return anomaly score

        Returns:
            loss: Adaptive MSE loss
            anomaly_score: Anomaly score map (if return_anomaly_score=True)
        """
        # Initialize buffers if needed
        self._initialize_buffers(output.shape)

        # Get current step
        current_step = self.step_counter.get(self.step_name)

        # Update weights if needed
        if current_step > 0 and current_step % self.update_interval == 0:
            self._update_weights(output, target)

        # Create clones for loss calculation
        mask_clone = self.mask.detach().clone()

        # Calculate weighted MSE loss
        weighted_output = output * mask_clone
        weighted_target = target * mask_clone
        loss = F.mse_loss(weighted_output, weighted_target)

        # Return loss and optionally anomaly score
        if return_anomaly_score:
            anomaly_score = self.get_anomaly_score()
            return loss, anomaly_score
        else:
            return loss, None

    def get_anomaly_score(self) -> torch.Tensor:
        """
        Get current anomaly score map

        Returns:
            anomaly_score: Normalized anomaly score map [B, H, W]
        """
        if self.current_residual is None:
            # Return uniform scores if no residuals calculated yet
            return torch.ones_like(self.residual)

        # Normalize anomaly scores (normalize each sample independently)
        anomaly_score = self.current_residual
        batch_size = anomaly_score.shape[0]

        # Normalize each sample in the batch separately
        for b in range(batch_size):
            score_min = anomaly_score[b].min()
            score_max = anomaly_score[b].max()

            if score_max > score_min:
                anomaly_score[b] = (anomaly_score[b] - score_min) / (
                    score_max - score_min + self.epsilon
                )

        return anomaly_score

    def get_current_mask(self) -> torch.Tensor:
        """Get current spatial mask"""
        return self.mask.detach().clone()

    def get_current_residual(self) -> torch.Tensor:
        """Get current residual map"""
        return self.residual.detach().clone()

    def reset_buffers(self):
        """Reset mask and residual to initial values"""
        if self.mask is not None:
            self.mask.fill_(self.init_mask_value)
        if self.residual is not None:
            self.residual.fill_(self.init_residual_value)
        self.current_residual = None

    def extra_repr(self) -> str:
        return (
            f"update_interval={self.update_interval}, "
            f"step_name='{self.step_name}', "
            f"epsilon={self.epsilon}"
        )


# Example usage and testing
def test_adaptive_had_mse():
    """Test function to demonstrate usage"""
    print("Testing AdaptiveHADMSE...")

    # Initialize all step counters first
    step_names = [f"test_batch_{batch_size}" for batch_size in [1, 2, 4]]
    global_step_counter = StepsCounter(step_names=step_names)

    # Test with different batch sizes
    for batch_size in [1, 2, 4]:
        print(f"\n--- Testing with batch_size = {batch_size} ---")

        channels, height, width = 10, 64, 64

        # Initialize loss function
        loss_fn = AdaptiveHADMSE(
            update_interval=25,
            step_name=f"test_batch_{batch_size}",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        # Use the global step counter
        loss_fn.step_counter = global_step_counter

        # Create sample tensors with different patterns per sample
        device = loss_fn.device
        target = torch.randn(batch_size, channels, height, width, device=device)

        # Add some "anomalies" to different samples
        output = target + 0.1 * torch.randn(
            batch_size, channels, height, width, device=device
        )

        # Add specific anomalies to each sample
        for b in range(batch_size):
            # Each sample has anomaly in different location
            anomaly_y = (height // (batch_size + 1)) * (b + 1)
            anomaly_x = width // 2
            output[
                b, :, anomaly_y - 2 : anomaly_y + 3, anomaly_x - 2 : anomaly_x + 3
            ] *= 2.0

        # Simulate training loop
        for step in range(50):
            # Update step counter
            loss_fn.step_counter.update(f"test_batch_{batch_size}")

            # Calculate loss
            loss, anomaly_score = loss_fn(output, target)

            if step % 25 == 0:
                print(
                    f"Step {step}: Loss = {loss.item():.6f}, "
                    f"Anomaly score shape = {anomaly_score.shape}"
                )

                # Show anomaly score range for each sample
                for b in range(batch_size):
                    print(
                        f"  Sample {b}: Anomaly score range = [{anomaly_score[b].min():.4f}, {anomaly_score[b].max():.4f}]"
                    )

                # Show mask statistics (should be same for all samples)
                mask = loss_fn.get_current_mask()
                print(
                    f"  Mask shape = {mask.shape}, range = [{mask.min():.4f}, {mask.max():.4f}]"
                )

                # Show residual shape
                residual = loss_fn.get_current_residual()
                print(f"  Residual shape = {residual.shape}")

    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    test_adaptive_had_mse()
