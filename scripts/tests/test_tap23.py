import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class Interp23Tap(nn.Module):
    """
    PyTorch implementation of the interp23tap MATLAB function.

    Interpolates the input tensor using a 23-coefficient polynomial interpolator,
    upsampling by the given ratio. The ratio must be a power of 2.

    Args:
        ratio (int): Scale ratio for upsampling. Must be a power of 2.
    """

    def __init__(self, ratio: int):
        super().__init__()

        if not (ratio > 0 and (ratio & (ratio - 1) == 0)):
            raise ValueError("Error: Only resize factors power of 2 are supported.")
        self.ratio = ratio
        self.num_upsamples = int(math.log2(ratio))

        # Define the 23-tap filter coefficients (CDF23 from MATLAB code)
        cdf23_coeffs = 2.0 * np.array(
            [
                0.5,
                0.305334091185,
                0.0,
                -0.072698593239,
                0.0,
                0.021809577942,
                0.0,
                -0.005192756653,
                0.0,
                0.000807762146,
                0.0,
                -0.000060081482,
            ]
        )
        # Make symmetric
        base_coeffs = np.concatenate([np.flip(cdf23_coeffs[1:]), cdf23_coeffs])
        base_coeffs_t = torch.tensor(base_coeffs, dtype=torch.float32)

        # Reshape kernel for 2D convolution (separable filter)
        # Kernel for filtering along height (columns in MATLAB)
        kernel_h = base_coeffs_t.view(1, 1, -1, 1)  # Shape (1, 1, 23, 1)
        # Kernel for filtering along width (rows in MATLAB)
        kernel_w = base_coeffs_t.view(1, 1, 1, -1)  # Shape (1, 1, 1, 23)

        # Register kernels as buffers
        self.register_buffer("kernel_h", kernel_h)
        self.register_buffer("kernel_w", kernel_w)

        # Calculate padding size (kernel_size=23)
        self.padding = (base_coeffs_t.shape[0] - 1) // 2  # Should be 11

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for interpolation.

        Args:
            x (torch.Tensor): Input tensor of shape (bs, c, h, w).

        Returns:
            torch.Tensor: Interpolated tensor of shape (bs, c, h * ratio, w * ratio).
        """
        if self.ratio == 1:
            return x

        current_img = x
        bs, c, h_curr, w_curr = current_img.shape

        for k in range(self.num_upsamples):
            h_curr *= 2
            w_curr *= 2

            # Upsample by inserting zeros
            upsampled = torch.zeros(
                bs, c, h_curr, w_curr, device=x.device, dtype=x.dtype
            )

            # Place original pixels according to MATLAB logic
            if k == 0:
                # I1LRU(2:2:end,2:2:end,:) = I_Interpolated;
                upsampled[..., 1::2, 1::2] = current_img
            else:
                # I1LRU(1:2:end,1:2:end,:) = I_Interpolated;
                upsampled[..., ::2, ::2] = current_img

            # Apply separable convolution with circular padding
            # Grouped convolution: apply filter independently per channel
            # Reshape for grouped conv: (1, bs*c, H, W) or apply channel-wise
            # Using conv2d with groups=c is efficient

            # Pad for horizontal filter (width)
            # Pad width dimension (dim 3) by self.padding on both sides
            padded_w = F.pad(
                upsampled, (self.padding, self.padding, 0, 0), mode="circular"
            )
            # Apply horizontal filter
            # Input: (bs, c, H, W_padded), Kernel: (1, 1, 1, K) -> Output: (bs, c, H, W)
            # We need kernel shape (c, 1, 1, K) for grouped convolution
            kernel_w_grouped = self.kernel_w.repeat(c, 1, 1, 1)
            filtered_w = F.conv2d(padded_w, kernel_w_grouped, groups=c)

            # Pad for vertical filter (height)
            # Pad height dimension (dim 2) by self.padding on both sides
            padded_h = F.pad(
                filtered_w, (0, 0, self.padding, self.padding), mode="circular"
            )
            # Apply vertical filter
            # Input: (bs, c, H_padded, W), Kernel: (1, 1, K, 1) -> Output: (bs, c, H, W)
            # We need kernel shape (c, 1, K, 1) for grouped convolution
            kernel_h_grouped = self.kernel_h.repeat(c, 1, 1, 1)
            filtered_h = F.conv2d(padded_h, kernel_h_grouped, groups=c)

            current_img = filtered_h  # Update image for next iteration

        return current_img


# Example Usage:
if __name__ == "__main__":
    ratio = 4  # Example ratio (must be power of 2)
    interpolator = Interp23Tap(ratio)

    # Create a dummy input tensor (batch_size=1, channels=3, height=32, width=32)
    dummy_input = torch.randn(1, 3, 32, 32)

    # Perform interpolation
    output_tensor = interpolator(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output_tensor.shape}")

    # Check if output shape is correct
    expected_h = dummy_input.shape[2] * ratio
    expected_w = dummy_input.shape[3] * ratio
    assert output_tensor.shape == (
        dummy_input.shape[0],
        dummy_input.shape[1],
        expected_h,
        expected_w,
    )
    print("Output shape is correct.")

    # Test with ratio=1
    interpolator_r1 = Interp23Tap(ratio=1)
    output_r1 = interpolator_r1(dummy_input)
    assert output_r1.shape == dummy_input.shape
    assert torch.allclose(output_r1, dummy_input)
    print("Ratio=1 test passed.")

    # Test with non-power of 2 ratio (should raise ValueError)
    try:
        Interp23Tap(ratio=3)
    except ValueError as e:
        print(f"Caught expected error for ratio=3: {e}")
