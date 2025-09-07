"""
Window sliding utilities for processing large images in patches.
"""

from collections.abc import Generator, Iterable
from typing import Any

import numpy as np
import torch


class WindowSlider:
    """
    A sliding window generator that extracts patches from full images.

    Args:
        slide_keys: List of keys in the sample that should be processed with window sliding
        window_size: Size of the square window to extract
        stride: Stride between windows
        overlap: Overlap ratio between windows (0 to 1). If specified, stride is calculated automatically.
    """

    def __init__(
        self,
        slide_keys: list[str],
        window_size: int = 64,
        stride: int | None = None,
        overlap: float | None = None,
    ):
        self.slide_keys = slide_keys
        self.window_size = window_size

        if overlap is not None:
            if not 0 <= overlap < 1:
                raise ValueError("Overlap must be between 0 and 1")
            self.stride = int(window_size * (1 - overlap))
        elif stride is not None:
            self.stride = stride
        else:
            self.stride = window_size  # no overlap by default

    def slide_windows(
        self, sample: dict[str, Any]
    ) -> Generator[dict[str, Any], None, None]:
        """
        Generate sliding windows from a full image sample.

        Args:
            sample: Dictionary containing full image data

        Yields:
            Dictionary containing windowed data with the same keys as input.
        """
        # Get image dimensions from the first slide key
        if not self.slide_keys or self.slide_keys[0] not in sample:
            raise ValueError(
                f"None of the slide keys {self.slide_keys} found in sample"
            )

        # Convert first image to tensor to get dimensions
        img = sample[self.slide_keys[0]]
        if not torch.is_tensor(img):
            img = torch.as_tensor(img)

        # Determine if we have batch dimension and get spatial dimensions
        has_batch = img.ndim == 4  # (B, C, H, W)
        if has_batch:
            batch_size = img.shape[0]
            # Use first image to determine spatial dimensions
            if img.ndim == 4:
                _, h, w = img.shape[1], img.shape[2], img.shape[3]
            else:
                raise ValueError(f"Unsupported image dimensions: {img.shape}")
        elif img.ndim == 3:  # (C, H, W)
            batch_size = 1
            _, h, w = img.shape
        elif img.ndim == 2:  # (H, W)
            batch_size = 1
            h, w = img.shape
            img = img.unsqueeze(0)  # Add channel dimension
        else:
            raise ValueError(f"Unsupported image dimensions: {img.shape}")

        # Calculate number of windows
        num_windows_h = (h - self.window_size) // self.stride + 1
        num_windows_w = (w - self.window_size) // self.stride + 1

        # Store original image info for merging
        original_info = {
            "original_shape": (h, w),
            "window_size": self.window_size,
            "stride": self.stride,
            "num_windows_h": num_windows_h,
            "num_windows_w": num_windows_w,
            "has_batch": has_batch,
            "batch_size": batch_size,
        }

        # Generate all windows, each containing the full batch
        for i in range(num_windows_h):
            for j in range(num_windows_w):
                h_start = i * self.stride
                h_end = h_start + self.window_size
                w_start = j * self.stride
                w_end = w_start + self.window_size

                # Extract window for the entire batch at once
                window_sample = self._extract_window_batch(
                    sample, h_start, h_end, w_start, w_end, i, j, original_info
                )
                yield window_sample

    def _extract_window_batch(
        self,
        sample: dict[str, Any],
        h_start: int,
        h_end: int,
        w_start: int,
        w_end: int,
        i: int,
        j: int,
        original_info: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Extract a window from the sample, preserving the full batch dimension.

        Args:
            sample: Dictionary containing full image data
            h_start, h_end: Height start and end indices
            w_start, w_end: Width start and end indices
            i, j: Window grid indices
            original_info: Original image information

        Returns:
            Dictionary containing windowed data with full batch dimension
        """
        window_sample = {}

        # Extract window for each key in the sample
        for key, value in sample.items():
            if key.startswith("__"):
                continue

            # Only process slide keys with window sliding
            if key in self.slide_keys:
                # Convert to tensor for uniform processing
                value_tensor = (
                    torch.as_tensor(value) if not torch.is_tensor(value) else value
                )
                original_type = type(value)

                # Extract window based on dimensions, preserving batch
                if value_tensor.ndim == 4:  # (B, C, H, W)
                    window = value_tensor[:, :, h_start:h_end, w_start:w_end]
                elif value_tensor.ndim == 3:  # (C, H, W)
                    # Add batch dimension to maintain consistency
                    window = value_tensor.unsqueeze(0)[
                        :, :, h_start:h_end, w_start:w_end
                    ]
                elif value_tensor.ndim == 2:  # (H, W)
                    # Add batch and channel dimensions to maintain consistency
                    window = value_tensor.unsqueeze(0).unsqueeze(0)[
                        :, :, h_start:h_end, w_start:w_end
                    ]
                else:
                    # For other dimensions, keep as is but add batch dimension if needed
                    window = value_tensor
                    if window.ndim == 0:
                        window = window.unsqueeze(0)  # Add batch dimension

                # Convert back to original type if needed
                if original_type is np.ndarray and not isinstance(value, torch.Tensor):
                    window = window.cpu().numpy()

                window_sample[key] = window
            else:
                # For non-slide keys, keep the original value
                window_sample[key] = value

        # Add window position info (batch_idx is now -1 to indicate full batch)
        window_sample["window_info"] = {
            "h_start": h_start,
            "h_end": h_end,
            "w_start": w_start,
            "w_end": w_end,
            "i": i,
            "j": j,
            "batch_idx": -1,  # -1 indicates full batch window
            **original_info,
        }

        return window_sample

    def create_window_generator(
        self, dataloader: torch.utils.data.DataLoader | Iterable[dict[str, Any]]
    ) -> Generator[dict[str, Any], None, None]:
        """
        Create a generator that yields windows from all samples in the dataloader.

        Args:
            dataloader: DataLoader that yields full image samples

        Yields:
            Dictionary containing windowed data
        """
        for sample in dataloader:
            # Remove batch dimension if present
            if isinstance(sample, dict):
                batch_sample = sample
            else:
                batch_sample = sample

            # Generate windows from this sample
            for window_sample in self.slide_windows(batch_sample):
                yield window_sample

    @staticmethod
    def merge_windows(
        window_results: list[dict[str, Any]],
        merge_method: str = "average",
        merged_keys: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Merge windowed results back to full image size.

        Args:
            window_results: List containing results from all windows,
                            each element is a dictionary with data and window_info
            merge_method: Method for merging overlapping regions ('average', 'first', 'last')

        Returns:
            Dictionary containing merged full image results
        """
        if not window_results:
            return {}

        # Get info from first window
        first_window = window_results[0]
        window_info = first_window.get("window_info", {})

        if not window_info:
            raise ValueError("Window info not found in results")

        original_shape = window_info["original_shape"]
        has_batch = window_info["has_batch"]
        batch_size = window_info["batch_size"]

        # Get all data keys (excluding metadata)
        if merged_keys is None:
            data_keys = [
                k
                for k in first_window.keys()
                if k not in ["window_info", "__key__", "__url__"]
            ]
        else:
            data_keys = merged_keys

        # Initialize merged results
        merged_results = {}

        # Process each key
        for key in data_keys:
            first_data = first_window[key]

            # Skip keys that are not tensor or numpy (keep original value)
            if not torch.is_tensor(first_data) and not isinstance(
                first_data, np.ndarray
            ):
                merged_results[key] = first_data
                continue

            # Convert to tensor for uniform processing
            first_tensor = (
                torch.as_tensor(first_data)
                if not torch.is_tensor(first_data)
                else first_data
            )
            original_type = type(first_data)

            # Determine if this key should be merged spatially
            should_merge_spatially = False
            if first_tensor.ndim == 4:  # (B, C, H, W)
                if (
                    first_tensor.shape[2] == window_info["window_size"]
                    and first_tensor.shape[3] == window_info["window_size"]
                ):
                    should_merge_spatially = True
            elif first_tensor.ndim == 3:  # (C, H, W)
                if (
                    first_tensor.shape[1] == window_info["window_size"]
                    and first_tensor.shape[2] == window_info["window_size"]
                ):
                    should_merge_spatially = True
            elif first_tensor.ndim == 2:  # (H, W)
                if (
                    first_tensor.shape[0] == window_info["window_size"]
                    and first_tensor.shape[1] == window_info["window_size"]
                ):
                    should_merge_spatially = True

            if not should_merge_spatially:
                # For non-spatial data, just keep the first window's value
                merged_results[key] = first_data
                continue

            # Initialize output tensor for spatial merging
            if first_tensor.ndim == 4:  # (B, C, H, W)
                output_shape = (first_tensor.shape[1],) + original_shape
                merged_tensor = torch.zeros(
                    (first_tensor.shape[0],) + output_shape,
                    dtype=first_tensor.dtype,
                    device=first_tensor.device,
                )
            elif first_tensor.ndim == 3:  # (C, H, W)
                output_shape = (first_tensor.shape[0],) + original_shape
                if has_batch:
                    merged_tensor = torch.zeros(
                        (batch_size,) + output_shape,
                        dtype=first_tensor.dtype,
                        device=first_tensor.device,
                    )
                else:
                    merged_tensor = torch.zeros(
                        output_shape,
                        dtype=first_tensor.dtype,
                        device=first_tensor.device,
                    )
            elif first_tensor.ndim == 2:  # (H, W)
                output_shape = original_shape
                if has_batch:
                    merged_tensor = torch.zeros(
                        (batch_size, 1) + output_shape,  # Add channel dimension
                        dtype=first_tensor.dtype,
                        device=first_tensor.device,
                    )
                else:
                    merged_tensor = torch.zeros(
                        output_shape,
                        dtype=first_tensor.dtype,
                        device=first_tensor.device,
                    )
            else:
                output_shape = first_tensor.shape
                merged_tensor = torch.zeros(
                    output_shape, dtype=first_tensor.dtype, device=first_tensor.device
                )

            # Initialize count tensor for averaging if needed
            count_tensor = None
            if merge_method == "average":
                if has_batch:
                    count_tensor = torch.zeros(
                        (batch_size,) + original_shape,
                        dtype=torch.float32,
                        device=first_tensor.device,
                    )
                else:
                    count_tensor = torch.zeros(
                        original_shape, dtype=torch.float32, device=first_tensor.device
                    )

            # Merge all windows
            for window_data in window_results:
                if key not in window_data:
                    continue

                # Convert to tensor for uniform processing
                data = (
                    torch.as_tensor(window_data[key])
                    if not torch.is_tensor(window_data[key])
                    else window_data[key]
                )
                info = window_data["window_info"]
                h_start, h_end = info["h_start"], info["h_end"]
                w_start, w_end = info["w_start"], info["w_end"]

                if merge_method == "average" and count_tensor is not None:
                    if data.dim() == 4:  # (B, C, H, W)
                        merged_tensor[:, :, h_start:h_end, w_start:w_end] += data
                        count_tensor[:, h_start:h_end, w_start:w_end] += 1
                    elif data.dim() == 3:  # (C, H, W)
                        if has_batch:
                            # This should not happen with the new format
                            merged_tensor[:, :, h_start:h_end, w_start:w_end] += (
                                data.unsqueeze(0)
                            )
                            count_tensor[:, h_start:h_end, w_start:w_end] += 1
                        else:
                            merged_tensor[:, h_start:h_end, w_start:w_end] += data
                            count_tensor[h_start:h_end, w_start:w_end] += 1
                    elif data.dim() == 2:  # (H, W)
                        if has_batch:
                            merged_tensor[:, 0, h_start:h_end, w_start:w_end] += data
                            count_tensor[:, h_start:h_end, w_start:w_end] += 1
                        else:
                            merged_tensor[h_start:h_end, w_start:w_end] += data
                            count_tensor[h_start:h_end, w_start:w_end] += 1
                elif merge_method == "first":
                    if data.dim() == 4:  # (B, C, H, W)
                        mask = (
                            merged_tensor[:, :, h_start:h_end, w_start:w_end].sum(
                                dim=(0, 1)
                            )
                            == 0
                        )
                        if mask.any():
                            merged_tensor[:, :, h_start:h_end, w_start:w_end][
                                :, :, mask
                            ] = data[:, :, mask]
                    elif data.dim() == 3:  # (C, H, W)
                        if has_batch:
                            mask = (
                                merged_tensor[:, :, h_start:h_end, w_start:w_end].sum(
                                    dim=(0, 1)
                                )
                                == 0
                            )
                            if mask.any():
                                merged_tensor[:, :, h_start:h_end, w_start:w_end][
                                    :, :, mask
                                ] = data.unsqueeze(0)[:, :, mask]
                        else:
                            mask = (
                                merged_tensor[:, h_start:h_end, w_start:w_end].sum(
                                    dim=0
                                )
                                == 0
                            )
                            if mask.any():
                                merged_tensor[:, h_start:h_end, w_start:w_end][
                                    :, mask
                                ] = data[:, mask]
                    elif data.dim() == 2:  # (H, W)
                        if has_batch:
                            mask = (
                                merged_tensor[:, 0, h_start:h_end, w_start:w_end] == 0
                            )
                            if mask.any():
                                merged_tensor[:, 0, h_start:h_end, w_start:w_end][
                                    :, mask
                                ] = data[mask]
                        else:
                            mask = merged_tensor[h_start:h_end, w_start:w_end] == 0
                            if mask.any():
                                merged_tensor[h_start:h_end, w_start:w_end][mask] = (
                                    data[mask]
                                )
                elif merge_method == "last":
                    if data.dim() == 4:  # (B, C, H, W)
                        merged_tensor[:, :, h_start:h_end, w_start:w_end] = data
                    elif data.dim() == 3:  # (C, H, W)
                        if has_batch:
                            merged_tensor[:, :, h_start:h_end, w_start:w_end] = (
                                data.unsqueeze(0)
                            )
                        else:
                            merged_tensor[:, h_start:h_end, w_start:w_end] = data
                    elif data.dim() == 2:  # (H, W)
                        if has_batch:
                            merged_tensor[:, 0, h_start:h_end, w_start:w_end] = data
                        else:
                            merged_tensor[h_start:h_end, w_start:w_end] = data

            # Apply averaging if needed
            if merge_method == "average" and count_tensor is not None:
                count_tensor = count_tensor.clamp(min=1)
                if merged_tensor.dim() == 4:  # (B, C, H, W)
                    merged_tensor = merged_tensor / count_tensor.unsqueeze(1)
                elif merged_tensor.dim() == 3:  # (C, H, W)
                    merged_tensor = merged_tensor / count_tensor.unsqueeze(0)
                else:
                    merged_tensor = merged_tensor / count_tensor

            # Convert back to original type if needed
            if original_type is np.ndarray:
                merged_tensor = merged_tensor.cpu().numpy()

            merged_results[key] = merged_tensor

        return merged_results


def create_windowed_dataloader(
    dataloader,
    slide_keys: list[str],
    window_size: int = 64,
    stride: int | None = None,
    overlap: float | None = None,
) -> Generator[dict[str, Any], None, None]:
    """
    Create a windowed dataloader generator from a full image dataloader.

    Args:
        dataloader: DataLoader that yields full image samples
        slide_keys: List of keys in the sample that should be processed with window sliding
        window_size: Size of the square window to extract
        stride: Stride between windows
        overlap: Overlap ratio between windows (0 to 1)

    Yields:
        Dictionary containing windowed data
    """
    slider = WindowSlider(
        slide_keys=slide_keys, window_size=window_size, stride=stride, overlap=overlap
    )
    return slider.create_window_generator(dataloader)
