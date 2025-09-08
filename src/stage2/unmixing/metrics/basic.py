from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchmetrics.aggregation import MeanMetric


class UnmixingMetrics(torch.nn.Module):
    """
    Hyperspectral Unmixing Metrics Class

    This class provides metrics for evaluating hyperspectral unmixing models,
    including SAD (Spectral Angle Distance) and MSE (Mean Squared Error).
    It supports PyTorch tensors and multi-process synchronization through MeanMetric.
    All calculations are performed using PyTorch operations.

    Attributes:
        sad_metrics (Dict[str, MeanMetric]): Dictionary of MeanMetric objects for SAD calculation
        mse_metrics (Dict[str, MeanMetric]): Dictionary of MeanMetric objects for MSE calculation
        num_endmembers (Optional[int]): Number of endmembers
        call_count (int): Number of times the metrics have been called
        sad_avg (Optional[float]): Average SAD value
        mse_avg (Optional[float]): Average MSE value
    """

    def __init__(
        self, num_endmembers: Optional[int] = None, device: str = "cuda"
    ) -> None:
        """
        Initialize the UnmixingMetrics class

        Args:
            num_endmembers (Optional[int]): Number of endmembers.
                                           If None, will be determined during first call.
        """
        super().__init__()
        self.num_endmembers = num_endmembers
        self.call_count = 0

        # Initialize metrics dictionaries
        self.sad_metrics: Dict[str, MeanMetric] = {}
        self.mse_metrics: Dict[str, MeanMetric] = {}

        # Initialize average metrics
        self.sad_avg: Optional[float] = None
        self.mse_avg: Optional[float] = None

        self.device = device

    @staticmethod
    def _normalize_spectra(spectra: torch.Tensor) -> torch.Tensor:
        """
        Normalize spectra to [0, 1] range using PyTorch

        Args:
            spectra (torch.Tensor): Input spectra

        Returns:
            torch.Tensor: Normalized spectra
        """
        if spectra.dim() == 1:
            # Single spectrum
            return spectra / spectra.max()
        else:
            # Multiple spectra
            return spectra / spectra.max(dim=-1, keepdim=True)[0]

    @staticmethod
    def _compute_sad(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute Spectral Angle Distance (SAD) between two spectra using PyTorch

        Args:
            y_true (torch.Tensor): True spectrum
            y_pred (torch.Tensor): Predicted spectrum

        Returns:
            torch.Tensor: SAD value in radians
        """
        # Compute dot product
        dot_product = torch.dot(y_pred, y_true)

        # Compute norms
        norm_true = torch.norm(y_true)
        norm_pred = torch.norm(y_pred)

        # Compute cosine similarity and clip to avoid numerical issues
        cos_sim = dot_product / (norm_true * norm_pred)
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)

        # Compute SAD
        sad = torch.acos(cos_sim)

        return sad

    @staticmethod
    def _compute_sad_matrix(
        endmembers: torch.Tensor, endmembers_gt: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute SAD matrix between all pairs of endmembers using PyTorch

        Args:
            endmembers (torch.Tensor): Predicted endmembers [num_em, bands]
            endmembers_gt (torch.Tensor): Ground truth endmembers [num_em, bands]

        Returns:
            torch.Tensor: SAD matrix [num_em, num_em]
        """
        num_endmembers = endmembers.shape[0]

        # Create SAD matrix
        sad_matrix = torch.ones(
            (num_endmembers, num_endmembers), device=endmembers.device
        )

        # Normalize endmembers
        for i in range(num_endmembers):
            endmembers[i, :] = UnmixingMetrics._normalize_spectra(endmembers[i, :])
            endmembers_gt[i, :] = UnmixingMetrics._normalize_spectra(
                endmembers_gt[i, :]
            )

        # Compute SAD matrix
        for i in range(num_endmembers):
            for j in range(num_endmembers):
                sad_matrix[i, j] = UnmixingMetrics._compute_sad(
                    endmembers[i, :], endmembers_gt[j, :]
                )

        return sad_matrix

    @staticmethod
    def _order_endmembers(
        endmembers: torch.Tensor, endmembers_gt: torch.Tensor
    ) -> Tuple[Dict[int, int], List[float], torch.Tensor]:
        """
        Match predicted endmembers to ground truth endmembers using PyTorch

        Args:
            endmembers (torch.Tensor): Predicted endmembers [num_em, bands]
            endmembers_gt (torch.Tensor): Ground truth endmembers [num_em, bands]

        Returns:
            Tuple[Dict[int, int], List[float], torch.Tensor]:
                - Mapping from GT index to predicted index
                - SAD values for matched endmembers
                - Average SAD value
        """
        num_endmembers = endmembers.shape[0]
        match_dict: Dict[int, int] = {}
        sad_values: List[float] = []

        # Compute SAD matrix
        sad_matrix = UnmixingMetrics._compute_sad_matrix(endmembers, endmembers_gt)

        # Find best matches
        matched_rows = 0
        while matched_rows < num_endmembers:
            min_sad = sad_matrix.min()
            if min_sad >= 100:  # Sentinel value
                break

            # Find indices of minimum SAD
            indices = torch.where(sad_matrix == min_sad)
            if len(indices[0]) == 0:
                break

            pred_idx = int(indices[0][0].item())
            gt_idx = int(indices[1][0].item())

            # Store match
            match_dict[gt_idx] = pred_idx
            sad_values.append(min_sad.item())

            # Mark as matched
            sad_matrix[pred_idx, gt_idx] = 100
            sad_matrix[pred_idx, :] = 100
            sad_matrix[:, gt_idx] = 100

            matched_rows += 1

        # Compute average SAD
        sad_values_tensor = torch.tensor(sad_values, device=endmembers.device)
        avg_sad = (
            torch.sum(sad_values_tensor) / len(sad_values_tensor)
            if len(sad_values_tensor) > 0
            else torch.tensor(0.0, device=endmembers.device)
        )

        return match_dict, sad_values, avg_sad

    @staticmethod
    def _plot_endmembers(
        endmembers: torch.Tensor,
        endmembers_gt: torch.Tensor,
        abundances: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any, np.ndarray]:
        """
        Plot predicted and ground truth endmembers with SAD values

        Args:
            endmembers (torch.Tensor): Predicted endmembers [num_em, bands]
            endmembers_gt (torch.Tensor): Ground truth endmembers [num_em, bands]
            abundances (torch.Tensor): Predicted abundances [num_em, H, W]

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, plt.Figure, np.ndarray]:
                - SAD values for matched endmembers
                - Ordered predicted endmembers
                - Ordered predicted abundances
                - Matplotlib figure object (Any type to avoid import issues)
                - Matplotlib axes array
        """
        num_endmembers = endmembers.shape[0]
        n_cols = (num_endmembers + 1) // 2  # Number of columns in subplot grid

        # Match endmembers
        match_dict, sad_values, avg_sad = UnmixingMetrics._order_endmembers(
            endmembers, endmembers_gt
        )

        # Order endmembers and abundances according to matches
        ordered_endmembers_list: List[torch.Tensor] = []
        ordered_abundances_list: List[torch.Tensor] = []

        for i in range(num_endmembers):
            if i in match_dict:
                ordered_endmembers_list.append(endmembers[match_dict[i]])
                ordered_abundances_list.append(abundances[match_dict[i]])
            else:
                # If no match, use the first remaining endmember
                for j in range(num_endmembers):
                    if j not in match_dict.values():
                        ordered_endmembers_list.append(endmembers[j])
                        ordered_abundances_list.append(abundances[j])
                        match_dict[i] = j
                        break

        ordered_endmembers = torch.stack(ordered_endmembers_list)
        ordered_abundances = torch.stack(ordered_abundances_list)

        # Compute SAD for ordered endmembers
        ordered_sad_list: List[float] = []
        for i in range(num_endmembers):
            sad = UnmixingMetrics._compute_sad(ordered_endmembers[i], endmembers_gt[i])
            ordered_sad_list.append(sad.item())

        ordered_sad = torch.tensor(ordered_sad_list, device=endmembers.device)

        # Create figure
        fig, axes = plt.subplots(2, n_cols, figsize=(12, 8))
        if num_endmembers == 1:
            axes = np.array([[axes]])
        elif n_cols == 1:
            axes = axes.reshape(2, 1)

        # Convert to numpy for plotting
        endmembers_gt_np = endmembers_gt.detach().cpu().numpy()
        ordered_endmembers_np = ordered_endmembers.detach().cpu().numpy()

        # Plot endmembers
        for i in range(num_endmembers):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col]

            # Plot ground truth and predicted endmembers
            ax.plot(endmembers_gt_np[i, :], "r-", linewidth=1.5, label="Ground Truth")
            ax.plot(
                ordered_endmembers_np[i, :], "k--", linewidth=1.5, label="Predicted"
            )

            # Set title with SAD value
            ax.set_title(f"Endmember {i + 1}, SAD: {ordered_sad[i]:.4f} rad")
            ax.set_xlabel("Band Index")
            ax.set_ylabel("Reflectance")

            if i == 0:
                ax.legend()

        # Set main title
        fig.suptitle(
            f"Endmember Comparison (Average SAD: {avg_sad:.4f} rad)", fontsize=14
        )
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        # Add average SAD to the list
        ordered_sad_with_avg = torch.cat([ordered_sad, avg_sad.unsqueeze(0)])

        return ordered_sad_with_avg, ordered_endmembers, ordered_abundances, fig, axes

    def _compute_mse(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute Mean Squared Error between true and predicted abundances using PyTorch

        Args:
            y_true (torch.Tensor): True abundances [num_em, H, W]
            y_pred (torch.Tensor): Predicted abundances [num_em, H, W]

        Returns:
            torch.Tensor: MSE values for each endmember and average MSE
        """
        num_em = y_true.shape[0]

        # Reshape to 2D
        y_true_flat = y_true.reshape(num_em, -1)
        y_pred_flat = y_pred.reshape(num_em, -1)

        # Compute squared differences
        squared_diff = (y_pred_flat - y_true_flat) ** 2

        # Compute MSE for each endmember
        mse_values = torch.mean(squared_diff, dim=1)

        # Compute average MSE
        avg_mse = torch.sum(mse_values) / num_em

        # Append average MSE to the tensor
        mse_with_avg = torch.cat([mse_values, avg_mse.unsqueeze(0)])

        return mse_with_avg

    def _initialize_metrics(self, num_endmembers: int) -> None:
        """
        Initialize MeanMetric objects for SAD and MSE

        Args:
            num_endmembers (int): Number of endmembers
        """
        # Initialize SAD metrics
        self.sad_metrics = {}
        for i in range(num_endmembers):
            self.sad_metrics[f"sad_{i}"] = MeanMetric().to(self.device)
        self.sad_metrics["sad_avg"] = MeanMetric().to(self.device)

        # Initialize MSE metrics
        self.mse_metrics = {}
        for i in range(num_endmembers):
            self.mse_metrics[f"mse_{i}"] = MeanMetric().to(self.device)
        self.mse_metrics["mse_avg"] = MeanMetric().to(self.device)

        # Store number of endmembers
        self.num_endmembers = num_endmembers

    def _update_metrics(
        self, sad_values: torch.Tensor, mse_values: torch.Tensor
    ) -> None:
        """
        Update MeanMetric objects with new values

        Args:
            sad_values (torch.Tensor): SAD values for each endmember and average
            mse_values (torch.Tensor): MSE values for each endmember and average
        """
        # Update SAD metrics
        assert self.num_endmembers is not None, (
            "Number of endmembers is not initialized."
        )

        for i in range(self.num_endmembers):
            self.sad_metrics[f"sad_{i}"].update(sad_values[i])
        self.sad_metrics["sad_avg"].update(sad_values[-1])

        # Update MSE metrics
        for i in range(self.num_endmembers):
            self.mse_metrics[f"mse_{i}"].update(mse_values[i])
        self.mse_metrics["mse_avg"].update(mse_values[-1])

    def _compute_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute current metric values from MeanMetric objects

        Returns:
            Dict[str, Dict[str, float]]: Dictionary containing current SAD and MSE values
        """
        # Compute SAD values
        sad_values: Dict[str, float] = {}
        for name, metric in self.sad_metrics.items():
            sad_values[name] = metric.compute().item()

        # Compute MSE values
        mse_values: Dict[str, float] = {}
        for name, metric in self.mse_metrics.items():
            mse_values[name] = metric.compute().item()

        # Store average values
        self.sad_avg = sad_values["sad_avg"]
        self.mse_avg = mse_values["mse_avg"]

        return {"sad": sad_values, "mse": mse_values}

    def update(self, sad_values: torch.Tensor, mse_values: torch.Tensor):
        self._update_metrics(sad_values, mse_values)

    def compute(self):
        self._compute_metrics()

    def forward(
        self,
        endmembers: torch.Tensor | np.ndarray,
        endmembers_gt: torch.Tensor | np.ndarray,
        abundances: torch.Tensor | np.ndarray,
        abundances_gt: torch.Tensor | np.ndarray | None = None,
        plot: bool = False,
    ) -> (
        Dict[str, Dict[str, float]]
        | Tuple[Dict[str, Dict[str, float]], Any, np.ndarray]
    ):
        """
        Compute unmixing metrics for the given predictions

        Args:
            endmembers (torch.Tensor | np.ndarray): Predicted endmembers [num_em, bands]
            endmembers_gt (torch.Tensor | np.ndarray): Ground truth endmembers [num_em, bands]
            abundances (torch.Tensor | np.ndarray): Predicted abundances [num_em, H, W]
            abundances_gt (torch.Tensor | np.ndarray | None): Ground truth abundances [num_em, H, W]
            plot (bool): Whether to plot endmember comparison

        Returns:
            Dict[str, Dict[str, float]] | Tuple[Dict[str, Dict[str, float]], Any, np.ndarray]:
                - Dictionary containing SAD and MSE values
                - If plot=True, also returns figure and axes objects
        """
        # Initialize metrics if not already done
        if self.num_endmembers is None:
            self.num_endmembers = endmembers.shape[0]
            self._initialize_metrics(self.num_endmembers)

        # Ensure inputs are PyTorch tensors
        if not isinstance(endmembers, torch.Tensor):
            endmembers = torch.tensor(endmembers, dtype=torch.float32)
        if not isinstance(endmembers_gt, torch.Tensor):
            endmembers_gt = torch.tensor(endmembers_gt, dtype=torch.float32)
        if not isinstance(abundances, torch.Tensor):
            abundances = torch.tensor(abundances, dtype=torch.float32)
        if abundances_gt is not None and not isinstance(abundances_gt, torch.Tensor):
            abundances_gt = torch.tensor(abundances_gt, dtype=torch.float32)

        # Compute SAD metrics and order endmembers
        fig, axes = None, None
        if plot:
            sad_values, ordered_endmembers, ordered_abundances, fig, axes = (
                self._plot_endmembers(endmembers, endmembers_gt, abundances)
            )
        else:
            match_dict, sad_values_list, avg_sad = self._order_endmembers(
                endmembers, endmembers_gt
            )

            # Order endmembers and abundances
            ordered_endmembers_list: List[torch.Tensor] = []
            ordered_abundances_list: List[torch.Tensor] = []

            for i in range(self.num_endmembers):
                if i in match_dict:
                    ordered_endmembers_list.append(endmembers[match_dict[i]])
                    ordered_abundances_list.append(abundances[match_dict[i]])
                else:
                    # If no match, use the first remaining endmember
                    for j in range(self.num_endmembers):
                        if j not in match_dict.values():
                            ordered_endmembers_list.append(endmembers[j])
                            ordered_abundances_list.append(abundances[j])
                            match_dict[i] = j
                            break

            ordered_endmembers = torch.stack(ordered_endmembers_list)
            ordered_abundances = torch.stack(ordered_abundances_list)

            # Compute SAD for ordered endmembers
            sad_values_tensor_list: List[torch.Tensor] = []
            for i in range(self.num_endmembers):
                sad = self._compute_sad(ordered_endmembers[i], endmembers_gt[i])
                sad_values_tensor_list.append(sad)

            sad_values = torch.stack(sad_values_tensor_list)
            sad_values = torch.cat([sad_values, avg_sad.unsqueeze(0)])

        # Compute MSE metrics if ground truth abundances are provided
        if abundances_gt is not None:
            mse_values = self._compute_mse(abundances_gt, ordered_abundances)
        else:
            # If no ground truth abundances, set MSE to zeros
            mse_values = torch.zeros(self.num_endmembers + 1, device=endmembers.device)

        # Update metrics
        self._update_metrics(sad_values, mse_values)

        # Increment call count
        self.call_count += 1

        # Compute and return current metrics
        metrics = self._compute_metrics()

        if plot:
            assert fig is not None and axes is not None, (
                "Figure and axes should not be None when plot=True"
            )
            return metrics, fig, axes
        else:
            return metrics

    def reset(self) -> None:
        """
        Reset all metrics
        """
        # Reset SAD metrics
        for metric in self.sad_metrics.values():
            metric.reset()

        # Reset MSE metrics
        for metric in self.mse_metrics.values():
            metric.reset()

        # Reset call count
        self.call_count = 0

        # Reset average values
        self.sad_avg = None
        self.mse_avg = None

    def get_results(self) -> Dict[str, Dict[str, float]]:
        """
        Get current metric results

        Returns:
            Dict[str, Dict[str, float]]: Dictionary containing current SAD and MSE values
        """
        return self._compute_metrics()

    def save_results(self, filepath: str) -> None:
        """
        Save current metric results to a .npz file

        Args:
            filepath (str): Path to save the results
        """
        # Get current metrics
        metrics = self.get_results()

        # Prepare data for saving
        save_data: Dict[str, Any] = {
            "sad_avg": metrics["sad"]["sad_avg"],
            "mse_avg": metrics["mse"]["mse_avg"],
            "call_count": self.call_count,
        }

        # Add individual endmember metrics
        num_endmembers = cast(int, self.num_endmembers)
        for i in range(num_endmembers):
            save_data[f"sad_{i}"] = metrics["sad"][f"sad_{i}"]
            save_data[f"mse_{i}"] = metrics["mse"][f"mse_{i}"]

        # Save to .npz file
        # sio.savemat(filepath, save_data)
        np.savez(filepath, **save_data)

    def __str__(self) -> str:
        """
        String representation of the metrics

        Returns:
            str: String representation
        """
        if self.sad_avg is None or self.mse_avg is None:
            return "UnmixingMetrics (no data yet)"

        return (
            f"UnmixingMetrics (calls={self.call_count}): "
            f"SAD_avg={self.sad_avg:.4f}, MSE_avg={self.mse_avg:.6f}"
        )


# Alias
endmembers_visualize = UnmixingMetrics._plot_endmembers
