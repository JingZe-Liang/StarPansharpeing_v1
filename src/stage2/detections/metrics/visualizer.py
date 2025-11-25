"""
Anomaly Detection Visualization Tools

This module provides comprehensive visualization tools for hyperspectral anomaly detection results,
including ROC curves, heatmaps, and box plots. The implementation is abstracted from GT-HAD
scripts with improved Pythonic interfaces and better flexibility.

Key features:
- ROC curve plotting with multiple algorithms comparison
- Heatmap visualization for anomaly detection results
- Box plot for anomaly vs background distribution comparison
- Support for batch processing and multiple datasets
- Customizable styling and output formats
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import auc, roc_curve


class AnomalyDetectionVisualizer:
    """
    Comprehensive visualization tools for hyperspectral anomaly detection.

    This class provides methods to create various types of plots commonly used
    in anomaly detection evaluation, abstracted from GT-HAD scripts with
    improved interfaces and flexibility.

    Example:
        >>> visualizer = AnomalyDetectionVisualizer()
        >>> # Plot ROC curves
        >>> visualizer.plot_roc_curves(results_dict, save_path="roc.pdf")
        >>> # Plot heatmap
        >>> visualizer.plot_heatmap(anomaly_map, save_path="heatmap.pdf")
        >>> # Plot boxplot
        >>> visualizer.plot_boxplot(scores, labels, save_path="boxplot.pdf")
    """

    def __init__(
        self,
        figsize: Tuple[int, int] = (7, 7),
        dpi: int = 300,
        style: str = "seaborn-v0_8",
        font_family: str = "Times New Roman",
    ):
        """
        Initialize the visualizer with default plotting parameters.

        Args:
            figsize: Default figure size for plots
            dpi: Resolution for saved figures
            style: Matplotlib style to use
            font_family: Default font family for text
        """
        self.figsize = figsize
        self.dpi = dpi
        self.style = style
        self.font_family = font_family

        # Set default style
        try:
            plt.style.use(style)
        except:
            # Fallback to default style if specified style not available
            pass

        # Set default font
        plt.rcParams["font.family"] = font_family

    def plot_roc_curves(
        self,
        results_dict: Dict[str, Dict],
        save_path: str,
        title: Optional[str] = None,
        colors: Optional[List[str]] = None,
        linewidth: float = 3.5,
        show_auc: bool = True,
        xlim: Tuple[float, float] = (1e-3, 1.0),
        ylim: Tuple[float, float] = (0.0, 1.05),
        xlabel: str = "False alarm rate",
        ylabel: str = "Probability of detection",
        fontsize: int = 15,
        ticksize: int = 13,
        legend_fontsize: int = 12,
    ) -> None:
        """
        Plot ROC curves for multiple algorithms comparison.

        Args:
            results_dict: Dictionary containing ROC data for each algorithm
                         Format: {algorithm_name: {"fpr": array, "tpr": array, "thresholds": array}}
            save_path: Path to save the plot
            title: Optional title for the plot
            colors: List of colors for each algorithm
            linewidth: Line width for ROC curves
            show_auc: Whether to show AUC values in legend
            xlim: X-axis limits (log scale)
            ylim: Y-axis limits
            xlabel: X-axis label
            ylabel: Y-axis label
            fontsize: Font size for axis labels
            ticksize: Font size for tick labels
            legend_fontsize: Font size for legend
        """
        # Default colors from GT-HAD
        if colors is None:
            colors = [
                "b",
                "c",
                "g",
                "lawngreen",
                "k",
                "m",
                "pink",
                "slategray",
                "orange",
                "y",
                "r",
            ]

        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, constrained_layout=True)

        # Set log scale for x-axis
        ax.set_xscale("log", base=10)
        ax.grid(False)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # Set labels
        ax.set_xlabel(xlabel, fontsize=fontsize, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=fontsize, fontweight="bold")
        ax.tick_params(axis="both", which="major", labelsize=ticksize)

        # Plot ROC curves for each algorithm
        for idx, (algorithm, data) in enumerate(results_dict.items()):
            fpr = data["fpr"]
            tpr = data["tpr"]

            # Handle different array shapes
            if len(fpr.shape) == 2 and fpr.shape[0] == 1:
                fpr = fpr[0]
                tpr = tpr[0]

            # Calculate AUC
            roc_auc = auc(fpr, tpr)

            # Choose color
            color = colors[idx % len(colors)]

            # Plot ROC curve
            label = f"{algorithm}" if not show_auc else f"{algorithm}: {roc_auc:.4f}"
            ax.semilogx(fpr, tpr, color=color, lw=linewidth, label=label)

            print(f"{algorithm}: {roc_auc:.4f}")

        # Add legend
        ax.legend(loc="lower right", fontsize=legend_fontsize, prop={"weight": "bold"})

        # Add title if provided
        if title:
            ax.set_title(title, fontsize=fontsize, fontweight="bold")

        # Create directory if it doesn't exist
        save_dir = os.path.dirname(save_path)
        if save_dir:  # Only create if directory path is not empty
            os.makedirs(save_dir, exist_ok=True)

        # Save figure
        plt.savefig(save_path, format="pdf", bbox_inches="tight", pad_inches=0.0, dpi=self.dpi)
        plt.close()

        print(f"ROC curves saved to: {save_path}")

    def plot_heatmap(
        self,
        anomaly_map: Union[np.ndarray, torch.Tensor],
        save_path: str,
        cmap: str = "turbo",
        vmin: float = 0.0,
        vmax: float = 1.0,
        figsize: Optional[Tuple[int, int]] = None,
        show_colorbar: bool = False,
        show_annotations: bool = False,
        show_ticks: bool = False,
    ) -> None:
        """
        Plot anomaly detection results as heatmap.

        Args:
            anomaly_map: 2D anomaly detection scores
            save_path: Path to save the plot
            cmap: Colormap for heatmap
            vmin: Minimum value for colormap
            vmax: Maximum value for colormap
            figsize: Figure size (overrides default)
            show_colorbar: Whether to show colorbar
            show_annotations: Whether to show value annotations
            show_ticks: Whether to show axis ticks
        """
        # Convert to numpy if tensor
        if isinstance(anomaly_map, torch.Tensor):
            anomaly_map = anomaly_map.detach().cpu().numpy()

        # Normalize to [0, 1] if not already
        if vmin is None or vmax is None:
            anomaly_map = anomaly_map - anomaly_map.min()
            anomaly_map = anomaly_map / anomaly_map.max()
        else:
            anomaly_map = np.clip(anomaly_map, vmin, vmax)
            anomaly_map = (anomaly_map - vmin) / (vmax - vmin)

        # Use provided figsize or default
        fig_size = figsize or self.figsize

        # Create figure
        plt.figure(figsize=fig_size)

        # Plot heatmap
        sns.heatmap(
            anomaly_map,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            annot=show_annotations,
            xticklabels=show_ticks,
            yticklabels=show_ticks,
            cbar=show_colorbar,
            linewidths=0.0,
            rasterized=True,
        )

        # Create directory if it doesn't exist
        save_dir = os.path.dirname(save_path)
        if save_dir:  # Only create if directory path is not empty
            os.makedirs(save_dir, exist_ok=True)

        # Save figure
        plt.savefig(save_path, format="pdf", bbox_inches="tight", pad_inches=0.0, dpi=self.dpi)
        plt.close()

        print(f"Heatmap saved to: {save_path}")

    def plot_boxplot(
        self,
        anomaly_scores: Union[np.ndarray, torch.Tensor],
        ground_truth: Union[np.ndarray, torch.Tensor],
        save_path: str,
        method_name: str = "Method",
        ylim: Tuple[float, float] = (0.0, 1.19),
        figsize: Optional[Tuple[int, int]] = None,
        ylabel: str = "Normalized detection statistic range",
        ticksize: int = 12,
        fontsize: int = 15,
        show_fliers: bool = False,
        box_width: float = 0.3,
    ) -> None:
        """
        Plot box plot comparing anomaly vs background distributions.

        Args:
            anomaly_scores: 2D anomaly detection scores
            ground_truth: 2D ground truth labels (binary)
            save_path: Path to save the plot
            method_name: Name of the method for labeling
            ylim: Y-axis limits
            figsize: Figure size (overrides default)
            ylabel: Y-axis label
            ticksize: Font size for tick labels
            fontsize: Font size for axis labels
            show_fliers: Whether to show outlier points
            box_width: Width of boxes in plot
        """
        # Convert to numpy if tensors
        if isinstance(anomaly_scores, torch.Tensor):
            anomaly_scores = anomaly_scores.detach().cpu().numpy()
        if isinstance(ground_truth, torch.Tensor):
            ground_truth = ground_truth.detach().cpu().numpy()

        # Normalize scores to [0, 1]
        anomaly_scores = anomaly_scores - anomaly_scores.min()
        anomaly_scores = anomaly_scores / anomaly_scores.max()

        # Separate anomaly and background regions
        scores_flat = anomaly_scores.flatten()
        gt_flat = ground_truth.flatten()

        # Remove zeros (background) for anomaly region
        anomaly_scores_masked = scores_flat.copy()
        anomaly_scores_masked[gt_flat == 0] = 0

        # Remove anomalies for background region
        background_scores = scores_flat.copy()
        background_scores[gt_flat != 0] = 0

        # Remove zero values
        anomaly_scores_filtered = anomaly_scores_masked[anomaly_scores_masked != 0]
        background_scores_filtered = background_scores[background_scores != 0]

        # Prepare data for plotting
        data = [anomaly_scores_filtered, background_scores_filtered]

        # Use provided figsize or default
        fig_size = figsize or self.figsize

        # Create figure
        fig, ax = plt.subplots(figsize=fig_size)
        ax.grid(False)
        ax.set_ylim(ylim)

        # Set labels
        ax.set_ylabel(ylabel, fontsize=fontsize, fontweight="bold")
        ax.tick_params(axis="both", which="major", labelsize=ticksize)

        # Plot boxplot
        box_positions = [1.0, 1.4]
        color_list = [(1, 0, 0), (0, 0, 1)]  # Red for anomaly, Blue for background

        bp = ax.boxplot(
            data,
            widths=box_width,
            patch_artist=True,
            showfliers=show_fliers,
            positions=box_positions,
            medianprops={"color": "black"},
            whiskerprops={"linestyle": "--"},
        )

        # Customize boxes
        for patch, color in zip(bp["boxes"], color_list):
            patch.set_facecolor(color)
            patch.set(linewidth=0.75)

        # Customize whiskers
        color_list_double = [(1, 0, 0), (1, 0, 0), (0, 0, 1), (0, 0, 1)]
        for patch, color in zip(bp["whiskers"], color_list_double):
            patch.set(color=color, linewidth=2.5)

        # Customize caps
        for patch, color in zip(bp["caps"], color_list_double):
            patch.set(color=color, linewidth=2.5)

        # Customize medians
        for patch, color in zip(bp["medians"], color_list):
            patch.set(color=color, linewidth=0.1)

        # Set x-axis labels
        plt.xticks([1.2], [method_name], fontsize=ticksize, fontweight="bold")

        # Add legend
        labels = ["Anomaly", "Background"]
        plt.legend(
            bp["boxes"],
            labels,
            loc="upper right",
            fontsize=ticksize,
            prop={"weight": "bold"},
        )

        # Create directory if it doesn't exist
        save_dir = os.path.dirname(save_path)
        if save_dir:  # Only create if directory path is not empty
            os.makedirs(save_dir, exist_ok=True)

        # Save figure
        plt.savefig(save_path, format="pdf", bbox_inches="tight", pad_inches=0.0, dpi=self.dpi)
        plt.close()

        print(f"Box plot saved to: {save_path}")

    def plot_multiple_boxplots(
        self,
        results_dict: Dict[str, Dict],
        ground_truth: Union[np.ndarray, torch.Tensor],
        save_path: str,
        ylim: Tuple[float, float] = (0.0, 1.19),
        figsize: Optional[Tuple[int, int]] = None,
        ylabel: str = "Normalized detection statistic range",
        ticksize: int = 11,
        fontsize: int = 15,
        rotation: int = 18,
        show_fliers: bool = False,
    ) -> None:
        """
        Plot box plots for multiple methods comparison.

        Args:
            results_dict: Dictionary containing anomaly scores for each method
                         Format: {method_name: {"scores": array}}
            ground_truth: 2D ground truth labels (binary)
            save_path: Path to save the plot
            ylim: Y-axis limits
            figsize: Figure size (overrides default)
            ylabel: Y-axis label
            ticksize: Font size for tick labels
            fontsize: Font size for axis labels
            rotation: Rotation angle for x-axis labels
            show_fliers: Whether to show outlier points
        """
        # Convert to numpy if tensor
        if isinstance(ground_truth, torch.Tensor):
            ground_truth = ground_truth.detach().cpu().numpy()

        # Prepare data for all methods
        all_data = []
        method_names = list(results_dict.keys())

        for method_name in method_names:
            scores = results_dict[method_name]["scores"]

            # Convert to numpy if tensor
            if isinstance(scores, torch.Tensor):
                scores = scores.detach().cpu().numpy()

            # Normalize scores
            scores = scores - scores.min()
            scores = scores / scores.max()

            # Separate anomaly and background
            scores_flat = scores.flatten()
            gt_flat = ground_truth.flatten()

            # Anomaly region
            anomaly_scores = scores_flat.copy()
            anomaly_scores[gt_flat == 0] = 0
            anomaly_filtered = anomaly_scores[anomaly_scores != 0]

            # Background region
            background_scores = scores_flat.copy()
            background_scores[gt_flat != 0] = 0
            background_filtered = background_scores[background_scores != 0]

            all_data.append(anomaly_filtered)
            all_data.append(background_filtered)

        # Use provided figsize or default
        fig_size = figsize or self.figsize

        # Create figure
        fig, ax = plt.subplots(figsize=fig_size)
        ax.grid(False)
        ax.set_ylim(ylim)

        # Set labels
        ax.set_ylabel(ylabel, fontsize=fontsize, fontweight="bold")
        ax.tick_params(axis="both", which="major", labelsize=ticksize)

        # Calculate box positions
        num_methods = len(method_names)
        box_position = []
        method_position = []
        pos = 0.0

        for i in range(num_methods * 2):
            if i % 2 == 0:
                pos += 1.0
                method_position.append(pos + 0.2)
            else:
                pos += 0.4
            box_position.append(pos)

        # Colors for alternating pattern
        color_list = [(1, 0, 0), (0, 0, 1)] * num_methods

        # Plot boxplot
        bp = ax.boxplot(
            all_data,
            widths=0.3,
            patch_artist=True,
            showfliers=show_fliers,
            positions=box_position,
            medianprops={"color": "black"},
            whiskerprops={"linestyle": "--"},
        )

        # Customize boxes
        for patch, color in zip(bp["boxes"], color_list):
            patch.set_facecolor(color)
            patch.set(linewidth=0.75)

        # Customize whiskers
        color_list_double = [(1, 0, 0), (1, 0, 0), (0, 0, 1), (0, 0, 1)] * num_methods
        for patch, color in zip(bp["whiskers"], color_list_double):
            patch.set(color=color, linewidth=2.5)

        # Customize caps
        for patch, color in zip(bp["caps"], color_list_double):
            patch.set(color=color, linewidth=2.5)

        # Customize medians
        for patch, color in zip(bp["medians"], color_list):
            patch.set(color=color, linewidth=0.1)

        # Set x-axis labels
        plt.xticks(
            method_position,
            method_names,
            rotation=rotation,
            fontsize=ticksize,
            fontweight="bold",
        )

        # Add legend
        labels = ["Anomaly", "Background"]
        plt.legend(
            bp["boxes"],
            labels,
            loc="upper right",
            fontsize=ticksize,
            prop={"weight": "bold"},
        )

        # Create directory if it doesn't exist
        save_dir = os.path.dirname(save_path)
        if save_dir:  # Only create if directory path is not empty
            os.makedirs(save_dir, exist_ok=True)

        # Save figure
        plt.savefig(save_path, format="pdf", bbox_inches="tight", pad_inches=0.0, dpi=self.dpi)
        plt.close()

        print(f"Multiple box plots saved to: {save_path}")


def test_visualizer():
    """Test function to demonstrate visualization capabilities."""
    print("Testing AnomalyDetectionVisualizer...")

    # Create visualizer
    visualizer = AnomalyDetectionVisualizer()

    # Create sample data
    np.random.seed(42)

    # Test ROC curves
    print("\n1. Testing ROC curves...")
    results_dict = {}
    methods = ["Method A", "Method B", "Method C"]

    for method in methods:
        # Generate synthetic ROC curve data
        fpr = np.linspace(0, 1, 100)
        tpr = 1 - np.exp(-5 * fpr)  # Exponential-like curve
        # Add some noise
        tpr += np.random.normal(0, 0.02, len(tpr))
        tpr = np.clip(tpr, 0, 1)

        results_dict[method] = {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": np.linspace(1, 0, 100),
        }

    visualizer.plot_roc_curves(results_dict, save_path="test_roc_curves.pdf", title="ROC Curves Comparison")

    # Test heatmap
    print("\n2. Testing heatmap...")
    # Create synthetic anomaly map
    height, width = 64, 64
    anomaly_map = np.random.rand(height, width)

    # Add some anomaly regions
    anomaly_map[20:30, 20:30] = 0.8 + 0.2 * np.random.rand(10, 10)
    anomaly_map[40:50, 40:50] = 0.9 + 0.1 * np.random.rand(10, 10)

    visualizer.plot_heatmap(anomaly_map, save_path="test_heatmap.pdf", cmap="turbo")

    # Test boxplot
    print("\n3. Testing boxplot...")
    # Create synthetic data
    scores = np.random.rand(height, width)
    gt = np.zeros((height, width))

    # Add anomaly regions
    gt[20:30, 20:30] = 1
    gt[40:50, 40:50] = 1

    # Make anomaly regions have higher scores
    scores[gt == 1] = 0.7 + 0.3 * np.random.rand(np.sum(gt == 1))
    scores[gt == 0] = 0.1 + 0.3 * np.random.rand(np.sum(gt == 0))

    visualizer.plot_boxplot(scores, gt, save_path="test_boxplot.pdf", method_name="Test Method")

    # Test multiple boxplots
    print("\n4. Testing multiple boxplots...")
    multi_results = {}
    for method in methods:
        method_scores = np.random.rand(height, width)
        method_scores[gt == 1] = 0.6 + 0.4 * np.random.rand(np.sum(gt == 1))
        method_scores[gt == 0] = 0.1 + 0.3 * np.random.rand(np.sum(gt == 0))
        multi_results[method] = {"scores": method_scores}

    visualizer.plot_multiple_boxplots(multi_results, gt, save_path="test_multiple_boxplots.pdf")

    print("\nAll visualization tests completed!")
    print("Generated files:")
    print("  - test_roc_curves.pdf")
    print("  - test_heatmap.pdf")
    print("  - test_boxplot.pdf")
    print("  - test_multiple_boxplots.pdf")


if __name__ == "__main__":
    test_visualizer()
