import importlib.util
import os
import sys
from typing import Any

import numpy as np
import torch
from scipy import io as sio


def _load_module_from_path(path: str, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None:
        raise ImportError(f"Could not load spec from {path}")
    mod = importlib.util.module_from_spec(spec)
    if spec.loader is not None:
        spec.loader.exec_module(mod)  # type: ignore
    return mod


def test_unmixing_metrics_agree():
    # Prepare paths to modules
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "stage2"))
    func_path = os.path.join(base, "HyperSIGMA", "HyperspectralUnmixing", "func.py")
    basic_path = os.path.join(base, "unmixing", "metrics", "basic.py")

    func = _load_module_from_path(func_path, "func_mod")
    basic = _load_module_from_path(basic_path, "basic_mod")

    # Fixed RNG for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Small synthetic example
    num_em = 4
    bands = 128
    H, W = 256, 256

    # Ground truth endmembers and abundances
    endmembers_gt = np.random.rand(num_em, bands).astype(float)
    abundances_gt = np.random.rand(num_em, H, W).astype(float)

    # Create predicted endmembers as a permutation of GT plus small noise
    perm = np.random.permutation(num_em)
    endmembers_pred = endmembers_gt[perm].copy() + 0.1 * np.random.randn(num_em, bands)
    abundances_pred = abundances_gt[perm].copy() + 0.2 * np.random.randn(num_em, H, W)

    # Compute reference results using func (numpy)
    # Use order_endmembers to get mapping, then compute ordered arrays like plotEndmembersAndGT
    mapping, sad_list, avg_sad = func.order_endmembers(endmembers_pred.copy(), endmembers_gt.copy())

    # Build ordered predicted endmembers/abundances following func.plotEndmembersAndGT
    endmember_sordered = np.array([endmembers_pred[mapping[i]] for i in range(num_em)])
    abundance_sordered = np.array([abundances_pred[mapping[i]] for i in range(num_em)])

    # Compute SAD_ordered (per-endmember) consistent with plotEndmembersAndGT
    SAD_ordered = np.array([func.numpy_SAD(endmember_sordered[i], endmembers_gt[i]) for i in range(num_em)])
    SAD_ordered = np.append(SAD_ordered, avg_sad)

    # Compute MSE reference
    mse_ref = func.alter_MSE(abundances_gt, abundance_sordered)

    # Compute results using UnmixingMetrics (torch)
    um = basic.UnmixingMetrics()
    # call expects (endmembers, endmembers_gt, abundances, abundances_gt)
    res = um(endmembers_pred, endmembers_gt, abundances_pred, abundances_gt, plot=False)

    # Extract SAD vector from UnmixingMetrics result in same order: sad_0..sad_{n-1}, sad_avg
    sad_dict = res["sad"]
    sad_vec = np.array([sad_dict[f"sad_{i}"] for i in range(num_em)] + [sad_dict["sad_avg"]])

    # Extract MSE vector
    mse_dict = res["mse"]
    mse_vec = np.array([mse_dict[f"mse_{i}"] for i in range(num_em)] + [mse_dict["mse_avg"]])

    # Compare numeric equality (allow small tolerance)
    # Allow small numerical differences between numpy and torch implementations
    assert np.allclose(SAD_ordered, sad_vec, atol=1e-4), f"SAD mismatch: ref={SAD_ordered}, um={sad_vec}"
    assert np.allclose(mse_ref, mse_vec, atol=1e-5), f"MSE mismatch: ref={mse_ref}, um={mse_vec}"

    # Also check that ordering mapping is consistent: mapping from func should equal internal mapping
    # We can re-run func.order_endmembers and basic._order_endmembers and compare keys -> predicted idx
    # (basic._order_endmembers is internal; call via instance)
    mapping_basic, _, _ = um._order_endmembers(
        torch.tensor(endmembers_pred, dtype=torch.float32),
        torch.tensor(endmembers_gt, dtype=torch.float32),
    )
    # Convert func mapping dict to same format: func returns dict[gt]=pred
    assert mapping_basic == mapping, f"Mapping mismatch: func={mapping}, basic={mapping_basic}"


def test_unmixing_metrics_with_plot():
    """Test unmixing metrics with real data and plotting functionality."""
    # Prepare paths to modules
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "stage2"))
    basic_path = os.path.join(base, "unmixing", "metrics", "basic.py")
    basic = _load_module_from_path(basic_path, "basic_mod")

    # Load real data
    data_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "data",
        "Downstreams",
        "UrbanUnmixing",
        "Urban_188_em4_init.mat",
    )

    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return

    # Load the .mat file
    dataset = sio.loadmat(data_path)
    endmembers_gt = dataset["endmember"]  # [162, 4]
    abundances_gt = dataset["abundance"]  # [4, 307, 307]

    # Transpose to match expected format [num_em, bands]
    endmembers_gt = endmembers_gt.T  # [4, 162]

    print(f"Loaded data - Endmembers shape: {endmembers_gt.shape}, Abundances shape: {abundances_gt.shape}")

    # Create synthetic predictions with some noise
    np.random.seed(42)
    num_em, bands = endmembers_gt.shape
    H, W = abundances_gt.shape[1], abundances_gt.shape[2]

    # Add noise to create predicted endmembers
    endmembers_pred = endmembers_gt.copy() + 0.05 * np.random.randn(num_em, bands)

    # Create predicted abundances with noise
    abundances_pred = abundances_gt.copy() + 0.02 * np.random.randn(num_em, H, W)

    # Ensure abundances are non-negative
    abundances_pred = np.clip(abundances_pred, 0, None)

    # Normalize abundances to sum to 1 for each pixel
    abundances_pred_sum = np.sum(abundances_pred, axis=0, keepdims=True)
    # Avoid division by zero
    abundances_pred_sum = np.where(abundances_pred_sum == 0, 1, abundances_pred_sum)
    abundances_pred = abundances_pred / abundances_pred_sum

    print(f"Predicted endmembers shape: {endmembers_pred.shape}")
    print(f"Predicted abundances shape: {abundances_pred.shape}")

    # Compute results using UnmixingMetrics with plotting
    um = basic.UnmixingMetrics()
    res, fig, axes = um(endmembers_pred, endmembers_gt, abundances_pred, abundances_gt, plot=True)

    # Print results
    print("SAD Results:")
    for i in range(num_em):
        print(f"  Endmember {i}: {res['sad'][f'sad_{i}']:.6f}")
    print(f"  Average SAD: {res['sad']['sad_avg']:.6f}")

    print("MSE Results:")
    for i in range(num_em):
        print(f"  Endmember {i}: {res['mse'][f'mse_{i}']:.6f}")
    print(f"  Average MSE: {res['mse']['mse_avg']:.6f}")

    # Save the plot to a file
    import matplotlib.pyplot as plt

    output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "unmixing_metrics_plot.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    test_unmixing_metrics_agree()
    print("Numerical consistency test passed!")

    test_unmixing_metrics_with_plot()
    print("Plot test completed!")
