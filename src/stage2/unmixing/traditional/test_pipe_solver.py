"""
Test pipe solver with DC hyperspectral dataset

This script tests the vca_fclsu_nnls_solver with real hyperspectral data
and validates the reconstruction quality.
"""

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch

from src.stage2.unmixing.traditional.pipe import vca_fclsu_nnls_solver


def load_hyperspectral_data(file_path: str) -> Tuple[torch.Tensor, dict]:
    """
    Load hyperspectral data from tiff file

    Parameters
    ----------
    file_path : str
        Path to the hyperspectral image file

    Returns
    -------
    Tuple[torch.Tensor, dict]
        hyperspectral_image: torch.Tensor[bands, height, width]
        info: dictionary with data information
    """
    # Load data
    data = tifffile.imread(file_path)

    print(f"原始数据形状: {data.shape}")
    print(f"数据类型: {data.dtype}")
    print(f"数值范围: [{data.min():.6f}, {data.max():.6f}]")

    # Assume format is (height, width, bands)
    height, width, bands = data.shape

    # Convert to torch tensor and rearrange to (bands, height, width)
    hyper_img = torch.from_numpy(data).permute(2, 0, 1).float()

    info = {
        "height": height,
        "width": width,
        "bands": bands,
        "original_shape": data.shape,
        "value_range": (data.min(), data.max()),
    }

    return hyper_img, info


def test_solver_with_dc_data():
    """
    Test the pipe solver with DC hyperspectral data
    """
    print("=" * 60)
    print("Testing Pipe Solver with DC Hyperspectral Data")
    print("=" * 60)

    # Load data
    file_path = "data/Downstreams/UrbanUnmixing/splits/DC/DC.img.tiff"
    hyper_img, info = load_hyperspectral_data(file_path)

    print(f"\n数据信息:")
    print(f"  高度: {info['height']}")
    print(f"  宽度: {info['width']}")
    print(f"  波段数: {info['bands']}")
    print(f"  数值范围: [{info['value_range'][0]:.6f}, {info['value_range'][1]:.6f}]")

    # Test with different number of endmembers
    n_endmembers_list = [3, 4, 5, 6]

    for n_endmembers in n_endmembers_list:
        print(f"\n{'=' * 40}")
        print(f"Testing with {n_endmembers} endmembers")
        print(f"{'=' * 40}")

        try:
            # Use the pipe solver
            endmembers, abundances = vca_fclsu_nnls_solver(
                hyper_img=hyper_img,
                n_endmembers=n_endmembers,
                fclsu_solver_kwargs={
                    "backend": "scipy",
                    "max_iter": 500,
                    "lr": 0.01,
                },
            )

            print(f"端元形状: {endmembers.shape}")
            print(f"丰度形状: {abundances.shape}")
            print(f"端元数值范围: [{endmembers.min():.6f}, {endmembers.max():.6f}]")
            print(f"丰度数值范围: [{abundances.min():.6f}, {abundances.max():.6f}]")

            # Check physical constraints
            print(f"\n物理约束检查:")
            print(f"  丰度非负性: {(abundances >= 0).all().item()}")
            print(f"  丰度和为1的偏差: {(abundances.sum(dim=0) - 1).abs().mean().item():.6f}")

            # Reconstruction validation
            print(f"\n重建验证:")

            # Reshape for matrix multiplication
            h, w = hyper_img.shape[-2:]
            img_1d = hyper_img.permute(1, 2, 0).reshape(h * w, -1)  # [h*w, bands]
            abundances_1d = abundances.reshape(n_endmembers, h * w).T  # [h*w, endmembers]

            # Reconstruct image
            reconstructed = abundances_1d @ endmembers.T  # [h*w, bands]

            # Calculate reconstruction error
            mse = torch.mean((img_1d - reconstructed) ** 2)
            rmse = torch.sqrt(mse)
            mae = torch.mean(torch.abs(img_1d - reconstructed))

            # Calculate PSNR (Peak Signal-to-Noise Ratio)
            # PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)
            # For hyperspectral data, we use the maximum value in the original image
            max_pixel_value = torch.max(img_1d)
            if mse > 0:
                psnr = 20 * torch.log10(max_pixel_value) - 10 * torch.log10(mse)
            else:
                psnr = torch.tensor(float("inf"))  # Perfect reconstruction

            # Calculate relative error
            relative_error = torch.norm(img_1d - reconstructed) / torch.norm(img_1d)

            print(f"  均方误差 (MSE): {mse.item():.6f}")
            print(f"  均方根误差 (RMSE): {rmse.item():.6f}")
            print(f"  平均绝对误差 (MAE): {mae.item():.6f}")
            print(f"  峰值信噪比 (PSNR): {psnr.item():.6f} dB")
            print(f"  相对误差: {relative_error.item():.6f}")

            # Spectral Angle Mapper (SAM)
            def calculate_sam(y_true, y_pred):
                y_true_norm = torch.nn.functional.normalize(y_true, dim=1)
                y_pred_norm = torch.nn.functional.normalize(y_pred, dim=1)
                cos_sim = torch.sum(y_true_norm * y_pred_norm, dim=1)
                cos_sim = torch.clamp(cos_sim, -1, 1)
                sam = torch.arccos(cos_sim)
                return torch.mean(sam)

            sam = calculate_sam(img_1d, reconstructed)
            print(f"  光谱角距离 (SAM): {sam.item():.6f} radians")

            # Visualize results for the first endmember
            if n_endmembers == 4:  # Only visualize for one configuration
                visualize_results(hyper_img, endmembers, abundances, reconstructed)

        except Exception as e:
            print(f"Error with {n_endmembers} endmembers: {e}")
            continue


def visualize_results(original_img, endmembers, abundances, reconstructed):
    """
    Visualize the unmixing results
    """
    print(f"\n可视化结果...")

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original image (RGB approximation)
    if original_img.shape[0] >= 3:
        rgb_bands = [
            original_img.shape[0] // 4,
            original_img.shape[0] // 2,
            3 * original_img.shape[0] // 4,
        ]
        rgb_img = original_img[rgb_bands].permute(1, 2, 0)
        rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
        axes[0, 0].imshow(rgb_img)
        axes[0, 0].set_title("Original RGB")
        axes[0, 0].axis("off")

    # Reconstructed image (RGB approximation)
    if reconstructed.shape[1] >= 3:
        rgb_rec = reconstructed[:, rgb_bands].reshape(original_img.shape[1], original_img.shape[2], 3)
        rgb_rec = (rgb_rec - rgb_rec.min()) / (rgb_rec.max() - rgb_rec.min())
        axes[0, 1].imshow(rgb_rec)
        axes[0, 1].set_title("Reconstructed RGB")
        axes[0, 1].axis("off")

    # Error map
    error_map = torch.norm(
        original_img.permute(1, 2, 0) - reconstructed.reshape(original_img.shape[1], original_img.shape[2], -1),
        dim=2,
    )
    im = axes[0, 2].imshow(error_map, cmap="hot")
    axes[0, 2].set_title("Reconstruction Error")
    axes[0, 2].axis("off")
    plt.colorbar(im, ax=axes[0, 2])

    # Endmember spectra
    for i in range(min(endmembers.shape[0], 6)):
        axes[1, i].plot(endmembers[i].cpu().numpy())
        axes[1, i].set_title(f"Endmember {i + 1}")
        axes[1, i].set_xlabel("Band")
        axes[1, i].set_ylabel("Reflectance")

    plt.tight_layout()
    plt.savefig("unmixing_results.png", dpi=300, bbox_inches="tight")
    print(f"可视化结果已保存到 'unmixing_results.png'")
    plt.close()


def main():
    """
    Main function to run the test
    """
    try:
        test_solver_with_dc_data()
        print(f"\n{'=' * 60}")
        print("测试完成！")
        print(f"{'=' * 60}")
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
