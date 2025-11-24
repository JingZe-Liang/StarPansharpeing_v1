from typing import Any, List, Optional, Tuple, Union

import einops
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA as PCA_sk

from src.utilities.logging import log

"""
src.stage1.utilities.losses.repa.feature_pca
"""


def shape_to_1d(img_feat: torch.Tensor, pca_k: int):
    assert img_feat.ndim in (3, 4), "must be 1d vit or 2d cnn features"
    _shape = img_feat.shape
    if len(_shape) == 3:
        c = _shape[-1]  # [bs, l, c]
        data = einops.rearrange(img_feat, "bs l c -> (bs l) c")
        _back_kwargs = dict(pattern="(bs l) k -> bs l k", k=pca_k, bs=_shape[0])
    else:
        _shape[1]
        data = einops.rearrange(img_feat, "bs c h w -> (bs h w) c")
        _back_kwargs = dict(
            pattern="(bs h w) k -> bs k h w",
            k=pca_k,
            bs=_shape[0],
            h=_shape[2],
            w=_shape[3],
        )

    return data, _back_kwargs


def norm_pca_feats_per_channel(feat_pca):
    f_min = feat_pca.min(dim=1)[0]
    f_max = feat_pca.max(dim=1)[0]
    feat_pca = (feat_pca - f_min) / (f_max - f_min)
    return feat_pca


def feature_pca_sk(img_feat: torch.Tensor, pca_k: int = 3, norm_pca=True):
    """
    Perform PCA dimension reduction on image features using scikit-learn
    """

    data, _back_kwargs = shape_to_1d(img_feat, pca_k)

    # pca in numpy
    data = data.detach().cpu().numpy()
    pca = PCA_sk(n_components=pca_k, whiten=True)
    pca_data = pca.fit_transform(data)
    pca_data = einops.rearrange(pca_data, **_back_kwargs)
    pca_data = torch.as_tensor(pca_data).to(img_feat.device)

    if norm_pca:
        pca_data = norm_pca_feats_per_channel(pca_data)
    return pca_data


def feature_pca_cuml(img_feat: torch.Tensor, pca_k: int = 3, norm_pca=True):
    from cuml.decomposition import PCA as cuML_PCA

    assert img_feat.is_cuda, "img_feat must be on GPU"
    data, _back_kwargs = shape_to_1d(img_feat, pca_k)
    pca = cuML_PCA(n_components=pca_k)

    projected_data = pca.fit_transform(data)

    img_feats_reduced_flat = torch.as_tensor(
        projected_data, device=data.device, dtype=img_feat.dtype
    )

    img_feats_reduced = einops.rearrange(img_feats_reduced_flat, **_back_kwargs)
    if norm_pca:
        img_feats_reduced = norm_pca_feats_per_channel(img_feats_reduced)

    log(f"Original shape: {img_feat.shape}")
    log(f"Reshaped for cuML PCA: {data.shape}")
    log(f"Projected data shape (cuML): {projected_data.shape}")
    log(f"Reduced features shape (PyTorch tensor): {img_feats_reduced.shape}")

    return img_feats_reduced


@torch.no_grad()
def feature_pca_torch(img_feat: torch.Tensor, pca_k: int, norm_pca=True):
    """
    Perform PCA dimension reduction on image features using PyTorch's SVD on CUDA.

    Handles 3D features [bs, l, c] (e.g., ViT patch tokens)
    or 4D features [bs, c, h, w] (e.g., CNN features).
    Reduces the feature dimension 'c' to 'pca_k'.

    Args:
        img_feat (torch.Tensor): Input features on CUDA. Shape [bs, l, c] or [bs, c, h, w].
            Ensure img_feat.is_cuda is True.
        pca_k (int): The target dimension after PCA. Must be <= original feature dimension.

    Returns:
        torch.Tensor: PCA reduced features on CUDA.
                      Shape [bs, l, pca_k] for 3D input,
                      Shape [bs, pca_k, h, w] for 4D input (reshaped back to channels first).
    """
    assert img_feat.ndim in (
        3,
        4,
    ), "Input feature tensor must be 3D ([bs, l, c]) or 4D ([bs, c, h, w])"
    # assert img_feat.is_cuda, "Input feature tensor must be on CUDA"
    # SVD usually works best with float types
    assert img_feat.dtype in (
        torch.float32,
        torch.float64,
    ), "Input feature tensor dtype must be float32 or float64 for SVD"

    _shape = img_feat.shape
    img_feat.device
    img_feat.dtype

    # Determine original feature dimension and reshape for PCA
    # PCA expects input in [N_samples, N_features] format.
    # We treat the last dimension (channel/feature dimension) as N_features,
    # and everything else (batch, spatial, sequence) as N_samples.
    if len(_shape) == 3:
        # Input is [bs, l, c] (e.g., ViT patch tokens)
        # N_samples = bs * l, N_features = c
        c = _shape[-1]
        data = einops.rearrange(img_feat, "bs l c -> (bs l) c")
        # Define pattern for reshaping back to [bs, l, pca_k]
        _back_kwargs = dict(
            pattern="(bs l) k -> bs l k", k=pca_k, bs=_shape[0], l=_shape[1]
        )
    else:  # len(_shape) == 4
        # Input is [bs, c, h, w] (e.g., CNN features)
        # We want to reduce the channel dimension c.
        # N_samples = bs * h * w, N_features = c
        c = _shape[1]  # Channel dimension is the second dimension
        # Reshape to treat spatial locations as samples and channels as features
        data = einops.rearrange(img_feat, "bs c h w -> (bs h w) c")
        # Define pattern for reshaping back from [(bs h w), k] to [bs, k, h, w]
        _back_kwargs = dict(
            pattern="(bs h w) k -> bs k h w",  # Reshape to channels first
            k=pca_k,
            bs=_shape[0],
            h=_shape[2],
            w=_shape[3],
        )

    # Assert that the target dimension is achievable
    assert pca_k <= c, (
        f"Target dimension pca_k ({pca_k}) cannot be larger than original feature dimension c ({c})"
    )
    # Also assert that the number of samples is sufficient for SVD
    N_samples = data.shape[0]
    # SVD of A (m x n) where m >= n with full_matrices=False gives Vh (n x n).
    # If m < n, Vh is (m x n). The principal components are rows of Vh.
    # For PCA to work on the features (columns of A), we need enough samples (rows of A).
    # At least c samples are needed to find c components.

    # assert N_samples >= c, (
    #     f"Number of samples ({N_samples}) is less than feature dimension ({c}). PCA may not be meaningful or stable. Consider increasing batch size or input size."
    # )

    if N_samples > c:
        log(
            f"Number of samples ({N_samples}) is greater than feature dimension ({c}). PCA may be more stable with fewer samples.",
            warn_once=True,
        )

    # --- PyTorch Manual PCA Steps (on CUDA) ---

    # 1. Center the data
    # Compute mean across the sample dimension (dim=0)
    # Keep mean on the same device/dtype
    mean = torch.mean(data, dim=0, keepdim=True)  # shape [1, c]
    centered_data = data - mean  # shape [N_samples, c]

    # 2. Perform Singular Value Decomposition (SVD) on the centered data
    # centered_data has shape [N_samples, c].
    # SVD of A (m x n) is U (m x m or m x min(m,n)) @ diag(S) (min(m,n) x min(m,n)) @ Vh (n x n or min(m,n) x n).
    # With full_matrices=False and N_samples >= c, Vh has shape [c, c].
    # The rows of Vh are the principal components (eigenvectors of the covariance matrix of A.T @ A).
    # They are sorted by singular value in descending order.
    U, S, Vh = torch.linalg.svd(centered_data, full_matrices=False)

    # 3. Select the first pca_k principal components (rows of Vh) and transpose
    # The rows of Vh are the principal directions. To project data [N_samples, c],
    # we multiply by the transpose of the matrix whose columns are the principal directions.
    # This projection matrix is Vh[:pca_k].T.
    # Vh[:pca_k, :] selects the top pca_k rows, shape [pca_k, c].
    # Transposing gives shape [c, pca_k].
    principal_components = Vh[
        :pca_k, :
    ].T  # Take first pca_k rows of Vh, then transpose

    # 4. Project the centered data onto the principal components
    # centered_data [N_samples, c] @ principal_components [c, pca_k] -> projected_data [N_samples, pca_k]
    projected_data = centered_data @ principal_components

    # --- End PyTorch Manual PCA Steps ---

    # Reshape the projected data back to the original structure (with reduced dimension)
    # The shape is now [N_samples, pca_k] -> reshape according to _back_kwargs
    # projected_data is already a PyTorch tensor on the correct device/dtype
    img_feats_reduced = einops.rearrange(projected_data, **_back_kwargs)

    # Add print statements for verification (optional)
    # logger.debug(f"Original shape: {img_feat.shape}")
    # logger.debug(f"Reshaped for PCA: {data.shape}")
    # logger.debug(f"Centered data shape: {centered_data.shape}")
    # print(f"U shape: {U.shape}, S shape: {S.shape}, Vh shape: {Vh.shape}") # Optional detailed prints
    # logger.debug(f"Principal components matrix shape: {principal_components.shape}")
    # logger.debug(f"Projected data shape (flat): {projected_data.shape}")
    # logger.debug(f"Reduced features shape (reshaped): {img_feats_reduced.shape}")

    if norm_pca:
        img_feats_reduced = norm_pca_feats_per_channel(img_feats_reduced)

    return img_feats_reduced


def pca_list(
    image_feats_list: List[torch.Tensor],
    dim: int = 3,
    fit_pca: Optional[Union["TorchPCA", PCA_sk]] = None,
    use_torch_pca: bool = True,
    max_samples: Optional[int] = None,
) -> Tuple[List[torch.Tensor], Union["TorchPCA", PCA_sk]]:
    """
    Apply PCA dimensionality reduction to a list of image feature tensors.

    This function processes multiple image feature tensors by flattening them,
    fitting a PCA model (either using PyTorch implementation or scikit-learn),
    and then transforming each tensor to the reduced dimension. The function
    handles both 2D features and 4D feature maps.

    Parameters
    ----------
    image_feats_list : List[torch.Tensor]
        List of image feature tensors. Each tensor can be either:
        - 2D tensor: [num_features, feature_dim]
        - 4D tensor: [batch_size, channels, height, width]
    dim : int, optional
        Target dimension after PCA reduction. Default is 3.
    fit_pca : Optional[Union['TorchPCA', PCA_sk]], optional
        Pre-fitted PCA model to use for transformation. If None, a new
        PCA model will be fitted on the concatenated features. Default is None.
    use_torch_pca : bool, optional
        Whether to use PyTorch-based PCA implementation or scikit-learn.
        Default is True (use PyTorch implementation).
    max_samples : Optional[int], optional
        Maximum number of samples to use for PCA fitting. If provided and
        the number of samples exceeds this value, random subsampling will be
        applied. Default is None (use all samples).

    Returns
    -------
    Tuple[List[torch.Tensor], Union['TorchPCA', PCA_sk]]
        A tuple containing:
        - List of reduced feature tensors with the same structure as input
        - The fitted PCA model (either TorchPCA or sklearn.PCA instance)

    Notes
    -----
    - For 4D input tensors, spatial dimensions are preserved and only the
      channel dimension is reduced.
    - The output features are normalized to [0, 1] range per dimension.
    - When multiple tensors are provided with different spatial sizes,
      they will be interpolated to match the first tensor's spatial size.
    """
    device = image_feats_list[0].device

    def flatten(
        tensor: torch.Tensor, target_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Flatten feature tensor for PCA processing.

        Parameters
        ----------
        tensor : torch.Tensor
            Input feature tensor (2D or 4D)
        target_size : Optional[int], optional
            Target spatial size for interpolation. Default is None.

        Returns
        -------
        torch.Tensor
            Flattened tensor with shape [num_samples, feature_dim]
        """
        if len(tensor.shape) == 2:
            return tensor.detach().cpu()
        if target_size is not None and fit_pca is None:
            tensor = F.interpolate(tensor, (target_size, target_size), mode="bilinear")
        B, C, H, W = tensor.shape
        return (
            tensor.permute(1, 0, 2, 3)
            .reshape(C, B * H * W)
            .permute(1, 0)
            .detach()
            .cpu()
        )

    if len(image_feats_list) > 1 and fit_pca is None:
        if len(image_feats_list[0].shape) == 2:
            target_size = None
        else:
            target_size = image_feats_list[0].shape[2]
    else:
        target_size = None

    flattened_feats = []
    for feats in image_feats_list:
        flattened_feats.append(flatten(feats, target_size))
    x = torch.cat(flattened_feats, dim=0)

    # Subsample the data if max_samples is set and the number of samples exceeds max_samples
    if max_samples is not None and x.shape[0] > max_samples:
        indices = torch.randperm(x.shape[0])[:max_samples]
        x = x[indices]

    if fit_pca is None:
        if use_torch_pca:
            fit_pca = TorchPCA(n_components=dim).fit(x)
        else:
            fit_pca = PCA_sk(n_components=dim).fit(x)

    reduced_feats = []
    for feats in image_feats_list:
        x_red = fit_pca.transform(flatten(feats))
        if isinstance(x_red, np.ndarray):
            x_red = torch.from_numpy(x_red)
        x_red -= x_red.min(dim=0, keepdim=True).values
        x_red /= x_red.max(dim=0, keepdim=True).values
        if len(feats.shape) == 2:
            reduced_feats.append(x_red)  # 1D
        else:
            B, C, H, W = feats.shape
            reduced_feats.append(
                x_red.reshape(B, H, W, dim).permute(0, 3, 1, 2).to(device)
            )  # 3D

    return reduced_feats, fit_pca


class TorchPCA:
    """
    PyTorch-based PCA implementation using torch.pca_lowrank.

    This class provides a scikit-learn compatible interface for PCA
    dimensionality reduction using PyTorch operations.
    """

    def __init__(self, n_components: int) -> None:
        """
        Initialize TorchPCA with specified number of components.

        Parameters
        ----------
        n_components : int
            Number of principal components to keep.
        """
        self.n_components: int = n_components
        self.mean_: Optional[torch.Tensor] = None
        self.components_: Optional[torch.Tensor] = None
        self.singular_values_: Optional[torch.Tensor] = None

    def fit(self, X: torch.Tensor) -> "TorchPCA":
        """
        Fit PCA model to the data.

        Parameters
        ----------
        X : torch.Tensor
            Input data with shape [n_samples, n_features].

        Returns
        -------
        TorchPCA
            Fitted PCA model instance.
        """
        self.mean_ = X.mean(dim=0)
        unbiased = X - self.mean_.unsqueeze(0)
        U, S, V = torch.pca_lowrank(
            unbiased, q=self.n_components, center=False, niter=4
        )
        self.components_ = V.T
        self.singular_values_ = S
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Transform data using fitted PCA model.

        Parameters
        ----------
        X : torch.Tensor
            Input data with shape [n_samples, n_features].

        Returns
        -------
        torch.Tensor
            Transformed data with shape [n_samples, n_components].

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCA model has not been fitted yet. Call fit() first.")

        t0 = X - self.mean_.unsqueeze(0)
        projected = t0 @ self.components_.T
        return projected


# * --- Test --- #


def test_speed_torch_version():
    import time

    from tqdm import trange

    a = torch.randn(2, 128, 256, 256).cuda()
    t1 = time.time()
    for _ in trange(100):
        _ = feature_pca_torch(a, pca_k=3)
    t2 = time.time()
    print(f"Time taken: {(t2 - t1) / 100:.2f} s")


# --- Example Usage (Requires CUDA enabled PyTorch and einops) ---

if __name__ == "__main__":
    test_speed_torch_version()
    exit(0)

    # Make sure CUDA is available
    if not torch.cuda.is_available():
        log("CUDA is not available. Cannot run CUDA PCA example.")
    else:
        device = torch.device("cuda")
        log(f"Using device: {device}")

        # Example 1: Simulate 3D ViT features [bs, l, c]
        bs_vit, l_vit, c_vit = 4, 256, 768
        pca_k_vit = 64
        log("\n--- Testing 3D ViT-like Features ---")
        vit_features = torch.randn(
            bs_vit, l_vit, c_vit, device=device, dtype=torch.float32
        )

        log("Running PCA with PyTorch manual SVD...")
        reduced_vit_features_torch = feature_pca_torch(vit_features, pca_k=pca_k_vit)
        log(f"Final reduced shape (torch): {reduced_vit_features_torch.shape}")
        assert reduced_vit_features_torch.shape == (bs_vit, l_vit, pca_k_vit)
        assert reduced_vit_features_torch.device == device

        # Example 2: Simulate 4D CNN features [bs, c, h, w]
        bs_cnn, c_cnn, h_cnn, w_cnn = 2, 512, 14, 14
        pca_k_cnn = 32
        log("\n--- Testing 4D CNN-like Features ---")
        cnn_features = torch.randn(
            bs_cnn, c_cnn, h_cnn, w_cnn, device=device, dtype=torch.float32
        )

        log("Running PCA with PyTorch manual SVD...")
        reduced_cnn_features_torch = feature_pca_torch(cnn_features, pca_k=pca_k_cnn)
        log(f"Final reduced shape (torch): {reduced_cnn_features_torch.shape}")
        assert reduced_cnn_features_torch.shape == (bs_cnn, pca_k_cnn, h_cnn, w_cnn)
        assert reduced_cnn_features_torch.device == device

        # Optional: Compare with cuML if installed
        try:
            from cuml.decomposition import PCA as cuML_PCA

            log("\n--- Testing with cuML PCA (for comparison) ---")

            def feature_pca_cuml2(img_feat: torch.Tensor, pca_k: int):
                assert img_feat.ndim in (3, 4), "must be 1d vit or 2d cnn features"
                assert img_feat.is_cuda, "img_feat must be on GPU"

                _shape = img_feat.shape
                if len(_shape) == 3:
                    c = _shape[-1]  # [bs, l, c]
                    data = einops.rearrange(img_feat, "bs l c -> (bs l) c")
                    _back_kwargs = dict(
                        pattern="(bs l) k -> bs l k", k=pca_k, bs=_shape[0], l=_shape[1]
                    )
                else:
                    c = _shape[1]
                    data = einops.rearrange(img_feat, "bs c h w -> (bs h w) c")
                    _back_kwargs = dict(
                        pattern="(bs h w) k -> bs k h w",
                        k=pca_k,
                        bs=_shape[0],
                        h=_shape[2],
                        w=_shape[3],
                    )
                assert pca_k <= c, (
                    f"Target dimension pca_k ({pca_k}) cannot be larger than original feature dimension c ({c})"
                )

                pca = cuML_PCA(n_components=pca_k)
                projected_data = pca.fit_transform(data)

                img_feats_reduced_flat = torch.as_tensor(
                    projected_data, device=data.device, dtype=img_feat.dtype
                )
                img_feats_reduced = einops.rearrange(
                    img_feats_reduced_flat, **_back_kwargs
                )
                return img_feats_reduced

            log("Running PCA with cuML...")
            reduced_vit_features_cuml = feature_pca_cuml2(vit_features, pca_k=pca_k_vit)
            log(f"Final reduced shape (cuML): {reduced_vit_features_cuml.shape}")
            assert reduced_vit_features_cuml.shape == (bs_vit, l_vit, pca_k_vit)
            assert reduced_vit_features_cuml.device == device

            reduced_cnn_features_cuml = feature_pca_cuml2(cnn_features, pca_k=pca_k_cnn)
            log(f"Final reduced shape (cuML): {reduced_cnn_features_cuml.shape}")
            assert reduced_cnn_features_cuml.shape == (bs_cnn, pca_k_cnn, h_cnn, w_cnn)
            assert reduced_cnn_features_cuml.device == device

            # Note: The exact numerical results might differ slightly between PyTorch SVD
            # and cuML PCA due to implementation details and floating-point precision,
            # but the shapes and general transformed structure should be comparable.
            # To check numerical similarity, you could compare the projected data
            # after aligning their signs (as PCA components are directionally ambiguous).

        except ImportError:
            log("\ncuML not installed. Skipping cuML comparison.")
        except Exception as e:
            log(f"\nError during cuML comparison: {e}. Skipping.")
