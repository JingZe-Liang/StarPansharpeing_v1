"""
Fully Constrained Least Squares Unmixing (FCLSU) implementations

This module provides multiple implementations of the FCLSU algorithm for hyperspectral unmixing,
including PyTorch-native versions and improved scipy-based versions.

"""

from typing import Union

import numpy as np
import numpy as np_orig
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import lsq_linear as ls_linear_func
from scipy.optimize import nnls


class FCLSUSolver:
    """
    FCLSU solver with multiple backend options
    """

    def __init__(self, backend: str = "scipy", **kwargs):
        """
        Initialize FCLSU solver

        Parameters
        ----------
        backend : str
            Backend to use ('scipy', 'ls_linear', 'torch_pgd', 'torch_adam')
        **kwargs
            Additional parameters for specific backends
        """
        self.backend = backend
        self.kwargs = kwargs

    def solve(
        self,
        M: Union[np.ndarray, torch.Tensor],
        Y: Union[np.ndarray, torch.Tensor],
        sigma: float = 1.0,
    ) -> torch.Tensor:
        """
        Solve FCLSU problem

        Parameters
        ----------
        M : Union[np.ndarray, torch.Tensor]
            Endmember matrix [bands, endmembers]
        Y : Union[np.ndarray, torch.Tensor]
            Observation matrix [pixels, bands]
        sigma : float
            Regularization parameter for sum-to-one constraint

        Returns
        -------
        torch.Tensor
            Abundance matrix [endmembers, pixels]
        """
        if self.backend == "scipy":
            return self._solve_scipy(M, Y, sigma)
        elif self.backend == "ls_linear":
            return self._solve_ls_linear(M, Y, sigma)
        elif self.backend == "torch_pgd":
            return self._solve_torch_pgd(M, Y, sigma)
        elif self.backend == "torch_adam":
            return self._solve_torch_adam(M, Y, sigma)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _solve_scipy(
        self,
        M: Union[np.ndarray, torch.Tensor],
        Y: Union[np.ndarray, torch.Tensor],
        sigma: float = 1.0,
    ) -> torch.Tensor:
        """
        Original scipy.optimize.nnls implementation
        """
        # Convert to numpy if needed
        if isinstance(M, torch.Tensor):
            M_np = M.detach().cpu().numpy()
        else:
            M_np = M.copy()

        if isinstance(Y, torch.Tensor):
            Y_np = Y.detach().cpu().numpy()
        else:
            Y_np = Y.copy()

        P = M_np.shape[1]  # Number of endmembers
        N = Y_np.shape[0]  # Number of pixels

        # Add sum-to-one constraint
        M_constrained = np_orig.vstack((sigma * M_np, np_orig.ones((1, P))))
        Y_constrained = np_orig.vstack((sigma * Y_np.T, np_orig.ones((1, N))))

        A_hat = np.zeros((P, N))

        # Solve for each pixel
        for i in range(N):
            A_hat[:, i], _ = nnls(M_constrained, Y_constrained[:, i])

        return torch.tensor(A_hat, dtype=torch.float32)

    def _solve_ls_linear(
        self,
        M: Union[np.ndarray, torch.Tensor],
        Y: Union[np.ndarray, torch.Tensor],
        sigma: float = 1.0,
    ) -> torch.Tensor:
        """
        scipy.optimize.ls_linear implementation
        """
        # Convert to numpy if needed
        if isinstance(M, torch.Tensor):
            M_np = M.detach().cpu().numpy()
        else:
            M_np = M.copy()

        if isinstance(Y, torch.Tensor):
            Y_np = Y.detach().cpu().numpy()
        else:
            Y_np = Y.copy()

        P = M_np.shape[1]  # Number of endmembers
        N = Y_np.shape[0]  # Number of pixels

        # Add sum-to-one constraint
        M_constrained = np_orig.vstack((sigma * M_np, np_orig.ones((1, P))))
        Y_constrained = np_orig.vstack((sigma * Y_np.T, np_orig.ones((1, N))))

        A_hat = np.zeros((P, N))

        # Solve for each pixel using ls_linear
        for i in range(N):
            result = ls_linear_func(M_constrained, Y_constrained[:, i], bounds=(0, np.inf))
            A_hat[:, i] = result.x

        return torch.tensor(A_hat, dtype=torch.float32)

    def _solve_torch_pgd(
        self,
        M: Union[np.ndarray, torch.Tensor],
        Y: Union[np.ndarray, torch.Tensor],
        sigma: float = 1.0,
    ) -> torch.Tensor:
        """
        PyTorch implementation using Projected Gradient Descent
        """
        # Convert to torch tensors
        if isinstance(M, np.ndarray):
            M_torch = torch.tensor(M, dtype=torch.float32)
        else:
            M_torch = M.float()

        if isinstance(Y, np.ndarray):
            Y_torch = torch.tensor(Y, dtype=torch.float32)
        else:
            Y_torch = Y.float()

        P = M_torch.shape[1]  # Number of endmembers
        N = Y_torch.shape[0]  # Number of pixels

        # Add sum-to-one constraint
        M_constrained = torch.vstack([sigma * M_torch, torch.ones(1, P)])
        Y_constrained = torch.vstack([sigma * Y_torch.T, torch.ones(1, N)])

        # Parameters
        max_iter = self.kwargs.get("max_iter", 1000)
        lr = self.kwargs.get("lr", 0.01)
        tol = self.kwargs.get("tol", 1e-6)

        A_hat = torch.zeros(P, N, device=M_torch.device)

        # Solve for each pixel
        for i in range(N):
            x = torch.zeros(P, device=M_torch.device, requires_grad=True)

            for _ in range(max_iter):
                # Compute gradient
                residual = M_constrained @ x - Y_constrained[:, i]
                gradient = 2 * M_constrained.T @ residual

                # Update with gradient descent
                with torch.no_grad():
                    x_new = x - lr * gradient

                    # Project onto constraints: x >= 0
                    x_new = torch.clamp(x_new, min=0)

                    # Check convergence
                    if torch.norm(x_new - x) < tol:
                        break

                    x.copy_(x_new)

            A_hat[:, i] = x.detach()

        return A_hat

    def _solve_torch_adam(
        self,
        M: Union[np.ndarray, torch.Tensor],
        Y: Union[np.ndarray, torch.Tensor],
        sigma: float = 1.0,
    ) -> torch.Tensor:
        """
        PyTorch implementation using Adam optimizer
        """
        # Convert to torch tensors
        if isinstance(M, np.ndarray):
            M_torch = torch.tensor(M, dtype=torch.float32)
        else:
            M_torch = M.float()

        if isinstance(Y, np.ndarray):
            Y_torch = torch.tensor(Y, dtype=torch.float32)
        else:
            Y_torch = Y.float()

        P = M_torch.shape[1]  # Number of endmembers
        N = Y_torch.shape[0]  # Number of pixels

        # Add sum-to-one constraint
        M_constrained = torch.vstack([sigma * M_torch, torch.ones(1, P)])
        Y_constrained = torch.vstack([sigma * Y_torch.T, torch.ones(1, N)])

        # Parameters
        max_iter = self.kwargs.get("max_iter", 1000)
        lr = self.kwargs.get("lr", 0.01)
        tol = self.kwargs.get("tol", 1e-6)

        A_hat = torch.zeros(P, N, device=M_torch.device)

        # Solve for each pixel
        for i in range(N):
            # Use ReLU to ensure non-negativity
            x_raw = torch.zeros(P, device=M_torch.device, requires_grad=True)
            optimizer = torch.optim.Adam([x_raw], lr=lr)

            prev_loss = float("inf")

            for _ in range(max_iter):
                optimizer.zero_grad()

                # Apply ReLU constraint
                x = torch.relu(x_raw)

                # Compute loss
                residual = M_constrained @ x - Y_constrained[:, i]
                loss = torch.norm(residual) ** 2

                # Backward pass
                loss.backward()
                optimizer.step()

                # Check convergence
                if abs(prev_loss - loss.detach().item()) < tol:
                    break
                prev_loss = loss.item()

            A_hat[:, i] = torch.relu(x_raw).detach()

        return A_hat


def torch_nnls(
    A: torch.Tensor,
    b: torch.Tensor,
    max_iter: int = 1000,
    lr: float = 0.01,
    method: str = "pgd",
) -> torch.Tensor:
    """
    PyTorch implementation of Non-Negative Least Squares

    Parameters
    ----------
    A : torch.Tensor
        Coefficient matrix [m, n]
    b : torch.Tensor
        Observation vector [m]
    max_iter : int
        Maximum number of iterations
    lr : float
        Learning rate
    method : str
        Optimization method ('pgd', 'adam')

    Returns
    -------
    torch.Tensor
        Solution vector [n]
    """
    assert A.dim() == 2, "A must be a 2D tensor"
    assert b.dim() == 1, "b must be a 1D tensor"
    assert A.shape[0] == b.shape[0], "Dimension mismatch between A and b"

    device = A.device

    if method == "pgd":
        # Projected Gradient Descent
        x = torch.zeros(A.shape[1], device=device, requires_grad=True)

        for _ in range(max_iter):
            # Compute gradient
            residual = A @ x - b
            gradient = 2 * A.T @ residual

            # Update with gradient descent
            with torch.no_grad():
                x_new = x - lr * gradient
                # Project onto non-negative orthant
                x_new = torch.clamp(x_new, min=0)

                # Check convergence
                if torch.norm(x_new - x) < 1e-6:
                    break

                x.copy_(x_new)

        return x.detach()

    elif method == "adam":
        # Adam optimizer with ReLU constraint
        x_raw = torch.zeros(A.shape[1], device=device, requires_grad=True)
        optimizer = torch.optim.Adam([x_raw], lr=lr)

        for _ in range(max_iter):
            optimizer.zero_grad()

            # Apply ReLU constraint
            x = torch.relu(x_raw)

            # Compute loss
            residual = A @ x - b
            loss = torch.norm(residual) ** 2

            # Backward pass
            loss.backward()
            optimizer.step()

        return torch.relu(x_raw).detach()

    else:
        raise ValueError(f"Unknown method: {method}")


def batch_torch_nnls(
    A: torch.Tensor,
    B: torch.Tensor,
    max_iter: int = 1000,
    lr: float = 0.01,
    method: str = "pgd",
) -> torch.Tensor:
    """
    Batched PyTorch implementation of NNLS

    Parameters
    ----------
    A : torch.Tensor
        Coefficient matrix [m, n]
    B : torch.Tensor
        Observation matrix [m, batch_size]
    max_iter : int
        Maximum number of iterations
    lr : float
        Learning rate
    method : str
        Optimization method ('pgd', 'adam')

    Returns
    -------
    torch.Tensor
        Solution matrix [n, batch_size]
    """
    assert A.dim() == 2, "A must be a 2D tensor"
    assert B.dim() == 2, "B must be a 2D tensor"
    assert A.shape[0] == B.shape[0], "Dimension mismatch between A and B"

    device = A.device
    batch_size = B.shape[1]

    if method == "pgd":
        # Projected Gradient Descent for batch
        X = torch.zeros(A.shape[1], batch_size, device=device, requires_grad=True)

        for _ in range(max_iter):
            # Compute gradients for all batch
            residuals = A @ X - B
            gradients = 2 * A.T @ residuals

            # Update with gradient descent
            with torch.no_grad():
                X_new = X - lr * gradients
                # Project onto non-negative orthant
                X_new = torch.clamp(X_new, min=0)

                # Check convergence
                if torch.norm(X_new - X) < 1e-6:
                    break

                X.copy_(X_new)

        return X.detach()

    elif method == "adam":
        # Adam optimizer with ReLU constraint for batch
        X_raw = torch.zeros(A.shape[1], batch_size, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([X_raw], lr=lr)

        for _ in range(max_iter):
            optimizer.zero_grad()

            # Apply ReLU constraint
            X = torch.relu(X_raw)

            # Compute loss
            residuals = A @ X - B
            loss = torch.norm(residuals) ** 2

            # Backward pass
            loss.backward()
            optimizer.step()

        return torch.relu(X_raw).detach()

    else:
        raise ValueError(f"Unknown method: {method}")


def FCLSU(
    M: Union[np.ndarray, torch.Tensor],
    Y: Union[np.ndarray, torch.Tensor],
    sigma: float = 1.0,
    backend: str = "scipy",
    **kwargs,
) -> torch.Tensor:
    """
    Fully Constrained Least Squares Unmixing

    Parameters
    ----------
    M : Union[np.ndarray, torch.Tensor]
        Endmember matrix [bands, endmembers]
    Y : Union[np.ndarray, torch.Tensor]
        Observation matrix [pixels, bands]
    sigma : float
        Regularization parameter for sum-to-one constraint
    backend : str
        Backend to use ('scipy', 'ls_linear', 'torch_pgd', 'torch_adam')
    **kwargs
        Additional parameters for specific backends

    Returns
    -------
    torch.Tensor
        Abundance matrix [endmembers, pixels]
    """
    solver = FCLSUSolver(backend, **kwargs)
    return solver.solve(M, Y, sigma)


# Legacy function for backward compatibility
def FCLSU_legacy(M: np.ndarray, Y: np.ndarray, sigma: float = 1) -> torch.Tensor:
    """
    Legacy FCLSU implementation matching the original interface

    Parameters
    ----------
    M : np.ndarray
        Endmember matrix [bands, endmembers]
    Y : np.ndarray
        Observation matrix [pixels, bands]
    sigma : float
        Regularization parameter

    Returns
    -------
    torch.Tensor
        Abundance matrix [endmembers, pixels]
    """
    P = M.shape[1]  # [c, em]
    N = Y.shape[1]  # [hw, c]
    M_constrained = np_orig.vstack((sigma * M, np_orig.ones((1, P))))
    Y_constrained = np_orig.vstack((sigma * Y, np_orig.ones((1, N))))
    A_hat = np.zeros((P, N))

    for i in np.arange(N):
        A_hat[:, i], _ = nnls(M_constrained, Y_constrained[:, i])

    return torch.tensor(A_hat)


if __name__ == "__main__":
    # Test the implementations
    print("Testing FCLSU implementations...")

    # Create test data
    np.random.seed(42)
    torch.manual_seed(42)

    # Simple test case
    M = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)  # 3 bands, 2 endmembers
    Y = np.array([[2, 1, 1], [1, 2, 0]], dtype=np.float32)  # 2 pixels, 3 bands

    print(f"Endmember matrix shape: {M.shape}")
    print(f"Observation matrix shape: {Y.shape}")

    # Test different backends
    backends = ["scipy", "ls_linear", "torch_pgd", "torch_adam"]

    for backend in backends:
        try:
            result = FCLSU(M, Y, backend=backend, max_iter=1000, lr=0.01)
            print(f"{backend}: {result.shape}, sum constraint: {result.sum(dim=0)}")
        except Exception as e:
            print(f"{backend}: Error - {e}")

    print("Test completed!")
