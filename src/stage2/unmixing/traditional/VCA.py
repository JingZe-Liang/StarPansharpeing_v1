# taken from https://github.com/Laadr/VCA/blob/master/VCA.py
# modified into Torch version


import sys

import numpy as np
import torch

from loguru import logger

#############################################
# Internal functions
#############################################


def estimate_snr(Y, r_m, x):
    [L, N] = Y.shape  # L number of bands (channels), N number of pixels
    [p, N] = x.shape  # p number of endmembers (reduced dimension)

    P_y = np.sum(Y**2) / float(N)
    P_x = np.sum(x**2) / float(N) + np.sum(r_m**2)
    snr_est = 10 * np.log10((P_x - p / L * P_y) / (P_y - P_x))

    return snr_est


def vca(Y, R, verbose=True, snr_input=0):
    # Vertex Component Analysis
    #
    # Ae, indice, Yp = vca(Y,R,verbose = True,snr_input = 0)
    #
    # ------- Input variables -------------
    #  Y - matrix with dimensions L(channels) x N(pixels)
    #      each pixel is a linear mixture of R endmembers
    #      signatures Y = M x s, where s = gamma x alfa
    #      gamma is a illumination perturbation factor and
    #      alfa are the abundance fractions of each endmember.
    #  R - positive integer number of endmembers in the scene
    #
    # ------- Output variables -----------
    # Ae     - estimated mixing matrix (endmembers signatures)
    # indice - pixels that were chosen to be the most pure
    # Yp     - Data matrix Y projected.
    #
    # ------- Optional parameters---------
    # snr_input - (float) signal to noise ratio (dB)
    # v         - [True | False]
    # ------------------------------------
    #
    # Author: Adrien Lagrange (adrien.lagrange@enseeiht.fr)
    # This code is a translation of a matlab code provided by
    # Jose Nascimento (zen@isel.pt) and Jose Bioucas Dias (bioucas@lx.it.pt)
    # available at http://www.lx.it.pt/~bioucas/code.htm under a non-specified Copyright (c)
    # Translation of last version at 22-February-2018 (Matlab version 2.1 (7-May-2004))
    #
    # more details on:
    # Jose M. P. Nascimento and Jose M. B. Dias
    # "Vertex Component Analysis: A Fast Algorithm to Unmix Hyperspectral Data"
    # submited to IEEE Trans. Geosci. Remote Sensing, vol. .., no. .., pp. .-., 2004
    #
    #

    #############################################
    # Initializations
    #############################################
    if len(Y.shape) != 2:
        raise ValueError(
            "Input data must be of size L (number of bands i.e. channels) by N (number of pixels)"
        )

    [L, N] = Y.shape  # L number of bands (channels), N number of pixels

    R = int(R)
    if R < 0 or R > L:
        raise ValueError("ENDMEMBER parameter must be integer between 1 and L")

    #############################################``
    # SNR Estimates
    #############################################

    if snr_input == 0:
        y_m = np.mean(Y, axis=1, keepdims=True)
        Y_o = Y - y_m  # data with zero-mean
        Ud = np.linalg.svd(np.dot(Y_o, Y_o.T) / float(N))[0][
            :, :R
        ]  # computes the R-projection matrix
        x_p = np.dot(Ud.T, Y_o)  # project the zero-mean data onto p-subspace

        SNR = estimate_snr(Y, y_m, x_p)

        if verbose:
            logger.debug("SNR estimated = {}[dB]".format(SNR))
    else:
        SNR = snr_input
        if verbose:
            logger.debug("input SNR = {}[dB]\n".format(SNR))

    SNR_th = 15 + 10 * np.log10(R)

    #############################################
    # Choosing Projective Projection or
    #          projection to p-1 subspace
    #############################################

    if SNR < SNR_th:
        if verbose:
            logger.debug("... Select proj. to R-1")

        d = R - 1
        if snr_input == 0:  # it means that the projection is already computed
            Ud = Ud[:, :d]
        else:
            y_m = np.mean(Y, axis=1, keepdims=True)
            Y_o = Y - y_m  # data with zero-mean

            Ud = np.linalg.svd(np.dot(Y_o, Y_o.T) / float(N))[0][
                :, :d
            ]  # computes the p-projection matrix
            x_p = np.dot(Ud.T, Y_o)  # project thezeros mean data onto p-subspace

        Yp = np.dot(Ud, x_p[:d, :]) + y_m  # again in dimension L

        x = x_p[:d, :]  #  x_p =  Ud.T * Y_o is on a R-dim subspace
        c = np.amax(np.sum(x**2, axis=0)) ** 0.5
        y = np.vstack((x, c * np.ones((1, N))))
    else:
        if verbose:
            logger.debug("... Select the projective proj.")

        d = R
        Ud = np.linalg.svd(np.dot(Y, Y.T) / float(N))[0][
            :, :d
        ]  # computes the p-projection matrix

        x_p = np.dot(Ud.T, Y)
        Yp = np.dot(
            Ud, x_p[:d, :]
        )  # again in dimension L (note that x_p has no null mean)

        x = np.dot(Ud.T, Y)
        u = np.mean(x, axis=1, keepdims=True)  # equivalent to  u = Ud.T * r_m
        y = x / (np.dot(u.T, x) + 1e-7)

    #############################################
    # VCA algorithm
    #############################################

    indice = np.zeros((R), dtype=int)
    A = np.zeros((R, R))
    A[-1, 0] = 1

    for i in range(R):
        w = np.random.rand(R, 1)
        f = w - np.dot(A, np.dot(np.linalg.pinv(A), w))
        f = f / np.linalg.norm(f)

        v = np.dot(f.T, y)

        indice[i] = np.argmax(np.absolute(v))
        A[:, i] = y[:, indice[i]]  # same as x(:,indice(i))

    Ae = Yp[:, indice]

    return Ae, indice, Yp


#############################################
# PyTorch Implementation
#############################################


def estimate_snr_torch(Y, r_m, x):
    """
    PyTorch implementation of SNR estimation

    Args:
        Y: torch.Tensor of shape [L, N] - input data
        r_m: torch.Tensor of shape [L, 1] - mean vector
        x: torch.Tensor of shape [p, N] - projected data

    Returns:
        snr_est: torch.Tensor - estimated SNR in dB
    """
    L, N = Y.shape
    p, _ = x.shape

    P_y = torch.sum(Y**2) / float(N)
    P_x = torch.sum(x**2) / float(N) + torch.sum(r_m**2)
    snr_est = 10 * torch.log10((P_x - p / L * P_y) / (P_y - P_x + 1e-10))

    return snr_est


def vca_torch(Y, R, verbose=True, snr_input=0, device: str | torch.device = "cuda"):
    """
    PyTorch implementation of Vertex Component Analysis

    Args:
        Y: torch.Tensor of shape [L, N] - input hyperspectral data
           L: number of bands (channels)
           N: number of pixels
        R: int - number of endmembers to extract
        verbose: bool - whether to logger.debug verbose output
        snr_input: float - input SNR in dB (0 for estimation)
        device: str - device to use ('cuda' or 'cpu')

    Returns:
        Ae: torch.Tensor of shape [L, R] - estimated endmember signatures
        indice: torch.Tensor of shape [R] - indices of selected pixels
        Yp: torch.Tensor of shape [L, N] - projected data matrix
    """
    # Input validation
    if len(Y.shape) != 2:
        raise ValueError(
            "Input data must be of size L (number of bands) by N (number of pixels)"
        )

    L, N = Y.shape
    R = int(R)

    if R < 0 or R > L:
        raise ValueError("ENDMEMBER parameter must be integer between 1 and L")

    # Move data to specified device
    Y = Y.to(device)

    #############################################
    # SNR Estimates
    #############################################

    if snr_input == 0:
        # Estimate SNR from data
        y_m = torch.mean(Y, dim=1, keepdim=True)
        Y_o = Y - y_m  # data with zero-mean

        # Compute SVD for projection matrix
        Y_cov = torch.mm(Y_o, Y_o.T) / float(N)
        U, S, V = torch.svd(Y_cov)
        Ud = U[:, :R]  # R-projection matrix

        x_p = torch.mm(Ud.T, Y_o)  # project zero-mean data onto R-subspace

        SNR = estimate_snr_torch(Y, y_m, x_p)

        if verbose:
            logger.debug(f"SNR estimated = {SNR.item():.2f}[dB]")
    else:
        SNR = torch.tensor(snr_input, device=device)
        if verbose:
            logger.debug(f"input SNR = {SNR.item():.2f}[dB]")

    SNR_th = 15 + 10 * torch.log10(torch.tensor(R, dtype=torch.float32, device=device))

    #############################################
    # Choosing Projective Projection or
    #          projection to R-1 subspace
    #############################################

    if SNR < SNR_th:
        if verbose:
            logger.debug("... Select proj. to R-1")

        d = R - 1
        if snr_input == 0:
            Ud = Ud[:, :d]
        else:
            y_m = torch.mean(Y, dim=1, keepdim=True)
            Y_o = Y - y_m  # data with zero-mean

            Y_cov = torch.mm(Y_o, Y_o.T) / float(N)
            U, S, V = torch.svd(Y_cov)
            Ud = U[:, :d]

            x_p = torch.mm(Ud.T, Y_o)

        Yp = torch.mm(Ud, x_p[:d, :]) + y_m  # back to original dimension L
        x = x_p[:d, :]  # projected data
        c = torch.max(torch.sum(x**2, dim=0)) ** 0.5
        y = torch.vstack((x, c * torch.ones((1, N), device=device)))
    else:
        if verbose:
            logger.debug("... Select the projective proj.")

        d = R
        Y_cov = torch.mm(Y, Y.T) / float(N)
        U, S, V = torch.svd(Y_cov)
        Ud = U[:, :d]

        x_p = torch.mm(Ud.T, Y)
        Yp = torch.mm(Ud, x_p[:d, :])  # back to original dimension L

        x = torch.mm(Ud.T, Y)
        u = torch.mean(x, dim=1, keepdim=True)
        y = x / (torch.mm(u.T, x) + 1e-7)

    #############################################
    # VCA algorithm
    #############################################

    # Initialize variables
    indice = torch.zeros(R, dtype=torch.long, device=device)
    A = torch.zeros(R, R, device=device)
    A[-1, 0] = 1

    # Random projections to find extreme pixels
    for i in range(R):
        # Generate random projection vector
        w = torch.rand(R, 1, device=device)

        # Project onto orthogonal complement of A
        f = w - torch.mm(A, torch.mm(torch.pinverse(A), w))
        f = f / (torch.norm(f) + 1e-10)

        # Project data onto f and find extreme pixel
        v = torch.mm(f.T, y)
        indice[i] = torch.argmax(torch.abs(v))

        # Update A matrix
        A[:, i] = y[:, indice[i]]

    # Extract endmember signatures
    Ae = Yp[:, indice]

    return Ae, indice, Yp


def vca_torch_batch(Y, R, batch_size=10000, verbose=True, snr_input=0, device="cuda"):
    """
    Batch implementation of VCA for large datasets

    Args:
        Y: torch.Tensor of shape [L, N] - input hyperspectral data.
            L: number of bands (channels), N: number of pixels
        R: int - number of endmembers to extract
        batch_size: int - batch size for processing
        verbose: bool - whether to logger.debug verbose output
        snr_input: float - input SNR in dB (0 for estimation)
        device: str - device to use ('cuda' or 'cpu')

    Returns:
        Ae: torch.Tensor of shape [L, R] - estimated endmember signatures
        indice: torch.Tensor of shape [R] - indices of selected pixels
        Yp: torch.Tensor of shape [L, N] - projected data matrix
    """
    L, N = Y.shape

    if N <= batch_size:
        # Use standard implementation for small datasets
        return vca_torch(Y, R, verbose, snr_input, device)

    if verbose:
        logger.debug(
            f"Processing large dataset ({N} pixels) in batches of {batch_size}"
        )

    # Process in batches for memory efficiency
    Y_batches = torch.split(Y, batch_size, dim=1)
    endmember_candidates = []
    indices_candidates = []

    for i, Y_batch in enumerate(Y_batches):
        if verbose:
            logger.debug(f"Processing batch {i + 1}/{len(Y_batches)}")

        # Run VCA on batch
        Ae_batch, indice_batch, _ = vca_torch(
            Y_batch, R, verbose=False, snr_input=snr_input, device=device
        )

        # Adjust indices to global coordinates
        global_offset = i * batch_size
        indice_global = indice_batch + global_offset

        endmember_candidates.append(Ae_batch)
        indices_candidates.extend(indice_global.cpu().numpy())

    # Combine results and select best endmembers
    all_endmembers = torch.cat(endmember_candidates, dim=1)

    # Use original VCA on the combined candidates to select final endmembers
    if verbose:
        logger.debug("Selecting final endmembers from candidates...")

    # Run VCA on candidate endmembers
    Ae_final, indice_final, Yp = vca_torch(
        all_endmembers, R, verbose=False, snr_input=snr_input, device=device
    )

    # Map back to original indices
    original_indices = torch.tensor(
        [indices_candidates[idx] for idx in indice_final.cpu().numpy()],
        dtype=torch.long,
        device=device,
    )

    # Compute full projection
    Yp_full = project_data_torch(Y, Ae_final, device=device)

    return Ae_final, original_indices, Yp_full


def project_data_torch(Y, endmembers, device="cuda"):
    """
    Project data onto endmember subspace

    Args:
        Y: torch.Tensor of shape [L, N] - input data
        endmembers: torch.Tensor of shape [L, R] - endmember signatures
        device: str - device to use

    Returns:
        Yp: torch.Tensor of shape [L, N] - projected data
    """
    Y = Y.to(device)
    endmembers = endmembers.to(device)

    # Compute projection matrix
    M_pinv = torch.pinverse(endmembers)

    # Project data
    abundances = torch.mm(M_pinv, Y)
    Yp = torch.mm(endmembers, abundances)

    return Yp


def find_best_matching(endmembers_pred, endmembers_gt):
    """
    Find the best matching between predicted and ground truth endmembers
    using Hungarian algorithm based on correlation
    """
    R = endmembers_pred.shape[1]
    correlation_matrix = np.zeros((R, R))

    for i in range(R):
        for j in range(R):
            corr = np.corrcoef(endmembers_pred[:, i], endmembers_gt[:, j])[0, 1]
            correlation_matrix[i, j] = corr

    # Find best matching
    from scipy.optimize import linear_sum_assignment

    row_ind, col_ind = linear_sum_assignment(-correlation_matrix)

    return row_ind, col_ind, correlation_matrix[row_ind, col_ind]


def test_vca_torch():
    """
    Test function to compare numpy and torch implementations
    """
    logger.debug("Testing PyTorch VCA implementation...")

    # Set seeds for reproducibility
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Generate synthetic data
    L, N, R = 200, 10000, 5  # bands, pixels, endmembers

    # Create random endmembers
    endmembers_np = np.random.rand(L, R)

    # Create random abundances (sum to 1)
    abundances_np = np.random.rand(R, N)
    abundances_np = abundances_np / abundances_np.sum(axis=0, keepdims=True)

    # Generate mixed data
    Y_np = np.dot(endmembers_np, abundances_np)

    # Add noise
    noise_level = 0.01
    Y_np += noise_level * np.random.randn(L, N)

    # Convert to torch
    Y_torch = torch.from_numpy(Y_np).float()

    # Run numpy version
    logger.debug("Running numpy VCA...")
    Ae_np, indice_np, Yp_np = vca(Y_np, R, verbose=False)

    # Run torch version with same seed
    logger.debug("Running torch VCA...")
    Ae_torch, indice_torch, Yp_torch = vca_torch(Y_torch, R, verbose=False, snr_input=0)

    # Compare results
    logger.debug(f"Results comparison:")
    logger.debug(f"Numpy endmembers shape: {Ae_np.shape}")
    logger.debug(f"Torch endmembers shape: {Ae_torch.shape}")

    # Find best matching between endmembers
    row_ind, col_ind, correlations = find_best_matching(Ae_np, Ae_torch.cpu().numpy())

    logger.debug(f"\nOptimal matching results:")
    for i, (np_idx, torch_idx) in enumerate(zip(row_ind, col_ind)):
        corr = correlations[i]
        logger.debug(
            f"Match {i + 1}: NP endmember {np_idx} <-> Torch endmember {torch_idx}, correlation: {corr:.4f}"
        )

    logger.debug(f"Average correlation: {np.mean(correlations):.4f}")

    # Also compare direct correspondence (without matching)
    logger.debug(f"\nDirect correspondence (without matching):")
    direct_correlations = []
    for i in range(R):
        corr = np.corrcoef(Ae_np[:, i], Ae_torch[:, i].cpu().numpy())[0, 1]
        direct_correlations.append(corr)
        logger.debug(f"Endmember {i + 1} correlation: {corr:.4f}")
    logger.debug(f"Average direct correlation: {np.mean(direct_correlations):.4f}")

    return Ae_np, Ae_torch, indice_np, indice_torch


if __name__ == "__main__":
    # Run test if executed directly
    test_vca_torch()
