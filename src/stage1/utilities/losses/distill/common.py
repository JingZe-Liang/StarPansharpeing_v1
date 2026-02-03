import math

import torch

from .hadamard import get_hadamard_matrix


def get_phi_s_matrix(X: torch.Tensor):
    mean_val = X.mean(dim=0)

    X = X - mean_val

    cov = torch.cov(X.T)

    e, U = torch.linalg.eigh(cov)
    mask = e >= 0
    e = torch.where(mask, e, torch.zeros_like(e))

    H = get_hadamard_matrix(e.shape[0]).to(U)

    normalizer = e.mean().sqrt()
    inv_normalizer = 1.0 / normalizer

    whiten = inv_normalizer * (H @ U.T)

    return mean_val, whiten


def phi_s_inverse_to_teacher_space(student_feat_norm, mean, alpha, H, U):
    # y = y' @ ((1/alpha) * H @ U.T) + mean
    W_inv = (1.0 / alpha) * (H @ U.t())  # (C,C)
    if student_feat_norm.dim() == 3:  # BNC
        return student_feat_norm @ W_inv.t() + mean[None, None, :]
    else:  # BCHW
        B, C, Hh, Ww = student_feat_norm.shape
        x = student_feat_norm.permute(0, 2, 3, 1).contiguous()
        y = x @ W_inv.t() + mean[None, None, None, :]
        return y.permute(0, 3, 1, 2).contiguous()
