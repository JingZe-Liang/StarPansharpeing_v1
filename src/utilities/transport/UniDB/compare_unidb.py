import torch
import sys
import os
import math

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import plan_continuous
import plan_disc


def compare_tensors(name, t1, t2, tol=1e-5):
    diff = (t1 - t2).abs().max().item()
    if diff < tol:
        print(f"[PASS] {name} match (max diff: {diff:.2e})")
    else:
        print(f"[FAIL] {name} mismatch (max diff: {diff:.2e})")
        print(f"  t1: {t1.flatten()[:5]}")
        print(f"  t2: {t2.flatten()[:5]}")


def main():
    device = torch.device("cpu")
    dtype = torch.float32

    # Params
    lambda_sq = 1.0
    gamma = 1.0
    T = 20  # Small T for testing loop
    eps = 0.01

    # 1. Init Continuous
    uni_cont = plan_continuous.UniDBContinuous(lambda_square=lambda_sq, gamma=gamma, T=T, schedule="cosine", eps=eps)

    # 2. Init Discrete
    # plan_disc.UniDB expects lambda_square NOT divided by 255 if >= 1?
    # plan_disc.py: "self.lambda_square = lambda_square / 255 if lambda_square >= 1 else lambda_square"
    # plan_continuous.py: "lambda_square = lambda_square / 255.0 if lambda_square >= 1.0 else lambda_square"
    # Matches.
    uni_disc = plan_disc.UniDB(lambda_square=lambda_sq, gamma=gamma, T=T, schedule="cosine", eps=eps, device=device)

    print(f"--- Comparing UniDB (T={T}) ---")

    # 3. Check Schedule (Thetas)
    # Continuous: _thetas (size T+1)?
    # Discrete: self.thetas (size T+2? or T+1?)
    # plan_disc cosine: "timesteps = timesteps + 2", then "betas = 1 - alphas_cumprod[1:-1]" => size T+1.
    # plan_continuous cosine: "t = timesteps + 2", ... "betas = 1 - alphas_cumprod[1:-1]" => size T+1.
    compare_tensors("Thetas", uni_cont._thetas, uni_disc.thetas)

    # 4. Check Sigmas/SigmaBars/m at each step t (1 to T)
    # Generate dummy data
    x = torch.randn(1, 1, dtype=dtype, device=device)  # scalar for coeffs
    mu = torch.randn(1, 1, dtype=dtype, device=device)

    # We will loop t from 1 to T
    for t_idx in range(1, T + 1):
        t_tensor = torch.tensor([t_idx], device=device, dtype=torch.long)
        t01 = uni_cont._index_to_t01(t_idx, device=device, dtype=dtype)

        # NOTE: plan_disc usually takes integer t in [1, T] but uses it as index for arrays.
        # But wait, plan_disc arrays like self.sigmas ... are they 0-indexed or 1-indexed (padded)?
        # plan_disc.py: "thetas = cosine_theta_schedule(T)". returns tensor of size T+1.
        # In forward: "for t in tqdm(range(1, T + 1)): ... self.sigmas[t] ..."
        # So they seem to rely on indices 1..T.
        # Let's check size of self.thetas in python if we could but code analysis:
        # cosine_theta_schedule returns size T+1.
        # So indices 0..T exist.
        # t=1 uses index 1.

        # A. m(t)
        # continuous
        m_cont = uni_cont.m(t01, device=device, dtype=dtype)
        # discrete
        m_disc = uni_disc.m(t_idx)  # returns tensor or scalar?
        # plan_disc.m(t) uses self.thetas_cumsum[t], etc. Arrays are tensors.
        # m_disc should be tensor.

        compare_tensors(f"m(t={t_idx})", m_cont, m_disc.cpu())

        # B. sigma(t) (marginal sigma)
        sigma_cont = uni_cont.sigma(t01, device=device, dtype=dtype)
        sigma_disc = uni_disc.f_sigma(t_idx)
        compare_tensors(f"sigma(t={t_idx})", sigma_cont, sigma_disc.cpu())

        # C. Reverse Mean (Posterior) - r_mean_1 vs _reverse_mean_gamma_adjacent
        if t_idx > 1:
            x_val = torch.randn(10, 3, 32, 32, device=device)
            mu_val = torch.randn(10, 3, 32, 32, device=device)
            x0_val = torch.randn(10, 3, 32, 32, device=device)

            # disc
            rm_disc = uni_disc.r_mean_1(x_val, x0_val, t_idx)

            # cont
            rm_cont = uni_cont._reverse_mean_gamma_adjacent(x_val, x0_val, mu_val, t_tensor)

            compare_tensors(f"r_mean_1(t={t_idx})", rm_cont, rm_disc)

    # 5. Reverse Drift (SDE)
    # Check sde_reverse_drift vs _reverse_mean_theta_adjacent (partially)
    # _reverse_mean_theta_adjacent returns "xt - drift * dt".
    # plan_disc reverse_sde_step_mean returns "x - sde_reverse_drift".
    # So we compare drift * dt vs sde_reverse_drift.

    t_idx = 10
    t_tensor = torch.tensor([t_idx], device=device, dtype=torch.long)
    t01 = uni_cont._index_to_t01(t_idx, device=device, dtype=dtype)

    x_val = torch.randn(10, 3, 32, 32, device=device)
    mu_val = torch.randn(10, 3, 32, 32, device=device)
    eps_hat = torch.randn(10, 3, 32, 32, device=device)

    # In plan_disc, score = - noise / f_sigma(t).
    # If we treat eps_hat as "noise", then score = - eps_hat / sigma(t).
    sigma_val = uni_cont.sigma(t01, device=device, dtype=dtype)
    score_val = -eps_hat / sigma_val.view(-1, 1, 1, 1)

    # Disc
    # sde_reverse_drift(self, x, score, t)
    drift_disc = uni_disc.sde_reverse_drift(x_val, score_val, t_tensor)

    # Cont
    # _reverse_mean_theta_adjacent computes the next point.
    # next_point = xt - drift * dt.
    # So drift_cont_integrated = drift * dt = xt - next_point.
    next_point_cont = uni_cont._reverse_mean_theta_adjacent(x_val, mu_val, t_tensor, eps_hat)
    drift_cont = x_val - next_point_cont

    compare_tensors(f"Reverse Drift (t={t_idx})", drift_cont, drift_disc, tol=1e-4)

    print("Comparison Finished.")


if __name__ == "__main__":
    main()
