import torch

from src.stage2.cloud_removal.diffusion.loss import estimate_x0_from_v, interpolant, pack_conditions


def test_pack_conditions_concatenates_channel_dim():
    b, h, w = 2, 8, 8
    c1, c2 = 3, 5
    a = torch.randn(b, c1, h, w)
    b2 = torch.randn(b, c2, h, w)
    out = pack_conditions([a, b2])
    assert out is not None
    assert out.shape == (b, c1 + c2, h, w)


def test_estimate_x0_from_v_recovers_linear():
    torch.manual_seed(0)
    b, c, h, w = 4, 3, 8, 8
    x0 = torch.randn(b, c, h, w)
    eps = torch.randn_like(x0)
    t = torch.rand(b, 1, 1, 1)

    alpha, sigma, d_alpha, d_sigma = interpolant(t, path_type="linear")
    x_t = alpha * x0 + sigma * eps
    v_t = d_alpha * x0 + d_sigma * eps

    x0_hat = estimate_x0_from_v(x_t, v_t, t, path_type="linear")
    torch.testing.assert_close(x0_hat, x0, rtol=1e-6, atol=1e-6)


def test_estimate_x0_from_v_recovers_cosine():
    torch.manual_seed(0)
    b, c, h, w = 4, 3, 8, 8
    x0 = torch.randn(b, c, h, w)
    eps = torch.randn_like(x0)
    t = torch.rand(b, 1, 1, 1)

    alpha, sigma, d_alpha, d_sigma = interpolant(t, path_type="cosine")
    x_t = alpha * x0 + sigma * eps
    v_t = d_alpha * x0 + d_sigma * eps

    x0_hat = estimate_x0_from_v(x_t, v_t, t, path_type="cosine")
    torch.testing.assert_close(x0_hat, x0, rtol=1e-5, atol=1e-5)
