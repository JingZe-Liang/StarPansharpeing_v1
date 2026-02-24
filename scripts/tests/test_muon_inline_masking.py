from __future__ import annotations

import torch

from src.utilities.optim.muon_fused import MuonFSDP


def _train_step_mse(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    pred = model(x)
    loss = torch.nn.functional.mse_loss(pred, y)
    loss.backward()
    optimizer.step()


def _build_small_model() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(4, 4, bias=True),
        torch.nn.LayerNorm(4),
    )


def test_muon_reg_mask_none_matches_baseline() -> None:
    torch.manual_seed(0)
    model_base = _build_small_model()
    model_none = _build_small_model()
    model_none.load_state_dict(model_base.state_dict())

    optimizer_base = MuonFSDP.create_muon_optimizer(
        model_base.named_parameters(),
        lr=1e-3,
        muon_steps=1,
    )
    optimizer_none = MuonFSDP.create_muon_optimizer(
        model_none.named_parameters(),
        lr=1e-3,
        muon_steps=1,
        reg_mask_mode=None,
    )

    x = torch.randn(8, 4)
    y = torch.randn(8, 4)
    _train_step_mse(model_base, optimizer_base, x, y)
    _train_step_mse(model_none, optimizer_none, x, y)

    for p_base, p_none in zip(model_base.parameters(), model_none.parameters(), strict=True):
        torch.testing.assert_close(p_base, p_none)


def test_muon_skipupdate_p1_matches_no_masking() -> None:
    torch.manual_seed(0)
    model_ref = _build_small_model()
    model_skip = _build_small_model()
    model_skip.load_state_dict(model_ref.state_dict())

    optimizer_ref = MuonFSDP.create_muon_optimizer(
        model_ref.named_parameters(),
        lr=1e-3,
        muon_steps=1,
    )
    optimizer_skip = MuonFSDP.create_muon_optimizer(
        model_skip.named_parameters(),
        lr=1e-3,
        muon_steps=1,
        reg_mask_mode="skipupdate",
        magma_wrap_kwargs={"survival_prob": 1.0, "seed": 0},
    )

    x = torch.randn(8, 4)
    y = torch.randn(8, 4)
    _train_step_mse(model_ref, optimizer_ref, x, y)
    _train_step_mse(model_skip, optimizer_skip, x, y)

    for p_ref, p_skip in zip(model_ref.parameters(), model_skip.parameters(), strict=True):
        torch.testing.assert_close(p_ref, p_skip)


def test_magma_scale_tracks_alignment() -> None:
    torch.manual_seed(0)
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = MuonFSDP(
        params=[{"params": [param], "algorithm": "lion"}],
        lr=0.1,
        mu=0.95,
        betas=(0.9, 0.95),
        muon_params_defaults={"reg_mask_mode": "magma"},
        oned_params_defaults={
            "reg_mask_mode": "magma",
            "reg_mask_survival_prob": 1.0,
            "reg_mask_tau": 2.0,
            "reg_mask_ema_beta": 0.9,
            "reg_mask_seed": 0,
        },
    )

    optimizer.zero_grad(set_to_none=True)
    param.grad = -torch.ones_like(param)
    optimizer.step()
    s1 = optimizer.state[param]["mask_s_t"]

    optimizer.zero_grad(set_to_none=True)
    # Use a tiny opposite-sign recovery gradient so momentum and grad become anti-aligned.
    param.grad = torch.full_like(param, 1e-4)
    optimizer.step()
    s2 = optimizer.state[param]["mask_s_t"]

    assert isinstance(s1, float)
    assert isinstance(s2, float)
    assert s2 < s1


def test_muon_inline_masking_covers_muon_and_oned_groups() -> None:
    torch.manual_seed(0)
    model = _build_small_model()
    optimizer = MuonFSDP.create_muon_optimizer(
        model.named_parameters(),
        oned_param_algo="adamw",
        lr=1e-3,
        muon_steps=1,
        reg_mask_mode="magma",
        magma_wrap_kwargs={"survival_prob": 1.0, "seed": 3},
    )

    x = torch.randn(8, 4)
    y = torch.randn(8, 4)
    _train_step_mse(model, optimizer, x, y)

    muon_params: list[torch.nn.Parameter] = []
    oned_params: list[torch.nn.Parameter] = []
    for group in optimizer.param_groups:
        if group["algorithm"] == "muon":
            muon_params.extend(group["params"])
        elif group["algorithm"] in ("adamw", "lion"):
            oned_params.extend(group["params"])

    assert muon_params
    assert oned_params
    assert any("mask_s_t" in optimizer.state[p] for p in muon_params)
    assert any("mask_s_t" in optimizer.state[p] for p in oned_params)


def test_muon_load_state_dict_normalizes_rng_state_dtype_and_device() -> None:
    torch.manual_seed(0)
    model_a = _build_small_model()
    model_b = _build_small_model()
    model_b.load_state_dict(model_a.state_dict())

    optimizer_a = MuonFSDP.create_muon_optimizer(
        model_a.named_parameters(),
        lr=1e-3,
        muon_steps=1,
        reg_mask_mode="magma",
        magma_wrap_kwargs={"survival_prob": 1.0, "seed": 2026},
    )
    optimizer_b = MuonFSDP.create_muon_optimizer(
        model_b.named_parameters(),
        lr=1e-3,
        muon_steps=1,
        reg_mask_mode="magma",
        magma_wrap_kwargs={"survival_prob": 1.0, "seed": 2026},
    )

    state = optimizer_a.state_dict()
    state["_mask_rng_state"] = state["_mask_rng_state"].to(dtype=torch.int64)
    if torch.cuda.is_available():
        state["_mask_rng_state"] = state["_mask_rng_state"].to(device="cuda")

    optimizer_b.load_state_dict(state)
