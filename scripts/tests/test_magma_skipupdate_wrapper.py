from __future__ import annotations

import torch

from src.utilities.optim.magma_skipupdate_wrapper import (
    MagmaSkipUpdateWrapper,
    create_torch_magma_optimizer,
    wrap_optimizer_with_magma,
)
from src.utilities.optim.muon_fused import MuonFSDP


def _step_to_int(step_value: torch.Tensor | int | float) -> int:
    if torch.is_tensor(step_value):
        return int(step_value.item())
    return int(step_value)


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


def test_skipupdate_p1_matches_base_optimizer() -> None:
    torch.manual_seed(0)
    model_base = torch.nn.Linear(4, 3, bias=True)
    model_wrap = torch.nn.Linear(4, 3, bias=True)
    model_wrap.load_state_dict(model_base.state_dict())

    base_optimizer = torch.optim.AdamW(model_base.parameters(), lr=1e-2, foreach=False)
    wrapped_base_optimizer = torch.optim.AdamW(model_wrap.parameters(), lr=1e-2, foreach=False)
    wrapped_optimizer = wrap_optimizer_with_magma(
        wrapped_base_optimizer,
        mode="skipupdate",
        survival_prob=1.0,
        seed=0,
    )

    x = torch.randn(8, 4)
    y = torch.randn(8, 3)
    _train_step_mse(model_base, base_optimizer, x, y)
    _train_step_mse(model_wrap, wrapped_optimizer, x, y)

    for p_base, p_wrap in zip(model_base.parameters(), model_wrap.parameters(), strict=True):
        torch.testing.assert_close(p_base, p_wrap)


def test_skipupdate_mask_can_skip_update_but_keep_dense_state() -> None:
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 4, bias=False)
    base_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, foreach=False)
    optimizer = wrap_optimizer_with_magma(
        base_optimizer,
        mode="skipupdate",
        survival_prob=0.5,
        seed=1,
    )

    x = torch.randn(4, 4)
    y = torch.randn(4, 4)
    param = model.weight
    before = param.detach().clone()
    _train_step_mse(model, optimizer, x, y)

    torch.testing.assert_close(param.detach(), before)

    base_state = optimizer.base_state[param]
    assert _step_to_int(base_state["step"]) > 0
    assert "exp_avg" in base_state
    assert not torch.allclose(base_state["exp_avg"], torch.zeros_like(base_state["exp_avg"]))


def test_magma_scale_tracks_alignment() -> None:
    torch.manual_seed(0)
    model = torch.nn.Linear(1, 1, bias=False)
    base_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer = wrap_optimizer_with_magma(
        base_optimizer,
        mode="magma",
        survival_prob=1.0,
        tau=2.0,
        ema_beta=0.9,
    )

    param = model.weight
    optimizer.zero_grad(set_to_none=True)
    param.grad = -torch.ones_like(param)
    optimizer.step()
    s1 = optimizer.get_param_mask_state(param)["s_t"]

    optimizer.zero_grad(set_to_none=True)
    param.grad = torch.ones_like(param)
    optimizer.step()
    s2 = optimizer.get_param_mask_state(param)["s_t"]

    assert isinstance(s1, float)
    assert isinstance(s2, float)
    assert s2 < s1


def test_wrapper_supports_muon_fsdp() -> None:
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=True), torch.nn.LayerNorm(4))
    first_layer = model[0]
    assert isinstance(first_layer, torch.nn.Linear)
    base_optimizer = MuonFSDP.create_muon_optimizer(
        model.named_parameters(),
        lr=1e-3,
        muon_steps=1,
    )
    optimizer = wrap_optimizer_with_magma(
        base_optimizer,
        mode="magma",
        survival_prob=1.0,
        seed=0,
    )

    x = torch.randn(4, 4)
    target = torch.randn(4, 4)
    before = first_layer.weight.detach().clone()
    loss = torch.nn.functional.mse_loss(model(x), target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    assert not torch.allclose(first_layer.weight.detach(), before)


def test_factory_for_torch_optimizer_and_muon_optimizer() -> None:
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=True), torch.nn.LayerNorm(4))

    torch_wrapper = create_torch_magma_optimizer(
        model.named_parameters(),
        optimizer_cls="torch.optim.AdamW",
        optimizer_kwargs={"lr": 1e-3, "foreach": False},
    )
    assert isinstance(torch_wrapper, MagmaSkipUpdateWrapper)
    assert isinstance(torch_wrapper.base_optimizer, torch.optim.AdamW)

    muon_optimizer = MuonFSDP.create_muon_optimizer(
        model.named_parameters(),
        lr=1e-3,
        muon_steps=1,
    )
    assert isinstance(muon_optimizer, MuonFSDP)


def test_muon_create_optimizer_supports_reg_mask_mode() -> None:
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=True), torch.nn.LayerNorm(4))

    optimizer = MuonFSDP.create_muon_optimizer(
        model.named_parameters(),
        lr=1e-3,
        muon_steps=1,
        reg_mask_mode="magma",
        magma_wrap_kwargs={"survival_prob": 1.0, "seed": 0},
    )
    assert isinstance(optimizer, MuonFSDP)
    for group in optimizer.param_groups:
        assert group["reg_mask_mode"] == "magma"
        assert float(group["reg_mask_survival_prob"]) == 1.0
        assert group["reg_mask_seed"] == 0


def test_wrapper_foreach_bucket_mixed_dtype_cpu() -> None:
    torch.manual_seed(0)
    p32 = torch.nn.Parameter(torch.randn(3, dtype=torch.float32))
    p64 = torch.nn.Parameter(torch.randn(3, dtype=torch.float64))
    base_optimizer = torch.optim.SGD([p32, p64], lr=0.1)
    optimizer = wrap_optimizer_with_magma(
        base_optimizer,
        mode="skipupdate",
        survival_prob=1.0,
        seed=0,
    )

    before_p32 = p32.detach().clone()
    before_p64 = p64.detach().clone()
    p32.grad = torch.ones_like(p32)
    p64.grad = torch.ones_like(p64)
    optimizer.step()

    assert not torch.allclose(p32.detach(), before_p32)
    assert not torch.allclose(p64.detach(), before_p64)


def test_state_dict_roundtrip() -> None:
    torch.manual_seed(0)
    model_a = torch.nn.Linear(2, 2, bias=False)
    model_b = torch.nn.Linear(2, 2, bias=False)
    model_b.load_state_dict(model_a.state_dict())

    optimizer_a = create_torch_magma_optimizer(
        model_a.named_parameters(),
        optimizer_cls=torch.optim.SGD,
        optimizer_kwargs={"lr": 1e-2},
        mode="magma",
        survival_prob=0.5,
        seed=7,
    )
    optimizer_b = create_torch_magma_optimizer(
        model_b.named_parameters(),
        optimizer_cls=torch.optim.SGD,
        optimizer_kwargs={"lr": 1e-2},
        mode="magma",
        survival_prob=0.5,
        seed=123,
    )

    param_a = model_a.weight
    param_b = model_b.weight
    optimizer_a.zero_grad(set_to_none=True)
    param_a.grad = torch.full_like(param_a, -1.0)
    optimizer_a.step()

    state = optimizer_a.state_dict()
    model_b.load_state_dict(model_a.state_dict())
    optimizer_b.load_state_dict(state)

    s_a = optimizer_a.get_param_mask_state(param_a)["s_t"]
    s_b = optimizer_b.get_param_mask_state(param_b)["s_t"]
    assert isinstance(s_a, float)
    assert isinstance(s_b, float)
    assert abs(s_a - s_b) < 1e-12

    optimizer_a.zero_grad(set_to_none=True)
    optimizer_b.zero_grad(set_to_none=True)
    param_a.grad = torch.full_like(param_a, 1.0)
    param_b.grad = torch.full_like(param_b, 1.0)
    optimizer_a.step()
    optimizer_b.step()

    torch.testing.assert_close(model_a.weight, model_b.weight)

    s_a_next = optimizer_a.get_param_mask_state(param_a)["s_t"]
    s_b_next = optimizer_b.get_param_mask_state(param_b)["s_t"]
    assert isinstance(s_a_next, float)
    assert isinstance(s_b_next, float)
    assert abs(s_a_next - s_b_next) < 1e-12
