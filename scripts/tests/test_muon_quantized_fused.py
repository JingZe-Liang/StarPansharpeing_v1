from __future__ import annotations

import copy

import pytest
import torch

from src.utilities.optim.flashoptim.flashoptim.optimizers import FlashAdamW, FlashLion, cast_model
from src.utilities.optim.muon_quantized_fused import QuantizedMuonFSDP


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _train_step(
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


def test_create_quantized_muon_optimizer_adamw_1d() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=True), torch.nn.LayerNorm(4)).to(_device())
    optimizer = QuantizedMuonFSDP.create_muon_optimizer(
        model.named_parameters(),
        oned_param_algo="adamw",
        lr=1e-3,
        muon_steps=1,
    )

    assert isinstance(optimizer, QuantizedMuonFSDP)
    assert isinstance(optimizer._flash_adamw_optimizer, FlashAdamW)
    assert optimizer._flash_lion_optimizer is None


def test_create_quantized_muon_optimizer_lion_1d() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=True), torch.nn.LayerNorm(4)).to(_device())
    optimizer = QuantizedMuonFSDP.create_muon_optimizer(
        model.named_parameters(),
        oned_param_algo="lion",
        lr=1e-3,
        muon_steps=1,
    )

    assert isinstance(optimizer, QuantizedMuonFSDP)
    assert isinstance(optimizer._flash_lion_optimizer, FlashLion)
    assert optimizer._flash_adamw_optimizer is None


def test_create_quantized_muon_optimizer_adamw_1d_with_cast_model() -> None:
    device = _device()
    model = torch.nn.Sequential(torch.nn.Linear(8, 8, bias=True), torch.nn.LayerNorm(8)).to(device)
    cast_model(model, dtype=torch.bfloat16)

    optimizer = QuantizedMuonFSDP.create_muon_optimizer(
        model.named_parameters(),
        oned_param_algo="adamw",
        lr=1e-3,
        muon_steps=1,
    )

    x = torch.randn(8, 8, device=device, dtype=torch.bfloat16)
    y = torch.randn(8, 8, device=device, dtype=torch.bfloat16)
    _train_step(model, optimizer, x, y)

    assert isinstance(optimizer._flash_adamw_optimizer, FlashAdamW)
    assert next(model.parameters()).dtype == torch.bfloat16


def test_create_quantized_muon_optimizer_adamw_1d_fp32_auto_disable_master_bits() -> None:
    device = _device()
    model = torch.nn.Sequential(torch.nn.Linear(8, 8, bias=True), torch.nn.LayerNorm(8)).to(
        device=device,
        dtype=torch.float32,
    )

    optimizer = QuantizedMuonFSDP.create_muon_optimizer(
        model.named_parameters(),
        oned_param_algo="adamw",
        lr=1e-3,
        muon_steps=1,
    )

    x = torch.randn(8, 8, device=device, dtype=torch.float32)
    y = torch.randn(8, 8, device=device, dtype=torch.float32)
    _train_step(model, optimizer, x, y)

    assert isinstance(optimizer._flash_adamw_optimizer, FlashAdamW)
    assert optimizer._flash_adamw_optimizer._master_bytewidth == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Muon momentum quantization path requires CUDA.")
def test_muon_momentum_quant_pack_unpack() -> None:
    device = _device()
    model = torch.nn.Linear(8, 8, bias=True).to(device=device, dtype=torch.bfloat16)

    optimizer = QuantizedMuonFSDP.create_muon_optimizer(
        model.named_parameters(),
        oned_param_algo="lion",
        lr=1e-3,
        muon_steps=1,
        use_triton=True,
    )

    x = torch.randn(8, 8, device=device, dtype=torch.bfloat16)
    y = torch.randn(8, 8, device=device, dtype=torch.bfloat16)
    _train_step(model, optimizer, x, y)

    muon_weight = model.weight
    assert id(muon_weight) in optimizer._muon_qstate
    assert optimizer.state[muon_weight]["momentum"] is None

    optimizer._materialize_muon_momentum_states()
    assert isinstance(optimizer.state[muon_weight]["momentum"], torch.Tensor)

    optimizer._quantize_muon_momentum_states()
    assert optimizer.state[muon_weight]["momentum"] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Groupwise quantization check requires CUDA.")
def test_groupwise_muon_quant_switch() -> None:
    device = _device()

    class TwoWeightModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w1 = torch.nn.Parameter(torch.randn(4, 4, device=device, dtype=torch.bfloat16))
            self.w2 = torch.nn.Parameter(torch.randn(4, 4, device=device, dtype=torch.bfloat16))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x @ self.w1 + x @ self.w2

    model = TwoWeightModel()

    optimizer = QuantizedMuonFSDP(
        params=[
            {"params": [model.w1], "algorithm": "muon", "muon_quantize_momentum": True},
            {"params": [model.w2], "algorithm": "muon", "muon_quantize_momentum": False},
        ],
        lr=1e-3,
        muon_steps=1,
        use_triton=True,
    )

    x = torch.randn(8, 4, device=device, dtype=torch.bfloat16)
    y = torch.randn(8, 4, device=device, dtype=torch.bfloat16)
    _train_step(model, optimizer, x, y)

    assert id(model.w1) in optimizer._muon_qstate
    assert id(model.w2) not in optimizer._muon_qstate


def test_state_dict_roundtrip_new_format() -> None:
    device = _device()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    torch.manual_seed(0)
    model_a = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=True), torch.nn.LayerNorm(4)).to(
        device=device, dtype=dtype
    )
    model_b = copy.deepcopy(model_a)

    optimizer_a = QuantizedMuonFSDP.create_muon_optimizer(
        model_a.named_parameters(),
        oned_param_algo="lion",
        lr=1e-3,
        muon_steps=1,
        use_triton=device.type == "cuda",
    )
    optimizer_b = QuantizedMuonFSDP.create_muon_optimizer(
        model_b.named_parameters(),
        oned_param_algo="lion",
        lr=1e-3,
        muon_steps=1,
        use_triton=device.type == "cuda",
    )

    x = torch.randn(8, 4, device=device, dtype=dtype)
    y = torch.randn(8, 4, device=device, dtype=dtype)

    _train_step(model_a, optimizer_a, x, y)

    sd = optimizer_a.state_dict()
    assert set(sd.keys()) == {"base_state", "oned_flash_state", "muon_qstate", "meta"}
    assert sd["meta"]["version"] == 1

    model_b.load_state_dict(model_a.state_dict())
    optimizer_b.load_state_dict(sd)

    _train_step(model_a, optimizer_a, x, y)
    _train_step(model_b, optimizer_b, x, y)

    for p_a, p_b in zip(model_a.parameters(), model_b.parameters(), strict=True):
        torch.testing.assert_close(p_a, p_b)


def test_no_double_update_for_1d_params() -> None:
    device = _device()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    torch.manual_seed(0)
    model_a = torch.nn.LayerNorm(4).to(device=device, dtype=dtype)
    model_b = copy.deepcopy(model_a)

    optimizer_a = QuantizedMuonFSDP.create_muon_optimizer(
        model_a.named_parameters(),
        oned_param_algo="adamw",
        lr=1e-3,
        muon_steps=1,
    )

    flash_quantize = torch.cuda.is_available()
    flash_fused = torch.cuda.is_available()
    flash_master_weight_bits = 24 if torch.cuda.is_available() else None
    optimizer_b = FlashAdamW(
        model_b.parameters(),
        lr=1e-3,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
        quantize=flash_quantize,
        compress_state_dict=False,
        master_weight_bits=flash_master_weight_bits,
        check_numerics=False,
        fused=flash_fused,
    )

    x = torch.randn(8, 4, device=device, dtype=dtype)
    y = torch.randn(8, 4, device=device, dtype=dtype)

    _train_step(model_a, optimizer_a, x, y)
    _train_step(model_b, optimizer_b, x, y)

    for p_a, p_b in zip(model_a.parameters(), model_b.parameters(), strict=True):
        torch.testing.assert_close(p_a, p_b)
