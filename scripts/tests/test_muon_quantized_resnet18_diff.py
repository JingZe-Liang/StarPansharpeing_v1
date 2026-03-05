from __future__ import annotations

import gc

import pytest
import torch
import torch.nn.functional as F
from torchvision.models import resnet18  # type: ignore[unresolved-import]

from src.utilities.optim.flashoptim.flashoptim.optimizers import cast_model
from src.utilities.optim.muon_fused import MuonFSDP
from src.utilities.optim.muon_quantized_fused import QuantizedMuonFSDP


def _run_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    pred = model(x)
    loss = F.cross_entropy(pred.float(), y)
    loss.backward()
    optimizer.step()
    return float(loss.detach().float().cpu())


def _mean_abs_param_diff(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    total_numel = 0
    total_abs = 0.0
    for pa, pb in zip(a, b, strict=True):
        d = (pa.detach().float() - pb.detach().float()).abs()
        total_numel += d.numel()
        total_abs += float(d.sum().cpu())
    return total_abs / max(total_numel, 1)


def _make_resnet18_state(dtype: torch.dtype, device: torch.device, seed: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    model = resnet18(num_classes=10).to(device)
    cast_model(model, dtype=dtype)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return state


def _build_batches(
    *,
    seed: int,
    steps: int,
    batch_size: int,
    image_hw: int,
    num_classes: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for i in range(steps):
        torch.manual_seed(seed + i + 1)
        x = torch.randn(batch_size, 3, image_hw, image_hw, device="cpu", dtype=torch.float32)
        y = torch.randint(0, num_classes, (batch_size,), device="cpu")
        batches.append((x, y))
    return batches


def _run_experiment(
    *,
    mode: str,
    base_state: dict[str, torch.Tensor],
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    dtype: torch.dtype,
    common_group_hparams: dict[str, dict[str, float]],
) -> tuple[list[float], list[torch.Tensor], int | None]:
    model = resnet18(num_classes=10).to(device)
    cast_model(model, dtype=dtype)
    model.load_state_dict(base_state, strict=True)

    if mode == "muon_ref":
        optimizer: torch.optim.Optimizer = MuonFSDP.create_muon_optimizer(
            model.named_parameters(),
            oned_param_algo="adamw",
            lr=1e-3,
            muon_steps=2,
            use_triton=True,
            **common_group_hparams,
        )
    elif mode == "quant_no_quant":
        optimizer = QuantizedMuonFSDP.create_muon_optimizer(
            model.named_parameters(),
            oned_param_algo="adamw",
            lr=1e-3,
            muon_steps=2,
            use_triton=True,
            muon_params_defaults={
                "lr": 1e-3,
                "weight_decay": 0.01,
                "muon_quantize_momentum": False,
            },
            oned_params_defaults=common_group_hparams["oned_params_defaults"],
            oned_flash_quantize=False,
            oned_flash_master_weight_bits=None,
        )
    elif mode == "quant_yes_quant":
        optimizer = QuantizedMuonFSDP.create_muon_optimizer(
            model.named_parameters(),
            oned_param_algo="adamw",
            lr=1e-3,
            muon_steps=2,
            use_triton=True,
            muon_params_defaults={
                "lr": 1e-3,
                "weight_decay": 0.01,
                "muon_quantize_momentum": True,
                "muon_quant_group_size": 32,
                "muon_quant_softsign": True,
            },
            oned_params_defaults=common_group_hparams["oned_params_defaults"],
            oned_flash_quantize=True,
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    losses: list[float] = []
    for x_cpu, y_cpu in batches:
        x = x_cpu.to(device=device, dtype=dtype, non_blocking=True)
        y = y_cpu.to(device=device, non_blocking=True)
        losses.append(_run_step(model, optimizer, x, y))

    final_params = [p.detach().float().cpu().clone() for p in model.parameters()]
    qstate_len = len(optimizer._muon_qstate) if isinstance(optimizer, QuantizedMuonFSDP) else None

    del optimizer
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return losses, final_params, qstate_len


@pytest.mark.skipif(not torch.cuda.is_available(), reason="This comparison test requires CUDA quantization paths.")
def test_resnet18_muon_adamw_vs_quantized_muon_difference() -> None:
    seed = 3407
    device = torch.device("cuda")
    dtype = torch.bfloat16
    steps = 4
    base_state = _make_resnet18_state(dtype=dtype, device=device, seed=seed)
    batches = _build_batches(seed=seed, steps=steps, batch_size=8, image_hw=64, num_classes=10)

    common_group_hparams = {
        "muon_params_defaults": {"lr": 1e-3, "weight_decay": 0.01},
        "oned_params_defaults": {"lr": 1e-3, "weight_decay": 0.01},
    }

    losses_ref, params_ref, qstate_ref = _run_experiment(
        mode="muon_ref",
        base_state=base_state,
        batches=batches,
        device=device,
        dtype=dtype,
        common_group_hparams=common_group_hparams,
    )
    losses_no_quant, params_no_quant, qstate_no_quant = _run_experiment(
        mode="quant_no_quant",
        base_state=base_state,
        batches=batches,
        device=device,
        dtype=dtype,
        common_group_hparams=common_group_hparams,
    )
    losses_quant, params_quant, qstate_quant = _run_experiment(
        mode="quant_yes_quant",
        base_state=base_state,
        batches=batches,
        device=device,
        dtype=dtype,
        common_group_hparams=common_group_hparams,
    )

    mean_loss_diff_no_quant = sum(abs(a - b) for a, b in zip(losses_ref, losses_no_quant, strict=True)) / steps
    mean_loss_diff_quant = sum(abs(a - b) for a, b in zip(losses_ref, losses_quant, strict=True)) / steps

    mean_param_diff_no_quant = _mean_abs_param_diff(params_ref, params_no_quant)
    mean_param_diff_quant = _mean_abs_param_diff(params_ref, params_quant)

    assert qstate_ref is None
    assert qstate_no_quant == 0
    assert qstate_quant is not None and qstate_quant > 0

    assert mean_loss_diff_no_quant <= 5e-3
    assert mean_loss_diff_quant <= 5e-2
    assert mean_param_diff_no_quant <= 1e-3
    assert mean_param_diff_quant <= 5e-3
