from __future__ import annotations

import torch

from src.utilities.optim.muon_fused import MuonFSDP


def test_muon_fsdp_forces_weight_decay_zero_for_hyperball() -> None:
    torch.manual_seed(0)

    model = torch.nn.Sequential(torch.nn.Linear(8, 4, bias=True), torch.nn.LayerNorm(4))
    opt = MuonFSDP.create_muon_optimizer(
        model.named_parameters(),
        muon_params_defaults={"weight_update_method": "hyperball", "weight_decay": 0.1},
        oned_params_defaults={"weight_decay": 0.2},
        lr=1e-3,
    )

    muon_groups = [g for g in opt.param_groups if g.get("algorithm") == "muon"]
    assert len(muon_groups) == 1
    assert muon_groups[0]["weight_update_method"] == "hyperball"
    assert float(muon_groups[0]["weight_decay"]) == 0.0

    oned_groups = [g for g in opt.param_groups if g.get("algorithm") in ("adamw", "lion")]
    assert len(oned_groups) == 1
    assert float(oned_groups[0]["weight_decay"]) == 0.2
