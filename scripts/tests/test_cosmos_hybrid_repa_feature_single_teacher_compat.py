from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.stage1.cosmos.cosmos_hybrid import CosmosHybridTokenizer


def _build_single_teacher_stub() -> CosmosHybridTokenizer:
    tokenizer = CosmosHybridTokenizer.__new__(CosmosHybridTokenizer)
    nn.Module.__init__(tokenizer)

    tokenizer._use_repa_loss = True
    tokenizer._vf_on_z_or_module = "z"
    tokenizer.low_lvl_repa_proj_is_multi = True
    tokenizer.sem_repa_proj_is_multi_layer_cached = True
    tokenizer.sem_repa_proj_is_multi_layer_cached_by_teacher = {}
    tokenizer._use_multi_teacher_proj = False
    tokenizer._teacher_names = []
    tokenizer._legacy_multi_teacher_warned = False
    tokenizer.phis_student_source = "semantic"
    tokenizer.training = True
    tokenizer.low_lvl_repa_proj_chans = [2, 2]

    tokenizer.z = [torch.randn(2, 2, 4, 4), torch.randn(2, 2, 4, 4)]
    tokenizer.sem_z = [torch.randn(2, 3, 4, 4), torch.randn(2, 3, 4, 4)]

    tokenizer._repa_proj = nn.ModuleDict(
        {
            "low_lvl_repa_proj": nn.ModuleList([nn.Conv2d(2, 4, 1), nn.Conv2d(2, 4, 1)]),
            "sem_repa_proj": nn.ModuleList([nn.Conv2d(3, 5, 1), nn.Conv2d(3, 5, 1)]),
        }
    )
    return tokenizer


def test_get_repa_feature_single_teacher_compat() -> None:
    tokenizer = _build_single_teacher_stub()
    repa_feature = tokenizer.get_repa_feature(force_to=True)

    assert isinstance(repa_feature, tuple)
    low_lvl_feature, sem_feature = repa_feature
    assert isinstance(low_lvl_feature, list)
    assert isinstance(sem_feature, list)
    assert len(low_lvl_feature) == 2
    assert len(sem_feature) == 2
    assert low_lvl_feature[0].shape[1] == 4
    assert sem_feature[0].shape[1] == 5
