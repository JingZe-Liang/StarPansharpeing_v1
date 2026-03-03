from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.stage1.cosmos.cosmos_hybrid import CosmosHybridTokenizer


def _build_multi_teacher_stub() -> CosmosHybridTokenizer:
    tokenizer = CosmosHybridTokenizer.__new__(CosmosHybridTokenizer)
    nn.Module.__init__(tokenizer)

    tokenizer._use_repa_loss = True
    tokenizer._vf_on_z_or_module = "z"
    tokenizer.low_lvl_repa_proj_is_multi = False
    tokenizer.sem_repa_proj_is_multi_layer_cached = False
    tokenizer.sem_repa_proj_is_multi_layer_cached_by_teacher = {"dino": False, "siglip": False}
    tokenizer._use_multi_teacher_proj = True
    tokenizer._teacher_names = ["dino", "siglip"]
    tokenizer._legacy_multi_teacher_warned = False
    tokenizer.phis_student_source = "semantic"
    tokenizer.training = True

    tokenizer.z = torch.randn(2, 2, 4, 4)
    tokenizer.sem_z = torch.randn(2, 3, 4, 4)
    tokenizer._repa_proj = nn.ModuleDict(
        {
            "dino": nn.ModuleDict(
                {
                    "low_lvl_repa_proj": nn.Conv2d(2, 4, 1),
                    "sem_repa_proj": nn.Conv2d(3, 6, 1),
                }
            ),
            "siglip": nn.ModuleDict(
                {
                    "low_lvl_repa_proj": nn.Conv2d(2, 5, 1),
                    "sem_repa_proj": nn.Conv2d(3, 7, 1),
                }
            ),
        }
    )
    return tokenizer


def test_get_repa_feature_legacy_fallback_uses_primary_teacher() -> None:
    tokenizer = _build_multi_teacher_stub()

    repa_feature = tokenizer.get_repa_feature(force_to=True)
    assert repa_feature is not None
    low_lvl_feature, sem_feature = repa_feature
    assert low_lvl_feature.shape[1] == 4
    assert sem_feature.shape[1] == 6
    assert tokenizer._legacy_multi_teacher_warned
