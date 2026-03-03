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


def test_get_repa_feature_dict_multi_teacher_semantic() -> None:
    tokenizer = _build_multi_teacher_stub()
    repa_feature_dict = tokenizer.get_repa_feature_dict(force_to=True)
    assert repa_feature_dict is not None

    assert set(repa_feature_dict.keys()) == {"dino", "siglip"}
    assert torch.is_tensor(repa_feature_dict["dino"])
    assert torch.is_tensor(repa_feature_dict["siglip"])
    assert repa_feature_dict["dino"].shape[1] == 6
    assert repa_feature_dict["siglip"].shape[1] == 7


def test_get_repa_feature_dict_multi_teacher_low_level() -> None:
    tokenizer = _build_multi_teacher_stub()
    tokenizer.phis_student_source = "low_level"

    repa_feature_dict = tokenizer.get_repa_feature_dict(force_to=True)
    assert repa_feature_dict is not None
    assert repa_feature_dict["dino"].shape[1] == 4
    assert repa_feature_dict["siglip"].shape[1] == 5
