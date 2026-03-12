import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.stage1.cosmos.cosmos_hybrid import CosmosHybridTokenizer


def _make_tokenizer_stub() -> CosmosHybridTokenizer:
    tokenizer = object.__new__(CosmosHybridTokenizer)
    tokenizer.cache_layers = {
        "low_level": [0, 1, 2, -1],
        "semantic": [2, 5, 8, 11],
    }
    tokenizer.cnn_cfg = SimpleNamespace(
        model=SimpleNamespace(
            channels=256,
            channels_mult=[1, 2, 2],
        )
    )
    return tokenizer


def test_set_low_level_proj_chans_is_instance_local() -> None:
    first_tokenizer = _make_tokenizer_stub()
    first_tokenizer._set_low_level_proj_chans()

    second_tokenizer = _make_tokenizer_stub()
    second_tokenizer._set_low_level_proj_chans()

    expected_channels = [256, 512, 512, 512]
    assert first_tokenizer.low_lvl_repa_proj_chans == expected_channels
    assert second_tokenizer.low_lvl_repa_proj_chans == expected_channels
    assert first_tokenizer.low_lvl_repa_proj_chans is not second_tokenizer.low_lvl_repa_proj_chans
    assert CosmosHybridTokenizer.low_lvl_repa_proj_chans == []
