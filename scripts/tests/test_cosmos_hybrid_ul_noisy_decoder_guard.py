from __future__ import annotations

import pytest

from src.stage1.cosmos.cosmos_hybrid import CosmosHybridTokenizer


def _make_tokenizer_stub(
    *,
    decoder_use_ul_noisy_latent: bool = True,
    latent_bottleneck_type: str = "after_semantic",
    st_skip_sem_decoder: bool = False,
) -> CosmosHybridTokenizer:
    tokenizer = object.__new__(CosmosHybridTokenizer)
    tokenizer.decoder_use_ul_noisy_latent = decoder_use_ul_noisy_latent
    tokenizer.latent_bottleneck_type = latent_bottleneck_type
    tokenizer.st_skip_sem_decoder = st_skip_sem_decoder
    return tokenizer


def test_ul_noisy_decoder_guard_accepts_after_semantic_single_latent() -> None:
    tokenizer = _make_tokenizer_stub()
    tokenizer._validate_ul_noisy_decoder_init_constraints()


def test_ul_noisy_decoder_guard_accepts_before_semantic_single_latent() -> None:
    tokenizer = _make_tokenizer_stub(latent_bottleneck_type="before_semantic")
    tokenizer._validate_ul_noisy_decoder_init_constraints()


def test_ul_noisy_decoder_guard_rejects_skip_path() -> None:
    tokenizer = _make_tokenizer_stub(st_skip_sem_decoder=True)
    with pytest.raises(ValueError, match="clean latent"):
        tokenizer._validate_ul_noisy_decoder_init_constraints()


def test_ul_noisy_decoder_guard_rejects_dual_latent() -> None:
    tokenizer = _make_tokenizer_stub()
    with pytest.raises(ValueError, match="dual/mixed latents"):
        tokenizer._raise_if_ul_noisy_decoder_dual_latent(latent_is_dual=True)
