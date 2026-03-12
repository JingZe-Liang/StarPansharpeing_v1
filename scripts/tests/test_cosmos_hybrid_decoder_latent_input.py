import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.stage1.cosmos.cosmos_hybrid import CosmosHybridTokenizer


def test_prepare_single_decoder_latent_inputs_uses_transmitted_sampled_latent() -> None:
    tokenizer = object.__new__(CosmosHybridTokenizer)
    tokenizer._post_quantize_latent_for_transmission = lambda latent: latent + 1  # type: ignore[method-assign]

    sampled_latent = torch.zeros(1, 2, 3, 3)
    decoder_inputs = tokenizer._prepare_single_decoder_latent_inputs(sampled_latent)

    expected = sampled_latent + 1
    assert torch.equal(decoder_inputs["to_dec"], expected)
    assert torch.equal(decoder_inputs["to_dec_clean"], expected)
    assert decoder_inputs["to_dec_is_dual_latent"] is False
