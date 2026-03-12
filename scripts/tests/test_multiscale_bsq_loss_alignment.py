from pathlib import Path
import sys

import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.stage1.discretization.collections.multiscale_bsq import BSQ, MultiScaleBSQ, l2norm


def test_bsq_loss_matches_binary_spherical_quantizer_semantics():
    quantizer = BSQ(
        dim=4,
        codebook_size=16,
        commitment_loss_weight=0.25,
        gamma0=1.3,
        gamma=0.7,
        zeta=1.1,
        inv_temperature=2.0,
        new_quant=False,
        force_quantization_f32=False,
    )
    quantizer.train()

    inputs = torch.tensor(
        [
            [[0.2, -0.3, 0.5, -0.7], [0.8, 0.1, -0.4, -0.2]],
            [[-0.6, 0.9, 0.4, -0.1], [0.3, -0.5, -0.8, 0.6]],
        ],
        dtype=torch.float32,
    )

    ret, breakdown = quantizer(inputs, return_loss_breakdown=True)

    projected = quantizer.project_in(inputs)
    split = projected.reshape(projected.shape[0], projected.shape[1], quantizer.num_codebooks, quantizer.codebook_dim)
    normalized = l2norm(split)
    quantized = quantizer.quantize(normalized)
    expected_commitment = torch.mean(((quantized.detach() - normalized) ** 2).sum(dim=-1))
    expected_per_sample, expected_codebook, _ = quantizer.soft_entropy_loss(normalized)
    expected_entropy_penalty = quantizer.gamma0 * expected_per_sample - quantizer.gamma * expected_codebook
    expected_aux_loss = expected_commitment * quantizer.commitment_loss_weight + (
        quantizer.zeta * expected_entropy_penalty / quantizer.inv_temperature
    )

    assert breakdown.commitment.item() > 0
    assert torch.allclose(breakdown.commitment, expected_commitment)
    assert torch.allclose(breakdown.per_sample_entropy, expected_per_sample)
    assert torch.allclose(breakdown.batch_entropy, expected_codebook)
    assert torch.allclose(ret.entropy_aux_loss, expected_aux_loss)
    assert ret.indices.shape == (2, 2)
    assert ret.bit_indices.shape == (2, 2, 4)


@torch.no_grad()
def test_bsq_supports_both_quantize_paths():
    inputs = torch.tensor(
        [[[0.1, -0.2, 0.3, -0.4], [-0.5, 0.6, -0.7, 0.8]]],
        dtype=torch.float32,
    )

    quantizer_old = BSQ(dim=4, codebook_size=16, new_quant=False, force_quantization_f32=False)
    quantizer_new = BSQ(dim=4, codebook_size=16, new_quant=True, force_quantization_f32=False)
    quantizer_old.eval()
    quantizer_new.eval()

    ret_old = quantizer_old(inputs)
    ret_new = quantizer_new(inputs)

    assert ret_old.quantized.shape == inputs.shape
    assert ret_new.quantized.shape == inputs.shape
    assert ret_old.indices.shape == (1, 2)
    assert ret_new.indices.shape == (1, 2)
    assert ret_old.bit_indices.shape == (1, 2, 4)
    assert ret_new.bit_indices.shape == (1, 2, 4)


@torch.no_grad()
def test_multiscale_bsq_flip_happens_after_quantization():
    inputs = torch.tensor(
        [
            [
                [[0.2, -0.3], [0.4, -0.5]],
                [[-0.6, 0.7], [-0.8, 0.9]],
                [[0.1, 0.2], [-0.3, -0.4]],
                [[0.5, -0.6], [0.7, -0.8]],
            ]
        ],
        dtype=torch.float32,
    )
    scale_schedule = [(1, 2, 2)]

    base_quantizer = MultiScaleBSQ(
        dim=4,
        codebook_size=16,
        schedule_mode="same1",
        random_flip=False,
        new_quant=True,
        force_quantization_f32=False,
    )
    flip_quantizer = MultiScaleBSQ(
        dim=4,
        codebook_size=16,
        schedule_mode="same1",
        random_flip=True,
        flip_prob=1.0,
        max_flip_lvl=1,
        new_quant=True,
        force_quantization_f32=False,
    )
    base_quantizer.eval()
    flip_quantizer.eval()

    base_out = base_quantizer(inputs, scale_schedule=scale_schedule)
    flip_out = flip_quantizer(inputs, scale_schedule=scale_schedule)

    base_quantized, base_indices, base_bit_indices, _, base_losses, _, base_entropies = base_out
    flip_quantized, flip_indices, flip_bit_indices, _, flip_losses, _, flip_entropies = flip_out

    assert torch.allclose(flip_quantized, -base_quantized)
    assert len(base_indices) == len(flip_indices) == 1
    assert len(base_bit_indices) == len(flip_bit_indices) == 1
    assert base_losses.shape == flip_losses.shape == (1,)
    assert base_entropies.shape == flip_entropies.shape == (2, 1)
