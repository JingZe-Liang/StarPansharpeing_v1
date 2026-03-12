from pathlib import Path
import sys

import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.stage1.discretization.collections.bsq import BinarySphericalQuantizer


def test_bsq_flip_aug_flips_only_returned_latent_in_training():
    inputs = torch.tensor(
        [
            [
                [[0.2, -0.3], [0.8, 0.1]],
                [[0.5, -0.7], [-0.4, -0.2]],
                [[-0.6, 0.9], [0.4, -0.1]],
                [[0.3, -0.5], [-0.8, 0.6]],
            ]
        ],
        dtype=torch.float32,
    )

    base = BinarySphericalQuantizer(
        embed_dim=4,
        beta=0.25,
        gamma0=1.0,
        gamma=1.0,
        zeta=1.0,
        input_format="bchw",
        soft_entropy=True,
        group_size=2,
        persample_entropy_compute="analytical",
        cb_entropy_compute="group",
        l2_norm=True,
        flip_bit_prob=0.0,
    )
    flip = BinarySphericalQuantizer(
        embed_dim=4,
        beta=0.25,
        gamma0=1.0,
        gamma=1.0,
        zeta=1.0,
        input_format="bchw",
        soft_entropy=True,
        group_size=2,
        persample_entropy_compute="analytical",
        cb_entropy_compute="group",
        l2_norm=True,
        flip_bit_prob=1.0,
    )
    base.train()
    flip.train()

    base_zq, base_loss, base_info = base(inputs)
    flip_zq, flip_loss, flip_info = flip(inputs)

    assert torch.allclose(flip_zq, -base_zq)
    assert torch.allclose(flip_loss, base_loss)
    assert torch.equal(flip_info.indices, base_info.indices)
    assert torch.equal(flip_info.group_indices, base_info.group_indices)


@torch.no_grad()
def test_bsq_flip_aug_is_disabled_in_eval():
    inputs = torch.tensor(
        [
            [
                [[0.2, -0.3], [0.8, 0.1]],
                [[0.5, -0.7], [-0.4, -0.2]],
                [[-0.6, 0.9], [0.4, -0.1]],
                [[0.3, -0.5], [-0.8, 0.6]],
            ]
        ],
        dtype=torch.float32,
    )

    model = BinarySphericalQuantizer(
        embed_dim=4,
        beta=0.25,
        gamma0=1.0,
        gamma=1.0,
        zeta=1.0,
        input_format="bchw",
        soft_entropy=True,
        group_size=2,
        persample_entropy_compute="analytical",
        cb_entropy_compute="group",
        l2_norm=True,
        flip_bit_prob=1.0,
    )
    baseline = BinarySphericalQuantizer(
        embed_dim=4,
        beta=0.25,
        gamma0=1.0,
        gamma=1.0,
        zeta=1.0,
        input_format="bchw",
        soft_entropy=True,
        group_size=2,
        persample_entropy_compute="analytical",
        cb_entropy_compute="group",
        l2_norm=True,
        flip_bit_prob=0.0,
    )
    model.eval()
    baseline.eval()

    eval_zq, _, eval_info = model(inputs)
    baseline_zq, _, baseline_info = baseline(inputs)

    assert torch.allclose(eval_zq, baseline_zq)
    assert torch.equal(eval_info.indices, baseline_info.indices)
    assert torch.equal(eval_info.group_indices, baseline_info.group_indices)
