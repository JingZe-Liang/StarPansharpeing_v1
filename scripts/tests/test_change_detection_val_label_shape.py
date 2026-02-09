import torch

from scripts.trainer.hyper_latent_change_detection_trainer import HyperCDTrainer


def test_to_label_map_3d_accepts_3d_or_single_channel_4d() -> None:
    trainer = HyperCDTrainer.__new__(HyperCDTrainer)

    x3 = torch.zeros(2, 64, 64, dtype=torch.long)
    x4 = torch.zeros(2, 1, 64, 64, dtype=torch.long)

    y3 = trainer._to_label_map_3d(x3)
    y4 = trainer._to_label_map_3d(x4)

    assert y3.shape == (2, 64, 64)
    assert y4.shape == (2, 64, 64)


def test_to_label_map_3d_rejects_invalid_shape() -> None:
    trainer = HyperCDTrainer.__new__(HyperCDTrainer)
    x = torch.zeros(2, 2, 64, 64, dtype=torch.long)

    try:
        trainer._to_label_map_3d(x)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for invalid label shape")
