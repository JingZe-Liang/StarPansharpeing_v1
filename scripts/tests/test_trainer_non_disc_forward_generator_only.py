from __future__ import annotations

import contextlib
import types

import torch

from scripts.tests._trainer_non_disc_test_utils import DummyAccelerator, get_trainer_cls


class _DummyVQLoss:
    def __init__(self) -> None:
        self.kwargs: dict = {}

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return {"gen_loss": torch.tensor(2.0), "q_loss": torch.tensor(1.0)}, {"nll_loss": torch.tensor(0.1)}


def test_trainer_non_disc_forward_generator_only() -> None:
    trainer_cls = get_trainer_cls()
    trainer = trainer_cls.__new__(trainer_cls)

    trainer._use_disc = False
    trainer.accelerator = DummyAccelerator()
    trainer.vq_loss_fn = _DummyVQLoss()
    trainer._record_time = lambda *_args, **_kwargs: contextlib.nullcontext()
    trainer.get_last_layer = lambda mode="dec": torch.ones(1, requires_grad=True)
    trainer.train_state = {"train": 0}

    x = torch.zeros(2, 3, 8, 8)
    out_d = {"recon": x, "latent": x}

    loss, log = trainer.forward_generator_only(x, out_d, split="train")

    assert torch.isclose(loss, torch.tensor(3.0))
    assert trainer.vq_loss_fn.kwargs["optimizer_idx"] == 0
    assert "nll_loss" in log
