from __future__ import annotations

import types

import torch
import torch.nn as nn

from scripts.tests._trainer_non_disc_test_utils import DummyAccelerator, DummyOptim, DummySched, get_trainer_cls


def test_trainer_disc_mode_regression_smoke() -> None:
    trainer_cls = get_trainer_cls()
    trainer = trainer_cls.__new__(trainer_cls)

    trainer._use_disc = True
    trainer.accelerator = DummyAccelerator()
    trainer.vq_loss_fn = types.SimpleNamespace(discriminator=nn.Conv2d(3, 4, kernel_size=1))
    trainer.disc_optim = DummyOptim()
    trainer.disc_sched = DummySched()
    trainer.ema_update = lambda mode="tokenizer": None
    trainer.gradient_check = lambda model: None
    trainer._raise_if_nonfinite_params = lambda model, model_name: None
    trainer.train_cfg = types.SimpleNamespace(grad_check=False)
    trainer.forward_discriminator = lambda x, tokenizer_out, train_tokenizer, split: (
        torch.tensor(1.0, requires_grad=True),
        {"disc_loss": torch.tensor(1.0)},
    )

    disc_loss, log_disc = trainer.train_disc_step(torch.zeros(2, 3, 8, 8), {"recon": torch.zeros(2, 3, 8, 8)})

    assert torch.is_tensor(disc_loss)
    assert "disc_loss" in log_disc
    assert trainer.disc_optim.zero_grad_called == 1
    assert trainer.disc_optim.step_called == 1
    assert trainer.disc_sched.step_called == 1
