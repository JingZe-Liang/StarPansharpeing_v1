from __future__ import annotations

import types

import torch
import torch.nn as nn

from scripts.tests._trainer_non_disc_test_utils import DummyAccelerator, DummyOptim, DummySched, get_trainer_cls


def test_trainer_non_disc_no_disc_objects() -> None:
    trainer_cls = get_trainer_cls()
    trainer = trainer_cls.__new__(trainer_cls)

    trainer._use_disc = False
    trainer._is_fsdp = False
    trainer.no_ema = True
    trainer.device = torch.device("cpu")
    trainer.dtype = torch.float32
    trainer.use_training_aug = False
    trainer.proxy_model = None
    trainer.log_msg = lambda *args, **kwargs: None
    trainer.tokenizer = nn.Linear(4, 4)
    trainer.tokenizer_optim = DummyOptim()
    trainer.tokenizer_sched = DummySched()
    trainer.disc_optim = None
    trainer.disc_sched = None
    trainer.accelerator = DummyAccelerator()
    trainer.vq_loss_fn = types.SimpleNamespace(discriminator=None, use_repa=False, use_vf=False, use_gram=False)

    trainer.prepare_for_training()
    trainer.prepare_ema_models()

    assert trainer.disc_optim is None
    assert trainer.disc_sched is None
    assert trainer.ema_vq_disc is None
    assert all(len(call) == 1 or call[0] is trainer.tokenizer for call in trainer.accelerator.prepare_calls)
