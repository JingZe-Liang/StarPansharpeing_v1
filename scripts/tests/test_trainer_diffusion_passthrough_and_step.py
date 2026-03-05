from __future__ import annotations

import contextlib
import types

import torch
import torch.nn as nn

from scripts.tests._trainer_non_disc_test_utils import DummyAccelerator, DummyOptim, DummySched, get_trainer_cls


class _DummyVQLoss:
    def __init__(self) -> None:
        self.kwargs: dict = {}

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return {"gen_loss": torch.tensor(2.0), "q_loss": torch.tensor(1.0)}, {"nll_loss": torch.tensor(0.1)}


def test_trainer_diffusion_passthrough_forward_generator_only() -> None:
    trainer_cls = get_trainer_cls()
    trainer = trainer_cls.__new__(trainer_cls)

    trainer._use_disc = False
    trainer._use_diffusion = True
    trainer.diffusion_model = nn.Conv2d(8, 8, kernel_size=1)
    trainer.accelerator = DummyAccelerator()
    trainer.vq_loss_fn = _DummyVQLoss()
    trainer._record_time = lambda *_args, **_kwargs: contextlib.nullcontext()
    trainer.get_last_layer = lambda mode="dec": torch.ones(1, requires_grad=True)
    trainer.train_state = {"train": 0}

    x = torch.zeros(2, 3, 8, 8)
    out_d = {"recon": x, "latent": torch.zeros(2, 8, 2, 2)}

    loss, _log = trainer.forward_generator_only(x, out_d, split="train")
    assert torch.isclose(loss, torch.tensor(3.0))
    assert trainer.vq_loss_fn.kwargs["diffusion_model"] is trainer.diffusion_model


def test_trainer_diffusion_train_tokenizer_step_updates_diffusion_optim() -> None:
    trainer_cls = get_trainer_cls()
    trainer = trainer_cls.__new__(trainer_cls)

    trainer._use_disc = False
    trainer._use_diffusion = True
    trainer._has_proxy_task = False
    trainer.use_training_aug = False
    trainer.accelerator = DummyAccelerator()
    trainer.tokenizer = nn.Identity()
    trainer.train_cfg = types.SimpleNamespace(proxy_task_train_type="unified", grad_check=False)
    trainer._record_time = lambda *_args, **_kwargs: contextlib.nullcontext()
    trainer.tokenizer_optim = DummyOptim()
    trainer.tokenizer_sched = DummySched()
    trainer.diffusion_optim = DummyOptim()
    trainer.diffusion_sched = DummySched()
    trainer.antideg_net = None
    trainer.ema_update = lambda mode="tokenizer": None
    trainer.forward_generator_only = lambda x, out_d, split="train": (
        torch.tensor(1.0, requires_grad=True),
        {},
    )

    x = torch.zeros(2, 3, 8, 8)
    tok_dict: dict = {}
    trainer.train_tokenizer_step(x, tok_dict)

    assert trainer.diffusion_optim.zero_grad_called == 1
    assert trainer.diffusion_optim.step_called == 1
    assert trainer.diffusion_sched.step_called == 1


def test_trainer_diffusion_optional_compat_when_disabled() -> None:
    trainer_cls = get_trainer_cls()
    trainer = trainer_cls.__new__(trainer_cls)

    trainer._use_disc = False
    trainer._use_diffusion = False
    trainer.accelerator = DummyAccelerator()
    trainer.vq_loss_fn = _DummyVQLoss()
    trainer._record_time = lambda *_args, **_kwargs: contextlib.nullcontext()
    trainer.get_last_layer = lambda mode="dec": torch.ones(1, requires_grad=True)
    trainer.train_state = {"train": 0}

    x = torch.zeros(2, 3, 8, 8)
    out_d = {"recon": x, "latent": torch.zeros(2, 8, 2, 2)}

    loss, _log = trainer.forward_generator_only(x, out_d, split="train")

    assert torch.isclose(loss, torch.tensor(3.0))
    assert trainer.vq_loss_fn.kwargs["diffusion_model"] is None
