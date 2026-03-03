from __future__ import annotations

import contextlib
import types

import torch

from scripts.tests._trainer_non_disc_test_utils import DummyAccelerator, DummyOptim, DummySched, get_trainer_cls


class _TrainState:
    def __init__(self) -> None:
        self._state = {"train": 0}

    def update(self, mode: str) -> None:
        self._state[mode] += 1

    def __getitem__(self, mode: str) -> int:
        return self._state[mode]


def test_trainer_non_disc_train_step_skip_disc() -> None:
    trainer_cls = get_trainer_cls()
    trainer = trainer_cls.__new__(trainer_cls)

    trainer._use_disc = False
    trainer._has_proxy_task = False
    trainer.use_training_aug = False
    trainer.device = torch.device("cpu")
    trainer.dtype = torch.float32
    trainer.accelerator = DummyAccelerator()
    trainer.tokenizer = torch.nn.Identity()
    trainer.tokenizer_optim = DummyOptim()
    trainer.tokenizer_sched = DummySched()
    trainer.disc_optim = None
    trainer.disc_sched = None
    trainer.vq_loss_fn = types.SimpleNamespace(discriminator=None)
    trainer.proxy_model = None
    trainer.train_state = _TrainState()
    trainer.train_cfg = types.SimpleNamespace(
        track_metrics_duration=-1,
        track_metrics_after=0,
        proxy_task_train_type="unified",
        log=types.SimpleNamespace(log_every=1000, visualize_every=1000),
        max_steps=10,
    )
    trainer.log_msg = lambda *args, **kwargs: None
    trainer.tenb_log_any = lambda *args, **kwargs: None
    trainer._record_time = lambda *_args, **_kwargs: contextlib.nullcontext()
    trainer.forward_tokenizer = lambda x: {"recon": x, "latent": x}
    trainer.train_tokenizer_step = lambda x, out_d, proxy_aug_future=None: (
        torch.tensor(1.0),
        {"gen_loss": torch.tensor(1.0)},
        None,
    )
    trainer.train_proxy_model_step = lambda x, proxy_aug_future=None: None
    trainer.proxy_aug_manager = types.SimpleNamespace(maybe_prefetch=lambda *args, **kwargs: None)
    trainer.to_rgb = lambda x: x
    trainer._filter_item_value_in_dict = lambda d, to_item=True: {}

    def _should_not_call(*args, **kwargs):
        raise AssertionError("train_disc_step should not be called when discriminator is disabled")

    trainer.train_disc_step = _should_not_call

    batch = {"img": torch.zeros(2, 3, 8, 8)}
    trainer.train_step(batch)

    assert trainer.global_step == 1
