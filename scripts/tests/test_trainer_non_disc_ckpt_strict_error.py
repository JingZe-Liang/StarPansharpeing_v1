from __future__ import annotations

import types
from pathlib import Path

import accelerate
import pytest
import torch.nn as nn

from scripts.tests._trainer_non_disc_test_utils import DummyAccelerator, get_trainer_cls


def test_trainer_non_disc_ckpt_strict_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    trainer_cls = get_trainer_cls()
    trainer = trainer_cls.__new__(trainer_cls)

    trainer._use_disc = False
    trainer._is_peft_tuning = False
    trainer._is_fsdp = False
    trainer._has_proxy_task = False
    trainer.train_cfg = types.SimpleNamespace(
        finetune_strategy="full",
        only_load_tokenizer=False,
        load_weight_shard=False,
    )
    trainer.tokenizer = nn.Identity()
    trainer.proxy_model = None
    trainer.log_msg = lambda *args, **kwargs: None
    trainer.prepare_ema_models = lambda: None
    trainer.accelerator = DummyAccelerator()
    trainer.accelerator.distributed_type = accelerate.utils.DistributedType.NO

    monkeypatch.setattr(accelerate.utils, "load_checkpoint_in_model", lambda *args, **kwargs: None)

    (tmp_path / "tokenizer").mkdir(parents=True, exist_ok=True)
    (tmp_path / "discriminator").mkdir(parents=True, exist_ok=True)

    with pytest.raises(RuntimeError, match="Non-disc strict mode"):
        trainer.load_from_ema_or_lora(tmp_path)
