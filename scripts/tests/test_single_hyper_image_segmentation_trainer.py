from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import torch

from scripts.trainer.hyper_latent_segmentation_trainer import HyperSegmentationTrainer, SegmentationOutput
from scripts.trainer.single_hyper_image_segmentation_trainer import SingleHyperImageSegmentationTrainer


class _DummySteps:
    def __init__(self) -> None:
        self._state = {"train": 0, "val": 0}

    def update(self, mode: str = "train") -> None:
        self._state[mode] += 1

    def __getitem__(self, mode: str) -> int:
        return self._state[mode]


class _DummyAccelerator:
    def accumulate(self, _model):
        return nullcontext()


class _DummySingleImageDataset:
    def __init__(self) -> None:
        self.full_img = False
        self.patch_size = 32

    def _get_full_img_tensor(self):
        img = torch.randn(8, 16, 16)
        gt = torch.randint(0, 3, (16, 16), dtype=torch.long)
        return {"img": img, "gt": gt}


def test_initialize_indexing_mode_falls_back_to_standard_with_warning() -> None:
    trainer = HyperSegmentationTrainer.__new__(HyperSegmentationTrainer)
    trainer.train_cfg = SimpleNamespace(use_single_img_indexing=True, macro_batch_size=7)
    logs: list[tuple[str, str]] = []

    def _log(msg: str, **kwargs) -> None:
        logs.append((msg, kwargs.get("level", "INFO")))

    trainer.log_msg = _log
    trainer._initialize_indexing_mode()

    assert trainer._use_single_img_indexing is False
    assert trainer.macro_batch_size == 7
    assert any("deprecated" in msg and level == "WARNING" for msg, level in logs)


def test_single_trainer_train_step_consumes_patch_batch() -> None:
    trainer = SingleHyperImageSegmentationTrainer.__new__(SingleHyperImageSegmentationTrainer)
    trainer.accelerator = _DummyAccelerator()
    trainer.model = object()
    trainer.train_state = _DummySteps()
    trainer.train_cfg = SimpleNamespace(
        log=SimpleNamespace(log_every=100),
        max_steps=1000,
    )
    trainer.optim = SimpleNamespace(param_groups=[{"lr": 1e-3}])
    trainer.log_msg = lambda *args, **kwargs: None
    trainer.tenb_log_any = lambda *args, **kwargs: None

    captured: dict[str, torch.Tensor] = {}

    def _train_segment_step(img: torch.Tensor, gt: torch.Tensor) -> SegmentationOutput:
        captured["img"] = img
        captured["gt"] = gt
        pred = torch.randn(img.shape[0], 3, img.shape[-2], img.shape[-1])
        return SegmentationOutput(loss=torch.tensor(1.0), pred_pixel=pred)

    trainer.train_segment_step = _train_segment_step

    img = torch.randn(4, 8, 32, 32)
    gt = torch.randint(0, 3, (4, 32, 32), dtype=torch.long)
    trainer.train_step({"img": img, "gt": gt})

    assert torch.equal(captured["img"], img)
    assert torch.equal(captured["gt"], gt)
    assert trainer.global_step == 1


def test_single_trainer_validation_uses_full_image_batch() -> None:
    trainer = SingleHyperImageSegmentationTrainer.__new__(SingleHyperImageSegmentationTrainer)
    trainer.ds_is_single_image = True
    trainer.device = torch.device("cpu")
    trainer.val_dataloader = object()
    trainer.val_dataset = _DummySingleImageDataset()
    trainer.val_cfg = SimpleNamespace(max_val_iters=10)
    trainer.log_msg = lambda *args, **kwargs: None

    batches = list(trainer.get_val_loader_iter())
    assert len(batches) == 1
    batch = batches[0]
    assert batch["img"].ndim == 4
    assert batch["gt"].ndim == 3
    assert batch["img"].shape[0] == 1
    assert batch["gt"].shape[0] == 1
