from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import torch

from scripts.trainer.hyper_latent_change_detection_trainer import CDModelStepOutput
from scripts.trainer.single_hyper_image_change_detection_trainer import SingleHyperImageCDTrainer


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

    def autocast(self):
        return nullcontext()


def test_single_cd_train_step_uses_single_image_accumulation() -> None:
    trainer = SingleHyperImageCDTrainer.__new__(SingleHyperImageCDTrainer)
    trainer.accelerator = _DummyAccelerator()
    trainer.model = object()
    trainer.macro_batch_size = 7
    trainer.train_state = _DummySteps()
    trainer.train_cfg = SimpleNamespace(
        log=SimpleNamespace(log_every=100),
        max_steps=1000,
    )
    trainer.optim = SimpleNamespace(param_groups=[{"lr": 1e-3}])
    trainer.log_msg = lambda *args, **kwargs: None
    trainer.tenb_log_any = lambda *args, **kwargs: None
    trainer._cast_to_dtype = lambda x: x

    called: dict[str, int] = {}

    def _single_step(batch: dict, macro_batch_size: int) -> CDModelStepOutput:
        called["count"] = called.get("count", 0) + 1
        called["mbs"] = macro_batch_size
        return CDModelStepOutput(
            pred_pixel=torch.randn(1, 2, 8, 8),
            loss=torch.tensor(1.0),
            log_losses={"seg_loss": torch.tensor(1.0)},
        )

    trainer.train_single_img_accum_step = _single_step
    trainer.train_step({"img1": torch.randn(1, 4, 8, 8), "img2": torch.randn(1, 4, 8, 8), "gt": torch.zeros(1, 8, 8)})

    assert called["count"] == 1
    assert called["mbs"] == 7
    assert trainer.global_step == 1


def test_single_cd_val_step_runs_standard_inference_path() -> None:
    trainer = SingleHyperImageCDTrainer.__new__(SingleHyperImageCDTrainer)
    trainer.accelerator = _DummyAccelerator()
    trainer.train_cfg = SimpleNamespace(val_slide_window=False, val_slide_window_kwargs={})
    trainer.no_ema = True

    def _should_not_call(*args, **kwargs):
        raise AssertionError("single-image val_step should not call _macro_forward_seg_model")

    trainer._macro_forward_seg_model = _should_not_call

    trainer.model = lambda pair: torch.randn(pair[0].shape[0], 2, pair[0].shape[-2], pair[0].shape[-1])

    batch = {
        "img1": torch.randn(2, 8, 16, 16),
        "img2": torch.randn(2, 8, 16, 16),
        "gt": torch.randint(0, 2, (2, 16, 16)),
    }
    pred = trainer.val_step(batch)

    assert pred.shape == (2, 16, 16)


def test_single_cd_train_single_step_passes_4d_center_gt_to_loss() -> None:
    trainer = SingleHyperImageCDTrainer.__new__(SingleHyperImageCDTrainer)
    trainer.macro_batch_size = 4
    trainer._macro_forward_seg_model = lambda batch, macro_batch_size: [torch.randn(3, 2, 64, 64)]  # noqa: ARG005
    trainer._optimize_step = lambda loss: None  # noqa: ARG005

    captured: dict[str, torch.Size] = {}

    def _fake_compute_seg_loss(pred_for_loss, gt_for_loss):
        captured["pred_shape"] = pred_for_loss.shape
        captured["gt_shape"] = gt_for_loss.shape
        return torch.tensor(1.0), {"seg_loss": torch.tensor(1.0)}

    trainer.compute_segmentation_loss = _fake_compute_seg_loss
    batch = {
        "img1": torch.randn(3, 8, 64, 64),
        "img2": torch.randn(3, 8, 64, 64),
        "gt": torch.randint(0, 2, (3, 64, 64)),
    }

    trainer.train_single_img_accum_step(batch, macro_batch_size=4)

    assert captured["pred_shape"] == torch.Size([3, 2, 1, 1])
    assert captured["gt_shape"] == torch.Size([3, 1, 1, 1])
