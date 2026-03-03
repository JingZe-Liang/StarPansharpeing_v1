from __future__ import annotations

import contextlib
import sys
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parents[2]))


class _DummyAccelerator:
    def autocast(self):
        return contextlib.nullcontext()

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        return model


class _DummyTokenizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._is_deep_supervision = False
        self._use_repa_loss = True

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"latent": x, "recon": x}

    def get_repa_feature(self):
        return torch.zeros(2, 4, 2, 2)

    def get_repa_feature_dict(self, force_to: bool = False) -> dict[str, list[torch.Tensor]]:
        _ = force_to
        return {"dino": [torch.zeros(2, 4, 2, 2)]}


class _DummyVQLoss:
    def __init__(self, use_phis: bool) -> None:
        self.use_phis = use_phis
        self.last_kwargs: dict[str, Any] = {}
        self.discriminator = nn.Identity()

    def __call__(self, **kwargs: Any):
        self.last_kwargs = kwargs
        return {"gen_loss": torch.tensor(1.0), "q_loss": torch.tensor(0.0), "disc_loss": torch.tensor(0.0)}, {}


def _build_trainer(use_phis: bool) -> Any:
    from scripts.trainer.hyperspectral_image_tokenizer_trainer import CosmosHyperspectralTokenizerTrainer

    trainer = CosmosHyperspectralTokenizerTrainer.__new__(CosmosHyperspectralTokenizerTrainer)
    trainer.accelerator = _DummyAccelerator()
    trainer.no_ema = True
    trainer.tokenizer = _DummyTokenizer()
    trainer.use_quantizer = False
    trainer.vq_loss_fn = _DummyVQLoss(use_phis=use_phis)
    trainer.train_state = {"train": 0}
    trainer._record_time = lambda _: contextlib.nullcontext()
    trainer.get_last_layer = lambda mode="dec": torch.ones(1, requires_grad=True)
    return trainer


def _ensure_litdata_stub() -> None:
    if "litdata" in sys.modules:
        return

    class _Serializer:
        def serialize(self, obj):
            _ = obj
            return b"", ""

        def deserialize(self, data):
            _ = data
            return None

    class _TIFFSerializer(_Serializer):
        pass

    class _JPEGSerializer(_Serializer):
        pass

    litdata = types.ModuleType("litdata")
    litdata.CombinedStreamingDataset = object
    litdata.ParallelStreamingDataset = object
    litdata.StreamingDataLoader = object
    litdata.StreamingDataset = object
    streaming = types.ModuleType("litdata.streaming")
    serializers = types.ModuleType("litdata.streaming.serializers")
    serializers.Serializer = _Serializer
    serializers.TIFFSerializer = _TIFFSerializer
    serializers.JPEGSerializer = _JPEGSerializer
    serializers._SERIALIZERS = {}
    streaming.serializers = serializers
    litdata.streaming = streaming
    sys.modules["litdata"] = litdata
    sys.modules["litdata.streaming"] = streaming
    sys.modules["litdata.streaming.serializers"] = serializers


def test_forward_tokenizer_sets_phis_student_feature_when_enabled() -> None:
    _ensure_litdata_stub()
    trainer = _build_trainer(use_phis=True)
    out_d = trainer.forward_tokenizer(torch.zeros(2, 3, 8, 8))
    assert "phis_student_feature" in out_d
    assert "dino" in out_d["phis_student_feature"]

    trainer.forward_discriminator(torch.zeros(2, 3, 8, 8), out_d, train_tokenizer=True)
    assert "phis_student_feature" in trainer.vq_loss_fn.last_kwargs
    assert trainer.vq_loss_fn.last_kwargs["phis_student_feature"] == out_d["phis_student_feature"]


def test_forward_tokenizer_skips_phis_student_feature_when_disabled() -> None:
    _ensure_litdata_stub()
    trainer = _build_trainer(use_phis=False)
    out_d = trainer.forward_tokenizer(torch.zeros(2, 3, 8, 8))
    assert "phis_student_feature" not in out_d
