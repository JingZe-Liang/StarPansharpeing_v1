from __future__ import annotations

import contextlib
import sys
import types
from pathlib import Path
from typing import Any

import accelerate
import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parents[2]))


def ensure_litdata_stub() -> None:
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


def get_trainer_cls():
    ensure_litdata_stub()
    from scripts.trainer.hyperspectral_image_tokenizer_trainer import CosmosHyperspectralTokenizerTrainer

    return CosmosHyperspectralTokenizerTrainer


class DummyAccelerator:
    def __init__(self) -> None:
        self.distributed_type = accelerate.utils.DistributedType.NO
        self.is_fsdp2 = False
        self.sync_gradients = False
        self.use_distributed = False
        self.is_main_process = True
        self.device = torch.device("cpu")
        self.state = types.SimpleNamespace(fsdp_plugin=None)
        self.prepare_calls: list[tuple[Any, ...]] = []

    def autocast(self):
        return contextlib.nullcontext()

    def accumulate(self, *models):
        _ = models
        return contextlib.nullcontext()

    def unwrap_model(self, model):
        return model

    def backward(self, loss):
        _ = loss
        return None

    def prepare(self, *objs):
        self.prepare_calls.append(objs)
        if len(objs) == 1:
            return objs[0]
        return objs


class DummyOptim:
    def __init__(self) -> None:
        self.zero_grad_called = 0
        self.step_called = 0
        self.param_groups = [{"lr": 1e-4}]

    def zero_grad(self) -> None:
        self.zero_grad_called += 1

    def step(self) -> None:
        self.step_called += 1

    def train(self) -> None:
        return None

    def eval(self) -> None:
        return None


class DummySched:
    def __init__(self) -> None:
        self.step_called = 0

    def step(self) -> None:
        self.step_called += 1
