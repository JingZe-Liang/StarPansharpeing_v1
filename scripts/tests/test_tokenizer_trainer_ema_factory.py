import sys
from pathlib import Path

import hydra
import torch
import torch.nn as nn
from ema_pytorch import EMA
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.trainer.hyperspectral_image_tokenizer_trainer import CosmosHyperspectralTokenizerTrainer


class _NonCopyableModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0]))

    def __deepcopy__(self, memo):
        raise RuntimeError("deepcopy disabled for test")


class _CloneModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.0]))


def test_prepare_ema_models_uses_factory_for_tokenizer(monkeypatch):
    trainer = CosmosHyperspectralTokenizerTrainer.__new__(CosmosHyperspectralTokenizerTrainer)
    trainer.cfg = OmegaConf.create(
        {
            "ema": {
                "_target_": "ema_pytorch.EMA",
                "_partial_": True,
                "beta": 0.999,
                "update_after_step": 0,
                "update_every": 1,
                "use_foreach": False,
                "allow_different_devices": True,
            }
        }
    )
    trainer.tokenizer_cfg = OmegaConf.create({"_target_": "tests.dummy.tokenizer"})
    trainer.tokenizer = _NonCopyableModule()
    trainer.device = torch.device("cpu")
    trainer.no_ema = False
    trainer._use_disc = False
    trainer.proxy_model = None

    def _instantiate(cfg, *args, **kwargs):
        target = cfg.get("_target_")
        if target == "ema_pytorch.EMA":
            return lambda model, **ema_kwargs: EMA(model, **ema_kwargs)
        if target == "tests.dummy.tokenizer":
            return _CloneModule()
        raise AssertionError(f"Unexpected instantiate target: {target}")

    monkeypatch.setattr(hydra.utils, "instantiate", _instantiate)

    trainer.prepare_ema_models()
    assert isinstance(trainer.ema_tokenizer.ema_model, _CloneModule)

    trainer.ema_tokenizer.update()

    online_weight = next(trainer.tokenizer.parameters())
    ema_weight = next(trainer.ema_tokenizer.ema_model.parameters())
    assert torch.equal(ema_weight, online_weight)
