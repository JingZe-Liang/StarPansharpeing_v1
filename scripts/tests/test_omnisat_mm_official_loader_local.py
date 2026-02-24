from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.stage2.classification.data.TreeSatAI_ts_official import TreeSAT
from src.stage2.classification.model import OmniSatMMClassifier


def test_omnisat_mm_with_official_loader_local() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = repo_root / "scripts/configs/classification/dataset/treesatai_ts_official.yaml"
    cfg = OmegaConf.load(cfg_path)

    root = repo_root / str(cfg.root)
    if not root.exists():
        return

    raw_dataset_kwargs = OmegaConf.to_container(cfg.train.dataset_kwargs, resolve=True)
    raw_loader_kwargs = OmegaConf.to_container(cfg.train.loader_kwargs, resolve=True)
    assert isinstance(raw_dataset_kwargs, dict)
    assert isinstance(raw_loader_kwargs, dict)
    dataset_kwargs: dict[str, object] = {str(k): v for k, v in raw_dataset_kwargs.items()}
    loader_kwargs: dict[str, object] = {str(k): v for k, v in raw_loader_kwargs.items()}

    loader_kwargs["batch_size"] = 1
    loader_kwargs["num_workers"] = 0
    loader_kwargs["shuffle"] = False

    _, dataloader = TreeSAT.create_dataloader(dataset_kwargs=dataset_kwargs, loader_kwargs=loader_kwargs)
    batch = next(iter(dataloader))

    model = OmniSatMMClassifier(
        num_classes=int(cfg.consts.num_classes),
        modalities=tuple(cfg.modalities),
        embed_dim=128,
        depth=6,
        num_heads=8,
    )
    model.eval()

    with torch.no_grad():
        out = model(batch)

    assert isinstance(out, dict)
    assert "logits" in out
    assert out["logits"].shape == (1, int(cfg.consts.num_classes))
    assert not torch.isnan(out["logits"]).any()
