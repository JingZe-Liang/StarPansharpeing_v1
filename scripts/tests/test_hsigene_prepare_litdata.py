from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.dataset.HISGene.prepare_litdata_and_conditions import (
    configure_caption_backend_env,
    repair_litdata_index_caption_format,
    repair_litdata_index_img_format,
    split_paths,
    stratified_split_groups,
)


def _fake_paths(n: int) -> list[Path]:
    return [Path(f"sample_{i:04d}.tiff") for i in range(n)]


def test_split_paths_ratio_disjoint_and_cover() -> None:
    paths = _fake_paths(10)
    train_paths, test_paths = split_paths(paths, test_ratio=0.2, seed=2026, split_key="mdas_hyspex")

    assert len(train_paths) == 8
    assert len(test_paths) == 2
    assert set(train_paths).isdisjoint(set(test_paths))
    assert set(train_paths).union(set(test_paths)) == set(paths)


def test_split_paths_small_dataset_keeps_train_non_empty() -> None:
    train_paths, test_paths = split_paths(_fake_paths(1), test_ratio=0.2, seed=2026, split_key="xiongan")
    assert len(train_paths) == 1
    assert len(test_paths) == 0


def test_split_paths_deterministic() -> None:
    paths = _fake_paths(37)
    train_a, test_a = split_paths(paths, test_ratio=0.2, seed=123, split_key="houston")
    train_b, test_b = split_paths(paths, test_ratio=0.2, seed=123, split_key="houston")

    assert train_a == train_b
    assert test_a == test_b


def test_stratified_split_groups_keeps_group_ratio() -> None:
    group_to_paths = {
        "chikusei": _fake_paths(10),
        "houston": [Path(f"hou_{i:04d}.tiff") for i in range(15)],
    }
    train_paths, test_paths, stats = stratified_split_groups(group_to_paths, test_ratio=0.2, seed=2026)

    assert len(train_paths) == 20
    assert len(test_paths) == 5
    assert stats["chikusei"]["train"] == 8
    assert stats["chikusei"]["test"] == 2
    assert stats["houston"]["train"] == 12
    assert stats["houston"]["test"] == 3


def test_repair_litdata_index_img_format(tmp_path: Path) -> None:
    index_path = tmp_path / "index.json"
    payload = {
        "config": {
            "data_format": ["str", "bytes"],
            "data_spec": '[1, {"type": "builtins.dict", "context": "[\\"__key__\\", \\"img\\"]", "children_spec": []}]',
        }
    }
    index_path.write_text(json.dumps(payload), encoding="utf-8")

    changed = repair_litdata_index_img_format(index_path)
    assert changed is True

    updated = json.loads(index_path.read_text(encoding="utf-8"))
    assert updated["config"]["data_format"] == ["str", "tifffile"]


def test_repair_litdata_index_caption_format(tmp_path: Path) -> None:
    index_path = tmp_path / "index.json"
    payload = {
        "config": {
            "data_format": ["str", "bytes", "bytes"],
            "data_spec": (
                '[1, {"type": "builtins.dict", '
                '"context": "[\\"__key__\\", \\"caption.json\\", \\"features.safetensors\\"]", '
                '"children_spec": []}]'
            ),
        }
    }
    index_path.write_text(json.dumps(payload), encoding="utf-8")

    changed = repair_litdata_index_caption_format(index_path)
    assert changed is True

    updated = json.loads(index_path.read_text(encoding="utf-8"))
    assert updated["config"]["data_format"] == ["str", "json", "bytes"]
    spec = json.loads(updated["config"]["data_spec"])
    keys = json.loads(spec[1]["context"])
    assert keys == ["__key__", "caption", "features.safetensors"]


def test_configure_caption_backend_env(monkeypatch: Any) -> None:
    args = argparse.Namespace(
        caption_backend="internvl35",
        caption_ckpt="/Data/ZiHanCao/checkpoints/Qwen/Qwen2___5-VL-7B-Instruct",
        hf_cache_dir="/Data/ZiHanCao/checkpoints",
        caption_local_files_only=True,
    )

    monkeypatch.delenv("CAPTION_BACKEND", raising=False)
    monkeypatch.delenv("QWEN25VL_CKPT", raising=False)
    monkeypatch.delenv("QWEN25VL_LOCAL_ONLY", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)

    configure_caption_backend_env(args)
    assert os.environ["CAPTION_BACKEND"] == "internvl35"
    assert os.environ["QWEN25VL_CKPT"] == args.caption_ckpt
    assert os.environ["QWEN25VL_LOCAL_ONLY"] == "1"
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ["HF_HOME"] == args.hf_cache_dir
    assert os.environ["TRANSFORMERS_CACHE"] == args.hf_cache_dir
    assert os.environ["HUGGINGFACE_HUB_CACHE"] == args.hf_cache_dir
