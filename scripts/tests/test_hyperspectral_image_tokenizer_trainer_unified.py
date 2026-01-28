from __future__ import annotations

import ast
from pathlib import Path


def _trainer_path() -> Path:
    # scripts/tests -> scripts/trainer/...
    return Path(__file__).resolve().parents[1] / "trainer" / "hyperspectral_image_tokenizer_trainer.py"


def test_trainer_file_parses() -> None:
    src = _trainer_path().read_text(encoding="utf-8")
    ast.parse(src)


def test_trainer_has_no_separate_enc_dec_runtime_branch() -> None:
    src = _trainer_path().read_text(encoding="utf-8")
    assert "self.sep_enc_dec" not in src
    assert "tokenizer_encoder" not in src
    assert "tokenizer_decoder" not in src
    assert "ema_encoder" not in src
    assert "ema_decoder" not in src
    assert "load_jit_model_shape_matched" not in src


def test_trainer_rejects_separate_enc_dec_config() -> None:
    src = _trainer_path().read_text(encoding="utf-8")
    assert "train.seperate_enc_dec=True" in src
