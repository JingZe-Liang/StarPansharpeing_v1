from types import MethodType

import torch
from torch import nn

from src.stage2.classification.model.cosmos_tokenizer_classifier_ts import CosmosTokenizerClassifierTS


def _build_stub_model(freeze_tokenizer: bool) -> CosmosTokenizerClassifierTS:
    model = CosmosTokenizerClassifierTS.__new__(CosmosTokenizerClassifierTS)
    nn.Module.__init__(model)
    model.freeze_tokenizer = freeze_tokenizer
    model.modality_order = ("aerial", "s1_asc", "s1_des", "s2")
    model.fusion_dim = 16
    model.fusion_pool = "cls"
    model.modality_embedding = nn.Parameter(torch.zeros((4, 16)))
    model.fusion_cls_token = nn.Parameter(torch.zeros((1, 1, 16)))
    layer = nn.TransformerEncoderLayer(
        d_model=16,
        nhead=4,
        dim_feedforward=64,
        dropout=0.0,
        activation="gelu",
        batch_first=True,
    )
    model.fusion_encoder = nn.TransformerEncoder(layer, num_layers=1)
    model.classifier = nn.Linear(16, 5)
    model.latent_proj = nn.Identity()
    model.aerial_proj = nn.Linear(8, 16, bias=False)
    model.ltae_s1 = nn.Conv2d(2, 8, kernel_size=1)
    model.ltae_s1_des = model.ltae_s1
    model.ltae_s2 = nn.Conv2d(10, 8, kernel_size=1)
    model.s1_proj = nn.Linear(8, 16, bias=False)
    model.s2_proj = nn.Linear(8, 16, bias=False)
    model.tokenizer = nn.Linear(3, 3, bias=False)
    model.batch_key_type = "legacy"
    return model


def test_forward_shape_and_nan_free_with_valid() -> None:
    model = _build_stub_model(freeze_tokenizer=True)

    def _fake_encode_modality_token(
        self: CosmosTokenizerClassifierTS,
        batch: dict[str, object],
        modality: str,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del batch
        base = {"aerial": 1.0, "s1_asc": 2.0, "s1_des": 3.0, "s2": 4.0}[modality]
        token = torch.full((batch_size, self.fusion_dim), base, dtype=torch.float32, device=device)
        valid = torch.ones(batch_size, dtype=torch.bool, device=device)
        return token, valid

    model._encode_modality_token = MethodType(_fake_encode_modality_token, model)  # type: ignore[invalid-assignment]
    batch = {
        "image_aerial": torch.randn((2, 4, 8, 8)),
        "image_s1_asc": torch.randn((2, 3, 2, 8, 8)),
        "image_s1_asc_valid": torch.ones((2, 3), dtype=torch.bool),
        "image_s1_des": torch.randn((2, 3, 2, 8, 8)),
        "image_s1_des_valid": torch.ones((2, 3), dtype=torch.bool),
        "image_s2": torch.randn((2, 3, 10, 8, 8)),
        "image_s2_valid": torch.ones((2, 3), dtype=torch.bool),
    }
    output = model(batch)
    logits = output["logits"]
    assert logits.shape == (2, 5)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()


def test_forward_works_without_any_valid_mask() -> None:
    model = _build_stub_model(freeze_tokenizer=True)

    def _fake_encode_modality_token(
        self: CosmosTokenizerClassifierTS,
        batch: dict[str, object],
        modality: str,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del batch, modality
        token = torch.randn((batch_size, self.fusion_dim), device=device)
        valid = torch.ones(batch_size, dtype=torch.bool, device=device)
        return token, valid

    model._encode_modality_token = MethodType(_fake_encode_modality_token, model)  # type: ignore[invalid-assignment]
    batch = {
        "image_aerial": torch.randn((2, 4, 8, 8)),
        "image_s1_asc": torch.randn((2, 3, 2, 8, 8)),
        "image_s1_des": torch.randn((2, 3, 2, 8, 8)),
        "image_s2": torch.randn((2, 3, 10, 8, 8)),
    }
    output = model(batch)
    assert output["logits"].shape == (2, 5)


def test_named_parameters_freeze_excludes_tokenizer() -> None:
    model = _build_stub_model(freeze_tokenizer=True)
    names = [name for name, _ in model.named_parameters()]
    assert any(name.startswith("classifier.") for name in names)
    assert not any(name.startswith("tokenizer.") for name in names)


def test_named_parameters_unfreeze_includes_tokenizer() -> None:
    model = _build_stub_model(freeze_tokenizer=False)
    names = [name for name, _ in model.named_parameters()]
    assert any(name.startswith("tokenizer.") for name in names)


def test_state_dict_policy_depends_on_freeze_flag() -> None:
    model_freeze = _build_stub_model(freeze_tokenizer=True)
    freeze_state = model_freeze.state_dict()
    assert "tokenizer.weight" not in freeze_state
    assert "classifier.weight" in freeze_state

    model_unfreeze = _build_stub_model(freeze_tokenizer=False)
    unfreeze_state = model_unfreeze.state_dict()
    assert "tokenizer.weight" in unfreeze_state


def test_load_state_dict_compat_with_old_head_only_weights() -> None:
    model = _build_stub_model(freeze_tokenizer=True)
    head_only = {
        "classifier.weight": torch.randn_like(model.classifier.weight),
        "classifier.bias": torch.randn_like(model.classifier.bias),
    }
    incompatible = model.load_state_dict(head_only, strict=False)
    assert isinstance(incompatible.missing_keys, list)


def test_resolve_modality_batch_keys_official() -> None:
    model = _build_stub_model(freeze_tokenizer=True)
    model.batch_key_type = "official"
    batch = {
        "aerial": torch.randn((2, 4, 8, 8)),
        "s1-asc": torch.randn((2, 3, 2, 8, 8)),
        "s1-asc_dates": torch.ones((2, 3), dtype=torch.long),
        "s2": torch.randn((2, 3, 10, 8, 8)),
        "s2_dates": torch.ones((2, 3), dtype=torch.long),
    }
    aerial_keys = model._resolve_modality_batch_keys(batch, "aerial")
    s1_keys = model._resolve_modality_batch_keys(batch, "s1_asc")
    s2_keys = model._resolve_modality_batch_keys(batch, "s2")
    assert aerial_keys == ("aerial", None, None)
    assert s1_keys[0] == "s1-asc"
    assert s1_keys[2] == "s1-asc_dates"
    assert s2_keys[0] == "s2"
    assert s2_keys[2] == "s2_dates"
