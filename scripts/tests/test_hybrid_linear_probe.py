from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from src.stage1.cosmos import cosmos_hybrid
from src.stage2.classification.model.cosmos_tokenizer_classifier import CosmosTokenizerClassifier


class DummyHybridTokenizer:
    def __init__(self, embed_dim: int) -> None:
        self.trans_enc_cfg = SimpleNamespace(embed_dim=embed_dim)

    def requires_grad_(self, _requires_grad: bool) -> "DummyHybridTokenizer":
        return self

    def eval(self) -> "DummyHybridTokenizer":
        return self

    def encode(self, x: torch.Tensor, get_intermediate_features: bool = False, **_kwargs) -> SimpleNamespace:
        if not get_intermediate_features:
            raise ValueError("Expected get_intermediate_features=True for hybrid probe test.")

        batch_size = x.shape[0]
        embed_dim = self.trans_enc_cfg.embed_dim
        features = [torch.randn(batch_size, embed_dim, 4, 4) for _ in range(3)]
        return SimpleNamespace(sem_z=features)


def _create_dummy_hybrid_tokenizer(*_args, **kwargs) -> DummyHybridTokenizer:
    trans_cfg = kwargs.get("trans_enc_cfg")
    if isinstance(trans_cfg, dict):
        embed_dim = int(trans_cfg.get("embed_dim", 64))
    else:
        embed_dim = int(getattr(trans_cfg, "get", lambda key, default: default)("embed_dim", 64))
    return DummyHybridTokenizer(embed_dim)


def test_hybrid_linear_probe_no_cls(monkeypatch) -> None:
    monkeypatch.setattr(
        cosmos_hybrid.CosmosHybridTokenizer,
        "create_model",
        staticmethod(_create_dummy_hybrid_tokenizer),
    )

    tokenizer_cfg = OmegaConf.create(
        {
            "trans_enc_cfg": {
                "embed_dim": 64,
            }
        }
    )

    model = CosmosTokenizerClassifier(
        tokenizer_cfg=tokenizer_cfg,
        num_classes=7,
        classifier={"classifier_type": "linear_probe"},
        tokenizer_pretrained_path=None,
        freeze_tokenizer=True,
        use_intermediate_features=True,
        n_last_blocks=2,
        use_avgpool=True,
    )

    x = torch.randn(2, 3, 32, 32)
    out = model(x)

    assert out["logits"].shape == (2, 7)
