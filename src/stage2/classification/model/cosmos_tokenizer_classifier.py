from collections.abc import Mapping
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn

from src.stage1.cosmos.cosmos_tokenizer import ContinuousImageTokenizer
from src.stage2.segmentation.models.head import get_classifier


class CosmosTokenizerClassifier(nn.Module):
    def __init__(
        self,
        tokenizer_cfg: DictConfig,
        num_classes: int,
        classifier: DictConfig | dict[str, Any] | None = None,
        tokenizer_pretrained_path: str | None = None,
        freeze_tokenizer: bool = True,
        use_quantizer: bool | None = None,
        use_z: bool = True,
        latent_proj_dim: int | None = None,
        use_intermediate_features: bool = False,
        n_last_blocks: int = 1,
        use_avgpool: bool = True,
    ) -> None:
        super().__init__()
        tokenizer_cfg = self._normalize_tokenizer_cfg(tokenizer_cfg)

        # 1. Create Tokenizer
        if "trans_enc_cfg" in tokenizer_cfg:
            from src.stage1.cosmos.cosmos_hybrid import CosmosHybridTokenizer

            self.tokenizer = CosmosHybridTokenizer.create_model(**tokenizer_cfg)  # type: ignore[arg-type]
        else:
            self.tokenizer = ContinuousImageTokenizer.create_model(config=tokenizer_cfg)

        if tokenizer_pretrained_path not in (None, ""):
            self.tokenizer.load_pretrained(uni_tokenizer_path=tokenizer_pretrained_path)

        self.freeze_tokenizer = True
        self.use_quantizer = use_quantizer
        self.use_z = use_z and not use_intermediate_features
        self.use_intermediate_features = use_intermediate_features
        self.n_last_blocks = n_last_blocks
        self.use_avgpool = use_avgpool
        self.use_cls_token = self.use_intermediate_features

        self.tokenizer.requires_grad_(False)
        self.tokenizer.eval()

        classifier_cfg = self._normalize_classifier_cfg(classifier)
        classifier_type = classifier_cfg.pop("classifier_type", "default")
        in_features = classifier_cfg.pop("in_features", None)

        feature_channels = self._get_feature_channels()
        target_channels = in_features or latent_proj_dim or feature_channels

        # 2. Latent Projection
        if self.use_cls_token:
            # If using CLS token, input is (B, D), no need for spatial projection
            # We assume target_channels == feature_channels for linear probe
            if target_channels != feature_channels:
                self.latent_proj = nn.Linear(feature_channels, target_channels, bias=False)
            else:
                self.latent_proj = nn.Identity()
        else:
            self.latent_proj = (
                nn.Identity()
                if target_channels == feature_channels
                else nn.Conv2d(feature_channels, target_channels, kernel_size=1, bias=False)
            )

        if classifier_type == "norm_mlp":
            classifier_cfg.setdefault("hidden_size", 1024)

        self.classifier = get_classifier(
            classifier_type,
            in_features=target_channels,
            num_classes=num_classes,
            **classifier_cfg,
        )

    def train(self, mode: bool = True) -> "CosmosTokenizerClassifier":
        super().train(mode)
        self.tokenizer.eval()
        return self

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        latent = self._encode_latent(x)
        latent = self.latent_proj(latent)
        logits = self.classifier(latent)
        return {"logits": logits}

    def _encode_latent(self, x: Tensor) -> Tensor:
        if self.use_intermediate_features:
            return self._encode_intermediate_features(x)
        if self.use_z:
            return self._encode_z(x)
        if self.freeze_tokenizer:
            with torch.no_grad():
                enc_out = self.tokenizer.encode(x, use_quantizer=self.use_quantizer)
        else:
            enc_out = self.tokenizer.encode(x, use_quantizer=self.use_quantizer)
        return enc_out.latent

    def _encode_intermediate_features(self, x: Tensor) -> Tensor:
        """Extract intermediate features from hybrid tokenizer's semantic encoder.

        Hybrid semantic encoder doesn't provide CLS tokens by default, so we use
        global average pooled patch features for each selected block.
        """
        if self.freeze_tokenizer:
            with torch.no_grad():
                enc_out = self.tokenizer.encode(x, get_intermediate_features=True)
        else:
            enc_out = self.tokenizer.encode(x, get_intermediate_features=True)

        sem_z = enc_out.sem_z
        if sem_z is None or not isinstance(sem_z, list):
            raise ValueError("Intermediate features (sem_z) not available from tokenizer.")

        last_n_features = sem_z[-self.n_last_blocks :]
        pooled_features: list[Tensor] = []

        for feat in last_n_features:
            if isinstance(feat, tuple):
                feat = feat[0]

            if feat.ndim == 4:
                pooled = feat.mean(dim=(-2, -1))
            elif feat.ndim == 3:
                pooled = feat.mean(dim=1)
            else:
                raise ValueError(f"Unsupported intermediate feature shape: {feat.shape}")

            pooled_features.append(pooled)

        output = torch.cat(pooled_features, dim=-1)
        return output.float()

    def _encode_z(self, x: Tensor) -> Tensor:
        if self.freeze_tokenizer:
            with torch.no_grad():
                z = self.tokenizer.encoder.encoder(x)
        else:
            z = self.tokenizer.encoder.encoder(x)
        return z

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Tensor]:
        full_state = super().state_dict(*args, **kwargs)
        return self._filter_classifier_state_dict(full_state)

    def load_state_dict(
        self, state_dict: Mapping[str, Tensor], strict: bool = True
    ) -> torch.nn.modules.module._IncompatibleKeys:
        filtered_state = self._filter_classifier_state_dict(dict(state_dict))
        if not filtered_state:
            classifier_state = self.classifier.state_dict()
            if set(state_dict.keys()) <= set(classifier_state.keys()):
                filtered_state = {f"classifier.{k}": v for k, v in state_dict.items()}
        return super().load_state_dict(filtered_state, strict=False)

    @staticmethod
    def _filter_classifier_state_dict(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        return {k: v for k, v in state_dict.items() if k.startswith("classifier.")}

    def _get_feature_channels(self) -> int:
        if self.use_intermediate_features:
            # Infer feature dim from hybrid tokenizer's semantic encoder
            if hasattr(self.tokenizer, "trans_enc_cfg"):
                embed_dim = self.tokenizer.trans_enc_cfg.embed_dim
                return self.n_last_blocks * embed_dim
            raise ValueError("Cannot infer intermediate feature dimension.")

        if self.use_z:
            model_cfg = getattr(self.tokenizer, "model_cfg", None)
            if model_cfg is None:
                # Fallback for hybrid tokenizer which might store it in cnn_cfg.model
                model_cfg = getattr(self.tokenizer, "cnn_cfg", None)
                if model_cfg:
                    model_cfg = model_cfg.model

            if model_cfg is None or getattr(model_cfg, "z_channels", None) is None:
                raise ValueError("Tokenizer missing model_cfg.z_channels for z feature.")
            return model_cfg.z_channels

        latent_channels = getattr(self.tokenizer, "latent_channels", None)
        if latent_channels is None:
            raise ValueError("Tokenizer missing latent_channels attribute.")
        return latent_channels

    @staticmethod
    def _normalize_classifier_cfg(classifier: DictConfig | dict[str, Any] | None) -> dict[str, Any]:
        if classifier is None:
            return {}
        if isinstance(classifier, DictConfig):
            return OmegaConf.to_container(classifier, resolve=True)  # type: ignore[return-value]
        return dict(classifier)

    @staticmethod
    def _normalize_tokenizer_cfg(tokenizer_cfg: DictConfig | dict[str, Any]) -> DictConfig:
        if isinstance(tokenizer_cfg, DictConfig):
            cfg_dict = OmegaConf.to_container(tokenizer_cfg, resolve=True)
        else:
            cfg_dict = dict(tokenizer_cfg)

        if isinstance(cfg_dict, dict) and "config" in cfg_dict and "model" not in cfg_dict:
            cfg_dict = cfg_dict["config"]

        if not isinstance(cfg_dict, dict):
            raise ValueError("tokenizer_cfg must be a dict-like config.")

        cfg = OmegaConf.create(cfg_dict)
        if not isinstance(cfg, DictConfig):
            raise ValueError("tokenizer_cfg must be a DictConfig.")
        return cfg


def test_model():
    """Test CosmosTokenizerClassifier with tokenizer weights from pansharpening config."""
    import yaml
    from pathlib import Path

    # Load config
    config_path = (
        Path(__file__).parent.parent.parent.parent.parent
        / "scripts/configs/pansharpening/pansharp_model/pansharp_wrapper_nafnet_cosmos.yaml"
    )

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Extract tokenizer config and path
    tokenizer_cfg = config["tokenizer"]["model"]
    tokenizer_pretrained_path = config["tokenizer"]["uni_path"]

    print(f"Loading tokenizer from: {tokenizer_pretrained_path}")
    print(
        f"Tokenizer config: latent_channels={tokenizer_cfg['latent_channels']}, spatial_compression={tokenizer_cfg['spatial_compression']}"
    )

    # Create model
    num_classes = 10
    model = CosmosTokenizerClassifier(
        tokenizer_cfg=OmegaConf.create(tokenizer_cfg),
        num_classes=num_classes,
        tokenizer_pretrained_path=tokenizer_pretrained_path,
        freeze_tokenizer=True,
        use_quantizer=False,
        classifier={"classifier_type": "default"},
    )

    # Test with random input
    batch_size = 2
    in_channels = tokenizer_cfg["in_channels"]
    height, width = 64, 64

    x = torch.randn(batch_size, in_channels, height, width)

    print(f"\nInput shape: {x.shape}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(x)

    logits = output["logits"]
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected shape: ({batch_size}, {num_classes})")

    # Verify output
    assert logits.shape == (batch_size, num_classes), f"Expected shape {(batch_size, num_classes)}, got {logits.shape}"

    # Check if logits are valid (not NaN or Inf)
    assert not torch.isnan(logits).any(), "Logits contain NaN values"
    assert not torch.isinf(logits).any(), "Logits contain Inf values"

    # Compute probabilities
    probs = torch.softmax(logits, dim=-1)
    print(f"\nProbabilities shape: {probs.shape}")
    print(f"Probabilities sum: {probs.sum(dim=-1)}")

    # Check probabilities sum to 1
    assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size)), "Probabilities should sum to 1"

    print("\n✅ Test passed! Model outputs are valid.")
    print(f"Sample logits: {logits[0]}")
    print(f"Sample probabilities: {probs[0]}")

    return model, output


if __name__ == "__main__":
    test_model()
