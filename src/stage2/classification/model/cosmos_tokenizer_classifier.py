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
        latent_proj_dim: int | None = None,
    ) -> None:
        super().__init__()
        tokenizer_cfg = self._normalize_tokenizer_cfg(tokenizer_cfg)
        self.tokenizer = ContinuousImageTokenizer.create_model(config=tokenizer_cfg)
        if tokenizer_pretrained_path not in (None, ""):
            self.tokenizer.load_pretrained(uni_tokenizer_path=tokenizer_pretrained_path)

        self.freeze_tokenizer = freeze_tokenizer
        self.use_quantizer = use_quantizer
        if self.freeze_tokenizer:
            self.tokenizer.requires_grad_(False)
            self.tokenizer.eval()

        classifier_cfg = self._normalize_classifier_cfg(classifier)
        classifier_type = classifier_cfg.pop("classifier_type", "default")
        in_features = classifier_cfg.pop("in_features", None)

        latent_channels = getattr(self.tokenizer, "latent_channels", None)
        if latent_channels is None:
            raise ValueError("Tokenizer missing latent_channels attribute.")

        target_channels = in_features or latent_proj_dim or latent_channels
        self.latent_proj = (
            nn.Identity()
            if target_channels == latent_channels
            else nn.Conv2d(latent_channels, target_channels, kernel_size=1, bias=False)
        )

        self.classifier = get_classifier(
            classifier_type,
            in_features=target_channels,
            num_classes=num_classes,
            **classifier_cfg,
        )

    def train(self, mode: bool = True) -> "CosmosTokenizerClassifier":
        super().train(mode)
        if self.freeze_tokenizer:
            self.tokenizer.eval()
        return self

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        latent = self._encode_latent(x)
        latent = self.latent_proj(latent)
        logits = self.classifier(latent)
        return {"logits": logits}

    def _encode_latent(self, x: Tensor) -> Tensor:
        if self.freeze_tokenizer:
            with torch.no_grad():
                enc_out = self.tokenizer.encode(x, use_quantizer=self.use_quantizer)
        else:
            enc_out = self.tokenizer.encode(x, use_quantizer=self.use_quantizer)
        return enc_out.latent

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
