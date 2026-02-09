from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Literal

import torch
from omegaconf import DictConfig
from torch import Tensor, nn

from src.stage2.classification.model.cosmos_tokenizer_classifier import CosmosTokenizerClassifier
from src.stage2.segmentation.models.head import get_classifier


class CosmosTokenizerClassifierTS(CosmosTokenizerClassifier):
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
        modality_order: Sequence[str] = ("aerial", "s1_asc", "s1_des", "s2"),
        modality_fusion: Literal["mean", "concat"] = "mean",
        expected_in_chans: int | None = None,
        enable_input_proj: bool = True,
        max_frames_per_forward: int = 32,
    ) -> None:
        super().__init__(
            tokenizer_cfg=tokenizer_cfg,
            num_classes=num_classes,
            classifier=classifier,
            tokenizer_pretrained_path=tokenizer_pretrained_path,
            freeze_tokenizer=freeze_tokenizer,
            use_quantizer=use_quantizer,
            use_z=use_z,
            latent_proj_dim=latent_proj_dim,
            use_intermediate_features=use_intermediate_features,
            n_last_blocks=n_last_blocks,
            use_avgpool=use_avgpool,
        )

        self.modality_order = tuple(modality_order)
        self.modality_fusion = modality_fusion
        self.expected_in_chans = expected_in_chans
        self.enable_input_proj = enable_input_proj
        self.max_frames_per_forward = int(max_frames_per_forward)
        if not self.modality_order:
            raise ValueError("modality_order must not be empty.")
        if self.modality_fusion not in {"mean", "concat"}:
            raise ValueError(f"Unsupported modality_fusion={modality_fusion}")
        if self.max_frames_per_forward <= 0:
            raise ValueError("max_frames_per_forward must be > 0.")

        self.frame_feature_dim = self._infer_frame_feature_dim()
        fused_dim = self._get_fused_dim()
        classifier_cfg = self._normalize_classifier_cfg(classifier)
        classifier_type = classifier_cfg.pop("classifier_type", "linear_probe")
        classifier_cfg.pop("in_features", None)
        self.classifier = get_classifier(
            classifier_type,
            in_features=fused_dim,
            num_classes=num_classes,
            **classifier_cfg,
        )

    def forward(self, batch: Mapping[str, Any]) -> dict[str, Tensor]:
        batch_size = self._infer_batch_size(batch)
        device = self._infer_device(batch)
        modality_features: list[Tensor] = []
        modality_valid: list[Tensor] = []

        for modality in self.modality_order:
            image_key = f"image_{modality}"
            valid_key = f"{image_key}_valid"
            x = batch.get(image_key, None)
            valid = batch.get(valid_key, None)
            if isinstance(x, Tensor):
                feat = self._encode_modality(x, valid, modality)
                cur_valid = torch.ones(batch_size, device=feat.device, dtype=torch.bool)
            else:
                feat = torch.zeros((batch_size, self.frame_feature_dim), device=device, dtype=torch.float32)
                cur_valid = torch.zeros(batch_size, device=device, dtype=torch.bool)
            modality_features.append(feat)
            modality_valid.append(cur_valid)

        fused = self._fuse_features(modality_features, modality_valid)
        logits = self.classifier(fused)
        return {"logits": logits, "features": fused}

    def _infer_frame_feature_dim(self) -> int:
        if isinstance(self.latent_proj, nn.Conv2d):
            return int(self.latent_proj.out_channels)
        if isinstance(self.latent_proj, nn.Linear):
            return int(self.latent_proj.out_features)
        if isinstance(self.latent_proj, nn.Identity):
            return int(self._get_feature_channels())
        raise ValueError(f"Unsupported latent_proj type: {type(self.latent_proj)}")

    def _get_fused_dim(self) -> int:
        if self.modality_fusion == "concat":
            return self.frame_feature_dim * len(self.modality_order)
        return self.frame_feature_dim

    @staticmethod
    def _infer_batch_size(batch: Mapping[str, Any]) -> int:
        for value in batch.values():
            if isinstance(value, Tensor) and value.ndim >= 1:
                return int(value.shape[0])
        raise ValueError("Cannot infer batch size from input batch.")

    @staticmethod
    def _infer_device(batch: Mapping[str, Any]) -> torch.device:
        for value in batch.values():
            if isinstance(value, Tensor):
                return value.device
        return torch.device("cpu")

    def _encode_frame_feature(self, x: Tensor) -> Tensor:
        latent = self._encode_latent_allow_input_grad(x)
        latent = self.latent_proj(latent)
        if latent.ndim == 4:
            return latent.mean(dim=(-2, -1))
        if latent.ndim == 2:
            return latent
        raise ValueError(f"Unsupported latent shape: {latent.shape}")

    @torch.no_grad()
    def _encode_latent_allow_input_grad(self, x: Tensor) -> Tensor:
        if self.use_intermediate_features:
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
            return torch.cat(pooled_features, dim=-1).float()

        if self.use_z:
            return self.tokenizer.encoder.encoder(x)

        enc_out = self.tokenizer.encode(x, use_quantizer=self.use_quantizer)
        return enc_out.latent

    @staticmethod
    def _prepare_temporal_tensor(x: Tensor) -> Tensor:
        if x.ndim == 4:
            return x.unsqueeze(1)
        if x.ndim == 5:
            return x
        raise ValueError(f"Expected image tensor shape [B,C,H,W] or [B,T,C,H,W], got {tuple(x.shape)}")

    @staticmethod
    def _prepare_valid_mask(valid: Any, b: int, t: int, device: torch.device) -> Tensor:
        if isinstance(valid, Tensor):
            if valid.ndim == 1:
                return valid.to(device=device, dtype=torch.bool).unsqueeze(1).expand(b, t)
            if valid.ndim == 2:
                return valid.to(device=device, dtype=torch.bool)
            raise ValueError(f"Expected valid mask [B] or [B,T], got {tuple(valid.shape)}")
        return torch.ones((b, t), device=device, dtype=torch.bool)

    def _encode_modality(self, x: Tensor, valid: Any, modality: str) -> Tensor:
        seq = self._prepare_temporal_tensor(x)
        b, t, c, h, w = seq.shape
        frames = seq.reshape(b * t, c, h, w)
        feats: list[Tensor] = []
        total_frames = int(frames.shape[0])
        for start in range(0, total_frames, self.max_frames_per_forward):
            end = min(start + self.max_frames_per_forward, total_frames)
            feat_chunk = self._encode_frame_feature(frames[start:end])
            feats.append(feat_chunk)
        feat = torch.cat(feats, dim=0)
        feat = torch.nan_to_num(feat, nan=0.0, posinf=1e4, neginf=-1e4)
        feat = feat.reshape(b, t, -1)

        valid_mask = self._prepare_valid_mask(valid, b, t, device=feat.device)
        weight = valid_mask.to(dtype=feat.dtype).unsqueeze(-1)
        denom = weight.sum(dim=1).clamp_min(1.0)
        pooled = (feat * weight).sum(dim=1) / denom
        return pooled

    def _fuse_features(self, features: list[Tensor], valid: list[Tensor]) -> Tensor:
        if self.modality_fusion == "concat":
            return torch.cat(features, dim=-1)

        stacked = torch.stack(features, dim=1)
        valid_mask = torch.stack(valid, dim=1).to(dtype=stacked.dtype).unsqueeze(-1)
        denom = valid_mask.sum(dim=1).clamp_min(1.0)
        return (stacked * valid_mask).sum(dim=1) / denom

    @staticmethod
    def _is_head_state_key(key: str) -> bool:
        return (
            key.startswith("classifier.")
            or key.startswith("latent_proj.")
            or ".classifier." in key
            or ".latent_proj." in key
        )

    def _iter_head_named_parameters(self, recurse: bool = True) -> Iterator[tuple[str, nn.Parameter]]:
        for name, param in self.latent_proj.named_parameters(prefix="latent_proj", recurse=recurse):
            yield name, param
        for name, param in self.classifier.named_parameters(prefix="classifier", recurse=recurse):
            yield name, param

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        for _, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ) -> Iterator[tuple[str, nn.Parameter]]:
        del remove_duplicate
        if prefix:
            prefix = f"{prefix}."
        for name, param in self._iter_head_named_parameters(recurse=recurse):
            yield f"{prefix}{name}", param

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Tensor]:
        full_state = nn.Module.state_dict(self, *args, **kwargs)
        return {key: value for key, value in full_state.items() if self._is_head_state_key(key)}

    def load_state_dict(
        self, state_dict: Mapping[str, Tensor], strict: bool = True
    ) -> torch.nn.modules.module._IncompatibleKeys:
        filtered = {k: v for k, v in state_dict.items() if self._is_head_state_key(k)}
        if not filtered:
            latent_proj_state = self.latent_proj.state_dict()
            classifier_state = self.classifier.state_dict()
            if set(state_dict.keys()) <= (set(latent_proj_state.keys()) | set(classifier_state.keys())):
                for key, value in state_dict.items():
                    if key in latent_proj_state:
                        filtered[f"latent_proj.{key}"] = value
                    elif key in classifier_state:
                        filtered[f"classifier.{key}"] = value
        return nn.Module.load_state_dict(self, filtered, strict=False)
