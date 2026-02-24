from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
import copy
from typing import Any, Literal

import torch
import torch.nn.functional as F
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
        fusion_dim: int = 512,
        fusion_layers: int = 2,
        fusion_heads: int = 8,
        fusion_dropout: float = 0.1,
        fusion_pool: Literal["cls", "mean"] = "cls",
        ltae_d_model: int = 256,
        ltae_heads: int = 16,
        ltae_d_k: int = 8,
        ltae_dropout: float = 0.2,
        share_s1_ltae: bool = True,
        batch_key_type: str = "legacy",
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

        self.batch_key_type = batch_key_type
        self.modality_order = tuple(modality_order)
        self.modality_fusion = modality_fusion
        self.expected_in_chans = expected_in_chans
        self.enable_input_proj = enable_input_proj
        self.max_frames_per_forward = int(max_frames_per_forward)
        self.fusion_dim = int(fusion_dim)
        self.fusion_layers = int(fusion_layers)
        self.fusion_heads = int(fusion_heads)
        self.fusion_dropout = float(fusion_dropout)
        self.fusion_pool = fusion_pool
        self.ltae_d_model = int(ltae_d_model)
        self.ltae_heads = int(ltae_heads)
        self.ltae_d_k = int(ltae_d_k)
        self.ltae_dropout = float(ltae_dropout)
        self.share_s1_ltae = bool(share_s1_ltae)
        if not self.modality_order:
            raise ValueError("modality_order must not be empty.")
        if self.modality_fusion not in {"mean", "concat"}:
            raise ValueError(f"Unsupported modality_fusion={modality_fusion}")
        if self.max_frames_per_forward <= 0:
            raise ValueError("max_frames_per_forward must be > 0.")
        if self.fusion_dim <= 0:
            raise ValueError("fusion_dim must be > 0.")
        if self.fusion_layers <= 0:
            raise ValueError("fusion_layers must be > 0.")
        if self.fusion_heads <= 0:
            raise ValueError("fusion_heads must be > 0.")
        if self.fusion_pool not in {"cls", "mean"}:
            raise ValueError(f"Unsupported fusion_pool={fusion_pool}")
        if self.batch_key_type not in {"my", "legacy", "official"}:
            raise ValueError(f"Unsupported batch_key_type={batch_key_type}")

        self.freeze_tokenizer = bool(freeze_tokenizer)
        self.tokenizer.requires_grad_(not self.freeze_tokenizer)
        if self.freeze_tokenizer:
            self.tokenizer.eval()

        self.frame_feature_dim = self._infer_frame_feature_dim()
        self.aerial_proj = nn.Linear(self.frame_feature_dim, self.fusion_dim, bias=False)
        self.ltae_s1 = LTAE2d(
            in_channels=2,
            n_head=self.ltae_heads,
            d_k=self.ltae_d_k,
            mlp=[self.ltae_d_model, self.ltae_d_model],
            mlp_in=[64, self.ltae_d_model],
            dropout=self.ltae_dropout,
            t=367,
            in_norm=False,
            return_att=False,
            positional_encoding=True,
        )
        if self.share_s1_ltae:
            self.ltae_s1_des = self.ltae_s1
        else:
            self.ltae_s1_des = copy.deepcopy(self.ltae_s1)
        self.ltae_s2 = LTAE2d(
            in_channels=10,
            n_head=self.ltae_heads,
            d_k=self.ltae_d_k,
            mlp=[self.ltae_d_model, self.ltae_d_model],
            mlp_in=[128, self.ltae_d_model],
            dropout=self.ltae_dropout,
            t=367,
            in_norm=True,
            return_att=False,
            positional_encoding=True,
        )
        self.s1_proj = nn.Linear(self.ltae_d_model, self.fusion_dim, bias=False)
        self.s2_proj = nn.Linear(self.ltae_d_model, self.fusion_dim, bias=False)
        self.modality_embedding = nn.Parameter(torch.zeros((len(self.modality_order), self.fusion_dim)))
        self.fusion_cls_token = nn.Parameter(torch.zeros((1, 1, self.fusion_dim)))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.fusion_dim,
            nhead=self.fusion_heads,
            dim_feedforward=self.fusion_dim * 4,
            dropout=self.fusion_dropout,
            activation="gelu",
            batch_first=True,
        )
        self.fusion_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.fusion_layers)
        classifier_cfg = self._normalize_classifier_cfg(classifier)
        classifier_type = classifier_cfg.pop("classifier_type", "linear_probe")
        classifier_cfg.pop("in_features", None)
        self.classifier = get_classifier(
            classifier_type,
            in_features=self.fusion_dim,
            num_classes=num_classes,
            **classifier_cfg,
        )
        nn.init.trunc_normal_(self.modality_embedding, std=0.02)
        nn.init.trunc_normal_(self.fusion_cls_token, std=0.02)

    def forward(self, batch: Mapping[str, Any]) -> dict[str, Tensor]:
        batch_size = self._infer_batch_size(batch)
        device = self._infer_device(batch)
        modality_tokens: list[Tensor] = []
        modality_valid: list[Tensor] = []
        for modality in self.modality_order:
            token, cur_valid = self._encode_modality_token(batch, modality, batch_size, device)
            modality_tokens.append(token)
            modality_valid.append(cur_valid)
        fused = self._fuse_tokens(modality_tokens, modality_valid)
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

    def _encode_latent_allow_input_grad(self, x: Tensor) -> Tensor:
        if self.use_intermediate_features:
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
            return torch.cat(pooled_features, dim=-1).float()

        if self.use_z:
            if self.freeze_tokenizer:
                with torch.no_grad():
                    return self.tokenizer.encoder.encoder(x)
            return self.tokenizer.encoder.encoder(x)

        if self.freeze_tokenizer:
            with torch.no_grad():
                enc_out = self.tokenizer.encode(x, use_quantizer=self.use_quantizer)
        else:
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

    @staticmethod
    def _prepare_dates(dates: Any, b: int, t: int, device: torch.device) -> Tensor:
        if isinstance(dates, Tensor):
            if dates.ndim == 1:
                return dates.to(device=device, dtype=torch.long).unsqueeze(0).expand(b, t)
            if dates.ndim == 2:
                return dates.to(device=device, dtype=torch.long)
            raise ValueError(f"Expected dates shape [T] or [B,T], got {tuple(dates.shape)}")
        base = torch.arange(t, device=device, dtype=torch.long)
        return base.unsqueeze(0).expand(b, t)

    def _encode_frames_in_chunks(self, frames: Tensor) -> Tensor:
        feats: list[Tensor] = []
        total_frames = int(frames.shape[0])
        for start in range(0, total_frames, self.max_frames_per_forward):
            end = min(start + self.max_frames_per_forward, total_frames)
            feat_chunk = self._encode_frame_feature(frames[start:end])
            feats.append(feat_chunk)
        return torch.cat(feats, dim=0)

    @staticmethod
    def _pool_temporal_feature(feat: Tensor, valid_mask: Tensor) -> Tensor:
        weight = valid_mask.to(dtype=feat.dtype).unsqueeze(-1)
        denom = weight.sum(dim=1).clamp_min(1.0)
        return (feat * weight).sum(dim=1) / denom

    def _encode_aerial(self, x: Tensor, valid: Any) -> tuple[Tensor, Tensor]:
        seq = self._prepare_temporal_tensor(x)
        b, t, c, h, w = seq.shape
        valid_mask = self._prepare_valid_mask(valid, b, t, device=seq.device)
        flat_valid = valid_mask.reshape(-1)
        frames = seq.reshape(b * t, c, h, w)

        if bool(flat_valid.any()):
            valid_frames = frames[flat_valid]
            valid_feat = self._encode_frames_in_chunks(valid_frames)
            feat = torch.zeros(
                (b * t, int(valid_feat.shape[-1])),
                device=valid_feat.device,
                dtype=valid_feat.dtype,
            )
            feat[flat_valid] = valid_feat
        else:
            feat = torch.zeros((b * t, self.frame_feature_dim), device=seq.device, dtype=torch.float32)

        feat = torch.nan_to_num(feat, nan=0.0, posinf=1e4, neginf=-1e4)
        feat = feat.reshape(b, t, -1)
        pooled = self._pool_temporal_feature(feat, valid_mask)
        token = self.aerial_proj(pooled)
        modality_valid = valid_mask.any(dim=1)
        return token, modality_valid

    def _encode_sentinel(
        self,
        x: Tensor,
        valid: Any,
        dates: Any,
        ltae: LTAE2d,
        proj: nn.Linear,
    ) -> tuple[Tensor, Tensor]:
        seq = self._prepare_temporal_tensor(x)
        b, t, _, _, _ = seq.shape
        valid_mask = self._prepare_valid_mask(valid, b, t, device=seq.device)
        date_tensor = self._prepare_dates(dates, b, t, device=seq.device)
        ltae_out = ltae(seq, batch_positions=date_tensor, pad_mask=~valid_mask)
        pooled = ltae_out.mean(dim=(-2, -1))
        token = proj(pooled)
        modality_valid = valid_mask.any(dim=1)
        return token, modality_valid

    def _encode_modality_token(
        self,
        batch: Mapping[str, Any],
        modality: str,
        batch_size: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        image_key, valid_key, doy_key = self._resolve_modality_batch_keys(batch, modality)
        x = batch.get(image_key, None)
        valid = batch.get(valid_key, None) if valid_key is not None else None
        dates = batch.get(doy_key, None) if doy_key is not None else None

        if not isinstance(x, Tensor):
            token = torch.zeros((batch_size, self.fusion_dim), device=device, dtype=torch.float32)
            modality_valid = torch.zeros(batch_size, device=device, dtype=torch.bool)
            return token, modality_valid
        if modality == "aerial":
            return self._encode_aerial(x, valid)
        if modality == "s2":
            return self._encode_sentinel(x, valid, dates, self.ltae_s2, self.s2_proj)
        if modality == "s1_asc":
            return self._encode_sentinel(x, valid, dates, self.ltae_s1, self.s1_proj)
        if modality == "s1_des":
            return self._encode_sentinel(x, valid, dates, self.ltae_s1_des, self.s1_proj)
        raise ValueError(f"Unsupported modality={modality}")

    @staticmethod
    def _first_existing_key(
        batch: Mapping[str, Any],
        candidates: list[str],
        fallback_first: bool = False,
    ) -> str | None:
        for key in candidates:
            if key in batch:
                return key
        if fallback_first and candidates:
            return candidates[0]
        return None

    def _resolve_modality_batch_keys(
        self, batch: Mapping[str, Any], modality: str
    ) -> tuple[str, str | None, str | None]:
        dash_modality = modality.replace("_", "-")
        underscore_modality = modality.replace("-", "_")
        image_my = f"image_{underscore_modality}"
        valid_my = f"{image_my}_valid"
        doy_my = f"doy_{underscore_modality}"
        date_official = f"{dash_modality}_dates"
        date_underscore = f"{underscore_modality}_dates"

        if self.batch_key_type == "my":
            image_candidates = [image_my]
            valid_candidates = [valid_my, f"{underscore_modality}_valid", f"{dash_modality}_valid"]
            doy_candidates = [doy_my, date_underscore, date_official]
        elif self.batch_key_type == "official":
            image_candidates = [dash_modality, underscore_modality, image_my]
            valid_candidates = [f"{dash_modality}_valid", f"{underscore_modality}_valid", valid_my]
            doy_candidates = [date_official, date_underscore, doy_my]
        else:  # legacy
            image_candidates = [image_my, underscore_modality, dash_modality]
            valid_candidates = [valid_my, f"{underscore_modality}_valid", f"{dash_modality}_valid"]
            doy_candidates = [doy_my, date_underscore, date_official]

        image_key = self._first_existing_key(batch, image_candidates, fallback_first=True)
        if image_key is None:
            raise ValueError(f"Cannot resolve image key for modality={modality}")
        valid_key = self._first_existing_key(batch, valid_candidates)
        doy_key = self._first_existing_key(batch, doy_candidates)
        return image_key, valid_key, doy_key

    def _fuse_tokens(self, tokens: list[Tensor], valid: list[Tensor]) -> Tensor:
        stacked = torch.stack(tokens, dim=1)
        valid_mask = torch.stack(valid, dim=1)
        stacked = stacked + self.modality_embedding.unsqueeze(0).to(dtype=stacked.dtype, device=stacked.device)

        if self.fusion_pool == "cls":
            cls = self.fusion_cls_token.expand(stacked.shape[0], -1, -1).to(dtype=stacked.dtype, device=stacked.device)
            src = torch.cat([cls, stacked], dim=1)
            padding_mask = torch.cat(
                [torch.zeros((valid_mask.shape[0], 1), dtype=torch.bool, device=valid_mask.device), ~valid_mask],
                dim=1,
            )
            fused = self.fusion_encoder(src, src_key_padding_mask=padding_mask)
            return fused[:, 0]

        fused_tokens = self.fusion_encoder(stacked, src_key_padding_mask=~valid_mask)
        weight = valid_mask.to(dtype=fused_tokens.dtype).unsqueeze(-1)
        denom = weight.sum(dim=1).clamp_min(1.0)
        return (fused_tokens * weight).sum(dim=1) / denom

    @staticmethod
    def _is_trainable_state_key(key: str) -> bool:
        return (
            key.startswith("classifier.")
            or key.startswith("latent_proj.")
            or key.startswith("aerial_proj.")
            or key.startswith("ltae_s1.")
            or key.startswith("ltae_s2.")
            or key.startswith("ltae_s1_des.")
            or key.startswith("s1_proj.")
            or key.startswith("s2_proj.")
            or key.startswith("fusion_encoder.")
            or key.startswith("modality_embedding")
            or key.startswith("fusion_cls_token")
        )

    def train(self, mode: bool = True) -> CosmosTokenizerClassifierTS:
        nn.Module.train(self, mode)
        if self.freeze_tokenizer:
            self.tokenizer.eval()
        return self

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        if not self.freeze_tokenizer:
            for param in nn.Module.parameters(self, recurse=recurse):
                yield param
            return
        for name, param in nn.Module.named_parameters(self, recurse=recurse):
            if self._is_trainable_state_key(name):
                yield param

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ) -> Iterator[tuple[str, nn.Parameter]]:
        if not self.freeze_tokenizer:
            for name, param in nn.Module.named_parameters(
                self,
                prefix=prefix,
                recurse=recurse,
                remove_duplicate=remove_duplicate,
            ):
                yield name, param
            return
        for name, param in nn.Module.named_parameters(
            self,
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        ):
            if self._is_trainable_state_key(name):
                yield name, param

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Tensor]:
        full_state = nn.Module.state_dict(self, *args, **kwargs)
        if not self.freeze_tokenizer:
            return full_state
        return {key: value for key, value in full_state.items() if self._is_trainable_state_key(key)}

    def load_state_dict(
        self, state_dict: Mapping[str, Tensor], strict: bool = True
    ) -> torch.nn.modules.module._IncompatibleKeys:
        full_state_keys = set(nn.Module.state_dict(self).keys())
        incoming_keys = set(state_dict.keys())
        if incoming_keys and incoming_keys <= full_state_keys and (not self.freeze_tokenizer):
            return nn.Module.load_state_dict(self, dict(state_dict), strict=False)

        filtered = {k: v for k, v in state_dict.items() if self._is_trainable_state_key(k)}
        if not filtered:
            classifier_state = self.classifier.state_dict()
            latent_proj_state = self.latent_proj.state_dict()
            if incoming_keys <= (set(classifier_state.keys()) | set(latent_proj_state.keys())):
                for key, value in state_dict.items():
                    if key in classifier_state:
                        filtered[f"classifier.{key}"] = value
                    if key in latent_proj_state:
                        filtered[f"latent_proj.{key}"] = value
        return nn.Module.load_state_dict(self, filtered, strict=False)


class PositionalEncoder(nn.Module):
    def __init__(self, d: int, t: int = 1000, repeat: int | None = None, offset: int = 0) -> None:
        super().__init__()
        self.repeat = repeat
        denom = torch.pow(t, 2 * (torch.arange(offset, offset + d, dtype=torch.float32) // 2) / d)
        self.register_buffer("denom", denom, persistent=False)

    def forward(self, batch_positions: Tensor) -> Tensor:
        denom = torch.as_tensor(self.denom, dtype=torch.float32, device=batch_positions.device)
        sinusoid = batch_positions[:, :, None].to(torch.float32) / denom.view(1, 1, -1)
        sinusoid[:, :, 0::2] = torch.sin(sinusoid[:, :, 0::2])
        sinusoid[:, :, 1::2] = torch.cos(sinusoid[:, :, 1::2])
        if self.repeat is not None:
            sinusoid = torch.cat([sinusoid for _ in range(self.repeat)], dim=-1)
        return sinusoid


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature: float, attn_dropout: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, pad_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2)) / self.temperature
        if pad_mask is not None:
            attn = attn.masked_fill(pad_mask.unsqueeze(1), -1e3)
        attn = self.dropout(self.softmax(attn))
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head: int, d_k: int, d_in: int) -> None:
        super().__init__()
        if d_in % n_head != 0:
            raise ValueError(f"d_in ({d_in}) must be divisible by n_head ({n_head}).")
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in
        self.d_v = d_in // n_head
        self.q = nn.Parameter(torch.zeros((n_head, d_k)), requires_grad=True)
        nn.init.normal_(self.q, mean=0.0, std=(2.0 / d_k) ** 0.5)
        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0.0, std=(2.0 / d_k) ** 0.5)

    def forward(self, v: Tensor, pad_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        sz_b, seq_len, _ = v.size()
        q = self.q.unsqueeze(0).unsqueeze(2).expand(sz_b, -1, -1, -1)
        k = self.fc1_k(v).view(sz_b, seq_len, self.n_head, self.d_k).permute(0, 2, 1, 3).contiguous()
        v_heads = v.view(sz_b, seq_len, self.n_head, self.d_v).permute(0, 2, 1, 3).contiguous()

        attn_bias: Tensor | None = None
        if pad_mask is not None:
            attn_bias = torch.zeros((sz_b, 1, 1, seq_len), dtype=q.dtype, device=q.device)
            attn_bias = attn_bias.masked_fill(pad_mask[:, None, None, :], float("-inf"))

        output = F.scaled_dot_product_attention(
            q,
            k,
            v_heads,
            attn_mask=attn_bias,
            dropout_p=0.0,
        )
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / (float(self.d_k) ** 0.5)
        if attn_bias is not None:
            attn_logits = attn_logits + attn_bias
        attn = torch.softmax(attn_logits, dim=-1).squeeze(2).permute(1, 0, 2).contiguous()
        output = output.squeeze(2).permute(1, 0, 2).contiguous()
        return output, attn


class LTAE2d(nn.Module):
    def __init__(
        self,
        in_channels: int = 10,
        n_head: int = 16,
        d_k: int = 8,
        mlp: list[int] | None = None,
        dropout: float = 0.2,
        mlp_in: list[int] | None = None,
        t: int = 367,
        in_norm: bool = True,
        return_att: bool = False,
        positional_encoding: bool = True,
    ) -> None:
        super().__init__()
        mlp = mlp or [256, 256]
        mlp_in = mlp_in or [64, 256]
        self.return_att = return_att
        self.n_head = n_head
        self.in_channels = in_channels

        if len(mlp_in) > 0:
            self.d_model = int(mlp_in[-1])
            mlp_in_layers = [self.in_channels, *mlp_in]
            in_layers: list[nn.Module] = []
            for i in range(len(mlp_in_layers) - 1):
                in_layers.extend(
                    [
                        nn.Linear(mlp_in_layers[i], mlp_in_layers[i + 1]),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    ]
                )
            self.inconv = nn.Sequential(*in_layers)
        else:
            self.d_model = self.in_channels
            self.inconv = None

        mlp_layers = [self.d_model, *mlp]
        self.positional_encoder = (
            PositionalEncoder(self.d_model // n_head, t=t, repeat=n_head) if positional_encoding else None
        )
        self.attention_heads = MultiHeadAttention(n_head=n_head, d_k=d_k, d_in=self.d_model)
        self.in_norm = nn.GroupNorm(num_groups=n_head, num_channels=self.d_model) if in_norm else nn.Identity()
        self.out_norm = nn.GroupNorm(num_groups=n_head, num_channels=mlp_layers[-1])
        out_layers: list[nn.Module] = []
        for i in range(len(mlp_layers) - 1):
            out_layers.extend(
                [
                    nn.Linear(mlp_layers[i], mlp_layers[i + 1]),
                    nn.ReLU(),
                ]
            )
        self.mlp = nn.Sequential(*out_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, batch_positions: Tensor | None = None, pad_mask: Tensor | None = None) -> Tensor:
        sz_b, seq_len, _, h, w = x.shape
        ltae_pad_mask: Tensor | None = None
        if pad_mask is not None:
            ltae_pad_mask = (
                (pad_mask.unsqueeze(-1).repeat((1, 1, h)).unsqueeze(-1).repeat((1, 1, 1, w)))
                .permute(0, 2, 3, 1)
                .contiguous()
                .view(sz_b * h * w, seq_len)
            )

        out = x.permute(0, 3, 4, 1, 2).contiguous().view(sz_b * h * w, seq_len, self.in_channels)
        if self.inconv is not None:
            out = self.inconv(out.view(-1, out.size(-1))).view(out.shape[0], out.shape[1], -1)

        out = self.in_norm(out.permute(0, 2, 1)).permute(0, 2, 1)
        if self.positional_encoder is not None and batch_positions is not None:
            bp = (
                (batch_positions.unsqueeze(-1).repeat((1, 1, h)).unsqueeze(-1).repeat((1, 1, 1, w)))
                .permute(0, 2, 3, 1)
                .contiguous()
                .view(sz_b * h * w, seq_len)
            )
            out = out + self.positional_encoder(bp)

        out, attn = self.attention_heads(out, pad_mask=ltae_pad_mask)
        out = out.permute(1, 0, 2).contiguous().view(sz_b * h * w, -1)
        out = self.dropout(self.mlp(out))
        out = self.out_norm(out)
        out = out.view(sz_b, h, w, -1).permute(0, 3, 1, 2)
        if self.return_att:
            attn_map = attn.view(self.n_head, sz_b, h, w, seq_len).permute(0, 1, 4, 2, 3)
            _ = torch.mean(attn_map, dim=(0, 3, 4), keepdim=True).squeeze()
        return out
