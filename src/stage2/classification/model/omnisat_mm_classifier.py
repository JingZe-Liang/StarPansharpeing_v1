from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
from typing import Any

import torch
from torch import Tensor, nn


def _append_omnisat_src_to_path() -> None:
    omni_src = Path(__file__).resolve().parents[2] / "SSL_third_party" / "OmniSat" / "src"
    omni_src_str = str(omni_src)
    if omni_src_str not in sys.path:
        sys.path.append(omni_src_str)


_append_omnisat_src_to_path()
from models.networks.encoder.Omni import OmniModule  # type: ignore
from models.networks.encoder.utils.ltae import LTAE2d  # type: ignore
from models.networks.encoder.utils.patch_embeddings import PatchEmbed  # type: ignore


def _build_head(in_dim: int, num_classes: int, hidden_dims: Sequence[int]) -> nn.Sequential:
    dims = [in_dim, *[int(x) for x in hidden_dims], num_classes]
    layers: list[nn.Module] = [nn.LayerNorm(dims[0])]
    for idx in range(len(dims) - 1):
        layers.append(nn.Linear(dims[idx], dims[idx + 1]))
        if idx < len(dims) - 2:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


class OmniSatMMClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        modalities: Sequence[str] = ("aerial", "s2", "s1-asc"),
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        class_token: bool = True,
        pre_norm: bool = False,
        drop_rate: float = 0.2,
        pos_drop_rate: float = 0.2,
        patch_drop_rate: float = 0.0,
        drop_path_rate: float = 0.2,
        attn_drop_rate: float = 0.2,
        num_patches: int = 36,
        head_hidden_dims: Sequence[int] = (),
    ) -> None:
        super().__init__()
        self.modalities = tuple(str(x) for x in modalities)
        self.class_token = bool(class_token)

        projectors: dict[str, nn.Module] = {
            "aerial": PatchEmbed(
                patch_size=50,
                in_chans=4,
                embed_dim=int(embed_dim),
                bias=bool(pre_norm),
                res=True,
                gp_norm=4,
            ),
            "s2": LTAE2d(
                in_channels=10,
                n_head=16,
                d_k=8,
                mlp=[256, 512, int(embed_dim)],
                mlp_in=[32, 128, int(embed_dim)],
                dropout=0.2,
                T=367,
                in_norm=True,
                return_att=True,
                positional_encoding=True,
            ),
            "s1-asc": LTAE2d(
                in_channels=2,
                n_head=16,
                d_k=8,
                mlp=[256, 512, int(embed_dim)],
                mlp_in=[32, 128, int(embed_dim)],
                dropout=0.2,
                T=367,
                in_norm=False,
                return_att=True,
                positional_encoding=True,
            ),
            "s1-des": LTAE2d(
                in_channels=2,
                n_head=16,
                d_k=8,
                mlp=[256, 512, int(embed_dim)],
                mlp_in=[32, 128, int(embed_dim)],
                dropout=0.2,
                T=367,
                in_norm=False,
                return_att=True,
                positional_encoding=True,
            ),
        }

        self.encoder = OmniModule(
            projectors=projectors,
            modalities=list(self.modalities),
            num_patches=int(num_patches),
            embed_dim=int(embed_dim),
            depth=int(depth),
            num_heads=int(num_heads),
            mlp_ratio=float(mlp_ratio),
            class_token=self.class_token,
            pre_norm=bool(pre_norm),
            drop_rate=float(drop_rate),
            pos_drop_rate=float(pos_drop_rate),
            patch_drop_rate=float(patch_drop_rate),
            drop_path_rate=float(drop_path_rate),
            attn_drop_rate=float(attn_drop_rate),
        )
        self.head = _build_head(in_dim=int(embed_dim), num_classes=int(num_classes), hidden_dims=head_hidden_dims)

    @staticmethod
    def _get_dates_or_default(batch: Mapping[str, Any], modality: str, x: Tensor) -> Tensor:
        key = f"{modality}_dates"
        value = batch.get(key, None)
        if isinstance(value, Tensor):
            if value.ndim == 1:
                return value.unsqueeze(0).expand(x.shape[0], x.shape[1]).to(device=x.device, dtype=torch.long)
            if value.ndim == 2:
                return value.to(device=x.device, dtype=torch.long)
        t = int(x.shape[1])
        return torch.zeros((x.shape[0], t), device=x.device, dtype=torch.long) + 120

    def _build_encoder_input(self, batch: Mapping[str, Any]) -> dict[str, Tensor]:
        encoder_input: dict[str, Tensor] = {}
        for modality in self.modalities:
            x = batch.get(modality, None)
            if not isinstance(x, Tensor):
                raise ValueError(f"Missing tensor input for modality='{modality}'")
            encoder_input[modality] = x
            if modality != "aerial" and not modality.endswith("-mono"):
                encoder_input[f"{modality}_dates"] = self._get_dates_or_default(batch, modality, x)
        return encoder_input

    def forward(self, batch: Mapping[str, Any]) -> dict[str, Tensor]:
        encoder_input = self._build_encoder_input(batch)
        tokens = self.encoder(encoder_input)
        if self.class_token:
            features = tokens[:, 0]
        else:
            features = tokens.mean(dim=1)
        logits = self.head(features)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
        return {"logits": logits, "features": features}
