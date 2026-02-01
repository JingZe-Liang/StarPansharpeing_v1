"""SpectralGPT backbone for remote sensing scene classification."""

from __future__ import annotations

from pathlib import Path
import importlib
import sys
from typing import Literal, cast

import torch
from torch import Tensor, nn

from src.stage2.segmentation.models.head import get_classifier


class SpectralGPTClassifier(nn.Module):
    """SpectralGPT backbone with a trainable classification head.

    Args:
        model_type: "tensor" (default), "group_c", or "default".
        model_name: SpectralGPT model function name (e.g., "vit_base_patch8_128").
        checkpoint_path: Path to SpectralGPT pretrained checkpoint (optional).
        pretrained: Whether to load pretrained weights from checkpoint_path.
        freeze_backbone: If True, only the classification head is trainable.
        num_classes: Number of classes for scene classification.
        classifier_type: "linear" or "linear_probe".
        drop_path_rate: Drop path rate for the backbone.
        backbone_kwargs: Extra kwargs forwarded to the SpectralGPT model factory.
        expected_in_chans: Expected input channels for SpectralGPT (default: 12).
        enable_input_proj: If True, create a 1x1 conv when input channels != expected_in_chans.
        img_is_neg_1_1: If True, map input from [-1, 1] to [0, 1].
        input_scale: Optional scale factor applied after range mapping, before normalization.
        input_mean: Per-channel mean for input normalization (applied after range mapping).
        input_std: Per-channel std for input normalization (applied after range mapping).
    """

    SUPPORTED_MODEL_TYPES = ("tensor", "group_c", "default")

    def __init__(
        self,
        model_type: Literal["tensor", "group_c", "default"] = "tensor",
        model_name: str = "vit_base_patch8_128",
        checkpoint_path: str | None = None,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        num_classes: int = 21,
        classifier_type: Literal["linear", "linear_probe"] = "linear",
        drop_path_rate: float | None = None,
        backbone_kwargs: dict[str, object] | None = None,
        expected_in_chans: int = 12,  # pretrained on 12 bands, if not 12 bands, add one new conv to 12 chans.
        enable_input_proj: bool = True,
        img_is_neg_1_1: bool = False,
        input_scale: float | None = None,
        input_mean: list[float] | None = None,
        input_std: list[float] | None = None,
    ) -> None:
        super().__init__()

        if model_type not in self.SUPPORTED_MODEL_TYPES:
            raise ValueError(f"Unsupported model_type: {model_type}. Choose from {self.SUPPORTED_MODEL_TYPES}")

        self.model_type = model_type
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        self.img_is_neg_1_1 = img_is_neg_1_1
        self.expected_in_chans = expected_in_chans
        self.enable_input_proj = enable_input_proj
        self.input_scale = self._validate_input_scale(input_scale)
        self._set_input_norm(input_mean, input_std)

        self.backbone = self._build_backbone(
            model_type=model_type,
            model_name=model_name,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
            backbone_kwargs=backbone_kwargs,
        )
        self._log_position_embedding_mode()

        if classifier_type != "linear":
            embed_dim = self._infer_embed_dim(self.backbone)
            self.backbone.head = self._build_classifier(  # type: ignore[assignment]
                classifier_type,
                embed_dim,
                num_classes,
            )

        self.input_proj = self._build_input_proj()

        if pretrained and checkpoint_path not in (None, ""):
            self._load_pretrained_backbone(checkpoint_path)

        self._set_backbone_trainable()

    def train(self, mode: bool = True) -> "SpectralGPTClassifier":
        super().train(mode)
        if self.freeze_backbone:
            for name, module in self.backbone.named_modules():
                if name == "head":
                    module.train(mode)
                elif name != "":
                    module.eval()
        return self

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        x = self._preprocess_input(x)
        x = self._maybe_project_input(x)
        logits = self.backbone(x)
        return {"logits": logits}

    def state_dict(self, *args, **kwargs) -> dict[str, Tensor]:
        full_state = super().state_dict(*args, **kwargs)
        if self.freeze_backbone:
            return {
                k: v for k, v in full_state.items() if (not k.startswith("backbone.") or k.startswith("backbone.head."))
            }
        return full_state

    def load_state_dict(
        self,
        state_dict: dict[str, Tensor],
        strict: bool = True,
    ) -> torch.nn.modules.module._IncompatibleKeys:
        if not self.freeze_backbone:
            return super().load_state_dict(state_dict, strict=strict)

        filtered_state = {
            k: v for k, v in state_dict.items() if (not k.startswith("backbone.") or k.startswith("backbone.head."))
        }
        if not filtered_state:
            head_state = self.backbone.head.state_dict()
            if set(state_dict.keys()) <= set(head_state.keys()):
                filtered_state = {f"backbone.head.{k}": v for k, v in state_dict.items()}
        return super().load_state_dict(filtered_state, strict=False)

    def parameters(self, recurse: bool = True):
        for _, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True):
        for name, param in super().named_parameters(prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate):
            if self.freeze_backbone:
                if param.requires_grad:
                    yield name, param
            else:
                yield name, param

    def _set_backbone_trainable(self) -> None:
        for name, param in self.backbone.named_parameters():
            param.requires_grad_(not self.freeze_backbone or name.startswith("head."))
        if self.freeze_backbone:
            self.backbone.eval()

    def _preprocess_input(self, x: Tensor) -> Tensor:
        if self.img_is_neg_1_1:
            x = (x + 1) / 2
        x = self._scale_input(x)
        return self._normalize_input(x)

    @staticmethod
    def _validate_input_scale(input_scale: float | None) -> float | None:
        if input_scale is None:
            return None
        if input_scale <= 0:
            raise ValueError("input_scale must be a positive number.")
        return float(input_scale)

    def _set_input_norm(
        self,
        input_mean: list[float] | None,
        input_std: list[float] | None,
    ) -> None:
        if (input_mean is None) != (input_std is None):
            raise ValueError("input_mean and input_std must both be set or both be None.")
        if input_mean is None:
            return
        mean = torch.tensor(input_mean).view(1, -1, 1, 1)
        std = torch.tensor(input_std).view(1, -1, 1, 1)
        self.register_buffer("_input_mean", mean, persistent=False)
        self.register_buffer("_input_std", std, persistent=False)

    def _scale_input(self, x: Tensor) -> Tensor:
        if self.input_scale is None:
            return x
        return x * self.input_scale

    def _normalize_input(self, x: Tensor) -> Tensor:
        if not hasattr(self, "_input_mean") or not hasattr(self, "_input_std"):
            return x
        mean = cast(Tensor, getattr(self, "_input_mean")).to(device=x.device, dtype=x.dtype)
        std = cast(Tensor, getattr(self, "_input_std")).to(device=x.device, dtype=x.dtype)
        if mean.numel() not in (1, x.shape[1]):
            raise ValueError(f"input_mean channels {mean.numel()} do not match input channels {x.shape[1]}.")
        if std.numel() not in (1, x.shape[1]):
            raise ValueError(f"input_std channels {std.numel()} do not match input channels {x.shape[1]}.")
        return (x - mean) / std

    def _maybe_project_input(self, x: Tensor) -> Tensor:
        if x.shape[1] == self.expected_in_chans:
            return x
        if not self.enable_input_proj:
            raise ValueError(
                f"Input channels {x.shape[1]} != expected {self.expected_in_chans}, and enable_input_proj is False."
            )
        return self.input_proj(x)

    def _build_input_proj(self) -> nn.Module:
        if not self.enable_input_proj:
            return nn.Identity()
        return nn.LazyConv2d(self.expected_in_chans, kernel_size=1, bias=False)

    @staticmethod
    def _build_classifier(
        classifier_type: Literal["linear", "linear_probe"],
        embed_dim: int,
        num_classes: int,
    ) -> nn.Module:
        if classifier_type == "linear":
            return nn.Linear(embed_dim, num_classes)
        if classifier_type == "linear_probe":
            return get_classifier("linear_probe", in_features=embed_dim, num_classes=num_classes)
        raise ValueError(f"Unsupported classifier_type: {classifier_type}")

    @staticmethod
    def _infer_embed_dim(backbone: nn.Module) -> int:
        head = getattr(backbone, "head", None)
        if isinstance(head, nn.Linear):
            return int(head.in_features)
        if hasattr(backbone, "embed_dim"):
            return int(getattr(backbone, "embed_dim"))
        raise ValueError("Cannot infer embed_dim from SpectralGPT backbone.")

    def _build_backbone(
        self,
        model_type: str,
        model_name: str,
        num_classes: int,
        drop_path_rate: float | None,
        backbone_kwargs: dict[str, object] | None,
    ) -> nn.Module:
        repo_dir = self._spectralgpt_repo_dir()
        if str(repo_dir) not in sys.path:
            sys.path.insert(0, str(repo_dir))

        module_name = {
            "tensor": "models_vit_tensor",
            "group_c": "models_vit_group_channels",
            "default": "models_vit",
        }[model_type]
        models = importlib.import_module(module_name)

        if not hasattr(models, model_name):
            available = [name for name in dir(models) if name.startswith("vit_")]
            raise ValueError(f"Unsupported model_name: {model_name}. Available: {available}")

        model_fn = getattr(models, model_name)
        kwargs = {"num_classes": num_classes}
        if drop_path_rate is not None:
            kwargs["drop_path_rate"] = drop_path_rate
        if backbone_kwargs:
            if "num_classes" in backbone_kwargs:
                raise ValueError("backbone_kwargs should not override num_classes.")
            if drop_path_rate is not None and "drop_path_rate" in backbone_kwargs:
                raise ValueError("backbone_kwargs should not override drop_path_rate.")
            # Handle kwargs that may conflict with model_fn's hardcoded defaults (e.g., img_size)
            # These keys would cause "multiple values for keyword argument" error
            conflicting_keys = {
                "img_size",
                "in_chans",
                "patch_size",
                "embed_dim",
                "depth",
                "num_heads",
                "mlp_ratio",
                "num_frames",
                "t_patch_size",
            }
            override_kwargs = {k: v for k, v in backbone_kwargs.items() if k in conflicting_keys}
            filtered_backbone_kwargs = {k: v for k, v in backbone_kwargs.items() if k not in conflicting_keys}
            kwargs.update(filtered_backbone_kwargs)
            if override_kwargs:
                # Build model by directly calling VisionTransformer with merged args
                vit_cls = models.VisionTransformer
                vit_kwargs = self._get_model_defaults(model_name)
                vit_kwargs.update(override_kwargs)
                vit_kwargs.update(kwargs)
                from functools import partial

                vit_kwargs["norm_layer"] = partial(nn.LayerNorm, eps=1e-6)
                return vit_cls(**vit_kwargs)
        backbone = model_fn(**kwargs)
        return backbone

    def _get_model_defaults(self, model_name: str) -> dict[str, object]:
        """Get default kwargs for a model function."""
        defaults = {
            "img_size": 96,
            "patch_size": 8,
            "in_chans": 1,
            "embed_dim": 768,
            "depth": 12,
            "num_heads": 12,
            "mlp_ratio": 4.0,
            "num_frames": 12,
            "t_patch_size": 3,
        }
        match model_name:
            case "vit_base_patch16":
                defaults.update({"patch_size": 16})
            case "vit_base_patch8_128":
                defaults.update({"img_size": 128})
            case "vit_base_patch8_channel10":
                defaults.update({"img_size": 128, "num_frames": 10, "t_patch_size": 2})
            case "vit_base_patch16_128":
                defaults.update({"patch_size": 16, "img_size": 128})
            case "vit_large_patch8_128":
                defaults.update({"embed_dim": 1024, "depth": 24, "num_heads": 16, "img_size": 128})
            case "vit_huge_patch8_128":
                defaults.update({"embed_dim": 1280, "depth": 32, "num_heads": 16, "img_size": 128, "t_patch_size": 12})
            case "vit_base_patch8_120":
                defaults.update({"img_size": 120, "t_patch_size": 12})
            case "vit_huge_patch14":
                defaults.update({"patch_size": 16, "embed_dim": 1280, "depth": 32, "num_heads": 16})
        return defaults

    def _load_pretrained_backbone(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(Path(checkpoint_path).expanduser(), map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)
        if not isinstance(state_dict, dict):
            raise ValueError("Checkpoint does not contain a valid state dict.")

        self._report_position_embedding_loading(state_dict)
        # Interpolate position embeddings BEFORE filtering to handle shape mismatches
        self._interpolate_position_embedding(state_dict)
        self._interpolate_sep_position_embedding(state_dict)
        state_dict = self._filter_state_dict(state_dict)

        msg = self.backbone.load_state_dict(state_dict, strict=False)
        print(
            f"SpectralGPT checkpoint loaded. Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}"
        )
        if msg.missing_keys:
            print(f"  Missing keys: {msg.missing_keys}")
        if msg.unexpected_keys:
            print(f"  Unexpected keys: {msg.unexpected_keys}")

    def _filter_state_dict(self, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        model_state = self.backbone.state_dict()
        filtered: dict[str, Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("head."):
                continue
            if key in model_state and model_state[key].shape != value.shape:
                continue
            filtered[key] = value
        return filtered

    def _interpolate_position_embedding(self, state_dict: dict[str, Tensor]) -> None:
        if "pos_embed" not in state_dict or not hasattr(self.backbone, "pos_embed"):
            return
        repo_dir = self._spectralgpt_repo_dir()
        if str(repo_dir) not in sys.path:
            sys.path.insert(0, str(repo_dir))
        pos_embed = importlib.import_module("util.pos_embed")
        pos_embed.interpolate_pos_embed(self.backbone, state_dict)

    def _interpolate_sep_position_embedding(self, state_dict: dict[str, Tensor]) -> None:
        """Interpolate separated spatial and temporal position embeddings."""
        # Handle spatial position embedding
        if "pos_embed_spatial" in state_dict and hasattr(self.backbone, "pos_embed_spatial"):
            ckpt_spatial = state_dict["pos_embed_spatial"]
            model_spatial = self.backbone.pos_embed_spatial
            if ckpt_spatial.shape != model_spatial.shape:
                print(f"Interpolating pos_embed_spatial from {ckpt_spatial.shape} to {model_spatial.shape}")
                state_dict["pos_embed_spatial"] = self._interpolate_1d_or_2d_pos_embed(
                    ckpt_spatial,
                    model_spatial.shape,  # type: ignore[arg-type]
                )
        # Handle temporal position embedding
        if "pos_embed_temporal" in state_dict and hasattr(self.backbone, "pos_embed_temporal"):
            ckpt_temporal = state_dict["pos_embed_temporal"]
            model_temporal = self.backbone.pos_embed_temporal
            if ckpt_temporal.shape != model_temporal.shape:
                print(f"Interpolating pos_embed_temporal from {ckpt_temporal.shape} to {model_temporal.shape}")
                state_dict["pos_embed_temporal"] = self._interpolate_1d_or_2d_pos_embed(
                    ckpt_temporal,
                    model_temporal.shape,  # type: ignore[arg-type]
                )

    @staticmethod
    def _interpolate_1d_or_2d_pos_embed(checkpoint_embed: Tensor, target_shape: tuple) -> Tensor:
        """Interpolate position embedding to target shape."""
        ckpt_shape = checkpoint_embed.shape
        if len(ckpt_shape) != len(target_shape):
            raise ValueError(f"Shape mismatch: {ckpt_shape} vs {target_shape}")
        # Handle 2D spatial: [1, H*W, C] -> interpolate as 2D grid
        if len(target_shape) == 3 and target_shape[0] == 1:
            embed_dim = target_shape[-1]
            ckpt_len = ckpt_shape[1]
            target_len = target_shape[1]
            # Assume square spatial grid
            ckpt_size = int(ckpt_len**0.5)
            target_size = int(target_len**0.5)
            if ckpt_size * ckpt_size == ckpt_len and target_size * target_size == target_len:
                # Reshape to [1, C, H, W] for interpolation
                embed = checkpoint_embed.reshape(1, ckpt_size, ckpt_size, embed_dim).permute(0, 3, 1, 2)
                embed = torch.nn.functional.interpolate(
                    embed, size=(target_size, target_size), mode="bicubic", align_corners=False
                )
                embed = embed.permute(0, 2, 3, 1).reshape(1, target_size * target_size, embed_dim)
                return embed
        # Handle 1D temporal: [1, T, C] -> interpolate along temporal dim
        if len(target_shape) == 3 and target_shape[0] == 1:
            embed = checkpoint_embed.permute(0, 2, 1)  # [1, C, T]
            embed = torch.nn.functional.interpolate(embed, size=target_shape[1], mode="linear", align_corners=False)
            return embed.permute(0, 2, 1)  # [1, T, C]
        return checkpoint_embed

    @staticmethod
    def _spectralgpt_repo_dir() -> Path:
        return Path(__file__).resolve().parents[2] / "SSL_third_party" / "SpectralGPT"

    def _position_embedding_mode(self) -> str:
        if hasattr(self.backbone, "pos_embed_spatial") or hasattr(self.backbone, "pos_embed_temporal"):
            return "sep_pos_embed"
        if hasattr(self.backbone, "pos_embed"):
            return "pos_embed"
        return "none"

    def _log_position_embedding_mode(self) -> None:
        mode = self._position_embedding_mode()
        print(f"SpectralGPT position embedding mode: {mode}")

    def _report_position_embedding_loading(self, state_dict: dict[str, Tensor]) -> None:
        mode = self._position_embedding_mode()
        has_pos_embed = "pos_embed" in state_dict
        has_sep = "pos_embed_spatial" in state_dict or "pos_embed_temporal" in state_dict
        if mode == "sep_pos_embed" and has_pos_embed and not has_sep:
            print(
                "SpectralGPT checkpoint uses pos_embed; model uses sep_pos_embed. "
                "Position embeddings will be reinitialized."
            )
        elif mode == "pos_embed" and has_sep and not has_pos_embed:
            print(
                "SpectralGPT checkpoint uses sep_pos_embed; model uses pos_embed. "
                "Position embeddings will be reinitialized."
            )


def test_spectralgpt_checkpoint_load() -> None:
    """Test loading SpectralGPT checkpoint and a forward pass."""
    ckpt_path = Path("/Data2/ZihanCao/Checkpoints/SpectralGPT/SpectralGPT.pth")
    if not ckpt_path.exists():
        print(f"⚠️  CKPT not found: {ckpt_path}")
        return

    model = SpectralGPTClassifier(
        model_type="tensor",
        model_name="vit_base_patch8",
        checkpoint_path=str(ckpt_path),
        pretrained=True,
        freeze_backbone=True,
        num_classes=21,
        backbone_kwargs={"sep_pos_embed": True},
    )

    x = torch.randn(2, 12, 96, 96)
    model.eval()
    with torch.no_grad():
        output = model(x)
    logits = output["logits"]
    print(f"forward logits shape: {logits.shape}")


if __name__ == "__main__":
    test_spectralgpt_checkpoint_load()
