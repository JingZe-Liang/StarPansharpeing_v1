"""DINOv3 backbone + Linear Probe for classification.

Supports:
- ViT-L and ConvNeXt backbones
- Remote sensing pretrained weights (SAT493M for ViT)
- Frozen backbone with trainable linear probe
"""

from typing import Any, Literal, cast

import torch
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import Tensor, nn
from torchvision.transforms import Normalize

from src.stage2.segmentation.models.head import create_linear_input, get_classifier


class DINOv3Classifier(nn.Module):
    """DINOv3 backbone with linear probe for classification.

    Supports both ViT and ConvNeXt backbones with DINOv3 pretrained weights.

    Args:
        backbone_type: Type of backbone - "vitl16", "convnext_base", "convnext_large"
        weights: Pretrained weights - "SAT493M" (remote sensing), "LVD1689M" (general),
                 or path to local weights
        pretrained: Whether to load pretrained weights
        freeze_backbone: Whether to freeze backbone during training
        n_last_blocks: Number of last blocks to use for ViT features
        use_avgpool: Whether to use avgpool of patch tokens (for ViT)
        num_classes: Number of output classes
    """

    SUPPORTED_BACKBONES = {
        # ViT backbones
        "vits16": "dinov3_vits16",
        "vitb16": "dinov3_vitb16",
        "vitl16": "dinov3_vitl16",
        # ConvNeXt backbones
        "convnext_tiny": "dinov3_convnext_tiny",
        "convnext_small": "dinov3_convnext_small",
        "convnext_base": "dinov3_convnext_base",
        "convnext_large": "dinov3_convnext_large",
    }

    def __init__(
        self,
        backbone_type: Literal[
            "vits16",
            "vitb16",
            "vitl16",
            "convnext_tiny",
            "convnext_small",
            "convnext_base",
            "convnext_large",
        ] = "vitl16",
        weights: str = "SAT493M",
        pretrained: bool = True,
        freeze_backbone: bool = True,
        n_last_blocks: int = 1,
        use_avgpool: bool = True,
        num_classes: int = 1000,
        img_is_neg_1_1: bool = True,
    ) -> None:
        super().__init__()

        if backbone_type not in self.SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported backbone: {backbone_type}. Choose from {list(self.SUPPORTED_BACKBONES.keys())}"
            )

        self.backbone_type = backbone_type
        self.weights = weights
        self.freeze_backbone = freeze_backbone
        self.n_last_blocks = n_last_blocks
        self.use_avgpool = use_avgpool
        self.is_vit = backbone_type.startswith("vit")
        self.img_is_neg_1_1 = img_is_neg_1_1

        # ImageNet normalization for DINOv3 backbone
        self.normalize = Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

        # Build backbone
        self.backbone = self._build_backbone(backbone_type, weights, pretrained)

        if freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

        # Build classifier head
        self.classifier = self._build_classifier(num_classes)

    def _build_backbone(
        self,
        backbone_type: str,
        weights: str,
        pretrained: bool,
    ) -> nn.Module:
        """Build DINOv3 backbone from local weights.

        Weights are loaded from:
        - SAT493M (satellite): src/stage1/utilities/losses/dinov3/weights/remote_sensing_image_pretrained_SAT_493M/
        - LVD1689M (web): src/stage1/utilities/losses/dinov3/weights/web_image_pretrained_lvd/
        """
        from pathlib import Path
        import sys

        backbone_fn_name = self.SUPPORTED_BACKBONES[backbone_type]

        # DINOv3 repo directory
        repo_dir = Path(__file__).parents[3] / "stage1/utilities/losses/dinov3"
        assert repo_dir.exists(), f"DINOv3 repo directory {repo_dir} does not exist"

        if not pretrained:
            # Load without pretrained weights
            sys.path.insert(0, str(repo_dir))
            model = torch.hub.load(str(repo_dir), backbone_fn_name, source="local", pretrained=False)
            return cast(nn.Module, model)

        # Determine weight directory based on weights type
        if weights == "SAT493M":
            weight_dir = repo_dir / "weights/remote_sensing_image_pretrained_SAT_493M"
        elif weights == "LVD1689M":
            weight_dir = repo_dir / "weights/web_image_pretrained_lvd"
        elif Path(weights).exists():
            # Direct path to weight file
            weight_path = weights
            sys.path.insert(0, str(repo_dir))
            model = torch.hub.load(str(repo_dir), backbone_fn_name, source="local", weights=weight_path)
            return cast(nn.Module, model)
        else:
            raise ValueError(f"Invalid weights: {weights}. Use 'SAT493M', 'LVD1689M', or a valid path.")

        # Search for weight file matching the backbone name
        weight_path: str | None = None
        search_name = backbone_fn_name + "_pretrain"
        for p in weight_dir.rglob("*.pth"):
            if search_name in p.stem:
                weight_path = str(p)
                break

        assert weight_path is not None, f"Cannot find weight for {backbone_fn_name} at {weight_dir}"
        assert Path(weight_path).exists(), f"Weight file does not exist: {weight_path}"

        sys.path.insert(0, str(repo_dir))
        model = torch.hub.load(str(repo_dir), backbone_fn_name, source="local", weights=weight_path)
        return cast(nn.Module, model)

    def _build_classifier(self, num_classes: int) -> nn.Module:
        """Build linear probe classifier."""
        embed_dim: int = cast(int, self.backbone.embed_dim)
        if self.is_vit:
            # For ViT: compute feature dimension based on n_last_blocks and avgpool
            feature_dim = embed_dim * self.n_last_blocks
            if self.use_avgpool:
                feature_dim += embed_dim
            return get_classifier(
                "linear_probe",
                in_features=feature_dim,
                num_classes=num_classes,
            )
        return get_classifier(
            "convnext_linear",
            in_features=embed_dim,
            num_classes=num_classes,
        )

    def train(self, mode: bool = True) -> "DINOv3Classifier":
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        x = self._preprocess_input(x)
        features = self._extract_features(x)
        logits = self.classifier(features)
        return {"logits": logits}

    def _preprocess_input(self, x: Tensor) -> Tensor:
        """Preprocess input: convert [-1,1] to [0,1] and apply ImageNet normalization."""
        if self.img_is_neg_1_1:
            x = (x + 1) / 2  # [-1, 1] -> [0, 1]
        x = self.normalize(x)  # ImageNet normalization
        return x

    def _extract_features(self, x: Tensor) -> Tensor:
        """Extract features from backbone."""
        if self.freeze_backbone:
            with torch.no_grad():
                return self._forward_backbone(x)
        return self._forward_backbone(x)

    def _forward_backbone(self, x: Tensor) -> Tensor:
        """Forward pass through backbone."""
        if self.is_vit:
            outputs = self.backbone.get_intermediate_layers(  # type: ignore[union-attr]
                x,
                n=self.n_last_blocks,
                reshape=False,
                return_class_token=True,
                norm=True,
            )
            return create_linear_input(list(outputs), self.n_last_blocks, self.use_avgpool)

        out: dict[str, Tensor] = self.backbone.forward_features(x)  # type: ignore[union-attr]
        return out["x_norm_clstoken"]

    def state_dict(self, *args, **kwargs) -> dict[str, Tensor]:
        """Return only classifier weights for checkpoint."""
        full_state = super().state_dict(*args, **kwargs)
        return {k: v for k, v in full_state.items() if k.startswith("classifier.")}

    def load_state_dict(
        self,
        state_dict: dict[str, Tensor],
        strict: bool = True,
    ) -> torch.nn.modules.module._IncompatibleKeys:
        """Load classifier weights only."""
        filtered_state = {k: v for k, v in state_dict.items() if k.startswith("classifier.")}
        if not filtered_state:
            # If state_dict doesn't have classifier prefix, add it
            filtered_state = {f"classifier.{k}": v for k, v in state_dict.items()}
        return super().load_state_dict(filtered_state, strict=False)


def test_vit_classifier():
    """Test DINOv3Classifier with ViT backbone."""
    print("Testing ViT-L16 classifier...")

    # Create model with ViT-L16 backbone
    model = DINOv3Classifier(
        backbone_type="vitl16",
        weights="LVD1689M",  # Use general weights for testing
        pretrained=True,
        freeze_backbone=True,
        n_last_blocks=1,
        use_avgpool=True,
        num_classes=21,  # UCMerced classes
    )

    # Test input
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)

    print(f"Input shape: {x.shape}")
    print(f"Backbone embed_dim: {model.backbone.embed_dim}")

    model.eval()
    with torch.no_grad():
        output = model(x)

    logits = output["logits"]
    print(f"Output logits shape: {logits.shape}")
    assert logits.shape == (batch_size, 21), f"Expected (2, 21), got {logits.shape}"
    print("✅ ViT test passed!")


def test_convnext_classifier():
    """Test DINOv3Classifier with ConvNeXt backbone."""
    print("\nTesting ConvNeXt-Base classifier...")

    # Create model with ConvNeXt backbone
    model = DINOv3Classifier(
        backbone_type="convnext_base",
        weights="LVD1689M",
        pretrained=True,
        freeze_backbone=True,
        num_classes=21,
    )

    # Test input
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)

    print(f"Input shape: {x.shape}")
    print(f"Backbone embed_dim: {model.backbone.embed_dim}")

    model.eval()
    with torch.no_grad():
        output = model(x)

    logits = output["logits"]
    print(f"Output logits shape: {logits.shape}")
    assert logits.shape == (batch_size, 21), f"Expected (2, 21), got {logits.shape}"
    print("✅ ConvNeXt test passed!")


if __name__ == "__main__":
    test_vit_classifier()
    test_convnext_classifier()
