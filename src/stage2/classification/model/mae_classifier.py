"""MAE backbone for remote sensing scene classification using timm.

This model uses standard Vision Transformer (ViT) backbones pre-trained with
Masked Autoencoders (MAE) on natural images (ImageNet-1K).
"""

from typing import Literal

import torch
import torch.nn as nn
import timm


class MAEClassifier(nn.Module):
    """MAE backbone with classifier for scene classification.

    This class provides a wrapper around timm's ViT models pre-trained with MAE.
    It is designed for remote sensing scene classification tasks, supporting
    RGB input and optional backbone freezing.

    Attributes:
        backbone_type: One of "vit_base_patch16_224.mae", "vit_large_patch16_224.mae",
            "vit_huge_patch14_224.mae".
        pretrained: Whether to load ImageNet-1K MAE weights.
        freeze_backbone: If True, only the classification head will be trained.
        num_classes: Number of target categories.
        img_size: Size to which input images will be resized/interpolated.
        img_is_neg_1_1: If True, input is assumed to be in [-1, 1] range and
            will be scaled to [0, 1] for the backbone.
    """

    SUPPORTED_BACKBONES = [
        "vit_base_patch16_224.mae",
        "vit_large_patch16_224.mae",
        "vit_huge_patch14_224.mae",
    ]

    def __init__(
        self,
        backbone_type: Literal[
            "vit_base_patch16_224.mae",
            "vit_large_patch16_224.mae",
            "vit_huge_patch14_224.mae",
        ] = "vit_base_patch16_224.mae",
        pretrained: bool = True,
        freeze_backbone: bool = True,
        num_classes: int = 21,
        img_size: int = 224,
        img_is_neg_1_1: bool = True,
    ) -> None:
        super().__init__()

        if backbone_type not in self.SUPPORTED_BACKBONES:
            raise ValueError(f"Unsupported backbone: {backbone_type}. Choose from {self.SUPPORTED_BACKBONES}")

        self.backbone_type = backbone_type
        self.freeze_backbone = freeze_backbone
        self.img_is_neg_1_1 = img_is_neg_1_1

        print(f"🚀 Creating MAE Classifier with backbone: {backbone_type}")
        print(f"   Pretrained: {pretrained}")
        print(f"   Freeze Backbone: {freeze_backbone}")
        print(f"   Num Classes: {num_classes}")

        # Create MAE backbone from timm
        # For MAE classification, global average pooling ('avg') is typically used
        # which averages all patch tokens (often including the CLS token if present).
        self.backbone = timm.create_model(
            backbone_type,
            pretrained=pretrained,
            num_classes=num_classes,
            img_size=img_size,
            global_pool="avg",
        )

        if freeze_backbone:
            # Freeze all parameters except the head
            for name, param in self.backbone.named_parameters():
                if not name.startswith("head"):
                    param.requires_grad_(False)
            # Ensure backbone is in eval mode initially
            self.backbone.eval()

    def train(self, mode: bool = True) -> "MAEClassifier":
        """Override train to maintain eval mode for backbone if frozen."""
        super().train(mode)
        if self.freeze_backbone:
            # Keep backbone in eval mode, but allow head to be in train mode
            for name, module in self.backbone.named_modules():
                if name != "" and not name.startswith("head"):
                    module.eval()
        return self

    def parameters(self, recurse: bool = True):
        """Standard parameters generator, filters if frozen."""
        for name, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True):
        """Filter named parameters based on freeze_backbone setting."""
        for name, param in super().named_parameters(prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate):
            if self.freeze_backbone:
                # Only expose trainable parameters (the head)
                if param.requires_grad:
                    yield name, param
            else:
                yield name, param

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Perform forward pass and return dict with logits."""
        x = self._preprocess_input(x)
        logits = self.backbone(x)
        return {"logits": logits}

    def _preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """Handle [-1, 1] normalization conversion if required."""
        if self.img_is_neg_1_1:
            # Map [-1, 1] -> [0, 1] as expected by timm models
            x = (x + 1) / 2
        return x

    def state_dict(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """Filter state_dict to save only the head weights if frozen."""
        full_state = super().state_dict(*args, **kwargs)
        if self.freeze_backbone:
            return {k: v for k, v in full_state.items() if "head" in k}
        return full_state

    def load_state_dict(
        self, state_dict: dict[str, torch.Tensor], strict: bool = True
    ) -> torch.nn.modules.module._IncompatibleKeys:
        """Filter state_dict for loading, supporting both full and head-only dicts."""
        # Only load head parameters if they are the only ones provided
        filtered_state = {k: v for k, v in state_dict.items() if "head" in k}

        # If strict or full state provided, handle accordingly
        if not filtered_state:
            # Maybe the state_dict doesn't have the 'backbone.' prefix
            filtered_state = {f"backbone.head.{k}": v for k, v in state_dict.items() if "head" not in k}

        return super().load_state_dict(filtered_state, strict=False)


def test_mae_classifier() -> None:
    """Test MAEClassifier."""
    print("Testing MAE Classifier...")

    for backbone_type in MAEClassifier.SUPPORTED_BACKBONES[:1]:  # Test vit_base
        print(f"\n{'=' * 60}")
        print(f"Testing {backbone_type}")
        print(f"{'=' * 60}")

        model = MAEClassifier(
            backbone_type=backbone_type,  # type: ignore[arg-type]
            pretrained=False,  # Don't download rewards for quick test
            freeze_backbone=True,
            num_classes=21,
            img_size=224,
        )

        # Count parameters
        total_p = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {total_p}")
        # Base ViT head is embed_dim * num_classes = 768 * 21 = 16128 (+ bias)
        # 16128 + 21 = 16149.
        assert total_p < 20000, f"Too many trainable parameters: {total_p}"

        # Test input
        batch_size = 2
        x = torch.randn(batch_size, 3, 224, 224)

        print(f"Input shape: {x.shape}")

        model.eval()
        with torch.no_grad():
            output = model(x)

        logits = output["logits"]
        print(f"Output logits shape: {logits.shape}")
        assert logits.shape == (batch_size, 21), f"Expected (2, 21), got {logits.shape}"
        print(f"✅ {backbone_type} test passed!")


if __name__ == "__main__":
    test_mae_classifier()
