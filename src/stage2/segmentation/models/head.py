import torch
from timm.layers.classifier import (
    ClassifierHead,
    NormMlpClassifierHead,
)
from timm.models.layers import trunc_normal_
from torch import Tensor, nn


def create_linear_input(
    x_tokens_list: list[tuple[Tensor, Tensor]],
    use_n_blocks: int,
    use_avgpool: bool,
) -> Tensor:
    """Create linear classifier input from intermediate layer outputs.

    Args:
        x_tokens_list: List of (patch_tokens, class_token) tuples from last n blocks
        use_n_blocks: Number of last blocks to use
        use_avgpool: Whether to concatenate avgpooled patch tokens

    Returns:
        Tensor of shape (B, feature_dim)
    """
    intermediate_output = x_tokens_list[-use_n_blocks:]
    output = torch.cat([class_token for _, class_token in intermediate_output], dim=-1)
    if use_avgpool:
        output = torch.cat(
            (
                output,
                torch.mean(intermediate_output[-1][0], dim=1),  # patch tokens
            ),
            dim=-1,
        )
        output = output.reshape(output.shape[0], -1)
    return output.float()


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features.

    Following DINOv3's official implementation with BatchNorm + Linear.
    Used for ViT backbones with intermediate token outputs.
    """

    def __init__(
        self,
        out_dim: int,
        num_classes: int = 1000,
        use_n_blocks: int = 1,
        use_avgpool: bool = True,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.use_n_blocks = use_n_blocks
        self.use_avgpool = use_avgpool
        self.num_classes = num_classes

        self.bn = nn.BatchNorm1d(out_dim, affine=False, eps=1e-6)
        self.linear = nn.Linear(out_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x_tokens_list: list[tuple[Tensor, Tensor]]) -> Tensor:
        output = create_linear_input(x_tokens_list, self.use_n_blocks, self.use_avgpool)
        return self.linear(self.bn(output))


class ConvNextLinearClassifier(nn.Module):
    """Linear classifier for ConvNeXt backbone.

    ConvNeXt uses global average pooled features directly.
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int = 1000,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes

        self.bn = nn.BatchNorm1d(in_dim, affine=False, eps=1e-6)
        self.linear = nn.Linear(in_dim, num_classes)
        trunc_normal_(self.linear.weight, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(self.bn(x))


class LinearProbeHead(nn.Module):
    """Simple linear probe head for 1D features.

    BatchNorm1d(affine=False) + Linear. Suitable for CLS token classification.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(in_features, affine=False, eps=1e-6)
        self.linear = nn.Linear(in_features, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(self.bn(x))


def get_classifier(classifier_type: str = "default", **classifier_kwargs):
    """Get a classifier head by type.

    Supported types:
    - "default": timm's ClassifierHead (expects NCHW input)
    - "norm_mlp": timm's NormMlpClassifierHead
    - "linear_probe": Simple BN + Linear (expects 1D input)
    - "convnext_linear": ConvNeXt style BN + Linear

    # Default/norm_mlp head kwargs
    in_features: int,
    num_classes: int,
    pool_type: str = 'avg',
    drop_rate: float = 0.,
    ...

    # linear_probe kwargs
    in_features: int,
    num_classes: int,
    """

    head_cls_mapping = {
        "default": ClassifierHead,
        "norm_mlp": NormMlpClassifierHead,
        "linear_probe": LinearProbeHead,
        "convnext_linear": ConvNextLinearClassifier,
    }

    # Handle key name differences
    if classifier_type in ("linear_probe", "convnext_linear"):
        # These use in_dim instead of in_features for ConvNextLinearClassifier
        if classifier_type == "convnext_linear" and "in_features" in classifier_kwargs:
            classifier_kwargs["in_dim"] = classifier_kwargs.pop("in_features")

    head_cls = head_cls_mapping.get(classifier_type, None)
    assert head_cls is not None, f"Unsupported classifier type: {classifier_type}"

    # Init
    classifier = head_cls(**classifier_kwargs)
    return classifier
