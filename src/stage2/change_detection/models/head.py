from timm.layers.classifier import (
    ClassifierHead,
    NormMlpClassifierHead,
)


def get_classifier(classifier_type: str = "default", **classifier_kwargs):
    """
    # Default head kwargs
    in_features: int,
    num_classes: int,
    pool_type: str = 'avg',
    drop_rate: float = 0.,
    use_conv: bool = False,
    input_fmt: str = 'NCHW',

    # norm_mlp head kwargs
    in_features: int,
    num_classes: int,
    hidden_size: Optional[int] = None,
    pool_type: str = 'avg',
    drop_rate: float = 0.,
    norm_layer: Union[str, Callable] = 'layernorm2d',
    act_layer: Union[str, Callable] = 'tanh',
    """

    head_cls_mapping = {
        "default": ClassifierHead,
        "norm_mlp": NormMlpClassifierHead,
    }
    head_cls = head_cls_mapping.get(classifier_type, None)
    assert head_cls is not None, f"Unsupported classifier type: {classifier_type}"

    # Init
    classifier = head_cls(**classifier_kwargs)
    return classifier
