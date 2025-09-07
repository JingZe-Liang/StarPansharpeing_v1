from torchmetrics.classification import ConfusionMatrix

from ...segmentation.metrics import HyperSegmentationScore


class ChangeDetectionScore(HyperSegmentationScore):
    def __init__(
        self,
        n_classes: int = 2,
        ignore_index: int | None = None,
        top_k: int = 1,
        reduction: str = "macro",
        per_class: bool = False,
        include_bg: bool = False,
        use_aggregation: bool = False,
    ):
        super().__init__(
            n_classes=n_classes,
            ignore_index=ignore_index,
            top_k=top_k,
            reduction=reduction,
            per_class=per_class,
            include_bg=include_bg,
            use_aggregation=use_aggregation,
        )
        self.confusion_matrix = ConfusionMatrix(
            task="multiclass", num_classes=n_classes, ignore_index=ignore_index
        )
        self._all_metric_fns.update(dict(confusion_matrix=self.confusion_matrix))
