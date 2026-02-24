from __future__ import annotations

from typing import Any, Literal, cast

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.classification import (
    Accuracy,
    F1Score,
    MultilabelAccuracy,
    MultilabelAveragePrecision,
    MultilabelF1Score,
)
from torchmetrics.functional.classification import multilabel_average_precision, multilabel_f1_score


def _reduce_metric_tensor(metric: Tensor) -> Tensor:
    if metric.ndim == 0:
        return metric
    return metric.mean()


def multi_hot_to_index(multi_hot: Tensor, num_classes: int) -> Tensor:
    if num_classes <= 0:
        raise ValueError(f"num_classes must be > 0, got {num_classes}.")
    if multi_hot.ndim < 1:
        raise ValueError(f"multi_hot must have at least 1 dimension, got shape={tuple(multi_hot.shape)}.")
    if int(multi_hot.shape[-1]) != int(num_classes):
        raise ValueError(
            f"Last dimension of multi_hot must equal num_classes, got shape={tuple(multi_hot.shape)}, "
            f"num_classes={num_classes}."
        )

    positive_mask = multi_hot > 0
    positive_count = positive_mask.sum(dim=-1)
    if not torch.all(positive_count == 1):
        invalid_samples = int((positive_count != 1).sum().item())
        raise ValueError(
            "multi_hot_to_index expects exactly one positive label per sample, "
            f"but found {invalid_samples} invalid sample(s)."
        )
    return torch.argmax(multi_hot, dim=-1).long()


class MultilabelMetricsBase(Metric):
    def __init__(
        self,
        num_labels: int,
        threshold: float,
        map_average: Literal["micro", "macro", "weighted", "none"] = "macro",
        num_eval_labels: int | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if not (0.0 < threshold < 1.0):
            raise ValueError(f"threshold must be in (0,1), got {threshold}")
        self.threshold = float(threshold)
        self.map_average = map_average
        self.num_labels = int(num_labels)
        self.num_eval_labels = int(num_eval_labels) if num_eval_labels is not None else self.num_labels
        if self.num_eval_labels <= 0:
            raise ValueError(f"num_eval_labels must be > 0, got {self.num_eval_labels}.")

        self.acc_metric: MultilabelAccuracy = MultilabelAccuracy(
            num_labels=self.num_eval_labels,
            threshold=self.threshold,
            average="micro",
        )
        self.f1_metric: MultilabelF1Score = MultilabelF1Score(
            num_labels=self.num_eval_labels,
            threshold=self.threshold,
            average="macro",
        )
        self.map_metric: MultilabelAveragePrecision = MultilabelAveragePrecision(
            num_labels=self.num_eval_labels,
            average=cast(Literal["micro", "macro", "weighted", "none"], self.map_average),
        )
        if device is not None:
            self.to(device)

    def _prepare_logits_target(self, logits: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        return logits, target

    def compute_train_metrics(self, logits: Tensor, target: Tensor) -> dict[str, Tensor]:
        eval_logits, eval_target = self._prepare_logits_target(logits, target)
        threshold_tensor = torch.tensor(self.threshold, dtype=eval_logits.dtype, device=eval_logits.device)
        pred = (eval_logits >= torch.logit(threshold_tensor)).float()
        acc = (pred == eval_target.float()).float().mean()
        train_f1 = multilabel_f1_score(
            eval_logits.detach(),
            eval_target,
            num_labels=self.num_eval_labels,
            threshold=self.threshold,
            average="macro",
        )
        train_map = multilabel_average_precision(
            eval_logits.detach(),
            eval_target,
            num_labels=self.num_eval_labels,
            average=cast(Literal["micro", "macro", "weighted", "none"], self.map_average),
        )
        return {
            "acc": acc.detach(),
            "mf1": _reduce_metric_tensor(train_f1).detach(),
            "map": _reduce_metric_tensor(train_map).detach(),
        }

    def reset(self) -> None:
        super().reset()
        self.acc_metric.reset()
        self.f1_metric.reset()
        self.map_metric.reset()

    def update(self, logits: Tensor, target: Tensor) -> None:
        eval_logits, eval_target = self._prepare_logits_target(logits, target)
        cast(Any, self.acc_metric).update(eval_logits, eval_target)
        cast(Any, self.f1_metric).update(eval_logits, eval_target)
        cast(Any, self.map_metric).update(eval_logits, eval_target)

    def compute(self) -> dict[str, Tensor]:
        val_acc = cast(Tensor, cast(Any, self.acc_metric).compute())
        val_f1 = cast(Tensor, cast(Any, self.f1_metric).compute())
        val_map = cast(Tensor, cast(Any, self.map_metric).compute())
        return {
            "val_acc": _reduce_metric_tensor(val_acc),
            "val_f1": _reduce_metric_tensor(val_f1),
            "val_map": _reduce_metric_tensor(val_map),
        }

    def compute_val_metrics(self) -> dict[str, Tensor]:
        val_acc = cast(Tensor, cast(Any, self.acc_metric).compute())
        val_f1 = cast(Tensor, cast(Any, self.f1_metric).compute())
        val_map = cast(Tensor, cast(Any, self.map_metric).compute())
        return {
            "val_acc": _reduce_metric_tensor(val_acc),
            "val_f1": _reduce_metric_tensor(val_f1),
            "val_map": _reduce_metric_tensor(val_map),
        }


class TreeSatMultilabelMetrics(MultilabelMetricsBase):
    def __init__(
        self,
        num_labels: int,
        threshold: float,
        map_average: Literal["micro", "macro", "weighted", "none"] = "macro",
        device: torch.device | None = None,
    ) -> None:
        super().__init__(
            num_labels=num_labels,
            threshold=threshold,
            map_average=map_average,
            num_eval_labels=num_labels - 2,
            device=device,
        )

    def _prepare_logits_target(self, logits: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        target_int = target.int()
        return logits[:, 1:-1], target_int[:, 1:-1]


class MultilabelMetrics(MultilabelMetricsBase):
    def _prepare_logits_target(self, logits: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        return logits, target.int()


class MulticlassMetrics(Metric):
    def __init__(
        self,
        num_classes: int,
        top_k: int = 1,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.acc_metric: Accuracy = Accuracy(task="multiclass", num_classes=num_classes, top_k=top_k, average="micro")
        self.f1_metric: F1Score = F1Score(task="multiclass", num_classes=num_classes, top_k=top_k, average="macro")
        self.acc_metric_top5: Accuracy = Accuracy(task="multiclass", num_classes=num_classes, top_k=5, average="micro")
        if device is not None:
            self.to(device)

    @staticmethod
    def compute_train_metrics(logits: Tensor, target: Tensor) -> dict[str, Tensor]:
        pred = torch.argmax(logits, dim=1)
        acc = (pred == target).float().mean()
        return {"acc": acc.detach()}

    def reset(self) -> None:
        super().reset()
        self.acc_metric.reset()
        self.f1_metric.reset()
        self.acc_metric_top5.reset()

    def update(self, logits: Tensor, target: Tensor) -> None:
        cast(Any, self.acc_metric).update(logits, target)
        cast(Any, self.f1_metric).update(logits, target)
        cast(Any, self.acc_metric_top5).update(logits, target)

    def compute(self) -> dict[str, Tensor]:
        val_acc = cast(Tensor, cast(Any, self.acc_metric).compute())
        val_f1 = cast(Tensor, cast(Any, self.f1_metric).compute())
        val_acc_top5 = cast(Tensor, cast(Any, self.acc_metric_top5).compute())
        return {
            "val_acc": _reduce_metric_tensor(val_acc),
            "val_f1": _reduce_metric_tensor(val_f1),
            "val_acc_top5": _reduce_metric_tensor(val_acc_top5),
        }

    def compute_val_metrics(self) -> dict[str, Tensor]:
        val_acc = cast(Tensor, cast(Any, self.acc_metric).compute())
        val_f1 = cast(Tensor, cast(Any, self.f1_metric).compute())
        val_acc_top5 = cast(Tensor, cast(Any, self.acc_metric_top5).compute())
        return {
            "val_acc": _reduce_metric_tensor(val_acc),
            "val_f1": _reduce_metric_tensor(val_f1),
            "val_acc_top5": _reduce_metric_tensor(val_acc_top5),
        }
