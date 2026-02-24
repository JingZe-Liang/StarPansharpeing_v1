from __future__ import annotations

from typing import Any, cast

import torch
from torchmetrics.functional.classification import multilabel_f1_score

from src.stage2.SSL_third_party.OmniSat.src.models.metrics.metrics import MetricsMultiModal
from src.stage2.classification.metrics.basic import MulticlassMetrics, MultilabelMetrics, TreeSatMultilabelMetrics


def _assert_close(a: torch.Tensor, b: torch.Tensor, atol: float = 1e-6) -> None:
    assert torch.allclose(a, b, atol=atol, rtol=0), f"{a.item()} != {b.item()} (atol={atol})"


def test_treesat_metric_matches_official_f1_macro_micro_weighted() -> None:
    official = MetricsMultiModal(modalities=["aerial", "s2", "s1-asc"], num_classes=15)
    ours = TreeSatMultilabelMetrics(num_labels=15, threshold=0.5, map_average="macro")

    test_cases: list[tuple[torch.Tensor, torch.Tensor]] = []
    for seed in [7, 23, 101]:
        g = torch.Generator().manual_seed(seed)
        logits = torch.randn((16, 15), generator=g)
        label = torch.randint(0, 2, (16, 15), generator=g).float()
        test_cases.append((logits, label))

    test_cases.append((torch.zeros((8, 15)), torch.zeros((8, 15))))
    test_cases.append((torch.randn((8, 15)), torch.ones((8, 15))))

    for logits, label in test_cases:
        cast(Any, official).update({"aerial": logits, "s2": logits, "s1-asc": logits}, {"label": label})
        cast(Any, ours).update(logits, label)

    off = cast(dict[str, torch.Tensor], cast(Any, official).compute())
    out = cast(dict[str, torch.Tensor], cast(Any, ours).compute())

    eval_logits, eval_target = ours._prepare_logits_target(test_cases[0][0], test_cases[0][1])
    assert eval_logits.shape[1] == 13
    assert eval_target.shape[1] == 13

    _assert_close(out["val_f1"], off["F1_Score_macro"])

    all_logits = torch.cat([x[0] for x in test_cases], dim=0)[:, 1:-1]
    all_target = torch.cat([x[1] for x in test_cases], dim=0)[:, 1:-1].int()
    micro = multilabel_f1_score(all_logits, all_target, num_labels=13, threshold=0.5, average="micro")
    weighted = multilabel_f1_score(all_logits, all_target, num_labels=13, threshold=0.5, average="weighted")
    _assert_close(micro, off["F1_Score_micro"])
    _assert_close(weighted, off["F1_Score_weighted"])


def test_multilabel_metrics_non_treesat_matches_torchmetrics_macro() -> None:
    metric = MultilabelMetrics(num_labels=15, threshold=0.5, map_average="macro")
    g = torch.Generator().manual_seed(9)
    logits = torch.randn((20, 15), generator=g)
    target = torch.randint(0, 2, (20, 15), generator=g).float()

    logs = metric.compute_train_metrics(logits, target)
    macro_ref = multilabel_f1_score(logits, target.int(), num_labels=15, threshold=0.5, average="macro")
    _assert_close(logs["mf1"], macro_ref)


def test_multiclass_metrics_outputs_scalars() -> None:
    metric = MulticlassMetrics(num_classes=6, top_k=1)
    g = torch.Generator().manual_seed(42)
    logits = torch.randn((24, 6), generator=g)
    target = torch.randint(0, 6, (24,), generator=g)

    cast(Any, metric).update(logits, target)
    out = cast(dict[str, torch.Tensor], cast(Any, metric).compute())
    assert set(out.keys()) == {"val_acc", "val_f1", "val_acc_top5"}
    for value in out.values():
        assert isinstance(value, torch.Tensor)
        assert value.ndim == 0
