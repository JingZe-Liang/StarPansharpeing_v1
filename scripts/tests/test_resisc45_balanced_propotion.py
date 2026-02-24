from collections import Counter

import pytest

from src.stage2.classification.data.RESISC45_torchgeo import RESISC45


def test_build_balanced_indices_with_propotion() -> None:
    labels = [0] * 10 + [1] * 6 + [2] * 4

    sampled = RESISC45._build_balanced_indices(labels=labels, propotion=0.5, seed=123)

    sampled_labels = [labels[index] for index in sampled]
    class_counts = Counter(sampled_labels)

    assert len(sampled) == 6
    assert class_counts == Counter({0: 2, 1: 2, 2: 2})


def test_build_balanced_indices_propotion_one_keeps_all() -> None:
    labels = [0] * 10 + [1] * 6 + [2] * 4

    sampled = RESISC45._build_balanced_indices(labels=labels, propotion=1.0, seed=123)

    assert sampled == list(range(len(labels)))


def test_build_balanced_indices_invalid_propotion() -> None:
    labels = [0, 0, 1, 1]

    with pytest.raises(ValueError, match="propotion must be in"):
        RESISC45._build_balanced_indices(labels=labels, propotion=0.0, seed=123)
