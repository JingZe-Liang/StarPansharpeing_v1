import numpy as np
import pytest
import torch

from src.utilities.train_utils.state import LossMetricTracker


def _reset_tracker_state() -> None:
    LossMetricTracker.remove_instance()


def test_add_tracked_accepts_tensor_and_ndarray() -> None:
    _reset_tracker_state()
    tracker = LossMetricTracker(
        loss_metrics_values={"loss": 0.0},
        loss_metrics_tracked={"loss": [0.1]},
    )

    tracker.add_tracked("loss", torch.tensor([0.2, 0.3]))
    tracker.add_tracked("loss", np.array([0.4, 0.5]))

    values, _ = tracker.get(name=["loss"], track_value_op="all")
    tracked = tracker.get_tracked_values_op(name=["loss"], track_value_op="all")
    assert values["loss"] == 0.0
    assert tracked["loss"] == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5])


def test_track_new_key_uses_tracked_dict() -> None:
    _reset_tracker_state()
    tracker = LossMetricTracker(loss_metrics_values={"loss": 0.0}, loss_metrics_tracked={"loss": [0.1]})

    tracker.track("new_metric", [1.0, 2.0])
    tracked = tracker.get_tracked_values_op(name=["new_metric"], track_value_op="all")
    assert tracked["new_metric"] == [1.0, 2.0]


def test_sync_state_raises_on_length_mismatch(monkeypatch) -> None:
    _reset_tracker_state()
    tracker = LossMetricTracker(loss_metrics_values={"loss": 1.0}, loss_metrics_tracked={"loss": [0.1, 0.2]})

    class _FakeDist:
        @staticmethod
        def is_initialized() -> bool:
            return True

        @staticmethod
        def get_world_size() -> int:
            return 2

        @staticmethod
        def all_gather_object(out, obj) -> None:
            other = LossMetricTracker(
                loss_metrics_values={"loss": 2.0},
                loss_metrics_tracked={"loss": [0.3]},
                __force_new_instance__=True,
            )
            out[0] = obj
            out[1] = other

    monkeypatch.setattr(torch, "distributed", _FakeDist)

    with pytest.raises(ValueError, match="length mismatch"):
        LossMetricTracker.sync_state()
