# Protocol
from typing import Protocol, runtime_checkable

from ...anomaly_detection.metrics import AnomalyDetectionMetrics, HADDetectionMetrics
from ...pansharpening.metrics import AnalysisPanAcc, PansharpeningMetrics
from ...segmentation.metrics import HyperSegmentationScore
from ...unmixing.metrics import UnmixingMetrics


@runtime_checkable
class MetricProtocol(Protocol):
    def update(self, *args, **kwargs) -> None: ...

    def compute(self) -> dict: ...

    def reset(self) -> None: ...


def is_basic_metric(metric):
    return isinstance(metric, MetricProtocol)
