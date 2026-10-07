"""Production-safe NoOp observability implementations."""

from __future__ import annotations

from app.observability.contracts import ProtectedTelemetrySinkPort
from app.observability.protection import (
    ProtectedMetricsRecorder,
    ProtectedObservability,
    build_metric_protection,
    build_observability_protection,
)
from app.runtime.data_protection import ProtectedPayload


class _NoOpProtectedSink(ProtectedTelemetrySinkPort):
    def emit(self, payload: ProtectedPayload) -> None:
        del payload

    def flush(self, *, timeout_seconds: float) -> bool:
        del timeout_seconds
        return True

    def shutdown(self, *, timeout_seconds: float) -> bool:
        del timeout_seconds
        return True


class NoOpObservability(ProtectedObservability):
    def __init__(self) -> None:
        super().__init__(
            protection=build_observability_protection(),
            sink=_NoOpProtectedSink(),
        )


class NoOpMetricsRecorder(ProtectedMetricsRecorder):
    def __init__(self) -> None:
        super().__init__(
            protection=build_metric_protection(),
            sink=_NoOpProtectedSink(),
        )


__all__ = ["NoOpMetricsRecorder", "NoOpObservability"]
