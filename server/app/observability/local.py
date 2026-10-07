"""Local stdlib JSON sink for already-protected canonical telemetry bytes."""

from __future__ import annotations

import logging

from app.observability.contracts import ObservabilityContractError
from app.runtime.data_protection import ProtectedPayload


class LocalJsonTelemetrySink:
    """Emit one canonical protected JSON object per stdlib log record."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or _build_default_local_logger()

    def emit(self, payload: ProtectedPayload) -> None:
        if type(payload) is not ProtectedPayload:
            raise ObservabilityContractError("local telemetry payload is invalid")
        canonical = payload.canonical_bytes
        if type(canonical) is not bytes:
            raise ObservabilityContractError("local telemetry bytes are invalid")
        self._logger.info(canonical.decode("utf-8"))

    def flush(self, *, timeout_seconds: float) -> bool:
        if type(timeout_seconds) not in {int, float} or timeout_seconds <= 0:
            raise ObservabilityContractError("telemetry timeout is invalid")
        for handler in tuple(self._logger.handlers):
            handler.flush()
        return True

    def shutdown(self, *, timeout_seconds: float) -> bool:
        return self.flush(timeout_seconds=timeout_seconds)


def _build_default_local_logger() -> logging.Logger:
    """Build one self-contained INFO logger without relying on root config."""

    logger = logging.Logger("app.observability.local", level=logging.INFO)
    logger.propagate = False
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger


__all__ = ["LocalJsonTelemetrySink"]
