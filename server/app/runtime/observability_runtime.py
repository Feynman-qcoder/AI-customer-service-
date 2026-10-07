"""Attempt-scoped orchestration for optional provider-neutral telemetry."""

from __future__ import annotations

import asyncio
import time
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from threading import RLock

from app.observability import (
    ActionOperationRecordV1,
    AttemptIdentityV1,
    AttemptOutcome,
    AttemptResumeKind,
    AttemptScopeV1,
    HITLOperationRecordV1,
    LLMOperationRecordV1,
    MemoryOperationRecordV1,
    MetricKind,
    MetricLabelV1,
    MetricName,
    MetricPointV1,
    MetricsRecorder,
    MetricUnit,
    NodeOperationRecordV1,
    NormalizedErrorCode,
    NormalizedErrorTypeV1,
    ObservabilityContractError,
    ObservabilityPort,
    ObservabilityStatus,
    OperationCode,
    OperationRecordV1,
    RetrievalOperationRecordV1,
    RevalidationOperationRecordV1,
    RunIdentityV1,
    SpanOperation,
    SpanScopeV1,
    ToolOperationRecordV1,
)
from app.runtime.context import ExecutionScope


@dataclass(frozen=True, slots=True)
class AttemptObservationV1:
    scope: AttemptScopeV1 | None
    started_ns: int


class ObservabilityRuntime:
    """One injected facade; telemetry failures never change business results."""

    def __init__(
        self,
        observability: ObservabilityPort,
        metrics: MetricsRecorder,
    ) -> None:
        self._observability = observability
        self._metrics = metrics
        self._scope_lock = RLock()
        self._scopes: dict[str, AttemptScopeV1] = {}

    def begin(
        self,
        execution: ExecutionScope,
        *,
        resume: AttemptResumeKind,
    ) -> AttemptObservationV1:
        started_ns = time.monotonic_ns()
        try:
            run = RunIdentityV1(
                conversation_id=execution.conversation_id,
                thread_id=execution.thread_id,
                run_id=execution.run_id,
                subject_user_id=execution.subject.user_id,
            )
            fence = execution.lease.fence_token
            if type(fence) is not int or fence <= 0:
                raise ObservabilityContractError("attempt fence is invalid")
            self._observability.register_run(run)
            scope = self._observability.start_attempt(
                AttemptIdentityV1(
                    run=run,
                    attempt_id=execution.attempt_id,
                    fence_token=fence,
                ),
                resume=resume,
            )
        except Exception:
            scope = None
        if scope is not None:
            with self._scope_lock:
                self._scopes[execution.attempt_id] = scope
        return AttemptObservationV1(scope=scope, started_ns=started_ns)

    def finish(
        self,
        observation: AttemptObservationV1 | None,
        *,
        outcome: AttemptOutcome,
        error_type: NormalizedErrorTypeV1 | None = None,
    ) -> None:
        if observation is None or observation.scope is None:
            return
        if (
            error_type is not None
            and error_type.code is NormalizedErrorCode.CANCELLED
        ):
            outcome = AttemptOutcome.CANCELLED
        duration_seconds = max(
            0.0,
            (time.monotonic_ns() - observation.started_ns) / 1_000_000_000,
        )
        status = _metric_status(ObservabilityStatus(outcome.value))
        self._record_metric(
            MetricPointV1(
                name=MetricName.ATTEMPT_TOTAL,
                kind=MetricKind.COUNTER,
                unit=MetricUnit.COUNT,
                value=1,
                labels=(
                    MetricLabelV1(key="status", value=status),
                    MetricLabelV1(key="resume", value=observation.scope.resume.value),
                ),
            )
        )
        self._record_metric(
            MetricPointV1(
                name=MetricName.ATTEMPT_DURATION_SECONDS,
                kind=MetricKind.HISTOGRAM,
                unit=MetricUnit.SECONDS,
                value=duration_seconds,
                labels=(MetricLabelV1(key="status", value=status),),
            )
        )
        try:
            self._observability.end_attempt(
                observation.scope,
                outcome=outcome,
                error_type=error_type,
            )
        except Exception:
            pass
        finally:
            with self._scope_lock:
                current = self._scopes.get(observation.scope.identity.attempt_id)
                if current is observation.scope:
                    self._scopes.pop(observation.scope.identity.attempt_id, None)

    def scope_for(self, attempt_id: str) -> AttemptScopeV1 | None:
        with self._scope_lock:
            return self._scopes.get(attempt_id)

    def flush(self, *, timeout_seconds: float = 1.0) -> None:
        try:
            self._observability.flush(timeout_seconds=timeout_seconds)
        except Exception:
            return

    def shutdown(self, *, timeout_seconds: float = 1.0) -> None:
        try:
            self._observability.shutdown(timeout_seconds=timeout_seconds)
        except Exception:
            return

    def record(
        self,
        scope: AttemptScopeV1 | None,
        record: OperationRecordV1,
    ) -> None:
        if scope is None:
            return
        try:
            self._observability.record_operation(scope, record)
        except Exception:
            return
        self._record_operation_metrics(record)

    def span(
        self,
        scope: AttemptScopeV1 | None,
        *,
        operation: SpanOperation,
    ) -> AbstractContextManager[SpanScopeV1 | None]:
        if scope is None:
            return nullcontext(None)
        try:
            return self._observability.span(scope, operation=operation)
        except Exception:
            return nullcontext(None)

    def _record_operation_metrics(self, record: OperationRecordV1) -> None:
        status = _metric_status(record.status)
        seconds = record.duration_ms / 1000
        points: list[MetricPointV1] = []
        if isinstance(record, LLMOperationRecordV1):
            operation = record.operation.value.removeprefix("LLM_")
            points.extend(
                (
                    MetricPointV1(
                        MetricName.LLM_CALL_TOTAL,
                        MetricKind.COUNTER,
                        MetricUnit.COUNT,
                        1,
                        (
                            MetricLabelV1("operation", operation),
                            MetricLabelV1("status", status),
                            MetricLabelV1("model_family", record.model_family.value),
                        ),
                    ),
                    MetricPointV1(
                        MetricName.LLM_DURATION_SECONDS,
                        MetricKind.HISTOGRAM,
                        MetricUnit.SECONDS,
                        seconds,
                        (
                            MetricLabelV1("operation", operation),
                            MetricLabelV1("status", status),
                        ),
                    ),
                )
            )
            for direction, value in (
                ("INPUT", record.prompt_tokens),
                ("OUTPUT", record.completion_tokens),
            ):
                if value is not None and value > 0:
                    points.append(
                        MetricPointV1(
                            MetricName.LLM_TOKENS_TOTAL,
                            MetricKind.COUNTER,
                            MetricUnit.TOKENS,
                            value,
                            (
                                MetricLabelV1("operation", operation),
                                MetricLabelV1("direction", direction),
                            ),
                        )
                    )
        elif isinstance(record, RetrievalOperationRecordV1):
            points.append(
                MetricPointV1(
                    MetricName.RETRIEVAL_DURATION_SECONDS,
                    MetricKind.HISTOGRAM,
                    MetricUnit.SECONDS,
                    seconds,
                    (
                        MetricLabelV1("channel", record.channel.value),
                        MetricLabelV1("status", status),
                    ),
                )
            )
        elif isinstance(record, ToolOperationRecordV1):
            points.extend(
                (
                    MetricPointV1(
                        MetricName.TOOL_CALL_TOTAL,
                        MetricKind.COUNTER,
                        MetricUnit.COUNT,
                        1,
                        (
                            MetricLabelV1("tool_name", record.tool_name.value),
                            MetricLabelV1("status", status),
                            MetricLabelV1("risk_level", record.risk_level.value),
                        ),
                    ),
                    MetricPointV1(
                        MetricName.TOOL_DURATION_SECONDS,
                        MetricKind.HISTOGRAM,
                        MetricUnit.SECONDS,
                        seconds,
                        (
                            MetricLabelV1("tool_name", record.tool_name.value),
                            MetricLabelV1("status", status),
                        ),
                    ),
                )
            )
        elif isinstance(record, ActionOperationRecordV1):
            points.append(
                MetricPointV1(
                    MetricName.ACTION_TOTAL,
                    MetricKind.COUNTER,
                    MetricUnit.COUNT,
                    1,
                    (
                        MetricLabelV1("action_type", record.action_type.value),
                        MetricLabelV1("status", status),
                    ),
                )
            )
        elif isinstance(record, MemoryOperationRecordV1) and record.operation is OperationCode.MEMORY_SUMMARY:
            points.append(
                MetricPointV1(
                    MetricName.MEMORY_SUMMARY_TOTAL,
                    MetricKind.COUNTER,
                    MetricUnit.COUNT,
                    1,
                    (MetricLabelV1("status", status),),
                )
            )
        elif isinstance(record, HITLOperationRecordV1):
            if record.operation in {
                OperationCode.HITL_CUSTOMER_INTERRUPT,
                OperationCode.HITL_ADMIN_INTERRUPT,
            }:
                reason = (
                    "CUSTOMER_CONFIRMATION"
                    if record.operation is OperationCode.HITL_CUSTOMER_INTERRUPT
                    else "ADMIN_APPROVAL"
                )
                points.append(
                    MetricPointV1(
                        MetricName.INTERRUPT_TOTAL,
                        MetricKind.COUNTER,
                        MetricUnit.COUNT,
                        1,
                        (MetricLabelV1("reason", reason),),
                    )
                )
            else:
                points.append(
                    MetricPointV1(
                        MetricName.RESUME_TOTAL,
                        MetricKind.COUNTER,
                        MetricUnit.COUNT,
                        1,
                        (MetricLabelV1("status", status),),
                    )
                )
        elif isinstance(record, NodeOperationRecordV1 | RevalidationOperationRecordV1):
            return
        for point in points:
            self._record_metric(point)

    def _record_metric(self, point: MetricPointV1) -> None:
        try:
            self._metrics.record(point)
        except Exception:
            return


def normalized_error_type(error: BaseException) -> NormalizedErrorTypeV1:
    if isinstance(error, asyncio.CancelledError):
        code = NormalizedErrorCode.CANCELLED
    elif isinstance(error, TimeoutError):
        code = NormalizedErrorCode.TIMEOUT
    elif isinstance(error, PermissionError):
        code = NormalizedErrorCode.AUTHORIZATION_DENIED
    elif isinstance(error, ValueError):
        code = NormalizedErrorCode.INTEGRITY_FAILURE
    else:
        code = NormalizedErrorCode.UNCLASSIFIED
    return NormalizedErrorTypeV1(code)


def _metric_status(status: ObservabilityStatus) -> str:
    return {
        ObservabilityStatus.SUCCEEDED: "SUCCESS",
        ObservabilityStatus.FAILED: "ERROR",
        ObservabilityStatus.CANCELLED: "CANCELLED",
        ObservabilityStatus.WAITING: "WAITING",
        ObservabilityStatus.REJECTED: "REJECTED",
        ObservabilityStatus.CONFLICT: "CONFLICT",
        ObservabilityStatus.STALE: "STALE",
        ObservabilityStatus.STARTED: "UNKNOWN",
        ObservabilityStatus.UNKNOWN: "UNKNOWN",
    }[status]


__all__ = [
    "AttemptObservationV1",
    "ObservabilityRuntime",
    "normalized_error_type",
]
