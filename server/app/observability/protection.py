"""Shared-protection adapters for optional observability sinks."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import RLock

from app.observability.contracts import (
    OBSERVABILITY_SCHEMA_VERSION_V1,
    AttemptIdentityV1,
    AttemptOutcome,
    AttemptResumeKind,
    AttemptScopeV1,
    FlushResultV1,
    NormalizedErrorCode,
    NormalizedErrorTypeV1,
    ObservabilityContractError,
    ObservabilityEventCode,
    ObservabilityEventV1,
    ObservabilityStatus,
    ProtectedTelemetrySinkPort,
    RunIdentityV1,
    ShutdownResultV1,
    SpanOperation,
    SpanScopeV1,
    TelemetryFailureCode,
)
from app.observability.metrics import (
    DEFAULT_METRIC_REGISTRY_V1,
    METRIC_SCHEMA_VERSION_V1,
    MetricKind,
    MetricLabelV1,
    MetricName,
    MetricPointV1,
    MetricRecordReason,
    MetricRecordResultV1,
    MetricUnit,
)
from app.observability.records import (
    OPERATION_RECORD_FIELDS,
    operation_record_payload,
    validate_operation_record_payload,
)
from app.runtime.data_protection import (
    DataProtectionPolicy,
    DataProtectionPort,
    DataProtectionProfile,
    DataProtectionSchemaPolicy,
    FieldProtection,
    SchemaPolicyError,
)

_OBSERVABILITY_FIELDS = frozenset(
    {
        "schema_version",
        "event_code",
        "status",
        "run",
        "attempt",
        "resume",
        "span_operation",
        "error_type",
        "record",
    }
)
_RUN_FIELDS = frozenset(
    {"conversation_id", "thread_id", "run_id", "subject_user_id"}
)
_ATTEMPT_FIELDS = frozenset({"attempt_id", "fence_token"})
_METRIC_FIELDS = frozenset(
    {"schema_version", "name", "kind", "unit", "value", "labels"}
)


@dataclass(frozen=True, slots=True)
class _ObservabilityEmissionV1:
    run: RunIdentityV1
    event: object
    attempt: AttemptIdentityV1 | None = None
    resume: AttemptResumeKind | None = None
    span_operation: SpanOperation | None = None
    record: object | None = None


def _run_payload(identity: RunIdentityV1) -> dict[str, object]:
    return {
        "conversation_id": identity.conversation_id,
        "thread_id": identity.thread_id,
        "run_id": identity.run_id,
        "subject_user_id": identity.subject_user_id,
    }


def _attempt_payload(identity: AttemptIdentityV1 | None) -> dict[str, object] | None:
    if identity is None:
        return None
    return {
        "attempt_id": identity.attempt_id,
        "fence_token": identity.fence_token,
    }


def _normalized_observability_payload(payload: object) -> dict[str, object]:
    if type(payload) is not _ObservabilityEmissionV1:
        raise SchemaPolicyError("OBSERVABILITY_SCHEMA_INVALID")
    if type(payload.run) is not RunIdentityV1 or type(payload.event) is not ObservabilityEventV1:
        raise SchemaPolicyError("OBSERVABILITY_SCHEMA_INVALID")
    if payload.attempt is not None:
        if type(payload.attempt) is not AttemptIdentityV1 or payload.attempt.run != payload.run:
            raise SchemaPolicyError("OBSERVABILITY_SCHEMA_INVALID")
    if payload.resume is not None and not isinstance(payload.resume, AttemptResumeKind):
        raise SchemaPolicyError("OBSERVABILITY_SCHEMA_INVALID")
    if payload.span_operation is not None and not isinstance(payload.span_operation, SpanOperation):
        raise SchemaPolicyError("OBSERVABILITY_SCHEMA_INVALID")
    event = payload.event
    return {
        "schema_version": OBSERVABILITY_SCHEMA_VERSION_V1,
        "event_code": event.code.value,
        "status": event.status.value,
        "run": _run_payload(payload.run),
        "attempt": _attempt_payload(payload.attempt),
        "resume": payload.resume.value if payload.resume is not None else None,
        "span_operation": (
            payload.span_operation.value if payload.span_operation is not None else None
        ),
        "error_type": (
            event.error_type.code.value if event.error_type is not None else None
        ),
        "record": (
            operation_record_payload(payload.record)
            if payload.record is not None
            else None
        ),
    }


def _validate_observability_mapping(payload: object) -> None:
    if not isinstance(payload, Mapping) or set(payload) != _OBSERVABILITY_FIELDS:
        raise SchemaPolicyError("OBSERVABILITY_SCHEMA_INVALID")
    try:
        if payload["schema_version"] != OBSERVABILITY_SCHEMA_VERSION_V1:
            raise ObservabilityContractError("observability version is invalid")
        run = payload["run"]
        if not isinstance(run, Mapping) or set(run) != _RUN_FIELDS:
            raise ObservabilityContractError("observability run is invalid")
        RunIdentityV1(
            conversation_id=run["conversation_id"],
            thread_id=run["thread_id"],
            run_id=run["run_id"],
            subject_user_id=run["subject_user_id"],
        )
        attempt = payload["attempt"]
        if attempt is not None:
            if not isinstance(attempt, Mapping) or set(attempt) != _ATTEMPT_FIELDS:
                raise ObservabilityContractError("observability attempt is invalid")
            if type(attempt["attempt_id"]) is not str:
                raise ObservabilityContractError("observability attempt is invalid")
            if type(attempt["fence_token"]) is not int or attempt["fence_token"] <= 0:
                raise ObservabilityContractError("observability attempt is invalid")
        ObservabilityEventV1(
            code=ObservabilityEventCode(payload["event_code"]),
            status=ObservabilityStatus(payload["status"]),
            error_type=(
                NormalizedErrorTypeV1(NormalizedErrorCode(payload["error_type"]))
                if payload["error_type"] is not None
                else None
            ),
        )
        if payload["resume"] is not None:
            AttemptResumeKind(payload["resume"])
        if payload["span_operation"] is not None:
            SpanOperation(payload["span_operation"])
        if payload["record"] is not None:
            validate_operation_record_payload(payload["record"])
    except (KeyError, TypeError, ValueError, ObservabilityContractError):
        raise SchemaPolicyError("OBSERVABILITY_SCHEMA_INVALID") from None


class ObservabilitySchemaPolicy(DataProtectionSchemaPolicy):
    """Closed adapter for lifecycle and diagnostic event records."""

    def normalize(self, payload: object) -> object:
        return _normalized_observability_payload(payload)

    def classify(self, path: tuple[str, ...], value: object) -> FieldProtection:
        del value
        if path in {("run",), ("attempt",), ("record",)}:
            return FieldProtection.STRUCTURE
        if len(path) == 1 and path[0] in _OBSERVABILITY_FIELDS:
            return FieldProtection.EXACT
        if len(path) == 2 and path[0] == "run" and path[1] in _RUN_FIELDS:
            return FieldProtection.EXACT
        if len(path) == 2 and path[0] == "attempt" and path[1] in _ATTEMPT_FIELDS:
            return FieldProtection.EXACT
        if len(path) == 2 and path[0] == "record" and path[1] in OPERATION_RECORD_FIELDS:
            return FieldProtection.EXACT
        return FieldProtection.REJECT

    def validate_protected(self, payload: object) -> None:
        _validate_observability_mapping(payload)


class MetricSchemaPolicy(DataProtectionSchemaPolicy):
    """Closed adapter for a validated metric point."""

    def __init__(self) -> None:
        self._registry = DEFAULT_METRIC_REGISTRY_V1

    def normalize(self, payload: object) -> object:
        reason = self._registry.rejection_reason(payload)
        if reason is not None:
            raise SchemaPolicyError("METRIC_SCHEMA_INVALID")
        assert isinstance(payload, MetricPointV1)
        return {
            "schema_version": self._registry.schema_version,
            "name": str(payload.name),
            "kind": payload.kind.value,
            "unit": payload.unit.value,
            "value": payload.value,
            "labels": {label.key: label.value for label in payload.labels},
        }

    def classify(self, path: tuple[str, ...], value: object) -> FieldProtection:
        del value
        if path == ("labels",):
            return FieldProtection.STRUCTURE
        if len(path) == 1 and path[0] in _METRIC_FIELDS:
            return FieldProtection.EXACT
        if len(path) == 2 and path[0] == "labels":
            return FieldProtection.EXACT
        return FieldProtection.REJECT

    def validate_protected(self, payload: object) -> None:
        if not isinstance(payload, Mapping) or set(payload) != _METRIC_FIELDS:
            raise SchemaPolicyError("METRIC_SCHEMA_INVALID")
        labels = payload["labels"]
        if not isinstance(labels, Mapping):
            raise SchemaPolicyError("METRIC_SCHEMA_INVALID")
        try:
            point = MetricPointV1(
                name=MetricName(payload["name"]),
                kind=MetricKind(payload["kind"]),
                unit=MetricUnit(payload["unit"]),
                value=payload["value"],
                labels=tuple(
                    MetricLabelV1(key=key, value=value)
                    for key, value in labels.items()
                ),
            )
        except (TypeError, ValueError):
            raise SchemaPolicyError("METRIC_SCHEMA_INVALID") from None
        if payload["schema_version"] != METRIC_SCHEMA_VERSION_V1:
            raise SchemaPolicyError("METRIC_SCHEMA_INVALID")
        if self._registry.rejection_reason(point) is not None:
            raise SchemaPolicyError("METRIC_SCHEMA_INVALID")


def build_observability_protection() -> DataProtectionPolicy:
    return DataProtectionPolicy(
        schema_policy=ObservabilitySchemaPolicy(),
        profile=DataProtectionProfile.OBSERVABILITY,
    )


def build_metric_protection() -> DataProtectionPolicy:
    return DataProtectionPolicy(
        schema_policy=MetricSchemaPolicy(),
        profile=DataProtectionProfile.OBSERVABILITY,
    )


def _require_timeout(timeout_seconds: float) -> None:
    if type(timeout_seconds) not in {int, float} or timeout_seconds <= 0:
        raise ObservabilityContractError("telemetry timeout is invalid")


class ProtectedObservability:
    """Fail-open facade whose physical sink sees protected bytes only."""

    def __init__(
        self,
        *,
        protection: DataProtectionPort,
        sink: ProtectedTelemetrySinkPort,
    ) -> None:
        self._protection = protection
        self._sink = sink
        self._registry_lock = RLock()
        self._runs: dict[str, RunIdentityV1] = {}
        self._attempts: dict[int, _AttemptRuntime] = {}
        self._attempt_identities: dict[str, AttemptIdentityV1] = {}

    def register_run(self, identity: RunIdentityV1) -> None:
        if type(identity) is not RunIdentityV1:
            raise ObservabilityContractError("run identity is invalid")
        with self._registry_lock:
            registered = self._runs.get(identity.run_id)
            if registered is not None:
                if registered != identity:
                    raise ObservabilityContractError(
                        "run identity conflicts with its registered scope"
                    )
                return
            self._runs[identity.run_id] = identity
        self._emit(
            _ObservabilityEmissionV1(
                run=identity,
                event=ObservabilityEventV1(
                    code=ObservabilityEventCode.RUN_REGISTERED,
                    status=ObservabilityStatus.SUCCEEDED,
                ),
            )
        )

    def start_attempt(
        self,
        identity: AttemptIdentityV1,
        *,
        resume: AttemptResumeKind,
    ) -> AttemptScopeV1:
        scope = AttemptScopeV1(identity=identity, resume=resume)
        state = _AttemptRuntime(scope=scope)
        with self._registry_lock:
            if identity.attempt_id in self._attempt_identities:
                raise ObservabilityContractError("attempt identity is already registered")
            self._attempt_identities[identity.attempt_id] = identity
            self._attempts[id(scope)] = state
        with state.lock:
            self._emit_for_scope(
                scope,
                ObservabilityEventV1(
                    code=ObservabilityEventCode.ATTEMPT_STARTED,
                    status=ObservabilityStatus.STARTED,
                ),
            )
        return scope

    def end_attempt(
        self,
        scope: AttemptScopeV1,
        *,
        outcome: AttemptOutcome,
        error_type: NormalizedErrorTypeV1 | None = None,
    ) -> None:
        if not isinstance(outcome, AttemptOutcome):
            raise ObservabilityContractError("attempt outcome is invalid")
        if error_type is not None and type(error_type) is not NormalizedErrorTypeV1:
            raise ObservabilityContractError("attempt error type is invalid")
        state = self._attempt_state(scope)
        with state.lock:
            state = self._require_live_attempt(scope)
            if state.open_spans:
                raise ObservabilityContractError("attempt has open spans")
            state.ended = True
            self._emit_for_scope(
                scope,
                ObservabilityEventV1(
                    code=ObservabilityEventCode.ATTEMPT_ENDED,
                    status=ObservabilityStatus(outcome.value),
                    error_type=error_type,
                ),
            )

    @contextmanager
    def span(
        self,
        scope: AttemptScopeV1,
        *,
        operation: SpanOperation,
    ) -> Iterator[SpanScopeV1]:
        span_scope = SpanScopeV1(attempt=scope, operation=operation)
        state = self._attempt_state(scope)
        with state.lock:
            state = self._require_live_attempt(scope)
            state.open_spans.add(id(span_scope))
            self._emit_for_scope(
                span_scope,
                ObservabilityEventV1(
                    code=ObservabilityEventCode.SPAN_STARTED,
                    status=ObservabilityStatus.STARTED,
                ),
            )
        try:
            yield span_scope
        except BaseException as error:
            if isinstance(error, asyncio.CancelledError):
                status = ObservabilityStatus.CANCELLED
                error_code = NormalizedErrorCode.CANCELLED
            elif isinstance(error, TimeoutError):
                status = ObservabilityStatus.FAILED
                error_code = NormalizedErrorCode.TIMEOUT
            elif isinstance(error, ValueError):
                status = ObservabilityStatus.FAILED
                error_code = NormalizedErrorCode.INTEGRITY_FAILURE
            else:
                status = ObservabilityStatus.FAILED
                error_code = NormalizedErrorCode.UNCLASSIFIED
            with state.lock:
                self._require_open_span(state, span_scope)
                self._emit_for_scope(
                    span_scope,
                    ObservabilityEventV1(
                        code=ObservabilityEventCode.SPAN_ENDED,
                        status=status,
                        error_type=NormalizedErrorTypeV1(error_code),
                    ),
                )
                state.open_spans.remove(id(span_scope))
            raise
        else:
            with state.lock:
                self._require_open_span(state, span_scope)
                self._emit_for_scope(
                    span_scope,
                    ObservabilityEventV1(
                        code=ObservabilityEventCode.SPAN_ENDED,
                        status=ObservabilityStatus.SUCCEEDED,
                    ),
                )
                state.open_spans.remove(id(span_scope))

    def record_event(
        self,
        scope: AttemptScopeV1 | SpanScopeV1,
        event: ObservabilityEventV1,
    ) -> None:
        if type(event) is not ObservabilityEventV1:
            raise ObservabilityContractError("observability event is invalid")
        state = self._state_for_event_scope(scope)
        with state.lock:
            self._require_event_scope(scope, state)
            self._emit_for_scope(scope, event)

    def record_error(
        self,
        scope: AttemptScopeV1 | SpanScopeV1,
        error_type: NormalizedErrorTypeV1,
    ) -> None:
        if type(error_type) is not NormalizedErrorTypeV1:
            raise ObservabilityContractError("normalized error type is invalid")
        state = self._state_for_event_scope(scope)
        with state.lock:
            self._require_event_scope(scope, state)
            self._emit_for_scope(
                scope,
                ObservabilityEventV1(
                    code=ObservabilityEventCode.ERROR_RECORDED,
                    status=ObservabilityStatus.FAILED,
                    error_type=error_type,
                ),
            )

    def record_operation(
        self,
        scope: AttemptScopeV1 | SpanScopeV1,
        record: object,
    ) -> None:
        record_payload = operation_record_payload(record)
        state = self._state_for_event_scope(scope)
        with state.lock:
            self._require_event_scope(scope, state)
            status_value = record_payload["status"]
            if type(status_value) is not str:
                raise ObservabilityContractError("operation status is invalid")
            status = ObservabilityStatus(status_value)
            error_code = record_payload["error_type"]
            error_type = (
                NormalizedErrorTypeV1(NormalizedErrorCode(error_code))
                if type(error_code) is str
                else None
            )
            self._emit_for_scope(
                scope,
                ObservabilityEventV1(
                    code=ObservabilityEventCode.OPERATION_RECORDED,
                    status=status,
                    error_type=error_type,
                ),
                record=record,
            )

    def flush(self, *, timeout_seconds: float) -> FlushResultV1:
        _require_timeout(timeout_seconds)
        try:
            completed = self._sink.flush(timeout_seconds=float(timeout_seconds))
        except Exception:
            completed = False
        return FlushResultV1(
            completed=completed is True,
            failure_code=(
                None if completed is True else TelemetryFailureCode.SINK_UNAVAILABLE
            ),
        )

    def shutdown(self, *, timeout_seconds: float) -> ShutdownResultV1:
        _require_timeout(timeout_seconds)
        try:
            completed = self._sink.shutdown(timeout_seconds=float(timeout_seconds))
        except Exception:
            completed = False
        return ShutdownResultV1(
            completed=completed is True,
            failure_code=(
                None if completed is True else TelemetryFailureCode.SINK_UNAVAILABLE
            ),
        )

    def _emit_for_scope(
        self,
        scope: AttemptScopeV1 | SpanScopeV1,
        event: object,
        *,
        record: object | None = None,
    ) -> None:
        if isinstance(scope, SpanScopeV1):
            attempt_scope = scope.attempt
            span_operation: SpanOperation | None = scope.operation
        elif isinstance(scope, AttemptScopeV1):
            attempt_scope = scope
            span_operation = None
        else:
            raise ObservabilityContractError("observability scope is invalid")
        self._emit(
            _ObservabilityEmissionV1(
                run=attempt_scope.identity.run,
                attempt=attempt_scope.identity,
                resume=attempt_scope.resume,
                span_operation=span_operation,
                event=event,
                record=record,
            )
        )

    def _attempt_state(self, scope: AttemptScopeV1) -> _AttemptRuntime:
        if type(scope) is not AttemptScopeV1:
            raise ObservabilityContractError("observability scope is invalid")
        with self._registry_lock:
            state = self._attempts.get(id(scope))
        if state is None or state.scope is not scope:
            raise ObservabilityContractError("attempt scope is not provider-owned")
        return state

    def _require_live_attempt(self, scope: AttemptScopeV1) -> _AttemptRuntime:
        state = self._attempt_state(scope)
        if state.ended:
            raise ObservabilityContractError("attempt has ended")
        return state

    def _state_for_event_scope(
        self,
        scope: AttemptScopeV1 | SpanScopeV1,
    ) -> _AttemptRuntime:
        if type(scope) is SpanScopeV1:
            return self._attempt_state(scope.attempt)
        if type(scope) is AttemptScopeV1:
            return self._attempt_state(scope)
        raise ObservabilityContractError("observability scope is invalid")

    def _require_event_scope(
        self,
        scope: AttemptScopeV1 | SpanScopeV1,
        state: _AttemptRuntime,
    ) -> None:
        self._require_live_attempt(state.scope)
        if type(scope) is SpanScopeV1 and id(scope) not in state.open_spans:
            raise ObservabilityContractError("span is not open")

    def _require_open_span(
        self,
        state: _AttemptRuntime,
        scope: SpanScopeV1,
    ) -> None:
        self._require_live_attempt(scope.attempt)
        if id(scope) not in state.open_spans:
            raise ObservabilityContractError("span lifecycle is invalid")

    def _emit(self, emission: _ObservabilityEmissionV1) -> None:
        try:
            protected = self._protection.protect(
                emission,
                profile=DataProtectionProfile.OBSERVABILITY,
            )
        except Exception:
            return
        try:
            self._sink.emit(protected)
        except Exception:
            return


class ProtectedMetricsRecorder:
    """Validate, protect and best-effort emit one closed metric point."""

    def __init__(
        self,
        *,
        protection: DataProtectionPort,
        sink: ProtectedTelemetrySinkPort,
    ) -> None:
        self._registry = DEFAULT_METRIC_REGISTRY_V1
        self._protection = protection
        self._sink = sink

    def record(self, point: MetricPointV1) -> MetricRecordResultV1:
        reason = self._registry.rejection_reason(point)
        if reason is not None:
            return MetricRecordResultV1(accepted=False, reason=reason)
        try:
            protected = self._protection.protect(
                point,
                profile=DataProtectionProfile.OBSERVABILITY,
            )
        except Exception:
            return MetricRecordResultV1(
                accepted=False,
                reason=MetricRecordReason.PROTECTION_FAILED,
            )
        try:
            self._sink.emit(protected)
        except Exception:
            return MetricRecordResultV1(
                accepted=False,
                reason=MetricRecordReason.SINK_FAILED,
            )
        return MetricRecordResultV1(accepted=True)


@dataclass(slots=True)
class _AttemptRuntime:
    scope: AttemptScopeV1
    open_spans: set[int] = field(default_factory=set)
    ended: bool = False
    lock: RLock = field(default_factory=RLock, repr=False)


__all__ = [
    "MetricSchemaPolicy",
    "ObservabilitySchemaPolicy",
    "ProtectedMetricsRecorder",
    "ProtectedObservability",
    "build_metric_protection",
    "build_observability_protection",
]
