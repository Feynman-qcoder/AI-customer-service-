"""Provider-neutral observability contracts and immutable value objects."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import StrEnum
from string import ascii_letters, digits
from typing import TYPE_CHECKING, Protocol

from app.agent.thread_identity import ThreadIdentityError, derive_thread_id
from app.runtime.data_protection import ProtectedPayload
from app.runtime.sensitive_text import (
    SENSITIVE_TEXT_CLASSIFIER_VERSION_V1,
    SensitiveTextClassificationV1,
    SensitiveTextClassifierV1,
)

OBSERVABILITY_SCHEMA_VERSION_V1 = "OBSERVABILITY_EVENT_V1"

_IDENTIFIER_CHARACTERS = frozenset(ascii_letters + digits + "-_.:")
_SENSITIVE_TEXT_CLASSIFIER = SensitiveTextClassifierV1()

if TYPE_CHECKING:
    from app.observability.records import OperationRecordV1


class ObservabilityContractError(ValueError):
    """Sanitized observability contract rejection."""


def _require_identifier(value: object, *, field: str) -> str:
    if type(value) is not str or not value or len(value) > 128:
        raise ObservabilityContractError(f"{field} is invalid")
    try:
        classification = _SENSITIVE_TEXT_CLASSIFIER.classify(value)
    except Exception:
        raise ObservabilityContractError(f"{field} is invalid") from None
    if (
        type(classification) is not SensitiveTextClassificationV1
        or classification.classifier_version != SENSITIVE_TEXT_CLASSIFIER_VERSION_V1
        or classification.is_sensitive
        or classification.reason is not None
        or any(character not in _IDENTIFIER_CHARACTERS for character in value)
    ):
        raise ObservabilityContractError(f"{field} is invalid")
    return value


class AttemptResumeKind(StrEnum):
    FRESH = "FRESH"
    RESUME = "RESUME"


class AttemptPhase(StrEnum):
    ACTIVE = "ACTIVE"
    WAITING = "WAITING"


class AttemptOutcome(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    WAITING = "WAITING"
    REJECTED = "REJECTED"


class ObservabilityStatus(StrEnum):
    STARTED = "STARTED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    WAITING = "WAITING"
    REJECTED = "REJECTED"
    CONFLICT = "CONFLICT"
    STALE = "STALE"
    UNKNOWN = "UNKNOWN"


class SpanOperation(StrEnum):
    NODE = "NODE"
    LLM = "LLM"
    RETRIEVAL = "RETRIEVAL"
    TOOL = "TOOL"
    MEMORY = "MEMORY"
    HITL = "HITL"
    REVALIDATION = "REVALIDATION"
    ACTION = "ACTION"


class TelemetryFailureCode(StrEnum):
    SINK_UNAVAILABLE = "SINK_UNAVAILABLE"


class ObservabilityEventCode(StrEnum):
    """Closed event catalog; emitters may only use registered codes."""

    RUN_REGISTERED = "RUN_REGISTERED"
    ATTEMPT_STARTED = "ATTEMPT_STARTED"
    ATTEMPT_ENDED = "ATTEMPT_ENDED"
    SPAN_STARTED = "SPAN_STARTED"
    SPAN_ENDED = "SPAN_ENDED"
    ERROR_RECORDED = "ERROR_RECORDED"
    POLICY_CHECK = "POLICY_CHECK"
    LEASE_RENEWED = "LEASE_RENEWED"
    AUTHORITY_REVALIDATED = "AUTHORITY_REVALIDATED"
    OPERATION_RECORDED = "OPERATION_RECORDED"


class NormalizedErrorCode(StrEnum):
    """Frozen low-cardinality error catalog with no raw exception text."""

    UNCLASSIFIED = "UNCLASSIFIED"
    LEASE_CONFLICT = "LEASE_CONFLICT"
    TIMEOUT = "TIMEOUT"
    UNAVAILABLE = "UNAVAILABLE"
    AUTHORIZATION_DENIED = "AUTHORIZATION_DENIED"
    INTEGRITY_FAILURE = "INTEGRITY_FAILURE"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class RunIdentityV1:
    conversation_id: int
    thread_id: str
    run_id: str
    subject_user_id: int

    def __post_init__(self) -> None:
        if type(self.conversation_id) is not int or self.conversation_id <= 0:
            raise ObservabilityContractError("conversation identity is invalid")
        if type(self.subject_user_id) is not int or self.subject_user_id <= 0:
            raise ObservabilityContractError("subject identity is invalid")
        _require_identifier(self.thread_id, field="thread identity")
        _require_identifier(self.run_id, field="run identity")
        try:
            canonical_thread_id = derive_thread_id(self.conversation_id)
        except ThreadIdentityError:
            raise ObservabilityContractError("conversation identity is invalid") from None
        if self.thread_id != canonical_thread_id:
            raise ObservabilityContractError("thread identity is invalid")


@dataclass(frozen=True, slots=True)
class AttemptIdentityV1:
    run: RunIdentityV1
    attempt_id: str
    fence_token: int

    def __post_init__(self) -> None:
        if type(self.run) is not RunIdentityV1:
            raise ObservabilityContractError("attempt run identity is invalid")
        _require_identifier(self.attempt_id, field="attempt identity")
        if type(self.fence_token) is not int or self.fence_token <= 0:
            raise ObservabilityContractError("attempt fence is invalid")


@dataclass(frozen=True, slots=True)
class AttemptScopeV1:
    identity: AttemptIdentityV1
    resume: AttemptResumeKind

    def __post_init__(self) -> None:
        if type(self.identity) is not AttemptIdentityV1:
            raise ObservabilityContractError("attempt scope identity is invalid")
        if not isinstance(self.resume, AttemptResumeKind):
            raise ObservabilityContractError("attempt resume kind is invalid")


@dataclass(frozen=True, slots=True)
class SpanScopeV1:
    attempt: AttemptScopeV1
    operation: SpanOperation

    def __post_init__(self) -> None:
        if type(self.attempt) is not AttemptScopeV1:
            raise ObservabilityContractError("span attempt scope is invalid")
        if not isinstance(self.operation, SpanOperation):
            raise ObservabilityContractError("span operation is invalid")


@dataclass(frozen=True, slots=True)
class NormalizedErrorTypeV1:
    code: NormalizedErrorCode

    def __post_init__(self) -> None:
        if not isinstance(self.code, NormalizedErrorCode):
            raise ObservabilityContractError("normalized error type is invalid")


@dataclass(frozen=True, slots=True)
class ObservabilityEventV1:
    code: ObservabilityEventCode
    status: ObservabilityStatus
    error_type: NormalizedErrorTypeV1 | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.code, ObservabilityEventCode):
            raise ObservabilityContractError("event code is invalid")
        if not isinstance(self.status, ObservabilityStatus):
            raise ObservabilityContractError("event status is invalid")
        if self.error_type is not None and type(self.error_type) is not NormalizedErrorTypeV1:
            raise ObservabilityContractError("event error type is invalid")


@dataclass(frozen=True, slots=True)
class FlushResultV1:
    completed: bool
    failure_code: TelemetryFailureCode | None = None

    def __post_init__(self) -> None:
        if type(self.completed) is not bool:
            raise ObservabilityContractError("flush result is invalid")
        if self.completed == (self.failure_code is not None):
            raise ObservabilityContractError("flush result is inconsistent")


@dataclass(frozen=True, slots=True)
class ShutdownResultV1:
    completed: bool
    failure_code: TelemetryFailureCode | None = None

    def __post_init__(self) -> None:
        if type(self.completed) is not bool:
            raise ObservabilityContractError("shutdown result is invalid")
        if self.completed == (self.failure_code is not None):
            raise ObservabilityContractError("shutdown result is inconsistent")


class ProtectedTelemetrySinkPort(Protocol):
    """A physical sink that only receives shared-protection output."""

    def emit(self, payload: ProtectedPayload) -> None: ...

    def flush(self, *, timeout_seconds: float) -> bool: ...

    def shutdown(self, *, timeout_seconds: float) -> bool: ...


class ObservabilityPort(Protocol):
    def register_run(self, identity: RunIdentityV1) -> None: ...

    def start_attempt(
        self,
        identity: AttemptIdentityV1,
        *,
        resume: AttemptResumeKind,
    ) -> AttemptScopeV1: ...

    def end_attempt(
        self,
        scope: AttemptScopeV1,
        *,
        outcome: AttemptOutcome,
        error_type: NormalizedErrorTypeV1 | None = None,
    ) -> None: ...

    def span(
        self,
        scope: AttemptScopeV1,
        *,
        operation: SpanOperation,
    ) -> AbstractContextManager[SpanScopeV1]: ...

    def record_event(
        self,
        scope: AttemptScopeV1 | SpanScopeV1,
        event: ObservabilityEventV1,
    ) -> None: ...

    def record_error(
        self,
        scope: AttemptScopeV1 | SpanScopeV1,
        error_type: NormalizedErrorTypeV1,
    ) -> None: ...

    def record_operation(
        self,
        scope: AttemptScopeV1 | SpanScopeV1,
        record: OperationRecordV1,
    ) -> None: ...

    def flush(self, *, timeout_seconds: float) -> FlushResultV1: ...

    def shutdown(self, *, timeout_seconds: float) -> ShutdownResultV1: ...


__all__ = [
    "OBSERVABILITY_SCHEMA_VERSION_V1",
    "AttemptIdentityV1",
    "AttemptOutcome",
    "AttemptPhase",
    "AttemptResumeKind",
    "AttemptScopeV1",
    "FlushResultV1",
    "NormalizedErrorTypeV1",
    "NormalizedErrorCode",
    "ObservabilityContractError",
    "ObservabilityEventV1",
    "ObservabilityEventCode",
    "ObservabilityPort",
    "ObservabilityStatus",
    "ProtectedTelemetrySinkPort",
    "RunIdentityV1",
    "ShutdownResultV1",
    "SpanOperation",
    "SpanScopeV1",
    "TelemetryFailureCode",
]
