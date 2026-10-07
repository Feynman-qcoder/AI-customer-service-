"""Closed Tool/Retrieval telemetry DTOs protected by the shared boundary.

These records are optional diagnostics.  Raw arguments, results, exception
messages, candidate content, and metadata never become persistence values: the
only write DTOs in this module are reconstructed from ``ProtectedPayload``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import StrEnum
from string import ascii_letters, digits

from app.runtime.data_protection import (
    DataProtectionPolicy,
    DataProtectionPort,
    DataProtectionProfile,
    DataProtectionSchemaPolicy,
    FieldProtection,
    ProtectedPayload,
    SchemaPolicyError,
)
from app.runtime.sensitive_text import (
    SENSITIVE_TEXT_CLASSIFIER_VERSION_V1,
    SensitiveTextClassificationV1,
    SensitiveTextClassifierPort,
    SensitiveTextClassifierV1,
)

TOOL_AUDIT_SCHEMA_VERSION_V1 = "TOOL_AUDIT_V1"
RETRIEVAL_TRACE_SCHEMA_VERSION_V1 = "RETRIEVAL_TRACE_V1"

_IDENTIFIER_CHARACTERS = frozenset(ascii_letters + digits + "-_.:")
_TOOL_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "subject_user_id",
        "tool_name",
        "arguments",
        "result_summary",
        "success",
        "retry_count",
        "duration_ms",
    }
)
_RETRIEVAL_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "kind",
        "candidate_id",
        "source_type",
        "document_id",
        "chunk_id",
        "rule_id",
        "original_score",
        "fused_score",
        "rerank_score",
        "selected",
        "decision_reason",
        "metadata",
        "diagnostic_status",
        "error_type",
    }
)


class OptionalTelemetryContractError(ValueError):
    """Sanitized optional-telemetry contract rejection."""


class ToolNameV1(StrEnum):
    LIST_MY_ORDERS = "list_my_orders"
    GET_ORDER_DETAIL = "get_order_detail"
    GET_PRODUCT_INFORMATION = "get_product_information"
    SEARCH_KNOWLEDGE_BASE = "search_knowledge_base"
    CREATE_SUPPORT_TICKET = "create_support_ticket"
    REQUEST_ORDER_CANCELLATION = "request_order_cancellation"
    REQUEST_REFUND = "request_refund"


class RetrievalTraceKindV1(StrEnum):
    CANDIDATE = "CANDIDATE"
    DIAGNOSTIC = "DIAGNOSTIC"


class RetrievalSourceTypeV1(StrEnum):
    KEYWORD = "keyword"
    DENSE_VECTOR = "dense_vector"
    STRUCTURED_RULE = "structured_rule"
    CACHE = "cache"


class RetrievalDiagnosticStatusV1(StrEnum):
    OK = "OK"
    FAILED = "FAILED"
    DEGRADED = "DEGRADED"


class RetrievalErrorTypeV1(StrEnum):
    CACHE_READ_FAILED = "CACHE_READ_FAILED"
    CACHE_WRITE_FAILED = "CACHE_WRITE_FAILED"
    KEYWORD_RECALL_FAILED = "KEYWORD_RECALL_FAILED"
    DENSE_VECTOR_RECALL_FAILED = "DENSE_VECTOR_RECALL_FAILED"
    STRUCTURED_RULE_RECALL_FAILED = "STRUCTURED_RULE_RECALL_FAILED"
    UNKNOWN = "UNKNOWN"


def _require_identifier(value: object, *, field: str, maximum: int = 128) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > maximum
        or any(character not in _IDENTIFIER_CHARACTERS for character in value)
    ):
        raise OptionalTelemetryContractError(f"{field} is invalid")
    return value


def _require_content(value: object, *, field: str) -> str:
    if type(value) is not str or len(value.encode("utf-8")) > 16_384:
        raise OptionalTelemetryContractError(f"{field} is invalid")
    return value


def _require_score(value: object, *, nullable: bool) -> float | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise OptionalTelemetryContractError("retrieval score is invalid")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise OptionalTelemetryContractError("retrieval score is invalid")
    return numeric


@dataclass(frozen=True, slots=True)
class ToolAuditEventV1:
    run_id: str
    subject_user_id: int
    tool_name: ToolNameV1
    arguments_json: str
    result_summary: str
    success: bool
    retry_count: int
    duration_ms: int

    def __post_init__(self) -> None:
        _require_identifier(self.run_id, field="tool audit run", maximum=64)
        if type(self.subject_user_id) is not int or self.subject_user_id <= 0:
            raise OptionalTelemetryContractError("tool audit subject is invalid")
        if not isinstance(self.tool_name, ToolNameV1):
            raise OptionalTelemetryContractError("tool audit name is invalid")
        _require_content(self.arguments_json, field="tool audit arguments")
        _require_content(self.result_summary, field="tool audit result")
        if type(self.success) is not bool:
            raise OptionalTelemetryContractError("tool audit outcome is invalid")
        if type(self.retry_count) is not int or self.retry_count < 0:
            raise OptionalTelemetryContractError("tool audit retry count is invalid")
        if type(self.duration_ms) is not int or self.duration_ms < 0:
            raise OptionalTelemetryContractError("tool audit duration is invalid")


@dataclass(frozen=True, slots=True)
class RetrievalTraceEventV1:
    run_id: str
    kind: RetrievalTraceKindV1
    candidate_id: str
    source_type: RetrievalSourceTypeV1
    document_id: str | None
    chunk_id: str | None
    rule_id: str | None
    original_score: float
    fused_score: float | None
    rerank_score: float | None
    selected: bool
    decision_reason: str | None
    metadata_json: str
    diagnostic_status: RetrievalDiagnosticStatusV1 | None = None
    error_type: RetrievalErrorTypeV1 | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.run_id, field="retrieval run", maximum=64)
        if not isinstance(self.kind, RetrievalTraceKindV1):
            raise OptionalTelemetryContractError("retrieval trace kind is invalid")
        _require_identifier(self.candidate_id, field="retrieval candidate")
        if not isinstance(self.source_type, RetrievalSourceTypeV1):
            raise OptionalTelemetryContractError("retrieval source is invalid")
        for name, value in (
            ("retrieval document", self.document_id),
            ("retrieval chunk", self.chunk_id),
            ("retrieval rule", self.rule_id),
        ):
            if value is not None:
                _require_identifier(value, field=name, maximum=64)
        _require_score(self.original_score, nullable=False)
        _require_score(self.fused_score, nullable=True)
        _require_score(self.rerank_score, nullable=True)
        if type(self.selected) is not bool:
            raise OptionalTelemetryContractError("retrieval selection is invalid")
        if self.decision_reason is not None:
            _require_content(self.decision_reason, field="retrieval decision")
        _require_content(self.metadata_json, field="retrieval metadata")
        if self.diagnostic_status is not None and not isinstance(
            self.diagnostic_status,
            RetrievalDiagnosticStatusV1,
        ):
            raise OptionalTelemetryContractError("retrieval diagnostic status is invalid")
        if self.error_type is not None and not isinstance(
            self.error_type,
            RetrievalErrorTypeV1,
        ):
            raise OptionalTelemetryContractError("retrieval error type is invalid")
        if self.kind is RetrievalTraceKindV1.CANDIDATE:
            if self.diagnostic_status is not None or self.error_type is not None:
                raise OptionalTelemetryContractError("candidate diagnostic binding is invalid")
        elif self.diagnostic_status is None:
            raise OptionalTelemetryContractError("diagnostic status is required")


@dataclass(frozen=True, slots=True)
class ProtectedToolAuditWriteV1:
    run_id: str
    subject_user_id: int
    tool_name: str
    redacted_arguments_json: str
    result_summary: str | None
    success: bool
    retry_count: int
    duration_ms: int


@dataclass(frozen=True, slots=True)
class ProtectedRetrievalTraceWriteV1:
    run_id: str
    candidate_id: str
    source_type: str
    document_id: str | None
    chunk_id: str | None
    rule_id: str | None
    original_score: float
    fused_score: float | None
    rerank_score: float | None
    selected: bool
    decision_reason: str | None
    metadata_json: str


class _SensitiveExactPolicy:
    def __init__(self, classifier: SensitiveTextClassifierPort | None = None) -> None:
        self._classifier = classifier or SensitiveTextClassifierV1()

    def require_safe(self, value: str) -> None:
        try:
            classification = self._classifier.classify(value)
        except Exception:
            raise SchemaPolicyError("TELEMETRY_SENSITIVE_TEXT_REJECTED") from None
        if (
            type(classification) is not SensitiveTextClassificationV1
            or classification.classifier_version != SENSITIVE_TEXT_CLASSIFIER_VERSION_V1
            or classification.is_sensitive
            or classification.reason is not None
        ):
            raise SchemaPolicyError("TELEMETRY_SENSITIVE_TEXT_REJECTED")


class ToolAuditSchemaPolicy(DataProtectionSchemaPolicy):
    def __init__(self, classifier: SensitiveTextClassifierPort | None = None) -> None:
        self._sensitive = _SensitiveExactPolicy(classifier)

    def normalize(self, payload: object) -> object:
        if type(payload) is not ToolAuditEventV1:
            raise SchemaPolicyError("TOOL_AUDIT_SCHEMA_INVALID")
        self._sensitive.require_safe(payload.run_id)
        return {
            "schema_version": TOOL_AUDIT_SCHEMA_VERSION_V1,
            "run_id": payload.run_id,
            "subject_user_id": payload.subject_user_id,
            "tool_name": payload.tool_name.value,
            "arguments": payload.arguments_json,
            "result_summary": payload.result_summary,
            "success": payload.success,
            "retry_count": payload.retry_count,
            "duration_ms": payload.duration_ms,
        }

    def classify(self, path: tuple[str, ...], value: object) -> FieldProtection:
        del value
        if path in {("arguments",), ("result_summary",)}:
            return FieldProtection.CONTENT
        if len(path) == 1 and path[0] in _TOOL_FIELDS:
            return FieldProtection.EXACT
        return FieldProtection.REJECT

    def validate_protected(self, payload: object) -> None:
        if not isinstance(payload, dict) or set(payload) != _TOOL_FIELDS:
            raise SchemaPolicyError("TOOL_AUDIT_SCHEMA_INVALID")
        try:
            if payload["schema_version"] != TOOL_AUDIT_SCHEMA_VERSION_V1:
                raise OptionalTelemetryContractError("tool audit version is invalid")
            event = ToolAuditEventV1(
                run_id=payload["run_id"],
                subject_user_id=payload["subject_user_id"],
                tool_name=ToolNameV1(payload["tool_name"]),
                arguments_json=payload["arguments"],
                result_summary=payload["result_summary"],
                success=payload["success"],
                retry_count=payload["retry_count"],
                duration_ms=payload["duration_ms"],
            )
            self._sensitive.require_safe(event.run_id)
        except (KeyError, TypeError, ValueError, OptionalTelemetryContractError):
            raise SchemaPolicyError("TOOL_AUDIT_SCHEMA_INVALID") from None


class RetrievalTraceSchemaPolicy(DataProtectionSchemaPolicy):
    def __init__(self, classifier: SensitiveTextClassifierPort | None = None) -> None:
        self._sensitive = _SensitiveExactPolicy(classifier)

    def normalize(self, payload: object) -> object:
        if type(payload) is not RetrievalTraceEventV1:
            raise SchemaPolicyError("RETRIEVAL_TRACE_SCHEMA_INVALID")
        for value in (
            payload.run_id,
            payload.candidate_id,
            payload.document_id,
            payload.chunk_id,
            payload.rule_id,
        ):
            if value is not None:
                self._sensitive.require_safe(value)
        return {
            "schema_version": RETRIEVAL_TRACE_SCHEMA_VERSION_V1,
            "run_id": payload.run_id,
            "kind": payload.kind.value,
            "candidate_id": payload.candidate_id,
            "source_type": payload.source_type.value,
            "document_id": payload.document_id,
            "chunk_id": payload.chunk_id,
            "rule_id": payload.rule_id,
            "original_score": payload.original_score,
            "fused_score": payload.fused_score,
            "rerank_score": payload.rerank_score,
            "selected": payload.selected,
            "decision_reason": payload.decision_reason,
            "metadata": payload.metadata_json,
            "diagnostic_status": (
                payload.diagnostic_status.value
                if payload.diagnostic_status is not None
                else None
            ),
            "error_type": payload.error_type.value if payload.error_type is not None else None,
        }

    def classify(self, path: tuple[str, ...], value: object) -> FieldProtection:
        del value
        if path in {("decision_reason",), ("metadata",)}:
            return FieldProtection.CONTENT
        if len(path) == 1 and path[0] in _RETRIEVAL_FIELDS:
            return FieldProtection.EXACT
        return FieldProtection.REJECT

    def validate_protected(self, payload: object) -> None:
        if not isinstance(payload, dict) or set(payload) != _RETRIEVAL_FIELDS:
            raise SchemaPolicyError("RETRIEVAL_TRACE_SCHEMA_INVALID")
        try:
            if payload["schema_version"] != RETRIEVAL_TRACE_SCHEMA_VERSION_V1:
                raise OptionalTelemetryContractError("retrieval trace version is invalid")
            event = RetrievalTraceEventV1(
                run_id=payload["run_id"],
                kind=RetrievalTraceKindV1(payload["kind"]),
                candidate_id=payload["candidate_id"],
                source_type=RetrievalSourceTypeV1(payload["source_type"]),
                document_id=payload["document_id"],
                chunk_id=payload["chunk_id"],
                rule_id=payload["rule_id"],
                original_score=payload["original_score"],
                fused_score=payload["fused_score"],
                rerank_score=payload["rerank_score"],
                selected=payload["selected"],
                decision_reason=payload["decision_reason"],
                metadata_json=payload["metadata"],
                diagnostic_status=(
                    RetrievalDiagnosticStatusV1(payload["diagnostic_status"])
                    if payload["diagnostic_status"] is not None
                    else None
                ),
                error_type=(
                    RetrievalErrorTypeV1(payload["error_type"])
                    if payload["error_type"] is not None
                    else None
                ),
            )
            for value in (
                event.run_id,
                event.candidate_id,
                event.document_id,
                event.chunk_id,
                event.rule_id,
            ):
                if value is not None:
                    self._sensitive.require_safe(value)
        except (KeyError, TypeError, ValueError, OptionalTelemetryContractError):
            raise SchemaPolicyError("RETRIEVAL_TRACE_SCHEMA_INVALID") from None


def build_tool_audit_protection(
    classifier: SensitiveTextClassifierPort | None = None,
) -> DataProtectionPolicy:
    return DataProtectionPolicy(
        schema_policy=ToolAuditSchemaPolicy(classifier),
        profile=DataProtectionProfile.OBSERVABILITY,
    )


def build_retrieval_trace_protection(
    classifier: SensitiveTextClassifierPort | None = None,
) -> DataProtectionPolicy:
    return DataProtectionPolicy(
        schema_policy=RetrievalTraceSchemaPolicy(classifier),
        profile=DataProtectionProfile.OBSERVABILITY,
    )


def protect_tool_audit_event(
    event: ToolAuditEventV1,
    protection: DataProtectionPort,
) -> ProtectedToolAuditWriteV1:
    protected = protection.protect(
        event,
        profile=DataProtectionProfile.OBSERVABILITY,
    )
    payload = _decode_protected(protected, _TOOL_FIELDS, "TOOL_AUDIT_SCHEMA_INVALID")
    arguments = json.dumps(
        {
            "schema_version": TOOL_AUDIT_SCHEMA_VERSION_V1,
            "arguments": payload["arguments"],
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return ProtectedToolAuditWriteV1(
        run_id=_as_str(payload["run_id"]),
        subject_user_id=_as_int(payload["subject_user_id"]),
        tool_name=_as_str(payload["tool_name"]),
        redacted_arguments_json=arguments,
        result_summary=(
            _as_str(payload["result_summary"])
            if payload["result_summary"] != ""
            else None
        ),
        success=_as_bool(payload["success"]),
        retry_count=_as_int(payload["retry_count"]),
        duration_ms=_as_int(payload["duration_ms"]),
    )


def protect_retrieval_trace_event(
    event: RetrievalTraceEventV1,
    protection: DataProtectionPort,
) -> ProtectedRetrievalTraceWriteV1:
    protected = protection.protect(
        event,
        profile=DataProtectionProfile.OBSERVABILITY,
    )
    payload = _decode_protected(
        protected,
        _RETRIEVAL_FIELDS,
        "RETRIEVAL_TRACE_SCHEMA_INVALID",
    )
    return ProtectedRetrievalTraceWriteV1(
        run_id=_as_str(payload["run_id"]),
        candidate_id=_as_str(payload["candidate_id"]),
        source_type=_as_str(payload["source_type"]),
        document_id=_as_optional_str(payload["document_id"]),
        chunk_id=_as_optional_str(payload["chunk_id"]),
        rule_id=_as_optional_str(payload["rule_id"]),
        original_score=_as_float(payload["original_score"]),
        fused_score=_as_optional_float(payload["fused_score"]),
        rerank_score=_as_optional_float(payload["rerank_score"]),
        selected=_as_bool(payload["selected"]),
        decision_reason=(
            _as_optional_str(payload["decision_reason"])
            if payload["decision_reason"] is not None
            else None
        ),
        metadata_json=json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ),
    )


def _decode_protected(
    protected: ProtectedPayload,
    fields: frozenset[str],
    reason: str,
) -> dict[str, object]:
    try:
        value = json.loads(protected.canonical_bytes)
    except (TypeError, ValueError, UnicodeDecodeError):
        raise OptionalTelemetryContractError(reason) from None
    if type(value) is not dict or set(value) != fields:
        raise OptionalTelemetryContractError(reason)
    return value


def _as_str(value: object) -> str:
    if type(value) is not str:
        raise OptionalTelemetryContractError("protected telemetry is invalid")
    return value


def _as_optional_str(value: object) -> str | None:
    if value is None:
        return None
    return _as_str(value)


def _as_int(value: object) -> int:
    if type(value) is not int:
        raise OptionalTelemetryContractError("protected telemetry is invalid")
    return value


def _as_bool(value: object) -> bool:
    if type(value) is not bool:
        raise OptionalTelemetryContractError("protected telemetry is invalid")
    return value


def _as_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise OptionalTelemetryContractError("protected telemetry is invalid")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise OptionalTelemetryContractError("protected telemetry is invalid")
    return numeric


def _as_optional_float(value: object) -> float | None:
    if value is None:
        return None
    return _as_float(value)


__all__ = [
    "OptionalTelemetryContractError",
    "ProtectedRetrievalTraceWriteV1",
    "ProtectedToolAuditWriteV1",
    "RETRIEVAL_TRACE_SCHEMA_VERSION_V1",
    "RetrievalDiagnosticStatusV1",
    "RetrievalErrorTypeV1",
    "RetrievalSourceTypeV1",
    "RetrievalTraceEventV1",
    "RetrievalTraceKindV1",
    "TOOL_AUDIT_SCHEMA_VERSION_V1",
    "ToolAuditEventV1",
    "ToolNameV1",
    "build_retrieval_trace_protection",
    "build_tool_audit_protection",
    "protect_retrieval_trace_event",
    "protect_tool_audit_event",
]
