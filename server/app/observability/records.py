"""Closed, provider-neutral operation records for local V1 observability."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from app.observability.contracts import (
    NormalizedErrorTypeV1,
    ObservabilityContractError,
    ObservabilityStatus,
)

OPERATION_RECORD_SCHEMA_VERSION_V1 = "OBSERVABILITY_OPERATION_V1"


class OperationRecordKind(StrEnum):
    NODE = "NODE"
    LLM = "LLM"
    RETRIEVAL = "RETRIEVAL"
    TOOL = "TOOL"
    MEMORY = "MEMORY"
    HITL = "HITL"
    REVALIDATION = "REVALIDATION"
    ACTION = "ACTION"


class OperationCode(StrEnum):
    NODE_WORKING_MEMORY_LOAD = "NODE_WORKING_MEMORY_LOAD"
    NODE_CONTEXT_RESOLVER = "NODE_CONTEXT_RESOLVER"
    NODE_INPUT_GUARD = "NODE_INPUT_GUARD"
    NODE_BLOCKED_RESPONSE = "NODE_BLOCKED_RESPONSE"
    NODE_PLANNER = "NODE_PLANNER"
    NODE_TOOL_OR_RETRIEVAL = "NODE_TOOL_OR_RETRIEVAL"
    NODE_CUSTOMER_CONFIRMATION_GATE = "NODE_CUSTOMER_CONFIRMATION_GATE"
    NODE_ACTION_PREPARE = "NODE_ACTION_PREPARE"
    NODE_ADMIN_APPROVAL_GATE = "NODE_ADMIN_APPROVAL_GATE"
    NODE_BUSINESS_EXECUTE = "NODE_BUSINESS_EXECUTE"
    NODE_WORKING_MEMORY_UPDATE = "NODE_WORKING_MEMORY_UPDATE"
    NODE_ANSWER_POLISHER = "NODE_ANSWER_POLISHER"
    NODE_RESPONSE_GUARDRAIL = "NODE_RESPONSE_GUARDRAIL"
    NODE_AUDIT_FINALIZE = "NODE_AUDIT_FINALIZE"
    LLM_PLANNER = "LLM_PLANNER"
    LLM_ANSWER = "LLM_ANSWER"
    LLM_SUMMARY = "LLM_SUMMARY"
    RETRIEVAL_SEARCH = "RETRIEVAL_SEARCH"
    TOOL_CALL = "TOOL_CALL"
    MEMORY_LOAD = "MEMORY_LOAD"
    MEMORY_UPDATE = "MEMORY_UPDATE"
    MEMORY_SUMMARY = "MEMORY_SUMMARY"
    MEMORY_RECENT_FALLBACK = "MEMORY_RECENT_FALLBACK"
    HITL_CUSTOMER_INTERRUPT = "HITL_CUSTOMER_INTERRUPT"
    HITL_CUSTOMER_RESUME = "HITL_CUSTOMER_RESUME"
    HITL_ADMIN_INTERRUPT = "HITL_ADMIN_INTERRUPT"
    HITL_ADMIN_DECISION = "HITL_ADMIN_DECISION"
    HITL_ADMIN_RESUME = "HITL_ADMIN_RESUME"
    HITL_RECONCILER_RESUME = "HITL_RECONCILER_RESUME"
    REVALIDATION_OWNER = "REVALIDATION_OWNER"
    REVALIDATION_ORDER_STATUS = "REVALIDATION_ORDER_STATUS"
    REVALIDATION_POLICY = "REVALIDATION_POLICY"
    REVALIDATION_EXECUTION_RESULT = "REVALIDATION_EXECUTION_RESULT"
    ACTION_PREPARE = "ACTION_PREPARE"
    ACTION_EXECUTION_CLAIM = "ACTION_EXECUTION_CLAIM"
    ACTION_BUSINESS_EXECUTE = "ACTION_BUSINESS_EXECUTE"


class ModelFamily(StrEnum):
    OPENAI_COMPATIBLE = "OPENAI_COMPATIBLE"
    MOCK = "MOCK"
    UNKNOWN = "UNKNOWN"


class RetrievalChannel(StrEnum):
    CACHE = "CACHE"
    KEYWORD = "KEYWORD"
    VECTOR = "VECTOR"
    RULE = "RULE"
    FUSED = "FUSED"
    UNKNOWN = "UNKNOWN"


class ToolCode(StrEnum):
    LIST_MY_ORDERS = "list_my_orders"
    GET_ORDER_DETAIL = "get_order_detail"
    GET_PRODUCT_INFORMATION = "get_product_information"
    SEARCH_KNOWLEDGE_BASE = "search_knowledge_base"
    CREATE_SUPPORT_TICKET = "create_support_ticket"
    REQUEST_ORDER_CANCELLATION = "request_order_cancellation"
    REQUEST_REFUND = "request_refund"


class ActionCode(StrEnum):
    CANCEL_ORDER = "CANCEL_ORDER"
    REQUEST_REFUND = "REQUEST_REFUND"
    UNKNOWN = "UNKNOWN"


class RiskCode(StrEnum):
    READ_ONLY = "READ_ONLY"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


def _require_status(value: object) -> None:
    if not isinstance(value, ObservabilityStatus):
        raise ObservabilityContractError("operation status is invalid")


def _require_duration(value: object) -> None:
    if type(value) is not int or value < 0:
        raise ObservabilityContractError("operation duration is invalid")


def _require_retry(value: object) -> None:
    if type(value) is not int or value < 0:
        raise ObservabilityContractError("operation retry count is invalid")


def _require_error(value: object) -> None:
    if value is not None and type(value) is not NormalizedErrorTypeV1:
        raise ObservabilityContractError("operation error type is invalid")


@dataclass(frozen=True, slots=True)
class NodeOperationRecordV1:
    operation: OperationCode
    status: ObservabilityStatus
    duration_ms: int
    error_type: NormalizedErrorTypeV1 | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.operation, OperationCode) or not self.operation.value.startswith("NODE_"):
            raise ObservabilityContractError("node operation is invalid")
        _require_status(self.status)
        _require_duration(self.duration_ms)
        _require_error(self.error_type)


@dataclass(frozen=True, slots=True)
class LLMOperationRecordV1:
    operation: OperationCode
    status: ObservabilityStatus
    duration_ms: int
    retry_count: int
    model_family: ModelFamily
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error_type: NormalizedErrorTypeV1 | None = None

    def __post_init__(self) -> None:
        if self.operation not in {
            OperationCode.LLM_PLANNER,
            OperationCode.LLM_ANSWER,
            OperationCode.LLM_SUMMARY,
        }:
            raise ObservabilityContractError("LLM operation is invalid")
        _require_status(self.status)
        _require_duration(self.duration_ms)
        _require_retry(self.retry_count)
        if not isinstance(self.model_family, ModelFamily):
            raise ObservabilityContractError("LLM model family is invalid")
        for value in (self.prompt_tokens, self.completion_tokens):
            if value is not None and (type(value) is not int or value < 0):
                raise ObservabilityContractError("LLM token usage is invalid")
        _require_error(self.error_type)


@dataclass(frozen=True, slots=True)
class RetrievalOperationRecordV1:
    status: ObservabilityStatus
    duration_ms: int
    retry_count: int
    channel: RetrievalChannel
    error_type: NormalizedErrorTypeV1 | None = None
    operation: OperationCode = OperationCode.RETRIEVAL_SEARCH

    def __post_init__(self) -> None:
        if self.operation is not OperationCode.RETRIEVAL_SEARCH:
            raise ObservabilityContractError("retrieval operation is invalid")
        _require_status(self.status)
        _require_duration(self.duration_ms)
        _require_retry(self.retry_count)
        if not isinstance(self.channel, RetrievalChannel):
            raise ObservabilityContractError("retrieval channel is invalid")
        _require_error(self.error_type)


@dataclass(frozen=True, slots=True)
class ToolOperationRecordV1:
    status: ObservabilityStatus
    duration_ms: int
    retry_count: int
    tool_name: ToolCode
    risk_level: RiskCode
    error_type: NormalizedErrorTypeV1 | None = None
    operation: OperationCode = OperationCode.TOOL_CALL

    def __post_init__(self) -> None:
        if self.operation is not OperationCode.TOOL_CALL:
            raise ObservabilityContractError("tool operation is invalid")
        _require_status(self.status)
        _require_duration(self.duration_ms)
        _require_retry(self.retry_count)
        if not isinstance(self.tool_name, ToolCode) or not isinstance(self.risk_level, RiskCode):
            raise ObservabilityContractError("tool classification is invalid")
        _require_error(self.error_type)


@dataclass(frozen=True, slots=True)
class MemoryOperationRecordV1:
    operation: OperationCode
    status: ObservabilityStatus
    duration_ms: int
    error_type: NormalizedErrorTypeV1 | None = None

    def __post_init__(self) -> None:
        if self.operation not in {
            OperationCode.MEMORY_LOAD,
            OperationCode.MEMORY_UPDATE,
            OperationCode.MEMORY_SUMMARY,
            OperationCode.MEMORY_RECENT_FALLBACK,
        }:
            raise ObservabilityContractError("memory operation is invalid")
        _require_status(self.status)
        _require_duration(self.duration_ms)
        _require_error(self.error_type)


@dataclass(frozen=True, slots=True)
class HITLOperationRecordV1:
    operation: OperationCode
    status: ObservabilityStatus
    duration_ms: int
    error_type: NormalizedErrorTypeV1 | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.operation, OperationCode) or not self.operation.value.startswith("HITL_"):
            raise ObservabilityContractError("HITL operation is invalid")
        _require_status(self.status)
        _require_duration(self.duration_ms)
        _require_error(self.error_type)


@dataclass(frozen=True, slots=True)
class RevalidationOperationRecordV1:
    operation: OperationCode
    status: ObservabilityStatus
    duration_ms: int
    action_type: ActionCode
    risk_level: RiskCode
    error_type: NormalizedErrorTypeV1 | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.operation, OperationCode) or not self.operation.value.startswith("REVALIDATION_"):
            raise ObservabilityContractError("revalidation operation is invalid")
        _require_status(self.status)
        _require_duration(self.duration_ms)
        if not isinstance(self.action_type, ActionCode) or not isinstance(self.risk_level, RiskCode):
            raise ObservabilityContractError("revalidation classification is invalid")
        _require_error(self.error_type)


@dataclass(frozen=True, slots=True)
class ActionOperationRecordV1:
    operation: OperationCode
    status: ObservabilityStatus
    duration_ms: int
    retry_count: int
    action_type: ActionCode
    risk_level: RiskCode
    error_type: NormalizedErrorTypeV1 | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.operation, OperationCode) or not self.operation.value.startswith("ACTION_"):
            raise ObservabilityContractError("action operation is invalid")
        _require_status(self.status)
        _require_duration(self.duration_ms)
        _require_retry(self.retry_count)
        if not isinstance(self.action_type, ActionCode) or not isinstance(self.risk_level, RiskCode):
            raise ObservabilityContractError("action classification is invalid")
        _require_error(self.error_type)


type OperationRecordV1 = (
    NodeOperationRecordV1
    | LLMOperationRecordV1
    | RetrievalOperationRecordV1
    | ToolOperationRecordV1
    | MemoryOperationRecordV1
    | HITLOperationRecordV1
    | RevalidationOperationRecordV1
    | ActionOperationRecordV1
)

_RECORD_TYPES = (
    NodeOperationRecordV1,
    LLMOperationRecordV1,
    RetrievalOperationRecordV1,
    ToolOperationRecordV1,
    MemoryOperationRecordV1,
    HITLOperationRecordV1,
    RevalidationOperationRecordV1,
    ActionOperationRecordV1,
)

OPERATION_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "operation",
        "status",
        "duration_ms",
        "retry_count",
        "model_family",
        "prompt_tokens",
        "completion_tokens",
        "channel",
        "tool_name",
        "action_type",
        "risk_level",
        "error_type",
    }
)


def operation_record_payload(record: object) -> dict[str, object]:
    if type(record) not in _RECORD_TYPES:
        raise ObservabilityContractError("operation record is invalid")
    assert isinstance(record, _RECORD_TYPES)
    kind_by_type = {
        NodeOperationRecordV1: OperationRecordKind.NODE,
        LLMOperationRecordV1: OperationRecordKind.LLM,
        RetrievalOperationRecordV1: OperationRecordKind.RETRIEVAL,
        ToolOperationRecordV1: OperationRecordKind.TOOL,
        MemoryOperationRecordV1: OperationRecordKind.MEMORY,
        HITLOperationRecordV1: OperationRecordKind.HITL,
        RevalidationOperationRecordV1: OperationRecordKind.REVALIDATION,
        ActionOperationRecordV1: OperationRecordKind.ACTION,
    }
    model_family = getattr(record, "model_family", None)
    channel = getattr(record, "channel", None)
    tool_name = getattr(record, "tool_name", None)
    action_type = getattr(record, "action_type", None)
    risk_level = getattr(record, "risk_level", None)
    return {
        "schema_version": OPERATION_RECORD_SCHEMA_VERSION_V1,
        "kind": kind_by_type[type(record)].value,
        "operation": record.operation.value,
        "status": record.status.value,
        "duration_ms": record.duration_ms,
        "retry_count": getattr(record, "retry_count", None),
        "model_family": _optional_enum_value(model_family),
        "prompt_tokens": getattr(record, "prompt_tokens", None),
        "completion_tokens": getattr(record, "completion_tokens", None),
        "channel": _optional_enum_value(channel),
        "tool_name": _optional_enum_value(tool_name),
        "action_type": _optional_enum_value(action_type),
        "risk_level": _optional_enum_value(risk_level),
        "error_type": record.error_type.code.value if record.error_type is not None else None,
    }


def _optional_enum_value(value: object) -> str | None:
    return value.value if isinstance(value, StrEnum) else None


def validate_operation_record_payload(payload: object) -> None:
    if not isinstance(payload, Mapping) or set(payload) != OPERATION_RECORD_FIELDS:
        raise ObservabilityContractError("operation record is invalid")
    try:
        if payload["schema_version"] != OPERATION_RECORD_SCHEMA_VERSION_V1:
            raise ValueError
        kind = OperationRecordKind(payload["kind"])
        operation = OperationCode(payload["operation"])
        ObservabilityStatus(payload["status"])
        duration = payload["duration_ms"]
        if type(duration) is not int or duration < 0:
            raise ValueError
        error = payload["error_type"]
        if error is not None:
            from app.observability.contracts import NormalizedErrorCode

            NormalizedErrorCode(error)
        retry = payload["retry_count"]
        if kind in {
            OperationRecordKind.LLM,
            OperationRecordKind.RETRIEVAL,
            OperationRecordKind.TOOL,
            OperationRecordKind.ACTION,
        }:
            if type(retry) is not int or retry < 0:
                raise ValueError
        elif retry is not None:
            raise ValueError
        expected_prefix = {
            OperationRecordKind.NODE: "NODE_",
            OperationRecordKind.LLM: "LLM_",
            OperationRecordKind.RETRIEVAL: "RETRIEVAL_",
            OperationRecordKind.TOOL: "TOOL_",
            OperationRecordKind.MEMORY: "MEMORY_",
            OperationRecordKind.HITL: "HITL_",
            OperationRecordKind.REVALIDATION: "REVALIDATION_",
            OperationRecordKind.ACTION: "ACTION_",
        }[kind]
        if not operation.value.startswith(expected_prefix):
            raise ValueError
        if kind is OperationRecordKind.LLM:
            ModelFamily(payload["model_family"])
            for key in ("prompt_tokens", "completion_tokens"):
                value = payload[key]
                if value is not None and (type(value) is not int or value < 0):
                    raise ValueError
        elif any(
            payload[key] is not None
            for key in ("model_family", "prompt_tokens", "completion_tokens")
        ):
            raise ValueError
        if kind is OperationRecordKind.RETRIEVAL:
            RetrievalChannel(payload["channel"])
        elif payload["channel"] is not None:
            raise ValueError
        if kind is OperationRecordKind.TOOL:
            ToolCode(payload["tool_name"])
            RiskCode(payload["risk_level"])
        elif payload["tool_name"] is not None:
            raise ValueError
        if kind in {OperationRecordKind.REVALIDATION, OperationRecordKind.ACTION}:
            ActionCode(payload["action_type"])
            RiskCode(payload["risk_level"])
        elif kind is not OperationRecordKind.TOOL and (
            payload["action_type"] is not None or payload["risk_level"] is not None
        ):
            raise ValueError
        if kind is OperationRecordKind.TOOL and payload["action_type"] is not None:
            raise ValueError
    except (TypeError, ValueError):
        raise ObservabilityContractError("operation record is invalid") from None


__all__ = [
    "OPERATION_RECORD_SCHEMA_VERSION_V1",
    "OPERATION_RECORD_FIELDS",
    "ActionCode",
    "ActionOperationRecordV1",
    "HITLOperationRecordV1",
    "LLMOperationRecordV1",
    "MemoryOperationRecordV1",
    "ModelFamily",
    "NodeOperationRecordV1",
    "OperationCode",
    "OperationRecordKind",
    "OperationRecordV1",
    "RetrievalChannel",
    "RetrievalOperationRecordV1",
    "RevalidationOperationRecordV1",
    "RiskCode",
    "ToolCode",
    "ToolOperationRecordV1",
    "operation_record_payload",
    "validate_operation_record_payload",
]
