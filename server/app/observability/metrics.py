"""Closed V1 metric registry and provider-neutral recorder contract."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

METRIC_SCHEMA_VERSION_V1 = "OBSERVABILITY_METRIC_V1"


class MetricKind(StrEnum):
    COUNTER = "COUNTER"
    HISTOGRAM = "HISTOGRAM"


class MetricUnit(StrEnum):
    COUNT = "1"
    SECONDS = "s"
    TOKENS = "{token}"
    MILLISECONDS = "ms"


class MetricName(StrEnum):
    ATTEMPT_TOTAL = "agent_attempt_total"
    ATTEMPT_DURATION_SECONDS = "agent_attempt_duration_seconds"
    INTERRUPT_TOTAL = "agent_interrupt_total"
    RESUME_TOTAL = "agent_resume_total"
    LLM_CALL_TOTAL = "agent_llm_call_total"
    LLM_DURATION_SECONDS = "agent_llm_duration_seconds"
    LLM_TOKENS_TOTAL = "agent_llm_tokens_total"
    RETRIEVAL_DURATION_SECONDS = "agent_retrieval_duration_seconds"
    TOOL_CALL_TOTAL = "agent_tool_call_total"
    TOOL_DURATION_SECONDS = "agent_tool_duration_seconds"
    ACTION_TOTAL = "agent_action_total"
    MEMORY_SUMMARY_TOTAL = "agent_memory_summary_total"
    OBSERVABILITY_EXPORT_FAILURE_TOTAL = "agent_observability_export_failure_total"
    AUDIT_FAILURE_TOTAL = "agent_audit_failure_total"


_STATUS_VALUES = (
    "SUCCESS",
    "ERROR",
    "CANCELLED",
    "WAITING",
    "REJECTED",
    "CONFLICT",
    "STALE",
    "UNKNOWN",
)
_OPERATION_VALUES = ("PLANNER", "ANSWER", "SUMMARY", "UNKNOWN")
_TOOL_VALUES = (
    "list_my_orders",
    "get_order_detail",
    "get_product_information",
    "search_knowledge_base",
    "request_order_cancellation",
    "request_refund",
    "create_support_ticket",
)
_FROZEN_LABEL_VALUES: dict[str, frozenset[str]] = {
    "status": frozenset(_STATUS_VALUES),
    "resume": frozenset({"FRESH", "RESUME"}),
    "reason": frozenset({"CUSTOMER_CONFIRMATION", "ADMIN_APPROVAL", "UNKNOWN"}),
    "operation": frozenset(_OPERATION_VALUES),
    "model_family": frozenset({"OPENAI_COMPATIBLE", "MOCK", "UNKNOWN"}),
    "direction": frozenset({"INPUT", "OUTPUT"}),
    "channel": frozenset({"KEYWORD", "VECTOR", "RULE", "FUSED", "UNKNOWN"}),
    "tool_name": frozenset(_TOOL_VALUES),
    "risk_level": frozenset({"READ_ONLY", "LOW", "MEDIUM", "HIGH"}),
    "action_type": frozenset({"CANCEL_ORDER", "REQUEST_REFUND", "UNKNOWN"}),
    "provider": frozenset({"NOOP", "LOCAL", "UNKNOWN"}),
    "error_type": frozenset({"TIMEOUT", "UNAVAILABLE", "REJECTED", "UNKNOWN"}),
    "event_type": frozenset(
        {"ADMIN_DECISION_DENIAL", "BUSINESS_EXECUTION", "UNKNOWN"}
    ),
}


class MetricRecordReason(StrEnum):
    PAYLOAD_INVALID = "PAYLOAD_INVALID"
    UNKNOWN_METRIC = "UNKNOWN_METRIC"
    KIND_MISMATCH = "KIND_MISMATCH"
    UNIT_MISMATCH = "UNIT_MISMATCH"
    LABEL_KEYS_MISMATCH = "LABEL_KEYS_MISMATCH"
    LABEL_VALUE_REJECTED = "LABEL_VALUE_REJECTED"
    VALUE_INVALID = "VALUE_INVALID"
    UNKNOWN_VALUE_OMITTED = "UNKNOWN_VALUE_OMITTED"
    PROTECTION_FAILED = "PROTECTION_FAILED"
    SINK_FAILED = "SINK_FAILED"


class MetricContractError(ValueError):
    """Sanitized metric value-object construction failure."""


@dataclass(frozen=True, slots=True)
class MetricLabelV1:
    key: str
    value: str

    def __post_init__(self) -> None:
        if type(self.key) is not str or self.key not in _FROZEN_LABEL_VALUES:
            raise MetricContractError("metric label key is invalid")
        if (
            type(self.value) is not str
            or self.value not in _FROZEN_LABEL_VALUES[self.key]
        ):
            raise MetricContractError("metric label value is invalid")
        if "\n" in self.key or "\r" in self.key or "\n" in self.value or "\r" in self.value:
            raise MetricContractError("metric label contains an invalid character")


@dataclass(frozen=True, slots=True)
class MetricPointV1:
    name: MetricName
    kind: MetricKind
    unit: MetricUnit
    value: int | float | None
    labels: tuple[MetricLabelV1, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, MetricName):
            raise MetricContractError("metric name is invalid")
        if not isinstance(self.kind, MetricKind):
            raise MetricContractError("metric kind is invalid")
        if not isinstance(self.unit, MetricUnit):
            raise MetricContractError("metric unit is invalid")
        if self.value is not None and type(self.value) not in {int, float}:
            raise MetricContractError("metric value is invalid")
        if type(self.labels) is not tuple or any(type(item) is not MetricLabelV1 for item in self.labels):
            raise MetricContractError("metric labels are invalid")
        ordered = tuple(sorted(self.labels, key=lambda item: item.key))
        if len({item.key for item in ordered}) != len(ordered):
            raise MetricContractError("metric label keys are duplicated")
        object.__setattr__(self, "labels", ordered)


@dataclass(frozen=True, slots=True)
class MetricLabelRuleV1:
    key: str
    allowed_values: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.key) is not str or self.key not in _FROZEN_LABEL_VALUES:
            raise MetricContractError("metric label rule key is invalid")
        if (
            type(self.allowed_values) is not tuple
            or not self.allowed_values
            or any(type(value) is not str or not value for value in self.allowed_values)
            or len(set(self.allowed_values)) != len(self.allowed_values)
            or any(
                value not in _FROZEN_LABEL_VALUES[self.key]
                for value in self.allowed_values
            )
        ):
            raise MetricContractError("metric label rule values are invalid")


@dataclass(frozen=True, slots=True)
class MetricDefinitionV1:
    name: MetricName
    kind: MetricKind
    unit: MetricUnit
    label_rules: tuple[MetricLabelRuleV1, ...]
    allow_unknown_value: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, MetricName):
            raise MetricContractError("metric definition name is invalid")
        if not isinstance(self.kind, MetricKind) or not isinstance(self.unit, MetricUnit):
            raise MetricContractError("metric definition shape is invalid")
        if type(self.label_rules) is not tuple or any(
            type(rule) is not MetricLabelRuleV1 for rule in self.label_rules
        ):
            raise MetricContractError("metric definition labels are invalid")
        if len({rule.key for rule in self.label_rules}) != len(self.label_rules):
            raise MetricContractError("metric definition label keys are duplicated")
        if type(self.allow_unknown_value) is not bool:
            raise MetricContractError("metric unknown-value policy is invalid")


@dataclass(frozen=True, slots=True)
class MetricRecordResultV1:
    accepted: bool
    reason: MetricRecordReason | None = None

    def __post_init__(self) -> None:
        if type(self.accepted) is not bool or self.accepted == (self.reason is not None):
            raise MetricContractError("metric record result is inconsistent")


@dataclass(frozen=True, slots=True)
class MetricRegistryV1:
    schema_version: str
    definitions: tuple[MetricDefinitionV1, ...]

    def __post_init__(self) -> None:
        if self.schema_version != METRIC_SCHEMA_VERSION_V1:
            raise MetricContractError("metric registry version is invalid")
        if type(self.definitions) is not tuple or not self.definitions:
            raise MetricContractError("metric registry is empty")
        if len({definition.name for definition in self.definitions}) != len(self.definitions):
            raise MetricContractError("metric registry contains duplicate names")

    def definition(self, name: MetricName) -> MetricDefinitionV1 | None:
        return next((item for item in self.definitions if item.name == name), None)

    def rejection_reason(self, point: object) -> MetricRecordReason | None:
        if type(point) is not MetricPointV1:
            return MetricRecordReason.PAYLOAD_INVALID
        definition = self.definition(point.name)
        if definition is None:
            return MetricRecordReason.UNKNOWN_METRIC
        if point.kind is not definition.kind:
            return MetricRecordReason.KIND_MISMATCH
        if point.unit is not definition.unit:
            return MetricRecordReason.UNIT_MISMATCH
        expected_keys = tuple(sorted(rule.key for rule in definition.label_rules))
        actual_keys = tuple(label.key for label in point.labels)
        if actual_keys != expected_keys:
            return MetricRecordReason.LABEL_KEYS_MISMATCH
        allowed_by_key = {rule.key: frozenset(rule.allowed_values) for rule in definition.label_rules}
        if any(label.value not in allowed_by_key[label.key] for label in point.labels):
            return MetricRecordReason.LABEL_VALUE_REJECTED
        if point.value is None:
            return (
                MetricRecordReason.UNKNOWN_VALUE_OMITTED
                if definition.allow_unknown_value
                else MetricRecordReason.VALUE_INVALID
            )
        if type(point.value) not in {int, float} or not math.isfinite(point.value):
            return MetricRecordReason.VALUE_INVALID
        if point.kind is MetricKind.COUNTER:
            if type(point.value) is not int or point.value <= 0:
                return MetricRecordReason.VALUE_INVALID
        elif point.value < 0:
            return MetricRecordReason.VALUE_INVALID
        return None


class MetricsRecorder(Protocol):
    def record(self, point: MetricPointV1) -> MetricRecordResultV1: ...


def _rule(key: str, *values: str) -> MetricLabelRuleV1:
    return MetricLabelRuleV1(key=key, allowed_values=tuple(values))


DEFAULT_METRIC_REGISTRY_V1 = MetricRegistryV1(
    schema_version=METRIC_SCHEMA_VERSION_V1,
    definitions=(
        MetricDefinitionV1(
            MetricName.ATTEMPT_TOTAL,
            MetricKind.COUNTER,
            MetricUnit.COUNT,
            (_rule("status", *_STATUS_VALUES), _rule("resume", "FRESH", "RESUME")),
        ),
        MetricDefinitionV1(
            MetricName.ATTEMPT_DURATION_SECONDS,
            MetricKind.HISTOGRAM,
            MetricUnit.SECONDS,
            (_rule("status", *_STATUS_VALUES),),
        ),
        MetricDefinitionV1(
            MetricName.INTERRUPT_TOTAL,
            MetricKind.COUNTER,
            MetricUnit.COUNT,
            (_rule("reason", "CUSTOMER_CONFIRMATION", "ADMIN_APPROVAL", "UNKNOWN"),),
        ),
        MetricDefinitionV1(
            MetricName.RESUME_TOTAL,
            MetricKind.COUNTER,
            MetricUnit.COUNT,
            (_rule("status", *_STATUS_VALUES),),
        ),
        MetricDefinitionV1(
            MetricName.LLM_CALL_TOTAL,
            MetricKind.COUNTER,
            MetricUnit.COUNT,
            (
                _rule("operation", *_OPERATION_VALUES),
                _rule("status", *_STATUS_VALUES),
                _rule("model_family", "OPENAI_COMPATIBLE", "MOCK", "UNKNOWN"),
            ),
        ),
        MetricDefinitionV1(
            MetricName.LLM_DURATION_SECONDS,
            MetricKind.HISTOGRAM,
            MetricUnit.SECONDS,
            (_rule("operation", *_OPERATION_VALUES), _rule("status", *_STATUS_VALUES)),
        ),
        MetricDefinitionV1(
            MetricName.LLM_TOKENS_TOTAL,
            MetricKind.COUNTER,
            MetricUnit.TOKENS,
            (_rule("operation", *_OPERATION_VALUES), _rule("direction", "INPUT", "OUTPUT")),
            allow_unknown_value=True,
        ),
        MetricDefinitionV1(
            MetricName.RETRIEVAL_DURATION_SECONDS,
            MetricKind.HISTOGRAM,
            MetricUnit.SECONDS,
            (
                _rule("channel", "KEYWORD", "VECTOR", "RULE", "FUSED", "UNKNOWN"),
                _rule("status", *_STATUS_VALUES),
            ),
        ),
        MetricDefinitionV1(
            MetricName.TOOL_CALL_TOTAL,
            MetricKind.COUNTER,
            MetricUnit.COUNT,
            (
                _rule("tool_name", *_TOOL_VALUES),
                _rule("status", *_STATUS_VALUES),
                _rule("risk_level", "READ_ONLY", "LOW", "MEDIUM", "HIGH"),
            ),
        ),
        MetricDefinitionV1(
            MetricName.TOOL_DURATION_SECONDS,
            MetricKind.HISTOGRAM,
            MetricUnit.SECONDS,
            (_rule("tool_name", *_TOOL_VALUES), _rule("status", *_STATUS_VALUES)),
        ),
        MetricDefinitionV1(
            MetricName.ACTION_TOTAL,
            MetricKind.COUNTER,
            MetricUnit.COUNT,
            (
                _rule("action_type", "CANCEL_ORDER", "REQUEST_REFUND", "UNKNOWN"),
                _rule("status", *_STATUS_VALUES),
            ),
        ),
        MetricDefinitionV1(
            MetricName.MEMORY_SUMMARY_TOTAL,
            MetricKind.COUNTER,
            MetricUnit.COUNT,
            (_rule("status", *_STATUS_VALUES),),
        ),
        MetricDefinitionV1(
            MetricName.OBSERVABILITY_EXPORT_FAILURE_TOTAL,
            MetricKind.COUNTER,
            MetricUnit.COUNT,
            (
                _rule("provider", "NOOP", "LOCAL", "UNKNOWN"),
                _rule("error_type", "TIMEOUT", "UNAVAILABLE", "REJECTED", "UNKNOWN"),
            ),
        ),
        MetricDefinitionV1(
            MetricName.AUDIT_FAILURE_TOTAL,
            MetricKind.COUNTER,
            MetricUnit.COUNT,
            (_rule("event_type", "ADMIN_DECISION_DENIAL", "BUSINESS_EXECUTION", "UNKNOWN"),),
        ),
    ),
)


__all__ = [
    "DEFAULT_METRIC_REGISTRY_V1",
    "METRIC_SCHEMA_VERSION_V1",
    "MetricContractError",
    "MetricDefinitionV1",
    "MetricKind",
    "MetricLabelRuleV1",
    "MetricLabelV1",
    "MetricName",
    "MetricPointV1",
    "MetricRecordReason",
    "MetricRecordResultV1",
    "MetricRegistryV1",
    "MetricUnit",
    "MetricsRecorder",
]
