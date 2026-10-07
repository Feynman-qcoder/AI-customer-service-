"""Closed contracts for durable, replay-safe business execution."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from app.agent.state import ActionDraftSnapshot
from app.runtime.context import ExecutionScope
from app.runtime.data_protection import (
    DataProtectionPolicy,
    DataProtectionProfile,
    DataProtectionSchemaPolicy,
    FieldProtection,
    SchemaPolicyError,
)
from app.runtime.durable import EffectWriteScope

BUSINESS_EXECUTION_AUDIT_SCHEMA_VERSION = "BUSINESS_EXECUTION_AUDIT_V1"
BUSINESS_EXECUTION_NODE_NAME = "durable_business_execute"
BUSINESS_EXECUTION_PURPOSE = "BUSINESS_EXECUTION_AUDIT"
BUSINESS_EXECUTION_SEQUENCE = 0
BUSINESS_EXECUTION_UNAVAILABLE_STATUS = "ORDER_UNAVAILABLE"

_CODE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_ORDER_NO = re.compile(r"^[A-Z0-9][A-Z0-9_-]{0,63}$")
_LOGICAL_ACTION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")
_AUDIT_FIELDS = frozenset(
    {
        "schema_version",
        "logical_action_id",
        "action_type",
        "target_order_id",
        "target_order_no",
        "subject_user_id",
        "admin_decision",
        "policy_version",
        "before_status",
        "after_status",
        "result_code",
        "reason_code",
    }
)
_V1_ORDER_STATUSES = frozenset(
    {
        "PENDING_PAYMENT",
        "PAID",
        "WAITING_SHIPMENT",
        "SHIPPED",
        "IN_TRANSIT",
        "SIGNED",
        "REFUNDING",
        "REFUND_PENDING",
        "AFTER_SALE_REVIEWING",
        "REFUNDED",
        "CANCELLED",
    }
)
_V1_EXECUTED_TRANSITIONS = frozenset(
    {
        (
            "ORDER_CANCELLATION",
            "PAID",
            "CANCELLED",
            "ORDER_CANCELLED",
        ),
        (
            "ORDER_CANCELLATION",
            "WAITING_SHIPMENT",
            "CANCELLED",
            "ORDER_CANCELLED",
        ),
        ("REFUND", "PAID", "REFUND_PENDING", "REFUND_PENDING"),
        (
            "REFUND",
            "WAITING_SHIPMENT",
            "REFUND_PENDING",
            "REFUND_PENDING",
        ),
        (
            "REFUND",
            "SHIPPED",
            "AFTER_SALE_REVIEWING",
            "AFTER_SALE_REVIEWING",
        ),
        (
            "REFUND",
            "IN_TRANSIT",
            "AFTER_SALE_REVIEWING",
            "AFTER_SALE_REVIEWING",
        ),
        (
            "REFUND",
            "SIGNED",
            "AFTER_SALE_REVIEWING",
            "AFTER_SALE_REVIEWING",
        ),
    }
)
_V1_EXECUTABLE_BEFORE_STATUSES = {
    "ORDER_CANCELLATION": frozenset({"PAID", "WAITING_SHIPMENT"}),
    "REFUND": frozenset(
        {"PAID", "WAITING_SHIPMENT", "SHIPPED", "IN_TRANSIT", "SIGNED"}
    ),
}


class BusinessExecutionResultCode(StrEnum):
    ORDER_CANCELLED = "ORDER_CANCELLED"
    REFUND_PENDING = "REFUND_PENDING"
    AFTER_SALE_REVIEWING = "AFTER_SALE_REVIEWING"
    STALE = "STALE"


class BusinessExecutionReasonCode(StrEnum):
    BUSINESS_ACTION_EXECUTED = "BUSINESS_ACTION_EXECUTED"
    ORDER_MISSING = "ORDER_MISSING"
    ORDER_OWNER_CHANGED = "ORDER_OWNER_CHANGED"
    ORDER_STATUS_INELIGIBLE = "ORDER_STATUS_INELIGIBLE"
    POLICY_VERSION_INCOMPATIBLE = "POLICY_VERSION_INCOMPATIBLE"


class BusinessExecutionOutcome(StrEnum):
    EXECUTED = "EXECUTED"
    STALE = "STALE"


class BusinessExecutionError(RuntimeError):
    """Sanitized application or persistence failure."""


class BusinessExecutionConflict(BusinessExecutionError):
    """The current attempt does not own live execution authority."""


class BusinessExecutionCorruption(BusinessExecutionError):
    """Persisted action, prepare, or execution evidence is inconsistent."""


@dataclass(frozen=True, slots=True)
class BusinessExecutionScope:
    effect_scope: EffectWriteScope
    conversation_id: int
    run_id: str
    admin_actor_id: int
    subject_user_id: int


@dataclass(frozen=True, slots=True)
class BusinessExecutionWrite:
    scope: BusinessExecutionScope
    action_id: int
    draft: ActionDraftSnapshot


@dataclass(frozen=True, slots=True)
class StoredBusinessExecutionAudit:
    effect_id: int
    effect_attempt_id: str
    idempotency_key: str
    payload_digest: str
    canonical_json: str


@dataclass(frozen=True, slots=True)
class LockedBusinessExecution:
    action_id: int
    run_id: str
    logical_action_id: str
    action_type: str
    target_order_id: int
    target_order_no: str
    subject_user_id: int
    admin_actor_id: int
    policy_version: str
    action_status: str
    resume_status: str
    execution_result_code: str | None
    executed_at: datetime | None
    order_exists: bool
    current_order_owner_id: int | None
    current_order_status: str | None
    order_product_id: int | None
    order_quantity: int | None
    audit: StoredBusinessExecutionAudit | None


@dataclass(frozen=True, slots=True)
class BusinessExecutionTransition:
    outcome: BusinessExecutionOutcome
    result_code: BusinessExecutionResultCode
    reason_code: BusinessExecutionReasonCode
    before_status: str
    after_status: str
    restore_stock: bool = False


@dataclass(frozen=True, slots=True)
class ProtectedBusinessExecutionAudit:
    canonical_json: str
    payload_digest: str


@dataclass(frozen=True, slots=True)
class BusinessExecutionResult:
    action_id: int
    outcome: BusinessExecutionOutcome
    result_code: BusinessExecutionResultCode
    reason_code: BusinessExecutionReasonCode
    before_status: str
    after_status: str


@dataclass(frozen=True, slots=True)
class BusinessExecutionAuditEvent:
    logical_action_id: str
    action_type: str
    target_order_id: int
    target_order_no: str
    subject_user_id: int
    admin_decision: str
    policy_version: str
    before_status: str
    after_status: str
    result_code: str
    reason_code: str
    schema_version: str = BUSINESS_EXECUTION_AUDIT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != BUSINESS_EXECUTION_AUDIT_SCHEMA_VERSION:
            raise BusinessExecutionCorruption("business execution audit version is invalid")
        if _LOGICAL_ACTION.fullmatch(self.logical_action_id) is None:
            raise BusinessExecutionCorruption("business execution logical action is invalid")
        if self.action_type not in {"REFUND", "ORDER_CANCELLATION"}:
            raise BusinessExecutionCorruption("business execution action type is invalid")
        if type(self.target_order_id) is not int or self.target_order_id <= 0:
            raise BusinessExecutionCorruption("business execution target is invalid")
        if _ORDER_NO.fullmatch(self.target_order_no) is None:
            raise BusinessExecutionCorruption("business execution order reference is invalid")
        if type(self.subject_user_id) is not int or self.subject_user_id <= 0:
            raise BusinessExecutionCorruption("business execution subject is invalid")
        if self.admin_decision != "APPROVE":
            raise BusinessExecutionCorruption("business execution decision is invalid")
        if not self.policy_version or len(self.policy_version) > 64:
            raise BusinessExecutionCorruption("business execution policy version is invalid")
        if any(
            _CODE.fullmatch(value) is None
            for value in (
                self.before_status,
                self.after_status,
                self.result_code,
                self.reason_code,
            )
        ):
            raise BusinessExecutionCorruption("business execution audit code is invalid")
        try:
            BusinessExecutionResultCode(self.result_code)
            BusinessExecutionReasonCode(self.reason_code)
        except ValueError:
            raise BusinessExecutionCorruption("business execution audit code is unknown") from None

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "logical_action_id": self.logical_action_id,
            "action_type": self.action_type,
            "target_order_id": self.target_order_id,
            "target_order_no": self.target_order_no,
            "subject_user_id": self.subject_user_id,
            "admin_decision": self.admin_decision,
            "policy_version": self.policy_version,
            "before_status": self.before_status,
            "after_status": self.after_status,
            "result_code": self.result_code,
            "reason_code": self.reason_code,
        }

    @classmethod
    def from_canonical_json(cls, value: str) -> BusinessExecutionAuditEvent:
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            raise BusinessExecutionCorruption("business execution audit is invalid") from None
        if type(decoded) is not dict or set(decoded) != _AUDIT_FIELDS:
            raise BusinessExecutionCorruption("business execution audit shape is invalid")
        try:
            event = cls(
                schema_version=decoded["schema_version"],
                logical_action_id=decoded["logical_action_id"],
                action_type=decoded["action_type"],
                target_order_id=decoded["target_order_id"],
                target_order_no=decoded["target_order_no"],
                subject_user_id=decoded["subject_user_id"],
                admin_decision=decoded["admin_decision"],
                policy_version=decoded["policy_version"],
                before_status=decoded["before_status"],
                after_status=decoded["after_status"],
                result_code=decoded["result_code"],
                reason_code=decoded["reason_code"],
            )
        except (KeyError, TypeError, BusinessExecutionCorruption):
            raise BusinessExecutionCorruption("business execution audit is invalid") from None
        canonical = json.dumps(
            event.payload(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        if canonical != value:
            raise BusinessExecutionCorruption("business execution audit is not canonical")
        return event


def decode_business_execution_audit_v1(
    *,
    action_id: int,
    action_status: str,
    action_type: str,
    draft_policy_version: str,
    execution_result_code: str | None,
    executed_at: datetime | None,
    event: BusinessExecutionAuditEvent,
) -> BusinessExecutionResult:
    """Interpret a persisted V1 terminal audit without consulting live policy."""

    if event.schema_version != BUSINESS_EXECUTION_AUDIT_SCHEMA_VERSION:
        raise BusinessExecutionCorruption(
            "business execution audit version is invalid"
        )
    if (
        event.action_type != action_type
        or event.policy_version != draft_policy_version
    ):
        raise BusinessExecutionCorruption(
            "terminal business audit does not match its frozen draft"
        )
    result_code = BusinessExecutionResultCode(event.result_code)
    reason_code = BusinessExecutionReasonCode(event.reason_code)

    if action_status == "EXECUTED":
        frozen_transition = (
            event.action_type,
            event.before_status,
            event.after_status,
            event.result_code,
        )
        if (
            execution_result_code != result_code.value
            or executed_at is None
            or reason_code
            is not BusinessExecutionReasonCode.BUSINESS_ACTION_EXECUTED
            or frozen_transition not in _V1_EXECUTED_TRANSITIONS
        ):
            raise BusinessExecutionCorruption(
                "executed business result is inconsistent"
            )
        outcome = BusinessExecutionOutcome.EXECUTED
    elif action_status == "STALE":
        if (
            execution_result_code is not None
            or executed_at is not None
            or result_code is not BusinessExecutionResultCode.STALE
            or reason_code
            is BusinessExecutionReasonCode.BUSINESS_ACTION_EXECUTED
            or event.after_status != event.before_status
        ):
            raise BusinessExecutionCorruption(
                "stale business result is inconsistent"
            )
        status = event.before_status
        if reason_code is BusinessExecutionReasonCode.ORDER_MISSING:
            if status != BUSINESS_EXECUTION_UNAVAILABLE_STATUS:
                raise BusinessExecutionCorruption(
                    "missing order audit is inconsistent"
                )
        elif reason_code is BusinessExecutionReasonCode.ORDER_OWNER_CHANGED:
            if status not in _V1_ORDER_STATUSES:
                raise BusinessExecutionCorruption(
                    "owner-change audit status is invalid"
                )
        elif reason_code is BusinessExecutionReasonCode.ORDER_STATUS_INELIGIBLE:
            executable_statuses = _V1_EXECUTABLE_BEFORE_STATUSES.get(
                event.action_type
            )
            if (
                status not in _V1_ORDER_STATUSES
                or executable_statuses is None
                or status in executable_statuses
            ):
                raise BusinessExecutionCorruption(
                    "ineligible-order audit status is invalid"
                )
        elif reason_code is BusinessExecutionReasonCode.POLICY_VERSION_INCOMPATIBLE:
            if (
                status != BUSINESS_EXECUTION_UNAVAILABLE_STATUS
                and status not in _V1_ORDER_STATUSES
            ):
                raise BusinessExecutionCorruption(
                    "policy-mismatch audit status is invalid"
                )
        else:
            raise BusinessExecutionCorruption(
                "stale business reason is invalid"
            )
        outcome = BusinessExecutionOutcome.STALE
    else:
        raise BusinessExecutionCorruption("business result is not terminal")

    return BusinessExecutionResult(
        action_id=action_id,
        outcome=outcome,
        result_code=result_code,
        reason_code=reason_code,
        before_status=event.before_status,
        after_status=event.after_status,
    )


class BusinessExecutionAuditSchemaPolicy(DataProtectionSchemaPolicy):
    def normalize(self, payload: object) -> object:
        if type(payload) is not dict or set(payload) != _AUDIT_FIELDS:
            raise SchemaPolicyError("BUSINESS_EXECUTION_AUDIT_SCHEMA_INVALID")
        try:
            event = BusinessExecutionAuditEvent(
                schema_version=payload["schema_version"],
                logical_action_id=payload["logical_action_id"],
                action_type=payload["action_type"],
                target_order_id=payload["target_order_id"],
                target_order_no=payload["target_order_no"],
                subject_user_id=payload["subject_user_id"],
                admin_decision=payload["admin_decision"],
                policy_version=payload["policy_version"],
                before_status=payload["before_status"],
                after_status=payload["after_status"],
                result_code=payload["result_code"],
                reason_code=payload["reason_code"],
            )
        except (KeyError, TypeError, BusinessExecutionCorruption):
            raise SchemaPolicyError("BUSINESS_EXECUTION_AUDIT_SCHEMA_INVALID") from None
        return event.payload()

    def classify(self, path: tuple[str, ...], value: object) -> FieldProtection:
        del value
        if len(path) == 1 and path[0] in _AUDIT_FIELDS:
            return FieldProtection.EXACT
        return FieldProtection.REJECT

    def validate_protected(self, payload: object) -> None:
        self.normalize(payload)


def build_business_execution_audit_protection() -> DataProtectionPolicy:
    return DataProtectionPolicy(
        schema_policy=BusinessExecutionAuditSchemaPolicy(),
        profile=DataProtectionProfile.OBSERVABILITY,
    )


class BusinessExecutionStorePort(Protocol):
    async def lock_business_execution(
        self,
        write: BusinessExecutionWrite,
    ) -> LockedBusinessExecution: ...

    async def apply_business_execution(
        self,
        write: BusinessExecutionWrite,
        locked: LockedBusinessExecution,
        transition: BusinessExecutionTransition,
        audit: ProtectedBusinessExecutionAudit,
    ) -> BusinessExecutionResult: ...


class BusinessExecutionApplicationPort(Protocol):
    async def execute(
        self,
        *,
        execution: ExecutionScope,
        draft: ActionDraftSnapshot,
        pending_action_id: int,
    ) -> BusinessExecutionResult: ...

    async def verify_terminal(
        self,
        *,
        execution: ExecutionScope,
        draft: ActionDraftSnapshot,
        pending_action_id: int,
        expected_outcome: BusinessExecutionOutcome | None = None,
    ) -> BusinessExecutionResult: ...


__all__ = [
    "BUSINESS_EXECUTION_AUDIT_SCHEMA_VERSION",
    "BUSINESS_EXECUTION_NODE_NAME",
    "BUSINESS_EXECUTION_PURPOSE",
    "BUSINESS_EXECUTION_SEQUENCE",
    "BUSINESS_EXECUTION_UNAVAILABLE_STATUS",
    "BusinessExecutionApplicationPort",
    "BusinessExecutionAuditEvent",
    "BusinessExecutionConflict",
    "BusinessExecutionCorruption",
    "BusinessExecutionError",
    "BusinessExecutionOutcome",
    "BusinessExecutionReasonCode",
    "BusinessExecutionResult",
    "BusinessExecutionResultCode",
    "BusinessExecutionScope",
    "BusinessExecutionStorePort",
    "BusinessExecutionTransition",
    "BusinessExecutionWrite",
    "LockedBusinessExecution",
    "ProtectedBusinessExecutionAudit",
    "StoredBusinessExecutionAudit",
    "build_business_execution_audit_protection",
    "decode_business_execution_audit_v1",
]
