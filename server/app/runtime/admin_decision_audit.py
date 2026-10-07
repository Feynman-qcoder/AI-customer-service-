"""Closed security-audit contract for rejected admin-decision attempts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from app.runtime.data_protection import (
    DataProtectionPolicy,
    DataProtectionProfile,
    DataProtectionSchemaPolicy,
    FieldProtection,
    SchemaPolicyError,
)

ADMIN_DECISION_AUDIT_SCHEMA_VERSION = "ADMIN_DECISION_SECURITY_AUDIT_V1"
ADMIN_DECISION_DENIED_EVENT_CODE = "ADMIN_DECISION_ACCESS_DENIED"
ADMIN_DECISION_DENIED_RESULT_CODE = "DENIED"
ADMIN_DECISION_DENIED_REASON_CODE = "ACTOR_ROLE_NOT_ADMIN"

_AUDIT_FIELDS = frozenset(
    {
        "schema_version",
        "actor_user_id",
        "actor_role",
        "event_code",
        "result_code",
        "reason_code",
    }
)
_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")


class AdminDecisionSecurityAuditError(RuntimeError):
    """Sanitized failure at the mandatory unauthorized-audit boundary."""


@dataclass(frozen=True, slots=True)
class AdminDecisionDeniedAuditEvent:
    actor_user_id: int
    actor_role: str
    schema_version: str = ADMIN_DECISION_AUDIT_SCHEMA_VERSION
    event_code: str = ADMIN_DECISION_DENIED_EVENT_CODE
    result_code: str = ADMIN_DECISION_DENIED_RESULT_CODE
    reason_code: str = ADMIN_DECISION_DENIED_REASON_CODE

    def __post_init__(self) -> None:
        if type(self.actor_user_id) is not int or self.actor_user_id <= 0:
            raise AdminDecisionSecurityAuditError(
                "security audit actor identity is invalid"
            )
        if self.actor_role not in {"CUSTOMER", "OTHER"}:
            raise AdminDecisionSecurityAuditError(
                "security audit actor role is invalid"
            )
        expected = (
            ADMIN_DECISION_AUDIT_SCHEMA_VERSION,
            ADMIN_DECISION_DENIED_EVENT_CODE,
            ADMIN_DECISION_DENIED_RESULT_CODE,
            ADMIN_DECISION_DENIED_REASON_CODE,
        )
        actual = (
            self.schema_version,
            self.event_code,
            self.result_code,
            self.reason_code,
        )
        if actual != expected:
            raise AdminDecisionSecurityAuditError(
                "security audit codes are invalid"
            )

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "actor_user_id": self.actor_user_id,
            "actor_role": self.actor_role,
            "event_code": self.event_code,
            "result_code": self.result_code,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True, slots=True)
class ProtectedAdminDecisionAuditWrite:
    actor_user_id: int
    actor_role: str
    event_code: str
    result_code: str
    reason_code: str
    canonical_json: str
    payload_digest: str

    def __post_init__(self) -> None:
        event = AdminDecisionDeniedAuditEvent(
            actor_user_id=self.actor_user_id,
            actor_role=self.actor_role,
            event_code=self.event_code,
            result_code=self.result_code,
            reason_code=self.reason_code,
        )
        if type(self.canonical_json) is not str or not self.canonical_json:
            raise AdminDecisionSecurityAuditError(
                "protected security audit payload is invalid"
            )
        if (
            type(self.payload_digest) is not str
            or _DIGEST_PATTERN.fullmatch(self.payload_digest) is None
        ):
            raise AdminDecisionSecurityAuditError(
                "protected security audit digest is invalid"
            )
        del event


class AdminDecisionSecurityAuditPort(Protocol):
    async def record_denial(self, event: AdminDecisionDeniedAuditEvent) -> None: ...


class AdminDecisionSecurityAuditStorePort(Protocol):
    async def append_admin_decision_denial(
        self,
        write: ProtectedAdminDecisionAuditWrite,
    ) -> None: ...


class AdminDecisionAuditSchemaPolicy(DataProtectionSchemaPolicy):
    """Allow only the six fixed, low-cardinality audit fields."""

    def normalize(self, payload: object) -> object:
        if type(payload) is not dict or set(payload) != _AUDIT_FIELDS:
            raise SchemaPolicyError("AUDIT_SCHEMA_INVALID")
        values = payload
        try:
            event = AdminDecisionDeniedAuditEvent(
                actor_user_id=values["actor_user_id"],
                actor_role=values["actor_role"],
                schema_version=values["schema_version"],
                event_code=values["event_code"],
                result_code=values["result_code"],
                reason_code=values["reason_code"],
            )
        except (AdminDecisionSecurityAuditError, KeyError):
            raise SchemaPolicyError("AUDIT_SCHEMA_INVALID") from None
        return event.payload()

    def classify(
        self,
        path: tuple[str, ...],
        value: object,
    ) -> FieldProtection:
        del value
        if len(path) == 1 and path[0] in _AUDIT_FIELDS:
            return FieldProtection.EXACT
        return FieldProtection.REJECT

    def validate_protected(self, payload: object) -> None:
        self.normalize(payload)


def build_admin_decision_audit_protection() -> DataProtectionPolicy:
    return DataProtectionPolicy(
        schema_policy=AdminDecisionAuditSchemaPolicy(),
        profile=DataProtectionProfile.OBSERVABILITY,
    )


def normalized_non_admin_role(role: str) -> str:
    return "CUSTOMER" if role == "CUSTOMER" else "OTHER"


__all__ = [
    "ADMIN_DECISION_AUDIT_SCHEMA_VERSION",
    "ADMIN_DECISION_DENIED_EVENT_CODE",
    "ADMIN_DECISION_DENIED_REASON_CODE",
    "ADMIN_DECISION_DENIED_RESULT_CODE",
    "AdminDecisionAuditSchemaPolicy",
    "AdminDecisionDeniedAuditEvent",
    "AdminDecisionSecurityAuditError",
    "AdminDecisionSecurityAuditPort",
    "AdminDecisionSecurityAuditStorePort",
    "ProtectedAdminDecisionAuditWrite",
    "build_admin_decision_audit_protection",
    "normalized_non_admin_role",
]
