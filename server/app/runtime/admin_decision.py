"""Closed DTOs and narrow persistence port for durable admin decisions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from app.agent.state import ActionDraftSnapshot
from app.runtime.durable import EffectWriteScope


class AdminDecisionValue(StrEnum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"


def is_admin_decision_available(
    *,
    status: str,
    confirmation_mode: str,
    resume_status: str,
    admin_decision: str | None,
) -> bool:
    """Return the only two server-authorized admin-decision combinations."""

    if admin_decision is not None or status != "PENDING":
        return False
    return (
        confirmation_mode == "DURABLE_INTERRUPT"
        and resume_status == "WAITING_ADMIN_DECISION"
    ) or (
        confirmation_mode == "R2_STATELESS_COMPAT"
        and resume_status == "NOT_APPLICABLE"
    )


class AdminDecisionPersistenceError(RuntimeError):
    """Sanitized persistence invariant failure."""


class AdminDecisionPersistenceNotFound(AdminDecisionPersistenceError):
    """The requested action does not exist."""


class AdminDecisionPersistenceConflict(AdminDecisionPersistenceError):
    """The request conflicts with the first durable decision or lock version."""


class AdminDecisionPersistenceCorruption(AdminDecisionPersistenceError):
    """Persisted prepare/decision evidence is incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class AdminActionRecord:
    id: int
    run_id: str
    action_type: str
    target_order_id: int | None
    action_payload_json: str
    risk_level: str
    status: str
    idempotency_key: str
    lock_version: int
    created_by: int
    approved_by: int | None
    approval_note: str | None
    logical_action_id: str | None
    confirmation_mode: str
    customer_confirmed_actor_id: int | None
    customer_confirmed_at: datetime | None
    customer_confirmation_challenge_digest: str | None
    admin_decision: str | None
    admin_decided_actor_id: int | None
    admin_decided_at: datetime | None
    admin_reason_code: str | None
    resume_status: str
    execution_result_code: str | None
    execution_error_type: str | None
    execution_error_summary: str | None
    legacy_original_status: str | None
    created_at: datetime
    approved_at: datetime | None
    executed_at: datetime | None
    conversation_id: int
    thread_id: str
    subject_user_id: int
    run_status: str
    decider_username: str | None = None
    decider_display_name: str | None = None
    decider_role: str | None = None
    decider_status: str | None = None


@dataclass(frozen=True, slots=True)
class AdminResumeScope:
    effect_scope: EffectWriteScope
    conversation_id: int
    run_id: str
    admin_actor_id: int
    subject_user_id: int


@dataclass(frozen=True, slots=True)
class DurableAdminDecisionWrite:
    scope: AdminResumeScope
    action_id: int
    expected_lock_version: int
    decision: AdminDecisionValue
    approval_note: str | None
    draft: ActionDraftSnapshot


@dataclass(frozen=True, slots=True)
class DurableAdminResumeWrite:
    scope: AdminResumeScope
    action_id: int
    decision: AdminDecisionValue
    draft: ActionDraftSnapshot


@dataclass(frozen=True, slots=True)
class R2LockedAction:
    action_id: int
    action_type: str
    order_status: str | None


@dataclass(frozen=True, slots=True)
class R2AdminDecisionWrite:
    action_id: int
    expected_lock_version: int
    admin_actor_id: int
    decision: AdminDecisionValue
    approval_note: str | None


class AdminDecisionStorePort(Protocol):
    async def resolve_admin_action(self, action_id: int) -> AdminActionRecord | None: ...

    async def apply_durable_admin_decision(
        self,
        write: DurableAdminDecisionWrite,
    ) -> AdminActionRecord: ...

    async def validate_durable_admin_resume(
        self,
        write: DurableAdminResumeWrite,
    ) -> AdminActionRecord: ...

    async def mark_durable_admin_resume_succeeded(
        self,
        write: DurableAdminResumeWrite,
    ) -> AdminActionRecord: ...

    async def mark_durable_admin_resume_failed(
        self,
        write: DurableAdminResumeWrite,
    ) -> AdminActionRecord: ...

    async def list_durable_admin_resume_candidates(
        self,
        *,
        limit: int,
    ) -> tuple[int, ...]: ...

    async def lock_r2_admin_action(
        self,
        write: R2AdminDecisionWrite,
    ) -> R2LockedAction: ...

    async def apply_r2_admin_decision(
        self,
        write: R2AdminDecisionWrite,
        *,
        next_order_status: str | None,
        restore_stock: bool,
        execution_summary: str | None,
    ) -> AdminActionRecord: ...


__all__ = [
    "AdminActionRecord",
    "AdminDecisionPersistenceConflict",
    "AdminDecisionPersistenceCorruption",
    "AdminDecisionPersistenceError",
    "AdminDecisionPersistenceNotFound",
    "AdminDecisionStorePort",
    "AdminDecisionValue",
    "AdminResumeScope",
    "DurableAdminDecisionWrite",
    "DurableAdminResumeWrite",
    "R2AdminDecisionWrite",
    "R2LockedAction",
    "is_admin_decision_available",
]
