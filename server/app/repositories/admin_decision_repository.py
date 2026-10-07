"""SQLAlchemy adapter for the closed durable-admin-decision port."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime

from sqlalchemy import literal_column, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from app.agent.state import ActionDraftSnapshot
from app.agent.thread_identity import ThreadIdentity
from app.db.models import (
    AgentActionRequest,
    AgentEffect,
    AgentRun,
    AgentRunAttempt,
    AgentStep,
    AgentThreadExecution,
    CustomerOrder,
    ProductCatalog,
    UserAccount,
)
from app.runtime.admin_decision import (
    AdminActionRecord,
    AdminDecisionPersistenceConflict,
    AdminDecisionPersistenceCorruption,
    AdminDecisionPersistenceNotFound,
    AdminDecisionValue,
    AdminResumeScope,
    DurableAdminDecisionWrite,
    DurableAdminResumeWrite,
    R2AdminDecisionWrite,
    R2LockedAction,
)
from app.runtime.admin_decision_audit import (
    AdminDecisionDeniedAuditEvent,
    AdminDecisionSecurityAuditError,
    ProtectedAdminDecisionAuditWrite,
)
from app.runtime.durable import (
    DURABLE_INTERRUPT_CONFIRMATION_MODE,
    R2_STATELESS_COMPAT_CONFIRMATION_MODE,
    DurableContractError,
    canonical_action_payload_json,
    canonical_effect_idempotency_key,
    durable_action_prepare_replay_digest,
)
from app.runtime.uow import TransactionBoundStoreGuard

_DB_NOW: ColumnElement[datetime] = literal_column("CURRENT_TIMESTAMP(6)")


class SqlAlchemyAdminDecisionStore:
    """One transaction-bound adapter; it never commits or exposes ORM rows."""

    def __init__(
        self,
        session: AsyncSession,
        guard: TransactionBoundStoreGuard,
    ) -> None:
        self._session = session
        self._guard = guard

    async def resolve_admin_action(self, action_id: int) -> AdminActionRecord | None:
        self._guard.ensure_active()
        action = await self._session.get(AgentActionRequest, action_id)
        if action is None:
            return None
        return await self._record(action)

    async def apply_durable_admin_decision(
        self,
        write: DurableAdminDecisionWrite,
    ) -> AdminActionRecord:
        self._guard.ensure_active()
        action = await self._lock_action(write.action_id)
        await self._require_live_admin_authority(write.scope)
        await self._validate_durable_prepare_evidence(action, write.draft)
        await self._require_admin_actor(write.scope.admin_actor_id)

        if action.admin_decision is not None:
            self._validate_decided_action(action)
            if action.admin_decision != write.decision.value:
                raise AdminDecisionPersistenceConflict(
                    "the durable action already has the opposite admin decision"
                )
            return await self._record(action)

        self._validate_undecided_action(action)
        if action.lock_version != write.expected_lock_version:
            raise AdminDecisionPersistenceConflict(
                "the durable action lock version changed before the first decision"
            )
        now = await self._database_time()
        action.admin_decision = write.decision.value
        action.admin_decided_actor_id = write.scope.admin_actor_id
        action.admin_decided_at = now
        action.admin_reason_code = (
            "ADMIN_APPROVED"
            if write.decision is AdminDecisionValue.APPROVE
            else "ADMIN_REJECTED"
        )
        action.approved_by = write.scope.admin_actor_id
        action.approved_at = now
        action.approval_note = write.approval_note
        action.status = (
            "APPROVED"
            if write.decision is AdminDecisionValue.APPROVE
            else "REJECTED"
        )
        action.resume_status = "RESUME_PENDING"
        action.lock_version += 1
        await self._guard.flush(self._session)
        return await self._record(action)

    async def validate_durable_admin_resume(
        self,
        write: DurableAdminResumeWrite,
    ) -> AdminActionRecord:
        self._guard.ensure_active()
        action = await self._lock_action(write.action_id)
        await self._require_live_admin_authority(write.scope)
        await self._validate_durable_prepare_evidence(action, write.draft)
        self._validate_decided_action(action)
        self._require_matching_decision(action, write)
        return await self._record(action)

    async def mark_durable_admin_resume_succeeded(
        self,
        write: DurableAdminResumeWrite,
    ) -> AdminActionRecord:
        self._guard.ensure_active()
        action = await self._lock_action(write.action_id)
        await self._require_live_admin_authority(write.scope)
        run = await self._validate_durable_prepare_evidence(action, write.draft)
        self._validate_decided_action(action)
        self._require_matching_decision(action, write)
        if write.decision is AdminDecisionValue.APPROVE:
            if action.status not in {"EXECUTED", "STALE"}:
                raise AdminDecisionPersistenceConflict(
                    "approved durable action has not reached a business terminal"
                )
            target_resume_status = "COMPLETED"
            target_run_status = "COMPLETED"
        else:
            target_resume_status = "COMPLETED"
            target_run_status = "REJECTED"
        if action.resume_status == target_resume_status:
            if run.status != target_run_status:
                raise AdminDecisionPersistenceCorruption(
                    "durable action and run resume markers disagree"
                )
            return await self._record(action)
        if action.resume_status not in {"RESUME_PENDING", "FAILED_RETRYABLE"}:
            raise AdminDecisionPersistenceConflict(
                "durable action cannot complete resume from its current marker"
            )
        action.resume_status = target_resume_status
        action.lock_version += 1
        run.status = target_run_status
        if write.decision is AdminDecisionValue.REJECT or target_run_status == "COMPLETED":
            run.completed_at = await self._database_time()
        await self._guard.flush(self._session)
        return await self._record(action)

    async def mark_durable_admin_resume_failed(
        self,
        write: DurableAdminResumeWrite,
    ) -> AdminActionRecord:
        self._guard.ensure_active()
        action = await self._lock_action(write.action_id)
        await self._require_live_admin_authority(write.scope)
        await self._validate_durable_prepare_evidence(action, write.draft)
        self._validate_decided_action(action)
        self._require_matching_decision(action, write)
        if action.resume_status in {"RESUMED", "COMPLETED"}:
            return await self._record(action)
        if action.resume_status == "RESUME_PENDING":
            action.resume_status = "FAILED_RETRYABLE"
            action.lock_version += 1
            await self._guard.flush(self._session)
        elif action.resume_status != "FAILED_RETRYABLE":
            raise AdminDecisionPersistenceConflict(
                "durable action cannot enter retryable resume state"
            )
        return await self._record(action)

    async def list_durable_admin_resume_candidates(
        self,
        *,
        limit: int,
    ) -> tuple[int, ...]:
        self._guard.ensure_active()
        if type(limit) is not int or not 0 < limit <= 100:
            raise AdminDecisionPersistenceConflict(
                "durable admin resume scan limit is invalid"
            )
        values = await self._session.scalars(
            select(AgentActionRequest.id)
            .where(
                AgentActionRequest.confirmation_mode
                == DURABLE_INTERRUPT_CONFIRMATION_MODE,
                AgentActionRequest.admin_decision.is_not(None),
                AgentActionRequest.resume_status.in_(
                    ("RESUME_PENDING", "FAILED_RETRYABLE")
                ),
            )
            .order_by(AgentActionRequest.id.asc())
            .limit(limit)
        )
        return tuple(int(value) for value in values.all())

    async def lock_r2_admin_action(
        self,
        write: R2AdminDecisionWrite,
    ) -> R2LockedAction:
        self._guard.ensure_active()
        action = await self._lock_action(write.action_id)
        await self._require_admin_actor(write.admin_actor_id)
        if action.lock_version != write.expected_lock_version:
            raise AdminDecisionPersistenceConflict(
                "the R2 action lock version changed"
            )
        if (
            action.confirmation_mode != R2_STATELESS_COMPAT_CONFIRMATION_MODE
            or action.logical_action_id is not None
            or action.customer_confirmed_actor_id is not None
            or action.customer_confirmed_at is not None
            or action.customer_confirmation_challenge_digest is not None
            or action.legacy_original_status is not None
            or action.resume_status != "NOT_APPLICABLE"
            or action.prepared_order_status is not None
        ):
            raise AdminDecisionPersistenceConflict(
                "the action cannot use the R2 compatibility approval path"
            )
        if action.status != "PENDING" or action.admin_decision is not None:
            raise AdminDecisionPersistenceConflict(
                "the R2 action is not awaiting a decision"
            )
        if write.decision is AdminDecisionValue.REJECT:
            return R2LockedAction(
                action_id=int(action.id),
                action_type=str(action.action_type),
                order_status=None,
            )
        if action.target_order_id is None:
            raise AdminDecisionPersistenceConflict(
                "the R2 action has no target order"
            )
        order = await self._session.scalar(
            select(CustomerOrder)
            .where(CustomerOrder.id == action.target_order_id)
            .with_for_update()
        )
        if order is None:
            raise AdminDecisionPersistenceConflict(
                "the R2 target order is unavailable"
            )
        return R2LockedAction(
            action_id=int(action.id),
            action_type=str(action.action_type),
            order_status=str(order.status),
        )

    async def apply_r2_admin_decision(
        self,
        write: R2AdminDecisionWrite,
        *,
        next_order_status: str | None,
        restore_stock: bool,
        execution_summary: str | None,
    ) -> AdminActionRecord:
        self._guard.ensure_active()
        action = await self._lock_action(write.action_id)
        if (
            action.lock_version != write.expected_lock_version
            or action.status != "PENDING"
            or action.admin_decision is not None
            or action.confirmation_mode != R2_STATELESS_COMPAT_CONFIRMATION_MODE
        ):
            raise AdminDecisionPersistenceConflict(
                "the R2 action changed before its decision"
            )
        now = await self._database_time()
        if write.decision is AdminDecisionValue.APPROVE:
            if action.target_order_id is None or next_order_status is None:
                raise AdminDecisionPersistenceConflict(
                    "the R2 approval transition is incomplete"
                )
            order = await self._session.scalar(
                select(CustomerOrder)
                .where(CustomerOrder.id == action.target_order_id)
                .with_for_update()
            )
            if order is None:
                raise AdminDecisionPersistenceConflict(
                    "the R2 target order is unavailable"
                )
            order.status = next_order_status
            order.updated_at = now
            if restore_stock:
                product = await self._session.scalar(
                    select(ProductCatalog)
                    .where(ProductCatalog.id == order.product_id)
                    .with_for_update()
                )
                if product is not None:
                    product.stock_quantity += order.quantity
                    product.updated_at = now
            action.status = "EXECUTED"
            action.admin_decision = "APPROVE"
            action.admin_reason_code = "R2_ADMIN_APPROVED"
            action.execution_result_code = "R2_ACTION_EXECUTED"
            action.executed_at = now
            action.approval_note = _merge_approval_note(
                write.approval_note,
                execution_summary,
            )
        else:
            action.status = "REJECTED"
            action.admin_decision = "REJECT"
            action.admin_reason_code = "R2_ADMIN_REJECTED"
            action.approval_note = write.approval_note
        action.admin_decided_actor_id = write.admin_actor_id
        action.admin_decided_at = now
        action.approved_by = write.admin_actor_id
        action.approved_at = now
        action.lock_version += 1
        await self._guard.flush(self._session)
        return await self._record(action)

    async def _lock_action(self, action_id: int) -> AgentActionRequest:
        if type(action_id) is not int or action_id <= 0:
            raise AdminDecisionPersistenceNotFound(
                "the action identity is invalid"
            )
        action = await self._session.scalar(
            select(AgentActionRequest)
            .where(AgentActionRequest.id == action_id)
            .with_for_update()
        )
        if action is None:
            raise AdminDecisionPersistenceNotFound(
                "the action does not exist"
            )
        return action

    async def _require_admin_actor(self, actor_id: int) -> UserAccount:
        actor = await self._session.scalar(
            select(UserAccount).where(UserAccount.id == actor_id)
        )
        if (
            actor is None
            or actor.role != "ADMIN"
            or actor.status != "ACTIVE"
        ):
            raise AdminDecisionPersistenceConflict(
                "the current actor has no admin authority"
            )
        return actor

    async def _require_live_admin_authority(self, scope: AdminResumeScope) -> None:
        await self._require_admin_actor(scope.admin_actor_id)
        current_attempt = await self._session.scalar(
            select(AgentRunAttempt).where(
                AgentRunAttempt.attempt_id == scope.effect_scope.attempt_id,
                AgentRunAttempt.run_id == scope.run_id,
                AgentRunAttempt.thread_id == scope.effect_scope.thread_id,
                AgentRunAttempt.conversation_id == scope.conversation_id,
                AgentRunAttempt.fence_version
                == scope.effect_scope.fence_version,
            )
        )
        if (
            current_attempt is None
            or current_attempt.actor_user_id != scope.admin_actor_id
            or current_attempt.actor_role != "ADMIN"
            or current_attempt.service_principal is not None
            or current_attempt.subject_user_id != scope.subject_user_id
        ):
            raise AdminDecisionPersistenceConflict(
                "the admin resume attempt authority is invalid"
            )
        live = await self._session.scalar(
            select(AgentThreadExecution.thread_id).where(
                AgentThreadExecution.thread_id == scope.effect_scope.thread_id,
                AgentThreadExecution.conversation_id == scope.conversation_id,
                AgentThreadExecution.owner_attempt_id
                == scope.effect_scope.attempt_id,
                AgentThreadExecution.fence_version
                == scope.effect_scope.fence_version,
                AgentThreadExecution.lease_expires_at.is_not(None),
                AgentThreadExecution.lease_expires_at > _DB_NOW,
            )
        )
        if live is None:
            raise AdminDecisionPersistenceConflict(
                "the admin resume attempt lost its live lease"
            )

    async def _validate_durable_prepare_evidence(
        self,
        action: AgentActionRequest,
        draft: ActionDraftSnapshot,
    ) -> AgentRun:
        if (
            action.confirmation_mode != DURABLE_INTERRUPT_CONFIRMATION_MODE
            or action.logical_action_id is None
            or action.target_order_id is None
            or action.prepared_order_status is None
            or action.customer_confirmed_actor_id is None
            or action.customer_confirmed_at is None
            or action.customer_confirmation_challenge_digest is None
            or action.effect_id is None
            or action.attempt_id is None
            or action.effect_node_name is None
            or action.effect_purpose is None
            or action.effect_sequence is None
            or action.effect_idempotency_key is None
            or action.legacy_original_status is not None
        ):
            raise AdminDecisionPersistenceCorruption(
                "durable prepare evidence is incomplete"
            )
        run = await self._session.scalar(
            select(AgentRun).where(AgentRun.run_id == action.run_id)
        )
        effect = await self._session.scalar(
            select(AgentEffect).where(AgentEffect.id == action.effect_id)
        )
        if run is None or effect is None:
            raise AdminDecisionPersistenceCorruption(
                "durable prepare evidence is orphaned"
            )
        origin = await self._session.scalar(
            select(AgentRunAttempt).where(
                AgentRunAttempt.attempt_id == effect.attempt_id,
                AgentRunAttempt.run_id == effect.run_id,
            )
        )
        if origin is None:
            raise AdminDecisionPersistenceCorruption(
                "durable prepare origin attempt is missing"
            )
        expected_thread = ThreadIdentity.from_conversation_id(
            run.conversation_id
        ).thread_id
        try:
            payload = json.loads(action.action_payload_json)
            canonical_payload = canonical_action_payload_json(payload)
        except (DurableContractError, TypeError, ValueError, json.JSONDecodeError):
            raise AdminDecisionPersistenceCorruption(
                "durable prepare payload is invalid"
            ) from None
        expected_payload = canonical_action_payload_json(
            {"reason": draft.reason_code}
        )
        expected_key = canonical_effect_idempotency_key(
            run_id=action.run_id,
            node_name="durable_action_prepare",
            purpose="ACTION_PREPARE",
            sequence=0,
        )
        expected_digest = durable_action_prepare_replay_digest(
            logical_action_id=action.logical_action_id,
            action_type=action.action_type,
            target_order_id=action.target_order_id,
            target_order_no=draft.target_order_no,
            subject_user_id=draft.subject_user_id,
            created_by=action.created_by,
            action_payload_json=canonical_payload,
            risk_level=action.risk_level,
            validated_order_status=action.prepared_order_status,
            policy_version=draft.policy_version,
            draft_revision=draft.draft_revision,
            draft_expires_at=draft.expires_at,
            customer_confirmation_challenge_digest=draft.nonce_digest,
        )
        if (
            run.thread_id != expected_thread
            or run.user_id != draft.subject_user_id
            or run.status
            not in {
                "WAITING_ADMIN_APPROVAL",
                "RESUME_PENDING",
                "COMPLETED",
                "REJECTED",
            }
            or action.logical_action_id != draft.logical_action_id
            or action.action_type != draft.action_type
            or action.target_order_id != draft.target_order_id
            or action.created_by != draft.subject_user_id
            or action.customer_confirmed_actor_id != draft.subject_user_id
            or action.customer_confirmation_challenge_digest
            != draft.nonce_digest
            or action.action_payload_json != expected_payload
            or action.risk_level != "HIGH"
            or action.idempotency_key != f"durable:{draft.logical_action_id}"
            or effect.run_id != action.run_id
            or effect.attempt_id != action.attempt_id
            or effect.node_name != "durable_action_prepare"
            or effect.purpose != "ACTION_PREPARE"
            or effect.sequence != 0
            or effect.effect_type != "ACTION_PREPARE"
            or effect.idempotency_key != expected_key
            or effect.payload_digest != expected_digest
            or action.effect_node_name != effect.node_name
            or action.effect_purpose != effect.purpose
            or action.effect_sequence != effect.sequence
            or action.effect_idempotency_key != expected_key
            or origin.attempt_id != effect.attempt_id
            or origin.run_id != effect.run_id
            or origin.thread_id != run.thread_id
            or origin.conversation_id != run.conversation_id
            or origin.actor_user_id != action.created_by
            or origin.subject_user_id != draft.subject_user_id
            or origin.actor_role != "CUSTOMER"
            or origin.service_principal is not None
        ):
            raise AdminDecisionPersistenceCorruption(
                "durable prepare evidence does not match its frozen draft"
            )
        return run

    def _validate_undecided_action(self, action: AgentActionRequest) -> None:
        if (
            action.status != "PENDING"
            or action.resume_status != "WAITING_ADMIN_DECISION"
            or any(
                value is not None
                for value in (
                    action.admin_decision,
                    action.admin_decided_actor_id,
                    action.admin_decided_at,
                    action.admin_reason_code,
                    action.approved_by,
                    action.approved_at,
                    action.approval_note,
                    action.execution_result_code,
                    action.execution_error_type,
                    action.execution_error_summary,
                    action.executed_at,
                )
            )
        ):
            raise AdminDecisionPersistenceCorruption(
                "durable action is not at its first decision boundary"
            )

    def _validate_decided_action(self, action: AgentActionRequest) -> None:
        if (
            action.admin_decision not in {"APPROVE", "REJECT"}
            or action.admin_decided_actor_id is None
            or action.admin_decided_at is None
            or action.admin_reason_code is None
            or action.approved_by != action.admin_decided_actor_id
            or action.approved_at is None
            or action.resume_status
            not in {
                "RESUME_PENDING",
                "FAILED_RETRYABLE",
                "RESUMED",
                "COMPLETED",
            }
            or action.execution_error_type is not None
            or action.execution_error_summary is not None
        ):
            raise AdminDecisionPersistenceCorruption(
                "durable admin decision evidence is invalid"
            )
        if action.admin_decision == "REJECT":
            valid = (
                action.status == "REJECTED"
                and action.admin_reason_code == "ADMIN_REJECTED"
                and action.execution_result_code is None
                and action.executed_at is None
            )
        elif action.status == "APPROVED":
            valid = (
                action.admin_reason_code == "ADMIN_APPROVED"
                and action.execution_result_code is None
                and action.executed_at is None
            )
        elif action.status == "EXECUTED":
            valid = (
                action.admin_reason_code == "ADMIN_APPROVED"
                and action.execution_result_code is not None
                and action.executed_at is not None
            )
        elif action.status == "STALE":
            valid = (
                action.admin_reason_code == "ADMIN_APPROVED"
                and action.execution_result_code is None
                and action.executed_at is None
            )
        else:
            valid = False
        if not valid:
            raise AdminDecisionPersistenceCorruption(
                "durable admin decision status is inconsistent"
            )

    def _require_matching_decision(
        self,
        action: AgentActionRequest,
        write: DurableAdminResumeWrite,
    ) -> None:
        if (
            action.admin_decision != write.decision.value
        ):
            raise AdminDecisionPersistenceConflict(
                "admin resume does not match the persisted first decision"
            )

    async def _database_time(self) -> datetime:
        now = await self._session.scalar(select(_DB_NOW))
        if not isinstance(now, datetime):
            raise AdminDecisionPersistenceConflict(
                "database time is unavailable"
            )
        return now

    async def _record(self, action: AgentActionRequest) -> AdminActionRecord:
        run = await self._session.scalar(
            select(AgentRun).where(AgentRun.run_id == action.run_id)
        )
        if run is None:
            raise AdminDecisionPersistenceCorruption(
                "action run is unavailable"
            )
        decider: UserAccount | None = None
        if action.admin_decided_actor_id is not None:
            decider = await self._session.get(
                UserAccount,
                action.admin_decided_actor_id,
            )
            if decider is None:
                raise AdminDecisionPersistenceCorruption(
                    "admin decision actor is unavailable"
                )
        return AdminActionRecord(
            id=int(action.id),
            run_id=str(action.run_id),
            action_type=str(action.action_type),
            target_order_id=(
                int(action.target_order_id)
                if action.target_order_id is not None
                else None
            ),
            action_payload_json=str(action.action_payload_json),
            risk_level=str(action.risk_level),
            status=str(action.status),
            idempotency_key=str(action.idempotency_key),
            lock_version=int(action.lock_version),
            created_by=int(action.created_by),
            approved_by=(
                int(action.approved_by)
                if action.approved_by is not None
                else None
            ),
            approval_note=action.approval_note,
            logical_action_id=action.logical_action_id,
            confirmation_mode=str(action.confirmation_mode),
            customer_confirmed_actor_id=(
                int(action.customer_confirmed_actor_id)
                if action.customer_confirmed_actor_id is not None
                else None
            ),
            customer_confirmed_at=action.customer_confirmed_at,
            customer_confirmation_challenge_digest=(
                action.customer_confirmation_challenge_digest
            ),
            admin_decision=action.admin_decision,
            admin_decided_actor_id=(
                int(action.admin_decided_actor_id)
                if action.admin_decided_actor_id is not None
                else None
            ),
            admin_decided_at=action.admin_decided_at,
            admin_reason_code=action.admin_reason_code,
            resume_status=str(action.resume_status),
            execution_result_code=action.execution_result_code,
            execution_error_type=action.execution_error_type,
            execution_error_summary=action.execution_error_summary,
            legacy_original_status=action.legacy_original_status,
            created_at=action.created_at,
            approved_at=action.approved_at,
            executed_at=action.executed_at,
            conversation_id=int(run.conversation_id),
            thread_id=str(run.thread_id),
            subject_user_id=int(run.user_id),
            run_status=str(run.status),
            decider_username=(str(decider.username) if decider else None),
            decider_display_name=(
                str(decider.display_name) if decider else None
            ),
            decider_role=(str(decider.role) if decider else None),
            decider_status=(str(decider.status) if decider else None),
        )


class SqlAlchemyAdminDecisionSecurityAuditStore:
    """Persist one protected denial event without reading action state."""

    def __init__(
        self,
        session: AsyncSession,
        guard: TransactionBoundStoreGuard,
    ) -> None:
        self._session = session
        self._guard = guard

    async def append_admin_decision_denial(
        self,
        write: ProtectedAdminDecisionAuditWrite,
    ) -> None:
        self._guard.ensure_active()
        event = AdminDecisionDeniedAuditEvent(
            actor_user_id=write.actor_user_id,
            actor_role=write.actor_role,
            event_code=write.event_code,
            result_code=write.result_code,
            reason_code=write.reason_code,
        )
        try:
            payload = json.loads(write.canonical_json)
            canonical = json.dumps(
                event.payload(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            raise AdminDecisionSecurityAuditError(
                "protected security audit payload is invalid"
            ) from None
        if (
            payload != event.payload()
            or canonical != write.canonical_json
            or hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            != write.payload_digest
        ):
            raise AdminDecisionSecurityAuditError(
                "protected security audit binding is invalid"
            )
        now = await self._database_time()
        self._session.add(
            AgentStep(
                run_id=f"security-audit-actor-{write.actor_user_id}",
                node_name=write.event_code,
                input_summary=write.canonical_json,
                output_summary=write.result_code,
                status=write.result_code,
                duration_ms=0,
                error_summary=write.reason_code,
                created_at=now,
            )
        )
        await self._guard.flush(self._session)

    async def _database_time(self) -> datetime:
        now = await self._session.scalar(select(_DB_NOW))
        if not isinstance(now, datetime):
            raise AdminDecisionSecurityAuditError(
                "security audit database time is unavailable"
            )
        return now


def _merge_approval_note(
    note: str | None,
    execution_summary: str | None,
) -> str | None:
    if execution_summary is None:
        return note
    if note:
        return f"{note}\n执行结果：{execution_summary}"
    return f"执行结果：{execution_summary}"
__all__ = [
    "SqlAlchemyAdminDecisionSecurityAuditStore",
    "SqlAlchemyAdminDecisionStore",
]
