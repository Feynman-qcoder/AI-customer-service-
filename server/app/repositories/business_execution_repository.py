"""SQLAlchemy adapter for one-transaction durable business execution."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime

from sqlalchemy import literal_column, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

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
)
from app.runtime.business_execution import (
    BUSINESS_EXECUTION_NODE_NAME,
    BUSINESS_EXECUTION_PURPOSE,
    BUSINESS_EXECUTION_SEQUENCE,
    BusinessExecutionAuditEvent,
    BusinessExecutionConflict,
    BusinessExecutionCorruption,
    BusinessExecutionOutcome,
    BusinessExecutionResult,
    BusinessExecutionTransition,
    BusinessExecutionWrite,
    LockedBusinessExecution,
    ProtectedBusinessExecutionAudit,
    StoredBusinessExecutionAudit,
)
from app.runtime.durable import (
    DURABLE_INTERRUPT_CONFIRMATION_MODE,
    DurableContractError,
    canonical_action_payload_json,
    canonical_effect_idempotency_key,
    durable_action_prepare_replay_digest,
)
from app.runtime.uow import TransactionBoundStoreGuard

_DB_NOW: ColumnElement[datetime] = literal_column("CURRENT_TIMESTAMP(6)")


class SqlAlchemyBusinessExecutionStore:
    """A transaction-bound adapter that never commits or invokes policy."""

    def __init__(
        self,
        session: AsyncSession,
        guard: TransactionBoundStoreGuard,
    ) -> None:
        self._session = session
        self._guard = guard
        self._locked_action: AgentActionRequest | None = None
        self._locked_order: CustomerOrder | None = None
        self._locked_write: BusinessExecutionWrite | None = None

    async def lock_business_execution(
        self,
        write: BusinessExecutionWrite,
    ) -> LockedBusinessExecution:
        self._guard.ensure_active()
        if self._locked_action is not None:
            raise BusinessExecutionCorruption("business execution lock was requested twice")
        await self._lock_live_authority(write)
        action = await self._session.scalar(
            select(AgentActionRequest)
            .where(AgentActionRequest.id == write.action_id)
            .with_for_update()
        )
        if action is None:
            raise BusinessExecutionCorruption("durable action is unavailable")
        run = await self._validate_prepare_and_admin_evidence(action, write)
        del run
        if action.logical_action_id is None or action.target_order_id is None:
            raise BusinessExecutionCorruption("durable action identity is incomplete")

        if action.status == "EXECUTING":
            raise BusinessExecutionCorruption("persisted EXECUTING action is invalid")
        if action.status not in {"APPROVED", "EXECUTED", "STALE"}:
            raise BusinessExecutionCorruption("durable action is not executable")

        order: CustomerOrder | None = None
        if action.status == "APPROVED":
            order = await self._session.scalar(
                select(CustomerOrder)
                .where(CustomerOrder.id == action.target_order_id)
                .with_for_update()
            )
        audit = await self._load_execution_audit(action, write)
        if action.status == "APPROVED" and audit is not None:
            raise BusinessExecutionCorruption(
                "approved action already has execution audit evidence"
            )
        if action.status in {"EXECUTED", "STALE"} and audit is None:
            raise BusinessExecutionCorruption(
                "terminal action is missing execution audit evidence"
            )

        self._locked_action = action
        self._locked_order = order
        self._locked_write = write
        return LockedBusinessExecution(
            action_id=int(action.id),
            run_id=str(action.run_id),
            logical_action_id=action.logical_action_id,
            action_type=str(action.action_type),
            target_order_id=int(action.target_order_id),
            target_order_no=write.draft.target_order_no,
            subject_user_id=write.scope.subject_user_id,
            admin_actor_id=write.scope.admin_actor_id,
            policy_version=write.draft.policy_version,
            action_status=str(action.status),
            resume_status=str(action.resume_status),
            execution_result_code=action.execution_result_code,
            executed_at=action.executed_at,
            order_exists=order is not None,
            current_order_owner_id=(int(order.user_id) if order is not None else None),
            current_order_status=(str(order.status) if order is not None else None),
            order_product_id=(int(order.product_id) if order is not None else None),
            order_quantity=(int(order.quantity) if order is not None else None),
            audit=audit,
        )

    async def apply_business_execution(
        self,
        write: BusinessExecutionWrite,
        locked: LockedBusinessExecution,
        transition: BusinessExecutionTransition,
        audit: ProtectedBusinessExecutionAudit,
    ) -> BusinessExecutionResult:
        self._guard.ensure_active()
        action = self._locked_action
        if (
            action is None
            or self._locked_write != write
            or action.id != locked.action_id
            or action.status != "APPROVED"
            or locked.action_status != "APPROVED"
            or locked.audit is not None
        ):
            raise BusinessExecutionCorruption("business execution claim is invalid")
        self._validate_protected_audit(write, transition, audit)

        action.status = "EXECUTING"
        await self._guard.flush(self._session)
        now = await self._database_time()
        if transition.outcome is BusinessExecutionOutcome.EXECUTED:
            order = self._locked_order
            if (
                order is None
                or order.id != locked.target_order_id
                or order.user_id != locked.subject_user_id
                or str(order.status) != transition.before_status
            ):
                raise BusinessExecutionCorruption(
                    "locked order changed before execution apply"
                )
            if transition.restore_stock:
                product = await self._session.scalar(
                    select(ProductCatalog)
                    .where(ProductCatalog.id == order.product_id)
                    .with_for_update()
                )
                if product is None:
                    raise BusinessExecutionCorruption(
                        "cancellation inventory target is unavailable"
                    )
                product.stock_quantity += order.quantity
                product.updated_at = now
            order.status = transition.after_status
            order.updated_at = now

        effect = AgentEffect(
            run_id=write.scope.run_id,
            attempt_id=write.scope.effect_scope.attempt_id,
            node_name=BUSINESS_EXECUTION_NODE_NAME,
            purpose=BUSINESS_EXECUTION_PURPOSE,
            sequence=BUSINESS_EXECUTION_SEQUENCE,
            effect_type="LOCAL_AUDIT",
            idempotency_key=canonical_effect_idempotency_key(
                run_id=write.scope.run_id,
                node_name=BUSINESS_EXECUTION_NODE_NAME,
                purpose=BUSINESS_EXECUTION_PURPOSE,
                sequence=BUSINESS_EXECUTION_SEQUENCE,
            ),
            payload_digest=audit.payload_digest,
        )
        self._session.add(effect)
        await self._guard.flush(self._session)
        self._session.add(
            AgentStep(
                run_id=write.scope.run_id,
                node_name=BUSINESS_EXECUTION_NODE_NAME,
                input_summary=audit.canonical_json,
                output_summary=None,
                status="COMPLETED",
                duration_ms=0,
                error_summary=None,
                attempt_id=effect.attempt_id,
                effect_id=effect.id,
                effect_purpose=effect.purpose,
                effect_sequence=effect.sequence,
                effect_idempotency_key=effect.idempotency_key,
            )
        )

        if transition.outcome is BusinessExecutionOutcome.EXECUTED:
            action.status = "EXECUTED"
            action.execution_result_code = transition.result_code.value
            action.executed_at = now
        else:
            action.status = "STALE"
            action.execution_result_code = None
            action.executed_at = None
        action.execution_error_type = None
        action.execution_error_summary = None
        action.lock_version += 1
        await self._guard.flush(self._session)
        return BusinessExecutionResult(
            action_id=int(action.id),
            outcome=transition.outcome,
            result_code=transition.result_code,
            reason_code=transition.reason_code,
            before_status=transition.before_status,
            after_status=transition.after_status,
        )

    async def _lock_live_authority(self, write: BusinessExecutionWrite) -> None:
        scope = write.scope
        attempt = await self._session.scalar(
            select(AgentRunAttempt)
            .where(
                AgentRunAttempt.attempt_id == scope.effect_scope.attempt_id,
                AgentRunAttempt.run_id == scope.run_id,
                AgentRunAttempt.thread_id == scope.effect_scope.thread_id,
                AgentRunAttempt.conversation_id == scope.conversation_id,
            )
            .with_for_update()
        )
        if (
            attempt is None
            or attempt.status != "ACTIVE"
            or attempt.fence_version != scope.effect_scope.fence_version
            or attempt.actor_user_id != scope.admin_actor_id
            or attempt.actor_role != "ADMIN"
            or attempt.service_principal is not None
            or attempt.subject_user_id != scope.subject_user_id
        ):
            raise BusinessExecutionConflict("business execution attempt is invalid")
        execution = await self._session.scalar(
            select(AgentThreadExecution)
            .where(
                AgentThreadExecution.thread_id == scope.effect_scope.thread_id,
                AgentThreadExecution.conversation_id == scope.conversation_id,
            )
            .with_for_update()
        )
        if (
            execution is None
            or execution.owner_attempt_id != scope.effect_scope.attempt_id
            or execution.fence_version != scope.effect_scope.fence_version
            or execution.lease_expires_at is None
            or execution.lease_expires_at <= await self._database_time()
        ):
            raise BusinessExecutionConflict("business execution lease is not live")

    async def _validate_prepare_and_admin_evidence(
        self,
        action: AgentActionRequest,
        write: BusinessExecutionWrite,
    ) -> AgentRun:
        draft = write.draft
        scope = write.scope
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
            raise BusinessExecutionCorruption("durable prepare evidence is incomplete")
        run = await self._session.scalar(
            select(AgentRun).where(AgentRun.run_id == action.run_id)
        )
        effect = await self._session.scalar(
            select(AgentEffect).where(AgentEffect.id == action.effect_id)
        )
        if run is None or effect is None:
            raise BusinessExecutionCorruption("durable prepare evidence is orphaned")
        origin = await self._session.scalar(
            select(AgentRunAttempt).where(
                AgentRunAttempt.attempt_id == effect.attempt_id,
                AgentRunAttempt.run_id == effect.run_id,
            )
        )
        if origin is None:
            raise BusinessExecutionCorruption("durable prepare origin is unavailable")
        try:
            payload = json.loads(action.action_payload_json)
            canonical_payload = canonical_action_payload_json(payload)
        except (DurableContractError, TypeError, ValueError, json.JSONDecodeError):
            raise BusinessExecutionCorruption("durable prepare payload is invalid") from None
        expected_payload = canonical_action_payload_json({"reason": draft.reason_code})
        expected_prepare_key = canonical_effect_idempotency_key(
            run_id=action.run_id,
            node_name="durable_action_prepare",
            purpose="ACTION_PREPARE",
            sequence=0,
        )
        expected_prepare_digest = durable_action_prepare_replay_digest(
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
        expected_thread = ThreadIdentity.from_conversation_id(
            scope.conversation_id
        ).thread_id
        if (
            action.run_id != scope.run_id
            or run.conversation_id != scope.conversation_id
            or run.thread_id != expected_thread
            or run.user_id != scope.subject_user_id
            or run.status
            not in {"WAITING_ADMIN_APPROVAL", "RESUME_PENDING", "COMPLETED"}
            or action.logical_action_id != draft.logical_action_id
            or action.action_type != draft.action_type
            or action.target_order_id != draft.target_order_id
            or action.created_by != draft.subject_user_id
            or action.customer_confirmed_actor_id != draft.subject_user_id
            or action.customer_confirmation_challenge_digest != draft.nonce_digest
            or action.action_payload_json != expected_payload
            or action.risk_level != "HIGH"
            or action.idempotency_key != f"durable:{draft.logical_action_id}"
            or effect.run_id != action.run_id
            or effect.attempt_id != action.attempt_id
            or effect.node_name != "durable_action_prepare"
            or effect.purpose != "ACTION_PREPARE"
            or effect.sequence != 0
            or effect.effect_type != "ACTION_PREPARE"
            or effect.idempotency_key != expected_prepare_key
            or effect.payload_digest != expected_prepare_digest
            or action.effect_node_name != effect.node_name
            or action.effect_purpose != effect.purpose
            or action.effect_sequence != effect.sequence
            or action.effect_idempotency_key != expected_prepare_key
            or origin.run_id != action.run_id
            or origin.thread_id != run.thread_id
            or origin.conversation_id != run.conversation_id
            or origin.actor_user_id != action.created_by
            or origin.subject_user_id != scope.subject_user_id
            or origin.actor_role != "CUSTOMER"
            or origin.service_principal is not None
            or action.admin_decision != "APPROVE"
            or action.admin_decided_actor_id != scope.admin_actor_id
            or action.admin_decided_at is None
            or action.admin_reason_code != "ADMIN_APPROVED"
            or action.approved_by != scope.admin_actor_id
            or action.approved_at is None
            or action.execution_error_type is not None
            or action.execution_error_summary is not None
            or action.resume_status
            not in {"RESUME_PENDING", "FAILED_RETRYABLE", "COMPLETED"}
        ):
            raise BusinessExecutionCorruption(
                "durable action evidence does not match the approved draft"
            )
        if action.status == "APPROVED" and (
            action.execution_result_code is not None or action.executed_at is not None
        ):
            raise BusinessExecutionCorruption("approved action has terminal evidence")
        if action.status == "EXECUTED" and (
            action.execution_result_code is None or action.executed_at is None
        ):
            raise BusinessExecutionCorruption("executed action evidence is incomplete")
        if action.status == "STALE" and (
            action.execution_result_code is not None or action.executed_at is not None
        ):
            raise BusinessExecutionCorruption("stale action has execution evidence")
        if action.status == "APPROVED" and action.resume_status == "COMPLETED":
            raise BusinessExecutionCorruption("unexecuted action has completed resume marker")
        return run

    async def _load_execution_audit(
        self,
        action: AgentActionRequest,
        write: BusinessExecutionWrite,
    ) -> StoredBusinessExecutionAudit | None:
        effect = await self._session.scalar(
            select(AgentEffect).where(
                AgentEffect.run_id == action.run_id,
                AgentEffect.node_name == BUSINESS_EXECUTION_NODE_NAME,
                AgentEffect.purpose == BUSINESS_EXECUTION_PURPOSE,
                AgentEffect.sequence == BUSINESS_EXECUTION_SEQUENCE,
            )
        )
        orphan_step = await self._session.scalar(
            select(AgentStep).where(
                AgentStep.run_id == action.run_id,
                AgentStep.node_name == BUSINESS_EXECUTION_NODE_NAME,
                AgentStep.effect_purpose == BUSINESS_EXECUTION_PURPOSE,
                AgentStep.effect_sequence == BUSINESS_EXECUTION_SEQUENCE,
            )
        )
        if effect is None:
            if orphan_step is not None:
                raise BusinessExecutionCorruption("execution audit target is orphaned")
            return None
        step = await self._session.scalar(
            select(AgentStep).where(AgentStep.effect_id == effect.id)
        )
        origin = await self._session.scalar(
            select(AgentRunAttempt).where(
                AgentRunAttempt.attempt_id == effect.attempt_id,
                AgentRunAttempt.run_id == effect.run_id,
            )
        )
        expected_key = canonical_effect_idempotency_key(
            run_id=action.run_id,
            node_name=BUSINESS_EXECUTION_NODE_NAME,
            purpose=BUSINESS_EXECUTION_PURPOSE,
            sequence=BUSINESS_EXECUTION_SEQUENCE,
        )
        if (
            step is None
            or orphan_step is None
            or orphan_step.id != step.id
            or origin is None
            or effect.effect_type != "LOCAL_AUDIT"
            or effect.idempotency_key != expected_key
            or step.run_id != action.run_id
            or step.node_name != BUSINESS_EXECUTION_NODE_NAME
            or step.input_summary is None
            or step.output_summary is not None
            or step.status != "COMPLETED"
            or step.duration_ms != 0
            or step.error_summary is not None
            or step.attempt_id != effect.attempt_id
            or step.effect_id != effect.id
            or step.effect_purpose != effect.purpose
            or step.effect_sequence != effect.sequence
            or step.effect_idempotency_key != effect.idempotency_key
            or origin.thread_id != write.scope.effect_scope.thread_id
            or origin.conversation_id != write.scope.conversation_id
            or origin.actor_role != "ADMIN"
            or origin.actor_user_id != action.admin_decided_actor_id
            or origin.subject_user_id != write.scope.subject_user_id
            or origin.service_principal is not None
        ):
            raise BusinessExecutionCorruption("execution audit evidence is inconsistent")
        return StoredBusinessExecutionAudit(
            effect_id=int(effect.id),
            effect_attempt_id=str(effect.attempt_id),
            idempotency_key=str(effect.idempotency_key),
            payload_digest=str(effect.payload_digest),
            canonical_json=str(step.input_summary),
        )

    def _validate_protected_audit(
        self,
        write: BusinessExecutionWrite,
        transition: BusinessExecutionTransition,
        audit: ProtectedBusinessExecutionAudit,
    ) -> None:
        event = BusinessExecutionAuditEvent.from_canonical_json(audit.canonical_json)
        expected = BusinessExecutionAuditEvent(
            logical_action_id=write.draft.logical_action_id,
            action_type=write.draft.action_type,
            target_order_id=write.draft.target_order_id,
            target_order_no=write.draft.target_order_no,
            subject_user_id=write.scope.subject_user_id,
            admin_decision="APPROVE",
            policy_version=write.draft.policy_version,
            before_status=transition.before_status,
            after_status=transition.after_status,
            result_code=transition.result_code.value,
            reason_code=transition.reason_code.value,
        )
        digest = hashlib.sha256(audit.canonical_json.encode("utf-8")).hexdigest()
        if event != expected or digest != audit.payload_digest:
            raise BusinessExecutionCorruption("protected execution audit is invalid")

    async def _database_time(self) -> datetime:
        value = await self._session.scalar(select(_DB_NOW))
        if not isinstance(value, datetime):
            raise BusinessExecutionConflict("database time is unavailable")
        return value


__all__ = ["SqlAlchemyBusinessExecutionStore"]
