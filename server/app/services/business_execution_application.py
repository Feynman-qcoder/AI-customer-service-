"""Application orchestration for durable at-most-once business execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from app.agent.state import ActionDraftSnapshot
from app.runtime.business_execution import (
    BUSINESS_EXECUTION_UNAVAILABLE_STATUS,
    BusinessExecutionAuditEvent,
    BusinessExecutionConflict,
    BusinessExecutionCorruption,
    BusinessExecutionOutcome,
    BusinessExecutionReasonCode,
    BusinessExecutionResult,
    BusinessExecutionResultCode,
    BusinessExecutionScope,
    BusinessExecutionStorePort,
    BusinessExecutionTransition,
    BusinessExecutionWrite,
    LockedBusinessExecution,
    ProtectedBusinessExecutionAudit,
    decode_business_execution_audit_v1,
)
from app.runtime.context import ExecutionScope
from app.runtime.data_protection import DataProtectionPort, DataProtectionProfile
from app.runtime.durable import EffectWriteScope
from app.runtime.uow import (
    ApplicationTransactionCoordinator,
    ApplicationUnitOfWorkFactory,
    CommitOutcomeUnknown,
    UnitOfWorkOperation,
    bind_application_unit_of_work_factory,
)
from app.services.action_execution_service import (
    ActionExecutionError,
    resolve_order_action_transition,
)
from app.services.side_effect_policy_service import POLICY_VERSION


@dataclass(frozen=True, slots=True)
class _ExecutionLifecycleGuard:
    execution: ExecutionScope

    def ensure_active(self) -> None:
        self.execution.require_attempt_active()


class DurableBusinessExecutionApplication:
    """Resolve policy synchronously between lock and apply in one UoW."""

    def __init__(
        self,
        unit_of_work: ApplicationUnitOfWorkFactory[BusinessExecutionStorePort],
        transactions: ApplicationTransactionCoordinator,
        data_protection: DataProtectionPort,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._transactions = transactions
        self._data_protection = data_protection

    async def execute(
        self,
        *,
        execution: ExecutionScope,
        draft: ActionDraftSnapshot,
        pending_action_id: int,
    ) -> BusinessExecutionResult:
        execution.require_attempt_active()
        write = self._write(execution, draft, pending_action_id)
        factory = self._bound_factory(execution)

        async def transact(store: BusinessExecutionStorePort) -> BusinessExecutionResult:
            locked = await store.lock_business_execution(write)
            if locked.action_status in {"EXECUTED", "STALE"}:
                return self._validate_terminal(locked)
            transition = self._resolve_transition(locked)
            protected = self._protect_audit(write, transition)
            return await store.apply_business_execution(
                write,
                locked,
                transition,
                protected,
            )

        try:
            result = await self._transactions.run(
                UnitOfWorkOperation.AGENT_BUSINESS_EXECUTE,
                factory,
                transact,
            )
        except CommitOutcomeUnknown as unknown:
            locked = await self._read_locked(factory, write)
            if locked.action_status not in {"EXECUTED", "STALE"}:
                raise unknown
            result = self._validate_terminal(locked)
        execution.require_attempt_active()
        return result

    async def verify_terminal(
        self,
        *,
        execution: ExecutionScope,
        draft: ActionDraftSnapshot,
        pending_action_id: int,
        expected_outcome: BusinessExecutionOutcome | None = None,
    ) -> BusinessExecutionResult:
        execution.require_attempt_active()
        write = self._write(execution, draft, pending_action_id)
        locked = await self._read_locked(self._bound_factory(execution), write)
        if locked.action_status not in {"EXECUTED", "STALE"}:
            raise BusinessExecutionCorruption(
                "business execution has no exact terminal result"
            )
        result = self._validate_terminal(locked)
        if expected_outcome is not None and result.outcome is not expected_outcome:
            raise BusinessExecutionCorruption(
                "checkpoint and MySQL business outcomes disagree"
            )
        execution.require_attempt_active()
        return result

    async def _read_locked(
        self,
        factory: ApplicationUnitOfWorkFactory[BusinessExecutionStorePort],
        write: BusinessExecutionWrite,
    ) -> LockedBusinessExecution:
        async def transact(
            store: BusinessExecutionStorePort,
        ) -> LockedBusinessExecution:
            return await store.lock_business_execution(write)

        return await self._transactions.run(
            UnitOfWorkOperation.AGENT_BUSINESS_EXECUTION_VERIFY,
            factory,
            transact,
        )

    def _bound_factory(
        self,
        execution: ExecutionScope,
    ) -> ApplicationUnitOfWorkFactory[BusinessExecutionStorePort]:
        return cast(
            ApplicationUnitOfWorkFactory[BusinessExecutionStorePort],
            bind_application_unit_of_work_factory(
                self._unit_of_work,
                _ExecutionLifecycleGuard(execution),
            ),
        )

    def _write(
        self,
        execution: ExecutionScope,
        draft: ActionDraftSnapshot,
        pending_action_id: int,
    ) -> BusinessExecutionWrite:
        if execution.actor.role != "ADMIN":
            raise BusinessExecutionConflict(
                "business execution requires admin resume authority"
            )
        fence = execution.lease.fence_token
        if type(fence) is not int or fence <= 0:
            raise BusinessExecutionConflict(
                "business execution requires a positive live fence"
            )
        if type(pending_action_id) is not int or pending_action_id <= 0:
            raise BusinessExecutionCorruption("business action identity is invalid")
        if draft.subject_user_id != execution.subject.user_id:
            raise BusinessExecutionCorruption(
                "business action subject does not match execution"
            )
        return BusinessExecutionWrite(
            scope=BusinessExecutionScope(
                effect_scope=EffectWriteScope(
                    thread_id=execution.thread_id,
                    attempt_id=execution.attempt_id,
                    fence_version=fence,
                ),
                conversation_id=execution.conversation_id,
                run_id=execution.run_id,
                admin_actor_id=execution.actor.user_id,
                subject_user_id=execution.subject.user_id,
            ),
            action_id=pending_action_id,
            draft=draft,
        )

    def _resolve_transition(
        self,
        locked: LockedBusinessExecution,
    ) -> BusinessExecutionTransition:
        if locked.policy_version != POLICY_VERSION:
            return self._stale(
                locked,
                BusinessExecutionReasonCode.POLICY_VERSION_INCOMPATIBLE,
            )
        if not locked.order_exists or locked.current_order_status is None:
            return BusinessExecutionTransition(
                outcome=BusinessExecutionOutcome.STALE,
                result_code=BusinessExecutionResultCode.STALE,
                reason_code=BusinessExecutionReasonCode.ORDER_MISSING,
                before_status=BUSINESS_EXECUTION_UNAVAILABLE_STATUS,
                after_status=BUSINESS_EXECUTION_UNAVAILABLE_STATUS,
            )
        if locked.current_order_owner_id != locked.subject_user_id:
            return self._stale(
                locked,
                BusinessExecutionReasonCode.ORDER_OWNER_CHANGED,
            )
        try:
            transition = resolve_order_action_transition(
                locked.action_type,
                locked.current_order_status,
            )
        except ActionExecutionError:
            return self._stale(
                locked,
                BusinessExecutionReasonCode.ORDER_STATUS_INELIGIBLE,
            )
        result_code = {
            "CANCELLED": BusinessExecutionResultCode.ORDER_CANCELLED,
            "REFUND_PENDING": BusinessExecutionResultCode.REFUND_PENDING,
            "AFTER_SALE_REVIEWING": (
                BusinessExecutionResultCode.AFTER_SALE_REVIEWING
            ),
        }.get(transition.next_status)
        if result_code is None:
            raise BusinessExecutionCorruption(
                "business policy returned an unsupported result"
            )
        return BusinessExecutionTransition(
            outcome=BusinessExecutionOutcome.EXECUTED,
            result_code=result_code,
            reason_code=BusinessExecutionReasonCode.BUSINESS_ACTION_EXECUTED,
            before_status=locked.current_order_status,
            after_status=transition.next_status,
            restore_stock=transition.restore_stock,
        )

    def _stale(
        self,
        locked: LockedBusinessExecution,
        reason: BusinessExecutionReasonCode,
    ) -> BusinessExecutionTransition:
        status = (
            locked.current_order_status
            or BUSINESS_EXECUTION_UNAVAILABLE_STATUS
        )
        return BusinessExecutionTransition(
            outcome=BusinessExecutionOutcome.STALE,
            result_code=BusinessExecutionResultCode.STALE,
            reason_code=reason,
            before_status=status,
            after_status=status,
        )

    def _protect_audit(
        self,
        write: BusinessExecutionWrite,
        transition: BusinessExecutionTransition,
    ) -> ProtectedBusinessExecutionAudit:
        event = BusinessExecutionAuditEvent(
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
        protected = self._data_protection.protect(
            event.payload(),
            profile=DataProtectionProfile.OBSERVABILITY,
        )
        return ProtectedBusinessExecutionAudit(
            canonical_json=protected.canonical_bytes.decode("utf-8"),
            payload_digest=protected.sha256,
        )

    def _validate_terminal(
        self,
        locked: LockedBusinessExecution,
    ) -> BusinessExecutionResult:
        audit = locked.audit
        if audit is None:
            raise BusinessExecutionCorruption(
                "terminal business action has no audit"
            )
        event = BusinessExecutionAuditEvent.from_canonical_json(
            audit.canonical_json
        )
        protected = self._data_protection.protect(
            event.payload(),
            profile=DataProtectionProfile.OBSERVABILITY,
        )
        expected_base = (
            locked.logical_action_id,
            locked.action_type,
            locked.target_order_id,
            locked.target_order_no,
            locked.subject_user_id,
            "APPROVE",
            locked.policy_version,
        )
        actual_base = (
            event.logical_action_id,
            event.action_type,
            event.target_order_id,
            event.target_order_no,
            event.subject_user_id,
            event.admin_decision,
            event.policy_version,
        )
        if (
            actual_base != expected_base
            or protected.canonical_bytes.decode("utf-8") != audit.canonical_json
            or protected.sha256 != audit.payload_digest
        ):
            raise BusinessExecutionCorruption(
                "terminal business audit does not match its action"
            )
        return decode_business_execution_audit_v1(
            action_id=locked.action_id,
            action_status=locked.action_status,
            action_type=locked.action_type,
            draft_policy_version=locked.policy_version,
            execution_result_code=locked.execution_result_code,
            executed_at=locked.executed_at,
            event=event,
        )


__all__ = ["DurableBusinessExecutionApplication"]
