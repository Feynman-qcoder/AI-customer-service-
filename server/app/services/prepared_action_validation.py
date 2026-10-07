from __future__ import annotations

from typing import Protocol

from app.agent.state import ActionDraftSnapshot
from app.agent.tools.registry import EffectPhase
from app.runtime.context import ExecutionScope
from app.runtime.durable import (
    DurableActionPrepareReplayLookup,
    DurableContractError,
    EffectConflict,
    EffectCorruption,
    EffectWriteScope,
    PreparedActionValidationRequest,
    PreparedActionValidationStorePort,
    canonical_action_payload_json,
)
from app.runtime.uow import (
    ApplicationTransactionCoordinator,
    ApplicationUnitOfWorkFactory,
    UnitOfWorkOperation,
)
from app.services.side_effect_policy_service import POLICY_VERSION


class PreparedActionValidationError(RuntimeError):
    """Persisted ACTION_PREPARE evidence cannot authorize an admin pause."""


class PreparedActionValidationPort(Protocol):
    async def validate(
        self,
        *,
        execution: ExecutionScope,
        draft: ActionDraftSnapshot,
        pending_action_id: int,
    ) -> None: ...


class PreparedActionValidationService:
    """Validate one prepared action in a fresh fenced application UoW."""

    def __init__(
        self,
        unit_of_work: ApplicationUnitOfWorkFactory[PreparedActionValidationStorePort],
        transactions: ApplicationTransactionCoordinator,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._transactions = transactions

    async def validate(
        self,
        *,
        execution: ExecutionScope,
        draft: ActionDraftSnapshot,
        pending_action_id: int,
    ) -> None:
        execution.require_attempt_active()
        fence = execution.lease.fence_token
        if type(fence) is not int or fence <= 0:
            raise PreparedActionValidationError(
                "prepared action validation requires live fenced authority"
            )
        if (
            draft.subject_user_id != execution.subject.user_id
            or draft.policy_version != POLICY_VERSION
        ):
            raise PreparedActionValidationError(
                "prepared action evidence does not match current authority"
            )
        lookup = DurableActionPrepareReplayLookup(
            scope=EffectWriteScope(
                thread_id=execution.thread_id,
                attempt_id=execution.attempt_id,
                fence_version=fence,
            ),
            conversation_id=execution.conversation_id,
            run_id=execution.run_id,
            node_name="durable_action_prepare",
            purpose=EffectPhase.ACTION_PREPARE.value,
            sequence=0,
            logical_action_id=draft.logical_action_id,
            action_type=draft.action_type,
            target_order_id=draft.target_order_id,
            target_order_no=draft.target_order_no.upper(),
            subject_user_id=draft.subject_user_id,
            created_by=execution.actor.user_id,
            action_payload_json=canonical_action_payload_json(
                {"reason": draft.reason_code}
            ),
            risk_level="HIGH",
            policy_version=draft.policy_version,
            draft_revision=draft.draft_revision,
            draft_expires_at=draft.expires_at,
            customer_confirmation_challenge_digest=draft.nonce_digest,
        )
        request = PreparedActionValidationRequest(
            lookup=lookup,
            pending_action_id=pending_action_id,
        )

        async def validate(store: PreparedActionValidationStorePort) -> None:
            execution.require_attempt_active()
            await store.validate_prepared_action(request)
            execution.require_attempt_active()

        try:
            await self._transactions.run(
                UnitOfWorkOperation.AGENT_CUSTOMER_CONFIRMATION_REVALIDATE,
                self._unit_of_work,
                validate,
            )
        except (DurableContractError, EffectConflict, EffectCorruption):
            raise PreparedActionValidationError(
                "prepared action evidence is invalid"
            ) from None


__all__ = [
    "PreparedActionValidationError",
    "PreparedActionValidationPort",
    "PreparedActionValidationService",
]
