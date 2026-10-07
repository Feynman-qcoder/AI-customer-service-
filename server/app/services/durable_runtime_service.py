from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import NoReturn, Protocol, cast

from sqlalchemy.exc import DBAPIError

from app.agent.state import EffectIdentity
from app.agent.thread_identity import ThreadIdentity
from app.agent.tools.registry import EffectPhase
from app.repositories.mysql_transaction_retry import mysql_error_code
from app.runtime.content_source import PublicationContentReferenceSet
from app.runtime.context import ExecutionScope
from app.runtime.durable import (
    DURABLE_INTERRUPT_CONFIRMATION_MODE,
    ActionPrepareEffectWrite,
    ActionPrepareProof,
    ActionPrepareStorePort,
    AttemptLeaseStorePort,
    AttemptRecord,
    AttemptRegistration,
    AuditEffectStorePort,
    AuditEffectWrite,
    CandidateSnapshot,
    CanonicalEffectClaim,
    CanonicalPublication,
    CheckpointPointer,
    DurableActionPrepareReplayLookup,
    DurableActionPrepareReplayStorePort,
    DurableContentionError,
    DurableContractError,
    EffectType,
    EffectWriteResult,
    EffectWriteScope,
    LeaseGrant,
    LeaseState,
    LogicalRunStorePort,
    MessageEffectStorePort,
    MessageEffectWrite,
    MessagePurpose,
    PublicationRecord,
    PublicationScope,
    PublicationStorePort,
    RunRecord,
    RunRegistration,
    WriteManifestItem,
    action_prepare_replay_digest,
    canonical_action_payload_json,
    canonical_digest,
    canonical_effect_idempotency_key,
    durable_action_prepare_replay_digest,
)
from app.runtime.uow import (
    ApplicationTransactionCoordinator,
    ApplicationUnitOfWork,
    ApplicationUnitOfWorkFactory,
    CommitOutcomeUnknown,
    UnitOfWorkOperation,
    UnitOfWorkState,
)
from app.services.side_effect_policy_service import (
    POLICY_VERSION,
    SideEffectAuthorization,
    ToolAuthorizationContext,
)


class ActionPrepareAuthorizationPort(Protocol):
    def verify_tool_authorization(
        self,
        authorization: SideEffectAuthorization | None,
        context: ToolAuthorizationContext,
    ) -> None: ...


class DurableRuntimeAuthorityService:
    """Application boundary for attempt, lease/fence, and publication authority."""

    def __init__(
        self,
        attempt_lease_uow: ApplicationUnitOfWorkFactory[AttemptLeaseStorePort],
        publication_uow: ApplicationUnitOfWorkFactory[PublicationStorePort],
        transactions: ApplicationTransactionCoordinator,
        run_uow: ApplicationUnitOfWorkFactory[LogicalRunStorePort] | None = None,
    ) -> None:
        self._attempt_lease_uow = attempt_lease_uow
        self._publication_uow = publication_uow
        self._transactions = transactions
        self._run_uow = run_uow or cast(
            ApplicationUnitOfWorkFactory[LogicalRunStorePort],
            attempt_lease_uow,
        )

    async def begin_run(self, execution: ExecutionScope) -> RunRecord:
        ThreadIdentity.from_conversation_id(execution.conversation_id).assert_matches(
            execution.conversation_id,
            execution.thread_id,
        )
        registration = RunRegistration(
            run_id=execution.run_id,
            thread_id=execution.thread_id,
            conversation_id=execution.conversation_id,
            subject_user_id=execution.subject.user_id,
            request_id=canonical_digest(
                {
                    "conversation_id": execution.conversation_id,
                    "run_id": execution.run_id,
                    "thread_id": execution.thread_id,
                }
            ),
            started_at=execution.started_at,
        )
        return await self._transactions.run(
            "run.begin",
            self._run_uow,
            lambda store: store.begin_run(registration),
        )

    async def register_attempt(self, execution: ExecutionScope) -> AttemptRecord:
        ThreadIdentity.from_conversation_id(execution.conversation_id).assert_matches(
            execution.conversation_id,
            execution.thread_id,
        )
        registration = AttemptRegistration(
            attempt_id=execution.attempt_id,
            run_id=execution.run_id,
            thread_id=execution.thread_id,
            conversation_id=execution.conversation_id,
            actor_user_id=execution.actor.user_id,
            actor_role=execution.actor.role,
            subject_user_id=execution.subject.user_id,
            started_at=execution.started_at,
        )
        return await self._transactions.run(
            "attempt.register",
            self._attempt_lease_uow,
            lambda store: store.register_attempt(registration),
        )

    async def acquire_lease(
        self,
        execution: ExecutionScope,
        *,
        lease_duration: timedelta,
    ) -> LeaseGrant:
        lease_milliseconds = _lease_milliseconds(lease_duration)
        return await self._transactions.run(
            "lease.acquire",
            self._attempt_lease_uow,
            lambda store: store.acquire_lease(
                thread_id=execution.thread_id,
                attempt_id=execution.attempt_id,
                lease_milliseconds=lease_milliseconds,
            ),
        )

    async def renew_lease(
        self,
        execution: ExecutionScope,
        *,
        lease_duration: timedelta,
    ) -> LeaseGrant:
        fence = _required_fence(execution)
        lease_milliseconds = _lease_milliseconds(lease_duration)
        return await self._transactions.run(
            "lease.renew",
            self._attempt_lease_uow,
            lambda store: store.renew_lease(
                thread_id=execution.thread_id,
                attempt_id=execution.attempt_id,
                fence_version=fence,
                lease_milliseconds=lease_milliseconds,
            ),
        )

    async def release_lease(self, execution: ExecutionScope) -> LeaseState:
        fence = _required_fence(execution)
        return await self._transactions.run(
            "lease.release",
            self._attempt_lease_uow,
            lambda store: store.release_lease(
                thread_id=execution.thread_id,
                attempt_id=execution.attempt_id,
                fence_version=fence,
            ),
        )

    async def read_lease(self, *, conversation_id: int) -> LeaseState | None:
        identity = ThreadIdentity.from_conversation_id(conversation_id)
        return await self._transactions.run(
            "lease.read",
            self._attempt_lease_uow,
            lambda store: store.read_lease(identity.thread_id),
        )

    async def require_live_lease(self, execution: ExecutionScope) -> LeaseGrant:
        return await self._transactions.run(
            "lease.require_live",
            self._attempt_lease_uow,
            lambda store: store.require_live_lease(
                thread_id=execution.thread_id,
                run_id=execution.run_id,
                attempt_id=execution.attempt_id,
                fence_version=_required_fence(execution),
            ),
        )

    async def publish(
        self,
        execution: ExecutionScope,
        *,
        snapshot: CandidateSnapshot,
        manifest: tuple[WriteManifestItem, ...],
        content_references: PublicationContentReferenceSet | None = None,
        expected_publication_version: int,
        expected_previous_pointer: CheckpointPointer | None,
    ) -> PublicationRecord:
        publication = self.build_publication(
            execution,
            snapshot=snapshot,
            manifest=manifest,
            content_references=content_references,
            expected_publication_version=expected_publication_version,
            expected_previous_pointer=expected_previous_pointer,
        )
        return await self._transactions.run(
            "publication.publish",
            self._publication_uow,
            lambda store: store.publish(publication),
        )

    def build_publication(
        self,
        execution: ExecutionScope,
        *,
        snapshot: CandidateSnapshot,
        manifest: tuple[WriteManifestItem, ...],
        content_references: PublicationContentReferenceSet | None = None,
        expected_publication_version: int,
        expected_previous_pointer: CheckpointPointer | None,
    ) -> CanonicalPublication:
        publication = CanonicalPublication(
            scope=PublicationScope(
                thread_id=execution.thread_id,
                attempt_id=execution.attempt_id,
                fence_version=_required_fence(execution),
            ),
            snapshot=snapshot,
            manifest=manifest,
            expected_publication_version=expected_publication_version,
            expected_previous_pointer=expected_previous_pointer,
            content_references=(
                content_references
                if content_references is not None
                else PublicationContentReferenceSet.empty()
            ),
        )
        # This is intentionally repeated immediately before the repository boundary.
        # A repository accepts only this canonical, fully validated DTO.
        publication.assert_valid()
        return publication

    async def read_publication(self, *, conversation_id: int) -> PublicationRecord | None:
        identity = ThreadIdentity.from_conversation_id(conversation_id)
        return await self._transactions.run(
            "publication.read",
            self._publication_uow,
            lambda store: store.read_publication(identity.thread_id),
        )


class ReplaySafeEffectService:
    """Create effect identities from server execution scope and persist them atomically."""

    def __init__(
        self,
        message_uow: ApplicationUnitOfWorkFactory[MessageEffectStorePort],
        audit_uow: ApplicationUnitOfWorkFactory[AuditEffectStorePort],
        action_prepare_uow: ApplicationUnitOfWorkFactory[ActionPrepareStorePort],
        action_authorization: ActionPrepareAuthorizationPort,
        transactions: ApplicationTransactionCoordinator,
    ) -> None:
        self._message_uow = message_uow
        self._audit_uow = audit_uow
        self._action_prepare_uow = action_prepare_uow
        self._durable_action_replay_uow = cast(
            ApplicationUnitOfWorkFactory[DurableActionPrepareReplayStorePort],
            action_prepare_uow,
        )
        self._action_authorization = action_authorization
        self._transactions = transactions

    async def write_message(
        self,
        execution: ExecutionScope,
        *,
        identity: EffectIdentity,
        purpose: MessagePurpose,
        role: str,
        content: str,
        sources_json: str | None = None,
        retrieval_score: Decimal | None = None,
        confidence_level: str | None = None,
        need_human: bool = False,
    ) -> EffectWriteResult:
        execution.require_attempt_active()
        if identity.purpose != purpose.value:
            raise DurableContractError("effect identity purpose does not match message purpose")
        payload = {
            "confidence_level": confidence_level,
            "content": content,
            "conversation_id": execution.conversation_id,
            "need_human": need_human,
            "retrieval_score": str(retrieval_score) if retrieval_score is not None else None,
            "role": role,
            "sources_json": sources_json,
        }
        claim = _canonical_effect_claim(execution, identity, EffectType.CHAT_MESSAGE, payload)
        write = MessageEffectWrite(
            scope=_effect_write_scope(execution),
            claim=claim,
            conversation_id=execution.conversation_id,
            role=role,
            content=content,
            sources_json=sources_json,
            retrieval_score=retrieval_score,
            confidence_level=confidence_level,
            need_human=need_human,
        )
        async def persist(store: MessageEffectStorePort) -> EffectWriteResult:
            execution.require_attempt_active()
            result = await store.write_message_effect(write)
            execution.require_attempt_active()
            return result

        return await self._transactions.run(
            "effect.message",
            self._message_uow,
            persist,
        )

    async def write_audit(
        self,
        execution: ExecutionScope,
        *,
        identity: EffectIdentity,
        input_summary: str | None,
        output_summary: str | None,
        status: str,
        duration_ms: int = 0,
        error_summary: str | None = None,
    ) -> EffectWriteResult:
        execution.require_attempt_active()
        if type(duration_ms) is not int or duration_ms < 0:
            raise DurableContractError("audit duration must be a non-negative integer")
        payload = {
            "duration_ms": duration_ms,
            "error_summary": error_summary,
            "input_summary": input_summary,
            "output_summary": output_summary,
            "status": status,
        }
        claim = _canonical_effect_claim(execution, identity, EffectType.LOCAL_AUDIT, payload)
        write = AuditEffectWrite(
            scope=_effect_write_scope(execution),
            claim=claim,
            input_summary=input_summary,
            output_summary=output_summary,
            status=status,
            duration_ms=duration_ms,
            error_summary=error_summary,
        )
        async def persist(store: AuditEffectStorePort) -> EffectWriteResult:
            execution.require_attempt_active()
            result = await store.write_audit_effect(write)
            execution.require_attempt_active()
            return result

        return await self._transactions.run(
            "effect.audit",
            self._audit_uow,
            persist,
        )

    async def write_action_prepare(
        self,
        execution: ExecutionScope,
        *,
        identity: EffectIdentity,
        authorization: SideEffectAuthorization | None,
        logical_action_id: str,
        action_type: str,
        target_order_id: int,
        target_order_no: str,
        validated_order_status: str,
        action_payload: Mapping[str, object],
        risk_level: str,
        confirmation_mode: str = "R2_STATELESS_COMPAT",
        customer_confirmation_challenge_digest: str | None = None,
        draft_revision: int | None = None,
        draft_expires_at: str | None = None,
    ) -> EffectWriteResult:
        execution.require_attempt_active()
        tool_name = {
            "REFUND": "request_refund",
            "ORDER_CANCELLATION": "request_order_cancellation",
        }.get(action_type)
        if tool_name is None:
            raise DurableContractError("unsupported action prepare type")
        payload_values = dict(action_payload)
        authorization_context = ToolAuthorizationContext(
            run_id=execution.run_id,
            logical_action_id=logical_action_id,
            subject_user_id=execution.subject.user_id,
            tool_name=tool_name,
            action_type=action_type,
            target_order_id=target_order_id,
            target_order_no=target_order_no.upper(),
            effect_phase=EffectPhase.ACTION_PREPARE,
            policy_version=POLICY_VERSION,
        )

        action_payload_json = canonical_action_payload_json(payload_values)
        if confirmation_mode == DURABLE_INTERRUPT_CONFIRMATION_MODE:
            if (
                customer_confirmation_challenge_digest is None
                or draft_revision is None
                or draft_expires_at is None
            ):
                raise DurableContractError(
                    "durable action prepare requires complete customer confirmation evidence"
                )
            replay_digest = durable_action_prepare_replay_digest(
                logical_action_id=logical_action_id,
                action_type=action_type,
                target_order_id=target_order_id,
                target_order_no=target_order_no.upper(),
                subject_user_id=execution.subject.user_id,
                created_by=execution.actor.user_id,
                action_payload_json=action_payload_json,
                risk_level=risk_level,
                validated_order_status=validated_order_status,
                policy_version=POLICY_VERSION,
                draft_revision=draft_revision,
                draft_expires_at=draft_expires_at,
                customer_confirmation_challenge_digest=(
                    customer_confirmation_challenge_digest
                ),
            )
        else:
            replay_digest = action_prepare_replay_digest(
                action_type=action_type,
                target_order_id=target_order_id,
                target_order_no=target_order_no.upper(),
                subject_user_id=execution.subject.user_id,
                created_by=execution.actor.user_id,
                action_payload_json=action_payload_json,
                risk_level=risk_level,
                validated_order_status=validated_order_status,
                policy_version=POLICY_VERSION,
            )
        claim = _canonical_effect_claim(
            execution,
            identity,
            EffectType.ACTION_PREPARE,
            None,
            payload_digest=replay_digest,
        )
        write = ActionPrepareEffectWrite(
            scope=_effect_write_scope(execution),
            claim=claim,
            action_type=action_type,
            target_order_id=target_order_id,
            action_payload_json=action_payload_json,
            risk_level=risk_level,
            created_by=execution.actor.user_id,
            subject_user_id=execution.subject.user_id,
            logical_action_id=logical_action_id,
            target_order_no=target_order_no.upper(),
            policy_version=POLICY_VERSION,
            validated_order_status=validated_order_status,
            confirmation_mode=confirmation_mode,
            customer_confirmation_challenge_digest=(
                customer_confirmation_challenge_digest
            ),
            draft_revision=draft_revision,
            draft_expires_at=draft_expires_at,
        )

        duplicate_key_pending = False
        for attempt_count in range(1, self._transactions.max_attempts + 1):
            execution.require_attempt_active()
            unit_of_work: ApplicationUnitOfWork[ActionPrepareStorePort] | None = None
            contention = None
            unknown_operation: str | None = None
            try:
                async with self._action_prepare_uow.open(
                    operation=UnitOfWorkOperation.EFFECT_ACTION_PREPARE
                ) as opened:
                    unit_of_work = opened
                    store = opened.store
                    execution.require_attempt_active()
                    replay = await store.lock_action_prepare_effect(write)
                    execution.require_attempt_active()
                    if replay is not None:
                        result = replay
                    else:
                        if duplicate_key_pending:
                            # A 1062 loss may only succeed by reading and fully
                            # validating a legitimate compatibility winner in
                            # this fresh UoW. A miss here means the duplicate
                            # key came from a non-target unique index (or the
                            # winner vanished): fail closed, never re-create.
                            _raise_duplicate_key_contention(attempt_count)
                        self._action_authorization.verify_tool_authorization(
                            authorization,
                            authorization_context,
                        )
                        if authorization is None:
                            raise DurableContractError(
                                "action authorization is required for first create"
                            )
                        proof = _action_prepare_proof(execution, write, authorization)
                        execution.require_attempt_active()
                        result = await store.create_action_prepare_effect(write, proof)
                        execution.require_attempt_active()
                return result
            except CommitOutcomeUnknown as caught:
                unknown_operation = caught.operation
                unit_of_work = None
                store = None
            except DBAPIError as exc:
                outcome = unit_of_work.outcome if unit_of_work is not None else None
                contention = self._transactions.retry_after_rollback(
                    exc,
                    operation=UnitOfWorkOperation.EFFECT_ACTION_PREPARE,
                    attempt_count=attempt_count,
                    outcome=outcome,
                )
                if contention is None:
                    if (
                        outcome is UnitOfWorkState.ROLLED_BACK
                        and mysql_error_code(exc) == _MYSQL_DUPLICATE_KEY_ERROR
                    ):
                        # Another run won a unique index race. The loss may only
                        # be retried as a compatibility-winner readback in a
                        # fresh UoW; the sanitized error carries no driver detail.
                        duplicate_key_pending = True
                        if attempt_count >= self._transactions.max_attempts:
                            _raise_duplicate_key_contention(attempt_count)
                        execution.require_attempt_active()
                        continue
                    raise
            if unknown_operation is not None:
                del self, authorization, action_payload, unit_of_work, opened, store
                raise CommitOutcomeUnknown(operation=unknown_operation) from None
            if contention is None:
                raise AssertionError("action-prepare contention classification was lost")
            if contention.attempt_count == self._transactions.max_attempts:
                raise DurableContentionError(
                    operation=contention.operation,
                    attempt_count=contention.attempt_count,
                    mysql_error_code=contention.mysql_error_code,
                )
            execution.require_attempt_active()
            await self._transactions.wait_before_retry(contention)
        raise AssertionError("bounded action-prepare retry loop did not return or raise")

    async def replay_durable_action_prepare(
        self,
        execution: ExecutionScope,
        *,
        identity: EffectIdentity,
        logical_action_id: str,
        action_type: str,
        target_order_id: int,
        target_order_no: str,
        action_payload: Mapping[str, object],
        risk_level: str,
        customer_confirmation_challenge_digest: str,
        draft_revision: int,
        draft_expires_at: str,
    ) -> EffectWriteResult | None:
        """Replay a durable prepare using its immutable persisted status evidence."""

        execution.require_attempt_active()
        if action_type not in {"REFUND", "ORDER_CANCELLATION"}:
            raise DurableContractError("unsupported action prepare type")
        if identity.run_id != execution.run_id:
            raise DurableContractError(
                "effect identity run does not match execution scope"
            )
        lookup = DurableActionPrepareReplayLookup(
            scope=_effect_write_scope(execution),
            conversation_id=execution.conversation_id,
            run_id=execution.run_id,
            node_name=identity.node_name,
            purpose=identity.purpose,
            sequence=identity.sequence,
            logical_action_id=logical_action_id,
            action_type=action_type,
            target_order_id=target_order_id,
            target_order_no=target_order_no.upper(),
            subject_user_id=execution.subject.user_id,
            created_by=execution.actor.user_id,
            action_payload_json=canonical_action_payload_json(dict(action_payload)),
            risk_level=risk_level,
            policy_version=POLICY_VERSION,
            draft_revision=draft_revision,
            draft_expires_at=draft_expires_at,
            customer_confirmation_challenge_digest=(
                customer_confirmation_challenge_digest
            ),
        )

        async def replay(
            store: DurableActionPrepareReplayStorePort,
        ) -> EffectWriteResult | None:
            execution.require_attempt_active()
            result = await store.lock_durable_action_prepare_replay(lookup)
            execution.require_attempt_active()
            return result

        return await self._transactions.run(
            UnitOfWorkOperation.EFFECT_ACTION_PREPARE,
            self._durable_action_replay_uow,
            replay,
        )


_MYSQL_DUPLICATE_KEY_ERROR = 1062


def _raise_duplicate_key_contention(attempt_count: int) -> NoReturn:
    """Raise the sanitized duplicate-key contention without leaking driver detail.

    The original DBAPIError carries the raw SQL statement and parameters; the
    raised ``DurableContentionError`` keeps only structured, allowlisted
    fields, and its exception link is stripped the same way the commit-outcome
    sanitizer does it.
    """
    error = DurableContentionError(
        operation=UnitOfWorkOperation.EFFECT_ACTION_PREPARE.value,
        attempt_count=attempt_count,
        mysql_error_code=_MYSQL_DUPLICATE_KEY_ERROR,
    )
    try:
        raise error
    except DurableContentionError:
        error.__context__ = None
        raise


def _canonical_effect_claim(
    execution: ExecutionScope,
    identity: EffectIdentity,
    effect_type: EffectType,
    payload: object,
    *,
    payload_digest: str | None = None,
) -> CanonicalEffectClaim:
    if identity.run_id != execution.run_id:
        raise DurableContractError("effect identity run does not match execution scope")
    return CanonicalEffectClaim(
        run_id=identity.run_id,
        node_name=identity.node_name,
        purpose=identity.purpose,
        sequence=identity.sequence,
        effect_type=effect_type,
        idempotency_key=canonical_effect_idempotency_key(
            run_id=identity.run_id,
            node_name=identity.node_name,
            purpose=identity.purpose,
            sequence=identity.sequence,
        ),
        payload_digest=(
            payload_digest if payload_digest is not None else canonical_digest(payload)
        ),
    )


def _action_prepare_proof(
    execution: ExecutionScope,
    write: ActionPrepareEffectWrite,
    authorization: SideEffectAuthorization,
) -> ActionPrepareProof:
    try:
        expires_at = datetime.fromisoformat(authorization.expires_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DurableContractError("action authorization expiry is invalid") from exc
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    partial = ActionPrepareProof(
        authorization_id=authorization.authorization_id,
        thread_id=execution.thread_id,
        run_id=execution.run_id,
        attempt_id=execution.attempt_id,
        fence_version=_required_fence(execution),
        actor_user_id=execution.actor.user_id,
        subject_user_id=execution.subject.user_id,
        logical_action_id=write.logical_action_id,
        action_type=write.action_type,
        target_order_id=write.target_order_id,
        target_order_no=write.target_order_no,
        payload_digest=write.claim.payload_digest,
        policy_version=write.policy_version,
        expires_at=expires_at,
        proof_digest="0" * 64,
    )
    return replace(
        partial,
        proof_digest=canonical_digest(partial.canonical_payload()),
    )


def _effect_write_scope(execution: ExecutionScope) -> EffectWriteScope:
    return EffectWriteScope(
        thread_id=execution.thread_id,
        attempt_id=execution.attempt_id,
        fence_version=_required_fence(execution),
    )


def _required_fence(execution: ExecutionScope) -> int:
    fence = execution.lease.fence_token
    if type(fence) is not int or fence <= 0:
        raise DurableContractError("a positive server-issued fence is required")
    return fence


def _lease_milliseconds(duration: timedelta) -> int:
    milliseconds = duration // timedelta(milliseconds=1)
    if type(milliseconds) is not int or not 0 < milliseconds <= 86_400_000:
        raise DurableContractError("lease duration must be between 1 ms and 24 hours")
    return milliseconds
