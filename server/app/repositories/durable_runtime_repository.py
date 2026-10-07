from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Protocol, TypeVar

from sqlalchemy import and_, exists, literal_column, select, update
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from app.db.models.entities import (
    AgentActionRequest,
    AgentCheckpointPublication,
    AgentCheckpointWriteManifest,
    AgentEffect,
    AgentRun,
    AgentRunAttempt,
    AgentStep,
    AgentThreadExecution,
    ChatConversation,
    ChatMessage,
    CustomerOrder,
)
from app.repositories.checkpoint_content_source_repository import (
    insert_publication_content_holds,
    lock_publication_content_sources,
)
from app.runtime.durable import (
    DURABLE_INTERRUPT_CONFIRMATION_MODE,
    R2_STATELESS_COMPAT_CONFIRMATION_MODE,
    ActionPrepareEffectWrite,
    ActionPrepareProof,
    AttemptRecord,
    AttemptRegistration,
    AuditEffectWrite,
    CanonicalEffectClaim,
    CanonicalPublication,
    CheckpointPointer,
    DurableActionPrepareReplayLookup,
    DurableContractError,
    EffectConflict,
    EffectCorruption,
    EffectType,
    EffectWriteResult,
    EffectWriteScope,
    LeaseConflict,
    LeaseGrant,
    LeaseState,
    MessageEffectWrite,
    PreparedActionValidationRequest,
    PublicationConflict,
    PublicationRecord,
    PublicationScope,
    R2DayCompatibilityKey,
    RunRecord,
    RunRegistration,
    WriteManifestItem,
    build_r2_day_compatibility_key,
    canonical_effect_idempotency_key,
    manifest_root,
    parse_r2_day_compatibility_key,
)
from app.runtime.uow import TransactionBoundStoreGuard

_DB_NOW: ColumnElement[datetime] = literal_column("CURRENT_TIMESTAMP(6)")
_ResultT = TypeVar("_ResultT")


class _BoundTransactionExecutor(Protocol):
    async def run(
        self,
        operation: str,
        work: Callable[[AsyncSession], Awaitable[_ResultT]],
    ) -> _ResultT: ...


class _DirectBoundSessionTransactionExecutor:
    def __init__(self, session: AsyncSession, guard: TransactionBoundStoreGuard) -> None:
        self._session = session
        self._guard = guard

    async def run(
        self,
        operation: str,
        work: Callable[[AsyncSession], Awaitable[_ResultT]],
    ) -> _ResultT:
        del operation
        self._guard.ensure_active()
        return await work(self._session)


class _SqlAlchemyDurableRuntimeStoreBase:
    """Transaction-bound MySQL store; the application UoW owns its Session."""

    def __init__(
        self,
        session: AsyncSession,
        guard: TransactionBoundStoreGuard,
    ) -> None:
        self._bound_session = session
        self._guard = guard
        self._transactions: _BoundTransactionExecutor = _DirectBoundSessionTransactionExecutor(
            session,
            guard,
        )

    async def begin_run(self, registration: RunRegistration) -> RunRecord:
        async def transaction(session: AsyncSession) -> RunRecord:
            conversation = await session.scalar(
                select(ChatConversation)
                .where(ChatConversation.id == registration.conversation_id)
                .with_for_update()
            )
            if (
                conversation is None
                or conversation.user_id is None
                or conversation.user_id != registration.subject_user_id
            ):
                raise DurableContractError(
                    "logical run conversation ownership is invalid"
                )
            run = await session.scalar(
                select(AgentRun)
                .where(AgentRun.run_id == registration.run_id)
                .with_for_update()
            )
            if run is None:
                run = AgentRun(
                    run_id=registration.run_id,
                    thread_id=registration.thread_id,
                    conversation_id=registration.conversation_id,
                    user_id=registration.subject_user_id,
                    status="RUNNING",
                    started_at=registration.started_at,
                    request_id=registration.request_id,
                )
                session.add(run)
                await self._guard.flush(session)
            actual = (
                run.thread_id,
                run.conversation_id,
                run.user_id,
                run.request_id,
            )
            expected = (
                registration.thread_id,
                registration.conversation_id,
                registration.subject_user_id,
                registration.request_id,
            )
            if actual != expected:
                raise DurableContractError(
                    "run id is already bound to another logical identity"
                )
            return _run_record(run)

        return await self._transactions.run("run.begin", transaction)

    async def register_attempt(self, registration: AttemptRegistration) -> AttemptRecord:
        async def transaction(session: AsyncSession) -> AttemptRecord:
            run = await session.scalar(select(AgentRun).where(AgentRun.run_id == registration.run_id).with_for_update())
            if run is None:
                raise DurableContractError("attempt references an unknown logical run")
            if (
                run.thread_id != registration.thread_id
                or run.conversation_id != registration.conversation_id
                or run.user_id != registration.subject_user_id
            ):
                raise DurableContractError("attempt identity does not match its logical run")

            statement = mysql_insert(AgentRunAttempt).values(
                attempt_id=registration.attempt_id,
                run_id=registration.run_id,
                thread_id=registration.thread_id,
                conversation_id=registration.conversation_id,
                actor_user_id=registration.actor_user_id,
                actor_role=registration.actor_role,
                subject_user_id=registration.subject_user_id,
                status="REGISTERED",
                fence_version=None,
                started_at=registration.started_at,
            )
            statement = statement.on_duplicate_key_update(attempt_id=AgentRunAttempt.attempt_id)
            await session.execute(statement)
            attempt = await session.scalar(
                select(AgentRunAttempt).where(AgentRunAttempt.attempt_id == registration.attempt_id).with_for_update()
            )
            if attempt is None:
                raise DurableContractError("attempt registration did not persist")
            actual_identity = (
                attempt.run_id,
                attempt.thread_id,
                attempt.conversation_id,
                attempt.actor_user_id,
                attempt.actor_role,
                attempt.subject_user_id,
            )
            requested_identity = (
                registration.run_id,
                registration.thread_id,
                registration.conversation_id,
                registration.actor_user_id,
                registration.actor_role,
                registration.subject_user_id,
            )
            if actual_identity != requested_identity:
                raise DurableContractError("attempt id is already bound to another execution scope")
            return _attempt_record(attempt)

        return await self._transactions.run("attempt.register", transaction)

    async def acquire_lease(
        self,
        *,
        thread_id: str,
        attempt_id: str,
        lease_milliseconds: int,
    ) -> LeaseGrant:
        _validate_lease_duration(lease_milliseconds)

        async def transaction(session: AsyncSession) -> LeaseGrant:
            attempt = await _load_attempt_for_thread(session, thread_id, attempt_id)
            await _ensure_thread_rows(session, attempt)
            execution = await session.scalar(
                select(AgentThreadExecution).where(AgentThreadExecution.thread_id == thread_id).with_for_update()
            )
            if execution is None:
                raise LeaseConflict("thread authority row does not exist")
            now, lease_expires_at = await _database_lease_times(session, lease_milliseconds)
            is_current_owner = (
                execution.owner_attempt_id == attempt_id
                and execution.lease_expires_at is not None
                and execution.lease_expires_at > now
            )
            is_available = (
                execution.owner_attempt_id is None
                or execution.lease_expires_at is None
                or execution.lease_expires_at <= now
            )
            if is_current_owner:
                fence_version = execution.fence_version
            elif is_available:
                fence_version = execution.fence_version + 1
            else:
                raise LeaseConflict("thread lease is owned by another live attempt")

            execution.owner_attempt_id = attempt_id
            execution.fence_version = fence_version
            execution.lease_expires_at = lease_expires_at
            attempt.status = "ACTIVE"
            attempt.fence_version = fence_version
            return LeaseGrant(
                thread_id=thread_id,
                owner_attempt_id=attempt_id,
                fence_version=fence_version,
                lease_expires_at=lease_expires_at,
            )

        return await self._transactions.run("lease.acquire", transaction)

    async def renew_lease(
        self,
        *,
        thread_id: str,
        attempt_id: str,
        fence_version: int,
        lease_milliseconds: int,
    ) -> LeaseGrant:
        _validate_fence(fence_version)
        _validate_lease_duration(lease_milliseconds)

        async def transaction(session: AsyncSession) -> LeaseGrant:
            await _load_attempt_for_thread(session, thread_id, attempt_id)
            _, lease_expires_at = await _database_lease_times(session, lease_milliseconds)
            statement = (
                update(AgentThreadExecution)
                .where(
                    AgentThreadExecution.thread_id == thread_id,
                    AgentThreadExecution.owner_attempt_id == attempt_id,
                    AgentThreadExecution.fence_version == fence_version,
                    AgentThreadExecution.lease_expires_at.is_not(None),
                    AgentThreadExecution.lease_expires_at > _DB_NOW,
                )
                .values(lease_expires_at=lease_expires_at)
            )
            result = await session.execute(statement)
            if result.rowcount != 1:
                raise LeaseConflict("lease renewal lost owner, fence, or database-time validity")
            return LeaseGrant(
                thread_id=thread_id,
                owner_attempt_id=attempt_id,
                fence_version=fence_version,
                lease_expires_at=lease_expires_at,
            )

        return await self._transactions.run("lease.renew", transaction)

    async def release_lease(
        self,
        *,
        thread_id: str,
        attempt_id: str,
        fence_version: int,
    ) -> LeaseState:
        _validate_fence(fence_version)

        async def transaction(session: AsyncSession) -> LeaseState:
            attempt = await _load_attempt_for_thread(session, thread_id, attempt_id)
            statement = (
                update(AgentThreadExecution)
                .where(
                    AgentThreadExecution.thread_id == thread_id,
                    AgentThreadExecution.owner_attempt_id == attempt_id,
                    AgentThreadExecution.fence_version == fence_version,
                )
                .values(owner_attempt_id=None, lease_expires_at=None)
            )
            result = await session.execute(statement)
            if result.rowcount != 1:
                raise LeaseConflict("lease release lost owner or fence authority")
            attempt.status = "RELEASED"
            return LeaseState(
                thread_id=thread_id,
                owner_attempt_id=None,
                fence_version=fence_version,
                lease_expires_at=None,
            )

        return await self._transactions.run("lease.release", transaction)

    async def read_lease(self, thread_id: str) -> LeaseState | None:
        self._guard.ensure_active()
        execution = await self._bound_session.get(AgentThreadExecution, thread_id)
        return _lease_state(execution) if execution is not None else None

    async def require_live_lease(
        self,
        *,
        thread_id: str,
        run_id: str,
        attempt_id: str,
        fence_version: int,
    ) -> LeaseGrant:
        _validate_fence(fence_version)
        attempt = await self._bound_session.scalar(
            select(AgentRunAttempt)
            .where(
                AgentRunAttempt.attempt_id == attempt_id,
                AgentRunAttempt.run_id == run_id,
                AgentRunAttempt.thread_id == thread_id,
                AgentRunAttempt.fence_version == fence_version,
            )
            .with_for_update()
        )
        if attempt is None:
            raise LeaseConflict("live lease attempt binding is invalid")
        execution = await self._bound_session.scalar(
            select(AgentThreadExecution)
            .where(
                AgentThreadExecution.thread_id == thread_id,
                AgentThreadExecution.owner_attempt_id == attempt_id,
                AgentThreadExecution.fence_version == fence_version,
                AgentThreadExecution.lease_expires_at.is_not(None),
                AgentThreadExecution.lease_expires_at > _DB_NOW,
            )
            .with_for_update()
        )
        if execution is None or execution.lease_expires_at is None:
            raise LeaseConflict(
                "live lease lost owner, fence, or database-time validity"
            )
        return LeaseGrant(
            thread_id=thread_id,
            owner_attempt_id=attempt_id,
            fence_version=fence_version,
            lease_expires_at=execution.lease_expires_at,
        )

    async def publish(self, publication: CanonicalPublication) -> PublicationRecord:
        # This check happens before a transaction is opened, so domain mismatch cannot
        # issue a CAS, UPDATE, or INSERT of any kind.
        publication.assert_valid()
        scope = publication.scope
        pointer = publication.pointer
        previous = publication.expected_previous_pointer

        async def transaction(session: AsyncSession) -> PublicationRecord:
            current_attempt = await _require_live_publication_authority(session, scope)
            locked_sources = await lock_publication_content_sources(
                session,
                publication.content_references,
                expected_conversation_id=current_attempt.conversation_id,
                expected_subject_user_id=current_attempt.subject_user_id,
            )
            live_authority = exists(
                select(1).where(
                    AgentThreadExecution.thread_id == scope.thread_id,
                    AgentThreadExecution.owner_attempt_id == scope.attempt_id,
                    AgentThreadExecution.fence_version == scope.fence_version,
                    AgentThreadExecution.lease_expires_at.is_not(None),
                    AgentThreadExecution.lease_expires_at > _DB_NOW,
                )
            )
            statement = (
                update(AgentCheckpointPublication)
                .where(
                    AgentCheckpointPublication.thread_id == scope.thread_id,
                    AgentCheckpointPublication.publication_version == publication.expected_publication_version,
                    _pointer_matches(previous),
                    live_authority,
                )
                .values(
                    publication_version=publication.next_publication_version,
                    logical_namespace=pointer.logical_namespace,
                    physical_namespace=pointer.physical_namespace,
                    checkpoint_id=pointer.checkpoint_id,
                    checkpoint_digest=publication.snapshot.checkpoint_digest,
                    previous_logical_namespace=(previous.logical_namespace if previous else None),
                    previous_physical_namespace=(previous.physical_namespace if previous else None),
                    previous_checkpoint_id=(previous.checkpoint_id if previous else None),
                    manifest_root=publication.snapshot.manifest_root,
                    manifest_count=publication.snapshot.manifest_count,
                )
            )
            result = await session.execute(statement)
            if result.rowcount != 1:
                raise PublicationConflict("publication CAS lost lease/fence, version, or exact previous pointer")
            if publication.manifest:
                await session.execute(
                    mysql_insert(AgentCheckpointWriteManifest),
                    [
                        {
                            "thread_id": item.thread_id,
                            "publication_version": publication.next_publication_version,
                            "logical_namespace": item.logical_namespace,
                            "physical_namespace": item.physical_namespace,
                            "checkpoint_id": item.checkpoint_id,
                            "task_id": item.task_id,
                            "write_index": item.write_index,
                            "channel": item.channel,
                            "content_digest": item.content_digest,
                        }
                        for item in publication.manifest
                    ],
                )
            await insert_publication_content_holds(
                session,
                thread_id=scope.thread_id,
                publication_version=publication.next_publication_version,
                references=publication.content_references,
                locked_sources=locked_sources,
            )
            record = await _read_publication(session, scope.thread_id)
            if record is None:
                raise PublicationConflict("publication disappeared inside its transaction")
            return record

        return await self._transactions.run("publication.publish", transaction)

    async def read_publication(self, thread_id: str) -> PublicationRecord | None:
        self._guard.ensure_active()
        return await _read_publication(self._bound_session, thread_id)

    async def write_message_effect(self, write: MessageEffectWrite) -> EffectWriteResult:
        async def transaction(session: AsyncSession) -> EffectWriteResult:
            current_attempt = await _require_live_effect_authority(
                session,
                write.scope,
                write.claim.run_id,
            )
            if write.conversation_id != current_attempt.conversation_id:
                raise EffectConflict("message conversation does not match the current attempt")
            effect = await _find_effect(session, write.claim)
            existing: ChatMessage | None
            if effect is None:
                effect = await _create_effect(
                    session,
                    write.scope,
                    write.claim,
                    guard=self._guard,
                )
                existing = ChatMessage(
                    conversation_id=write.conversation_id,
                    role=write.role,
                    content=write.content,
                    sources_json=write.sources_json,
                    retrieval_score=write.retrieval_score,
                    confidence_level=write.confidence_level,
                    need_human=write.need_human,
                    source_run_id=effect.run_id,
                    source_attempt_id=effect.attempt_id,
                    message_purpose=effect.purpose,
                    message_sequence=effect.sequence,
                    message_idempotency_key=effect.idempotency_key,
                    effect_id=effect.id,
                )
                session.add(existing)
                await self._guard.flush(session)
                replayed = False
            else:
                existing = await session.scalar(
                    select(ChatMessage).where(ChatMessage.effect_id == effect.id).with_for_update()
                )
                if existing is None:
                    raise EffectCorruption("message effect exists without its target")
                _validate_message_target(existing, effect, write)
                replayed = True
            return EffectWriteResult(
                effect_id=effect.id,
                target_id=existing.id,
                idempotency_key=effect.idempotency_key,
                replayed=replayed,
            )

        return await self._transactions.run("effect.message", transaction)

    async def write_audit_effect(self, write: AuditEffectWrite) -> EffectWriteResult:
        async def transaction(session: AsyncSession) -> EffectWriteResult:
            await _require_live_effect_authority(session, write.scope, write.claim.run_id)
            effect = await _find_effect(session, write.claim)
            existing: AgentStep | None
            if effect is None:
                effect = await _create_effect(
                    session,
                    write.scope,
                    write.claim,
                    guard=self._guard,
                )
                existing = AgentStep(
                    run_id=effect.run_id,
                    node_name=effect.node_name,
                    input_summary=write.input_summary,
                    output_summary=write.output_summary,
                    status=write.status,
                    duration_ms=write.duration_ms,
                    error_summary=write.error_summary,
                    attempt_id=effect.attempt_id,
                    effect_id=effect.id,
                    effect_purpose=effect.purpose,
                    effect_sequence=effect.sequence,
                    effect_idempotency_key=effect.idempotency_key,
                )
                session.add(existing)
                await self._guard.flush(session)
                replayed = False
            else:
                existing = await session.scalar(
                    select(AgentStep).where(AgentStep.effect_id == effect.id).with_for_update()
                )
                if existing is None:
                    raise EffectCorruption("audit effect exists without its target")
                _validate_audit_target(existing, effect, write)
                replayed = True
            return EffectWriteResult(
                effect_id=effect.id,
                target_id=existing.id,
                idempotency_key=effect.idempotency_key,
                replayed=replayed,
            )

        return await self._transactions.run("effect.audit", transaction)

    async def write_action_prepare_effect(
        self,
        write: ActionPrepareEffectWrite,
        *,
        proof: ActionPrepareProof,
    ) -> EffectWriteResult:
        async def transaction(session: AsyncSession) -> EffectWriteResult:
            current_attempt = await _require_live_effect_authority(
                session,
                write.scope,
                write.claim.run_id,
            )
            if write.created_by != current_attempt.actor_user_id:
                raise EffectConflict("action creator does not match the current attempt actor")
            if write.subject_user_id != current_attempt.subject_user_id:
                raise EffectConflict("action subject does not match the current attempt subject")
            _validate_action_prepare_write_contract(write)
            if write.confirmation_mode == DURABLE_INTERRUPT_CONFIRMATION_MODE:
                replay = await _lock_durable_action_prepare_replay(
                    session,
                    _durable_lookup_from_write(write, current_attempt),
                    current_attempt=current_attempt,
                )
                if replay is None:
                    await _lock_action_target(session, write)
            else:
                await _lock_action_target(session, write)
                replay = await _lock_action_prepare_replay(
                    session,
                    write,
                    current_attempt=current_attempt,
                )
            if replay is not None:
                return replay
            await _validate_action_prepare_proof(session, write, proof, current_attempt)
            effect = await _create_effect(
                session,
                write.scope,
                write.claim,
                guard=self._guard,
            )
            existing = await _create_action_prepare_target(
                session,
                write,
                effect,
                guard=self._guard,
            )
            return EffectWriteResult(
                effect_id=effect.id,
                target_id=existing.id,
                idempotency_key=effect.idempotency_key,
                replayed=False,
            )

        return await self._transactions.run("effect.action_prepare", transaction)


class SqlAlchemyDurableRuntimeStore(_SqlAlchemyDurableRuntimeStoreBase):
    """One transaction-bound durable store; it never opens or finalizes a Session."""

    def __init__(self, session: AsyncSession, guard: TransactionBoundStoreGuard) -> None:
        super().__init__(session, guard)

    async def read_lease(self, thread_id: str) -> LeaseState | None:
        self._guard.ensure_active()
        execution = await self._bound_session.get(AgentThreadExecution, thread_id)
        return _lease_state(execution) if execution is not None else None

    async def read_publication(self, thread_id: str) -> PublicationRecord | None:
        self._guard.ensure_active()
        return await _read_publication(self._bound_session, thread_id)

    async def lock_action_prepare_effect(
        self,
        write: ActionPrepareEffectWrite,
    ) -> EffectWriteResult | None:
        self._guard.ensure_active()
        session = self._bound_session
        current_attempt = await _require_live_effect_authority(
            session,
            write.scope,
            write.claim.run_id,
        )
        if write.created_by != current_attempt.actor_user_id:
            raise EffectConflict("action creator does not match the current attempt actor")
        if write.subject_user_id != current_attempt.subject_user_id:
            raise EffectConflict("action subject does not match the current attempt subject")
        _validate_action_prepare_write_contract(write)
        if write.confirmation_mode == DURABLE_INTERRUPT_CONFIRMATION_MODE:
            replay = await _lock_durable_action_prepare_replay(
                session,
                _durable_lookup_from_write(write, current_attempt),
                current_attempt=current_attempt,
            )
            if replay is not None:
                return replay
        # A first create locks and revalidates the order inside this UoW.
        await _lock_action_target(session, write)
        return await _lock_action_prepare_replay(
            session,
            write,
            current_attempt=current_attempt,
        )

    async def lock_durable_action_prepare_replay(
        self,
        lookup: DurableActionPrepareReplayLookup,
    ) -> EffectWriteResult | None:
        self._guard.ensure_active()
        current_attempt = await _require_live_effect_authority(
            self._bound_session,
            lookup.scope,
            lookup.run_id,
        )
        return await _lock_durable_action_prepare_replay(
            self._bound_session,
            lookup,
            current_attempt=current_attempt,
        )

    async def validate_prepared_action(
        self,
        request: PreparedActionValidationRequest,
    ) -> None:
        self._guard.ensure_active()
        lookup = request.lookup
        current_attempt = await _require_live_effect_authority(
            self._bound_session,
            lookup.scope,
            lookup.run_id,
        )
        replay = await _lock_durable_action_prepare_replay(
            self._bound_session,
            lookup,
            current_attempt=current_attempt,
            required_target_id=request.pending_action_id,
        )
        if replay is None:
            raise EffectCorruption("prepared action evidence is unavailable")

    async def create_action_prepare_effect(
        self,
        write: ActionPrepareEffectWrite,
        proof: ActionPrepareProof,
    ) -> EffectWriteResult:
        self._guard.ensure_active()
        current_attempt = await _require_live_effect_authority(
            self._bound_session,
            write.scope,
            write.claim.run_id,
        )
        if write.created_by != current_attempt.actor_user_id:
            raise EffectConflict("action creator does not match the current attempt actor")
        if write.subject_user_id != current_attempt.subject_user_id:
            raise EffectConflict("action subject does not match the current attempt subject")
        _validate_action_prepare_write_contract(write)
        # The create port is a security boundary of its own.  Re-lock even
        # when the normal service path called lock_action_prepare_effect first,
        # so a direct/future caller cannot bypass owner/order/status checks.
        await _lock_action_target(self._bound_session, write)
        return await self.write_action_prepare_effect(write, proof=proof)


async def _load_attempt_for_thread(
    session: AsyncSession,
    thread_id: str,
    attempt_id: str,
) -> AgentRunAttempt:
    attempt = await session.scalar(
        select(AgentRunAttempt)
        .where(
            AgentRunAttempt.attempt_id == attempt_id,
            AgentRunAttempt.thread_id == thread_id,
        )
        .with_for_update()
    )
    if attempt is None:
        raise LeaseConflict("attempt does not belong to the requested thread")
    return attempt


async def _ensure_thread_rows(session: AsyncSession, attempt: AgentRunAttempt) -> None:
    execution_insert = mysql_insert(AgentThreadExecution).values(
        thread_id=attempt.thread_id,
        conversation_id=attempt.conversation_id,
        owner_attempt_id=None,
        fence_version=0,
        lease_expires_at=None,
    )
    await session.execute(execution_insert.on_duplicate_key_update(thread_id=AgentThreadExecution.thread_id))
    publication_insert = mysql_insert(AgentCheckpointPublication).values(
        thread_id=attempt.thread_id,
        conversation_id=attempt.conversation_id,
        publication_version=0,
        manifest_count=0,
    )
    await session.execute(publication_insert.on_duplicate_key_update(thread_id=AgentCheckpointPublication.thread_id))


async def _database_lease_times(
    session: AsyncSession,
    lease_milliseconds: int,
) -> tuple[datetime, datetime]:
    lease_microseconds = lease_milliseconds * 1_000
    expiry: ColumnElement[datetime] = literal_column(
        f"TIMESTAMPADD(MICROSECOND, {lease_microseconds}, CURRENT_TIMESTAMP(6))"
    )
    row = (await session.execute(select(_DB_NOW, expiry))).one()
    return row[0], row[1]


def _pointer_matches(pointer: CheckpointPointer | None) -> ColumnElement[bool]:
    if pointer is None:
        return and_(
            AgentCheckpointPublication.logical_namespace.is_(None),
            AgentCheckpointPublication.physical_namespace.is_(None),
            AgentCheckpointPublication.checkpoint_id.is_(None),
        )
    return and_(
        AgentCheckpointPublication.logical_namespace == pointer.logical_namespace,
        AgentCheckpointPublication.physical_namespace == pointer.physical_namespace,
        AgentCheckpointPublication.checkpoint_id == pointer.checkpoint_id,
    )


async def _read_publication(
    session: AsyncSession,
    thread_id: str,
) -> PublicationRecord | None:
    publication = await session.get(AgentCheckpointPublication, thread_id)
    if publication is None:
        return None
    if publication.publication_version == 0:
        return PublicationRecord(
            thread_id=thread_id,
            publication_version=0,
            pointer=None,
            checkpoint_digest=None,
            previous_pointer=None,
            manifest_root=None,
            manifest_count=0,
            manifest=(),
        )
    if (
        publication.logical_namespace is None
        or publication.physical_namespace is None
        or publication.checkpoint_id is None
        or publication.checkpoint_digest is None
        or publication.manifest_root is None
    ):
        raise PublicationConflict("published pointer metadata is incomplete")
    rows = (
        await session.scalars(
            select(AgentCheckpointWriteManifest)
            .where(
                AgentCheckpointWriteManifest.thread_id == thread_id,
                AgentCheckpointWriteManifest.publication_version == publication.publication_version,
            )
            .order_by(
                AgentCheckpointWriteManifest.task_id,
                AgentCheckpointWriteManifest.write_index,
                AgentCheckpointWriteManifest.channel,
            )
        )
    ).all()
    manifest = tuple(
        WriteManifestItem(
            thread_id=row.thread_id,
            logical_namespace=row.logical_namespace,
            physical_namespace=row.physical_namespace,
            checkpoint_id=row.checkpoint_id,
            task_id=row.task_id,
            write_index=row.write_index,
            channel=row.channel,
            content_digest=row.content_digest,
        )
        for row in rows
    )
    expected_domain = (
        thread_id,
        publication.logical_namespace,
        publication.physical_namespace,
        publication.checkpoint_id,
    )
    if any(
        (
            item.thread_id,
            item.logical_namespace,
            item.physical_namespace,
            item.checkpoint_id,
        )
        != expected_domain
        for item in manifest
    ):
        raise PublicationConflict("published manifest item does not match publication pointer")
    if len(manifest) != publication.manifest_count:
        raise PublicationConflict("published manifest is incomplete")
    if manifest_root(manifest) != publication.manifest_root:
        raise PublicationConflict("published manifest root does not match its exact rows")
    previous_pointer = None
    if publication.previous_checkpoint_id is not None:
        if publication.previous_logical_namespace is None or publication.previous_physical_namespace is None:
            raise PublicationConflict("published previous pointer is incomplete")
        previous_pointer = CheckpointPointer(
            logical_namespace=publication.previous_logical_namespace,
            physical_namespace=publication.previous_physical_namespace,
            checkpoint_id=publication.previous_checkpoint_id,
        )
    return PublicationRecord(
        thread_id=thread_id,
        publication_version=publication.publication_version,
        pointer=CheckpointPointer(
            logical_namespace=publication.logical_namespace,
            physical_namespace=publication.physical_namespace,
            checkpoint_id=publication.checkpoint_id,
        ),
        checkpoint_digest=publication.checkpoint_digest,
        previous_pointer=previous_pointer,
        manifest_root=publication.manifest_root,
        manifest_count=publication.manifest_count,
        manifest=manifest,
    )


async def _require_live_publication_authority(
    session: AsyncSession,
    scope: PublicationScope,
) -> AgentRunAttempt:
    attempt = await session.scalar(
        select(AgentRunAttempt)
        .where(
            AgentRunAttempt.attempt_id == scope.attempt_id,
            AgentRunAttempt.thread_id == scope.thread_id,
            AgentRunAttempt.fence_version == scope.fence_version,
        )
        .with_for_update()
    )
    if attempt is None:
        raise PublicationConflict("publication scope does not belong to the fenced attempt")
    execution = await session.scalar(
        select(AgentThreadExecution)
        .where(
            AgentThreadExecution.thread_id == scope.thread_id,
            AgentThreadExecution.owner_attempt_id == scope.attempt_id,
            AgentThreadExecution.fence_version == scope.fence_version,
            AgentThreadExecution.lease_expires_at.is_not(None),
            AgentThreadExecution.lease_expires_at > _DB_NOW,
        )
        .with_for_update()
    )
    if execution is None:
        raise PublicationConflict("publication lost owner, fence, or database-time lease authority")
    return attempt


async def _require_live_effect_authority(
    session: AsyncSession,
    scope: EffectWriteScope,
    run_id: str,
) -> AgentRunAttempt:
    current_attempt = await session.scalar(
        select(AgentRunAttempt)
        .where(
            AgentRunAttempt.attempt_id == scope.attempt_id,
            AgentRunAttempt.run_id == run_id,
            AgentRunAttempt.thread_id == scope.thread_id,
            AgentRunAttempt.fence_version == scope.fence_version,
        )
        .with_for_update()
    )
    if current_attempt is None:
        raise EffectConflict("effect scope does not belong to the logical run")
    execution = await session.scalar(
        select(AgentThreadExecution)
        .where(
            AgentThreadExecution.thread_id == scope.thread_id,
            AgentThreadExecution.owner_attempt_id == scope.attempt_id,
            AgentThreadExecution.fence_version == scope.fence_version,
            AgentThreadExecution.lease_expires_at.is_not(None),
            AgentThreadExecution.lease_expires_at > _DB_NOW,
        )
        .with_for_update()
    )
    if execution is None:
        raise EffectConflict("effect write lost owner, fence, or database-time lease authority")
    return current_attempt


async def _find_effect(
    session: AsyncSession,
    claim: CanonicalEffectClaim,
) -> AgentEffect | None:
    effect = await session.scalar(
        select(AgentEffect).where(AgentEffect.idempotency_key == claim.idempotency_key).with_for_update()
    )
    if effect is not None:
        _validate_effect_contract(effect, claim)
    return effect


async def _lock_action_target(
    session: AsyncSession,
    write: ActionPrepareEffectWrite,
) -> None:
    """Lock and revalidate the target order inside the caller's transaction.

    The row is locked FOR UPDATE in the same MySQL Session/transaction that
    performs the authority checks, compatibility lookup and the effect/request
    inserts. Owner and identity are persistence invariants validated here; the
    action-state admission policy itself stays in the application precheck and
    is anchored by ``validated_order_status`` so a state change after the
    precheck fails closed instead of writing a stale PENDING request.
    """
    order = await session.scalar(
        select(CustomerOrder)
        .where(
            CustomerOrder.id == write.target_order_id,
            CustomerOrder.order_no == write.target_order_no,
            CustomerOrder.user_id == write.subject_user_id,
        )
        .with_for_update()
    )
    if order is None:
        raise EffectConflict(
            "action target order is missing or no longer owned by the subject"
        )
    if order.status != write.validated_order_status:
        raise EffectConflict("action target order state changed before the fenced write")


async def _r2_compatibility_key(
    session: AsyncSession,
    write: ActionPrepareEffectWrite,
) -> R2DayCompatibilityKey:
    """Derive the day-scoped compatibility key from authoritative MySQL time.

    The day component always comes from the database clock inside this
    transaction so that concurrent runs on the same server agree; client,
    model, or tool values never participate in this key.
    """
    now = await session.scalar(select(_DB_NOW))
    if now is None:
        raise EffectConflict("database time is unavailable for the R2 compatibility key")
    return build_r2_day_compatibility_key(
        action_type=write.action_type,
        target_order_id=write.target_order_id,
        subject_user_id=write.subject_user_id,
        day=now.date().isoformat(),
    )


def _durable_action_request_key(write: ActionPrepareEffectWrite) -> str:
    if write.confirmation_mode != DURABLE_INTERRUPT_CONFIRMATION_MODE:
        raise EffectConflict("durable action request key requires durable confirmation mode")
    return f"durable:{write.logical_action_id}"


def _durable_lookup_from_write(
    write: ActionPrepareEffectWrite,
    current_attempt: AgentRunAttempt,
) -> DurableActionPrepareReplayLookup:
    if (
        write.confirmation_mode != DURABLE_INTERRUPT_CONFIRMATION_MODE
        or write.customer_confirmation_challenge_digest is None
        or write.draft_revision is None
        or write.draft_expires_at is None
    ):
        raise EffectConflict("durable action replay evidence is incomplete")
    return DurableActionPrepareReplayLookup(
        scope=write.scope,
        conversation_id=current_attempt.conversation_id,
        run_id=write.claim.run_id,
        node_name=write.claim.node_name,
        purpose=write.claim.purpose,
        sequence=write.claim.sequence,
        logical_action_id=write.logical_action_id,
        action_type=write.action_type,
        target_order_id=write.target_order_id,
        target_order_no=write.target_order_no,
        subject_user_id=write.subject_user_id,
        created_by=write.created_by,
        action_payload_json=write.action_payload_json,
        risk_level=write.risk_level,
        policy_version=write.policy_version,
        draft_revision=write.draft_revision,
        draft_expires_at=write.draft_expires_at,
        customer_confirmation_challenge_digest=(
            write.customer_confirmation_challenge_digest
        ),
    )


async def _lock_durable_action_prepare_replay(
    session: AsyncSession,
    lookup: DurableActionPrepareReplayLookup,
    *,
    current_attempt: AgentRunAttempt,
    required_target_id: int | None = None,
) -> EffectWriteResult | None:
    if (
        lookup.conversation_id != current_attempt.conversation_id
        or lookup.created_by != current_attempt.actor_user_id
        or lookup.subject_user_id != current_attempt.subject_user_id
        or current_attempt.actor_role != "CUSTOMER"
        or current_attempt.service_principal is not None
    ):
        raise EffectConflict("durable replay does not match current customer authority")

    effect = await session.scalar(
        select(AgentEffect)
        .where(AgentEffect.idempotency_key == lookup.effect_idempotency_key)
        .with_for_update()
    )
    durable_key = f"durable:{lookup.logical_action_id}"
    if effect is None:
        collision_predicate = (
            (AgentActionRequest.logical_action_id == lookup.logical_action_id)
            | (AgentActionRequest.idempotency_key == durable_key)
        )
        if required_target_id is not None:
            collision_predicate = collision_predicate | (
                AgentActionRequest.id == required_target_id
            )
        collision = await session.scalar(
            select(AgentActionRequest)
            .where(collision_predicate)
            .with_for_update()
        )
        if collision is not None or required_target_id is not None:
            raise EffectCorruption(
                "durable action target exists without its canonical effect identity"
            )
        return None

    target = await session.scalar(
        select(AgentActionRequest)
        .where(AgentActionRequest.effect_id == effect.id)
        .with_for_update()
    )
    if target is None:
        raise EffectCorruption("action-prepare effect exists without its target")
    if required_target_id is not None and target.id != required_target_id:
        raise EffectCorruption("pending action does not match its canonical effect")
    if target.prepared_order_status is None:
        raise EffectCorruption("durable action has no prepared order-status evidence")

    write = lookup.write_with_prepared_order_status(target.prepared_order_status)
    _validate_action_prepare_write_contract(write)
    originating_attempt = await _load_originating_action_attempt(session, effect)
    if (
        originating_attempt.thread_id != lookup.scope.thread_id
        or originating_attempt.conversation_id != lookup.conversation_id
        or current_attempt.thread_id != lookup.scope.thread_id
        or current_attempt.conversation_id != lookup.conversation_id
    ):
        raise EffectCorruption(
            "durable action provenance does not match the conversation domain"
        )
    _validate_action_target(
        target,
        effect,
        write,
        originating_attempt=originating_attempt,
        current_attempt=current_attempt,
    )
    if target.effect_idempotency_key is None:
        raise EffectCorruption("action target has no effect idempotency linkage")
    return EffectWriteResult(
        effect_id=effect.id,
        target_id=target.id,
        idempotency_key=target.effect_idempotency_key,
        replayed=True,
    )


async def _lock_action_prepare_replay(
    session: AsyncSession,
    write: ActionPrepareEffectWrite,
    *,
    current_attempt: AgentRunAttempt,
) -> EffectWriteResult | None:
    effect = await _find_effect(session, write.claim)
    if effect is None:
        if write.confirmation_mode == R2_STATELESS_COMPAT_CONFIRMATION_MODE:
            compatibility_key = await _r2_compatibility_key(session, write)
            return await _find_compatibility_replay(
                session,
                write,
                compatibility_key,
                current_attempt=current_attempt,
            )
        durable_key = _durable_action_request_key(write)
        collision = await session.scalar(
            select(AgentActionRequest)
            .where(
                (
                    AgentActionRequest.logical_action_id
                    == write.logical_action_id
                )
                | (AgentActionRequest.idempotency_key == durable_key)
            )
            .with_for_update()
        )
        if collision is not None:
            raise EffectCorruption(
                "durable action target exists without its canonical effect identity"
            )
        return None
    existing = await session.scalar(
        select(AgentActionRequest)
        .where(AgentActionRequest.effect_id == effect.id)
        .with_for_update()
    )
    if existing is None:
        raise EffectCorruption("action-prepare effect exists without its target")
    originating_attempt = await _load_originating_action_attempt(session, effect)
    _validate_action_target(
        existing,
        effect,
        write,
        originating_attempt=originating_attempt,
        current_attempt=current_attempt,
    )
    if existing.effect_idempotency_key is None:
        raise EffectCorruption("action target has no effect idempotency linkage")
    return EffectWriteResult(
        effect_id=effect.id,
        target_id=existing.id,
        idempotency_key=existing.effect_idempotency_key,
        replayed=True,
    )


async def _create_action_prepare_target(
    session: AsyncSession,
    write: ActionPrepareEffectWrite,
    effect: AgentEffect,
    *,
    guard: TransactionBoundStoreGuard,
) -> AgentActionRequest:
    if write.confirmation_mode == DURABLE_INTERRUPT_CONFIRMATION_MODE:
        if write.customer_confirmation_challenge_digest is None:
            raise EffectConflict("durable action confirmation evidence is incomplete")
        confirmed_at = await session.scalar(select(_DB_NOW))
        if confirmed_at is None:
            raise EffectConflict("database time is unavailable for customer confirmation")
        logical_action_id: str | None = write.logical_action_id
        confirmation_mode = DURABLE_INTERRUPT_CONFIRMATION_MODE
        resume_status = "WAITING_ADMIN_DECISION"
        idempotency_key = _durable_action_request_key(write)
        confirmed_actor_id: int | None = write.created_by
        challenge_digest: str | None = (
            write.customer_confirmation_challenge_digest
        )
    else:
        compatibility_key = await _r2_compatibility_key(session, write)
        logical_action_id = None
        confirmation_mode = R2_STATELESS_COMPAT_CONFIRMATION_MODE
        resume_status = "NOT_APPLICABLE"
        idempotency_key = compatibility_key.value
        confirmed_actor_id = None
        confirmed_at = None
        challenge_digest = None
    existing = AgentActionRequest(
        run_id=effect.run_id,
        action_type=write.action_type,
        target_order_id=write.target_order_id,
        action_payload_json=write.action_payload_json,
        risk_level=write.risk_level,
        status="PENDING",
        logical_action_id=logical_action_id,
        confirmation_mode=confirmation_mode,
        customer_confirmed_actor_id=confirmed_actor_id,
        customer_confirmed_at=confirmed_at,
        customer_confirmation_challenge_digest=challenge_digest,
        resume_status=resume_status,
        prepared_order_status=(
            write.validated_order_status
            if confirmation_mode == DURABLE_INTERRUPT_CONFIRMATION_MODE
            else None
        ),
        idempotency_key=idempotency_key,
        lock_version=0,
        created_by=write.created_by,
        attempt_id=effect.attempt_id,
        effect_id=effect.id,
        effect_node_name=effect.node_name,
        effect_purpose=effect.purpose,
        effect_sequence=effect.sequence,
        effect_idempotency_key=effect.idempotency_key,
    )
    session.add(existing)
    await guard.flush(session)
    return existing


async def _find_compatibility_replay(
    session: AsyncSession,
    write: ActionPrepareEffectWrite,
    key: R2DayCompatibilityKey,
    *,
    current_attempt: AgentRunAttempt,
) -> EffectWriteResult | None:
    """EXACT replay of the first run's committed compatibility target.

    A cross-run retry (response loss, or a concurrent new run) may only return
    the first committed ActionRequest when every persisted dimension matches:
    the linked AgentEffect must exist and agree with the request's linkage and
    semantics (effect type, purpose, sequence, identity keys), and the request
    must match action type, target order, subject/creator, payload, risk level
    and run/attempt identity. This path deliberately writes nothing, so no
    second AgentEffect (orphan) or ActionRequest can appear. The unique index
    on ``agent_action_request.idempotency_key`` arbitrates concurrent first
    creates; the loser reads the winner back through this exact lookup in a
    fresh transaction.
    """
    existing = await session.scalar(
        select(AgentActionRequest).where(
            AgentActionRequest.idempotency_key == key.value
        )
    )
    if existing is None:
        return None
    if existing.effect_id is None or existing.effect_idempotency_key is None:
        raise EffectCorruption("compatibility replay target has no effect linkage")
    effect = await session.scalar(
        select(AgentEffect).where(AgentEffect.id == existing.effect_id)
    )
    if effect is None:
        raise EffectCorruption("compatibility replay target references a missing effect")
    originating_attempt = await _load_originating_action_attempt(session, effect)
    _validate_compatibility_replay(
        existing,
        effect,
        write,
        key,
        originating_attempt=originating_attempt,
        current_attempt=current_attempt,
    )
    return EffectWriteResult(
        effect_id=effect.id,
        target_id=existing.id,
        idempotency_key=effect.idempotency_key,
        replayed=True,
    )


async def _load_originating_action_attempt(
    session: AsyncSession,
    effect: AgentEffect,
) -> AgentRunAttempt:
    originating_attempt = await session.scalar(
        select(AgentRunAttempt)
        .where(
            AgentRunAttempt.attempt_id == effect.attempt_id,
            AgentRunAttempt.run_id == effect.run_id,
        )
        .with_for_update()
    )
    if originating_attempt is None:
        raise EffectCorruption("action effect references a missing originating attempt")
    return originating_attempt


def _validate_action_prepare_write_contract(
    write: ActionPrepareEffectWrite,
) -> None:
    expected_effect_key = canonical_effect_idempotency_key(
        run_id=write.claim.run_id,
        node_name=write.claim.node_name,
        purpose=write.claim.purpose,
        sequence=write.claim.sequence,
    )
    if (
        write.claim.effect_type is not EffectType.ACTION_PREPARE
        or write.claim.idempotency_key != expected_effect_key
        or write.claim.payload_digest != write.replay_digest()
    ):
        raise EffectConflict("action-prepare write contract is invalid")


def _validate_action_replay_binding(
    target: AgentActionRequest,
    effect: AgentEffect,
    write: ActionPrepareEffectWrite,
    *,
    originating_attempt: AgentRunAttempt,
    current_attempt: AgentRunAttempt,
) -> None:
    expected_effect_key = canonical_effect_idempotency_key(
        run_id=effect.run_id,
        node_name=effect.node_name,
        purpose=effect.purpose,
        sequence=effect.sequence,
    )
    if (
        effect.effect_type != EffectType.ACTION_PREPARE.value
        or effect.purpose != write.claim.purpose
        or effect.node_name != write.claim.node_name
        or effect.sequence != write.claim.sequence
    ):
        raise EffectCorruption("action replay effect semantics are invalid")
    if (
        effect.idempotency_key != expected_effect_key
        or target.effect_idempotency_key != expected_effect_key
    ):
        raise EffectCorruption("action replay effect identity is not canonical")
    if effect.payload_digest != write.replay_digest():
        raise EffectCorruption("action replay contract digest is invalid")
    if (
        originating_attempt.attempt_id != effect.attempt_id
        or originating_attempt.run_id != effect.run_id
        or originating_attempt.actor_user_id != current_attempt.actor_user_id
        or originating_attempt.subject_user_id != current_attempt.subject_user_id
        or originating_attempt.actor_user_id != write.created_by
        or originating_attempt.subject_user_id != write.subject_user_id
        or originating_attempt.actor_role != "CUSTOMER"
        or current_attempt.actor_role != "CUSTOMER"
        or originating_attempt.service_principal is not None
        or current_attempt.service_principal is not None
    ):
        raise EffectConflict("action replay provenance does not match customer authority")


def _validate_compatibility_replay(
    target: AgentActionRequest,
    effect: AgentEffect,
    write: ActionPrepareEffectWrite,
    key: R2DayCompatibilityKey,
    *,
    originating_attempt: AgentRunAttempt,
    current_attempt: AgentRunAttempt,
) -> None:
    if target.idempotency_key != key.value:
        raise EffectConflict("compatibility replay target has an unexpected key")
    if (
        target.confirmation_mode != R2_STATELESS_COMPAT_CONFIRMATION_MODE
        or target.logical_action_id is not None
        or target.customer_confirmed_actor_id is not None
        or target.customer_confirmed_at is not None
        or target.customer_confirmation_challenge_digest is not None
        or target.resume_status != "NOT_APPLICABLE"
        or target.status != "PENDING"
        or target.action_type != write.action_type
        or target.target_order_id != write.target_order_id
        or target.created_by != write.created_by
        or target.action_payload_json != write.action_payload_json
        or target.risk_level != write.risk_level
    ):
        raise EffectCorruption("compatibility replay target does not match the locked request facts")
    if (
        target.run_id != effect.run_id
        or target.attempt_id != effect.attempt_id
        or target.effect_id != effect.id
        or target.effect_node_name != effect.node_name
        or target.effect_purpose != effect.purpose
        or target.effect_sequence != effect.sequence
        or target.effect_idempotency_key != effect.idempotency_key
    ):
        raise EffectCorruption("compatibility replay linkage does not match its effect")
    _validate_action_replay_binding(
        target,
        effect,
        write,
        originating_attempt=originating_attempt,
        current_attempt=current_attempt,
    )


async def _validate_action_prepare_proof(
    session: AsyncSession,
    write: ActionPrepareEffectWrite,
    proof: ActionPrepareProof,
    current_attempt: AgentRunAttempt,
) -> None:
    proof.assert_digest()
    expected = (
        write.scope.thread_id,
        write.claim.run_id,
        write.scope.attempt_id,
        write.scope.fence_version,
        write.created_by,
        write.subject_user_id,
        write.logical_action_id,
        write.action_type,
        write.target_order_id,
        write.target_order_no.upper(),
        write.claim.payload_digest,
        write.policy_version,
    )
    actual = (
        proof.thread_id,
        proof.run_id,
        proof.attempt_id,
        proof.fence_version,
        proof.actor_user_id,
        proof.subject_user_id,
        proof.logical_action_id,
        proof.action_type,
        proof.target_order_id,
        proof.target_order_no.upper(),
        proof.payload_digest,
        proof.policy_version,
    )
    if expected != actual:
        raise EffectConflict("action authorization proof does not match locked request facts")
    if current_attempt.actor_user_id != proof.actor_user_id or current_attempt.subject_user_id != proof.subject_user_id:
        raise EffectConflict("action authorization proof actor or subject is invalid")
    if not proof.authorization_id.startswith("auth_"):
        raise EffectConflict("action authorization proof identity is invalid")
    now = await session.scalar(select(_DB_NOW))
    if now is None:
        raise EffectConflict("database time is unavailable for action proof validation")
    expires_at = proof.expires_at.astimezone(UTC).replace(tzinfo=None)
    if expires_at <= now:
        raise EffectConflict("action authorization proof expired by database time")
    execution = await session.scalar(
        select(AgentThreadExecution)
        .where(
            AgentThreadExecution.thread_id == write.scope.thread_id,
            AgentThreadExecution.owner_attempt_id == write.scope.attempt_id,
            AgentThreadExecution.fence_version == write.scope.fence_version,
            AgentThreadExecution.lease_expires_at.is_not(None),
            AgentThreadExecution.lease_expires_at > _DB_NOW,
        )
        .with_for_update()
    )
    if execution is None:
        raise EffectConflict("action authority expired after policy validation")


async def _create_effect(
    session: AsyncSession,
    scope: EffectWriteScope,
    claim: CanonicalEffectClaim,
    *,
    guard: TransactionBoundStoreGuard | None = None,
) -> AgentEffect:
    effect = AgentEffect(
        run_id=claim.run_id,
        attempt_id=scope.attempt_id,
        node_name=claim.node_name,
        purpose=claim.purpose,
        sequence=claim.sequence,
        effect_type=claim.effect_type.value,
        idempotency_key=claim.idempotency_key,
        payload_digest=claim.payload_digest,
    )
    session.add(effect)
    if guard is None:
        await session.flush()
    else:
        await guard.flush(session)
    return effect


def _validate_effect_contract(effect: AgentEffect, claim: CanonicalEffectClaim) -> None:
    existing_contract = (
        effect.run_id,
        effect.node_name,
        effect.purpose,
        effect.sequence,
        effect.effect_type,
        effect.idempotency_key,
        effect.payload_digest,
    )
    requested_contract = (
        claim.run_id,
        claim.node_name,
        claim.purpose,
        claim.sequence,
        claim.effect_type.value,
        claim.idempotency_key,
        claim.payload_digest,
    )
    if existing_contract != requested_contract:
        raise EffectConflict("effect identity was replayed with different semantics")


def _validate_message_target(
    target: ChatMessage,
    effect: AgentEffect,
    write: MessageEffectWrite,
) -> None:
    existing_contract = (
        target.conversation_id,
        target.role,
        target.content,
        target.sources_json,
        target.retrieval_score,
        target.confidence_level,
        target.need_human,
        target.source_run_id,
        target.source_attempt_id,
        target.message_purpose,
        target.message_sequence,
        target.message_idempotency_key,
        target.effect_id,
    )
    requested_contract = (
        write.conversation_id,
        write.role,
        write.content,
        write.sources_json,
        write.retrieval_score,
        write.confidence_level,
        write.need_human,
        effect.run_id,
        effect.attempt_id,
        effect.purpose,
        effect.sequence,
        effect.idempotency_key,
        effect.id,
    )
    if existing_contract != requested_contract:
        raise EffectCorruption("message target does not match its committed effect")


def _validate_audit_target(
    target: AgentStep,
    effect: AgentEffect,
    write: AuditEffectWrite,
) -> None:
    existing_contract = (
        target.run_id,
        target.node_name,
        target.input_summary,
        target.output_summary,
        target.status,
        target.duration_ms,
        target.error_summary,
        target.attempt_id,
        target.effect_id,
        target.effect_purpose,
        target.effect_sequence,
        target.effect_idempotency_key,
    )
    requested_contract = (
        effect.run_id,
        effect.node_name,
        write.input_summary,
        write.output_summary,
        write.status,
        write.duration_ms,
        write.error_summary,
        effect.attempt_id,
        effect.id,
        effect.purpose,
        effect.sequence,
        effect.idempotency_key,
    )
    if existing_contract != requested_contract:
        raise EffectCorruption("audit target does not match its committed effect")


def _validate_action_target(
    target: AgentActionRequest,
    effect: AgentEffect,
    write: ActionPrepareEffectWrite,
    *,
    originating_attempt: AgentRunAttempt,
    current_attempt: AgentRunAttempt,
) -> None:
    _validate_action_replay_binding(
        target,
        effect,
        write,
        originating_attempt=originating_attempt,
        current_attempt=current_attempt,
    )
    existing_contract = (
        target.run_id,
        target.action_type,
        target.target_order_id,
        target.action_payload_json,
        target.risk_level,
        target.created_by,
        target.attempt_id,
        target.effect_id,
        target.effect_node_name,
        target.effect_purpose,
        target.effect_sequence,
        target.effect_idempotency_key,
    )
    requested_contract = (
        effect.run_id,
        write.action_type,
        write.target_order_id,
        write.action_payload_json,
        write.risk_level,
        write.created_by,
        effect.attempt_id,
        effect.id,
        effect.node_name,
        effect.purpose,
        effect.sequence,
        effect.idempotency_key,
    )
    if existing_contract != requested_contract:
        raise EffectCorruption("action target does not match its committed effect")
    if write.confirmation_mode == DURABLE_INTERRUPT_CONFIRMATION_MODE:
        expected_key = _durable_action_request_key(write)
        if (
            target.idempotency_key != expected_key
            or target.logical_action_id != write.logical_action_id
            or target.confirmation_mode != DURABLE_INTERRUPT_CONFIRMATION_MODE
            or target.customer_confirmed_actor_id != write.created_by
            or target.customer_confirmed_at is None
            or target.customer_confirmation_challenge_digest
            != write.customer_confirmation_challenge_digest
            or target.resume_status != "WAITING_ADMIN_DECISION"
            or target.prepared_order_status != write.validated_order_status
            or target.status != "PENDING"
            or target.admin_decision is not None
            or target.admin_decided_actor_id is not None
            or target.admin_decided_at is not None
            or target.admin_reason_code is not None
            or target.approved_by is not None
            or target.approved_at is not None
            or target.approval_note is not None
            or target.executed_at is not None
            or target.execution_result_code is not None
            or target.execution_error_type is not None
            or target.execution_error_summary is not None
            or target.legacy_original_status is not None
        ):
            raise EffectCorruption(
                "durable action target does not match its customer confirmation contract"
            )
        return
    if target.prepared_order_status is not None:
        raise EffectCorruption(
            "R2 action target cannot carry durable prepared status evidence"
        )
    # The stored day-scoped compatibility key is validated by its parsed
    # components instead of a recomputed day so that an exact effect replay on
    # a later date still validates its own committed facts.
    stored_compatibility = parse_r2_day_compatibility_key(target.idempotency_key)
    if (
        stored_compatibility is None
        or stored_compatibility.action_type != write.action_type
        or stored_compatibility.target_order_id != write.target_order_id
        or stored_compatibility.subject_user_id != write.subject_user_id
    ):
        raise EffectCorruption("action target compatibility key does not match its effect facts")


def _run_record(run: AgentRun) -> RunRecord:
    return RunRecord(
        run_id=run.run_id,
        thread_id=run.thread_id,
        conversation_id=run.conversation_id,
        subject_user_id=run.user_id,
        status=run.status,
        request_id=run.request_id,
    )


def _attempt_record(attempt: AgentRunAttempt) -> AttemptRecord:
    return AttemptRecord(
        attempt_id=attempt.attempt_id,
        run_id=attempt.run_id,
        thread_id=attempt.thread_id,
        conversation_id=attempt.conversation_id,
        status=attempt.status,
        fence_version=attempt.fence_version,
    )


def _lease_state(execution: AgentThreadExecution) -> LeaseState:
    return LeaseState(
        thread_id=execution.thread_id,
        owner_attempt_id=execution.owner_attempt_id,
        fence_version=execution.fence_version,
        lease_expires_at=execution.lease_expires_at,
    )


def _validate_fence(fence_version: int) -> None:
    if type(fence_version) is not int or fence_version <= 0:
        raise DurableContractError("fence version must be a positive integer")


def _validate_lease_duration(lease_milliseconds: int) -> None:
    if type(lease_milliseconds) is not int or not 0 < lease_milliseconds <= 86_400_000:
        raise DurableContractError("lease duration must be between 1 ms and 24 hours")
