from __future__ import annotations

from app.agent.checkpoint_projection import (
    CheckpointContentSlot,
    collect_checkpoint_content_slots,
)
from app.agent.state import ConversationCheckpointState
from app.runtime.checkpoint_projection import ContentRole, ContentSlotReferences, ContentSourceReferenceV1
from app.runtime.content_source import (
    CandidateWriteSurface,
    ChatMessageSourceCommand,
    ContentSourceAppendResult,
    ContentSourceAuthorityStorePort,
    ContentSourceIntegrityError,
    ContentSourceRevisionRecord,
    ContentSourceRevisionWrite,
    PublicationContentReferenceSet,
    ServiceContentSourceCommand,
    SourceBeforeCandidateRecorderPort,
)
from app.runtime.context import ExecutionScope
from app.runtime.uow import (
    ApplicationTransactionCoordinator,
    ApplicationUnitOfWorkFactory,
    require_no_active_transaction,
)


class ContentSourceAuthorityService:
    """Application orchestration for source commits and exact-read gating."""

    def __init__(
        self,
        source_uow: ApplicationUnitOfWorkFactory[ContentSourceAuthorityStorePort],
        transactions: ApplicationTransactionCoordinator,
    ) -> None:
        self._source_uow = source_uow
        self._transactions = transactions

    async def append_chat_message_revision(
        self,
        command: ChatMessageSourceCommand,
    ) -> ContentSourceAppendResult:
        """Derive producer/domain/digest from the locked trusted origin."""

        if not isinstance(command, ChatMessageSourceCommand):
            raise TypeError("CHAT_MESSAGE source append requires the typed command")
        return await self._transactions.run(
            "application.transaction",
            self._source_uow,
            lambda store: _append_chat_message(store, command),
        )

    async def append_revision(
        self,
        write: ContentSourceRevisionWrite,
    ) -> ContentSourceAppendResult:
        return await self._transactions.run(
            "application.transaction",
            self._source_uow,
            lambda store: _append_one(store, write),
        )

    async def append_service_revision(
        self,
        command: ServiceContentSourceCommand,
    ) -> ContentSourceAppendResult:
        if not isinstance(command, ServiceContentSourceCommand):
            raise TypeError("service source append requires the typed command")
        return await self._transactions.run(
            "application.transaction",
            self._source_uow,
            lambda store: store.append_service_revision(command),
        )

    async def materialize_checkpoint_state(
        self,
        state: ConversationCheckpointState,
        execution: ExecutionScope,
    ) -> ContentSlotReferences:
        """Commit every referenced content slot in one short MySQL UoW.

        QUESTION is derived from the replay-safe USER_INPUT ChatMessage.  The
        remaining registered V1 slots are produced by the closed
        CHECKPOINT_RUNTIME service authority.  The returned references are
        only usable after this transaction has committed and closed.
        """

        identity = state.conversation_identity
        active_run = state.active_run
        if active_run is None:
            if collect_checkpoint_content_slots(state):
                raise ContentSourceIntegrityError(
                    "checkpoint without an active run has unsupported content"
                )
            return ContentSlotReferences()
        if (
            identity.conversation_id != execution.conversation_id
            or identity.thread_id != execution.thread_id
            or identity.subject_user_id != execution.subject.user_id
            or active_run.run_id != execution.run_id
            or active_run.attempt_id != execution.attempt_id
            or execution.lease.fence_token is None
        ):
            raise ContentSourceIntegrityError(
                "checkpoint source scope does not match the fenced execution"
            )
        slots = collect_checkpoint_content_slots(state)
        mapping = await self._transactions.run(
            "application.transaction",
            self._source_uow,
            lambda store: _materialize_slots(
                store,
                state=state,
                execution=execution,
                slots=slots,
            ),
        )
        return ContentSlotReferences(mapping)

    async def read_exact_revision(
        self,
        reference: ContentSourceReferenceV1,
    ) -> ContentSourceRevisionRecord:
        return await self._transactions.run(
            "application.transaction",
            self._source_uow,
            lambda store: store.read_exact_revision(reference),
        )

    async def read_publication_holds(
        self,
        *,
        thread_id: str,
        publication_version: int,
    ) -> PublicationContentReferenceSet:
        return await self._transactions.run(
            "application.transaction",
            self._source_uow,
            lambda store: store.read_publication_holds(
                thread_id=thread_id,
                publication_version=publication_version,
            ),
        )

    async def commit_sources_then_record_candidate(
        self,
        *,
        revisions: tuple[ContentSourceRevisionWrite, ...],
        references: PublicationContentReferenceSet,
        surface: CandidateWriteSurface,
        recorder: SourceBeforeCandidateRecorderPort,
    ) -> None:
        """Commit, re-open and exact-read every source before candidate I/O.

        The recorder is deliberately a narrow stage port used by 5.2-A proof;
        it is not a SQLite provider and receives no bytes or Runtime State.
        """

        if not isinstance(references, PublicationContentReferenceSet):
            raise TypeError("candidate gate requires a typed publication reference set")
        await self._transactions.run(
            "application.transaction",
            self._source_uow,
            lambda store: _append_all(store, revisions),
        )
        await self._transactions.run(
            "application.transaction",
            self._source_uow,
            lambda store: store.require_exact_references(references),
        )
        require_no_active_transaction("content-source candidate stage")
        if surface is CandidateWriteSurface.CHECKPOINT:
            await recorder.record_checkpoint_candidate(references)
            return
        if surface is CandidateWriteSurface.PENDING_WRITES:
            await recorder.record_pending_write_candidate(references)
            return
        raise ValueError("unsupported candidate write surface")


async def _append_all(
    store: ContentSourceAuthorityStorePort,
    revisions: tuple[ContentSourceRevisionWrite, ...],
) -> tuple[ContentSourceAppendResult, ...]:
    results: list[ContentSourceAppendResult] = []
    for revision in revisions:
        results.append(await _append_one(store, revision))
    return tuple(results)


async def _append_one(
    store: ContentSourceAuthorityStorePort,
    revision: ContentSourceRevisionWrite,
) -> ContentSourceAppendResult:
    reference = revision.reference
    if type(reference.source_record_id) is not int:
        raise ContentSourceIntegrityError(
            "source kind has no registered typed origin authority"
        )
    origin = await store.lock_chat_message_origin(
        ChatMessageSourceCommand(
            origin_chat_message_id=reference.source_record_id,
            content_role=reference.content_role,
        )
    )
    if origin.reference != reference or origin.raw_content != revision.raw_content:
        raise ContentSourceIntegrityError(
            "source write does not match origin-derived authority"
        )
    return await store.append_revision(revision, origin=origin)


async def _append_chat_message(
    store: ContentSourceAuthorityStorePort,
    command: ChatMessageSourceCommand,
) -> ContentSourceAppendResult:
    origin = await store.lock_chat_message_origin(command)
    return await store.append_revision(
        ContentSourceRevisionWrite(
            reference=origin.reference,
            raw_content=origin.raw_content,
        ),
        origin=origin,
    )


async def _materialize_slots(
    store: ContentSourceAuthorityStorePort,
    *,
    state: ConversationCheckpointState,
    execution: ExecutionScope,
    slots: tuple[CheckpointContentSlot, ...],
) -> dict[str, ContentSourceReferenceV1]:
    active_run = state.active_run
    if active_run is None:
        return {}
    mapping: dict[str, ContentSourceReferenceV1] = {}
    for slot in slots:
        slot_name = slot.slot
        role = slot.content_role
        content = slot.content
        if role is ContentRole.QUESTION:
            if active_run.current_user_message_id is None:
                raise ContentSourceIntegrityError(
                    "QUESTION requires a replay-safe USER_INPUT message"
                )
            origin = await store.lock_chat_message_origin(
                ChatMessageSourceCommand(
                    origin_chat_message_id=active_run.current_user_message_id,
                    content_role=ContentRole.QUESTION,
                )
            )
            if origin.raw_content != content:
                raise ContentSourceIntegrityError(
                    "QUESTION bytes differ from the authoritative USER_INPUT message"
                )
            result = await store.append_revision(
                ContentSourceRevisionWrite(
                    reference=origin.reference,
                    raw_content=origin.raw_content,
                ),
                origin=origin,
            )
        else:
            fence = execution.lease.fence_token
            if fence is None:
                raise ContentSourceIntegrityError(
                    "service content requires a fenced execution"
                )
            result = await store.append_service_revision(
                ServiceContentSourceCommand(
                    conversation_id=execution.conversation_id,
                    thread_id=execution.thread_id,
                    subject_user_id=execution.subject.user_id,
                    run_id=execution.run_id,
                    attempt_id=execution.attempt_id,
                    fence_version=fence,
                    content_role=role,
                    raw_content=content,
                )
            )
        mapping[slot_name] = result.record.reference
    return mapping
