from __future__ import annotations

import json
from typing import NoReturn, cast

from pydantic import ValidationError
from sqlalchemy import literal_column, select
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from app.db.models.entities import (
    AgentCheckpointContentReference,
    AgentCheckpointPublication,
    AgentContentSourceRevision,
    AgentEffect,
    AgentRunAttempt,
    AgentThreadExecution,
    ChatConversation,
    ChatMessage,
)
from app.runtime.checkpoint_projection import (
    ContentSourceReferenceV1,
    NormalizationVersion,
    ProducingPrincipalKind,
    SourceKind,
    compute_content_digest,
)
from app.runtime.content_source import (
    ChatMessageSourceCommand,
    ContentOriginAuthorityRecord,
    ContentSourceAppendResult,
    ContentSourceConflict,
    ContentSourceIntegrityError,
    ContentSourceMissing,
    ContentSourceRevisionRecord,
    ContentSourceRevisionWrite,
    PublicationContentReference,
    PublicationContentReferenceSet,
    RegisteredServicePrincipal,
    ServiceContentSourceCommand,
    derive_service_source_record_id,
    encode_source_record_id,
    require_canonical_projection_slot,
    require_chat_message_role_purpose,
)
from app.runtime.durable import canonical_digest
from app.runtime.uow import TransactionBoundStoreGuard

_DB_NOW: ColumnElement[object] = literal_column("CURRENT_TIMESTAMP(6)")


class SqlAlchemyCheckpointContentSourceStore:
    """Transaction-bound adapter for immutable exact source revisions only."""

    def __init__(
        self,
        session: AsyncSession,
        guard: TransactionBoundStoreGuard,
    ) -> None:
        self._session = session
        self._guard = guard

    async def lock_chat_message_origin(
        self,
        command: ChatMessageSourceCommand,
    ) -> ContentOriginAuthorityRecord:
        self._guard.ensure_active()
        return await _lock_and_build_chat_message_origin(
            self._session,
            command,
            for_update=True,
        )

    async def lock_exact_origin(
        self,
        write: ContentSourceRevisionWrite,
    ) -> ContentOriginAuthorityRecord:
        """Compatibility seam for already-projected references.

        The supplied producer fields are claims only.  The adapter derives a
        fresh canonical reference from locked origin/effect facts and requires
        exact equality before returning authority.
        """

        self._guard.ensure_active()
        origin = await _lock_and_build_chat_message_origin(
            self._session,
            ChatMessageSourceCommand(
                origin_chat_message_id=_strict_chat_message_id(write.reference),
                content_role=write.reference.content_role,
            ),
            for_update=True,
        )
        if origin.reference != write.reference or origin.raw_content != write.raw_content:
            raise ContentSourceIntegrityError(
                "source write does not match origin-derived authority"
            )
        return origin

    async def append_revision(
        self,
        write: ContentSourceRevisionWrite,
        *,
        origin: ContentOriginAuthorityRecord,
    ) -> ContentSourceAppendResult:
        self._guard.ensure_active()
        if (
            not isinstance(origin, ContentOriginAuthorityRecord)
            or origin.reference != write.reference
            or origin.raw_content != write.raw_content
            or origin.origin_chat_message_id != write.reference.source_record_id
        ):
            raise ContentSourceIntegrityError(
                "source append requires the exact origin authority from this write"
            )
        existing = await _find_source_row(
            self._session,
            write.reference,
            for_update=True,
        )
        if existing is not None:
            record = _validated_record(existing)
            if record.reference != write.reference or record.raw_content != write.raw_content:
                raise ContentSourceConflict("exact source identity is already bound to different content")
            return ContentSourceAppendResult(record=record, replayed=True)

        reference = write.reference
        statement = mysql_insert(AgentContentSourceRevision).values(
            source_kind=reference.source_kind.value,
            source_record_id=encode_source_record_id(reference.source_record_id),
            source_revision=reference.source_revision,
            content_role=reference.content_role.value,
            content_schema_version=reference.content_schema_version,
            normalization_version=reference.normalization_version.value,
            content_sha256=reference.content_sha256,
            raw_content_utf8=write.raw_content.encode("utf-8", errors="strict"),
            conversation_id=reference.conversation_id,
            subject_user_id=reference.subject_user_id,
            producing_principal_kind=reference.producing_principal_kind.value,
            producing_actor_id=reference.producing_actor_id,
            producing_service_principal=reference.producing_service_principal,
            run_id=reference.run_id,
            producing_attempt_id=reference.producing_attempt_id,
            origin_chat_message_id=origin.origin_chat_message_id,
        )
        result = await self._session.execute(statement.prefix_with("IGNORE"))
        row = await _find_source_row(
            self._session,
            reference,
            for_update=True,
        )
        if row is None:
            raise ContentSourceConflict("exact source revision insert was rejected")
        record = _validated_record(row)
        if (
            record.reference != reference
            or record.raw_content != write.raw_content
            or row.origin_chat_message_id != origin.origin_chat_message_id
        ):
            raise ContentSourceConflict("exact source identity is already bound to different content")
        return ContentSourceAppendResult(
            record=record,
            replayed=result.rowcount == 0,
        )

    async def append_service_revision(
        self,
        command: ServiceContentSourceCommand,
    ) -> ContentSourceAppendResult:
        """Append/replay one server-derived source under a live fenced attempt."""

        self._guard.ensure_active()
        if not isinstance(command, ServiceContentSourceCommand):
            raise TypeError("service source append requires the typed command")
        attempt = await self._session.scalar(
            select(AgentRunAttempt)
            .where(
                AgentRunAttempt.attempt_id == command.attempt_id,
                AgentRunAttempt.run_id == command.run_id,
                AgentRunAttempt.thread_id == command.thread_id,
                AgentRunAttempt.conversation_id == command.conversation_id,
                AgentRunAttempt.subject_user_id == command.subject_user_id,
                AgentRunAttempt.fence_version == command.fence_version,
            )
            .with_for_update()
        )
        if attempt is None:
            raise ContentSourceIntegrityError(
                "service source attempt binding is invalid"
            )
        execution = await self._session.scalar(
            select(AgentThreadExecution)
            .where(
                AgentThreadExecution.thread_id == command.thread_id,
                AgentThreadExecution.conversation_id == command.conversation_id,
                AgentThreadExecution.owner_attempt_id == command.attempt_id,
                AgentThreadExecution.fence_version == command.fence_version,
                AgentThreadExecution.lease_expires_at.is_not(None),
                AgentThreadExecution.lease_expires_at > _DB_NOW,
            )
            .with_for_update()
        )
        if execution is None:
            raise ContentSourceIntegrityError(
                "service source lost owner, fence, or database-time lease"
            )
        conversation = await self._session.scalar(
            select(ChatConversation)
            .where(ChatConversation.id == command.conversation_id)
            .with_for_update()
        )
        if (
            conversation is None
            or conversation.user_id is None
            or conversation.user_id != command.subject_user_id
        ):
            raise ContentSourceIntegrityError(
                "service source conversation subject is invalid"
            )

        digest = compute_content_digest(
            source_kind=SourceKind.AGENT_AUDIT_CONTENT,
            content_role=command.content_role,
            content_schema_version=1,
            normalization_version=NormalizationVersion.RAW_UTF8_V1,
            content=command.raw_content,
        )
        source_record_id = derive_service_source_record_id(
            conversation_id=command.conversation_id,
            run_id=command.run_id,
            content_role=command.content_role,
            content_sha256=digest,
        )
        existing = await _find_source_row_by_key(
            self._session,
            source_kind=SourceKind.AGENT_AUDIT_CONTENT.value,
            source_record_id=encode_source_record_id(source_record_id),
            source_revision=1,
            for_update=True,
        )
        if existing is not None:
            record = _validated_record(existing)
            _validate_service_record_against_command(record, command)
            return ContentSourceAppendResult(record=record, replayed=True)

        reference = ContentSourceReferenceV1(
            source_kind=SourceKind.AGENT_AUDIT_CONTENT,
            source_record_id=source_record_id,
            content_role=command.content_role,
            content_schema_version=1,
            content_sha256=digest,
            conversation_id=command.conversation_id,
            subject_user_id=command.subject_user_id,
            producing_principal_kind=ProducingPrincipalKind.SERVICE,
            producing_actor_id=None,
            producing_service_principal=(
                RegisteredServicePrincipal.CHECKPOINT_RUNTIME.value
            ),
            run_id=command.run_id,
            producing_attempt_id=command.attempt_id,
            source_revision=1,
            normalization_version=NormalizationVersion.RAW_UTF8_V1,
        )
        statement = mysql_insert(AgentContentSourceRevision).values(
            source_kind=reference.source_kind.value,
            source_record_id=encode_source_record_id(reference.source_record_id),
            source_revision=reference.source_revision,
            content_role=reference.content_role.value,
            content_schema_version=reference.content_schema_version,
            normalization_version=reference.normalization_version.value,
            content_sha256=reference.content_sha256,
            raw_content_utf8=command.raw_content.encode("utf-8", errors="strict"),
            conversation_id=reference.conversation_id,
            subject_user_id=reference.subject_user_id,
            producing_principal_kind=reference.producing_principal_kind.value,
            producing_actor_id=None,
            producing_service_principal=reference.producing_service_principal,
            run_id=reference.run_id,
            producing_attempt_id=reference.producing_attempt_id,
            origin_chat_message_id=None,
        )
        result = await self._session.execute(statement.prefix_with("IGNORE"))
        row = await _find_source_row_by_key(
            self._session,
            source_kind=reference.source_kind.value,
            source_record_id=encode_source_record_id(reference.source_record_id),
            source_revision=1,
            for_update=True,
        )
        if row is None:
            raise ContentSourceConflict("service source insert was rejected")
        record = _validated_record(row)
        _validate_service_record_against_command(record, command)
        return ContentSourceAppendResult(
            record=record,
            replayed=result.rowcount == 0,
        )

    async def read_exact_revision(
        self,
        reference: ContentSourceReferenceV1,
    ) -> ContentSourceRevisionRecord:
        self._guard.ensure_active()
        row = await _find_source_row(self._session, reference, for_update=False)
        if row is None:
            raise ContentSourceMissing("exact source revision is missing")
        record = _validated_record(row, reference)
        if record.reference.source_kind is SourceKind.CHAT_MESSAGE:
            await _lock_and_validate_origin(
                self._session,
                record.reference,
                expected_raw_content=record.raw_content,
                lock_attempt=False,
                for_update=False,
            )
        elif record.reference.source_kind is SourceKind.AGENT_AUDIT_CONTENT:
            await _validate_service_source_authority(self._session, record)
        else:
            raise ContentSourceIntegrityError(
                "source kind has no registered V1 exact-read authority"
            )
        return record

    async def require_exact_references(
        self,
        references: PublicationContentReferenceSet,
    ) -> tuple[ContentSourceRevisionRecord, ...]:
        self._guard.ensure_active()
        records: list[ContentSourceRevisionRecord] = []
        for item in references:
            row = await _find_source_row(
                self._session,
                item.reference,
                for_update=False,
            )
            if row is None:
                raise ContentSourceMissing("publication references a missing exact source revision")
            record = _validated_record(row, item.reference)
            await _lock_and_validate_origin(
                self._session,
                record.reference,
                expected_raw_content=record.raw_content,
                lock_attempt=False,
                for_update=False,
            )
            records.append(record)
        return tuple(records)

    async def read_publication_holds(
        self,
        *,
        thread_id: str,
        publication_version: int,
    ) -> PublicationContentReferenceSet:
        self._guard.ensure_active()
        _validate_publication_identity(thread_id, publication_version)
        publication = await self._session.scalar(
            select(AgentCheckpointPublication).where(
                AgentCheckpointPublication.thread_id == thread_id
            )
        )
        execution = await self._session.scalar(
            select(AgentThreadExecution).where(
                AgentThreadExecution.thread_id == thread_id
            )
        )
        if (
            publication is None
            or execution is None
            or publication.publication_version != publication_version
            or publication.conversation_id != execution.conversation_id
        ):
            raise ContentSourceIntegrityError(
                "publication hold read is not bound to the current publication domain"
            )
        conversation = await self._session.get(
            ChatConversation,
            publication.conversation_id,
        )
        if conversation is None or conversation.user_id is None:
            raise ContentSourceIntegrityError(
                "publication conversation has no authoritative subject"
            )
        rows = (
            await self._session.scalars(
                select(AgentCheckpointContentReference)
                .where(
                    AgentCheckpointContentReference.thread_id == thread_id,
                    AgentCheckpointContentReference.publication_version == publication_version,
                )
                .order_by(AgentCheckpointContentReference.reference_ordinal)
            )
        ).all()
        items: list[PublicationContentReference] = []
        for expected_ordinal, hold in enumerate(rows):
            if hold.reference_ordinal != expected_ordinal:
                raise ContentSourceIntegrityError("publication hold ordinals are incomplete")
            source = await _find_source_row_by_key(
                self._session,
                source_kind=hold.source_kind,
                source_record_id=hold.source_record_id,
                source_revision=hold.source_revision,
                for_update=False,
            )
            if source is None:
                raise ContentSourceIntegrityError("publication hold points to a missing source")
            record = _validated_record(source)
            await _lock_and_validate_origin(
                self._session,
                record.reference,
                expected_raw_content=record.raw_content,
                lock_attempt=False,
                for_update=False,
            )
            if (
                record.reference.content_role.value != hold.content_role
                or record.reference.content_sha256 != hold.content_sha256
                or record.reference.conversation_id != publication.conversation_id
                or record.reference.subject_user_id != conversation.user_id
            ):
                raise ContentSourceIntegrityError("publication hold disagrees with its exact source")
            require_canonical_projection_slot(
                hold.projection_slot,
                record.reference.content_role,
            )
            items.append(
                PublicationContentReference(
                    projection_slot=hold.projection_slot,
                    reference=record.reference,
                )
            )
        return PublicationContentReferenceSet.of(*items)


async def lock_publication_content_sources(
    session: AsyncSession,
    references: PublicationContentReferenceSet,
    *,
    expected_conversation_id: int,
    expected_subject_user_id: int,
) -> tuple[AgentContentSourceRevision, ...]:
    """Lock every exact source before pointer CAS to serialize deletion/publish."""

    locked: list[AgentContentSourceRevision] = []
    for item in references:
        origin = await _lock_and_validate_origin(
            session,
            item.reference,
            expected_raw_content=None,
            lock_attempt=False,
            for_update=True,
        )
        row = await _find_source_row(
            session,
            item.reference,
            for_update=True,
        )
        if row is None:
            raise ContentSourceMissing("publication references a missing exact source revision")
        record = _validated_record(row, item.reference)
        if (
            record.reference.conversation_id != expected_conversation_id
            or record.reference.subject_user_id != expected_subject_user_id
            or record.raw_content != origin.raw_content
            or row.origin_chat_message_id != origin.origin_chat_message_id
        ):
            raise ContentSourceIntegrityError("publication source crosses conversation or subject authority")
        locked.append(row)
    return tuple(locked)


async def insert_publication_content_holds(
    session: AsyncSession,
    *,
    thread_id: str,
    publication_version: int,
    references: PublicationContentReferenceSet,
    locked_sources: tuple[AgentContentSourceRevision, ...],
) -> None:
    _validate_publication_identity(thread_id, publication_version)
    if len(references) != len(locked_sources):
        raise ContentSourceIntegrityError("locked source set does not match publication references")
    for ordinal, (item, row) in enumerate(zip(references, locked_sources, strict=True)):
        _validated_record(row, item.reference)
        reference = item.reference
        require_canonical_projection_slot(item.projection_slot, reference.content_role)
        session.add(
            AgentCheckpointContentReference(
                thread_id=thread_id,
                publication_version=publication_version,
                reference_ordinal=ordinal,
                projection_slot=item.projection_slot,
                content_role=reference.content_role.value,
                source_kind=reference.source_kind.value,
                source_record_id=encode_source_record_id(reference.source_record_id),
                source_revision=reference.source_revision,
                content_sha256=reference.content_sha256,
            )
        )
    if len(references) > 0:
        await session.flush()


async def _find_source_row(
    session: AsyncSession,
    reference: ContentSourceReferenceV1,
    *,
    for_update: bool,
) -> AgentContentSourceRevision | None:
    return await _find_source_row_by_key(
        session,
        source_kind=reference.source_kind.value,
        source_record_id=encode_source_record_id(reference.source_record_id),
        source_revision=reference.source_revision,
        for_update=for_update,
    )


async def _lock_and_validate_origin(
    session: AsyncSession,
    reference: ContentSourceReferenceV1,
    *,
    expected_raw_content: str | None,
    lock_attempt: bool,
    for_update: bool,
) -> ContentOriginAuthorityRecord:
    del lock_attempt
    origin = await _lock_and_build_chat_message_origin(
        session,
        ChatMessageSourceCommand(
            origin_chat_message_id=_strict_chat_message_id(reference),
            content_role=reference.content_role,
        ),
        for_update=for_update,
    )
    if origin.reference != reference:
        raise ContentSourceIntegrityError(
            "source reference disagrees with origin-derived authority"
        )
    if expected_raw_content is not None and origin.raw_content != expected_raw_content:
        raise ContentSourceIntegrityError(
            "CHAT_MESSAGE origin bytes differ from the source revision"
        )
    return origin


async def _lock_and_build_chat_message_origin(
    session: AsyncSession,
    command: ChatMessageSourceCommand,
    *,
    for_update: bool,
) -> ContentOriginAuthorityRecord:
    # This read discovers lock keys only and grants no authority.  Authoritative
    # rows are re-read below in the common attempt -> execution -> origin order.
    snapshot = cast(
        ChatMessage | None,
        await session.scalar(
            select(ChatMessage).where(
                ChatMessage.id == command.origin_chat_message_id
            )
        ),
    )
    if snapshot is None:
        raise ContentSourceIntegrityError("CHAT_MESSAGE origin is missing")
    if (
        snapshot.source_run_id is None
        or snapshot.source_attempt_id is None
        or snapshot.effect_id is None
    ):
        raise ContentSourceIntegrityError(
            "CHAT_MESSAGE origin lacks immutable run/attempt/effect authority"
        )
    discovered = (
        snapshot.source_run_id,
        snapshot.source_attempt_id,
        snapshot.effect_id,
        snapshot.conversation_id,
    )

    attempt_statement = select(AgentRunAttempt).where(
        AgentRunAttempt.attempt_id == snapshot.source_attempt_id,
        AgentRunAttempt.run_id == snapshot.source_run_id,
    )
    if for_update:
        attempt_statement = attempt_statement.with_for_update()
    attempt = cast(AgentRunAttempt | None, await session.scalar(attempt_statement))
    if attempt is None:
        raise ContentSourceIntegrityError("source producing attempt is missing")

    execution_statement = select(AgentThreadExecution).where(
        AgentThreadExecution.thread_id == attempt.thread_id
    )
    conversation_statement = select(ChatConversation).where(
        ChatConversation.id == snapshot.conversation_id
    )
    effect_statement = select(AgentEffect).where(
        AgentEffect.id == snapshot.effect_id
    )
    message_statement = (
        select(ChatMessage)
        .where(ChatMessage.id == command.origin_chat_message_id)
        .execution_options(populate_existing=True)
    )
    if for_update:
        execution_statement = execution_statement.with_for_update()
        conversation_statement = conversation_statement.with_for_update()
        effect_statement = effect_statement.with_for_update()
        message_statement = message_statement.with_for_update()
    execution = cast(
        AgentThreadExecution | None,
        await session.scalar(execution_statement),
    )
    conversation = cast(
        ChatConversation | None,
        await session.scalar(conversation_statement),
    )
    effect = cast(AgentEffect | None, await session.scalar(effect_statement))
    message = cast(ChatMessage | None, await session.scalar(message_statement))
    if message is None:
        raise ContentSourceIntegrityError("CHAT_MESSAGE origin disappeared")
    if (
        message.source_run_id,
        message.source_attempt_id,
        message.effect_id,
        message.conversation_id,
    ) != discovered:
        raise ContentSourceIntegrityError(
            "CHAT_MESSAGE origin changed while authority was being locked"
        )
    if conversation is None or conversation.user_id is None:
        raise ContentSourceIntegrityError(
            "CHAT_MESSAGE origin has no authoritative subject"
        )
    if (
        execution is None
        or execution.conversation_id != attempt.conversation_id
        or attempt.conversation_id != message.conversation_id
    ):
        raise ContentSourceIntegrityError(
            "source attempt/execution is not bound to the origin conversation"
        )
    _validate_execution_actor(attempt)
    expected_role, expected_purpose = require_chat_message_role_purpose(
        command.content_role
    )
    expected_effect_key = (
        canonical_digest(
            {
                "node_name": effect.node_name,
                "purpose": effect.purpose,
                "run_id": effect.run_id,
                "sequence": effect.sequence,
            }
        )
        if effect is not None
        else None
    )
    expected_payload_digest = canonical_digest(
        {
            "confidence_level": message.confidence_level,
            "content": message.content,
            "conversation_id": message.conversation_id,
            "need_human": message.need_human,
            "retrieval_score": (
                str(message.retrieval_score)
                if message.retrieval_score is not None
                else None
            ),
            "role": message.role,
            "sources_json": message.sources_json,
        }
    )
    if (
        message.role != expected_role
        or message.message_purpose != expected_purpose
        or attempt.subject_user_id != conversation.user_id
        or effect is None
        or effect.effect_type != "CHAT_MESSAGE"
        or effect.run_id != message.source_run_id
        or effect.attempt_id != message.source_attempt_id
        or effect.purpose != expected_purpose
        or effect.sequence != message.message_sequence
        or effect.idempotency_key != message.message_idempotency_key
        or effect.idempotency_key != expected_effect_key
        or effect.payload_digest != expected_payload_digest
    ):
        raise ContentSourceIntegrityError(
            "CHAT_MESSAGE origin domain, role, purpose, identity, or digest is invalid"
        )

    if expected_role == "ASSISTANT":
        producing_kind = ProducingPrincipalKind.SERVICE
        producing_actor_id = None
        producing_service_principal = (
            RegisteredServicePrincipal.CHECKPOINT_RUNTIME.value
        )
    else:
        if (
            attempt.actor_role != "CUSTOMER"
            or attempt.actor_user_id != conversation.user_id
        ):
            raise ContentSourceIntegrityError(
                "USER content origin is not bound to the conversation customer"
            )
        producing_kind = ProducingPrincipalKind.USER
        producing_actor_id = attempt.actor_user_id
        producing_service_principal = None

    reference = ContentSourceReferenceV1(
        source_kind=SourceKind.CHAT_MESSAGE,
        source_record_id=message.id,
        content_role=command.content_role,
        content_schema_version=1,
        content_sha256=compute_content_digest(
            source_kind=SourceKind.CHAT_MESSAGE,
            content_role=command.content_role,
            content_schema_version=1,
            normalization_version=NormalizationVersion.RAW_UTF8_V1,
            content=message.content,
        ),
        conversation_id=message.conversation_id,
        subject_user_id=conversation.user_id,
        producing_principal_kind=producing_kind,
        producing_actor_id=producing_actor_id,
        producing_service_principal=producing_service_principal,
        run_id=message.source_run_id,
        producing_attempt_id=message.source_attempt_id,
        source_revision=1,
        normalization_version=NormalizationVersion.RAW_UTF8_V1,
    )
    return ContentOriginAuthorityRecord(
        reference=reference,
        origin_chat_message_id=message.id,
        raw_content=message.content,
    )


def _validate_execution_actor(
    attempt: AgentRunAttempt,
) -> None:
    if attempt.actor_role in {"CUSTOMER", "ADMIN"}:
        if attempt.actor_user_id is None or attempt.service_principal is not None:
            raise ContentSourceIntegrityError(
                "current execution user actor is incomplete"
            )
        return
    if (
        attempt.actor_role != "SYSTEM"
        or attempt.actor_user_id is not None
        or attempt.service_principal
        != RegisteredServicePrincipal.CHECKPOINT_RUNTIME.value
    ):
        raise ContentSourceIntegrityError(
            "current execution service actor is not registered"
        )


def _strict_chat_message_id(reference: ContentSourceReferenceV1) -> int:
    if (
        reference.source_kind is not SourceKind.CHAT_MESSAGE
        or type(reference.source_record_id) is not int
        or reference.source_record_id <= 0
        or reference.source_revision != 1
        or reference.content_schema_version != 1
    ):
        raise ContentSourceIntegrityError(
            "source kind has no registered V1 typed origin authority"
        )
    return reference.source_record_id


async def _find_source_row_by_key(
    session: AsyncSession,
    *,
    source_kind: str,
    source_record_id: str,
    source_revision: int,
    for_update: bool,
) -> AgentContentSourceRevision | None:
    statement = select(AgentContentSourceRevision).where(
        AgentContentSourceRevision.source_kind == source_kind,
        AgentContentSourceRevision.source_record_id == source_record_id,
        AgentContentSourceRevision.source_revision == source_revision,
    )
    if for_update:
        statement = statement.with_for_update()
    return cast(AgentContentSourceRevision | None, await session.scalar(statement))


def _validated_record(
    row: AgentContentSourceRevision,
    expected: ContentSourceReferenceV1 | None = None,
) -> ContentSourceRevisionRecord:
    reason: str | None = None
    reference: ContentSourceReferenceV1 | None = None
    raw_content: str | None = None
    try:
        source_record_id = json.loads(row.source_record_id)
        if type(source_record_id) not in (int, str):
            raise ValueError("invalid source record identity")
        if row.source_kind == SourceKind.CHAT_MESSAGE.value:
            if (
                type(source_record_id) is not int
                or row.origin_chat_message_id != source_record_id
            ):
                raise ValueError("source origin identity is invalid")
        elif row.source_kind == SourceKind.AGENT_AUDIT_CONTENT.value:
            if type(source_record_id) is not str or row.origin_chat_message_id is not None:
                raise ValueError("service source physical identity is invalid")
        else:
            raise ValueError("source kind has no registered V1 authority")
        raw_content = bytes(row.raw_content_utf8).decode("utf-8", errors="strict")
        reference = ContentSourceReferenceV1.model_validate(
            {
                "source_kind": row.source_kind,
                "source_record_id": source_record_id,
                "content_role": row.content_role,
                "content_schema_version": row.content_schema_version,
                "content_sha256": row.content_sha256,
                "conversation_id": row.conversation_id,
                "subject_user_id": row.subject_user_id,
                "producing_principal_kind": row.producing_principal_kind,
                "producing_actor_id": row.producing_actor_id,
                "producing_service_principal": row.producing_service_principal,
                "run_id": row.run_id,
                "producing_attempt_id": row.producing_attempt_id,
                "source_revision": row.source_revision,
                "normalization_version": row.normalization_version,
            }
        )
        actual_digest = compute_content_digest(
            source_kind=reference.source_kind,
            content_role=reference.content_role,
            content_schema_version=reference.content_schema_version,
            normalization_version=reference.normalization_version,
            content=raw_content,
        )
        if actual_digest != reference.content_sha256:
            raise ValueError("stored source digest mismatch")
        if expected is not None and reference != expected:
            raise ValueError("stored source binding mismatch")
    except (UnicodeDecodeError, ValueError, TypeError, json.JSONDecodeError, ValidationError):
        reason = "exact source revision failed integrity validation"
    if reason is not None or reference is None or raw_content is None:
        _raise_integrity(reason or "exact source revision is invalid")
    return ContentSourceRevisionRecord(reference=reference, raw_content=raw_content)


def _validate_service_record_against_command(
    record: ContentSourceRevisionRecord,
    command: ServiceContentSourceCommand,
) -> None:
    reference = record.reference
    expected_digest = compute_content_digest(
        source_kind=SourceKind.AGENT_AUDIT_CONTENT,
        content_role=command.content_role,
        content_schema_version=1,
        normalization_version=NormalizationVersion.RAW_UTF8_V1,
        content=command.raw_content,
    )
    expected_record_id = derive_service_source_record_id(
        conversation_id=command.conversation_id,
        run_id=command.run_id,
        content_role=command.content_role,
        content_sha256=expected_digest,
    )
    if (
        reference.source_kind is not SourceKind.AGENT_AUDIT_CONTENT
        or reference.source_record_id != expected_record_id
        or reference.source_revision != 1
        or reference.content_role is not command.content_role
        or reference.content_sha256 != expected_digest
        or reference.conversation_id != command.conversation_id
        or reference.subject_user_id != command.subject_user_id
        or reference.run_id != command.run_id
        or reference.producing_principal_kind is not ProducingPrincipalKind.SERVICE
        or reference.producing_actor_id is not None
        or reference.producing_service_principal
        != RegisteredServicePrincipal.CHECKPOINT_RUNTIME.value
        or record.raw_content != command.raw_content
    ):
        raise ContentSourceConflict(
            "service source identity is already bound to different content"
        )


async def _validate_service_source_authority(
    session: AsyncSession,
    record: ContentSourceRevisionRecord,
) -> None:
    reference = record.reference
    if (
        reference.source_kind is not SourceKind.AGENT_AUDIT_CONTENT
        or type(reference.source_record_id) is not str
        or reference.source_revision != 1
        or reference.producing_principal_kind is not ProducingPrincipalKind.SERVICE
        or reference.producing_actor_id is not None
        or reference.producing_service_principal
        != RegisteredServicePrincipal.CHECKPOINT_RUNTIME.value
    ):
        raise ContentSourceIntegrityError(
            "service source authority binding is invalid"
        )
    expected_record_id = derive_service_source_record_id(
        conversation_id=reference.conversation_id,
        run_id=reference.run_id,
        content_role=reference.content_role,
        content_sha256=reference.content_sha256,
    )
    if reference.source_record_id != expected_record_id:
        raise ContentSourceIntegrityError(
            "service source deterministic identity is invalid"
        )
    attempt = await session.scalar(
        select(AgentRunAttempt).where(
            AgentRunAttempt.attempt_id == reference.producing_attempt_id,
            AgentRunAttempt.run_id == reference.run_id,
            AgentRunAttempt.conversation_id == reference.conversation_id,
            AgentRunAttempt.subject_user_id == reference.subject_user_id,
        )
    )
    conversation = await session.scalar(
        select(ChatConversation).where(
            ChatConversation.id == reference.conversation_id,
            ChatConversation.user_id == reference.subject_user_id,
        )
    )
    if attempt is None or conversation is None:
        raise ContentSourceIntegrityError(
            "service source producer or conversation authority is missing"
        )


def _raise_integrity(message: str) -> NoReturn:
    error = ContentSourceIntegrityError(message)
    try:
        raise error
    except ContentSourceIntegrityError:
        error.__context__ = None
        raise


def _validate_publication_identity(thread_id: str, publication_version: int) -> None:
    if type(thread_id) is not str or not thread_id or len(thread_id) > 128:
        raise ContentSourceConflict("publication thread identity is invalid")
    if type(publication_version) is not int or publication_version <= 0:
        raise ContentSourceConflict("publication version must be a strict positive integer")
