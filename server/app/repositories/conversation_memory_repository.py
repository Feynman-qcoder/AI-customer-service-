from __future__ import annotations

import json
from datetime import UTC

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.entities import (
    AgentContentSourceRevision,
    AgentRunAttempt,
    ChatConversation,
    ChatMessage,
    ConversationRollingSummary,
    ConversationWorkingMemory,
)
from app.memory import (
    ConversationMemoryAccessDenied,
    ConversationMemoryIntegrityError,
    ConversationMemoryRevisionConflict,
    ConversationMemoryScopeV1,
    MemoryProvenanceV1,
    RecentMessageSourceV1,
    RollingSummaryV1,
    WorkingMemorySourceMessageV1,
    WorkingMemoryV1,
)
from app.runtime.checkpoint_projection import (
    ContentRole,
    NormalizationVersion,
    SourceKind,
    compute_content_digest,
)
from app.runtime.content_source import (
    RegisteredServicePrincipal,
    encode_source_record_id,
)
from app.runtime.uow import TransactionBoundStoreGuard

_MYSQL_SIGNED_BIGINT_MAX = 9_223_372_036_854_775_807
_ACCESS_DENIED = "conversation memory access denied"
_REVISION_CONFLICT = "conversation memory revision conflict"
_INVALID_DATA = "conversation memory data is invalid"


class SqlAlchemyConversationMemoryStore:
    """Transaction-bound adapter for owner-scoped memory rows and exact summary references."""

    def __init__(
        self,
        session: AsyncSession,
        guard: TransactionBoundStoreGuard,
    ) -> None:
        self._session = session
        self._guard = guard

    async def load_working_memory(
        self,
        scope: ConversationMemoryScopeV1,
    ) -> WorkingMemoryV1 | None:
        _require_scope(scope)
        self._guard.ensure_active()
        await self._assert_owned_conversation(scope, for_update=False)
        row = await self._session.scalar(
            select(ConversationWorkingMemory).where(
                ConversationWorkingMemory.conversation_id == scope.conversation_id
            )
        )
        self._guard.ensure_active()
        if row is None:
            return None
        if row.subject_user_id != scope.subject_user_id:
            raise ConversationMemoryIntegrityError(_INVALID_DATA)
        return _working_memory_from_row(row)

    async def write_working_memory(
        self,
        scope: ConversationMemoryScopeV1,
        *,
        expected_revision: int,
        memory: WorkingMemoryV1,
    ) -> WorkingMemoryV1:
        _require_scope(scope)
        _require_expected_revision(expected_revision)
        if not isinstance(memory, WorkingMemoryV1):
            raise TypeError("working memory write requires WorkingMemoryV1")
        _require_next_revision(expected_revision, memory.memory_revision)

        self._guard.ensure_active()
        await self._assert_owned_conversation(scope, for_update=True)
        row = await self._session.scalar(
            select(ConversationWorkingMemory)
            .where(ConversationWorkingMemory.conversation_id == scope.conversation_id)
            .with_for_update()
        )
        if row is None:
            if expected_revision != 0:
                raise ConversationMemoryRevisionConflict(_REVISION_CONFLICT)
            row = ConversationWorkingMemory(
                conversation_id=scope.conversation_id,
                subject_user_id=scope.subject_user_id,
                memory_revision=memory.memory_revision,
            )
            self._session.add(row)
        elif (
            row.subject_user_id != scope.subject_user_id
            or row.memory_revision != expected_revision
        ):
            raise ConversationMemoryRevisionConflict(_REVISION_CONFLICT)

        _apply_working_memory(row, memory)
        try:
            await self._guard.flush(self._session)
        except DBAPIError:
            raise ConversationMemoryIntegrityError(_INVALID_DATA) from None
        return _working_memory_from_row(row)

    async def load_user_input_messages(
        self,
        scope: ConversationMemoryScopeV1,
        *,
        message_ids: tuple[int, ...],
    ) -> tuple[WorkingMemorySourceMessageV1, ...]:
        _require_scope(scope)
        if (
            type(message_ids) is not tuple
            or len(message_ids) > 4
            or len(set(message_ids)) != len(message_ids)
            or any(
                type(message_id) is not int
                or message_id <= 0
                or message_id > _MYSQL_SIGNED_BIGINT_MAX
                for message_id in message_ids
            )
        ):
            raise ConversationMemoryIntegrityError(_INVALID_DATA)
        self._guard.ensure_active()
        await self._assert_owned_conversation(scope, for_update=False)
        if not message_ids:
            return ()
        rows = (
            await self._session.scalars(
                select(ChatMessage).where(
                    ChatMessage.id.in_(message_ids),
                    ChatMessage.conversation_id == scope.conversation_id,
                    ChatMessage.role == "USER",
                    ChatMessage.message_purpose == "USER_INPUT",
                    ChatMessage.effect_id.is_not(None),
                )
            )
        ).all()
        by_id = {int(row.id): row for row in rows}
        if set(by_id) != set(message_ids):
            raise ConversationMemoryIntegrityError(_INVALID_DATA)
        try:
            messages = tuple(
                WorkingMemorySourceMessageV1(
                    message_id=message_id,
                    content=by_id[message_id].content,
                    created_at=(
                        by_id[message_id].created_at.replace(tzinfo=UTC)
                        if by_id[message_id].created_at.tzinfo is None
                        else by_id[message_id].created_at.astimezone(UTC)
                    ),
                )
                for message_id in message_ids
            )
        except (ValidationError, TypeError, ValueError):
            raise ConversationMemoryIntegrityError(_INVALID_DATA) from None
        self._guard.ensure_active()
        return messages

    async def load_rolling_summary(
        self,
        scope: ConversationMemoryScopeV1,
    ) -> RollingSummaryV1 | None:
        _require_scope(scope)
        self._guard.ensure_active()
        await self._assert_owned_conversation(scope, for_update=False)
        row = await self._session.scalar(
            select(ConversationRollingSummary).where(
                ConversationRollingSummary.conversation_id == scope.conversation_id
            )
        )
        if row is None:
            self._guard.ensure_active()
            return None
        if row.subject_user_id != scope.subject_user_id:
            raise ConversationMemoryIntegrityError(_INVALID_DATA)
        summary = _rolling_summary_from_row(row)
        await self._validate_summary_links(scope, summary, for_update=False)
        self._guard.ensure_active()
        return summary

    async def load_recent_messages(
        self,
        scope: ConversationMemoryScopeV1,
        *,
        after_message_id: int | None,
        before_message_id: int | None = None,
    ) -> tuple[RecentMessageSourceV1, ...]:
        _require_scope(scope)
        _require_optional_cursor(after_message_id)
        _require_optional_cursor(before_message_id)
        if (
            after_message_id is not None
            and before_message_id is not None
            and after_message_id >= before_message_id
        ):
            raise ConversationMemoryIntegrityError(_INVALID_DATA)
        self._guard.ensure_active()
        await self._assert_owned_conversation(scope, for_update=False)
        statement = select(ChatMessage).where(
            ChatMessage.conversation_id == scope.conversation_id,
            ChatMessage.role.in_(("USER", "ASSISTANT")),
        )
        if after_message_id is not None:
            statement = statement.where(ChatMessage.id > after_message_id)
        if before_message_id is not None:
            statement = statement.where(ChatMessage.id < before_message_id)
        rows = (await self._session.scalars(statement.order_by(ChatMessage.id.asc()))).all()
        try:
            messages = tuple(
                RecentMessageSourceV1.model_validate(
                    {
                        "message_id": row.id,
                        "role": row.role,
                        "content": row.content,
                        "created_at": (
                            row.created_at.replace(tzinfo=UTC)
                            if row.created_at.tzinfo is None
                            else row.created_at.astimezone(UTC)
                        ),
                    }
                )
                for row in rows
            )
        except (ValidationError, TypeError, ValueError):
            raise ConversationMemoryIntegrityError(_INVALID_DATA) from None
        self._guard.ensure_active()
        return messages

    async def load_rolling_summary_content(
        self,
        scope: ConversationMemoryScopeV1,
        summary: RollingSummaryV1,
    ) -> str:
        _require_scope(scope)
        if not isinstance(summary, RollingSummaryV1):
            raise TypeError("rolling summary content read requires RollingSummaryV1")
        self._guard.ensure_active()
        await self._assert_owned_conversation(scope, for_update=False)
        row = await self._session.scalar(
            select(AgentContentSourceRevision).where(
                AgentContentSourceRevision.source_kind == summary.source_kind,
                AgentContentSourceRevision.source_record_id
                == encode_source_record_id(summary.source_record_id),
                AgentContentSourceRevision.source_revision == summary.source_revision,
                AgentContentSourceRevision.content_role == summary.content_role,
                AgentContentSourceRevision.content_schema_version
                == summary.content_schema_version,
                AgentContentSourceRevision.normalization_version
                == summary.normalization_version,
                AgentContentSourceRevision.content_sha256 == summary.content_sha256,
                AgentContentSourceRevision.conversation_id == scope.conversation_id,
                AgentContentSourceRevision.subject_user_id == scope.subject_user_id,
            )
        )
        if row is None:
            raise ConversationMemoryIntegrityError(_INVALID_DATA)
        try:
            content = row.raw_content_utf8.decode("utf-8", errors="strict")
            digest = compute_content_digest(
                source_kind=SourceKind(row.source_kind),
                content_role=ContentRole(row.content_role),
                content_schema_version=row.content_schema_version,
                normalization_version=NormalizationVersion(row.normalization_version),
                content=content,
            )
        except (UnicodeDecodeError, TypeError, ValueError):
            raise ConversationMemoryIntegrityError(_INVALID_DATA) from None
        if (
            not content
            or digest != summary.content_sha256
            or row.producing_principal_kind != "SERVICE"
            or row.producing_actor_id is not None
            or row.producing_service_principal
            != RegisteredServicePrincipal.CHECKPOINT_RUNTIME.value
            or row.origin_chat_message_id is not None
            or row.run_id is None
            or row.producing_attempt_id is None
        ):
            raise ConversationMemoryIntegrityError(_INVALID_DATA)
        attempt_id = await self._session.scalar(
            select(AgentRunAttempt.attempt_id).where(
                AgentRunAttempt.attempt_id == row.producing_attempt_id,
                AgentRunAttempt.run_id == row.run_id,
                AgentRunAttempt.conversation_id == scope.conversation_id,
                AgentRunAttempt.subject_user_id == scope.subject_user_id,
            )
        )
        if attempt_id is None:
            raise ConversationMemoryIntegrityError(_INVALID_DATA)
        self._guard.ensure_active()
        return content

    async def write_rolling_summary(
        self,
        scope: ConversationMemoryScopeV1,
        *,
        expected_revision: int,
        summary: RollingSummaryV1,
    ) -> RollingSummaryV1:
        _require_scope(scope)
        _require_expected_revision(expected_revision)
        if not isinstance(summary, RollingSummaryV1):
            raise TypeError("rolling summary write requires RollingSummaryV1")
        _require_next_revision(expected_revision, summary.summary_revision)

        self._guard.ensure_active()
        await self._assert_owned_conversation(scope, for_update=True)
        await self._validate_summary_links(scope, summary, for_update=True)
        row = await self._session.scalar(
            select(ConversationRollingSummary)
            .where(ConversationRollingSummary.conversation_id == scope.conversation_id)
            .with_for_update()
        )
        if row is None:
            if expected_revision != 0:
                raise ConversationMemoryRevisionConflict(_REVISION_CONFLICT)
            row = ConversationRollingSummary(
                conversation_id=scope.conversation_id,
                subject_user_id=scope.subject_user_id,
                summary_revision=summary.summary_revision,
                summary_until_message_id=summary.summary_until_message_id,
                source_kind=summary.source_kind,
                source_record_id=encode_source_record_id(summary.source_record_id),
                source_revision=summary.source_revision,
                content_role=summary.content_role,
                content_schema_version=summary.content_schema_version,
                normalization_version=summary.normalization_version,
                content_sha256=summary.content_sha256,
                token_counter_version=summary.token_counter_version,
            )
            self._session.add(row)
        else:
            if (
                row.subject_user_id != scope.subject_user_id
                or row.summary_revision != expected_revision
                or summary.summary_until_message_id <= row.summary_until_message_id
            ):
                raise ConversationMemoryRevisionConflict(_REVISION_CONFLICT)
            _apply_rolling_summary(row, summary)

        try:
            await self._guard.flush(self._session)
        except DBAPIError:
            raise ConversationMemoryIntegrityError(_INVALID_DATA) from None
        return _rolling_summary_from_row(row)

    async def _assert_owned_conversation(
        self,
        scope: ConversationMemoryScopeV1,
        *,
        for_update: bool,
    ) -> None:
        statement = select(ChatConversation.id).where(
            ChatConversation.id == scope.conversation_id,
            ChatConversation.user_id == scope.subject_user_id,
            ChatConversation.user_id.is_not(None),
            ChatConversation.status == "ACTIVE",
        )
        if for_update:
            statement = statement.with_for_update()
        owned_id = await self._session.scalar(statement)
        if owned_id is None:
            raise ConversationMemoryAccessDenied(_ACCESS_DENIED)

    async def _validate_summary_links(
        self,
        scope: ConversationMemoryScopeV1,
        summary: RollingSummaryV1,
        *,
        for_update: bool,
    ) -> None:
        source_statement = select(AgentContentSourceRevision.id).where(
            AgentContentSourceRevision.source_kind == summary.source_kind,
            AgentContentSourceRevision.source_record_id
            == encode_source_record_id(summary.source_record_id),
            AgentContentSourceRevision.source_revision == summary.source_revision,
            AgentContentSourceRevision.content_role == summary.content_role,
            AgentContentSourceRevision.content_schema_version
            == summary.content_schema_version,
            AgentContentSourceRevision.normalization_version
            == summary.normalization_version,
            AgentContentSourceRevision.content_sha256 == summary.content_sha256,
            AgentContentSourceRevision.conversation_id == scope.conversation_id,
            AgentContentSourceRevision.subject_user_id == scope.subject_user_id,
        )
        cursor_statement = select(ChatMessage.id).where(
            ChatMessage.id == summary.summary_until_message_id,
            ChatMessage.conversation_id == scope.conversation_id,
            ChatMessage.role.in_(("USER", "ASSISTANT")),
        )
        if for_update:
            source_statement = source_statement.with_for_update()
            cursor_statement = cursor_statement.with_for_update()
        source_id = await self._session.scalar(source_statement)
        cursor_id = await self._session.scalar(cursor_statement)
        if source_id is None or cursor_id is None:
            raise ConversationMemoryIntegrityError(_INVALID_DATA)


def _require_scope(scope: ConversationMemoryScopeV1) -> None:
    if not isinstance(scope, ConversationMemoryScopeV1):
        raise TypeError("conversation memory access requires ConversationMemoryScopeV1")


def _require_expected_revision(expected_revision: int) -> None:
    if (
        type(expected_revision) is not int
        or expected_revision < 0
        or expected_revision >= _MYSQL_SIGNED_BIGINT_MAX
    ):
        raise ConversationMemoryRevisionConflict(_REVISION_CONFLICT)


def _require_optional_cursor(cursor: int | None) -> None:
    if cursor is not None and (
        type(cursor) is not int
        or cursor <= 0
        or cursor > _MYSQL_SIGNED_BIGINT_MAX
    ):
        raise ConversationMemoryIntegrityError(_INVALID_DATA)


def _require_next_revision(expected_revision: int, next_revision: int) -> None:
    if next_revision != expected_revision + 1:
        raise ConversationMemoryRevisionConflict(_REVISION_CONFLICT)


def _serialize_provenance(
    provenance: MemoryProvenanceV1 | None,
) -> dict[str, object] | None:
    if provenance is None:
        return None
    return dict(provenance.model_dump(mode="json"))


def _parse_provenance(value: object) -> MemoryProvenanceV1 | None:
    if value is None:
        return None
    return MemoryProvenanceV1.model_validate(value)


def _working_memory_from_row(row: ConversationWorkingMemory) -> WorkingMemoryV1:
    try:
        return WorkingMemoryV1(
            active_order_no=row.active_order_no,
            active_order_no_provenance=_parse_provenance(row.active_order_no_provenance),
            active_product_code=row.active_product_code,
            active_product_code_provenance=_parse_provenance(
                row.active_product_code_provenance
            ),
            current_issue=row.current_issue,
            current_issue_provenance=_parse_provenance(row.current_issue_provenance),
            last_intent=row.last_intent,
            last_intent_provenance=_parse_provenance(row.last_intent_provenance),
            memory_revision=row.memory_revision,
        )
    except (ValidationError, TypeError, ValueError):
        raise ConversationMemoryIntegrityError(_INVALID_DATA) from None


def _apply_working_memory(
    row: ConversationWorkingMemory,
    memory: WorkingMemoryV1,
) -> None:
    row.active_order_no = memory.active_order_no
    row.active_order_no_provenance = _serialize_provenance(
        memory.active_order_no_provenance
    )
    row.active_product_code = memory.active_product_code
    row.active_product_code_provenance = _serialize_provenance(
        memory.active_product_code_provenance
    )
    row.current_issue = memory.current_issue
    row.current_issue_provenance = _serialize_provenance(
        memory.current_issue_provenance
    )
    row.last_intent = memory.last_intent
    row.last_intent_provenance = _serialize_provenance(
        memory.last_intent_provenance
    )
    row.memory_revision = memory.memory_revision


def _rolling_summary_from_row(row: ConversationRollingSummary) -> RollingSummaryV1:
    try:
        source_record_id = json.loads(row.source_record_id)
        if type(source_record_id) is not str:
            raise ValueError("summary source record identity is invalid")
        return RollingSummaryV1.model_validate(
            {
                "source_kind": row.source_kind,
                "source_record_id": source_record_id,
                "source_revision": row.source_revision,
                "content_role": row.content_role,
                "content_schema_version": row.content_schema_version,
                "normalization_version": row.normalization_version,
                "content_sha256": row.content_sha256,
                "summary_until_message_id": row.summary_until_message_id,
                "summary_revision": row.summary_revision,
                "token_counter_version": row.token_counter_version,
            }
        )
    except (json.JSONDecodeError, ValidationError, TypeError, ValueError):
        raise ConversationMemoryIntegrityError(_INVALID_DATA) from None


def _apply_rolling_summary(
    row: ConversationRollingSummary,
    summary: RollingSummaryV1,
) -> None:
    row.source_kind = summary.source_kind
    row.source_record_id = encode_source_record_id(summary.source_record_id)
    row.source_revision = summary.source_revision
    row.content_role = summary.content_role
    row.content_schema_version = summary.content_schema_version
    row.normalization_version = summary.normalization_version
    row.content_sha256 = summary.content_sha256
    row.summary_until_message_id = summary.summary_until_message_id
    row.summary_revision = summary.summary_revision
    row.token_counter_version = summary.token_counter_version
