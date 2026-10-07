from __future__ import annotations

import asyncio
from dataclasses import dataclass

from sqlalchemy.exc import DBAPIError

from app.agent.thread_identity import ThreadIdentity, ThreadIdentityError
from app.memory import (
    TOKEN_COUNTER_VERSION_V1,
    ConversationMemoryError,
    ConversationMemoryRevisionConflict,
    ConversationMemoryScopeV1,
    ConversationMemoryStorePort,
    GeneratedSummaryV1,
    GovernedSummaryPayloadV1,
    MemoryBudgetConfigV1,
    RecentMessageSourceV1,
    RecentMessagesV1,
    RecentMessageV1,
    RecentSummaryContextV1,
    RollingSummaryV1,
    SummaryContentSourcePort,
    SummaryFailureReason,
    SummaryGenerationRequestV1,
    SummaryGeneratorPort,
    SummaryRefreshCommandV1,
    SummaryRefreshResultV1,
    SummaryRefreshStatus,
    count_memory_tokens,
)
from app.runtime.checkpoint_projection import (
    ContentRole,
    ContentSourceReferenceV1,
    ProducingPrincipalKind,
    SourceKind,
)
from app.runtime.content_source import ContentSourceError, ServiceContentSourceCommand
from app.runtime.context import RuntimeContextError
from app.runtime.data_protection import DataProtectionPort, DataProtectionProfile
from app.runtime.durable import DurableContentionError, DurableRuntimeError
from app.runtime.uow import (
    ApplicationTransactionCoordinator,
    ApplicationUnitOfWorkFactory,
    CommitOutcomeUnknown,
    require_no_active_transaction,
)


@dataclass(frozen=True, slots=True)
class _MemorySnapshot:
    rolling_summary: RollingSummaryV1 | None
    previous_summary: str | None
    candidates: tuple[RecentMessageSourceV1, ...]


_FAIL_CLOSED_SUMMARY_ERRORS = (
    ConversationMemoryError,
    ContentSourceError,
    CommitOutcomeUnknown,
    DBAPIError,
    DurableRuntimeError,
    PermissionError,
    RuntimeContextError,
    ThreadIdentityError,
)


class RecentMessagesRollingSummaryService:
    """Owner-scoped L2 selection and monotonic L3 summary orchestration."""

    def __init__(
        self,
        *,
        memory_uow: ApplicationUnitOfWorkFactory[ConversationMemoryStorePort],
        transactions: ApplicationTransactionCoordinator,
        summary_generator: SummaryGeneratorPort,
        data_protection: DataProtectionPort,
        content_sources: SummaryContentSourcePort,
        config: MemoryBudgetConfigV1,
    ) -> None:
        if not isinstance(config, MemoryBudgetConfigV1):
            raise TypeError("recent/summary configuration must be typed")
        self._memory_uow = memory_uow
        self._transactions = transactions
        self._summary_generator = summary_generator
        self._data_protection = data_protection
        self._content_sources = content_sources
        self._config = config

    async def read_recent(
        self,
        scope: ConversationMemoryScopeV1,
    ) -> RecentMessagesV1:
        snapshot = await self._read_snapshot(scope)
        return self._select_recent(snapshot)

    async def read_context(
        self,
        scope: ConversationMemoryScopeV1,
        *,
        before_message_id: int,
    ) -> RecentSummaryContextV1:
        _require_message_upper_bound(before_message_id)
        snapshot = await self._read_snapshot(
            scope,
            before_message_id=before_message_id,
        )
        return RecentSummaryContextV1(
            rolling_summary=snapshot.rolling_summary,
            summary_text=snapshot.previous_summary,
            recent=self._select_recent(snapshot),
            config=self._config,
        )

    async def read_recent_without_summary(
        self,
        scope: ConversationMemoryScopeV1,
        *,
        before_message_id: int,
    ) -> RecentMessagesV1:
        _require_message_upper_bound(before_message_id)
        snapshot = await self._read_snapshot(
            scope,
            before_message_id=before_message_id,
            ignore_summary=True,
        )
        return self._select_recent(snapshot)

    async def refresh(
        self,
        command: SummaryRefreshCommandV1,
    ) -> SummaryRefreshResultV1:
        if not isinstance(command, SummaryRefreshCommandV1):
            raise TypeError("summary refresh requires SummaryRefreshCommandV1")
        ThreadIdentity.from_conversation_id(command.scope.conversation_id).assert_matches(
            command.scope.conversation_id,
            command.thread_id,
        )
        snapshot = await self._read_snapshot(command.scope)
        candidate_count = len(snapshot.candidates)
        candidate_tokens = _candidate_token_count(snapshot.candidates)
        if not self._is_triggered(candidate_count, candidate_tokens):
            return self._result(
                status=SummaryRefreshStatus.NOT_TRIGGERED,
                snapshot=snapshot,
                candidate_count=candidate_count,
                candidate_tokens=candidate_tokens,
            )

        request = SummaryGenerationRequestV1(
            previous_summary=snapshot.previous_summary,
            messages=snapshot.candidates,
            token_counter_version=TOKEN_COUNTER_VERSION_V1,
            candidate_message_count=candidate_count,
            candidate_token_count=candidate_tokens,
        )
        try:
            require_no_active_transaction("rolling-summary generator")
            generated = await self._summary_generator.generate(request)
            if type(generated) is not GeneratedSummaryV1:
                raise TypeError("summary generator returned an unregistered shape")
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if isinstance(error, _FAIL_CLOSED_SUMMARY_ERRORS):
                raise
            return self._fallback_from_snapshot(
                snapshot,
                reason="GENERATION_FAILED",
                candidate_count=candidate_count,
                candidate_tokens=candidate_tokens,
            )

        if count_memory_tokens(generated.summary_text) > self._config.summary_token_budget:
            return self._fallback_from_snapshot(
                snapshot,
                reason="SUMMARY_BUDGET_EXCEEDED",
                candidate_count=candidate_count,
                candidate_tokens=candidate_tokens,
            )

        try:
            protected = self._data_protection.protect(
                GovernedSummaryPayloadV1(summary_text=generated.summary_text),
                profile=DataProtectionProfile.FUTURE_MEMORY_GOVERNANCE,
            )
            governed = GovernedSummaryPayloadV1.model_validate_json(
                protected.canonical_bytes,
                strict=True,
            )
            if governed.summary_text != generated.summary_text:
                raise ValueError("summary protection changed governed text")
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if isinstance(error, _FAIL_CLOSED_SUMMARY_ERRORS):
                raise
            return self._fallback_from_snapshot(
                snapshot,
                reason="PROTECTION_FAILED",
                candidate_count=candidate_count,
                candidate_tokens=candidate_tokens,
            )

        try:
            appended = await self._content_sources.append_service_revision(
                ServiceContentSourceCommand(
                    conversation_id=command.scope.conversation_id,
                    thread_id=command.thread_id,
                    subject_user_id=command.scope.subject_user_id,
                    run_id=command.run_id,
                    attempt_id=command.attempt_id,
                    fence_version=command.fence_version,
                    content_role=ContentRole.CONVERSATION_SUMMARY,
                    raw_content=governed.summary_text,
                )
            )
            candidate_summary = _summary_from_reference(
                appended.record.reference,
                cursor=snapshot.candidates[-1].message_id,
                revision=(
                    snapshot.rolling_summary.summary_revision + 1
                    if snapshot.rolling_summary is not None
                    else 1
                ),
            )
        except asyncio.CancelledError:
            raise
        except DurableContentionError:
            return self._fallback_from_snapshot(
                snapshot,
                reason="SOURCE_WRITE_FAILED",
                candidate_count=candidate_count,
                candidate_tokens=candidate_tokens,
            )

        expected_revision = (
            snapshot.rolling_summary.summary_revision
            if snapshot.rolling_summary is not None
            else 0
        )
        try:
            await self._transactions.run(
                "application.transaction",
                self._memory_uow,
                lambda store: store.write_rolling_summary(
                    command.scope,
                    expected_revision=expected_revision,
                    summary=candidate_summary,
                ),
            )
        except asyncio.CancelledError:
            raise
        except ConversationMemoryRevisionConflict:
            return await self._fallback_after_cas_conflict(
                command.scope,
                original_snapshot=snapshot,
                reason="CAS_FAILED",
                candidate_count=candidate_count,
                candidate_tokens=candidate_tokens,
            )
        except DurableContentionError:
            return self._fallback_from_snapshot(
                snapshot,
                reason="CAS_FAILED",
                candidate_count=candidate_count,
                candidate_tokens=candidate_tokens,
            )

        fresh = await self._read_snapshot(command.scope)
        return SummaryRefreshResultV1(
            status=SummaryRefreshStatus.COMMITTED,
            rolling_summary=fresh.rolling_summary,
            recent=self._select_recent(fresh),
            config=self._config,
            candidate_message_count=candidate_count,
            candidate_token_count=candidate_tokens,
            failure_reason=None,
        )

    async def _read_snapshot(
        self,
        scope: ConversationMemoryScopeV1,
        *,
        before_message_id: int | None = None,
        ignore_summary: bool = False,
    ) -> _MemorySnapshot:
        async def read(store: ConversationMemoryStorePort) -> _MemorySnapshot:
            summary = None if ignore_summary else await store.load_rolling_summary(scope)
            previous_summary = (
                await store.load_rolling_summary_content(scope, summary)
                if summary is not None
                else None
            )
            after_message_id = (
                summary.summary_until_message_id if summary is not None else None
            )
            if before_message_id is None:
                messages = await store.load_recent_messages(
                    scope,
                    after_message_id=after_message_id,
                )
            else:
                messages = await store.load_recent_messages(
                    scope,
                    after_message_id=after_message_id,
                    before_message_id=before_message_id,
                )
            return _MemorySnapshot(
                rolling_summary=summary,
                previous_summary=previous_summary,
                candidates=messages,
            )

        return await self._transactions.run(
            "application.transaction",
            self._memory_uow,
            read,
        )

    def _select_recent(self, snapshot: _MemorySnapshot) -> RecentMessagesV1:
        selected_descending: list[RecentMessageV1] = []
        token_count = 0
        for candidate in reversed(snapshot.candidates):
            if len(selected_descending) >= self._config.recent_message_limit:
                break
            candidate_tokens = count_memory_tokens(candidate.content)
            if token_count + candidate_tokens > self._config.recent_token_budget:
                break
            selected_descending.append(
                RecentMessageV1(
                    **candidate.model_dump(),
                    token_count=candidate_tokens,
                )
            )
            token_count += candidate_tokens
        selected = tuple(reversed(selected_descending))
        return RecentMessagesV1(
            summary_until_message_id=(
                snapshot.rolling_summary.summary_until_message_id
                if snapshot.rolling_summary is not None
                else None
            ),
            messages=selected,
            token_counter_version=TOKEN_COUNTER_VERSION_V1,
            message_limit=self._config.recent_message_limit,
            recent_token_budget=self._config.recent_token_budget,
            actual_token_count=token_count,
            selected_message_count=len(selected),
            trimmed_message_count=len(snapshot.candidates) - len(selected),
        )

    def _is_triggered(self, candidate_count: int, candidate_tokens: int) -> bool:
        return (
            candidate_count >= self._config.summary_trigger_message_count
            or candidate_tokens >= self._config.summary_trigger_token_budget
        )

    async def _fallback_after_cas_conflict(
        self,
        scope: ConversationMemoryScopeV1,
        *,
        original_snapshot: _MemorySnapshot,
        reason: SummaryFailureReason,
        candidate_count: int,
        candidate_tokens: int,
    ) -> SummaryRefreshResultV1:
        try:
            snapshot = await self._read_snapshot(scope)
        except asyncio.CancelledError:
            raise
        except DurableContentionError:
            snapshot = original_snapshot
        return self._fallback_from_snapshot(
            snapshot,
            reason=reason,
            candidate_count=candidate_count,
            candidate_tokens=candidate_tokens,
        )

    def _fallback_from_snapshot(
        self,
        snapshot: _MemorySnapshot,
        *,
        reason: SummaryFailureReason,
        candidate_count: int,
        candidate_tokens: int,
    ) -> SummaryRefreshResultV1:
        return self._result(
            status=SummaryRefreshStatus.FALLBACK,
            snapshot=snapshot,
            candidate_count=candidate_count,
            candidate_tokens=candidate_tokens,
            failure_reason=reason,
        )

    def _result(
        self,
        *,
        status: SummaryRefreshStatus,
        snapshot: _MemorySnapshot,
        candidate_count: int,
        candidate_tokens: int,
        failure_reason: SummaryFailureReason | None = None,
    ) -> SummaryRefreshResultV1:
        return SummaryRefreshResultV1(
            status=status,
            rolling_summary=snapshot.rolling_summary,
            recent=self._select_recent(snapshot),
            config=self._config,
            candidate_message_count=candidate_count,
            candidate_token_count=candidate_tokens,
            failure_reason=failure_reason,
        )


def _candidate_token_count(messages: tuple[RecentMessageSourceV1, ...]) -> int:
    return sum(count_memory_tokens(message.content) for message in messages)


def _require_message_upper_bound(message_id: int) -> None:
    if type(message_id) is not int or not 0 < message_id <= 9_223_372_036_854_775_807:
        raise ValueError("recent message upper bound is invalid")


def _summary_from_reference(
    reference: ContentSourceReferenceV1,
    *,
    cursor: int,
    revision: int,
) -> RollingSummaryV1:
    if (
        reference.source_kind is not SourceKind.AGENT_AUDIT_CONTENT
        or type(reference.source_record_id) is not str
        or reference.content_role is not ContentRole.CONVERSATION_SUMMARY
        or reference.producing_principal_kind is not ProducingPrincipalKind.SERVICE
    ):
        raise ValueError("summary source binding is invalid")
    return RollingSummaryV1(
        source_kind="AGENT_AUDIT_CONTENT",
        source_record_id=reference.source_record_id,
        source_revision=reference.source_revision,
        content_role="CONVERSATION_SUMMARY",
        content_schema_version=reference.content_schema_version,
        normalization_version=reference.normalization_version.value,
        content_sha256=reference.content_sha256,
        summary_until_message_id=cursor,
        summary_revision=revision,
        token_counter_version=TOKEN_COUNTER_VERSION_V1,
    )


__all__ = ["RecentMessagesRollingSummaryService"]
