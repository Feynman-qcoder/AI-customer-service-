from __future__ import annotations

from typing import Protocol

from app.memory.models import (
    ConversationMemoryScopeV1,
    GeneratedSummaryV1,
    LoadedWorkingMemoryV1,
    RecentMessageSourceV1,
    RecentMessagesV1,
    RecentSummaryContextV1,
    RollingSummaryV1,
    SummaryGenerationRequestV1,
    SummaryRefreshCommandV1,
    SummaryRefreshResultV1,
    WorkingMemoryPromotionCommandV1,
    WorkingMemorySourceMessageV1,
    WorkingMemoryV1,
)
from app.runtime.content_source import (
    ContentSourceAppendResult,
    ServiceContentSourceCommand,
)


class ConversationMemoryError(RuntimeError):
    """Sanitized base error for the bounded conversation-memory boundary."""


class ConversationMemoryAccessDenied(ConversationMemoryError):
    pass


class ConversationMemoryRevisionConflict(ConversationMemoryError):
    pass


class ConversationMemoryIntegrityError(ConversationMemoryError):
    pass


class WorkingMemoryPromotionRejected(ConversationMemoryError):
    pass


class WorkingMemoryUpdateConflict(ConversationMemoryError):
    pass


class ConversationMemoryStorePort(Protocol):
    async def load_working_memory(
        self,
        scope: ConversationMemoryScopeV1,
    ) -> WorkingMemoryV1 | None: ...

    async def write_working_memory(
        self,
        scope: ConversationMemoryScopeV1,
        *,
        expected_revision: int,
        memory: WorkingMemoryV1,
    ) -> WorkingMemoryV1: ...

    async def load_user_input_messages(
        self,
        scope: ConversationMemoryScopeV1,
        *,
        message_ids: tuple[int, ...],
    ) -> tuple[WorkingMemorySourceMessageV1, ...]: ...

    async def load_rolling_summary(
        self,
        scope: ConversationMemoryScopeV1,
    ) -> RollingSummaryV1 | None: ...

    async def load_recent_messages(
        self,
        scope: ConversationMemoryScopeV1,
        *,
        after_message_id: int | None,
        before_message_id: int | None = None,
    ) -> tuple[RecentMessageSourceV1, ...]: ...

    async def load_rolling_summary_content(
        self,
        scope: ConversationMemoryScopeV1,
        summary: RollingSummaryV1,
    ) -> str: ...

    async def write_rolling_summary(
        self,
        scope: ConversationMemoryScopeV1,
        *,
        expected_revision: int,
        summary: RollingSummaryV1,
    ) -> RollingSummaryV1: ...


class WorkingMemoryApplicationPort(Protocol):
    async def load(
        self,
        scope: ConversationMemoryScopeV1,
    ) -> LoadedWorkingMemoryV1: ...

    async def update(
        self,
        scope: ConversationMemoryScopeV1,
        command: WorkingMemoryPromotionCommandV1,
    ) -> LoadedWorkingMemoryV1: ...


class SummaryGeneratorPort(Protocol):
    async def generate(
        self,
        request: SummaryGenerationRequestV1,
    ) -> GeneratedSummaryV1: ...


class SummaryContentSourcePort(Protocol):
    async def append_service_revision(
        self,
        command: ServiceContentSourceCommand,
    ) -> ContentSourceAppendResult: ...

class RecentSummaryApplicationPort(Protocol):
    async def read_recent(
        self,
        scope: ConversationMemoryScopeV1,
    ) -> RecentMessagesV1: ...

    async def read_context(
        self,
        scope: ConversationMemoryScopeV1,
        *,
        before_message_id: int,
    ) -> RecentSummaryContextV1: ...

    async def read_recent_without_summary(
        self,
        scope: ConversationMemoryScopeV1,
        *,
        before_message_id: int,
    ) -> RecentMessagesV1: ...

    async def refresh(
        self,
        command: SummaryRefreshCommandV1,
    ) -> SummaryRefreshResultV1: ...
