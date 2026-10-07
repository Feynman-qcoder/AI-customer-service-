from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.memory.token_counter import (
    TOKEN_COUNTER_V1_ALGORITHM as _TOKEN_COUNTER_V1_ALGORITHM,
)
from app.memory.token_counter import (
    TOKEN_COUNTER_VERSION_V1 as _TOKEN_COUNTER_VERSION_V1,
)
from app.memory.token_counter import (
    count_memory_tokens,
)

TOKEN_COUNTER_VERSION_V1 = _TOKEN_COUNTER_VERSION_V1
TOKEN_COUNTER_V1_ALGORITHM = _TOKEN_COUNTER_V1_ALGORITHM
_MYSQL_SIGNED_BIGINT_MAX = 9_223_372_036_854_775_807

StrictPositiveInt = Annotated[
    int,
    Field(gt=0, le=_MYSQL_SIGNED_BIGINT_MAX, strict=True),
]
StrictNonNegativeInt = Annotated[
    int,
    Field(ge=0, le=_MYSQL_SIGNED_BIGINT_MAX, strict=True),
]
BoundedMemoryIdentifier = Annotated[
    str,
    Field(min_length=1, max_length=256, strict=True),
]
BoundedIssue = Annotated[str, Field(min_length=1, max_length=4000, strict=True)]
NormalizedIntentCode = Annotated[
    str,
    Field(pattern=r"^[A-Z][A-Z0-9_]{0,63}$", strict=True),
]
ProvenanceReference = Annotated[
    str,
    Field(
        min_length=1,
        max_length=256,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$",
        strict=True,
    ),
]
SourceRecordIdentifier = Annotated[
    str,
    Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$",
        strict=True,
    ),
]
LowerHexDigest = Annotated[
    str,
    Field(pattern=r"^[0-9a-f]{64}$", strict=True),
]
MemoryFieldName = Literal[
    "active_order_no",
    "active_product_code",
    "current_issue",
    "last_intent",
]


class _ClosedMemoryModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ConversationMemoryScopeV1(_ClosedMemoryModel):
    conversation_id: StrictPositiveInt
    subject_user_id: StrictPositiveInt


class MemoryProvenanceV1(_ClosedMemoryModel):
    source_kind: Literal["CURRENT_INPUT", "AUTHORIZED_READ_RESULT"]
    source_reference: ProvenanceReference
    source_message_id: StrictPositiveInt


class WorkingMemorySourceMessageV1(_ClosedMemoryModel):
    message_id: StrictPositiveInt
    content: Annotated[str, Field(min_length=1, max_length=4000, strict=True)]
    created_at: datetime


class AuthorizedOrderReadResultV1(_ClosedMemoryModel):
    result_kind: Literal["ORDER"] = "ORDER"
    subject_user_id: StrictPositiveInt
    source_message_id: StrictPositiveInt
    order_id: StrictPositiveInt
    order_no: BoundedMemoryIdentifier
    product_id: StrictPositiveInt
    product_code: BoundedMemoryIdentifier


class AuthorizedProductReadResultV1(_ClosedMemoryModel):
    result_kind: Literal["PRODUCT"] = "PRODUCT"
    subject_user_id: StrictPositiveInt
    source_message_id: StrictPositiveInt
    product_id: StrictPositiveInt
    product_code: BoundedMemoryIdentifier


AuthorizedReadResultV1 = Annotated[
    AuthorizedOrderReadResultV1 | AuthorizedProductReadResultV1,
    Field(discriminator="result_kind"),
]


class WorkingMemoryPromotionCommandV1(_ClosedMemoryModel):
    current_input_message_id: StrictPositiveInt
    authorized_read_results: Annotated[
        tuple[AuthorizedReadResultV1, ...],
        Field(max_length=8),
    ] = ()


class RuntimeMemoryProvenanceV1(_ClosedMemoryModel):
    field_name: MemoryFieldName
    source_kind: Literal["CURRENT_INPUT", "AUTHORIZED_READ_RESULT"]
    source_reference: ProvenanceReference
    source_message_id: StrictPositiveInt
    observed_at: datetime
    memory_revision: StrictPositiveInt


class WorkingMemoryV1(_ClosedMemoryModel):
    active_order_no: BoundedMemoryIdentifier | None
    active_order_no_provenance: MemoryProvenanceV1 | None
    active_product_code: BoundedMemoryIdentifier | None
    active_product_code_provenance: MemoryProvenanceV1 | None
    current_issue: BoundedIssue | None
    current_issue_provenance: MemoryProvenanceV1 | None
    last_intent: NormalizedIntentCode | None
    last_intent_provenance: MemoryProvenanceV1 | None
    memory_revision: StrictPositiveInt

    @model_validator(mode="after")
    def _require_exact_per_field_provenance(self) -> WorkingMemoryV1:
        pairs = (
            (self.active_order_no, self.active_order_no_provenance),
            (self.active_product_code, self.active_product_code_provenance),
            (self.current_issue, self.current_issue_provenance),
            (self.last_intent, self.last_intent_provenance),
        )
        if any((value is None) is not (provenance is None) for value, provenance in pairs):
            raise ValueError("working memory value and provenance must be present together")
        return self


class LoadedWorkingMemoryV1(_ClosedMemoryModel):
    memory: WorkingMemoryV1 | None
    runtime_provenance: Annotated[
        tuple[RuntimeMemoryProvenanceV1, ...],
        Field(max_length=4),
    ] = ()

    @model_validator(mode="after")
    def _require_provenance_only_for_persisted_memory(self) -> LoadedWorkingMemoryV1:
        if self.memory is None and self.runtime_provenance:
            raise ValueError("runtime provenance requires persisted working memory")
        if self.memory is not None:
            expected = {
                field_name
                for field_name in (
                    "active_order_no",
                    "active_product_code",
                    "current_issue",
                    "last_intent",
                )
                if getattr(self.memory, field_name) is not None
            }
            actual = {item.field_name for item in self.runtime_provenance}
            if actual != expected or len(actual) != len(self.runtime_provenance):
                raise ValueError("runtime provenance must exactly cover populated memory fields")
        return self


class RollingSummaryV1(_ClosedMemoryModel):
    source_kind: Literal["AGENT_AUDIT_CONTENT"]
    source_record_id: SourceRecordIdentifier
    source_revision: StrictPositiveInt
    content_role: Literal["CONVERSATION_SUMMARY"]
    content_schema_version: Literal[1]
    normalization_version: Literal["RAW_UTF8_V1"]
    content_sha256: LowerHexDigest
    summary_until_message_id: StrictPositiveInt
    summary_revision: StrictPositiveInt
    token_counter_version: Literal["UTF8_BYTES_CEIL_DIV_3_V1"]


class RecentMessageSourceV1(_ClosedMemoryModel):
    message_id: StrictPositiveInt
    role: Literal["USER", "ASSISTANT"]
    content: Annotated[str, Field(strict=True)]
    created_at: datetime


class RecentMessageV1(RecentMessageSourceV1):
    token_count: StrictNonNegativeInt

    @model_validator(mode="after")
    def _require_exact_token_count(self) -> RecentMessageV1:
        if self.token_count != count_memory_tokens(self.content):
            raise ValueError("recent message token count is inconsistent")
        return self


class MemoryBudgetConfigV1(_ClosedMemoryModel):
    recent_message_limit: StrictPositiveInt = 12
    recent_token_budget: StrictPositiveInt = 2000
    summary_trigger_message_count: StrictPositiveInt = 20
    summary_trigger_token_budget: StrictPositiveInt = 4000
    summary_token_budget: StrictPositiveInt = 1200


class RecentMessagesV1(_ClosedMemoryModel):
    summary_until_message_id: StrictPositiveInt | None
    messages: tuple[RecentMessageV1, ...]
    token_counter_version: Literal["UTF8_BYTES_CEIL_DIV_3_V1"]
    message_limit: StrictPositiveInt
    recent_token_budget: StrictPositiveInt
    actual_token_count: StrictNonNegativeInt
    selected_message_count: StrictNonNegativeInt
    trimmed_message_count: StrictNonNegativeInt

    @model_validator(mode="after")
    def _validate_window_measurements(self) -> RecentMessagesV1:
        message_ids = tuple(message.message_id for message in self.messages)
        if tuple(sorted(message_ids)) != message_ids or len(set(message_ids)) != len(
            message_ids
        ):
            raise ValueError("recent message identities must be unique and ascending")
        if self.summary_until_message_id is not None and any(
            message_id <= self.summary_until_message_id for message_id in message_ids
        ):
            raise ValueError("recent messages must be after the summary cursor")
        if self.selected_message_count != len(self.messages):
            raise ValueError("recent selected count is inconsistent")
        if self.actual_token_count != sum(
            message.token_count for message in self.messages
        ):
            raise ValueError("recent token count is inconsistent")
        if self.selected_message_count > self.message_limit:
            raise ValueError("recent message limit was exceeded")
        if self.actual_token_count > self.recent_token_budget:
            raise ValueError("recent token budget was exceeded")
        return self


class SummaryGenerationRequestV1(_ClosedMemoryModel):
    previous_summary: Annotated[str, Field(min_length=1, strict=True)] | None
    messages: tuple[RecentMessageSourceV1, ...]
    token_counter_version: Literal["UTF8_BYTES_CEIL_DIV_3_V1"]
    candidate_message_count: StrictPositiveInt
    candidate_token_count: StrictNonNegativeInt

    @model_validator(mode="after")
    def _validate_candidate_measurements(self) -> SummaryGenerationRequestV1:
        message_ids = tuple(message.message_id for message in self.messages)
        if not message_ids or tuple(sorted(message_ids)) != message_ids:
            raise ValueError("summary candidates must be non-empty and ascending")
        if len(set(message_ids)) != len(message_ids):
            raise ValueError("summary candidate identities must be unique")
        if self.candidate_message_count != len(self.messages):
            raise ValueError("summary candidate count is inconsistent")
        if self.candidate_token_count != sum(
            count_memory_tokens(message.content) for message in self.messages
        ):
            raise ValueError("summary candidate token count is inconsistent")
        return self


class GeneratedSummaryV1(_ClosedMemoryModel):
    summary_text: Annotated[str, Field(min_length=1, strict=True)]


class GovernedSummaryPayloadV1(_ClosedMemoryModel):
    schema_version: Literal[1] = 1
    summary_text: Annotated[str, Field(min_length=1, strict=True)]


BoundedThreadIdentifier = Annotated[
    str,
    Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
        strict=True,
    ),
]
BoundedRunIdentifier = Annotated[
    str,
    Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
        strict=True,
    ),
]


class SummaryRefreshCommandV1(_ClosedMemoryModel):
    scope: ConversationMemoryScopeV1
    thread_id: BoundedThreadIdentifier
    run_id: BoundedRunIdentifier
    attempt_id: BoundedRunIdentifier
    fence_version: StrictPositiveInt


class SummaryRefreshStatus(StrEnum):
    NOT_TRIGGERED = "NOT_TRIGGERED"
    COMMITTED = "COMMITTED"
    FALLBACK = "FALLBACK"


SummaryFailureReason = Literal[
    "GENERATION_FAILED",
    "SUMMARY_BUDGET_EXCEEDED",
    "PROTECTION_FAILED",
    "SOURCE_READ_FAILED",
    "SOURCE_WRITE_FAILED",
    "CAS_FAILED",
]


class SummaryRefreshResultV1(_ClosedMemoryModel):
    status: SummaryRefreshStatus
    rolling_summary: RollingSummaryV1 | None
    recent: RecentMessagesV1
    config: MemoryBudgetConfigV1
    candidate_message_count: StrictNonNegativeInt
    candidate_token_count: StrictNonNegativeInt
    failure_reason: SummaryFailureReason | None

    @model_validator(mode="after")
    def _validate_refresh_result(self) -> SummaryRefreshResultV1:
        if self.status is SummaryRefreshStatus.FALLBACK:
            if self.failure_reason is None:
                raise ValueError("summary fallback requires a fixed reason code")
        elif self.failure_reason is not None:
            raise ValueError("non-fallback summary result cannot carry a failure")
        if self.status is SummaryRefreshStatus.COMMITTED and self.rolling_summary is None:
            raise ValueError("committed summary result requires a summary")
        if (
            self.recent.message_limit != self.config.recent_message_limit
            or self.recent.recent_token_budget != self.config.recent_token_budget
        ):
            raise ValueError("recent output does not record effective configuration")
        if self.rolling_summary is None:
            if self.recent.summary_until_message_id is not None:
                raise ValueError("recent cursor requires a rolling summary")
        elif (
            self.recent.summary_until_message_id
            != self.rolling_summary.summary_until_message_id
            or self.rolling_summary.token_counter_version
            != self.recent.token_counter_version
        ):
            raise ValueError("summary and recent cursor snapshot is inconsistent")
        return self


class RecentSummaryContextV1(_ClosedMemoryModel):
    rolling_summary: RollingSummaryV1 | None
    summary_text: Annotated[str, Field(min_length=1, strict=True)] | None
    recent: RecentMessagesV1
    config: MemoryBudgetConfigV1

    @model_validator(mode="after")
    def _require_consistent_context_snapshot(self) -> RecentSummaryContextV1:
        if (
            self.recent.message_limit != self.config.recent_message_limit
            or self.recent.recent_token_budget != self.config.recent_token_budget
        ):
            raise ValueError("recent context does not record effective configuration")
        if self.rolling_summary is None:
            if self.summary_text is not None or self.recent.summary_until_message_id is not None:
                raise ValueError("summary text and cursor require summary provenance")
        elif (
            self.summary_text is None
            or self.recent.summary_until_message_id
            != self.rolling_summary.summary_until_message_id
            or self.recent.token_counter_version
            != self.rolling_summary.token_counter_version
        ):
            raise ValueError("summary context snapshot is inconsistent")
        return self
