import json
import math
from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, TypedDict, cast
from uuid import UUID

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from app.agent.thread_identity import ThreadIdentity, require_positive_identity

CURRENT_STATE_SCHEMA_VERSION = 1
MAX_IDENTITY = (1 << 63) - 1

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

PositiveInt = Annotated[StrictInt, Field(gt=0, le=MAX_IDENTITY)]
NonNegativeInt = Annotated[StrictInt, Field(ge=0, le=MAX_IDENTITY)]
ShortText = Annotated[StrictStr, StringConstraints(max_length=256)]
BoundedText = Annotated[StrictStr, StringConstraints(max_length=4_000)]
LongText = Annotated[StrictStr, StringConstraints(max_length=16_000)]
NonEmptyText = Annotated[StrictStr, StringConstraints(min_length=1, max_length=256)]
NormalizedCode = Annotated[
    StrictStr,
    StringConstraints(min_length=1, max_length=128, pattern=r"^[A-Z][A-Z0-9_]*$"),
]
SafeIdentifier = Annotated[
    StrictStr,
    StringConstraints(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$"),
]

IntentType = Literal[
    "ORDER_QUERY",
    "SHIPPING_QUERY",
    "PRODUCT_QUERY",
    "KNOWLEDGE_QUERY",
    "CANCEL_ORDER",
    "REFUND_REQUEST",
    "CREATE_TICKET",
    "CLARIFICATION",
]
RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "FORBIDDEN"]


class StateContractError(ValueError):
    """A stable checkpoint payload violates the frozen state contract."""


class UnsupportedStateVersionError(StateContractError):
    """The payload declares a schema version this process must not interpret."""


class ActiveRunConflictError(StateContractError):
    status_code = 409


class RunStatus(StrEnum):
    RUNNING = "RUNNING"
    WAITING_CUSTOMER_CONFIRMATION = "WAITING_CUSTOMER_CONFIRMATION"
    WAITING_ADMIN_APPROVAL = "WAITING_ADMIN_APPROVAL"
    RESUME_PENDING = "RESUME_PENDING"
    EXECUTING = "EXECUTING"
    COMPLETED = "COMPLETED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ToolResultStatus(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"


class RetrievalChannel(StrEnum):
    KEYWORD = "keyword"
    DENSE = "dense"
    STRUCTURED_RULE = "structured_rule"
    FUSED = "fused"
    RERANKED = "reranked"


class EffectPhase(StrEnum):
    READ_ONLY = "READ_ONLY"
    ACTION_PREPARE = "ACTION_PREPARE"
    BUSINESS_EXECUTE = "BUSINESS_EXECUTE"


class CustomerConfirmationStatus(StrEnum):
    NOT_REQUIRED = "NOT_REQUIRED"
    PENDING = "PENDING"
    CONFIRMED = "CONFIRMED"
    REJECTED = "REJECTED"


class ApprovalDecision(StrEnum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    STALE = "STALE"


class ResumeKind(StrEnum):
    CUSTOMER_CONFIRMATION = "CUSTOMER_CONFIRMATION"
    ADMIN_DECISION = "ADMIN_DECISION"


class StableModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _validate_rfc3339(value: str) -> str:
    candidate = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ValueError("timestamp must be RFC3339") from exc
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return value


def _validate_uuid_text(value: str) -> str:
    try:
        parsed = UUID(value)
    except ValueError as exc:
        raise ValueError("logical_action_id must be UUID-form") from exc
    if str(parsed) != value.lower():
        raise ValueError("logical_action_id must be canonical UUID-form")
    return value.lower()


class OrderReferenceSnapshot(StableModel):
    order_no: ShortText | None = None
    ordinal_index: NonNegativeInt | None = None
    product_keyword: ShortText | None = None
    latest: StrictBool = False
    list_all: StrictBool = False


class PlanSnapshot(StableModel):
    intent: IntentType
    goal: BoundedText
    order_reference: OrderReferenceSnapshot | None = None
    product_reference: ShortText | None = None
    required_tools: Annotated[list[NonEmptyText], Field(max_length=32)]
    action_type: NormalizedCode | None = None
    risk_level: RiskLevel
    confirmation_required: StrictBool = Field(
        validation_alias=AliasChoices("confirmation_required", "requires_confirmation")
    )
    missing_information: Annotated[list[ShortText], Field(max_length=32)]
    decision_reason: BoundedText


class ToolResultSnapshot(StableModel):
    tool_name: NonEmptyText
    status: ToolResultStatus
    result_ref: SafeIdentifier | None = None
    safe_metadata: Annotated[dict[ShortText, JsonScalar], Field(max_length=32)]
    observed_at: NonEmptyText
    error_type: NormalizedCode | None = None

    @field_validator("observed_at")
    @classmethod
    def validate_observed_at(cls, value: str) -> str:
        return _validate_rfc3339(value)

    @field_validator("safe_metadata")
    @classmethod
    def validate_safe_metadata(cls, value: dict[str, JsonScalar]) -> dict[str, JsonScalar]:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(encoded) > 16 * 1024:
            raise ValueError("safe_metadata exceeds 16 KiB")
        return value


class RetrievalEvidence(StableModel):
    document_id: NonNegativeInt
    chunk_ref: SafeIdentifier
    file_name: ShortText
    snippet: Annotated[StrictStr, StringConstraints(max_length=800)]
    score: StrictFloat
    channel: RetrievalChannel

    @field_validator("score")
    @classmethod
    def validate_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("score must be finite")
        return value


class ResponseMeta(StableModel):
    sources: Annotated[list[RetrievalEvidence], Field(max_length=50)]
    retrieval_score: StrictFloat
    confidence_level: NormalizedCode
    need_human: StrictBool
    ticket_id: PositiveInt | None = None

    @field_validator("retrieval_score")
    @classmethod
    def validate_retrieval_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("retrieval_score must be finite")
        return value


class ActionDraftSnapshot(StableModel):
    logical_action_id: NonEmptyText
    action_type: NormalizedCode
    target_order_id: PositiveInt
    target_order_no: NonEmptyText
    subject_user_id: PositiveInt
    reason_code: NormalizedCode
    policy_version: NonEmptyText
    draft_revision: PositiveInt
    expires_at: NonEmptyText
    nonce_digest: SafeIdentifier

    @field_validator("logical_action_id")
    @classmethod
    def validate_logical_action_id(cls, value: str) -> str:
        return _validate_uuid_text(value)

    @field_validator("expires_at")
    @classmethod
    def validate_expires_at(cls, value: str) -> str:
        return _validate_rfc3339(value)


class SideEffectAuthorizationSnapshot(StableModel):
    authorization_id: SafeIdentifier
    run_id: SafeIdentifier
    logical_action_id: NonEmptyText
    action_type: NormalizedCode
    subject_user_id: PositiveInt
    target_order_id: PositiveInt
    target_order_no: NonEmptyText
    effect_phase: EffectPhase
    policy_version: NonEmptyText
    draft_revision: PositiveInt
    issued_at: NonEmptyText
    expires_at: NonEmptyText

    @field_validator("logical_action_id")
    @classmethod
    def validate_logical_action_id(cls, value: str) -> str:
        return _validate_uuid_text(value)

    @field_validator("issued_at", "expires_at")
    @classmethod
    def validate_timestamps(cls, value: str) -> str:
        return _validate_rfc3339(value)


class EffectIdentity(StableModel):
    run_id: SafeIdentifier
    node_name: SafeIdentifier
    purpose: NormalizedCode
    sequence: NonNegativeInt


class MemoryProvenanceRecord(StableModel):
    field_name: Literal["active_order_no", "active_product_code", "current_issue", "last_intent"]
    source_type: NormalizedCode
    source_ref: SafeIdentifier
    observed_at: NonEmptyText
    memory_revision: NonNegativeInt

    @field_validator("observed_at")
    @classmethod
    def validate_observed_at(cls, value: str) -> str:
        return _validate_rfc3339(value)


class ConversationIdentityState(StableModel):
    conversation_id: PositiveInt
    thread_id: NonEmptyText
    subject_user_id: PositiveInt
    subject_role_snapshot: NormalizedCode

    @model_validator(mode="after")
    def validate_thread_identity(self) -> "ConversationIdentityState":
        ThreadIdentity.from_conversation_id(self.conversation_id).assert_matches(
            self.conversation_id,
            self.thread_id,
        )
        return self


class ConversationMemoryState(StableModel):
    active_order_no: ShortText | None = None
    active_product_code: ShortText | None = None
    current_issue: BoundedText | None = None
    last_intent: NormalizedCode | None = None
    provenance: Annotated[list[MemoryProvenanceRecord], Field(max_length=32)] = Field(default_factory=list)
    memory_revision: NonNegativeInt = 0
    conversation_summary: LongText = ""
    summary_until_message_id: NonNegativeInt | None = None
    summary_revision: NonNegativeInt = 0

    @model_validator(mode="after")
    def validate_summary_cursor(self) -> "ConversationMemoryState":
        if self.conversation_summary and (
            self.summary_until_message_id is None or self.summary_revision <= 0
        ):
            raise ValueError("a non-empty summary requires a cursor and positive revision")
        if not self.conversation_summary and (
            self.summary_until_message_id is not None or self.summary_revision != 0
        ):
            raise ValueError("an empty summary cannot retain a cursor or revision")
        if any(record.memory_revision > self.memory_revision for record in self.provenance):
            raise ValueError("provenance cannot reference a future memory revision")
        return self


class ActiveRunState(StableModel):
    run_id: SafeIdentifier
    attempt_id: SafeIdentifier
    run_status: RunStatus = RunStatus.RUNNING
    question: BoundedText
    effective_question: BoundedText
    current_user_message_id: PositiveInt | None = None
    blocked: StrictBool = False
    intent: IntentType = "CLARIFICATION"
    risk_level: RiskLevel = "LOW"
    plan: PlanSnapshot | None = None
    selected_tools: Annotated[list[NonEmptyText], Field(max_length=32)] = Field(default_factory=list)
    tool_results: Annotated[list[ToolResultSnapshot], Field(max_length=64)] = Field(default_factory=list)
    retrieval_evidence: Annotated[list[RetrievalEvidence], Field(max_length=50)] = Field(default_factory=list)
    retrieval_score: StrictFloat | None = None
    action_draft: ActionDraftSnapshot | None = None
    side_effect_authorization: SideEffectAuthorizationSnapshot | None = None
    pending_action_id: PositiveInt | None = None
    customer_confirmation_status: CustomerConfirmationStatus = CustomerConfirmationStatus.NOT_REQUIRED
    approval_decision: ApprovalDecision | None = None
    decision_reason: BoundedText | None = None
    draft_answer: LongText | None = None
    final_answer: LongText | None = None
    response_meta: ResponseMeta | None = None
    error_type: NormalizedCode | None = None
    error_summary: BoundedText | None = None

    @field_validator("retrieval_score")
    @classmethod
    def validate_retrieval_score(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("retrieval_score must be finite")
        return value

    @model_validator(mode="after")
    def validate_run_invariants(self) -> "ActiveRunState":
        if self.run_status is RunStatus.COMPLETED and self.final_answer is None:
            raise ValueError("COMPLETED requires final_answer")
        if self.run_status is RunStatus.WAITING_CUSTOMER_CONFIRMATION:
            if self.action_draft is None or self.customer_confirmation_status is not CustomerConfirmationStatus.PENDING:
                raise ValueError("customer confirmation wait requires a pending action draft")
        if self.pending_action_id is not None:
            if (
                self.action_draft is None
                or self.customer_confirmation_status is not CustomerConfirmationStatus.CONFIRMED
            ):
                raise ValueError("pending action requires a confirmed action draft")
        if self.run_status is RunStatus.WAITING_ADMIN_APPROVAL and self.pending_action_id is None:
            raise ValueError("admin approval wait requires a pending action reference")
        if self.run_status in {RunStatus.RESUME_PENDING, RunStatus.EXECUTING} and self.action_draft is None:
            raise ValueError("resume or execution state requires an action draft")
        if self.approval_decision is not None and (self.pending_action_id is None or self.action_draft is None):
            raise ValueError("approval evidence requires a pending action draft")
        if self.side_effect_authorization is not None:
            authorization = self.side_effect_authorization
            if authorization.run_id != self.run_id:
                raise ValueError("authorization run mismatch")
            if self.action_draft is None:
                raise ValueError("authorization requires an action draft")
            draft = self.action_draft
            if (
                authorization.logical_action_id != draft.logical_action_id
                or authorization.action_type != draft.action_type
                or authorization.target_order_id != draft.target_order_id
                or authorization.target_order_no != draft.target_order_no
                or authorization.subject_user_id != draft.subject_user_id
                or authorization.policy_version != draft.policy_version
                or authorization.draft_revision != draft.draft_revision
            ):
                raise ValueError("authorization does not match the action draft")
            if (
                authorization.effect_phase in {EffectPhase.ACTION_PREPARE, EffectPhase.BUSINESS_EXECUTE}
                and self.customer_confirmation_status is not CustomerConfirmationStatus.CONFIRMED
            ):
                raise ValueError("side-effect authorization requires customer confirmation")
            if authorization.effect_phase is EffectPhase.BUSINESS_EXECUTE and (
                self.approval_decision is not ApprovalDecision.APPROVED or self.pending_action_id is None
            ):
                raise ValueError("business execution authorization requires approved pending action")
        return self


class ConversationCheckpointState(StableModel):
    schema_version: Literal[1] = 1
    conversation_identity: ConversationIdentityState
    memory: ConversationMemoryState = Field(default_factory=lambda: ConversationMemoryState())
    active_run: ActiveRunState | None = None

    @model_validator(mode="after")
    def validate_cross_scope_invariants(self) -> "ConversationCheckpointState":
        if self.active_run is None or self.active_run.action_draft is None:
            return self
        draft = self.active_run.action_draft
        if draft.subject_user_id != self.conversation_identity.subject_user_id:
            raise ValueError("action draft subject does not match conversation identity")
        return self

    def to_agent_state(self) -> "AgentState":
        payload = cast(dict[str, JsonValue], self.model_dump(mode="json"))
        _assert_json_value(payload)
        return cast(AgentState, payload)


class AgentState(TypedDict):
    schema_version: int
    conversation_identity: dict[str, JsonValue]
    memory: dict[str, JsonValue]
    active_run: dict[str, JsonValue] | None


TERMINAL_RUN_STATUSES = frozenset(
    {
        RunStatus.COMPLETED,
        RunStatus.REJECTED,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
    }
)
RESUMABLE_RUN_STATUSES = frozenset(
    {
        RunStatus.WAITING_CUSTOMER_CONFIRMATION,
        RunStatus.WAITING_ADMIN_APPROVAL,
        RunStatus.RESUME_PENDING,
        RunStatus.EXECUTING,
    }
)


class ResumeInput(StableModel):
    kind: ResumeKind
    conversation_id: PositiveInt
    thread_id: NonEmptyText
    run_id: SafeIdentifier
    attempt_id: SafeIdentifier
    actor_user_id: PositiveInt
    actor_role: NormalizedCode
    subject_user_id: PositiveInt
    logical_action_id: NonEmptyText
    action_type: NormalizedCode
    target_order_id: PositiveInt
    target_order_no: NonEmptyText
    draft_revision: PositiveInt
    pending_action_id: PositiveInt | None = None

    @field_validator("logical_action_id")
    @classmethod
    def validate_logical_action_id(cls, value: str) -> str:
        return _validate_uuid_text(value)

    @model_validator(mode="after")
    def validate_identity(self) -> "ResumeInput":
        ThreadIdentity.from_conversation_id(self.conversation_id).assert_matches(
            self.conversation_id,
            self.thread_id,
        )
        return self


class LegacyAgentStateV0(StableModel):
    schema_version: Literal[0] = 0
    conversation_id: PositiveInt
    authenticated_user_id: PositiveInt
    user_role: NormalizedCode
    thread_id: NonEmptyText | None = None
    run_id: SafeIdentifier | None = None
    attempt_id: SafeIdentifier | None = None
    question: BoundedText = ""
    effective_question: BoundedText = ""
    blocked: StrictBool = False
    intent: IntentType = "CLARIFICATION"
    risk_level: RiskLevel = "LOW"
    plan: PlanSnapshot | None = None
    selected_tools: Annotated[list[NonEmptyText], Field(max_length=32)] = Field(default_factory=list)
    tool_results: Annotated[list[ToolResultSnapshot], Field(max_length=64)] = Field(default_factory=list)
    retrieval_evidence: Annotated[list[RetrievalEvidence], Field(max_length=50)] = Field(default_factory=list)
    retrieval_score: StrictFloat | None = None
    pending_action_id: PositiveInt | None = None
    decision_reason: BoundedText | None = None
    draft_answer: LongText | None = None
    final_answer: LongText | None = None
    error_type: NormalizedCode | None = None
    error_summary: BoundedText | None = None


def _assert_json_value(value: JsonValue | object, *, path: str = "$") -> None:
    if value is None or type(value) in {str, int, bool}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise StateContractError(f"{path} contains a non-finite float")
        return
    if type(value) is list:
        for index, item in enumerate(cast(list[object], value)):
            _assert_json_value(item, path=f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, item in cast(dict[object, object], value).items():
            if type(key) is not str:
                raise StateContractError(f"{path} contains a non-string key")
            _assert_json_value(item, path=f"{path}.{key}")
        return
    raise StateContractError(f"{path} contains a non-JSON runtime value")


def new_conversation_state(
    *,
    conversation_id: object,
    subject_user_id: object,
    subject_role_snapshot: str,
) -> ConversationCheckpointState:
    identity = ThreadIdentity.from_conversation_id(conversation_id)
    return ConversationCheckpointState(
        conversation_identity=ConversationIdentityState(
            conversation_id=identity.conversation_id,
            thread_id=identity.thread_id,
            subject_user_id=require_positive_identity(subject_user_id, field_name="subject_user_id"),
            subject_role_snapshot=subject_role_snapshot,
        )
    )


def start_new_run(
    checkpoint: ConversationCheckpointState,
    *,
    run_id: str,
    attempt_id: str,
    question: str,
) -> ConversationCheckpointState:
    active_run = checkpoint.active_run
    if active_run is not None and active_run.run_status not in TERMINAL_RUN_STATUSES:
        raise ActiveRunConflictError("the conversation already has an active logical run")
    if active_run is not None and (
        run_id == active_run.run_id or attempt_id == active_run.attempt_id
    ):
        raise ActiveRunConflictError(
            "new run and attempt identity must differ from the terminal run"
        )
    replacement = ActiveRunState(
        run_id=run_id,
        attempt_id=attempt_id,
        question=question,
        effective_question=question,
    )
    result = checkpoint.model_copy(update={"active_run": replacement})
    return validate_state_json_round_trip(result)


def resume_active_run(
    checkpoint: ConversationCheckpointState,
    resume: ResumeInput,
) -> ConversationCheckpointState:
    active_run = checkpoint.active_run
    identity = checkpoint.conversation_identity
    if active_run is None or active_run.run_status not in RESUMABLE_RUN_STATUSES:
        raise ActiveRunConflictError("the conversation has no resumable active run")
    draft = active_run.action_draft
    if draft is None:
        raise ActiveRunConflictError("the active run has no resumable action draft")
    if resume.attempt_id == active_run.attempt_id:
        raise ActiveRunConflictError(
            "resume attempt identity must differ from the current attempt"
        )
    if (
        resume.conversation_id != identity.conversation_id
        or resume.thread_id != identity.thread_id
        or resume.run_id != active_run.run_id
        or resume.subject_user_id != identity.subject_user_id
        or resume.logical_action_id != draft.logical_action_id
        or resume.action_type != draft.action_type
        or resume.target_order_id != draft.target_order_id
        or resume.target_order_no != draft.target_order_no
        or resume.draft_revision != draft.draft_revision
    ):
        raise ActiveRunConflictError("resume input does not match the stored pause point")
    if resume.kind is ResumeKind.CUSTOMER_CONFIRMATION:
        if (
            active_run.run_status is not RunStatus.WAITING_CUSTOMER_CONFIRMATION
            or resume.actor_role != "CUSTOMER"
            or resume.actor_user_id != identity.subject_user_id
            or resume.pending_action_id is not None
        ):
            raise ActiveRunConflictError("customer resume is not authorized for this pause point")
    elif (
        active_run.run_status is not RunStatus.WAITING_ADMIN_APPROVAL
        or resume.actor_role != "ADMIN"
        or resume.pending_action_id != active_run.pending_action_id
    ):
        raise ActiveRunConflictError("admin resume is not authorized for this pause point")
    updates: dict[str, object] = {
        "attempt_id": resume.attempt_id,
        "run_status": RunStatus.RESUME_PENDING,
    }
    if resume.kind is ResumeKind.CUSTOMER_CONFIRMATION:
        updates["customer_confirmation_status"] = CustomerConfirmationStatus.CONFIRMED
    resumed = active_run.model_copy(update=updates)
    result = checkpoint.model_copy(update={"active_run": resumed})
    return validate_state_json_round_trip(result)


def validate_state_json_round_trip(
    checkpoint: ConversationCheckpointState,
) -> ConversationCheckpointState:
    payload = checkpoint.model_dump(mode="json")
    _assert_json_value(payload)
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    restored = json.loads(encoded)
    return ConversationCheckpointState.model_validate(restored)


def migrate_legacy_state(payload: Mapping[str, object]) -> ConversationCheckpointState:
    try:
        legacy = LegacyAgentStateV0.model_validate(dict(payload))
    except ValidationError as exc:
        raise StateContractError("unsupported or unsafe legacy state") from exc
    identity = ThreadIdentity.from_conversation_id(legacy.conversation_id)
    if legacy.thread_id is not None:
        identity.assert_matches(legacy.conversation_id, legacy.thread_id)
    checkpoint = new_conversation_state(
        conversation_id=legacy.conversation_id,
        subject_user_id=legacy.authenticated_user_id,
        subject_role_snapshot=legacy.user_role,
    )
    if legacy.run_id is None:
        declared_defaults = LegacyAgentStateV0(
            conversation_id=legacy.conversation_id,
            authenticated_user_id=legacy.authenticated_user_id,
            user_role=legacy.user_role,
            thread_id=legacy.thread_id,
        )
        current_payload = legacy.model_dump(mode="python")
        default_payload = declared_defaults.model_dump(mode="python")
        conversation_fields = {
            "schema_version",
            "conversation_id",
            "authenticated_user_id",
            "user_role",
            "thread_id",
        }
        if any(
            current_payload[field_name] != default_payload[field_name]
            for field_name in current_payload.keys() - conversation_fields
        ):
            raise StateContractError(
                "legacy state without run_id contains non-default run-scoped data"
            )
        return checkpoint
    active_run = ActiveRunState(
        run_id=legacy.run_id,
        attempt_id=legacy.attempt_id or f"migrated-{legacy.run_id}",
        question=legacy.question,
        effective_question=legacy.effective_question or legacy.question,
        blocked=legacy.blocked,
        intent=legacy.intent,
        risk_level=legacy.risk_level,
        plan=legacy.plan,
        selected_tools=legacy.selected_tools,
        tool_results=legacy.tool_results,
        retrieval_evidence=legacy.retrieval_evidence,
        retrieval_score=legacy.retrieval_score,
        pending_action_id=legacy.pending_action_id,
        decision_reason=legacy.decision_reason,
        draft_answer=legacy.draft_answer,
        final_answer=legacy.final_answer,
        error_type=legacy.error_type,
        error_summary=legacy.error_summary,
    )
    return validate_state_json_round_trip(checkpoint.model_copy(update={"active_run": active_run}))


def load_checkpoint_state(payload: Mapping[str, object]) -> ConversationCheckpointState:
    copied = dict(payload)
    _assert_json_value(copied)
    raw_version = copied.get("schema_version", 0)
    if type(raw_version) is not int:
        raise UnsupportedStateVersionError("schema_version must be an integer")
    if raw_version == 0:
        return migrate_legacy_state(copied)
    if raw_version != CURRENT_STATE_SCHEMA_VERSION:
        raise UnsupportedStateVersionError("unsupported checkpoint schema version")
    try:
        return ConversationCheckpointState.model_validate(copied)
    except ValidationError as exc:
        raise StateContractError("invalid checkpoint state") from exc
