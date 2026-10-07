"""Closed persisted projection contracts for the durable checkpoint.

This module defines the runtime-side contract layer:

- two closed root DTOs (``PersistedAgentStateV1``, ``PersistedPendingWriteV1``)
  with ``extra=forbid``, strict types, registered schema versions and no
  ``dict[str, Any]`` escape hatches;
- ``ContentSourceReferenceV1`` as the only content placeholder, with the
  stable domain-tagged SHA-256 preimage;
- ``ContentReferenceSequenceV1`` as the only sequence contract for content
  references: a named, closed, logically immutable, bounded DTO whose canonical
  JSON is an array, validated by a strict deserializer that never silently
  sorts and never accepts raw tuples or unvalidated lists;
- the typed tool metadata policy engine (rules injected by the upper adapter);
- ``SanitizedProjectionError`` and the sanitized raise helper.

Dependency direction: this module imports no Agent Pydantic models — only the
shared thread-identity utility (the same leaf dependency ``runtime/context.py``
already uses) — and the agent-side adapter/projector imports these contracts,
never the reverse.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Any, Literal, NoReturn

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    StringConstraints,
    TypeAdapter,
    field_serializer,
    field_validator,
    model_validator,
)

from app.agent.thread_identity import derive_thread_id

# ---------------------------------------------------------------------------
# Sanitized exception boundary
# ---------------------------------------------------------------------------


class SanitizedProjectionError(RuntimeError):
    """Stable, content-free rejection carrying only safe scalar facts."""

    __slots__ = ("reason", "stage", "detail")

    def __init__(self, reason: str, *, stage: str, detail: str = "") -> None:
        safe_reason = (
            reason
            if type(reason) is str and re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", reason)
            else "PROJECTION_REJECTED"
        )
        safe_stage = (
            stage
            if type(stage) is str and re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", stage)
            else "PROJECTION"
        )
        del detail
        self.reason = safe_reason
        self.stage = safe_stage
        self.detail = ""
        super().__init__(
            f"projection boundary rejected: {safe_reason} ({safe_stage})"
        )


def raise_sanitized_projection_error(
    reason: str,
    *,
    stage: str,
    detail: str = "",
) -> NoReturn:
    """Raise a sanitized error whose graph is clean in any caller scope.

    ``raise ... from None`` only suppresses the display of ``__context__``
    without clearing it; the raise -> self-catch -> strip -> bare re-raise
    sequence below keeps both ``__cause__`` and ``__context__`` as ``None``
    even when invoked inside an active ``except`` block.
    """
    sanitized = SanitizedProjectionError(reason, stage=stage, detail=detail)
    try:
        raise sanitized
    except SanitizedProjectionError:
        sanitized.__context__ = None
        raise


# ---------------------------------------------------------------------------
# Frozen content vocabulary
# ---------------------------------------------------------------------------

_DIGEST_DOMAIN_TAG = b"dianshang-agent/content-source-sha256/v1\x00"


class SourceKind(StrEnum):
    CHAT_MESSAGE = "CHAT_MESSAGE"
    AGENT_AUDIT_CONTENT = "AGENT_AUDIT_CONTENT"
    ACTION_RECORD = "ACTION_RECORD"
    RAG_DOCUMENT = "RAG_DOCUMENT"


class ContentRole(StrEnum):
    QUESTION = "QUESTION"
    EFFECTIVE_QUESTION = "EFFECTIVE_QUESTION"
    PLAN_GOAL = "PLAN_GOAL"
    PLAN_REASON = "PLAN_REASON"
    PLAN_MISSING_INFORMATION = "PLAN_MISSING_INFORMATION"
    CURRENT_ISSUE = "CURRENT_ISSUE"
    CONVERSATION_SUMMARY = "CONVERSATION_SUMMARY"
    RETRIEVAL_FILE_NAME = "RETRIEVAL_FILE_NAME"
    RETRIEVAL_SNIPPET = "RETRIEVAL_SNIPPET"
    TOOL_CONTENT = "TOOL_CONTENT"
    DRAFT_ANSWER = "DRAFT_ANSWER"
    FINAL_ANSWER = "FINAL_ANSWER"
    ERROR_DETAIL = "ERROR_DETAIL"


#: The frozen closed role set in canonical order (13 roles).
CONTENT_ROLE_ORDER: tuple[ContentRole, ...] = (
    ContentRole.QUESTION,
    ContentRole.EFFECTIVE_QUESTION,
    ContentRole.PLAN_GOAL,
    ContentRole.PLAN_REASON,
    ContentRole.PLAN_MISSING_INFORMATION,
    ContentRole.CURRENT_ISSUE,
    ContentRole.CONVERSATION_SUMMARY,
    ContentRole.RETRIEVAL_FILE_NAME,
    ContentRole.RETRIEVAL_SNIPPET,
    ContentRole.TOOL_CONTENT,
    ContentRole.DRAFT_ANSWER,
    ContentRole.FINAL_ANSWER,
    ContentRole.ERROR_DETAIL,
)

_ROLE_RANK: dict[ContentRole, int] = {
    role: rank for rank, role in enumerate(CONTENT_ROLE_ORDER)
}

_SEQUENCE_MAX_ITEMS = len(CONTENT_ROLE_ORDER)

_FIXED_CONTENT_SLOT_ROLES: dict[str, ContentRole] = {
    "memory.current_issue": ContentRole.CURRENT_ISSUE,
    "memory.conversation_summary": ContentRole.CONVERSATION_SUMMARY,
    "active_run.question": ContentRole.QUESTION,
    "active_run.effective_question": ContentRole.EFFECTIVE_QUESTION,
    "active_run.decision_reason": ContentRole.PLAN_REASON,
    "active_run.draft_answer": ContentRole.DRAFT_ANSWER,
    "active_run.final_answer": ContentRole.FINAL_ANSWER,
    "active_run.error_summary": ContentRole.ERROR_DETAIL,
    "active_run.plan.goal": ContentRole.PLAN_GOAL,
    "active_run.plan.decision_reason": ContentRole.PLAN_REASON,
}
_INDEXED_CONTENT_SLOT_ROLES: tuple[tuple[re.Pattern[str], ContentRole], ...] = (
    (
        re.compile(r"^active_run\.plan\.missing_information\[[0-9]+\]$"),
        ContentRole.PLAN_MISSING_INFORMATION,
    ),
    (
        re.compile(r"^active_run\.retrieval_evidence\[[0-9]+\]\.file_name$"),
        ContentRole.RETRIEVAL_FILE_NAME,
    ),
    (
        re.compile(r"^active_run\.retrieval_evidence\[[0-9]+\]\.snippet$"),
        ContentRole.RETRIEVAL_SNIPPET,
    ),
    (
        re.compile(r"^active_run\.response_meta\.sources\[[0-9]+\]\.file_name$"),
        ContentRole.RETRIEVAL_FILE_NAME,
    ),
    (
        re.compile(r"^active_run\.response_meta\.sources\[[0-9]+\]\.snippet$"),
        ContentRole.RETRIEVAL_SNIPPET,
    ),
)


def canonical_content_role_for_slot(slot: str) -> ContentRole:
    """Return the one frozen role for a persisted content slot.

    This leaf-level registry is shared by the Agent collector and MySQL source
    adapter.  It deliberately validates a single slot; it does not impose the
    pending-write sequence's role-uniqueness rule on a whole state.
    """

    role = _FIXED_CONTENT_SLOT_ROLES.get(slot)
    if role is not None:
        return role
    for pattern, indexed_role in _INDEXED_CONTENT_SLOT_ROLES:
        if pattern.fullmatch(slot) is not None:
            return indexed_role
    raise ValueError("content slot is not registered")


class NormalizationVersion(StrEnum):
    RAW_UTF8_V1 = "RAW_UTF8_V1"


class ProducingPrincipalKind(StrEnum):
    USER = "USER"
    SERVICE = "SERVICE"


def _lp64(data: bytes) -> bytes:
    return struct.pack(">Q", len(data)) + data


def compute_content_digest(
    *,
    source_kind: str | SourceKind,
    content_role: str | ContentRole,
    content_schema_version: int,
    normalization_version: str | NormalizationVersion,
    content: str,
) -> str:
    """The only V1 preimage:

    ``ASCII("dianshang-agent/content-source-sha256/v1\\0") || LP64(ASCII(source_kind))
    || LP64(ASCII(content_role)) || U64BE(content_schema_version)
    || LP64(ASCII(normalization_version)) || LP64(raw_utf8_content)``

    ``RAW_UTF8_V1`` performs no implicit normalization of any kind.
    """
    preimage = _DIGEST_DOMAIN_TAG
    preimage += _lp64(str(source_kind).encode("ascii"))
    preimage += _lp64(str(content_role).encode("ascii"))
    preimage += struct.pack(">Q", content_schema_version)
    preimage += _lp64(str(normalization_version).encode("ascii"))
    preimage += _lp64(content.encode("utf-8"))
    return hashlib.sha256(preimage).hexdigest()


# ---------------------------------------------------------------------------
# Restricted scalar vocabularies
# ---------------------------------------------------------------------------

StrictPositiveInt = Annotated[int, Field(gt=0, strict=True)]
StrictNonNegativeInt = Annotated[int, Field(ge=0, strict=True)]
StrictFiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]
SafeIdentifier = Annotated[
    str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
]
PendingWriteChannel = SafeIdentifier | Literal["__interrupt__", "__resume__"]
Hex64 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Rfc3339Text = Annotated[
    str,
    StringConstraints(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$"),
]
NormalizedCode = Annotated[str, StringConstraints(pattern=r"^[A-Z][A-Z0-9_]{0,63}$")]
OrderNoIdentifier = Annotated[
    str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9-]{0,31}$")
]
SkuIdentifier = Annotated[
    str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
]
UuidFormText = Annotated[
    str,
    StringConstraints(
        pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    ),
]

IntentProjection = Literal[
    "ORDER_QUERY",
    "SHIPPING_QUERY",
    "PRODUCT_QUERY",
    "KNOWLEDGE_QUERY",
    "CANCEL_ORDER",
    "REFUND_REQUEST",
    "CREATE_TICKET",
    "CLARIFICATION",
]
RiskLevelProjection = Literal["LOW", "MEDIUM", "HIGH", "FORBIDDEN"]
RunStatusProjection = Literal[
    "RUNNING",
    "WAITING_CUSTOMER_CONFIRMATION",
    "WAITING_ADMIN_APPROVAL",
    "RESUME_PENDING",
    "EXECUTING",
    "COMPLETED",
    "REJECTED",
    "FAILED",
    "CANCELLED",
]
ToolStatusProjection = Literal["SUCCEEDED", "FAILED", "BLOCKED"]
RetrievalChannelProjection = Literal[
    "keyword", "dense", "structured_rule", "fused", "reranked"
]
ConfirmationStatusProjection = Literal[
    "NOT_REQUIRED", "PENDING", "CONFIRMED", "REJECTED"
]
ApprovalDecisionProjection = Literal["PENDING", "APPROVED", "REJECTED", "STALE"]
EffectPhaseProjection = Literal["READ_ONLY", "ACTION_PREPARE", "BUSINESS_EXECUTE"]
MemoryProvenanceField = Literal[
    "active_order_no", "active_product_code", "current_issue", "last_intent"
]

CONFIRMATION_TEMPLATE_VERSION: Literal["customer-confirmation-v1"] = (
    "customer-confirmation-v1"
)

_ACTION_LABELS: Mapping[str, str] = {
    "REFUND": "退款",
    "ORDER_CANCELLATION": "取消订单",
}


def render_customer_confirmation_challenge(
    action_type: str,
    target_order_no: str,
) -> str:
    """Deterministically rebuild the confirmation challenge from the frozen
    template ``请回复：确认<动作> <完整订单号>`` and the finite action-label map.

    The challenge text itself is never persisted or referenced; only the
    template version travels inside the projection.
    """
    label = _ACTION_LABELS.get(action_type)
    if label is None:
        raise_sanitized_projection_error(
            "ACTION_LABEL_UNKNOWN", stage="CONFIRMATION_TEMPLATE"
        )
    return f"请回复：确认{label} {target_order_no}"


class ClosedModel(BaseModel):
    """Frozen, closed base: unknown fields are always rejected."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------------
# ContentSourceReferenceV1
# ---------------------------------------------------------------------------


class ContentSourceReferenceV1(ClosedModel):
    source_kind: SourceKind
    source_record_id: StrictPositiveInt | SafeIdentifier
    content_role: ContentRole
    content_schema_version: Literal[1]
    content_sha256: Hex64
    conversation_id: StrictPositiveInt
    subject_user_id: StrictPositiveInt
    producing_principal_kind: ProducingPrincipalKind
    producing_actor_id: StrictPositiveInt | None = None
    producing_service_principal: SafeIdentifier | None = None
    run_id: SafeIdentifier
    producing_attempt_id: SafeIdentifier
    source_revision: StrictPositiveInt
    normalization_version: NormalizationVersion

    @model_validator(mode="after")
    def _principal_exclusivity(self) -> ContentSourceReferenceV1:
        if self.producing_principal_kind is ProducingPrincipalKind.USER:
            if self.producing_actor_id is None or self.producing_service_principal is not None:
                raise ValueError(
                    "USER producing principal requires producing_actor_id only"
                )
        elif self.producing_actor_id is not None or self.producing_service_principal is None:
            raise ValueError(
                "SERVICE producing principal requires producing_service_principal only"
            )
        return self


# ---------------------------------------------------------------------------
# ContentReferenceSequenceV1
# ---------------------------------------------------------------------------


def _validate_sequence_items(
    items: tuple[ContentSourceReferenceV1, ...],
) -> tuple[ContentSourceReferenceV1, ...]:
    if len(items) > _SEQUENCE_MAX_ITEMS:
        raise_sanitized_projection_error(
            "ROLE_COUNT_EXCEEDED",
            stage="CONTENT_SEQUENCE",
            detail=f"sequence length {len(items)} exceeds {_SEQUENCE_MAX_ITEMS}",
        )
    seen: set[ContentRole] = set()
    last_rank = -1
    for item in items:
        if not isinstance(item, ContentSourceReferenceV1):
            raise_sanitized_projection_error(
                "SEQUENCE_ITEM_INVALID",
                stage="CONTENT_SEQUENCE",
                detail="items must be ContentSourceReferenceV1 instances",
            )
        role = item.content_role
        if role in seen:
            raise_sanitized_projection_error(
                "ROLE_DUPLICATED", stage="CONTENT_SEQUENCE"
            )
        rank = _ROLE_RANK[role]
        if rank <= last_rank:
            raise_sanitized_projection_error(
                "ROLE_ORDER_INVALID",
                stage="CONTENT_SEQUENCE",
                detail="items must already be in canonical role order",
            )
        seen.add(role)
        last_rank = rank
    return items


class ContentReferenceSequenceV1:
    """The only closed sequence contract for content references.

    A named, logically immutable, bounded DTO — never a raw Python tuple,
    ordinary list or arbitrary JSON container.  Canonical JSON is an array of
    ``ContentSourceReferenceV1`` objects; the strict deserializer is the only
    JSON-array entry point and never silently sorts its input.
    """

    __slots__ = ("_items",)
    _items: tuple[ContentSourceReferenceV1, ...]

    def __init__(self, items: object = None) -> None:
        del items
        raise TypeError(
            "ContentReferenceSequenceV1 must be built with of() or decode_canonical()"
        )

    @classmethod
    def _create(
        cls, items: tuple[ContentSourceReferenceV1, ...]
    ) -> ContentReferenceSequenceV1:
        instance = object.__new__(cls)
        object.__setattr__(instance, "_items", _validate_sequence_items(items))
        return instance

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("content reference sequence is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("content reference sequence is immutable")

    @classmethod
    def of(cls, *items: ContentSourceReferenceV1) -> ContentReferenceSequenceV1:
        """Strict constructor from already-validated reference instances."""
        return cls._create(tuple(items))

    @classmethod
    def decode_canonical(cls, data: bytes | str) -> ContentReferenceSequenceV1:
        """The only strict JSON-array deserializer into this nominal DTO."""
        raw = data.encode("utf-8") if type(data) is str else data
        if type(raw) is not bytes:
            raise_sanitized_projection_error(
                "SEQUENCE_CANONICAL_BYTES_REQUIRED", stage="CONTENT_SEQUENCE"
            )
        reason: str | None = None
        try:
            parsed = json.loads(raw)
            if type(parsed) is not list:
                raise ValueError("canonical content sequence must be an array")
            items = _REFERENCE_LIST_ADAPTER.validate_json(raw)
            sequence = cls._create(tuple(items))
            if sequence.encode_canonical() != raw:
                raise ValueError("content sequence is not canonical JSON")
        except Exception:
            reason = "SEQUENCE_CANONICAL_INVALID"
        else:
            return sequence
        del data, raw
        raise_sanitized_projection_error(reason, stage="CONTENT_SEQUENCE")

    def encode_canonical(self) -> bytes:
        return json.dumps(
            [item.model_dump(mode="json") for item in self._items],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")

    @property
    def items(self) -> tuple[ContentSourceReferenceV1, ...]:
        return self._items

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[ContentSourceReferenceV1]:
        return iter(self._items)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ContentReferenceSequenceV1):
            return self._items == other._items
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._items)


_REFERENCE_LIST_ADAPTER: TypeAdapter[list[ContentSourceReferenceV1]] = TypeAdapter(
    list[ContentSourceReferenceV1]
)


def _coerce_sequence_field(value: Any) -> ContentReferenceSequenceV1:
    """Before-validator for DTO fields typed as the nominal sequence.

    Only an already-constructed nominal instance passes.  Canonical arrays
    enter through ``ContentReferenceSequenceV1.decode_canonical`` before root
    DTO construction; Python lists and tuples are never auto-promoted.
    """
    if isinstance(value, ContentReferenceSequenceV1):
        return value
    raise ValueError("content_references must be the nominal sequence DTO")


def _serialize_sequence_field(value: ContentReferenceSequenceV1) -> list[dict[str, Any]]:
    return [item.model_dump(mode="json") for item in value.items]


# ---------------------------------------------------------------------------
# Content slot references (runtime-side closed input for the state projector)
# ---------------------------------------------------------------------------

_SLOT_KEY_PATTERN_MAX_PARTS = 8


class ContentSlotReferences:
    """Slot-key -> reference mapping consumed by the state projector.

    Keys follow the frozen slot vocabulary (``memory.current_issue``,
    ``active_run.question``, ``active_run.plan.missing_information[0]`` …).
    The projector derives the expected key set from the runtime state and
    requires an exact match, so an unknown or missing slot fails closed.
    """

    __slots__ = ("_mapping",)
    _mapping: dict[str, ContentSourceReferenceV1]

    def __init__(
        self,
        mapping: Mapping[str, ContentSourceReferenceV1] | None = None,
    ) -> None:
        validated: dict[str, ContentSourceReferenceV1] = {}
        for key, value in (mapping or {}).items():
            if not _is_valid_slot_key(key):
                raise ValueError(f"invalid content slot key: {key!r}")
            if not isinstance(value, ContentSourceReferenceV1):
                raise ValueError("slot values must be ContentSourceReferenceV1")
            validated[key] = value
        object.__setattr__(self, "_mapping", validated)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("content slot references are immutable")

    @property
    def mapping(self) -> Mapping[str, ContentSourceReferenceV1]:
        return dict(self._mapping)

    def get(self, slot: str) -> ContentSourceReferenceV1 | None:
        return self._mapping.get(slot)

    def require_exact(self, expected: Iterable[str]) -> None:
        expected_keys = set(expected)
        provided = set(self._mapping)
        missing = expected_keys - provided
        if missing:
            raise_sanitized_projection_error(
                "CONTENT_REFERENCE_MISSING",
                stage="CONTENT_BINDING",
                detail=f"missing slots: {len(missing)}",
            )
        extra = provided - expected_keys
        if extra:
            raise_sanitized_projection_error(
                "CONTENT_REFERENCE_UNEXPECTED",
                stage="CONTENT_BINDING",
                detail=f"unexpected slots: {len(extra)}",
            )


def _is_valid_slot_key(key: str) -> bool:
    if not key or len(key) > 256:
        return False
    parts = key.split(".")
    if not 2 <= len(parts) <= _SLOT_KEY_PATTERN_MAX_PARTS:
        return False
    for part in parts:
        name, _, index = part.partition("[")
        if not name or not name.replace("_", "a").islower() or not name.isidentifier():
            return False
        if index:
            if not index.endswith("]") or not index[:-1].isdigit():
                return False
    return True


# ---------------------------------------------------------------------------
# Typed tool metadata policy
# ---------------------------------------------------------------------------


class ToolMetadataTypeKind(StrEnum):
    STRICT_BOOL = "STRICT_BOOL"
    FINITE_ENUM = "FINITE_ENUM"
    POSITIVE_INT = "POSITIVE_INT"
    NON_NEGATIVE_INT = "NON_NEGATIVE_INT"
    RESTRICTED_IDENTIFIER = "RESTRICTED_IDENTIFIER"


@dataclass(frozen=True, slots=True)
class ToolMetadataKeyRule:
    key: str
    type_kind: ToolMetadataTypeKind
    allowed_values: frozenset[str] = frozenset()


class ToolMetadataEntryV1(ClosedModel):
    key: SafeIdentifier
    value: StrictBool | StrictInt | StrictStr


RegisteredToolNameProjection = Literal[
    "list_my_orders",
    "get_order_detail",
    "get_product_information",
    "search_knowledge_base",
    "create_support_ticket",
    "request_order_cancellation",
    "request_refund",
]

TOOL_METADATA_POLICY_VERSION: Literal[1] = 1
TOOL_METADATA_SCHEMA_VERSION: Literal[1] = 1


def _validate_unique_metadata_entries(
    entries: tuple[ToolMetadataEntryV1, ...],
) -> tuple[ToolMetadataEntryV1, ...]:
    seen: set[str] = set()
    for entry in entries:
        if entry.key in seen:
            raise ValueError("typed tool metadata keys must be unique")
        seen.add(entry.key)
    return entries


class ToolMetadataPolicyCatalog:
    """Versioned ``tool_name + metadata_key`` typed-rule catalog.

    The catalog is injected by the upper adapter; this runtime engine never
    imports Agent models or tool implementations.  Registered keys with wrong
    types, out-of-domain values, PII/free text, unknown keys and unknown
    tools all fail closed.
    """

    def __init__(
        self,
        rules: Mapping[str, Mapping[str, ToolMetadataKeyRule]],
    ) -> None:
        validated: dict[str, dict[str, ToolMetadataKeyRule]] = {}
        for tool_name, key_rules in rules.items():
            if not tool_name or len(tool_name) > 128:
                raise ValueError("tool rule keys must be bounded identifiers")
            if not key_rules:
                raise ValueError("every registered tool requires a metadata schema")
            entries: dict[str, ToolMetadataKeyRule] = {}
            for rule in key_rules.values():
                if not rule.key or len(rule.key) > 64:
                    raise ValueError("metadata rule keys must be bounded")
                if rule.type_kind is ToolMetadataTypeKind.FINITE_ENUM and not rule.allowed_values:
                    raise ValueError("finite enum rules must declare values")
                if rule.type_kind is not ToolMetadataTypeKind.FINITE_ENUM and rule.allowed_values:
                    raise ValueError("only finite enum rules declare values")
                entries[rule.key] = rule
            if len(entries) > 32:
                raise ValueError("tool metadata rules exceed 32 keys")
            validated[tool_name] = entries
        self._rules = validated

    def validate(
        self,
        tool_name: str,
        metadata: Mapping[str, Any],
        *,
        tool_policy_version: int = TOOL_METADATA_POLICY_VERSION,
        metadata_schema_version: int = TOOL_METADATA_SCHEMA_VERSION,
    ) -> tuple[ToolMetadataEntryV1, ...]:
        failure: tuple[str, str, str] | None = None
        try:
            result = self._validate_impl(
                tool_name,
                metadata,
                tool_policy_version=tool_policy_version,
                metadata_schema_version=metadata_schema_version,
            )
        except SanitizedProjectionError as error:
            failure = (error.reason, error.stage, error.detail)
        except Exception:
            failure = ("TOOL_METADATA_INVALID", "TOOL_METADATA", "")
        else:
            return result
        del self, tool_name, metadata, tool_policy_version, metadata_schema_version
        reason, stage, detail = failure
        raise_sanitized_projection_error(reason, stage=stage, detail=detail)

    def _validate_impl(
        self,
        tool_name: str,
        metadata: Mapping[str, Any],
        *,
        tool_policy_version: int,
        metadata_schema_version: int,
    ) -> tuple[ToolMetadataEntryV1, ...]:
        if type(tool_policy_version) is not int or (
            tool_policy_version != TOOL_METADATA_POLICY_VERSION
        ):
            raise_sanitized_projection_error(
                "TOOL_POLICY_VERSION_UNKNOWN", stage="TOOL_METADATA"
            )
        if type(metadata_schema_version) is not int or (
            metadata_schema_version != TOOL_METADATA_SCHEMA_VERSION
        ):
            raise_sanitized_projection_error(
                "TOOL_METADATA_SCHEMA_UNKNOWN", stage="TOOL_METADATA"
            )
        key_rules = self._rules.get(tool_name)
        if key_rules is None:
            raise_sanitized_projection_error(
                "TOOL_NOT_REGISTERED", stage="TOOL_METADATA"
            )
        if len(metadata) > 32:
            raise_sanitized_projection_error(
                "TOOL_METADATA_TOO_LARGE", stage="TOOL_METADATA"
            )
        entries: list[ToolMetadataEntryV1] = []
        for key, value in sorted(metadata.items()):
            rule = key_rules.get(key)
            if rule is None:
                raise_sanitized_projection_error(
                    "TOOL_METADATA_KEY_NOT_REGISTERED", stage="TOOL_METADATA"
                )
            validated_value = self._validate_value(rule, value)
            entries.append(ToolMetadataEntryV1(key=key, value=validated_value))
        return tuple(entries)

    def _validate_value(self, rule: ToolMetadataKeyRule, value: Any) -> Any:
        kind = rule.type_kind
        if kind is ToolMetadataTypeKind.STRICT_BOOL:
            if type(value) is not bool:
                raise_sanitized_projection_error(
                    "TOOL_METADATA_TYPE_MISMATCH", stage="TOOL_METADATA"
                )
            return value
        if kind is ToolMetadataTypeKind.FINITE_ENUM:
            if type(value) is not str or value not in rule.allowed_values:
                raise_sanitized_projection_error(
                    "TOOL_METADATA_VALUE_OUT_OF_DOMAIN", stage="TOOL_METADATA"
                )
            return value
        if kind is ToolMetadataTypeKind.POSITIVE_INT:
            if type(value) is not int or value <= 0:
                raise_sanitized_projection_error(
                    "TOOL_METADATA_VALUE_OUT_OF_DOMAIN", stage="TOOL_METADATA"
                )
            return value
        if kind is ToolMetadataTypeKind.NON_NEGATIVE_INT:
            if type(value) is not int or value < 0:
                raise_sanitized_projection_error(
                    "TOOL_METADATA_VALUE_OUT_OF_DOMAIN", stage="TOOL_METADATA"
                )
            return value
        if type(value) is not str or len(value) > 64 or not _is_restricted_identifier(value):
            raise_sanitized_projection_error(
                "TOOL_METADATA_VALUE_OUT_OF_DOMAIN", stage="TOOL_METADATA"
            )
        return value

    def registered_tools(self) -> frozenset[str]:
        return frozenset(self._rules)


def _is_restricted_identifier(value: str) -> bool:
    if not value:
        return False
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    return all(char in allowed for char in value)


# ---------------------------------------------------------------------------
# PersistedAgentStateV1 (closed root DTO)
# ---------------------------------------------------------------------------

_MYSQL_SIGNED_INT_MIN = -(2**31)
_MYSQL_SIGNED_INT_MAX = 2**31 - 1
SignedWriteIndex = Annotated[
    int,
    Field(
        ge=_MYSQL_SIGNED_INT_MIN,
        le=_MYSQL_SIGNED_INT_MAX,
        strict=True,
    ),
]


class PersistedConversationIdentityV1(ClosedModel):
    conversation_id: StrictPositiveInt
    thread_id: SafeIdentifier
    subject_user_id: StrictPositiveInt
    subject_role_snapshot: NormalizedCode

    @model_validator(mode="after")
    def _thread_formula(self) -> PersistedConversationIdentityV1:
        if self.thread_id != derive_thread_id(self.conversation_id):
            raise ValueError("thread_id must equal the canonical conversation thread")
        return self


class PersistedExecutionExpectationV1(ClosedModel):
    run_id: SafeIdentifier
    attempt_id: SafeIdentifier
    expected_fence_version: StrictPositiveInt | None = None


class MemoryProvenanceProjectionV1(ClosedModel):
    field_name: MemoryProvenanceField
    source_type: NormalizedCode
    source_ref: SafeIdentifier
    observed_at: Rfc3339Text
    memory_revision: StrictNonNegativeInt


class PersistedMemoryProjectionV1(ClosedModel):
    active_order_no: OrderNoIdentifier | None = None
    active_product_code: SkuIdentifier | None = None
    current_issue: ContentSourceReferenceV1 | None = None
    last_intent: NormalizedCode | None = None
    provenance: tuple[MemoryProvenanceProjectionV1, ...] = Field(
        default=(), max_length=32
    )
    memory_revision: StrictNonNegativeInt = 0
    conversation_summary: ContentSourceReferenceV1 | None = None
    summary_until_message_id: StrictNonNegativeInt | None = None
    summary_revision: StrictNonNegativeInt = 0

    @model_validator(mode="after")
    def _summary_invariants(self) -> PersistedMemoryProjectionV1:
        if self.conversation_summary is not None:
            if self.summary_until_message_id is None or self.summary_revision < 1:
                raise ValueError(
                    "a non-empty summary requires a cursor and a positive revision"
                )
        return self


class PersistedOrderReferenceProjectionV1(ClosedModel):
    order_no: OrderNoIdentifier | None = None
    ordinal_index: StrictPositiveInt | None = None
    latest: bool = Field(default=False, strict=True)
    list_all: bool = Field(default=False, strict=True)


class PersistedPlanProjectionV1(ClosedModel):
    intent: IntentProjection
    goal: ContentSourceReferenceV1 | None = None
    order_reference: PersistedOrderReferenceProjectionV1 | None = None
    product_reference: SkuIdentifier | None = None
    required_tools: tuple[SafeIdentifier, ...] = Field(default=(), max_length=32)
    action_type: NormalizedCode | None = None
    risk_level: RiskLevelProjection
    confirmation_required: bool = Field(strict=True)
    missing_information: tuple[ContentSourceReferenceV1, ...] = Field(
        default=(), max_length=32
    )
    decision_reason: ContentSourceReferenceV1 | None = None


class ToolProjectionV1(ClosedModel):
    tool_name: RegisteredToolNameProjection
    tool_policy_version: Literal[1] = TOOL_METADATA_POLICY_VERSION
    metadata_schema_version: Literal[1] = TOOL_METADATA_SCHEMA_VERSION
    status: ToolStatusProjection
    result_ref: SafeIdentifier | None = None
    observed_at: Rfc3339Text
    error_type: NormalizedCode | None = None
    safe_metadata: tuple[ToolMetadataEntryV1, ...] = Field(default=(), max_length=32)

    @field_validator("safe_metadata")
    @classmethod
    def _unique_safe_metadata(
        cls, value: tuple[ToolMetadataEntryV1, ...]
    ) -> tuple[ToolMetadataEntryV1, ...]:
        return _validate_unique_metadata_entries(value)


class RetrievalProjectionV1(ClosedModel):
    document_id: StrictNonNegativeInt
    chunk_ref: SafeIdentifier
    score: StrictFiniteFloat
    channel: RetrievalChannelProjection
    file_name: ContentSourceReferenceV1
    snippet: ContentSourceReferenceV1


class ActionDraftProjectionV1(ClosedModel):
    logical_action_id: UuidFormText
    action_type: NormalizedCode
    target_order_id: StrictPositiveInt
    target_order_no: OrderNoIdentifier
    subject_user_id: StrictPositiveInt
    reason_code: NormalizedCode
    policy_version: SafeIdentifier
    draft_revision: StrictPositiveInt
    expires_at: Rfc3339Text
    nonce_digest: Hex64


class AuthorizationAssociationProjectionV1(ClosedModel):
    authorization_id: SafeIdentifier
    run_id: SafeIdentifier
    logical_action_id: UuidFormText
    action_type: NormalizedCode
    subject_user_id: StrictPositiveInt
    target_order_id: StrictPositiveInt
    target_order_no: OrderNoIdentifier
    effect_phase: EffectPhaseProjection
    policy_version: SafeIdentifier
    draft_revision: StrictPositiveInt
    issued_at: Rfc3339Text
    expires_at: Rfc3339Text


class ResponseMetaProjectionV1(ClosedModel):
    sources: tuple[RetrievalProjectionV1, ...] = Field(default=(), max_length=50)
    retrieval_score: StrictFiniteFloat
    confidence_level: NormalizedCode
    need_human: bool = Field(strict=True)
    ticket_id: StrictPositiveInt | None = None


class PersistedActiveRunProjectionV1(ClosedModel):
    run_id: SafeIdentifier
    attempt_id: SafeIdentifier
    run_status: RunStatusProjection
    question: ContentSourceReferenceV1 | None = None
    effective_question: ContentSourceReferenceV1 | None = None
    current_user_message_id: StrictPositiveInt | None = None
    blocked: bool = Field(default=False, strict=True)
    intent: IntentProjection
    risk_level: RiskLevelProjection
    plan: PersistedPlanProjectionV1 | None = None
    selected_tools: tuple[SafeIdentifier, ...] = Field(default=(), max_length=32)
    tool_results: tuple[ToolProjectionV1, ...] = Field(default=(), max_length=64)
    retrieval_evidence: tuple[RetrievalProjectionV1, ...] = Field(
        default=(), max_length=50
    )
    retrieval_score: StrictFiniteFloat | None = None
    action_draft: ActionDraftProjectionV1 | None = None
    side_effect_authorization: AuthorizationAssociationProjectionV1 | None = None
    pending_action_id: StrictPositiveInt | None = None
    customer_confirmation_status: ConfirmationStatusProjection
    approval_decision: ApprovalDecisionProjection | None = None
    decision_reason: ContentSourceReferenceV1 | None = None
    confirmation_template_version: Literal["customer-confirmation-v1"] | None = None
    draft_answer: ContentSourceReferenceV1 | None = None
    final_answer: ContentSourceReferenceV1 | None = None
    response_meta: ResponseMetaProjectionV1 | None = None
    error_type: NormalizedCode | None = None
    error_detail: ContentSourceReferenceV1 | None = None

    @model_validator(mode="after")
    def _waiting_confirmation_invariants(self) -> PersistedActiveRunProjectionV1:
        if self.run_status == "WAITING_CUSTOMER_CONFIRMATION":
            if self.confirmation_template_version != CONFIRMATION_TEMPLATE_VERSION:
                raise ValueError(
                    "WAITING_CUSTOMER_CONFIRMATION requires the registered template version"
                )
            if self.action_draft is None:
                raise ValueError(
                    "WAITING_CUSTOMER_CONFIRMATION requires the action draft"
                )
        elif self.confirmation_template_version is not None:
            raise ValueError(
                "confirmation template version only applies to waiting confirmation"
            )
        if self.pending_action_id is not None and self.action_draft is None:
            raise ValueError("pending action requires the action draft")
        return self


class ProjectionPublicationBindingV1(ClosedModel):
    logical_namespace: SafeIdentifier
    physical_namespace: SafeIdentifier | None = None
    checkpoint_id: SafeIdentifier | None = None
    expected_publication_version: StrictNonNegativeInt | None = None
    expected_previous_pointer: SafeIdentifier | None = None


class PersistedAgentStateV1(ClosedModel):
    projection_schema_version: Literal[1] = 1
    runtime_state_schema_version: StrictPositiveInt
    protection_schema_version: StrictPositiveInt
    policy_schema_version: StrictPositiveInt
    identity: PersistedConversationIdentityV1
    execution: PersistedExecutionExpectationV1 | None = None
    memory: PersistedMemoryProjectionV1
    active_run: PersistedActiveRunProjectionV1 | None = None
    publication_binding: ProjectionPublicationBindingV1

    @model_validator(mode="after")
    def _root_invariants(self) -> PersistedAgentStateV1:
        registered = {1}
        for version in (
            self.runtime_state_schema_version,
            self.protection_schema_version,
            self.policy_schema_version,
        ):
            if version not in registered:
                raise ValueError(
                    "unknown runtime/protection/policy schema version"
                )
        if (self.execution is None) != (self.active_run is None):
            raise ValueError(
                "execution expectation exists exactly when an active run exists"
            )
        if self.active_run is not None:
            if self.active_run.run_id != self.execution.run_id:  # type: ignore[union-attr]
                raise ValueError("execution run must match the active run")
            if self.active_run.attempt_id != self.execution.attempt_id:  # type: ignore[union-attr]
                raise ValueError("execution attempt must match the active run")
        return self


# ---------------------------------------------------------------------------
# PersistedPendingWriteV1 (closed root DTO, second physical write surface)
# ---------------------------------------------------------------------------


class PendingWritePurposeV1(StrEnum):
    CHANNEL_WRITE = "CHANNEL_WRITE"
    STATE_UPDATE = "STATE_UPDATE"


class PendingWriteDomainV1(ClosedModel):
    thread_id: SafeIdentifier
    conversation_id: StrictPositiveInt
    run_id: SafeIdentifier
    attempt_id: SafeIdentifier
    fence_version: StrictPositiveInt | None = None
    logical_namespace: SafeIdentifier
    physical_namespace: SafeIdentifier | None = None
    checkpoint_id: SafeIdentifier | None = None
    expected_publication_version: StrictNonNegativeInt | None = None
    expected_previous_pointer: SafeIdentifier | None = None

    @model_validator(mode="after")
    def _thread_formula(self) -> PendingWriteDomainV1:
        if self.thread_id != derive_thread_id(self.conversation_id):
            raise ValueError("thread_id must equal the canonical conversation thread")
        return self


class StructuralControlValueV1(ClosedModel):
    value_kind: Literal["STRUCTURAL_CONTROL"]
    control: Literal["NO_OP", "INTERRUPT", "RESUME"]
    control_schema_version: Literal[1] = 1
    control_id: Hex64
    interrupt_namespaces: tuple[tuple[SafeIdentifier, ...], ...] = Field(
        default=(),
        max_length=16,
    )
    resume_shape: Literal["ACK", "ACK_SEQUENCE"] | None = None
    resume_count: StrictNonNegativeInt = 0

    @model_validator(mode="after")
    def _registered_structural_shape(self) -> StructuralControlValueV1:
        if any(not namespace or len(namespace) > 16 for namespace in self.interrupt_namespaces):
            raise ValueError("interrupt namespaces must be non-empty and bounded")
        if self.control == "INTERRUPT":
            if (
                not self.interrupt_namespaces
                or self.resume_shape is not None
                or self.resume_count != 0
            ):
                raise ValueError("interrupt structural control shape is invalid")
        elif self.control == "RESUME":
            if (
                self.interrupt_namespaces
                or self.resume_shape is None
                or self.resume_count <= 0
            ):
                raise ValueError("resume structural control shape is invalid")
        elif (
            self.interrupt_namespaces
            or self.resume_shape is not None
            or self.resume_count != 0
        ):
            raise ValueError("no-op structural control shape is invalid")
        return self


class StateProjectionValueV1(ClosedModel):
    value_kind: Literal["STATE_PROJECTION"]
    state: PersistedAgentStateV1


class ContentReferenceValueV1(ClosedModel):
    value_kind: Literal["CONTENT_REFERENCE"]
    reference: ContentSourceReferenceV1


class TypedToolMetadataValueV1(ClosedModel):
    value_kind: Literal["TYPED_TOOL_METADATA"]
    tool_name: RegisteredToolNameProjection
    tool_policy_version: Literal[1] = TOOL_METADATA_POLICY_VERSION
    metadata_schema_version: Literal[1] = TOOL_METADATA_SCHEMA_VERSION
    entries: tuple[ToolMetadataEntryV1, ...] = Field(default=(), max_length=32)

    @field_validator("entries")
    @classmethod
    def _unique_entries(
        cls, value: tuple[ToolMetadataEntryV1, ...]
    ) -> tuple[ToolMetadataEntryV1, ...]:
        return _validate_unique_metadata_entries(value)


class NullValueV1(ClosedModel):
    value_kind: Literal["NULL"]


PendingWriteValueV1 = Annotated[
    StructuralControlValueV1
    | StateProjectionValueV1
    | ContentReferenceValueV1
    | TypedToolMetadataValueV1
    | NullValueV1,
    Field(discriminator="value_kind"),
]


class PendingWriteSpecV1(ClosedModel):
    """Runtime-side closed input for the pending-write projector."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, arbitrary_types_allowed=True
    )

    runtime_state_schema_version: StrictPositiveInt = 1
    protection_schema_version: StrictPositiveInt = 1
    policy_schema_version: StrictPositiveInt = 1
    domain: PendingWriteDomainV1
    task_id: SafeIdentifier
    channel: PendingWriteChannel
    write_index: SignedWriteIndex
    batch_ordinal: StrictNonNegativeInt
    write_purpose: PendingWritePurposeV1
    value: PendingWriteValueV1
    content_references: ContentReferenceSequenceV1

    @field_validator("content_references", mode="before")
    @classmethod
    def _coerce_content_references(cls, value: Any) -> ContentReferenceSequenceV1:
        return _coerce_sequence_field(value)

    @field_serializer("content_references")
    def _serialize_content_references(
        self, value: ContentReferenceSequenceV1
    ) -> list[dict[str, Any]]:
        return _serialize_sequence_field(value)


class PersistedPendingWriteV1(ClosedModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, arbitrary_types_allowed=True
    )

    pending_write_schema_version: Literal[1] = 1
    projection_schema_version: Literal[1] = 1
    runtime_state_schema_version: StrictPositiveInt = 1
    protection_schema_version: StrictPositiveInt = 1
    policy_schema_version: StrictPositiveInt = 1
    payload_kind: Literal["PENDING_WRITE"] = "PENDING_WRITE"
    domain: PendingWriteDomainV1
    task_id: SafeIdentifier
    channel: PendingWriteChannel
    write_index: SignedWriteIndex
    batch_ordinal: StrictNonNegativeInt
    write_purpose: PendingWritePurposeV1
    value: PendingWriteValueV1
    content_references: ContentReferenceSequenceV1

    @field_validator("content_references", mode="before")
    @classmethod
    def _coerce_content_references(cls, value: Any) -> ContentReferenceSequenceV1:
        return _coerce_sequence_field(value)

    @field_serializer("content_references")
    def _serialize_content_references(
        self, value: ContentReferenceSequenceV1
    ) -> list[dict[str, Any]]:
        return _serialize_sequence_field(value)

    @model_validator(mode="after")
    def _registered_versions(self) -> PersistedPendingWriteV1:
        registered = {1}
        for version in (
            self.runtime_state_schema_version,
            self.protection_schema_version,
            self.policy_schema_version,
        ):
            if version not in registered:
                raise ValueError(
                    "unknown runtime/protection/policy schema version"
                )
        return self
