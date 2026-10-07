from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Protocol

from app.agent.thread_identity import ThreadIdentity
from app.runtime.checkpoint_projection import (
    CONTENT_ROLE_ORDER,
    ContentRole,
    ContentSourceReferenceV1,
    canonical_content_role_for_slot,
    compute_content_digest,
)

_MYSQL_SIGNED_BIGINT_MAX = 2**63 - 1
_PROJECTION_SLOT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\[\]-]{0,254}$")
_ROLE_RANK = {role: rank for rank, role in enumerate(CONTENT_ROLE_ORDER)}

_CHAT_MESSAGE_ROLE_PURPOSE: dict[ContentRole, tuple[str, str]] = {
    ContentRole.QUESTION: ("USER", "USER_INPUT"),
    ContentRole.EFFECTIVE_QUESTION: ("USER", "USER_INPUT"),
    ContentRole.CURRENT_ISSUE: ("USER", "USER_INPUT"),
    ContentRole.FINAL_ANSWER: ("ASSISTANT", "FINAL_ANSWER"),
}


class ContentSourceError(RuntimeError):
    """Base failure for the exact MySQL content-source authority."""


class ContentSourceConflict(ContentSourceError):
    """A caller attempted an overwrite or supplied a non-canonical hold set."""


class ContentSourceMissing(ContentSourceError):
    """An exact source revision does not exist."""


class ContentSourceIntegrityError(ContentSourceError):
    """Stored bytes or authority bindings do not match their exact reference."""


class CandidateWriteSurface(StrEnum):
    CHECKPOINT = "APUT"
    PENDING_WRITES = "APUT_WRITES"


class RegisteredServicePrincipal(StrEnum):
    """Closed V1 service principals allowed to produce authoritative content."""

    CHECKPOINT_RUNTIME = "CHECKPOINT_RUNTIME"


@dataclass(frozen=True, slots=True)
class ChatMessageSourceCommand:
    """Producer-free command for the only registered V1 origin adapter."""

    origin_chat_message_id: int
    content_role: ContentRole
    content_schema_version: Literal[1] = 1
    source_revision: Literal[1] = 1

    def __post_init__(self) -> None:
        if (
            type(self.origin_chat_message_id) is not int
            or self.origin_chat_message_id <= 0
            or self.origin_chat_message_id > _MYSQL_SIGNED_BIGINT_MAX
        ):
            raise ContentSourceConflict(
                "CHAT_MESSAGE origin identity must be a strict positive MySQL BIGINT"
            )
        if not isinstance(self.content_role, ContentRole):
            raise ContentSourceConflict("CHAT_MESSAGE content role is invalid")
        if self.content_schema_version != 1 or self.source_revision != 1:
            raise ContentSourceConflict("CHAT_MESSAGE V1 command uses an unknown version")


_SERVICE_CONTENT_ROLES = frozenset(
    {
        ContentRole.EFFECTIVE_QUESTION,
        ContentRole.PLAN_GOAL,
        ContentRole.PLAN_REASON,
        ContentRole.PLAN_MISSING_INFORMATION,
        ContentRole.CURRENT_ISSUE,
        ContentRole.RETRIEVAL_FILE_NAME,
        ContentRole.RETRIEVAL_SNIPPET,
        ContentRole.TOOL_CONTENT,
        ContentRole.DRAFT_ANSWER,
        ContentRole.FINAL_ANSWER,
        ContentRole.ERROR_DETAIL,
        ContentRole.CONVERSATION_SUMMARY,
    }
)
_SERVICE_SOURCE_ID_DOMAIN = "dianshang-agent/checkpoint-service-source/v1"


@dataclass(frozen=True, slots=True)
class ServiceContentSourceCommand:
    """Closed command whose physical identity and producer are server-derived."""

    conversation_id: int
    thread_id: str
    subject_user_id: int
    run_id: str
    attempt_id: str
    fence_version: int
    content_role: ContentRole
    raw_content: str

    def __post_init__(self) -> None:
        for numeric_name, numeric_value in (
            ("conversation_id", self.conversation_id),
            ("subject_user_id", self.subject_user_id),
            ("fence_version", self.fence_version),
        ):
            if (
                type(numeric_value) is not int
                or numeric_value <= 0
                or numeric_value > _MYSQL_SIGNED_BIGINT_MAX
            ):
                raise ContentSourceConflict(
                    f"service source {numeric_name} is invalid"
                )
        ThreadIdentity.from_conversation_id(self.conversation_id).assert_matches(
            self.conversation_id,
            self.thread_id,
        )
        identifiers: tuple[tuple[str, str, int], ...] = (
            ("run_id", self.run_id, 64),
            ("attempt_id", self.attempt_id, 64),
        )
        for identifier_name, identifier_value, maximum in identifiers:
            if (
                not identifier_value
                or len(identifier_value) > maximum
                or re.fullmatch(
                    r"[A-Za-z0-9][A-Za-z0-9._:-]*",
                    identifier_value,
                )
                is None
            ):
                raise ContentSourceConflict(
                    f"service source {identifier_name} is invalid"
                )
        if self.content_role not in _SERVICE_CONTENT_ROLES:
            raise ContentSourceConflict(
                "content role has no registered CHECKPOINT_RUNTIME authority"
            )
        if type(self.raw_content) is not str or not self.raw_content:
            raise ContentSourceConflict("service source content must be non-empty text")
        try:
            self.raw_content.encode("utf-8", errors="strict")
        except UnicodeEncodeError:
            raise ContentSourceConflict(
                "service source content is not strict UTF-8"
            ) from None


def derive_service_source_record_id(
    *,
    conversation_id: int,
    run_id: str,
    content_role: ContentRole,
    content_sha256: str,
) -> str:
    """Derive the immutable physical identity from a versioned domain tuple."""

    preimage = "\x00".join(
        (
            _SERVICE_SOURCE_ID_DOMAIN,
            str(conversation_id),
            run_id,
            content_role.value,
            content_sha256,
        )
    ).encode("ascii")
    return hashlib.sha256(preimage).hexdigest()


def encode_source_record_id(value: int | str) -> str:
    """Encode the strict scalar as canonical JSON while preserving its type."""

    if type(value) is int:
        if value <= 0 or value > _MYSQL_SIGNED_BIGINT_MAX:
            raise ContentSourceConflict("source record integer is outside the MySQL positive BIGINT range")
    elif type(value) is str:
        if not value or len(value) > 128 or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", value) is None:
            raise ContentSourceConflict("source record identifier is invalid")
    else:
        raise ContentSourceConflict("source record identity must be a strict integer or identifier")
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class ContentSourceRevisionWrite:
    reference: ContentSourceReferenceV1
    raw_content: str

    def __post_init__(self) -> None:
        if not isinstance(self.reference, ContentSourceReferenceV1):
            raise ContentSourceConflict("source write requires ContentSourceReferenceV1")
        if type(self.raw_content) is not str:
            raise ContentSourceConflict("source raw content must be strict text")
        if self.reference.content_schema_version != 1:
            raise ContentSourceConflict("source content schema version is not registered")
        try:
            self.raw_content.encode("utf-8", errors="strict")
        except UnicodeEncodeError:
            raise ContentSourceConflict("source raw content is not strict UTF-8") from None
        encode_source_record_id(self.reference.source_record_id)
        actual_digest = compute_content_digest(
            source_kind=self.reference.source_kind,
            content_role=self.reference.content_role,
            content_schema_version=self.reference.content_schema_version,
            normalization_version=self.reference.normalization_version,
            content=self.raw_content,
        )
        if actual_digest != self.reference.content_sha256:
            raise ContentSourceIntegrityError("source content digest does not match the exact reference")


@dataclass(frozen=True, slots=True)
class ContentSourceRevisionRecord:
    reference: ContentSourceReferenceV1
    raw_content: str


@dataclass(frozen=True, slots=True)
class ContentSourceAppendResult:
    record: ContentSourceRevisionRecord
    replayed: bool


@dataclass(frozen=True, slots=True)
class ContentOriginAuthorityRecord:
    """Exact origin row locked and validated inside the current application UoW."""

    reference: ContentSourceReferenceV1
    origin_chat_message_id: int
    raw_content: str


@dataclass(frozen=True, slots=True)
class PublicationContentReference:
    projection_slot: str
    reference: ContentSourceReferenceV1

    def __post_init__(self) -> None:
        if type(self.projection_slot) is not str or _PROJECTION_SLOT_PATTERN.fullmatch(self.projection_slot) is None:
            raise ContentSourceConflict("publication projection slot is invalid")
        if not isinstance(self.reference, ContentSourceReferenceV1):
            raise ContentSourceConflict("publication hold requires ContentSourceReferenceV1")
        if self.reference.content_schema_version != 1:
            raise ContentSourceConflict(
                "publication hold content schema version is not registered"
            )
        require_canonical_projection_slot(
            self.projection_slot,
            self.reference.content_role,
        )


class PublicationContentReferenceSet:
    """Closed immutable publication-hold set in frozen content-role order."""

    __slots__ = ("_items",)
    _items: tuple[PublicationContentReference, ...]

    def __init__(self, items: object = None) -> None:
        del items
        raise TypeError("PublicationContentReferenceSet must be built with of() or empty()")

    @classmethod
    def empty(cls) -> PublicationContentReferenceSet:
        return cls._create(())

    @classmethod
    def of(
        cls,
        *items: PublicationContentReference,
    ) -> PublicationContentReferenceSet:
        return cls._create(tuple(items))

    @classmethod
    def _create(
        cls,
        items: tuple[PublicationContentReference, ...],
    ) -> PublicationContentReferenceSet:
        if len(items) > len(CONTENT_ROLE_ORDER):
            raise ContentSourceConflict("publication content-reference count exceeds the frozen role set")
        slots: set[str] = set()
        roles: set[ContentRole] = set()
        last_rank = -1
        for item in items:
            if not isinstance(item, PublicationContentReference):
                raise ContentSourceConflict("publication references must be typed hold items")
            if item.projection_slot in slots:
                raise ContentSourceConflict("publication projection slot is duplicated")
            role = item.reference.content_role
            if role in roles:
                raise ContentSourceConflict("publication content role is duplicated")
            rank = _ROLE_RANK[role]
            if rank <= last_rank:
                raise ContentSourceConflict("publication content references are not in canonical role order")
            slots.add(item.projection_slot)
            roles.add(role)
            last_rank = rank
        instance = object.__new__(cls)
        object.__setattr__(instance, "_items", items)
        return instance

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("publication content-reference set is immutable")

    def __delattr__(self, name: str) -> None:
        del name
        raise AttributeError("publication content-reference set is immutable")

    @property
    def items(self) -> tuple[PublicationContentReference, ...]:
        return self._items

    def __iter__(self) -> Iterator[PublicationContentReference]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PublicationContentReferenceSet):
            return self._items == other._items
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._items)


class ContentOriginAuthorityPort(Protocol):
    async def lock_chat_message_origin(
        self,
        command: ChatMessageSourceCommand,
    ) -> ContentOriginAuthorityRecord: ...


class ContentSourceRevisionStorePort(Protocol):
    async def append_service_revision(
        self,
        command: ServiceContentSourceCommand,
    ) -> ContentSourceAppendResult: ...

    async def append_revision(
        self,
        write: ContentSourceRevisionWrite,
        *,
        origin: ContentOriginAuthorityRecord,
    ) -> ContentSourceAppendResult: ...

    async def read_exact_revision(
        self,
        reference: ContentSourceReferenceV1,
    ) -> ContentSourceRevisionRecord: ...

    async def require_exact_references(
        self,
        references: PublicationContentReferenceSet,
    ) -> tuple[ContentSourceRevisionRecord, ...]: ...

    async def read_publication_holds(
        self,
        *,
        thread_id: str,
        publication_version: int,
    ) -> PublicationContentReferenceSet: ...


class ContentSourceAuthorityStorePort(
    ContentOriginAuthorityPort,
    ContentSourceRevisionStorePort,
    Protocol,
):
    """Combined transaction-bound store; consumers should prefer the narrow ports."""


class SourceBeforeCandidateRecorderPort(Protocol):
    async def record_checkpoint_candidate(
        self,
        references: PublicationContentReferenceSet,
    ) -> None: ...

    async def record_pending_write_candidate(
        self,
        references: PublicationContentReferenceSet,
    ) -> None: ...


def require_canonical_projection_slot(
    projection_slot: str,
    content_role: ContentRole,
) -> None:
    """Reject any slot outside the projector's closed slot/role vocabulary."""

    if not isinstance(content_role, ContentRole):
        raise ContentSourceConflict("publication content role is invalid")
    try:
        expected_role = canonical_content_role_for_slot(projection_slot)
    except ValueError:
        raise ContentSourceConflict(
            "publication projection slot is not registered"
        ) from None
    if expected_role is not content_role:
        raise ContentSourceConflict("publication projection slot and role disagree")


def require_chat_message_role_purpose(
    content_role: ContentRole,
) -> tuple[str, str]:
    """Return the only registered ChatMessage role/purpose for a content role."""

    contract = _CHAT_MESSAGE_ROLE_PURPOSE.get(content_role)
    if contract is None:
        raise ContentSourceIntegrityError(
            "content role has no registered CHAT_MESSAGE origin authority"
        )
    return contract
