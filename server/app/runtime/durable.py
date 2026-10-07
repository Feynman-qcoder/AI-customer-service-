from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Protocol, cast

from app.runtime.content_source import PublicationContentReferenceSet

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MYSQL_SIGNED_INT_MIN = -(2**31)
_MYSQL_SIGNED_INT_MAX = 2**31 - 1


class DurableRuntimeError(RuntimeError):
    """Base error for durable authority and replay-safe effect contracts."""


class DurableContentionError(DurableRuntimeError):
    """A bounded MySQL contention retry was exhausted at a transaction boundary."""

    retriable = True

    def __init__(self, *, operation: str, attempt_count: int, mysql_error_code: int) -> None:
        self.operation = operation
        self.attempt_count = attempt_count
        self.mysql_error_code = mysql_error_code
        super().__init__(
            "durable MySQL contention exhausted "
            f"for {operation} after {attempt_count} attempts (error {mysql_error_code})"
        )


class DurableContractError(DurableRuntimeError):
    """Stable input violates the durable contract before persistence."""


class PublicationScopeMismatch(DurableContractError):
    """Publication scope, candidate, or manifest identity crosses domains."""


class LeaseConflict(DurableRuntimeError):
    """A lease operation does not own the current thread/fence."""


class PublicationConflict(DurableRuntimeError):
    """A monotonic publication compare-and-set was rejected."""


class EffectConflict(DurableRuntimeError):
    """The same effect identity was replayed with different content."""


class EffectCorruption(EffectConflict):
    """A committed effect is missing or disagrees with its exact target record."""


class EffectType(StrEnum):
    CHAT_MESSAGE = "CHAT_MESSAGE"
    ACTION_PREPARE = "ACTION_PREPARE"
    LOCAL_AUDIT = "LOCAL_AUDIT"


class MessagePurpose(StrEnum):
    USER_INPUT = "USER_INPUT"
    CONFIRMATION_CHALLENGE = "CONFIRMATION_CHALLENGE"
    WAITING_NOTICE = "WAITING_NOTICE"
    FINAL_ANSWER = "FINAL_ANSWER"
    BLOCKED_RESPONSE = "BLOCKED_RESPONSE"


@dataclass(frozen=True, slots=True)
class RunRegistration:
    run_id: str
    thread_id: str
    conversation_id: int
    subject_user_id: int
    request_id: str
    started_at: datetime

    def __post_init__(self) -> None:
        _require_identifier(self.run_id, "run_id", 64)
        _require_identifier(self.thread_id, "thread_id", 128)
        _require_positive_int(self.conversation_id, "conversation_id")
        _require_positive_int(self.subject_user_id, "subject_user_id")
        _require_identifier(self.request_id, "request_id", 64)
        if self.started_at.tzinfo is None:
            raise DurableContractError("run started_at must include a timezone")


@dataclass(frozen=True, slots=True)
class RunRecord:
    run_id: str
    thread_id: str
    conversation_id: int
    subject_user_id: int
    status: str
    request_id: str


@dataclass(frozen=True, slots=True)
class AttemptRegistration:
    attempt_id: str
    run_id: str
    thread_id: str
    conversation_id: int
    actor_user_id: int | None
    actor_role: str
    subject_user_id: int
    started_at: datetime

    def __post_init__(self) -> None:
        _require_identifier(self.attempt_id, "attempt_id", 64)
        _require_identifier(self.run_id, "run_id", 64)
        _require_identifier(self.thread_id, "thread_id", 128)
        _require_positive_int(self.conversation_id, "conversation_id")
        if self.actor_user_id is not None:
            _require_positive_int(self.actor_user_id, "actor_user_id")
        _require_identifier(self.actor_role, "actor_role", 32)
        _require_positive_int(self.subject_user_id, "subject_user_id")
        if self.started_at.tzinfo is None:
            raise DurableContractError("attempt started_at must include a timezone")


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    attempt_id: str
    run_id: str
    thread_id: str
    conversation_id: int
    status: str
    fence_version: int | None


@dataclass(frozen=True, slots=True)
class LeaseGrant:
    thread_id: str
    owner_attempt_id: str
    fence_version: int
    lease_expires_at: datetime


@dataclass(frozen=True, slots=True)
class LeaseState:
    thread_id: str
    owner_attempt_id: str | None
    fence_version: int
    lease_expires_at: datetime | None


@dataclass(frozen=True, slots=True)
class PublicationScope:
    thread_id: str
    attempt_id: str
    fence_version: int

    def __post_init__(self) -> None:
        _require_identifier(self.thread_id, "thread_id", 128)
        _require_identifier(self.attempt_id, "attempt_id", 64)
        _require_positive_int(self.fence_version, "fence_version")


@dataclass(frozen=True, slots=True)
class CandidateHandle:
    thread_id: str
    attempt_id: str
    fence_version: int
    logical_namespace: str
    physical_namespace: str
    checkpoint_id: str

    def __post_init__(self) -> None:
        _require_identifier(self.thread_id, "candidate.thread_id", 128)
        _require_identifier(self.attempt_id, "candidate.attempt_id", 64)
        _require_positive_int(self.fence_version, "candidate.fence_version")
        _require_identifier(
            self.logical_namespace,
            "candidate.logical_namespace",
            128,
            allow_empty=True,
        )
        _require_identifier(self.physical_namespace, "candidate.physical_namespace", 255)
        _require_identifier(self.checkpoint_id, "candidate.checkpoint_id", 128)


@dataclass(frozen=True, slots=True)
class CheckpointPointer:
    logical_namespace: str
    physical_namespace: str
    checkpoint_id: str

    def __post_init__(self) -> None:
        _require_identifier(self.logical_namespace, "pointer.logical_namespace", 128, allow_empty=True)
        _require_identifier(self.physical_namespace, "pointer.physical_namespace", 255)
        _require_identifier(self.checkpoint_id, "pointer.checkpoint_id", 128)


@dataclass(frozen=True, slots=True)
class WriteManifestItem:
    thread_id: str
    logical_namespace: str
    physical_namespace: str
    checkpoint_id: str
    task_id: str
    write_index: int
    channel: str
    content_digest: str

    def __post_init__(self) -> None:
        _require_identifier(self.thread_id, "manifest.thread_id", 128)
        _require_identifier(
            self.logical_namespace,
            "manifest.logical_namespace",
            128,
            allow_empty=True,
        )
        _require_identifier(self.physical_namespace, "manifest.physical_namespace", 255)
        _require_identifier(self.checkpoint_id, "manifest.checkpoint_id", 128)
        _require_identifier(self.task_id, "manifest.task_id", 128)
        _require_mysql_signed_int(self.write_index, "manifest.write_index")
        _require_identifier(self.channel, "manifest.channel", 128)
        _require_sha256(self.content_digest, "manifest.content_digest")


@dataclass(frozen=True, slots=True)
class CandidateSnapshot:
    handle: CandidateHandle
    checkpoint_digest: str
    manifest_root: str
    manifest_count: int

    def __post_init__(self) -> None:
        _require_sha256(self.checkpoint_digest, "checkpoint_digest")
        _require_sha256(self.manifest_root, "manifest_root")
        _require_nonnegative_int(self.manifest_count, "manifest_count")


@dataclass(frozen=True, slots=True)
class CanonicalPublication:
    scope: PublicationScope
    snapshot: CandidateSnapshot
    manifest: tuple[WriteManifestItem, ...]
    expected_publication_version: int
    expected_previous_pointer: CheckpointPointer | None
    content_references: PublicationContentReferenceSet = field(
        default_factory=PublicationContentReferenceSet.empty
    )

    def __post_init__(self) -> None:
        _require_nonnegative_int(
            self.expected_publication_version,
            "expected_publication_version",
        )
        if not isinstance(self.content_references, PublicationContentReferenceSet):
            raise DurableContractError(
                "publication content references require the closed typed hold set"
            )
        canonical_manifest = tuple(
            sorted(
                self.manifest,
                key=lambda item: (item.task_id, item.write_index, item.channel),
            )
        )
        object.__setattr__(self, "manifest", canonical_manifest)
        validate_publication_domain(
            self.scope,
            self.snapshot,
            canonical_manifest,
            expected_publication_version=self.expected_publication_version,
            expected_previous_pointer=self.expected_previous_pointer,
        )

    @property
    def next_publication_version(self) -> int:
        return self.expected_publication_version + 1

    @property
    def pointer(self) -> CheckpointPointer:
        handle = self.snapshot.handle
        return CheckpointPointer(
            logical_namespace=handle.logical_namespace,
            physical_namespace=handle.physical_namespace,
            checkpoint_id=handle.checkpoint_id,
        )

    def assert_valid(self) -> None:
        if not isinstance(self.content_references, PublicationContentReferenceSet):
            raise DurableContractError(
                "publication content references require the closed typed hold set"
            )
        validate_publication_domain(
            self.scope,
            self.snapshot,
            self.manifest,
            expected_publication_version=self.expected_publication_version,
            expected_previous_pointer=self.expected_previous_pointer,
        )


@dataclass(frozen=True, slots=True)
class PublicationRecord:
    thread_id: str
    publication_version: int
    pointer: CheckpointPointer | None
    checkpoint_digest: str | None
    previous_pointer: CheckpointPointer | None
    manifest_root: str | None
    manifest_count: int
    manifest: tuple[WriteManifestItem, ...]


@dataclass(frozen=True, slots=True)
class CanonicalEffectClaim:
    run_id: str
    node_name: str
    purpose: str
    sequence: int
    effect_type: EffectType
    idempotency_key: str
    payload_digest: str

    def __post_init__(self) -> None:
        _require_identifier(self.run_id, "effect.run_id", 64)
        _require_identifier(self.node_name, "effect.node_name", 64)
        _require_identifier(self.purpose, "effect.purpose", 64)
        _require_nonnegative_int(self.sequence, "effect.sequence")
        _require_sha256(self.idempotency_key, "effect.idempotency_key")
        _require_sha256(self.payload_digest, "effect.payload_digest")


@dataclass(frozen=True, slots=True)
class EffectWriteScope:
    thread_id: str
    attempt_id: str
    fence_version: int

    def __post_init__(self) -> None:
        _require_identifier(self.thread_id, "effect_scope.thread_id", 128)
        _require_identifier(self.attempt_id, "effect_scope.attempt_id", 64)
        _require_positive_int(self.fence_version, "effect_scope.fence_version")


@dataclass(frozen=True, slots=True)
class MessageEffectWrite:
    scope: EffectWriteScope
    claim: CanonicalEffectClaim
    conversation_id: int
    role: str
    content: str
    sources_json: str | None = None
    retrieval_score: Decimal | None = None
    confidence_level: str | None = None
    need_human: bool = False

    def __post_init__(self) -> None:
        _require_positive_int(self.conversation_id, "message.conversation_id")
        _require_identifier(self.role, "message.role", 32)
        if not isinstance(self.content, str):
            raise DurableContractError("message.content must be a string")
        if type(self.need_human) is not bool:
            raise DurableContractError("message.need_human must be a boolean")


@dataclass(frozen=True, slots=True)
class AuditEffectWrite:
    scope: EffectWriteScope
    claim: CanonicalEffectClaim
    input_summary: str | None
    output_summary: str | None
    status: str
    duration_ms: int = 0
    error_summary: str | None = None


_ACTION_PREPARE_REPLAY_CONTRACT_KIND = "ACTION_PREPARE_REPLAY"
_ACTION_PREPARE_REPLAY_CONTRACT_VERSION = 1
R2_STATELESS_COMPAT_CONFIRMATION_MODE = "R2_STATELESS_COMPAT"
DURABLE_INTERRUPT_CONFIRMATION_MODE = "DURABLE_INTERRUPT"
_DURABLE_ACTION_PREPARE_REPLAY_CONTRACT_KIND = "DURABLE_ACTION_PREPARE_REPLAY"
_DURABLE_ACTION_PREPARE_REPLAY_CONTRACT_VERSION = 1


@dataclass(frozen=True, slots=True)
class _ActionPrepareReplayContractV1:
    """Private, versioned digest preimage for exact ACTION_PREPARE replay."""

    action_type: str
    target_order_id: int
    target_order_no: str
    subject_user_id: int
    created_by: int
    action_payload_json: str
    risk_level: str
    validated_order_status: str
    policy_version: str

    def __post_init__(self) -> None:
        _require_identifier(self.action_type, "replay.action_type", 64)
        _require_positive_int(self.target_order_id, "replay.target_order_id")
        _require_identifier(self.target_order_no, "replay.target_order_no", 64)
        _require_positive_int(self.subject_user_id, "replay.subject_user_id")
        _require_positive_int(self.created_by, "replay.created_by")
        _decode_canonical_action_payload(self.action_payload_json)
        _require_identifier(self.risk_level, "replay.risk_level", 32)
        _require_identifier(
            self.validated_order_status,
            "replay.validated_order_status",
            32,
        )
        _require_identifier(self.policy_version, "replay.policy_version", 64)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "action_payload": _decode_canonical_action_payload(
                self.action_payload_json
            ),
            "action_type": self.action_type,
            "contract_kind": _ACTION_PREPARE_REPLAY_CONTRACT_KIND,
            "contract_version": _ACTION_PREPARE_REPLAY_CONTRACT_VERSION,
            "created_by": self.created_by,
            "policy_version": self.policy_version,
            "risk_level": self.risk_level,
            "subject_user_id": self.subject_user_id,
            "target_order_id": self.target_order_id,
            "target_order_no": self.target_order_no,
            "validated_order_status": self.validated_order_status,
        }

    def digest(self) -> str:
        return canonical_digest(self.canonical_payload())


@dataclass(frozen=True, slots=True)
class _DurableActionPrepareReplayContractV1:
    """Stable replay preimage for one durable logical customer action."""

    logical_action_id: str
    action_type: str
    target_order_id: int
    target_order_no: str
    subject_user_id: int
    created_by: int
    action_payload_json: str
    risk_level: str
    validated_order_status: str
    policy_version: str
    draft_revision: int
    draft_expires_at: str
    customer_confirmation_challenge_digest: str

    def __post_init__(self) -> None:
        _require_identifier(self.logical_action_id, "durable_replay.logical_action_id", 64)
        _require_identifier(self.action_type, "durable_replay.action_type", 64)
        _require_positive_int(self.target_order_id, "durable_replay.target_order_id")
        _require_identifier(self.target_order_no, "durable_replay.target_order_no", 64)
        _require_positive_int(self.subject_user_id, "durable_replay.subject_user_id")
        _require_positive_int(self.created_by, "durable_replay.created_by")
        _decode_canonical_action_payload(self.action_payload_json)
        _require_identifier(self.risk_level, "durable_replay.risk_level", 32)
        _require_identifier(
            self.validated_order_status,
            "durable_replay.validated_order_status",
            32,
        )
        _require_identifier(self.policy_version, "durable_replay.policy_version", 64)
        _require_positive_int(self.draft_revision, "durable_replay.draft_revision")
        _canonical_rfc3339(self.draft_expires_at, "durable_replay.draft_expires_at")
        _require_sha256(
            self.customer_confirmation_challenge_digest,
            "durable_replay.customer_confirmation_challenge_digest",
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "action_payload": _decode_canonical_action_payload(self.action_payload_json),
            "action_type": self.action_type,
            "confirmation_mode": DURABLE_INTERRUPT_CONFIRMATION_MODE,
            "contract_kind": _DURABLE_ACTION_PREPARE_REPLAY_CONTRACT_KIND,
            "contract_version": _DURABLE_ACTION_PREPARE_REPLAY_CONTRACT_VERSION,
            "created_by": self.created_by,
            "customer_confirmation_challenge_digest": (
                self.customer_confirmation_challenge_digest
            ),
            "draft_expires_at": _canonical_rfc3339(
                self.draft_expires_at,
                "durable_replay.draft_expires_at",
            ),
            "draft_revision": self.draft_revision,
            "logical_action_id": self.logical_action_id,
            "policy_version": self.policy_version,
            "risk_level": self.risk_level,
            "subject_user_id": self.subject_user_id,
            "target_order_id": self.target_order_id,
            "target_order_no": self.target_order_no,
            "validated_order_status": self.validated_order_status,
        }

    def digest(self) -> str:
        return canonical_digest(self.canonical_payload())


def action_prepare_replay_digest(
    *,
    action_type: str,
    target_order_id: int,
    target_order_no: str,
    subject_user_id: int,
    created_by: int,
    action_payload_json: str,
    risk_level: str,
    validated_order_status: str,
    policy_version: str,
) -> str:
    """Return the only accepted V1 ACTION_PREPARE replay-contract digest."""

    return _ActionPrepareReplayContractV1(
        action_type=action_type,
        target_order_id=target_order_id,
        target_order_no=target_order_no,
        subject_user_id=subject_user_id,
        created_by=created_by,
        action_payload_json=action_payload_json,
        risk_level=risk_level,
        validated_order_status=validated_order_status,
        policy_version=policy_version,
    ).digest()


def durable_action_prepare_replay_digest(
    *,
    logical_action_id: str,
    action_type: str,
    target_order_id: int,
    target_order_no: str,
    subject_user_id: int,
    created_by: int,
    action_payload_json: str,
    risk_level: str,
    validated_order_status: str,
    policy_version: str,
    draft_revision: int,
    draft_expires_at: str,
    customer_confirmation_challenge_digest: str,
) -> str:
    """Return the stable V1 durable logical-action replay digest."""

    return _DurableActionPrepareReplayContractV1(
        logical_action_id=logical_action_id,
        action_type=action_type,
        target_order_id=target_order_id,
        target_order_no=target_order_no,
        subject_user_id=subject_user_id,
        created_by=created_by,
        action_payload_json=action_payload_json,
        risk_level=risk_level,
        validated_order_status=validated_order_status,
        policy_version=policy_version,
        draft_revision=draft_revision,
        draft_expires_at=draft_expires_at,
        customer_confirmation_challenge_digest=(
            customer_confirmation_challenge_digest
        ),
    ).digest()


def canonical_action_payload_json(payload: object) -> str:
    """Serialize one closed action payload for persistence and digest binding."""

    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise DurableContractError("action payload must be stable JSON") from exc
    _decode_canonical_action_payload(encoded)
    return encoded


@dataclass(frozen=True, slots=True)
class ActionPrepareEffectWrite:
    scope: EffectWriteScope
    claim: CanonicalEffectClaim
    action_type: str
    target_order_id: int
    action_payload_json: str
    risk_level: str
    created_by: int
    subject_user_id: int
    logical_action_id: str
    target_order_no: str
    policy_version: str
    validated_order_status: str
    confirmation_mode: str = R2_STATELESS_COMPAT_CONFIRMATION_MODE
    customer_confirmation_challenge_digest: str | None = None
    draft_revision: int | None = None
    draft_expires_at: str | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.action_type, "action.action_type", 64)
        _require_positive_int(self.target_order_id, "action.target_order_id")
        if not isinstance(self.action_payload_json, str):
            raise DurableContractError("action payload must be canonical JSON text")
        _require_identifier(self.risk_level, "action.risk_level", 32)
        _require_positive_int(self.created_by, "action.created_by")
        _require_positive_int(self.subject_user_id, "action.subject_user_id")
        _require_identifier(self.logical_action_id, "action.logical_action_id", 128)
        _require_identifier(self.target_order_no, "action.target_order_no", 64)
        _require_identifier(self.policy_version, "action.policy_version", 64)
        _require_identifier(
            self.validated_order_status,
            "action.validated_order_status",
            32,
        )
        if self.confirmation_mode == DURABLE_INTERRUPT_CONFIRMATION_MODE:
            if (
                self.customer_confirmation_challenge_digest is None
                or self.draft_revision is None
                or self.draft_expires_at is None
            ):
                raise DurableContractError(
                    "durable action prepare requires complete customer confirmation evidence"
                )
            _require_sha256(
                self.customer_confirmation_challenge_digest,
                "action.customer_confirmation_challenge_digest",
            )
            _require_positive_int(self.draft_revision, "action.draft_revision")
            _canonical_rfc3339(self.draft_expires_at, "action.draft_expires_at")
        elif self.confirmation_mode != R2_STATELESS_COMPAT_CONFIRMATION_MODE:
            raise DurableContractError("action confirmation mode is unsupported")
        elif any(
            value is not None
            for value in (
                self.customer_confirmation_challenge_digest,
                self.draft_revision,
                self.draft_expires_at,
            )
        ):
            raise DurableContractError(
                "R2 stateless action prepare cannot carry durable confirmation evidence"
            )
        self.replay_digest()

    def replay_digest(self) -> str:
        if self.confirmation_mode == DURABLE_INTERRUPT_CONFIRMATION_MODE:
            assert self.customer_confirmation_challenge_digest is not None
            assert self.draft_revision is not None
            assert self.draft_expires_at is not None
            return durable_action_prepare_replay_digest(
                logical_action_id=self.logical_action_id,
                action_type=self.action_type,
                target_order_id=self.target_order_id,
                target_order_no=self.target_order_no,
                subject_user_id=self.subject_user_id,
                created_by=self.created_by,
                action_payload_json=self.action_payload_json,
                risk_level=self.risk_level,
                validated_order_status=self.validated_order_status,
                policy_version=self.policy_version,
                draft_revision=self.draft_revision,
                draft_expires_at=self.draft_expires_at,
                customer_confirmation_challenge_digest=(
                    self.customer_confirmation_challenge_digest
                ),
            )
        return action_prepare_replay_digest(
            action_type=self.action_type,
            target_order_id=self.target_order_id,
            target_order_no=self.target_order_no,
            subject_user_id=self.subject_user_id,
            created_by=self.created_by,
            action_payload_json=self.action_payload_json,
            risk_level=self.risk_level,
            validated_order_status=self.validated_order_status,
            policy_version=self.policy_version,
        )


@dataclass(frozen=True, slots=True)
class DurableActionPrepareReplayLookup:
    """Stable durable lookup whose historical order status comes from MySQL."""

    scope: EffectWriteScope
    conversation_id: int
    run_id: str
    node_name: str
    purpose: str
    sequence: int
    logical_action_id: str
    action_type: str
    target_order_id: int
    target_order_no: str
    subject_user_id: int
    created_by: int
    action_payload_json: str
    risk_level: str
    policy_version: str
    draft_revision: int
    draft_expires_at: str
    customer_confirmation_challenge_digest: str

    def __post_init__(self) -> None:
        _require_positive_int(self.conversation_id, "durable_lookup.conversation_id")
        _require_identifier(self.run_id, "durable_lookup.run_id", 64)
        _require_identifier(self.node_name, "durable_lookup.node_name", 64)
        _require_identifier(self.purpose, "durable_lookup.purpose", 64)
        _require_nonnegative_int(self.sequence, "durable_lookup.sequence")
        _require_identifier(
            self.logical_action_id,
            "durable_lookup.logical_action_id",
            64,
        )
        _require_identifier(self.action_type, "durable_lookup.action_type", 64)
        _require_positive_int(self.target_order_id, "durable_lookup.target_order_id")
        _require_identifier(
            self.target_order_no,
            "durable_lookup.target_order_no",
            64,
        )
        _require_positive_int(self.subject_user_id, "durable_lookup.subject_user_id")
        _require_positive_int(self.created_by, "durable_lookup.created_by")
        _decode_canonical_action_payload(self.action_payload_json)
        _require_identifier(self.risk_level, "durable_lookup.risk_level", 32)
        _require_identifier(self.policy_version, "durable_lookup.policy_version", 64)
        _require_positive_int(self.draft_revision, "durable_lookup.draft_revision")
        _canonical_rfc3339(self.draft_expires_at, "durable_lookup.draft_expires_at")
        _require_sha256(
            self.customer_confirmation_challenge_digest,
            "durable_lookup.customer_confirmation_challenge_digest",
        )

    @property
    def effect_idempotency_key(self) -> str:
        return canonical_effect_idempotency_key(
            run_id=self.run_id,
            node_name=self.node_name,
            purpose=self.purpose,
            sequence=self.sequence,
        )

    def write_with_prepared_order_status(
        self,
        prepared_order_status: str,
    ) -> ActionPrepareEffectWrite:
        digest = durable_action_prepare_replay_digest(
            logical_action_id=self.logical_action_id,
            action_type=self.action_type,
            target_order_id=self.target_order_id,
            target_order_no=self.target_order_no,
            subject_user_id=self.subject_user_id,
            created_by=self.created_by,
            action_payload_json=self.action_payload_json,
            risk_level=self.risk_level,
            validated_order_status=prepared_order_status,
            policy_version=self.policy_version,
            draft_revision=self.draft_revision,
            draft_expires_at=self.draft_expires_at,
            customer_confirmation_challenge_digest=(
                self.customer_confirmation_challenge_digest
            ),
        )
        return ActionPrepareEffectWrite(
            scope=self.scope,
            claim=CanonicalEffectClaim(
                run_id=self.run_id,
                node_name=self.node_name,
                purpose=self.purpose,
                sequence=self.sequence,
                effect_type=EffectType.ACTION_PREPARE,
                idempotency_key=self.effect_idempotency_key,
                payload_digest=digest,
            ),
            action_type=self.action_type,
            target_order_id=self.target_order_id,
            action_payload_json=self.action_payload_json,
            risk_level=self.risk_level,
            created_by=self.created_by,
            subject_user_id=self.subject_user_id,
            logical_action_id=self.logical_action_id,
            target_order_no=self.target_order_no,
            policy_version=self.policy_version,
            validated_order_status=prepared_order_status,
            confirmation_mode=DURABLE_INTERRUPT_CONFIRMATION_MODE,
            customer_confirmation_challenge_digest=(
                self.customer_confirmation_challenge_digest
            ),
            draft_revision=self.draft_revision,
            draft_expires_at=self.draft_expires_at,
        )


@dataclass(frozen=True, slots=True)
class PreparedActionValidationRequest:
    lookup: DurableActionPrepareReplayLookup
    pending_action_id: int

    def __post_init__(self) -> None:
        _require_positive_int(
            self.pending_action_id,
            "prepared_validation.pending_action_id",
        )


@dataclass(frozen=True, slots=True)
class ActionPrepareProof:
    authorization_id: str
    thread_id: str
    run_id: str
    attempt_id: str
    fence_version: int
    actor_user_id: int
    subject_user_id: int
    logical_action_id: str
    action_type: str
    target_order_id: int
    target_order_no: str
    payload_digest: str
    policy_version: str
    expires_at: datetime
    proof_digest: str

    def __post_init__(self) -> None:
        _require_identifier(self.authorization_id, "proof.authorization_id", 128)
        _require_identifier(self.thread_id, "proof.thread_id", 128)
        _require_identifier(self.run_id, "proof.run_id", 64)
        _require_identifier(self.attempt_id, "proof.attempt_id", 64)
        _require_positive_int(self.fence_version, "proof.fence_version")
        _require_positive_int(self.actor_user_id, "proof.actor_user_id")
        _require_positive_int(self.subject_user_id, "proof.subject_user_id")
        _require_identifier(self.logical_action_id, "proof.logical_action_id", 128)
        _require_identifier(self.action_type, "proof.action_type", 64)
        _require_positive_int(self.target_order_id, "proof.target_order_id")
        _require_identifier(self.target_order_no, "proof.target_order_no", 64)
        _require_sha256(self.payload_digest, "proof.payload_digest")
        _require_identifier(self.policy_version, "proof.policy_version", 64)
        if self.expires_at.tzinfo is None:
            raise DurableContractError("proof.expires_at must include a timezone")
        _require_sha256(self.proof_digest, "proof.proof_digest")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "action_type": self.action_type,
            "actor_user_id": self.actor_user_id,
            "attempt_id": self.attempt_id,
            "authorization_id": self.authorization_id,
            "expires_at": self.expires_at.astimezone(UTC).isoformat(),
            "fence_version": self.fence_version,
            "logical_action_id": self.logical_action_id,
            "payload_digest": self.payload_digest,
            "policy_version": self.policy_version,
            "run_id": self.run_id,
            "subject_user_id": self.subject_user_id,
            "target_order_id": self.target_order_id,
            "target_order_no": self.target_order_no,
            "thread_id": self.thread_id,
        }

    def assert_digest(self) -> None:
        if canonical_digest(self.canonical_payload()) != self.proof_digest:
            raise DurableContractError("action authorization proof digest is invalid")


@dataclass(frozen=True, slots=True)
class EffectWriteResult:
    effect_id: int
    target_id: int
    idempotency_key: str
    replayed: bool


R2_DAY_COMPATIBILITY_PREFIX = "r2compat"


@dataclass(frozen=True, slots=True)
class R2DayCompatibilityKey:
    """Day-scoped cross-run idempotency compatibility key.

    Distinct HTTP runs — a response-loss retry or two concurrent new runs —
    converge on the first committed ACTION_PREPARE ActionRequest instead of
    creating duplicates. This key is day-scoped and is not a resume/replay key.

    Contract:

    - the key is generated only by the server from the four facts below plus
      the authoritative MySQL database day; request, model, or tool payloads
      can never supply or influence it;
    - it deliberately excludes attempt, fence, and run components;
    - it is stored in the existing unique ``agent_action_request.idempotency_key``
      column.
    """

    action_type: str
    target_order_id: int
    subject_user_id: int
    day: str

    def __post_init__(self) -> None:
        _require_identifier(self.action_type, "compatibility.action_type", 64)
        _require_positive_int(self.target_order_id, "compatibility.target_order_id")
        _require_positive_int(self.subject_user_id, "compatibility.subject_user_id")
        if not isinstance(self.day, str) or datetime.strptime(self.day, "%Y-%m-%d") is None:
            raise DurableContractError("compatibility day must be an ISO YYYY-MM-DD date")
        if datetime.strptime(self.day, "%Y-%m-%d").strftime("%Y-%m-%d") != self.day:
            raise DurableContractError("compatibility day must be an ISO YYYY-MM-DD date")

    @property
    def value(self) -> str:
        return ":".join(
            (
                R2_DAY_COMPATIBILITY_PREFIX,
                self.action_type,
                str(self.target_order_id),
                str(self.subject_user_id),
                self.day,
            )
        )


def build_r2_day_compatibility_key(
    *,
    action_type: str,
    target_order_id: int,
    subject_user_id: int,
    day: str,
) -> R2DayCompatibilityKey:
    """Generate a day-scoped compatibility key from server-side facts."""
    return R2DayCompatibilityKey(
        action_type=action_type,
        target_order_id=target_order_id,
        subject_user_id=subject_user_id,
        day=day,
    )


def parse_r2_day_compatibility_key(stored: str) -> R2DayCompatibilityKey | None:
    """Parse a stored key, returning ``None`` for any non-conforming value."""
    if not isinstance(stored, str):
        return None
    parts = stored.split(":")
    if len(parts) != 5 or parts[0] != R2_DAY_COMPATIBILITY_PREFIX:
        return None
    try:
        return R2DayCompatibilityKey(
            action_type=parts[1],
            target_order_id=int(parts[2]),
            subject_user_id=int(parts[3]),
            day=parts[4],
        )
    except (ValueError, DurableContractError):
        return None


class LogicalRunStorePort(Protocol):
    async def begin_run(self, registration: RunRegistration) -> RunRecord: ...


class AttemptLeaseStorePort(Protocol):
    async def register_attempt(self, registration: AttemptRegistration) -> AttemptRecord: ...

    async def acquire_lease(
        self,
        *,
        thread_id: str,
        attempt_id: str,
        lease_milliseconds: int,
    ) -> LeaseGrant: ...

    async def renew_lease(
        self,
        *,
        thread_id: str,
        attempt_id: str,
        fence_version: int,
        lease_milliseconds: int,
    ) -> LeaseGrant: ...

    async def release_lease(
        self,
        *,
        thread_id: str,
        attempt_id: str,
        fence_version: int,
    ) -> LeaseState: ...

    async def read_lease(self, thread_id: str) -> LeaseState | None: ...

    async def require_live_lease(
        self,
        *,
        thread_id: str,
        run_id: str,
        attempt_id: str,
        fence_version: int,
    ) -> LeaseGrant: ...


class PublicationStorePort(Protocol):
    async def publish(self, publication: CanonicalPublication) -> PublicationRecord: ...

    async def read_publication(self, thread_id: str) -> PublicationRecord | None: ...


class DurableAuthorityStorePort(
    LogicalRunStorePort,
    AttemptLeaseStorePort,
    PublicationStorePort,
    Protocol,
):
    """Concrete-adapter wiring surface for the two narrow authority consumers."""


class MessageEffectStorePort(Protocol):
    async def write_message_effect(self, write: MessageEffectWrite) -> EffectWriteResult: ...


class AuditEffectStorePort(Protocol):
    async def write_audit_effect(self, write: AuditEffectWrite) -> EffectWriteResult: ...


class ActionPrepareStorePort(Protocol):
    async def lock_action_prepare_effect(
        self,
        write: ActionPrepareEffectWrite,
    ) -> EffectWriteResult | None: ...

    async def create_action_prepare_effect(
        self,
        write: ActionPrepareEffectWrite,
        proof: ActionPrepareProof,
    ) -> EffectWriteResult: ...


class DurableActionPrepareReplayStorePort(Protocol):
    async def lock_durable_action_prepare_replay(
        self,
        lookup: DurableActionPrepareReplayLookup,
    ) -> EffectWriteResult | None: ...


class PreparedActionValidationStorePort(Protocol):
    async def validate_prepared_action(
        self,
        request: PreparedActionValidationRequest,
    ) -> None: ...


class ReplaySafeEffectStorePort(
    MessageEffectStorePort,
    AuditEffectStorePort,
    ActionPrepareStorePort,
    Protocol,
):
    """Concrete-adapter wiring surface for three narrow effect consumers."""


def manifest_root(items: tuple[WriteManifestItem, ...]) -> str:
    canonical = [
        {
            "channel": item.channel,
            "checkpoint_id": item.checkpoint_id,
            "content_digest": item.content_digest,
            "logical_namespace": item.logical_namespace,
            "physical_namespace": item.physical_namespace,
            "task_id": item.task_id,
            "thread_id": item.thread_id,
            "write_index": item.write_index,
        }
        for item in sorted(items, key=lambda value: (value.task_id, value.write_index, value.channel))
    ]
    encoded = json.dumps(canonical, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_publication_domain(
    scope: PublicationScope,
    snapshot: CandidateSnapshot,
    manifest: tuple[WriteManifestItem, ...],
    *,
    expected_publication_version: int,
    expected_previous_pointer: CheckpointPointer | None,
) -> None:
    handle = snapshot.handle
    scope_domain = (scope.thread_id, scope.attempt_id, scope.fence_version)
    handle_domain = (handle.thread_id, handle.attempt_id, handle.fence_version)
    if scope_domain != handle_domain:
        raise PublicationScopeMismatch("publication scope does not match candidate handle")
    if expected_publication_version == 0 and expected_previous_pointer is not None:
        raise DurableContractError("initial publication cannot declare a previous pointer")
    if expected_publication_version > 0 and expected_previous_pointer is None:
        raise DurableContractError("non-initial publication requires the exact previous pointer")

    item_keys: set[tuple[str, int]] = set()
    for item in manifest:
        item_domain = (
            item.thread_id,
            item.logical_namespace,
            item.physical_namespace,
            item.checkpoint_id,
        )
        snapshot_domain = (
            handle.thread_id,
            handle.logical_namespace,
            handle.physical_namespace,
            handle.checkpoint_id,
        )
        if item_domain != snapshot_domain:
            raise PublicationScopeMismatch("manifest item does not match candidate snapshot")
        item_key = (item.task_id, item.write_index)
        if item_key in item_keys:
            raise DurableContractError("manifest contains a duplicate task/write index")
        item_keys.add(item_key)

    if snapshot.manifest_count != len(manifest):
        raise DurableContractError("candidate manifest count does not match exact manifest")
    if snapshot.manifest_root != manifest_root(manifest):
        raise DurableContractError("candidate manifest root does not match exact manifest")


def canonical_digest(payload: object) -> str:
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError) as exc:
        raise DurableContractError("effect payload must be stable JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def canonical_effect_idempotency_key(
    *,
    run_id: str,
    node_name: str,
    purpose: str,
    sequence: int,
) -> str:
    """Canonical key algorithm shared by effect creation and replay checks."""

    return canonical_digest(
        {
            "node_name": node_name,
            "purpose": purpose,
            "run_id": run_id,
            "sequence": sequence,
        }
    )


def _decode_canonical_action_payload(payload_json: str) -> dict[str, object]:
    if not isinstance(payload_json, str):
        raise DurableContractError("action payload must be canonical JSON text")
    try:
        decoded = json.loads(payload_json)
        canonical = json.dumps(
            decoded,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise DurableContractError("action payload must be stable JSON") from exc
    if not isinstance(decoded, dict) or canonical != payload_json:
        raise DurableContractError("action payload must be a canonical JSON object")
    return cast(dict[str, object], decoded)


def _require_identifier(
    value: str,
    field_name: str,
    max_length: int,
    *,
    allow_empty: bool = False,
) -> None:
    if not isinstance(value, str):
        raise DurableContractError(f"{field_name} must be a string")
    if (not allow_empty and not value) or len(value) > max_length:
        raise DurableContractError(f"{field_name} is empty or exceeds its storage bound")


def _require_positive_int(value: int, field_name: str) -> None:
    if type(value) is not int or value <= 0:
        raise DurableContractError(f"{field_name} must be a positive integer")


def _require_nonnegative_int(value: int, field_name: str) -> None:
    if type(value) is not int or value < 0:
        raise DurableContractError(f"{field_name} must be a non-negative integer")


def _require_mysql_signed_int(value: int, field_name: str) -> None:
    if (
        type(value) is not int
        or value < _MYSQL_SIGNED_INT_MIN
        or value > _MYSQL_SIGNED_INT_MAX
    ):
        raise DurableContractError(f"{field_name} must fit a MySQL signed INT")


def _require_sha256(value: str, field_name: str) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise DurableContractError(f"{field_name} must be a lowercase SHA-256 digest")


def _canonical_rfc3339(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise DurableContractError(f"{field_name} must be a non-empty RFC3339 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DurableContractError(f"{field_name} must be an RFC3339 timestamp") from exc
    if parsed.tzinfo is None:
        raise DurableContractError(f"{field_name} must include a timezone")
    return parsed.astimezone(UTC).isoformat()
