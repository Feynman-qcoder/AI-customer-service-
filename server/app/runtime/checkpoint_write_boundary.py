"""Sealed checkpoint write boundary.

This module has no database, saver, LangGraph or business-service dependency.
It accepts closed projection DTOs and hands one sealed state envelope or one
sealed atomic pending batch to a paired capability-verifying provider.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol, cast

from pydantic import BaseModel, Field, StrictBytes, model_validator

from app.runtime.checkpoint_projection import (
    ClosedModel,
    ContentReferenceSequenceV1,
    ContentSlotReferences,
    ContentSourceReferenceV1,
    Hex64,
    PendingWriteSpecV1,
    PersistedAgentStateV1,
    PersistedPendingWriteV1,
    ProjectionPublicationBindingV1,
    SafeIdentifier,
    SanitizedProjectionError,
    StrictNonNegativeInt,
    StrictPositiveInt,
    ToolMetadataEntryV1,
    ToolProjectionV1,
    TypedToolMetadataValueV1,
    raise_sanitized_projection_error,
)
from app.runtime.data_protection import (
    DataProtectionError,
    DataProtectionLimits,
    DataProtectionPolicy,
    DataProtectionProfile,
    FieldProtection,
    ProtectionMeasurements,
    SchemaPolicyError,
)


class CheckpointStateProjector(Protocol):
    def project(
        self,
        runtime_state: object,
        *,
        references: ContentSlotReferences,
        publication_binding: ProjectionPublicationBindingV1,
        expected_fence_version: int | None = None,
    ) -> PersistedAgentStateV1: ...


class PendingWriteProjector(Protocol):
    def project_pending_write(self, spec: PendingWriteSpecV1) -> PersistedPendingWriteV1: ...


class ContentReferenceBinder(Protocol):
    def bind(self, reference: ContentSourceReferenceV1) -> None: ...


@dataclass(frozen=True, slots=True)
class ProviderCommitResult:
    """Safe whole-operation result; there is no per-item partial result."""

    operation: Literal["PERSIST_STATE", "PERSIST_PENDING_BATCH"]
    envelope_sha256: str
    item_count: int


class _ReasonCountV1(ClosedModel):
    reason: SafeIdentifier
    count: StrictPositiveInt


class _ProtectedMeasurementsV1(ClosedModel):
    total_bytes: StrictNonNegativeInt
    reason_counts: tuple[_ReasonCountV1, ...] = Field(default=(), max_length=32)


class _WriteScopeV1(ClosedModel):
    thread_id: SafeIdentifier
    conversation_id: StrictPositiveInt
    run_id: SafeIdentifier | None = None
    attempt_id: SafeIdentifier | None = None
    fence_version: StrictPositiveInt | None = None
    logical_namespace: SafeIdentifier
    physical_namespace: SafeIdentifier | None = None
    checkpoint_id: SafeIdentifier | None = None
    expected_publication_version: StrictNonNegativeInt | None = None
    expected_previous_pointer: SafeIdentifier | None = None
    domain_sha256: Hex64

    @model_validator(mode="after")
    def _run_attempt_pair(self) -> _WriteScopeV1:
        if (self.run_id is None) != (self.attempt_id is None):
            raise ValueError("run and attempt identities must be present together")
        if self.run_id is None and self.fence_version is not None:
            raise ValueError("fence requires a run and attempt")
        return self


class _PendingItemIdentityV1(ClosedModel):
    task_id: SafeIdentifier
    channel: SafeIdentifier
    write_index: int = Field(strict=True, ge=-(2**31), le=2**31 - 1)
    batch_ordinal: StrictNonNegativeInt
    write_purpose: Literal["CHANNEL_WRITE", "STATE_UPDATE"]
    identity_sha256: Hex64


class _ProtectedPendingItemV1(ClosedModel):
    identity: _PendingItemIdentityV1
    schema_versions: tuple[StrictPositiveInt, ...] = Field(min_length=5, max_length=5)
    canonical_bytes: StrictBytes
    sha256: Hex64
    measurements: _ProtectedMeasurementsV1
    reference_count: StrictNonNegativeInt

    @model_validator(mode="after")
    def _digest_matches(self) -> _ProtectedPendingItemV1:
        if hashlib.sha256(self.canonical_bytes).hexdigest() != self.sha256:
            raise ValueError("pending item digest mismatch")
        if len(self.canonical_bytes) != self.measurements.total_bytes:
            raise ValueError("pending item byte measurement mismatch")
        return self


class _ProtectedStateEnvelope(ClosedModel):
    envelope_schema_version: Literal[1] = 1
    operation_kind: Literal["PERSIST_STATE"] = "PERSIST_STATE"
    envelope_kind: Literal["CHECKPOINT"] = "CHECKPOINT"
    envelope_nonce: SafeIdentifier
    scope: _WriteScopeV1
    profile: Literal["CHECKPOINT_PROJECTION"] = "CHECKPOINT_PROJECTION"
    schema_versions: tuple[StrictPositiveInt, ...] = Field(min_length=4, max_length=4)
    canonical_bytes: StrictBytes
    sha256: Hex64
    measurements: _ProtectedMeasurementsV1
    reference_count: StrictNonNegativeInt

    @model_validator(mode="after")
    def _digest_matches(self) -> _ProtectedStateEnvelope:
        if hashlib.sha256(self.canonical_bytes).hexdigest() != self.sha256:
            raise ValueError("state envelope digest mismatch")
        if len(self.canonical_bytes) != self.measurements.total_bytes:
            raise ValueError("state envelope byte measurement mismatch")
        return self


class _ProtectedPendingBatchEnvelope(ClosedModel):
    envelope_schema_version: Literal[1] = 1
    operation_kind: Literal["PERSIST_PENDING_BATCH"] = "PERSIST_PENDING_BATCH"
    envelope_kind: Literal["PENDING_BATCH"] = "PENDING_BATCH"
    envelope_nonce: SafeIdentifier
    scope: _WriteScopeV1
    profile: Literal["CHECKPOINT_PROJECTION"] = "CHECKPOINT_PROJECTION"
    schema_versions: tuple[StrictPositiveInt, ...] = Field(min_length=5, max_length=5)
    items: tuple[_ProtectedPendingItemV1, ...] = Field(min_length=1, max_length=128)
    batch_count: StrictPositiveInt
    batch_order: tuple[StrictNonNegativeInt, ...] = Field(min_length=1, max_length=128)
    item_digests: tuple[Hex64, ...] = Field(min_length=1, max_length=128)
    batch_sha256: Hex64
    reference_count: StrictNonNegativeInt

    @model_validator(mode="after")
    def _batch_snapshot_is_exact(self) -> _ProtectedPendingBatchEnvelope:
        if self.batch_count != len(self.items):
            raise ValueError("pending batch count mismatch")
        if self.batch_order != tuple(item.identity.batch_ordinal for item in self.items):
            raise ValueError("pending batch order mismatch")
        if self.batch_order != tuple(range(self.batch_count)):
            raise ValueError("pending batch ordinal sequence is not canonical")
        if self.item_digests != tuple(item.sha256 for item in self.items):
            raise ValueError("pending batch item digest list mismatch")
        if any(item.schema_versions != self.schema_versions for item in self.items):
            raise ValueError("pending batch schema versions differ")
        if self.batch_sha256 != _compute_pending_batch_digest(
            scope=self.scope, schema_versions=self.schema_versions, items=self.items
        ):
            raise ValueError("pending batch digest mismatch")
        if self.reference_count != sum(item.reference_count for item in self.items):
            raise ValueError("pending batch reference count mismatch")
        return self


_SealedEnvelope = _ProtectedStateEnvelope | _ProtectedPendingBatchEnvelope


@dataclass(frozen=True, slots=True)
class _CapabilityClaims:
    boundary_id: str
    provider_id: str
    operation_kind: str
    envelope_kind: str
    envelope_nonce: str
    envelope_sha256: str
    batch_count: int
    batch_order: tuple[int, ...]
    item_digests: tuple[str, ...]
    thread_id: str
    conversation_id: int
    run_id: str | None
    attempt_id: str | None
    fence_version: int | None
    logical_namespace: str
    physical_namespace: str | None
    checkpoint_id: str | None
    expected_publication_version: int | None
    expected_previous_pointer: str | None
    domain_sha256: str
    schema_versions: tuple[int, ...]
    profile: str
    one_use_nonce: str


@dataclass(frozen=True, slots=True)
class _WriteInvocationCapability:
    claims: _CapabilityClaims
    signature: str


_CapabilityValidator = Callable[[object, object, str], None]

class InternalPersistenceProvider(ABC):
    """Paired provider exposing only one state and one atomic-batch entry."""

    def __init__(self) -> None:
        self.__validator: _CapabilityValidator | None = None

    def _bind_capability_validator(
        self,
        *,
        boundary_id: str,
        provider_id: str,
        validator: _CapabilityValidator,
    ) -> None:
        del boundary_id, provider_id
        if self.__validator is not None:
            raise_sanitized_projection_error(
                "PROVIDER_ALREADY_BOUND", stage="INTERNAL_PROVIDER"
            )
        self.__validator = validator

    def persist_state(
        self, capability: object, envelope: object
    ) -> ProviderCommitResult:
        validator = self.__validator
        if validator is None:
            raise_sanitized_projection_error(
                "PROVIDER_NOT_BOUND", stage="INTERNAL_PROVIDER"
            )
        validator(capability, envelope, "PERSIST_STATE")
        if type(envelope) is not _ProtectedStateEnvelope:
            raise_sanitized_projection_error(
                "ENVELOPE_KIND_INVALID", stage="INTERNAL_PROVIDER"
            )
        result = self._commit_state(envelope)
        _validate_provider_result(result, envelope)
        return result

    def persist_pending_batch(
        self, capability: object, envelope: object
    ) -> ProviderCommitResult:
        validator = self.__validator
        if validator is None:
            raise_sanitized_projection_error(
                "PROVIDER_NOT_BOUND", stage="INTERNAL_PROVIDER"
            )
        validator(capability, envelope, "PERSIST_PENDING_BATCH")
        if type(envelope) is not _ProtectedPendingBatchEnvelope:
            raise_sanitized_projection_error(
                "ENVELOPE_KIND_INVALID", stage="INTERNAL_PROVIDER"
            )
        result = self._commit_pending_batch(envelope)
        _validate_provider_result(result, envelope)
        return result

    @abstractmethod
    def _commit_state(
        self, envelope: _ProtectedStateEnvelope
    ) -> ProviderCommitResult: ...

    @abstractmethod
    def _commit_pending_batch(
        self, envelope: _ProtectedPendingBatchEnvelope
    ) -> ProviderCommitResult: ...


def _validate_provider_result(
    result: object, envelope: _SealedEnvelope
) -> None:
    if type(result) is not ProviderCommitResult:
        raise_sanitized_projection_error(
            "PROVIDER_RESULT_INVALID", stage="INTERNAL_PROVIDER"
        )
    expected_digest = (
        envelope.sha256
        if isinstance(envelope, _ProtectedStateEnvelope)
        else envelope.batch_sha256
    )
    expected_count = (
        1 if isinstance(envelope, _ProtectedStateEnvelope) else envelope.batch_count
    )
    if (
        result.operation != envelope.operation_kind
        or result.envelope_sha256 != expected_digest
        or result.item_count != expected_count
    ):
        raise_sanitized_projection_error(
            "PROVIDER_RESULT_MISMATCH", stage="INTERNAL_PROVIDER"
        )


class ClosedProjectionSchemaPolicy:
    def normalize(self, payload: object) -> object:
        if type(payload) not in {PersistedAgentStateV1, PersistedPendingWriteV1}:
            raise SchemaPolicyError("CLOSED_DTO_REQUIRED")
        closed = cast(PersistedAgentStateV1 | PersistedPendingWriteV1, payload)
        return closed.model_dump(mode="json")

    def classify(self, path: tuple[str, ...], value: object) -> FieldProtection:
        del path
        if isinstance(value, dict | list | tuple):
            return FieldProtection.STRUCTURE
        return FieldProtection.EXACT

    def validate_protected(self, payload: object) -> None:
        if not isinstance(payload, Mapping):
            raise SchemaPolicyError("CLOSED_DTO_REQUIRED")
        if payload.get("payload_kind") == "PENDING_WRITE":
            values = dict(payload)
            canonical_references = json.dumps(
                values.get("content_references"),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            values["content_references"] = (
                ContentReferenceSequenceV1.decode_canonical(canonical_references)
            )
            PersistedPendingWriteV1.model_validate(values)
            return
        PersistedAgentStateV1.model_validate(payload)


def build_checkpoint_projection_protection(
    *, limits: DataProtectionLimits = DataProtectionLimits()
) -> DataProtectionPolicy:
    return DataProtectionPolicy(
        schema_policy=ClosedProjectionSchemaPolicy(),
        limits=limits,
        profile=DataProtectionProfile.CHECKPOINT_PROJECTION,
    )


def _iter_references(value: object) -> Iterable[ContentSourceReferenceV1]:
    if isinstance(value, ContentSourceReferenceV1):
        yield value
    elif isinstance(value, ContentReferenceSequenceV1):
        yield from value.items
    elif isinstance(value, BaseModel):
        for name in type(value).model_fields:
            yield from _iter_references(getattr(value, name))
    elif isinstance(value, list | tuple):
        for item in value:
            yield from _iter_references(item)


def _iter_tool_metadata(
    value: object,
) -> Iterable[tuple[str, int, int, tuple[ToolMetadataEntryV1, ...]]]:
    if isinstance(value, BaseModel):
        if isinstance(value, ToolProjectionV1):
            yield (
                value.tool_name,
                value.tool_policy_version,
                value.metadata_schema_version,
                value.safe_metadata,
            )
            return
        if isinstance(value, TypedToolMetadataValueV1):
            yield (
                value.tool_name,
                value.tool_policy_version,
                value.metadata_schema_version,
                value.entries,
            )
            return
        for name in type(value).model_fields:
            yield from _iter_tool_metadata(getattr(value, name))
    elif isinstance(value, list | tuple):
        for item in value:
            yield from _iter_tool_metadata(item)


@dataclass(frozen=True, slots=True)
class _BoundaryFailure:
    reason: str
    stage: str
    detail: str = ""

class _CheckpointWriteCore:
    """Internal paired write core; never retained on the public facade."""

    def __init__(
        self,
        *,
        binder: ContentReferenceBinder,
        protection: DataProtectionPolicy,
        provider: InternalPersistenceProvider,
        metadata_catalog: Any,
    ) -> None:
        if not isinstance(provider, InternalPersistenceProvider):
            raise TypeError("provider must derive from InternalPersistenceProvider")
        boundary_id = secrets.token_hex(16)
        provider_id = secrets.token_hex(16)
        secret = secrets.token_bytes(32)
        consumed: set[str] = set()
        consume_lock = threading.Lock()

        def _mint(envelope: _SealedEnvelope) -> _WriteInvocationCapability:
            claims = _claims_for_envelope(
                envelope,
                boundary_id=boundary_id,
                provider_id=provider_id,
                nonce=secrets.token_hex(16),
            )
            signature = hmac.new(
                secret, _canonical_claims(claims), hashlib.sha256
            ).hexdigest()
            return _WriteInvocationCapability(claims=claims, signature=signature)

        def _verify(
            capability: object, envelope: object, operation_kind: str
        ) -> None:
            if type(capability) is not _WriteInvocationCapability:
                raise_sanitized_projection_error(
                    "CAPABILITY_INVALID", stage="INTERNAL_PROVIDER"
                )
            if type(envelope) not in {
                _ProtectedStateEnvelope,
                _ProtectedPendingBatchEnvelope,
            }:
                raise_sanitized_projection_error(
                    "ENVELOPE_INVALID", stage="INTERNAL_PROVIDER"
                )
            sealed_envelope = cast(_SealedEnvelope, envelope)
            expected = _claims_for_envelope(
                sealed_envelope,
                boundary_id=boundary_id,
                provider_id=provider_id,
                nonce=capability.claims.one_use_nonce,
            )
            expected_signature = hmac.new(
                secret, _canonical_claims(expected), hashlib.sha256
            ).hexdigest()
            if (
                capability.claims != expected
                or operation_kind != expected.operation_kind
                or not hmac.compare_digest(capability.signature, expected_signature)
            ):
                raise_sanitized_projection_error(
                    "CAPABILITY_BINDING_MISMATCH", stage="INTERNAL_PROVIDER"
                )
            with consume_lock:
                if capability.claims.one_use_nonce in consumed:
                    raise_sanitized_projection_error(
                        "CAPABILITY_REUSED", stage="INTERNAL_PROVIDER"
                    )
                consumed.add(capability.claims.one_use_nonce)

        provider._bind_capability_validator(
            boundary_id=boundary_id,
            provider_id=provider_id,
            validator=_verify,
        )
        self._binder = binder
        self._protection = protection
        self._provider = provider
        self._metadata_catalog = metadata_catalog
        self._mint_capability = _mint

    def write_state(self, projection: PersistedAgentStateV1) -> ProviderCommitResult:
        outcome = self._write_state_safely(projection)
        if isinstance(outcome, ProviderCommitResult):
            return outcome
        reason, stage, detail = outcome.reason, outcome.stage, outcome.detail
        del outcome, projection, self
        raise_sanitized_projection_error(reason, stage=stage, detail=detail)

    def write_pending_batch(
        self, batch: Sequence[PersistedPendingWriteV1]
    ) -> ProviderCommitResult:
        outcome = self._write_pending_safely(batch)
        if isinstance(outcome, ProviderCommitResult):
            return outcome
        reason, stage, detail = outcome.reason, outcome.stage, outcome.detail
        del outcome, batch, self
        raise_sanitized_projection_error(reason, stage=stage, detail=detail)

    def _write_state_safely(
        self, projection: object
    ) -> ProviderCommitResult | _BoundaryFailure:
        try:
            envelope = self._prepare_state(projection)
            return self._provider.persist_state(
                self._mint_capability(envelope), envelope
            )
        except SanitizedProjectionError as error:
            return _BoundaryFailure(error.reason, error.stage, error.detail)
        except DataProtectionError as error:
            return _BoundaryFailure(
                f"DATA_PROTECTION_{error.reason}", "DATA_PROTECTION"
            )
        except Exception:
            return _BoundaryFailure("STATE_WRITE_FAILED", "WRITE_BOUNDARY")

    def _write_pending_safely(
        self, batch: object
    ) -> ProviderCommitResult | _BoundaryFailure:
        try:
            envelope = self._prepare_pending_batch(batch)
            return self._provider.persist_pending_batch(
                self._mint_capability(envelope), envelope
            )
        except SanitizedProjectionError as error:
            return _BoundaryFailure(error.reason, error.stage, error.detail)
        except DataProtectionError as error:
            return _BoundaryFailure(
                f"DATA_PROTECTION_{error.reason}", "DATA_PROTECTION"
            )
        except Exception:
            return _BoundaryFailure("PENDING_BATCH_WRITE_FAILED", "WRITE_BOUNDARY")

    def _prepare_state(self, projection: object) -> _ProtectedStateEnvelope:
        if type(projection) is not PersistedAgentStateV1:
            raise_sanitized_projection_error(
                "PAYLOAD_KIND_INVALID", stage="WRITE_BOUNDARY"
            )
        references = tuple(_iter_references(projection))
        for reference in references:
            self._binder.bind(reference)
        self._validate_tool_metadata(projection)
        self._validate_tool_names(projection)
        protected = self._protection.protect(
            projection, profile=DataProtectionProfile.CHECKPOINT_PROJECTION
        )
        return _ProtectedStateEnvelope(
            envelope_nonce=secrets.token_hex(16),
            scope=_state_scope(projection),
            schema_versions=(
                projection.projection_schema_version,
                projection.runtime_state_schema_version,
                projection.protection_schema_version,
                projection.policy_schema_version,
            ),
            canonical_bytes=protected.canonical_bytes,
            sha256=protected.sha256,
            measurements=_measurements(protected.measurements),
            reference_count=len(references),
        )

    def _prepare_pending_batch(
        self, batch: object
    ) -> _ProtectedPendingBatchEnvelope:
        if type(batch) not in {list, tuple}:
            raise_sanitized_projection_error(
                "BATCH_NOT_A_SEQUENCE", stage="WRITE_BOUNDARY"
            )
        raw_batch = cast(list[object] | tuple[object, ...], batch)
        if not raw_batch:
            raise_sanitized_projection_error("BATCH_EMPTY", stage="WRITE_BOUNDARY")
        if len(raw_batch) > 128:
            raise_sanitized_projection_error(
                "BATCH_TOO_LARGE", stage="WRITE_BOUNDARY"
            )
        if any(type(item) is not PersistedPendingWriteV1 for item in raw_batch):
            raise_sanitized_projection_error(
                "PAYLOAD_KIND_INVALID", stage="WRITE_BOUNDARY"
            )
        roots = cast(tuple[PersistedPendingWriteV1, ...], tuple(raw_batch))
        first = roots[0]
        scope = _pending_scope(first)
        schema_versions = _pending_schema_versions(first)
        items: list[_ProtectedPendingItemV1] = []
        for position, root in enumerate(roots):
            if _pending_scope(root) != scope:
                raise_sanitized_projection_error(
                    "PENDING_BATCH_SCOPE_MISMATCH", stage="WRITE_BOUNDARY"
                )
            if _pending_schema_versions(root) != schema_versions:
                raise_sanitized_projection_error(
                    "PENDING_BATCH_VERSION_MISMATCH", stage="WRITE_BOUNDARY"
                )
            if root.batch_ordinal != position:
                raise_sanitized_projection_error(
                    "PENDING_BATCH_ORDER_INVALID", stage="WRITE_BOUNDARY"
                )
            references = tuple(_iter_references(root))
            for reference in references:
                self._binder.bind(reference)
            self._validate_tool_metadata(root)
            protected = self._protection.protect(
                root, profile=DataProtectionProfile.CHECKPOINT_PROJECTION
            )
            identity_payload = {
                "task_id": root.task_id,
                "channel": root.channel,
                "write_index": root.write_index,
                "batch_ordinal": root.batch_ordinal,
                "write_purpose": root.write_purpose.value,
            }
            items.append(
                _ProtectedPendingItemV1(
                    identity=_PendingItemIdentityV1(
                        task_id=root.task_id,
                        channel=root.channel,
                        write_index=root.write_index,
                        batch_ordinal=root.batch_ordinal,
                        write_purpose=root.write_purpose.value,
                        identity_sha256=_sha256_json(identity_payload),
                    ),
                    schema_versions=schema_versions,
                    canonical_bytes=protected.canonical_bytes,
                    sha256=protected.sha256,
                    measurements=_measurements(protected.measurements),
                    reference_count=len(references),
                )
            )
        item_tuple = tuple(items)
        return _ProtectedPendingBatchEnvelope(
            envelope_nonce=secrets.token_hex(16),
            scope=scope,
            schema_versions=schema_versions,
            items=item_tuple,
            batch_count=len(item_tuple),
            batch_order=tuple(item.identity.batch_ordinal for item in item_tuple),
            item_digests=tuple(item.sha256 for item in item_tuple),
            batch_sha256=_compute_pending_batch_digest(
                scope=scope, schema_versions=schema_versions, items=item_tuple
            ),
            reference_count=sum(item.reference_count for item in item_tuple),
        )

    def _validate_tool_metadata(self, payload: object) -> None:
        for tool_name, policy_version, schema_version, entries in _iter_tool_metadata(
            payload
        ):
            metadata: dict[str, Any] = {}
            for entry in entries:
                if entry.key in metadata:
                    raise_sanitized_projection_error(
                        "TOOL_METADATA_KEY_DUPLICATED", stage="TOOL_METADATA"
                    )
                metadata[entry.key] = entry.value
            self._metadata_catalog.validate(
                tool_name,
                metadata,
                tool_policy_version=policy_version,
                metadata_schema_version=schema_version,
            )

    def _validate_tool_names(self, payload: PersistedAgentStateV1) -> None:
        run = payload.active_run
        if run is None:
            return
        names = set(run.selected_tools)
        if run.plan is not None:
            names.update(run.plan.required_tools)
        names.update(tool.tool_name for tool in run.tool_results)
        if names - self._metadata_catalog.registered_tools():
            raise_sanitized_projection_error(
                "TOOL_NOT_REGISTERED", stage="WRITE_BOUNDARY"
            )


class CheckpointWriteBoundary:
    """Capability-safe public facade composed around an inaccessible core.

    The returned per-composition subclass has no instance dictionary or slots.
    Its two formal methods close over the paired internal core; no provider,
    signer, issuer, mint function, envelope factory, or capability is retained
    on the facade.  Recovering the closure would require Python runtime
    reflection, which is explicitly outside the V1 ordinary-caller threat
    model.
    """

    __slots__ = ()

    def __new__(
        cls,
        *,
        binder: ContentReferenceBinder,
        protection: DataProtectionPolicy,
        provider: InternalPersistenceProvider,
        metadata_catalog: Any,
    ) -> CheckpointWriteBoundary:
        if cls is not CheckpointWriteBoundary:
            raise TypeError("checkpoint write boundary cannot be subclassed")
        core = _CheckpointWriteCore(
            binder=binder,
            protection=protection,
            provider=provider,
            metadata_catalog=metadata_catalog,
        )

        class _BoundCheckpointWriteBoundary(CheckpointWriteBoundary):
            __slots__ = ()

            def write_state(
                self, projection: PersistedAgentStateV1
            ) -> ProviderCommitResult:
                del self
                return core.write_state(projection)

            def write_pending_batch(
                self, batch: Sequence[PersistedPendingWriteV1]
            ) -> ProviderCommitResult:
                del self
                return core.write_pending_batch(batch)

        return object.__new__(_BoundCheckpointWriteBoundary)

    def __init__(
        self,
        *,
        binder: ContentReferenceBinder,
        protection: DataProtectionPolicy,
        provider: InternalPersistenceProvider,
        metadata_catalog: Any,
    ) -> None:
        # Pairing happened atomically in __new__; never retain constructor
        # inputs or an internal callable on this facade instance.
        del self, binder, protection, provider, metadata_catalog

    def write_state(self, projection: PersistedAgentStateV1) -> ProviderCommitResult:
        del self, projection
        raise_sanitized_projection_error(
            "BOUNDARY_NOT_COMPOSED", stage="WRITE_BOUNDARY"
        )

    def write_pending_batch(
        self, batch: Sequence[PersistedPendingWriteV1]
    ) -> ProviderCommitResult:
        del self, batch
        raise_sanitized_projection_error(
            "BOUNDARY_NOT_COMPOSED", stage="WRITE_BOUNDARY"
        )


def _measurements(value: ProtectionMeasurements) -> _ProtectedMeasurementsV1:
    return _ProtectedMeasurementsV1(
        total_bytes=value.total_bytes,
        reason_counts=tuple(
            _ReasonCountV1(reason=reason, count=count)
            for reason, count in value.reason_counts
        ),
    )


def _sha256_json(value: object) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _state_scope(payload: PersistedAgentStateV1) -> _WriteScopeV1:
    execution = payload.execution
    binding = payload.publication_binding
    domain = {
        "thread_id": payload.identity.thread_id,
        "conversation_id": payload.identity.conversation_id,
        "run_id": execution.run_id if execution is not None else None,
        "attempt_id": execution.attempt_id if execution is not None else None,
        "fence_version": (
            execution.expected_fence_version if execution is not None else None
        ),
        "logical_namespace": binding.logical_namespace,
        "physical_namespace": binding.physical_namespace,
        "checkpoint_id": binding.checkpoint_id,
        "expected_publication_version": binding.expected_publication_version,
        "expected_previous_pointer": binding.expected_previous_pointer,
    }
    return _WriteScopeV1(
        thread_id=payload.identity.thread_id,
        conversation_id=payload.identity.conversation_id,
        run_id=execution.run_id if execution is not None else None,
        attempt_id=execution.attempt_id if execution is not None else None,
        fence_version=(
            execution.expected_fence_version if execution is not None else None
        ),
        logical_namespace=binding.logical_namespace,
        physical_namespace=binding.physical_namespace,
        checkpoint_id=binding.checkpoint_id,
        expected_publication_version=binding.expected_publication_version,
        expected_previous_pointer=binding.expected_previous_pointer,
        domain_sha256=_sha256_json(domain),
    )


def _pending_scope(payload: PersistedPendingWriteV1) -> _WriteScopeV1:
    source = payload.domain
    domain = {
        "thread_id": source.thread_id,
        "conversation_id": source.conversation_id,
        "run_id": source.run_id,
        "attempt_id": source.attempt_id,
        "fence_version": source.fence_version,
        "logical_namespace": source.logical_namespace,
        "physical_namespace": source.physical_namespace,
        "checkpoint_id": source.checkpoint_id,
        "expected_publication_version": source.expected_publication_version,
        "expected_previous_pointer": source.expected_previous_pointer,
    }
    return _WriteScopeV1(
        thread_id=source.thread_id,
        conversation_id=source.conversation_id,
        run_id=source.run_id,
        attempt_id=source.attempt_id,
        fence_version=source.fence_version,
        logical_namespace=source.logical_namespace,
        physical_namespace=source.physical_namespace,
        checkpoint_id=source.checkpoint_id,
        expected_publication_version=source.expected_publication_version,
        expected_previous_pointer=source.expected_previous_pointer,
        domain_sha256=_sha256_json(domain),
    )


def _pending_schema_versions(payload: PersistedPendingWriteV1) -> tuple[int, ...]:
    return (
        payload.pending_write_schema_version,
        payload.projection_schema_version,
        payload.runtime_state_schema_version,
        payload.protection_schema_version,
        payload.policy_schema_version,
    )


def _compute_pending_batch_digest(
    *,
    scope: _WriteScopeV1,
    schema_versions: tuple[int, ...],
    items: tuple[_ProtectedPendingItemV1, ...],
) -> str:
    descriptor = {
        "domain_tag": "dianshang-agent/protected-pending-batch/v1",
        "scope": scope.model_dump(mode="json"),
        "schema_versions": list(schema_versions),
        "items": [
            {
                "identity": item.identity.model_dump(mode="json"),
                "sha256": item.sha256,
            }
            for item in items
        ],
    }
    return _sha256_json(descriptor)


def _claims_for_envelope(
    envelope: _SealedEnvelope,
    *,
    boundary_id: str,
    provider_id: str,
    nonce: str,
) -> _CapabilityClaims:
    batch_order: tuple[int, ...]
    item_digests: tuple[str, ...]
    if isinstance(envelope, _ProtectedStateEnvelope):
        digest = envelope.sha256
        batch_count = 1
        batch_order = (0,)
        item_digests = (envelope.sha256,)
    else:
        digest = envelope.batch_sha256
        batch_count = envelope.batch_count
        batch_order = envelope.batch_order
        item_digests = envelope.item_digests
    scope = envelope.scope
    return _CapabilityClaims(
        boundary_id=boundary_id,
        provider_id=provider_id,
        operation_kind=envelope.operation_kind,
        envelope_kind=envelope.envelope_kind,
        envelope_nonce=envelope.envelope_nonce,
        envelope_sha256=digest,
        batch_count=batch_count,
        batch_order=batch_order,
        item_digests=item_digests,
        thread_id=scope.thread_id,
        conversation_id=scope.conversation_id,
        run_id=scope.run_id,
        attempt_id=scope.attempt_id,
        fence_version=scope.fence_version,
        logical_namespace=scope.logical_namespace,
        physical_namespace=scope.physical_namespace,
        checkpoint_id=scope.checkpoint_id,
        expected_publication_version=scope.expected_publication_version,
        expected_previous_pointer=scope.expected_previous_pointer,
        domain_sha256=scope.domain_sha256,
        schema_versions=envelope.schema_versions,
        profile=envelope.profile,
        one_use_nonce=nonce,
    )


def _canonical_claims(claims: _CapabilityClaims) -> bytes:
    return json.dumps(
        asdict(claims),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


__all__ = [
    "CheckpointStateProjector",
    "PendingWriteProjector",
    "ContentReferenceBinder",
    "InternalPersistenceProvider",
    "ProviderCommitResult",
    "CheckpointWriteBoundary",
    "build_checkpoint_projection_protection",
    "SanitizedProjectionError",
]
