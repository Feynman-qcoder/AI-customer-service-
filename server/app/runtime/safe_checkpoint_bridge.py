"""The sole protected LangGraph ↔ managed AsyncSqliteSaver bridge for V1."""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, Literal, Protocol, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.constants import INTERRUPT, RESUME
from langgraph.types import Interrupt

from app.agent.checkpoint_projection import (
    AgentCheckpointStateProjector,
    AgentPendingWriteProjector,
)
from app.agent.state import ConversationCheckpointState, load_checkpoint_state
from app.agent.thread_identity import derive_checkpoint_namespace
from app.runtime.checkpoint_projection import (
    ContentReferenceSequenceV1,
    ContentSlotReferences,
    ContentSourceReferenceV1,
    NullValueV1,
    PendingWriteDomainV1,
    PendingWritePurposeV1,
    PendingWriteSpecV1,
    PendingWriteValueV1,
    PersistedAgentStateV1,
    ProjectionPublicationBindingV1,
    StateProjectionValueV1,
    StructuralControlValueV1,
)
from app.runtime.checkpoint_rehydration import (
    CheckpointRehydrationError,
    parse_persisted_pending_write,
    parse_persisted_state,
    rehydrate_agent_state,
)
from app.runtime.checkpoint_write_boundary import build_checkpoint_projection_protection
from app.runtime.content_source import ContentSourceRevisionRecord
from app.runtime.context import ExecutionScope, RuntimeContextProvider
from app.runtime.data_protection import DataProtectionPort, DataProtectionProfile
from app.runtime.durable import LeaseGrant
from app.runtime.uow import require_no_active_transaction

_STATE_CHANNELS = (
    "schema_version",
    "conversation_identity",
    "memory",
    "active_run",
)
_PROTECTED_STATE_CHANNEL = "__protected_agent_state_v1__"
_PROTECTED_SHAPE_CHANNEL = "__protected_agent_shape_v1__"
_SHAPE_START = "START"
_SHAPE_STATE = "STATE"
_LOGICAL_NAMESPACE = "root"


class CheckpointContentSourcePort(Protocol):
    async def materialize_checkpoint_state(
        self,
        state: ConversationCheckpointState,
        execution: ExecutionScope,
    ) -> ContentSlotReferences: ...

    async def read_exact_revision(
        self,
        reference: ContentSourceReferenceV1,
    ) -> ContentSourceRevisionRecord: ...


class CheckpointLeaseAuthorityPort(Protocol):
    async def require_live_lease(
        self,
        execution: ExecutionScope,
    ) -> LeaseGrant: ...


class AsyncCheckpointSaverPort(Protocol):
    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None: ...

    def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]: ...

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig: ...

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None: ...


class ProtectedCheckpointBridge(BaseCheckpointSaver[str]):
    """Protect every saver surface and rehydrate only exact MySQL content."""

    def __init__(
        self,
        *,
        saver: AsyncCheckpointSaverPort,
        runtime_contexts: RuntimeContextProvider,
        content_sources: CheckpointContentSourcePort,
        durable_authority: CheckpointLeaseAuthorityPort,
        state_projector: AgentCheckpointStateProjector,
        pending_projector: AgentPendingWriteProjector,
        data_protection: DataProtectionPort | None = None,
    ) -> None:
        super().__init__()
        self._saver = saver
        self._runtime_contexts = runtime_contexts
        self._content_sources = content_sources
        self._durable_authority = durable_authority
        self._state_projector = state_projector
        self._pending_projector = pending_projector
        self._data_protection = (
            data_protection or build_checkpoint_projection_protection()
        )

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        execution = self._execution_for_config(config)
        server_config = _server_checkpoint_config(config, execution)
        await self._require_live_write_authority(execution)
        persisted = await self._saver.aget_tuple(server_config)
        if persisted is None:
            return None
        return await self._rehydrate_tuple(persisted, execution=execution)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        if config is None:
            raise CheckpointRehydrationError(
                "checkpoint listing requires a server execution scope"
            )
        execution = self._execution_for_config(config)
        server_config = _server_checkpoint_config(config, execution)
        await self._require_live_write_authority(execution)
        async for persisted in self._saver.alist(
            server_config,
            filter=filter,
            before=before,
            limit=limit,
        ):
            yield await self._rehydrate_tuple(persisted, execution=execution)

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        execution = self._execution_for_config(config)
        server_config = _server_checkpoint_config(config, execution)
        state, shape, controls = _state_from_checkpoint(checkpoint)
        projection = await self._project_state(state, execution=execution)
        protected = self._data_protection.protect(
            projection,
            profile=DataProtectionProfile.CHECKPOINT_PROJECTION,
        )
        await self._require_live_write_authority(execution)
        protected_checkpoint = cast(Checkpoint, dict(checkpoint))
        protected_checkpoint["channel_values"] = {
            _PROTECTED_STATE_CHANNEL: protected.canonical_bytes,
            _PROTECTED_SHAPE_CHANNEL: shape,
            **controls,
        }
        execution.require_attempt_active()
        return await self._saver.aput(
            server_config,
            protected_checkpoint,
            _safe_metadata(metadata),
            new_versions,
        )

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        execution = self._execution_for_config(config)
        server_config = _server_checkpoint_config(config, execution)
        classified = [
            _classify_pending_write(
                channel=channel,
                value=value,
                task_id=task_id,
                task_path=task_path,
                ordinal=ordinal,
            )
            for ordinal, (channel, value) in enumerate(writes)
        ]
        has_state = any(kind == "STATE" for kind, _value in classified)
        projection: PersistedAgentStateV1 | None = None
        if has_state:
            state = _state_from_pending_writes(writes)
            projection = await self._project_state(state, execution=execution)
        domain = PendingWriteDomainV1(
            thread_id=execution.thread_id,
            conversation_id=execution.conversation_id,
            run_id=execution.run_id,
            attempt_id=execution.attempt_id,
            fence_version=_required_fence(execution),
            logical_namespace=_LOGICAL_NAMESPACE,
        )
        protected_writes: list[tuple[str, Any]] = []
        for ordinal, ((channel, _raw_value), (kind, classified_value)) in enumerate(
            zip(writes, classified, strict=True)
        ):
            if kind == "STATE":
                if projection is None:
                    raise AssertionError("pending state projection was not materialized")
                pending_value: PendingWriteValueV1 = StateProjectionValueV1(
                    value_kind="STATE_PROJECTION",
                    state=projection,
                )
                purpose = PendingWritePurposeV1.STATE_UPDATE
            elif kind == "NULL":
                pending_value = NullValueV1(value_kind="NULL")
                purpose = PendingWritePurposeV1.CHANNEL_WRITE
            elif kind == "STRUCTURAL":
                if not isinstance(classified_value, StructuralControlValueV1):
                    raise AssertionError("structural pending write was not classified")
                pending_value = classified_value
                purpose = PendingWritePurposeV1.CHANNEL_WRITE
            else:
                raise AssertionError("unknown classified pending-write kind")
            spec = PendingWriteSpecV1(
                domain=domain,
                task_id=task_id,
                channel=channel,
                write_index=WRITES_IDX_MAP.get(channel, ordinal),
                batch_ordinal=ordinal,
                write_purpose=purpose,
                value=pending_value,
                content_references=ContentReferenceSequenceV1.of(),
            )
            projected = self._pending_projector.project_pending_write(spec)
            protected = self._data_protection.protect(
                projected,
                profile=DataProtectionProfile.CHECKPOINT_PROJECTION,
            )
            protected_writes.append((channel, protected.canonical_bytes))
        await self._require_live_write_authority(execution)
        execution.require_attempt_active()
        await self._saver.aput_writes(
            server_config,
            protected_writes,
            task_id,
            task_path,
        )

    async def adelete_thread(self, thread_id: str) -> None:
        del thread_id
        raise CheckpointRehydrationError(
            "checkpoint deletion is unavailable before the protected retention phase"
        )

    async def _project_state(
        self,
        state: ConversationCheckpointState,
        *,
        execution: ExecutionScope,
    ) -> PersistedAgentStateV1:
        execution.require_attempt_active()
        references = await self._content_sources.materialize_checkpoint_state(
            state,
            execution,
        )
        execution.require_attempt_active()
        # The materializer owns and closes its MySQL transaction before any
        # projection/protection or SQLite await is allowed.
        require_no_active_transaction("checkpoint projection")
        return self._state_projector.project(
            state,
            references=references,
            publication_binding=ProjectionPublicationBindingV1(
                logical_namespace=_LOGICAL_NAMESPACE,
            ),
            expected_fence_version=_required_fence(execution),
        )

    async def _require_live_write_authority(
        self,
        execution: ExecutionScope,
    ) -> LeaseGrant:
        execution.require_attempt_active()
        require_no_active_transaction("checkpoint live-authority precheck")
        grant = await self._durable_authority.require_live_lease(execution)
        execution.require_attempt_active()
        require_no_active_transaction("checkpoint physical write")
        return grant

    async def _rehydrate_tuple(
        self,
        persisted: CheckpointTuple,
        *,
        execution: ExecutionScope,
    ) -> CheckpointTuple:
        checkpoint = cast(Checkpoint, dict(persisted.checkpoint))
        values = checkpoint.get("channel_values")
        if not isinstance(values, Mapping):
            raise CheckpointRehydrationError("checkpoint channel values are invalid")
        encoded = values.get(_PROTECTED_STATE_CHANNEL)
        shape = values.get(_PROTECTED_SHAPE_CHANNEL)
        if type(encoded) is not bytes or shape not in {_SHAPE_START, _SHAPE_STATE}:
            raise CheckpointRehydrationError("checkpoint protected state is missing")
        projection = parse_persisted_state(encoded)
        state = await rehydrate_agent_state(
            projection,
            reader=self._content_sources,
            execution=execution,
        )
        controls = {
            key: value
            for key, value in values.items()
            if key not in {_PROTECTED_STATE_CHANNEL, _PROTECTED_SHAPE_CHANNEL}
        }
        if shape == _SHAPE_START:
            channel_values: dict[str, Any] = {
                "__start__": state.to_agent_state(),
                **controls,
            }
        else:
            channel_values = {**state.to_agent_state(), **controls}
        checkpoint["channel_values"] = channel_values
        pending_writes = None
        if persisted.pending_writes is not None:
            pending_writes = []
            for task_id, channel, raw_value in persisted.pending_writes:
                if type(raw_value) is not bytes:
                    raise CheckpointRehydrationError(
                        "checkpoint pending write is not protected"
                    )
                pending = parse_persisted_pending_write(raw_value)
                if pending.task_id != task_id or pending.channel != channel:
                    raise CheckpointRehydrationError(
                        "checkpoint pending-write identity mismatch"
                    )
                _validate_pending_domain(pending.domain, execution)
                value = pending.value
                if isinstance(value, StateProjectionValueV1):
                    recovered = await rehydrate_agent_state(
                        value.state,
                        reader=self._content_sources,
                        execution=execution,
                    )
                    recovered_values = cast(dict[str, Any], recovered.to_agent_state())
                    if channel not in recovered_values:
                        raise CheckpointRehydrationError(
                            "checkpoint pending state channel is invalid"
                        )
                    restored = recovered_values[channel]
                elif isinstance(value, NullValueV1):
                    restored = None
                elif isinstance(value, StructuralControlValueV1):
                    restored = _rehydrate_structural_control(value)
                else:
                    raise CheckpointRehydrationError(
                        "checkpoint pending-write value is unsupported"
                    )
                pending_writes.append((task_id, channel, restored))
        return CheckpointTuple(
            config=persisted.config,
            checkpoint=checkpoint,
            metadata=persisted.metadata,
            parent_config=persisted.parent_config,
            pending_writes=pending_writes,
        )

    def _execution_for_config(self, config: RunnableConfig) -> ExecutionScope:
        execution = self._runtime_contexts.current().execution
        execution.require_attempt_active()
        configurable = config.get("configurable", {})
        if not isinstance(configurable, Mapping):
            raise CheckpointRehydrationError("checkpoint config is invalid")
        if configurable.get("thread_id") != execution.thread_id:
            raise CheckpointRehydrationError(
                "checkpoint thread is not the server execution thread"
            )
        if "run_id" in configurable or "attempt_id" in configurable:
            raise CheckpointRehydrationError(
                "checkpoint client execution override is forbidden"
            )
        expected_namespace = derive_checkpoint_namespace(execution.run_id)
        supplied_namespace = configurable.get("checkpoint_ns")
        if supplied_namespace not in (None, "", expected_namespace):
            raise CheckpointRehydrationError(
                "checkpoint namespace is not the server run namespace"
            )
        metadata = config.get("metadata")
        if isinstance(metadata, Mapping):
            if "run_id" in metadata or "attempt_id" in metadata:
                raise CheckpointRehydrationError(
                    "checkpoint metadata execution override is forbidden"
                )
            metadata_namespace = metadata.get("checkpoint_ns")
            if metadata_namespace not in (None, expected_namespace):
                raise CheckpointRehydrationError(
                    "checkpoint metadata namespace is not the server run namespace"
                )
        return execution


def _server_checkpoint_config(
    config: RunnableConfig,
    execution: ExecutionScope,
) -> RunnableConfig:
    configurable = config.get("configurable", {})
    if not isinstance(configurable, Mapping):
        raise CheckpointRehydrationError("checkpoint config is invalid")
    return cast(
        RunnableConfig,
        {
            **config,
            "configurable": {
                **configurable,
                "thread_id": execution.thread_id,
                "checkpoint_ns": derive_checkpoint_namespace(execution.run_id),
            },
        },
    )


def _classify_pending_write(
    *,
    channel: str,
    value: Any,
    task_id: str,
    task_path: str,
    ordinal: int,
) -> tuple[str, StructuralControlValueV1 | None]:
    if channel in _STATE_CHANNELS:
        return "STATE", None
    if channel == INTERRUPT:
        if type(value) is not tuple or not 0 < len(value) <= 16:
            raise CheckpointRehydrationError(
                "checkpoint interrupt shape is not registered"
            )
        namespaces: list[tuple[str, ...]] = []
        for raw_interrupt in value:
            if (
                not isinstance(raw_interrupt, Interrupt)
                or raw_interrupt.resumable is not True
                or raw_interrupt.when != "during"
                or isinstance(raw_interrupt.ns, str | bytes)
                or not raw_interrupt.ns
                or not all(type(part) is str for part in raw_interrupt.ns)
            ):
                raise CheckpointRehydrationError(
                    "checkpoint interrupt shape is not registered"
                )
            namespaces.append(tuple(raw_interrupt.ns))
        control_id = _structural_control_id(
            task_id=task_id,
            task_path=task_path,
            channel=channel,
            ordinal=ordinal,
            interrupt_namespaces=tuple(namespaces),
            resume_shape=None,
            resume_count=0,
        )
        try:
            control = StructuralControlValueV1(
                value_kind="STRUCTURAL_CONTROL",
                control="INTERRUPT",
                control_id=control_id,
                interrupt_namespaces=tuple(namespaces),
            )
        except Exception:
            raise CheckpointRehydrationError(
                "checkpoint interrupt shape is not registered"
            ) from None
        return "STRUCTURAL", control
    if channel == RESUME:
        resume_shape: Literal["ACK", "ACK_SEQUENCE"]
        if value is True:
            resume_shape = "ACK"
            resume_count = 1
        elif (
            type(value) is list
            and 0 < len(value) <= 16
            and all(item is True for item in value)
        ):
            resume_shape = "ACK_SEQUENCE"
            resume_count = len(value)
        else:
            raise CheckpointRehydrationError(
                "checkpoint resume shape is not registered"
            )
        control_id = _structural_control_id(
            task_id=task_id,
            task_path=task_path,
            channel=channel,
            ordinal=ordinal,
            interrupt_namespaces=(),
            resume_shape=resume_shape,
            resume_count=resume_count,
        )
        return (
            "STRUCTURAL",
            StructuralControlValueV1(
                value_kind="STRUCTURAL_CONTROL",
                control="RESUME",
                control_id=control_id,
                resume_shape=resume_shape,
                resume_count=resume_count,
            ),
        )
    if value is None:
        return "NULL", None
    raise CheckpointRehydrationError(
        "checkpoint pending-write shape is not registered"
    )


def _structural_control_id(
    *,
    task_id: str,
    task_path: str,
    channel: str,
    ordinal: int,
    interrupt_namespaces: tuple[tuple[str, ...], ...],
    resume_shape: Literal["ACK", "ACK_SEQUENCE"] | None,
    resume_count: int,
) -> str:
    canonical = json.dumps(
        {
            "channel": channel,
            "interrupt_namespaces": interrupt_namespaces,
            "ordinal": ordinal,
            "resume_count": resume_count,
            "resume_shape": resume_shape,
            "task_id": task_id,
            "task_path": task_path,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _rehydrate_structural_control(value: StructuralControlValueV1) -> Any:
    if value.control == "INTERRUPT":
        descriptor = {
            "control": "GENERIC_INTERRUPT_V1",
            "control_id": value.control_id,
        }
        return tuple(
            Interrupt(
                value=descriptor,
                resumable=True,
                ns=list(namespace),
                when="during",
            )
            for namespace in value.interrupt_namespaces
        )
    if value.control == "RESUME":
        if value.resume_shape == "ACK":
            return True
        if value.resume_shape == "ACK_SEQUENCE":
            return [True] * value.resume_count
    raise CheckpointRehydrationError(
        "checkpoint structural control is unsupported"
    )


def _validate_pending_domain(
    domain: PendingWriteDomainV1,
    execution: ExecutionScope,
) -> None:
    if (
        domain.thread_id != execution.thread_id
        or domain.conversation_id != execution.conversation_id
        or domain.run_id != execution.run_id
        or domain.logical_namespace != _LOGICAL_NAMESPACE
    ):
        raise CheckpointRehydrationError(
            "checkpoint pending-write domain mismatch"
        )


def _state_from_checkpoint(
    checkpoint: Checkpoint,
) -> tuple[ConversationCheckpointState, str, dict[str, Any]]:
    values = checkpoint.get("channel_values")
    if not isinstance(values, Mapping):
        raise CheckpointRehydrationError("checkpoint channel values are invalid")
    if "__start__" in values:
        raw_state = values["__start__"]
        shape = _SHAPE_START
        excluded = {"__start__"}
    elif all(channel in values for channel in _STATE_CHANNELS):
        raw_state = {channel: values[channel] for channel in _STATE_CHANNELS}
        shape = _SHAPE_STATE
        excluded = set(_STATE_CHANNELS)
    else:
        raise CheckpointRehydrationError("checkpoint Runtime State shape is unknown")
    if not isinstance(raw_state, Mapping):
        raise CheckpointRehydrationError("checkpoint Runtime State is invalid")
    controls: dict[str, Any] = {}
    for channel, value in values.items():
        if channel in excluded:
            continue
        if not channel.startswith("branch:") or value is not None:
            raise CheckpointRehydrationError(
                "checkpoint contains an unregistered control channel"
            )
        controls[channel] = None
    try:
        state = load_checkpoint_state(dict(raw_state))
    except Exception:
        raise CheckpointRehydrationError("checkpoint Runtime State is invalid") from None
    return state, shape, controls


def _state_from_pending_writes(
    writes: Sequence[tuple[str, Any]],
) -> ConversationCheckpointState:
    state_values: dict[str, Any] = {}
    for channel, value in writes:
        if channel in _STATE_CHANNELS:
            if channel in state_values:
                raise CheckpointRehydrationError(
                    "checkpoint pending state channel is duplicated"
                )
            state_values[channel] = value
        elif channel in {INTERRUPT, RESUME}:
            continue
        elif value is not None:
            raise CheckpointRehydrationError(
                "checkpoint pending-write shape is not registered"
            )
    if set(state_values) != set(_STATE_CHANNELS):
        raise CheckpointRehydrationError(
            "checkpoint pending writes do not contain a complete Runtime State"
        )
    try:
        return load_checkpoint_state(state_values)
    except Exception:
        raise CheckpointRehydrationError(
            "checkpoint pending Runtime State is invalid"
        ) from None


def _safe_metadata(metadata: CheckpointMetadata) -> CheckpointMetadata:
    safe: dict[str, Any] = {}
    source = metadata.get("source")
    step = metadata.get("step")
    parents = metadata.get("parents")
    if type(source) is str and source in {"input", "loop", "update", "fork"}:
        safe["source"] = source
    if type(step) is int:
        safe["step"] = step
    if isinstance(parents, Mapping) and all(
        type(key) is str and type(value) is str
        for key, value in parents.items()
    ):
        safe["parents"] = dict(parents)
    return cast(CheckpointMetadata, safe)


def _required_fence(execution: ExecutionScope) -> int:
    fence = execution.lease.fence_token
    if type(fence) is not int or fence <= 0:
        raise CheckpointRehydrationError(
            "checkpoint write requires a server-issued positive fence"
        )
    return fence


__all__ = [
    "AsyncCheckpointSaverPort",
    "CheckpointContentSourcePort",
    "CheckpointLeaseAuthorityPort",
    "ProtectedCheckpointBridge",
]
