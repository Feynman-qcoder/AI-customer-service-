"""Durable first-admin-decision and server-resolved graph resume boundary."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, Protocol, cast
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import CheckpointTuple
from langgraph.constants import INTERRUPT
from langgraph.types import Command, Interrupt

from app.agent.state import (
    ActionDraftSnapshot,
    ApprovalDecision,
    ConversationCheckpointState,
    CustomerConfirmationStatus,
    RunStatus,
    load_checkpoint_state,
)
from app.agent.thread_identity import ThreadIdentity, derive_checkpoint_namespace
from app.core.security import AuthenticatedUser
from app.observability import (
    AttemptOutcome,
    AttemptResumeKind,
    HITLOperationRecordV1,
    NoOpMetricsRecorder,
    NoOpObservability,
    NormalizedErrorTypeV1,
    ObservabilityStatus,
    OperationCode,
)
from app.runtime.admin_decision import (
    AdminActionRecord,
    AdminDecisionPersistenceConflict,
    AdminDecisionPersistenceCorruption,
    AdminDecisionPersistenceError,
    AdminDecisionPersistenceNotFound,
    AdminDecisionStorePort,
    AdminDecisionValue,
    AdminResumeScope,
    DurableAdminDecisionWrite,
    DurableAdminResumeWrite,
    R2AdminDecisionWrite,
    is_admin_decision_available,
)
from app.runtime.admin_decision_audit import (
    AdminDecisionDeniedAuditEvent,
    AdminDecisionSecurityAuditError,
    AdminDecisionSecurityAuditPort,
    AdminDecisionSecurityAuditStorePort,
    ProtectedAdminDecisionAuditWrite,
    normalized_non_admin_role,
)
from app.runtime.business_execution import (
    BusinessExecutionApplicationPort,
    BusinessExecutionOutcome,
)
from app.runtime.context import (
    ActorIdentity,
    AttemptWriteCapability,
    ControlledResourceFactories,
    ExecutionScope,
    LeaseExecutionScope,
    RuntimeConfigReader,
    RuntimeContextProvider,
    SubjectIdentity,
)
from app.runtime.data_protection import DataProtectionPort, DataProtectionProfile
from app.runtime.durable import EffectWriteScope
from app.runtime.observability_runtime import (
    AttemptObservationV1,
    ObservabilityRuntime,
    normalized_error_type,
)
from app.runtime.single_flight import PerThreadSingleFlight, ThreadSingleFlightConflict
from app.runtime.uow import (
    ApplicationUnitOfWorkFactory,
    CommitOutcomeUnknown,
    UnitOfWorkOperation,
)
from app.schemas.admin_agent import AgentActionResponse
from app.services.action_execution_service import (
    ActionExecutionError,
    resolve_order_action_transition,
)
from app.services.durable_runtime_service import DurableRuntimeAuthorityService

AdminDecision = AdminDecisionValue


class AdminDecisionError(RuntimeError):
    """Sanitized error at the durable admin decision boundary."""

    status_code = 409


class AdminDecisionAccessDenied(AdminDecisionError):
    status_code = 403


class AdminDecisionNotFound(AdminDecisionError):
    status_code = 404


class AdminDecisionConflict(AdminDecisionError):
    status_code = 409


class AdminDecisionResumeError(AdminDecisionError):
    status_code = 503


@dataclass(frozen=True, slots=True)
class AdminDecisionCommand:
    action_id: int
    lock_version: int
    decision: AdminDecision
    approval_note: str | None = None

    def __post_init__(self) -> None:
        if type(self.action_id) is not int or self.action_id <= 0:
            raise AdminDecisionError("action identity is invalid")
        if type(self.lock_version) is not int or self.lock_version < 0:
            raise AdminDecisionError("action lock version is invalid")
        if not isinstance(self.decision, AdminDecision):
            raise AdminDecisionError("admin decision is invalid")
        if self.approval_note is not None and (
            type(self.approval_note) is not str
            or len(self.approval_note) > 512
        ):
            raise AdminDecisionError("admin approval note is invalid")
        if (
            self.decision is AdminDecision.REJECT
            and not self.approval_note
        ):
            raise AdminDecisionError("admin rejection requires a note")


@dataclass(frozen=True, slots=True)
class _PersistedAdminDecision:
    action_id: int
    decision: AdminDecision
    admin_actor_id: int
    reason_code: str


@dataclass(slots=True)
class _AdminResumeCapability:
    conversation_id: int
    thread_id: str
    run_id: str
    attempt_id: str
    admin_actor_id: int
    subject_user_id: int
    action_id: int
    logical_action_id: str
    draft_revision: int
    decision: AdminDecision
    reason_code: str
    _issuer: object = field(repr=False, compare=False)
    _consumed: bool = field(default=False, init=False, repr=False, compare=False)


_CURRENT_ADMIN_RESUME: ContextVar[_AdminResumeCapability | None] = ContextVar(
    "admin_resume_capability",
    default=None,
)


class AdminDecisionResumeCapabilityProvider:
    """Issue and consume one process-local, attempt-bound admin decision."""

    __slots__ = ("_issuer",)

    def __init__(self) -> None:
        self._issuer = object()

    def issue(
        self,
        *,
        execution: ExecutionScope,
        state: ConversationCheckpointState,
        decision: _PersistedAdminDecision,
    ) -> _AdminResumeCapability:
        execution.require_attempt_active()
        active_run = state.active_run
        if active_run is None:
            raise AdminDecisionResumeError(
                "admin resume capability cannot bind this pause point"
            )
        draft = active_run.action_draft
        if (
            execution.actor.role != "ADMIN"
            or execution.actor.user_id != decision.admin_actor_id
            or draft is None
            or active_run.pending_action_id != decision.action_id
        ):
            raise AdminDecisionResumeError(
                "admin resume capability cannot bind this pause point"
            )
        return _AdminResumeCapability(
            conversation_id=execution.conversation_id,
            thread_id=execution.thread_id,
            run_id=execution.run_id,
            attempt_id=execution.attempt_id,
            admin_actor_id=execution.actor.user_id,
            subject_user_id=execution.subject.user_id,
            action_id=decision.action_id,
            logical_action_id=draft.logical_action_id,
            draft_revision=draft.draft_revision,
            decision=decision.decision,
            reason_code=decision.reason_code,
            _issuer=self._issuer,
        )

    @contextmanager
    def bind(self, capability: _AdminResumeCapability) -> Iterator[None]:
        if capability._issuer is not self._issuer or capability._consumed:
            raise AdminDecisionResumeError(
                "admin resume capability is invalid or already consumed"
            )
        token: Token[_AdminResumeCapability | None] = _CURRENT_ADMIN_RESUME.set(
            capability
        )
        try:
            yield
        finally:
            _CURRENT_ADMIN_RESUME.reset(token)

    def consume(
        self,
        *,
        execution: ExecutionScope,
        state: ConversationCheckpointState,
    ) -> _PersistedAdminDecision:
        execution.require_attempt_active()
        capability = _CURRENT_ADMIN_RESUME.get()
        active_run = state.active_run
        draft = active_run.action_draft if active_run is not None else None
        if capability is None or capability._issuer is not self._issuer:
            raise AdminDecisionResumeError(
                "admin resume capability is not installed"
            )
        actual = (
            execution.conversation_id,
            execution.thread_id,
            execution.run_id,
            execution.attempt_id,
            execution.actor.user_id,
            execution.actor.role,
            execution.subject.user_id,
            active_run.attempt_id if active_run is not None else None,
            active_run.pending_action_id if active_run is not None else None,
            draft.logical_action_id if draft is not None else None,
            draft.draft_revision if draft is not None else None,
        )
        expected = (
            capability.conversation_id,
            capability.thread_id,
            capability.run_id,
            capability.attempt_id,
            capability.admin_actor_id,
            "ADMIN",
            capability.subject_user_id,
            capability.attempt_id,
            capability.action_id,
            capability.logical_action_id,
            capability.draft_revision,
        )
        if capability._consumed or actual != expected:
            raise AdminDecisionResumeError(
                "admin resume capability does not match this attempt"
            )
        capability._consumed = True
        return _PersistedAdminDecision(
            action_id=capability.action_id,
            decision=capability.decision,
            admin_actor_id=capability.admin_actor_id,
            reason_code=capability.reason_code,
        )


class AdminDecisionGraphPort(Protocol):
    async def ainvoke(
        self,
        state: Command[Any] | None,
        config: RunnableConfig | None = None,
    ) -> Mapping[str, object]: ...


class AdminDecisionCheckpointReaderPort(Protocol):
    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None: ...


class _UnavailableRuntimeConfigReader(RuntimeConfigReader):
    async def read(self, unit_of_work: object) -> object:
        del unit_of_work
        raise RuntimeError("admin decision resume does not use model config")


@dataclass(frozen=True, slots=True)
class _FormalAdminCheckpoint:
    state: ConversationCheckpointState
    stage: Literal[
        "ADMIN_READY",
        "ADMIN_INTERRUPT",
        "ADMIN_CONSUMED_APPROVE",
        "ADMIN_CONSUMED_REJECT",
        "BUSINESS_TERMINAL_EXECUTED",
        "BUSINESS_TERMINAL_STALE",
    ]


@dataclass(frozen=True, slots=True)
class AdminReconcileResult:
    scanned: int
    resumed: int
    failed: int


class AdminDecisionSecurityAuditApplication(AdminDecisionSecurityAuditPort):
    """Protect and persist one mandatory unauthorized-decision event."""

    def __init__(
        self,
        *,
        unit_of_work: ApplicationUnitOfWorkFactory[
            AdminDecisionSecurityAuditStorePort
        ],
        data_protection: DataProtectionPort,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._data_protection = data_protection

    async def record_denial(self, event: AdminDecisionDeniedAuditEvent) -> None:
        try:
            protected = self._data_protection.protect(
                event.payload(),
                profile=DataProtectionProfile.OBSERVABILITY,
            )
            write = ProtectedAdminDecisionAuditWrite(
                actor_user_id=event.actor_user_id,
                actor_role=event.actor_role,
                event_code=event.event_code,
                result_code=event.result_code,
                reason_code=event.reason_code,
                canonical_json=protected.canonical_bytes.decode("utf-8"),
                payload_digest=protected.sha256,
            )
            async with self._unit_of_work.open(
                operation=UnitOfWorkOperation.AGENT_ADMIN_ACTION_DECIDE
            ) as uow:
                await uow.store.append_admin_decision_denial(write)
        except asyncio.CancelledError:
            raise
        except Exception:
            raise AdminDecisionSecurityAuditError(
                "admin decision security audit is unavailable"
            ) from None


class AdminDecisionApplicationService:
    """Persist the first admin decision, then resume only its server state."""

    def __init__(
        self,
        *,
        unit_of_work: ApplicationUnitOfWorkFactory[AdminDecisionStorePort],
        durable_authority: DurableRuntimeAuthorityService,
        runtime_contexts: RuntimeContextProvider,
        graph: AdminDecisionGraphPort,
        checkpoint_reader: AdminDecisionCheckpointReaderPort | None,
        single_flight: PerThreadSingleFlight,
        resume_capabilities: AdminDecisionResumeCapabilityProvider,
        security_audit: AdminDecisionSecurityAuditPort,
        business_execution: BusinessExecutionApplicationPort,
        lease_ttl_seconds: int,
        lease_renew_interval_seconds: int,
        enabled: bool,
        observability_runtime: ObservabilityRuntime | None = None,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._durable_authority = durable_authority
        self._runtime_contexts = runtime_contexts
        self._graph = graph
        self._checkpoint_reader = checkpoint_reader
        self._single_flight = single_flight
        self._resume_capabilities = resume_capabilities
        self._security_audit = security_audit
        self._business_execution = business_execution
        self._lease_ttl_seconds = lease_ttl_seconds
        self._lease_renew_interval_seconds = lease_renew_interval_seconds
        self._enabled = enabled
        self._observability_runtime = observability_runtime or ObservabilityRuntime(
            NoOpObservability(),
            NoOpMetricsRecorder(),
        )

    @property
    def available(self) -> bool:
        return self._enabled and self._checkpoint_reader is not None

    async def decide_and_resume(
        self,
        actor: AuthenticatedUser,
        command: AdminDecisionCommand,
    ) -> AgentActionResponse:
        if actor.role != "ADMIN":
            await self._audit_and_deny_non_admin(actor)
        resolved = await self._resolve(command.action_id)
        if resolved.confirmation_mode == "R2_STATELESS_COMPAT":
            return _response(await self._decide_r2(actor, command))
        if resolved.confirmation_mode != "DURABLE_INTERRUPT":
            raise AdminDecisionConflict(
                "the action has no trusted durable confirmation"
            )
        if (
            resolved.admin_decision == command.decision.value
            and resolved.admin_decided_actor_id is not None
            and resolved.admin_decided_actor_id != actor.user_id
        ):
            return await self.resume_persisted_action(resolved.id)
        record = await self._resume_durable(
            actor=actor,
            resolved=resolved,
            command=command,
            resume_operation=OperationCode.HITL_ADMIN_RESUME,
        )
        return _response(record)

    async def resume_persisted_action(self, action_id: int) -> AgentActionResponse:
        resolved = await self._resolve(action_id)
        if (
            resolved.confirmation_mode != "DURABLE_INTERRUPT"
            or resolved.admin_decision not in {"APPROVE", "REJECT"}
            or resolved.admin_decided_actor_id is None
            or resolved.decider_username is None
            or resolved.decider_display_name is None
            or resolved.decider_role != "ADMIN"
            or resolved.decider_status != "ACTIVE"
        ):
            raise AdminDecisionConflict(
                "the resume candidate has no active persisted admin authority"
            )
        actor = AuthenticatedUser(
            user_id=resolved.admin_decided_actor_id,
            username=resolved.decider_username,
            name=resolved.decider_display_name,
            role=resolved.decider_role,
        )
        record = await self._resume_durable(
            actor=actor,
            resolved=resolved,
            command=None,
            resume_operation=OperationCode.HITL_RECONCILER_RESUME,
        )
        return _response(record)

    async def list_resume_candidates(self, *, limit: int) -> tuple[int, ...]:
        try:
            async with self._unit_of_work.open(
                operation=UnitOfWorkOperation.AGENT_ADMIN_RECONCILE_SCAN
            ) as uow:
                return await uow.store.list_durable_admin_resume_candidates(
                    limit=limit
                )
        except AdminDecisionPersistenceError:
            raise AdminDecisionResumeError(
                "durable admin resume scan failed"
            ) from None

    async def _resume_durable(
        self,
        *,
        actor: AuthenticatedUser,
        resolved: AdminActionRecord,
        command: AdminDecisionCommand | None,
        resume_operation: OperationCode,
    ) -> AdminActionRecord:
        if not self.available:
            raise AdminDecisionResumeError(
                "durable admin resume is unavailable"
            )
        self._require_admin(actor)
        if resolved.thread_id != ThreadIdentity.from_conversation_id(
            resolved.conversation_id
        ).thread_id:
            raise AdminDecisionConflict("durable action thread identity is invalid")
        try:
            single_flight = await self._single_flight.acquire(resolved.thread_id)
        except ThreadSingleFlightConflict:
            latest = await self._resolve(resolved.id)
            if (
                command is not None
                and latest.admin_decision == command.decision.value
            ):
                return latest
            raise AdminDecisionConflict(
                "the durable action is already being resumed"
            ) from None

        attempt_capability = AttemptWriteCapability()
        execution: ExecutionScope | None = None
        lease_acquired = False
        operation_error: BaseException | None = None
        decision_committed = False
        resume_write: DurableAdminResumeWrite | None = None
        draft: ActionDraftSnapshot | None = None
        observation: AttemptObservationV1 | None = None
        observation_outcome = AttemptOutcome.FAILED
        observation_error: NormalizedErrorTypeV1 | None = None
        observation_started_ns = time.monotonic_ns()
        try:
            latest = await self._resolve(resolved.id)
            if (
                latest.run_id != resolved.run_id
                or latest.thread_id != resolved.thread_id
                or latest.conversation_id != resolved.conversation_id
                or latest.subject_user_id != resolved.subject_user_id
            ):
                raise AdminDecisionConflict(
                    "durable action identity changed before resume"
                )
            attempt_id = "attempt_" + uuid4().hex
            started_at = datetime.now(UTC)
            unfenced = ExecutionScope(
                conversation_id=latest.conversation_id,
                thread_id=latest.thread_id,
                run_id=latest.run_id,
                attempt_id=attempt_id,
                actor=ActorIdentity(
                    user_id=actor.user_id,
                    username=actor.username,
                    display_name=actor.name,
                    role=actor.role,
                ),
                subject=SubjectIdentity(
                    user_id=latest.subject_user_id,
                    role_snapshot="CUSTOMER",
                ),
                lease=LeaseExecutionScope(owner_attempt_id=attempt_id),
                started_at=started_at,
                attempt_capability=attempt_capability,
            )
            await self._durable_authority.register_attempt(unfenced)
            grant = await self._durable_authority.acquire_lease(
                unfenced,
                lease_duration=timedelta(seconds=self._lease_ttl_seconds),
            )
            lease_acquired = True
            execution = ExecutionScope(
                conversation_id=unfenced.conversation_id,
                thread_id=unfenced.thread_id,
                run_id=unfenced.run_id,
                attempt_id=unfenced.attempt_id,
                actor=unfenced.actor,
                subject=unfenced.subject,
                lease=LeaseExecutionScope(
                    owner_attempt_id=unfenced.attempt_id,
                    fence_token=grant.fence_version,
                ),
                started_at=unfenced.started_at,
                attempt_capability=attempt_capability,
            )
            observation = self._observability_runtime.begin(
                execution,
                resume=AttemptResumeKind.RESUME,
            )
            runtime = self._runtime_contexts.resolve(
                conversation_id=execution.conversation_id,
                thread_id=execution.thread_id,
                run_id=execution.run_id,
                attempt_id=execution.attempt_id,
                actor=execution.actor,
                subject=execution.subject,
                fence_token=grant.fence_version,
                attempt_capability=attempt_capability,
                resource_factory=lambda: ControlledResourceFactories(
                    unit_of_work=cast(
                        ApplicationUnitOfWorkFactory[object],
                        self._unit_of_work,
                    ),
                    runtime_config=_UnavailableRuntimeConfigReader(),
                ),
            )
            config = cast(
                RunnableConfig,
                {
                    "configurable": {
                        "thread_id": execution.thread_id,
                        "checkpoint_ns": derive_checkpoint_namespace(
                            execution.run_id
                        ),
                    }
                },
            )
            with self._runtime_contexts.bind(runtime):
                checkpoint_reader = self._checkpoint_reader
                if checkpoint_reader is None:
                    raise AdminDecisionResumeError(
                        "durable admin checkpoint reader is unavailable"
                    )
                formal = self._validate_formal_checkpoint(
                    await checkpoint_reader.aget_tuple(config),
                    execution=execution,
                    action_id=latest.id,
                )
                active_run = formal.state.active_run
                assert active_run is not None
                assert active_run.action_draft is not None
                draft = active_run.action_draft
                scope = self._resume_scope(execution)
                if command is not None:
                    decided = await self._apply_durable_decision(
                        DurableAdminDecisionWrite(
                            scope=scope,
                            action_id=latest.id,
                            expected_lock_version=command.lock_version,
                            decision=command.decision,
                            approval_note=command.approval_note,
                            draft=draft,
                        )
                    )
                else:
                    if latest.admin_decision is None:
                        raise AdminDecisionConflict(
                            "reconciler candidate has no admin decision"
                        )
                    decided = await self._validate_resume(
                        DurableAdminResumeWrite(
                            scope=scope,
                            action_id=latest.id,
                            decision=AdminDecision(latest.admin_decision),
                            draft=draft,
                        )
                    )
                decision_committed = decided.admin_decision is not None
                if (
                    decided.admin_decision is None
                    or decided.admin_reason_code is None
                ):
                    raise AdminDecisionConflict(
                        "persisted decision does not match the resume actor"
                    )
                self._observability_runtime.record(
                    observation.scope if observation is not None else None,
                    HITLOperationRecordV1(
                        operation=OperationCode.HITL_ADMIN_DECISION,
                        status=ObservabilityStatus.SUCCEEDED,
                        duration_ms=max(
                            0,
                            (time.monotonic_ns() - observation_started_ns)
                            // 1_000_000,
                        ),
                    ),
                )
                persisted_decision = AdminDecision(decided.admin_decision)
                resume_write = DurableAdminResumeWrite(
                    scope=scope,
                    action_id=decided.id,
                    decision=persisted_decision,
                    draft=draft,
                )
                terminal_outcome = {
                    "BUSINESS_TERMINAL_EXECUTED": BusinessExecutionOutcome.EXECUTED,
                    "BUSINESS_TERMINAL_STALE": BusinessExecutionOutcome.STALE,
                }.get(formal.stage)
                if (
                    formal.stage == "ADMIN_CONSUMED_APPROVE"
                    and persisted_decision is not AdminDecision.APPROVE
                ) or (
                    formal.stage == "ADMIN_CONSUMED_REJECT"
                    and persisted_decision is not AdminDecision.REJECT
                ) or (
                    terminal_outcome is not None
                    and persisted_decision is not AdminDecision.APPROVE
                ):
                    raise AdminDecisionConflict(
                        "checkpoint consumed a different admin decision"
                    )
                if terminal_outcome is not None:
                    await self._validate_resume(resume_write)
                    await self._business_execution.verify_terminal(
                        execution=execution,
                        draft=draft,
                        pending_action_id=decided.id,
                        expected_outcome=terminal_outcome,
                    )
                elif formal.stage == "ADMIN_CONSUMED_REJECT":
                    await self._validate_resume(resume_write)
                elif formal.stage == "ADMIN_CONSUMED_APPROVE":
                    await self._validate_resume(resume_write)
                    raw = await self._run_graph_with_renewer(
                        execution=execution,
                        config=config,
                        graph_input=None,
                    )
                    outcome = self._validate_resumed_graph_state(
                        raw,
                        execution=execution,
                        action_id=decided.id,
                        decision=persisted_decision,
                    )
                    await self._business_execution.verify_terminal(
                        execution=execution,
                        draft=draft,
                        pending_action_id=decided.id,
                        expected_outcome=outcome,
                    )
                else:
                    await self._validate_resume(resume_write)
                    capability = self._resume_capabilities.issue(
                        execution=execution,
                        state=formal.state,
                        decision=_PersistedAdminDecision(
                            action_id=decided.id,
                            decision=persisted_decision,
                            admin_actor_id=actor.user_id,
                            reason_code=decided.admin_reason_code,
                        ),
                    )
                    with self._resume_capabilities.bind(capability):
                        raw = await self._run_graph_with_renewer(
                            execution=execution,
                            config=config,
                            graph_input=Command(resume=True),
                        )
                    outcome = self._validate_resumed_graph_state(
                        raw,
                        execution=execution,
                        action_id=decided.id,
                        decision=persisted_decision,
                    )
                    if outcome is not None:
                        await self._business_execution.verify_terminal(
                            execution=execution,
                            draft=draft,
                            pending_action_id=decided.id,
                            expected_outcome=outcome,
                        )
                result = await self._mark_resume_succeeded(resume_write)
                observation_outcome = AttemptOutcome.SUCCEEDED
                self._observability_runtime.record(
                    observation.scope if observation is not None else None,
                    HITLOperationRecordV1(
                        operation=resume_operation,
                        status=ObservabilityStatus.SUCCEEDED,
                        duration_ms=max(
                            0,
                            (time.monotonic_ns() - observation_started_ns)
                            // 1_000_000,
                        ),
                    ),
                )
                return result
        except asyncio.CancelledError as caught:
            operation_error = caught
            observation_outcome = AttemptOutcome.CANCELLED
            observation_error = normalized_error_type(caught)
            raise
        except CommitOutcomeUnknown as caught:
            operation_error = caught
            raise
        except (
            AdminDecisionPersistenceNotFound,
            AdminDecisionPersistenceConflict,
            AdminDecisionPersistenceCorruption,
        ) as caught:
            operation_error = caught
            raise _map_persistence_error(caught) from None
        except AdminDecisionError as caught:
            operation_error = caught
            raise
        except BaseException as caught:
            operation_error = caught
            if (
                decision_committed
                and execution is not None
                and resume_write is not None
                and draft is not None
                and execution.attempt_capability is attempt_capability
            ):
                try:
                    await self._mark_resume_failed(resume_write)
                except BaseException:
                    pass
            raise AdminDecisionResumeError(
                "durable admin decision is saved and resume is retryable"
            ) from None
        finally:
            cleanup_error: BaseException | None = None
            if operation_error is not None:
                self._observability_runtime.record(
                    observation.scope if observation is not None else None,
                    HITLOperationRecordV1(
                        operation=resume_operation,
                        status=(
                            ObservabilityStatus.CANCELLED
                            if isinstance(operation_error, asyncio.CancelledError)
                            else ObservabilityStatus.FAILED
                        ),
                        duration_ms=max(
                            0,
                            (time.monotonic_ns() - observation_started_ns)
                            // 1_000_000,
                        ),
                        error_type=normalized_error_type(operation_error),
                    ),
                )
            attempt_capability.revoke()
            if lease_acquired and execution is not None:
                try:
                    await self._durable_authority.release_lease(execution)
                except BaseException as caught:
                    cleanup_error = caught
            try:
                await single_flight.release()
            except BaseException as caught:
                if cleanup_error is None:
                    cleanup_error = caught
            if operation_error is not None and observation_error is None:
                observation_error = normalized_error_type(operation_error)
            if cleanup_error is not None and operation_error is None:
                observation_outcome = AttemptOutcome.FAILED
                observation_error = normalized_error_type(cleanup_error)
            self._observability_runtime.finish(
                observation,
                outcome=observation_outcome,
                error_type=observation_error,
            )
            if cleanup_error is not None and operation_error is None:
                raise cleanup_error

    async def _resolve(self, action_id: int) -> AdminActionRecord:
        try:
            async with self._unit_of_work.open(
                operation=UnitOfWorkOperation.AGENT_ADMIN_ACTION_RESOLVE
            ) as uow:
                record = await uow.store.resolve_admin_action(action_id)
        except AdminDecisionPersistenceError as caught:
            raise _map_persistence_error(caught) from None
        if record is None:
            raise AdminDecisionNotFound("admin action is unavailable")
        return record

    async def _decide_r2(
        self,
        actor: AuthenticatedUser,
        command: AdminDecisionCommand,
    ) -> AdminActionRecord:
        write = R2AdminDecisionWrite(
            action_id=command.action_id,
            expected_lock_version=command.lock_version,
            admin_actor_id=actor.user_id,
            decision=command.decision,
            approval_note=command.approval_note,
        )
        try:
            async with self._unit_of_work.open(
                operation=UnitOfWorkOperation.AGENT_ADMIN_ACTION_DECIDE
            ) as uow:
                locked = await uow.store.lock_r2_admin_action(write)
                next_status: str | None = None
                restore_stock = False
                summary: str | None = None
                if command.decision is AdminDecision.APPROVE:
                    if locked.order_status is None:
                        raise AdminDecisionPersistenceConflict(
                            "R2 approval target is unavailable"
                        )
                    try:
                        transition = resolve_order_action_transition(
                            locked.action_type,
                            locked.order_status,
                        )
                    except ActionExecutionError:
                        raise AdminDecisionPersistenceConflict(
                            "R2 action is no longer executable"
                        ) from None
                    next_status = transition.next_status
                    restore_stock = transition.restore_stock
                    summary = transition.summary
                return await uow.store.apply_r2_admin_decision(
                    write,
                    next_order_status=next_status,
                    restore_stock=restore_stock,
                    execution_summary=summary,
                )
        except AdminDecisionPersistenceError as caught:
            raise _map_persistence_error(caught) from None

    async def _apply_durable_decision(
        self,
        write: DurableAdminDecisionWrite,
    ) -> AdminActionRecord:
        async with self._unit_of_work.open(
            operation=UnitOfWorkOperation.AGENT_ADMIN_ACTION_DECIDE
        ) as uow:
            return await uow.store.apply_durable_admin_decision(write)

    async def _validate_resume(
        self,
        write: DurableAdminResumeWrite,
    ) -> AdminActionRecord:
        async with self._unit_of_work.open(
            operation=UnitOfWorkOperation.AGENT_ADMIN_RESUME_VALIDATE
        ) as uow:
            return await uow.store.validate_durable_admin_resume(write)

    async def _mark_resume_succeeded(
        self,
        write: DurableAdminResumeWrite,
    ) -> AdminActionRecord:
        async with self._unit_of_work.open(
            operation=UnitOfWorkOperation.AGENT_ADMIN_RESUME_MARK
        ) as uow:
            return await uow.store.mark_durable_admin_resume_succeeded(write)

    async def _mark_resume_failed(
        self,
        write: DurableAdminResumeWrite,
    ) -> AdminActionRecord:
        async with self._unit_of_work.open(
            operation=UnitOfWorkOperation.AGENT_ADMIN_RESUME_MARK
        ) as uow:
            return await uow.store.mark_durable_admin_resume_failed(write)

    def _resume_scope(self, execution: ExecutionScope) -> AdminResumeScope:
        execution.require_attempt_active()
        fence = execution.lease.fence_token
        if type(fence) is not int or fence <= 0:
            raise AdminDecisionResumeError(
                "admin resume requires a positive live fence"
            )
        return AdminResumeScope(
            effect_scope=EffectWriteScope(
                thread_id=execution.thread_id,
                attempt_id=execution.attempt_id,
                fence_version=fence,
            ),
            conversation_id=execution.conversation_id,
            run_id=execution.run_id,
            admin_actor_id=execution.actor.user_id,
            subject_user_id=execution.subject.user_id,
        )

    def _validate_formal_checkpoint(
        self,
        checkpoint_tuple: CheckpointTuple | None,
        *,
        execution: ExecutionScope,
        action_id: int,
    ) -> _FormalAdminCheckpoint:
        if checkpoint_tuple is None:
            raise AdminDecisionResumeError(
                "durable admin checkpoint is unavailable"
            )
        values = checkpoint_tuple.checkpoint.get("channel_values")
        if not isinstance(values, Mapping):
            raise AdminDecisionResumeError("admin checkpoint state is unavailable")
        raw_state = values.get("__start__", values)
        state = self._load_graph_state(raw_state)
        active_run = state.active_run
        identity = state.conversation_identity
        if (
            identity.conversation_id != execution.conversation_id
            or identity.thread_id != execution.thread_id
            or identity.subject_user_id != execution.subject.user_id
            or active_run is None
            or active_run.run_id != execution.run_id
            or active_run.attempt_id != execution.attempt_id
            or active_run.action_draft is None
            or active_run.pending_action_id != action_id
            or active_run.customer_confirmation_status
            is not CustomerConfirmationStatus.CONFIRMED
            or active_run.side_effect_authorization is not None
        ):
            raise AdminDecisionConflict(
                "admin checkpoint does not match the server-resolved action"
            )
        interrupted: list[Interrupt] = []
        for _task_id, channel, value in checkpoint_tuple.pending_writes or ():
            if channel == INTERRUPT and type(value) is tuple:
                interrupted.extend(
                    item for item in value if isinstance(item, Interrupt)
                )
        if active_run.run_status is RunStatus.WAITING_ADMIN_APPROVAL:
            if active_run.approval_decision is not None:
                raise AdminDecisionConflict(
                    "waiting admin checkpoint already carries a decision"
                )
            if not interrupted:
                return _FormalAdminCheckpoint(state, "ADMIN_READY")
            if not _is_admin_interrupt(interrupted):
                raise AdminDecisionConflict(
                    "checkpoint is not the durable admin interrupt"
                )
            return _FormalAdminCheckpoint(state, "ADMIN_INTERRUPT")
        if interrupted:
            raise AdminDecisionConflict(
                "consumed admin checkpoint still carries an interrupt"
            )
        if (
            active_run.run_status is RunStatus.RESUME_PENDING
            and active_run.approval_decision is ApprovalDecision.APPROVED
        ):
            return _FormalAdminCheckpoint(state, "ADMIN_CONSUMED_APPROVE")
        if (
            active_run.run_status is RunStatus.REJECTED
            and active_run.approval_decision is ApprovalDecision.REJECTED
        ):
            return _FormalAdminCheckpoint(state, "ADMIN_CONSUMED_REJECT")
        if (
            active_run.run_status is RunStatus.COMPLETED
            and active_run.approval_decision is ApprovalDecision.APPROVED
        ):
            return _FormalAdminCheckpoint(state, "BUSINESS_TERMINAL_EXECUTED")
        if (
            active_run.run_status is RunStatus.COMPLETED
            and active_run.approval_decision is ApprovalDecision.STALE
        ):
            return _FormalAdminCheckpoint(state, "BUSINESS_TERMINAL_STALE")
        raise AdminDecisionConflict(
            "checkpoint is not at a supported admin decision stage"
        )

    async def _run_graph_with_renewer(
        self,
        *,
        execution: ExecutionScope,
        config: RunnableConfig,
        graph_input: Command[Any] | None,
    ) -> Mapping[str, object]:
        stop_renewal = asyncio.Event()
        graph_result: Mapping[str, object] | None = None
        graph_error: BaseException | None = None
        renewal_error: BaseException | None = None
        graph_task: asyncio.Task[None] | None = None

        async def execute() -> None:
            nonlocal graph_error, graph_result
            try:
                graph_result = await self._graph.ainvoke(
                    graph_input,
                    config,
                )
            except asyncio.CancelledError:
                if renewal_error is None:
                    raise
            except BaseException as caught:
                graph_error = caught
            finally:
                stop_renewal.set()

        async def renew() -> None:
            nonlocal renewal_error
            while not stop_renewal.is_set():
                try:
                    async with asyncio.timeout(
                        self._lease_renew_interval_seconds
                    ):
                        await stop_renewal.wait()
                    return
                except TimeoutError:
                    try:
                        await self._durable_authority.renew_lease(
                            execution,
                            lease_duration=timedelta(
                                seconds=self._lease_ttl_seconds
                            ),
                        )
                    except asyncio.CancelledError:
                        raise
                    except BaseException as caught:
                        renewal_error = caught
                        execution.revoke_attempt()
                        stop_renewal.set()
                        if graph_task is not None and not graph_task.done():
                            graph_task.cancel()
                        return

        try:
            async with asyncio.TaskGroup() as tasks:
                graph_task = tasks.create_task(
                    execute(),
                    name="admin-decision-graph",
                )
                tasks.create_task(
                    renew(),
                    name="admin-decision-lease-renewer",
                )
        finally:
            stop_renewal.set()
        if graph_error is not None:
            raise graph_error
        if renewal_error is not None:
            raise renewal_error
        if graph_result is None:
            raise AdminDecisionResumeError(
                "admin decision graph returned no state"
            )
        return graph_result

    def _validate_resumed_graph_state(
        self,
        raw: Mapping[str, object],
        *,
        execution: ExecutionScope,
        action_id: int,
        decision: AdminDecision,
    ) -> BusinessExecutionOutcome | None:
        state = self._load_graph_state(raw)
        active_run = state.active_run
        expected_status = RunStatus.COMPLETED if decision is AdminDecision.APPROVE else RunStatus.REJECTED
        if (
            active_run is None
            or active_run.run_id != execution.run_id
            or active_run.attempt_id != execution.attempt_id
            or active_run.run_status is not expected_status
            or active_run.pending_action_id != action_id
            or active_run.side_effect_authorization is not None
        ):
            raise AdminDecisionResumeError(
                "admin decision graph result is invalid"
            )
        if decision is AdminDecision.REJECT:
            if active_run.approval_decision is not ApprovalDecision.REJECTED:
                raise AdminDecisionResumeError(
                    "admin decision graph result is invalid"
                )
            return None
        if active_run.approval_decision is ApprovalDecision.APPROVED:
            return BusinessExecutionOutcome.EXECUTED
        if active_run.approval_decision is ApprovalDecision.STALE:
            return BusinessExecutionOutcome.STALE
        raise AdminDecisionResumeError("admin decision graph result is invalid")

    def _load_graph_state(
        self,
        raw: Mapping[str, object] | object,
    ) -> ConversationCheckpointState:
        if not isinstance(raw, Mapping):
            raise AdminDecisionResumeError("admin checkpoint state is invalid")
        required = (
            "schema_version",
            "conversation_identity",
            "memory",
            "active_run",
        )
        if not all(key in raw for key in required):
            raise AdminDecisionResumeError("admin checkpoint state is incomplete")
        try:
            return load_checkpoint_state({key: raw[key] for key in required})
        except Exception:
            raise AdminDecisionResumeError(
                "admin checkpoint state is invalid"
            ) from None

    def _require_admin(self, actor: AuthenticatedUser) -> None:
        if actor.role != "ADMIN":
            raise AdminDecisionAccessDenied(
                "admin decision access is denied"
            )

    async def _audit_and_deny_non_admin(
        self,
        actor: AuthenticatedUser,
    ) -> None:
        try:
            await self._security_audit.record_denial(
                AdminDecisionDeniedAuditEvent(
                    actor_user_id=actor.user_id,
                    actor_role=normalized_non_admin_role(actor.role),
                )
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
        raise AdminDecisionAccessDenied(
            "admin decision access is denied"
        ) from None


class AdminDecisionReconciler:
    """Bounded, repeatable scanner for committed but unconsumed decisions."""

    def __init__(
        self,
        application: AdminDecisionApplicationService,
        *,
        timeout_seconds: float = 30.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("admin reconciler timeout must be positive")
        self._application = application
        self._timeout_seconds = timeout_seconds

    async def run_once(self, *, limit: int = 32) -> AdminReconcileResult:
        if not self._application.available:
            return AdminReconcileResult(scanned=0, resumed=0, failed=0)
        resumed = 0
        failed = 0
        async with asyncio.timeout(self._timeout_seconds):
            candidates = await self._application.list_resume_candidates(
                limit=limit
            )
            for action_id in candidates:
                try:
                    await self._application.resume_persisted_action(action_id)
                except (
                    AdminDecisionError,
                    ThreadSingleFlightConflict,
                    CommitOutcomeUnknown,
                ):
                    failed += 1
                else:
                    resumed += 1
        return AdminReconcileResult(
            scanned=len(candidates),
            resumed=resumed,
            failed=failed,
        )


class AdminDecisionReconcilerRunner:
    """Lifespan-owned serial scanner with bounded periodic retry."""

    def __init__(
        self,
        reconciler: AdminDecisionReconciler,
        *,
        interval_seconds: float = 1.0,
        limit: int = 32,
    ) -> None:
        if (
            type(interval_seconds) not in {int, float}
            or interval_seconds <= 0
            or interval_seconds > 300
        ):
            raise ValueError(
                "admin reconciler interval must be within (0, 300] seconds"
            )
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("admin reconciler limit must be within [1, 100]")
        self._reconciler = reconciler
        self._interval_seconds = float(interval_seconds)
        self._limit = limit
        self._stop = asyncio.Event()
        self._wake = asyncio.Event()
        self._first_scan_complete = asyncio.Event()
        self._scan_condition = asyncio.Condition()
        self._task_group: asyncio.TaskGroup | None = None
        self._task: asyncio.Task[None] | None = None
        self._first_result: AdminReconcileResult | None = None
        self._last_result: AdminReconcileResult | None = None
        self._scan_count = 0

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def scan_count(self) -> int:
        return self._scan_count

    @property
    def last_result(self) -> AdminReconcileResult | None:
        return self._last_result

    async def start(self) -> AdminReconcileResult:
        if self.running:
            await self._first_scan_complete.wait()
            if self._first_result is None:
                raise AdminDecisionResumeError(
                    "admin reconciler runner did not complete its first scan"
                )
            return self._first_result
        if self._task is not None:
            await self.stop()
        self._stop.clear()
        self._wake.clear()
        self._first_scan_complete.clear()
        self._first_result = None
        self._last_result = None
        task_group = asyncio.TaskGroup()
        await task_group.__aenter__()
        self._task_group = task_group
        self._task = task_group.create_task(
            self._run(),
            name="admin-decision-reconciler-runner",
        )
        await self._first_scan_complete.wait()
        task = self._task
        if task is not None and task.done():
            await task
        if self._first_result is None:
            raise AdminDecisionResumeError(
                "admin reconciler runner did not complete its first scan"
            )
        return self._first_result

    def request_scan(self) -> None:
        if self.running:
            self._wake.set()

    async def wait_for_scan_count(
        self,
        expected_count: int,
        *,
        timeout_seconds: float,
    ) -> None:
        if type(expected_count) is not int or expected_count < 0:
            raise ValueError("expected scan count must be non-negative")
        if (
            type(timeout_seconds) not in {int, float}
            or timeout_seconds <= 0
        ):
            raise ValueError("scan wait timeout must be positive")
        async with asyncio.timeout(float(timeout_seconds)):
            async with self._scan_condition:
                await self._scan_condition.wait_for(
                    lambda: self._scan_count >= expected_count
                )

    async def stop(self) -> None:
        task = self._task
        if task is None:
            return
        task_group = self._task_group
        self._stop.set()
        self._wake.set()
        if not task.done():
            task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        finally:
            self._task = None
            self._task_group = None
            if task_group is not None:
                await task_group.__aexit__(None, None, None)

    async def _run(self) -> None:
        try:
            while not self._stop.is_set():
                result = await self._run_once_safely()
                self._last_result = result
                async with self._scan_condition:
                    self._scan_count += 1
                    self._scan_condition.notify_all()
                if self._first_result is None:
                    self._first_result = result
                    self._first_scan_complete.set()
                if self._stop.is_set():
                    return
                try:
                    async with asyncio.timeout(self._interval_seconds):
                        await self._wake.wait()
                except TimeoutError:
                    pass
                finally:
                    self._wake.clear()
        finally:
            self._first_scan_complete.set()
            async with self._scan_condition:
                self._scan_condition.notify_all()

    async def _run_once_safely(self) -> AdminReconcileResult:
        try:
            return await self._reconciler.run_once(limit=self._limit)
        except asyncio.CancelledError:
            raise
        except Exception:
            return AdminReconcileResult(scanned=0, resumed=0, failed=1)


def _is_admin_interrupt(interrupted: list[Interrupt]) -> bool:
    return (
        len(interrupted) == 1
        and interrupted[0].resumable is True
        and interrupted[0].when == "during"
        and interrupted[0].ns is not None
        and len(interrupted[0].ns) == 1
        and interrupted[0].ns[0].startswith("admin_approval_gate:")
        and isinstance(interrupted[0].value, Mapping)
        and interrupted[0].value.get("control") == "GENERIC_INTERRUPT_V1"
        and type(interrupted[0].value.get("control_id")) is str
    )


def _map_persistence_error(error: AdminDecisionPersistenceError) -> AdminDecisionError:
    if isinstance(error, AdminDecisionPersistenceNotFound):
        return AdminDecisionNotFound("admin action is unavailable")
    return AdminDecisionConflict("durable admin decision was rejected")


def _response(record: AdminActionRecord) -> AgentActionResponse:
    return AgentActionResponse(
        id=record.id,
        runId=record.run_id,
        actionType=record.action_type,
        targetOrderId=record.target_order_id,
        actionPayloadJson=record.action_payload_json,
        riskLevel=record.risk_level,
        status=record.status,
        idempotencyKey=record.idempotency_key,
        lockVersion=record.lock_version,
        createdBy=record.created_by,
        approvedBy=record.approved_by,
        approvalNote=record.approval_note,
        logicalActionId=record.logical_action_id,
        confirmationMode=record.confirmation_mode,
        customerConfirmedActorId=record.customer_confirmed_actor_id,
        customerConfirmedAt=record.customer_confirmed_at,
        customerConfirmationChallengeDigest=(
            record.customer_confirmation_challenge_digest
        ),
        adminDecision=record.admin_decision,
        adminDecidedActorId=record.admin_decided_actor_id,
        adminDecidedAt=record.admin_decided_at,
        adminReasonCode=record.admin_reason_code,
        resumeStatus=record.resume_status,
        decisionAvailable=is_admin_decision_available(
            status=record.status,
            confirmation_mode=record.confirmation_mode,
            resume_status=record.resume_status,
            admin_decision=record.admin_decision,
        ),
        executionResultCode=record.execution_result_code,
        executionErrorType=record.execution_error_type,
        executionErrorSummary=record.execution_error_summary,
        legacyOriginalStatus=record.legacy_original_status,
        createdAt=record.created_at,
        approvedAt=record.approved_at,
        executedAt=record.executed_at,
    )


__all__ = [
    "AdminDecision",
    "AdminDecisionAccessDenied",
    "AdminDecisionApplicationService",
    "AdminDecisionCheckpointReaderPort",
    "AdminDecisionCommand",
    "AdminDecisionConflict",
    "AdminDecisionError",
    "AdminDecisionGraphPort",
    "AdminDecisionNotFound",
    "AdminDecisionReconciler",
    "AdminDecisionResumeCapabilityProvider",
    "AdminDecisionResumeError",
    "AdminReconcileResult",
    "AdminDecisionReconcilerRunner",
    "AdminDecisionSecurityAuditApplication",
]
