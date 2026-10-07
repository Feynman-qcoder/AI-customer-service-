"""Authenticated durable customer-confirmation resume application boundary."""

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
from app.repositories.agent_workflow_repository import (
    ConfirmationStorePort,
    CustomerConfirmationOwnershipError,
    CustomerConfirmationRunSnapshot,
    CustomerConfirmationStoreError,
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
from app.runtime.durable import EffectWriteScope
from app.runtime.observability_runtime import (
    AttemptObservationV1,
    ObservabilityRuntime,
    normalized_error_type,
)
from app.runtime.single_flight import PerThreadSingleFlight
from app.runtime.uow import ApplicationUnitOfWorkFactory, UnitOfWorkOperation
from app.schemas.chat import CustomerConfirmationResponse
from app.services.durable_runtime_service import DurableRuntimeAuthorityService
from app.services.prepared_action_validation import (
    PreparedActionValidationError,
    PreparedActionValidationPort,
)
from app.services.side_effect_policy_service import (
    ActionDraftSnapshot as PolicyActionDraftSnapshot,
)
from app.services.side_effect_policy_service import (
    SideEffectPolicyError,
    SideEffectPolicyService,
)


class CustomerConfirmationError(RuntimeError):
    """Sanitized domain failure at the customer confirmation boundary."""

    status_code = 409


class CustomerConfirmationAccessDenied(CustomerConfirmationError):
    status_code = 403


class CustomerResumeCapabilityError(CustomerConfirmationError):
    """The graph ACK lacks the exact application-issued one-use capability."""


class CustomerConfirmationRetryable(CustomerConfirmationError):
    """The MySQL marker has no recoverable checkpoint and was closed safely."""


class _CheckpointRecoveryRequired(CustomerConfirmationError):
    """A fenced MySQL WAITING marker is ahead of its formal pause point."""


@dataclass(slots=True)
class _CustomerResumeCapability:
    conversation_id: int
    thread_id: str
    run_id: str
    attempt_id: str
    subject_user_id: int
    logical_action_id: str
    challenge_digest: str
    _issuer: object = field(repr=False, compare=False)
    _consumed: bool = field(default=False, init=False, repr=False, compare=False)


_CURRENT_CUSTOMER_RESUME: ContextVar[_CustomerResumeCapability | None] = ContextVar(
    "customer_resume_capability", default=None
)


class CustomerResumeCapabilityProvider:
    """Issues and consumes process-local, attempt-scoped ACK capabilities."""

    __slots__ = ("_issuer",)

    def __init__(self) -> None:
        self._issuer = object()

    def issue(
        self,
        *,
        execution: ExecutionScope,
        state: ConversationCheckpointState,
    ) -> _CustomerResumeCapability:
        execution.require_attempt_active()
        active_run = state.active_run
        if active_run is None or active_run.action_draft is None:
            raise CustomerResumeCapabilityError("customer resume capability requires a waiting action draft")
        return _CustomerResumeCapability(
            conversation_id=execution.conversation_id,
            thread_id=execution.thread_id,
            run_id=execution.run_id,
            attempt_id=execution.attempt_id,
            subject_user_id=execution.subject.user_id,
            logical_action_id=active_run.action_draft.logical_action_id,
            challenge_digest=active_run.action_draft.nonce_digest,
            _issuer=self._issuer,
        )

    @contextmanager
    def bind(self, capability: _CustomerResumeCapability) -> Iterator[None]:
        if capability._issuer is not self._issuer or capability._consumed:
            raise CustomerResumeCapabilityError("customer resume capability is invalid or already consumed")
        token: Token[_CustomerResumeCapability | None] = _CURRENT_CUSTOMER_RESUME.set(capability)
        try:
            yield
        finally:
            _CURRENT_CUSTOMER_RESUME.reset(token)

    def consume(
        self,
        *,
        execution: ExecutionScope,
        state: ConversationCheckpointState,
    ) -> None:
        execution.require_attempt_active()
        capability = _CURRENT_CUSTOMER_RESUME.get()
        active_run = state.active_run
        draft = active_run.action_draft if active_run is not None else None
        if capability is None or capability._issuer is not self._issuer:
            raise CustomerResumeCapabilityError("customer resume capability is not installed")
        actual = (
            execution.conversation_id,
            execution.thread_id,
            execution.run_id,
            execution.attempt_id,
            execution.subject.user_id,
            draft.logical_action_id if draft is not None else None,
            draft.nonce_digest if draft is not None else None,
            active_run.attempt_id if active_run is not None else None,
        )
        expected = (
            capability.conversation_id,
            capability.thread_id,
            capability.run_id,
            capability.attempt_id,
            capability.subject_user_id,
            capability.logical_action_id,
            capability.challenge_digest,
            capability.attempt_id,
        )
        if capability._consumed or actual != expected:
            raise CustomerResumeCapabilityError("customer resume capability does not match this attempt")
        capability._consumed = True


@dataclass(frozen=True, slots=True)
class CustomerConfirmationCommand:
    conversation_id: int
    confirmation_text: str
    confirmation_challenge_digest: str

    def __post_init__(self) -> None:
        if type(self.conversation_id) is not int or self.conversation_id <= 0:
            raise CustomerConfirmationError("conversation identity is invalid")
        if not self.confirmation_text or len(self.confirmation_text) > 256:
            raise CustomerConfirmationError("customer confirmation text is invalid")
        if len(self.confirmation_challenge_digest) != 64 or any(
            character not in "0123456789abcdef" for character in self.confirmation_challenge_digest
        ):
            raise CustomerConfirmationError("customer confirmation challenge is invalid")


class CustomerConfirmationGraphPort(Protocol):
    async def ainvoke(
        self,
        state: Command[Any] | None,
        config: RunnableConfig | None = None,
    ) -> Mapping[str, object]: ...


class CustomerConfirmationCheckpointReaderPort(Protocol):
    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None: ...


class _UnavailableRuntimeConfigReader(RuntimeConfigReader):
    async def read(self, unit_of_work: object) -> object:
        del unit_of_work
        raise RuntimeError("customer confirmation resume does not use model config")


@dataclass(frozen=True, slots=True)
class _ResolvedConfirmation:
    run: CustomerConfirmationRunSnapshot
    action_type: str
    target_order_no: str


@dataclass(frozen=True, slots=True)
class _FormalCheckpoint:
    state: ConversationCheckpointState
    stage: Literal[
        "CUSTOMER_INTERRUPT",
        "CUSTOMER_ACKED",
        "ADMIN_READY",
        "ADMIN_INTERRUPT",
    ]


class CustomerConfirmationApplicationService:
    """Own the server-side identity, lease and ACK orchestration for 7.2."""

    def __init__(
        self,
        *,
        unit_of_work: ApplicationUnitOfWorkFactory[ConfirmationStorePort],
        durable_authority: DurableRuntimeAuthorityService,
        runtime_contexts: RuntimeContextProvider,
        graph: CustomerConfirmationGraphPort,
        checkpoint_reader: CustomerConfirmationCheckpointReaderPort | None,
        single_flight: PerThreadSingleFlight,
        resume_capabilities: CustomerResumeCapabilityProvider,
        side_effect_policy: SideEffectPolicyService,
        prepared_action_validation: PreparedActionValidationPort,
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
        self._side_effect_policy = side_effect_policy
        self._prepared_action_validation = prepared_action_validation
        self._lease_ttl_seconds = lease_ttl_seconds
        self._lease_renew_interval_seconds = lease_renew_interval_seconds
        self._enabled = enabled
        self._observability_runtime = observability_runtime or ObservabilityRuntime(
            NoOpObservability(),
            NoOpMetricsRecorder(),
        )

    @property
    def available(self) -> bool:
        return self._enabled

    async def resume(
        self,
        actor: AuthenticatedUser,
        command: CustomerConfirmationCommand,
    ) -> CustomerConfirmationResponse:
        if not self._enabled:
            raise CustomerConfirmationError("durable customer confirmation is unavailable")
        if actor.role != "CUSTOMER":
            raise CustomerConfirmationAccessDenied("customer confirmation access is denied")
        identity = ThreadIdentity.from_conversation_id(command.conversation_id)
        await self._resolve_confirmation(actor, command)

        single_flight = await self._single_flight.acquire(identity.thread_id)
        attempt_capability = AttemptWriteCapability()
        execution: ExecutionScope | None = None
        lease_acquired = False
        operation_error: BaseException | None = None
        observation: AttemptObservationV1 | None = None
        observation_outcome = AttemptOutcome.FAILED
        observation_error: NormalizedErrorTypeV1 | None = None
        observation_started_ns = time.monotonic_ns()
        try:
            # Re-read inside the single-flight window so a winner completed
            # just before acquisition cannot cause another attempt.
            resolved = await self._resolve_confirmation(actor, command)
            run = resolved.run
            if run.thread_id != identity.thread_id:
                raise CustomerConfirmationError("customer confirmation run identity is invalid")
            attempt_id = "attempt_" + uuid4().hex
            if attempt_id == run.paused_attempt_id:
                raise CustomerConfirmationError("resume attempt must differ from the paused attempt")
            started_at = datetime.now(UTC)
            unfenced = ExecutionScope(
                conversation_id=identity.conversation_id,
                thread_id=identity.thread_id,
                run_id=run.run_id,
                attempt_id=attempt_id,
                actor=ActorIdentity(
                    user_id=actor.user_id,
                    username=actor.username,
                    display_name=actor.name,
                    role=actor.role,
                ),
                subject=SubjectIdentity(
                    user_id=actor.user_id,
                    role_snapshot=actor.role,
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
                conversation_id=identity.conversation_id,
                thread_id=identity.thread_id,
                run_id=run.run_id,
                attempt_id=attempt_id,
                actor=unfenced.actor,
                subject=unfenced.subject,
                lease=LeaseExecutionScope(
                    owner_attempt_id=attempt_id,
                    fence_token=grant.fence_version,
                ),
                started_at=started_at,
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
                        "checkpoint_ns": derive_checkpoint_namespace(execution.run_id),
                    }
                },
            )
            with self._runtime_contexts.bind(runtime):
                checkpoint_reader = self._checkpoint_reader
                if checkpoint_reader is None:
                    raise CustomerConfirmationError("customer confirmation checkpoint reader is unavailable")
                checkpoint_tuple = await checkpoint_reader.aget_tuple(config)
                if checkpoint_tuple is None:
                    if run.pending_action_id is None:
                        await self._mark_confirmation_retryable(execution)
                        raise CustomerConfirmationRetryable(
                            "customer confirmation checkpoint is unavailable; retry the original request"
                        )
                    raise CustomerConfirmationError(
                        "prepared customer action checkpoint is unavailable"
                    )
                try:
                    formal = self._validate_formal_checkpoint(
                        checkpoint_tuple=checkpoint_tuple,
                        execution=execution,
                        paused_attempt_id=run.paused_attempt_id,
                    )
                except _CheckpointRecoveryRequired:
                    if run.pending_action_id is not None:
                        raise CustomerConfirmationError(
                            "prepared customer action checkpoint is incomplete"
                        ) from None
                    await self._mark_confirmation_retryable(execution)
                    raise CustomerConfirmationRetryable(
                        "customer confirmation pause is incomplete; retry the original request"
                    ) from None
                self._validate_frozen_confirmation(
                    actor=actor,
                    command=command,
                    resolved=resolved,
                    state=formal.state,
                )
                if formal.stage == "ADMIN_INTERRUPT":
                    active_run = formal.state.active_run
                    assert active_run is not None
                    assert active_run.pending_action_id is not None
                    assert active_run.action_draft is not None
                    await self._prepared_action_validation.validate(
                        execution=execution,
                        draft=active_run.action_draft,
                        pending_action_id=active_run.pending_action_id,
                    )
                    await self._mark_admin_waiting(
                        execution,
                        pending_action_id=active_run.pending_action_id,
                    )
                    response = self._response(
                        command,
                        resolved,
                        pending_action_id=active_run.pending_action_id,
                    )
                    observation_outcome = AttemptOutcome.WAITING
                    self._record_successful_resume(
                        observation,
                        observation_started_ns,
                    )
                    return response
                if formal.stage == "CUSTOMER_INTERRUPT":
                    await self._validate_current_order_and_confirmation(
                        actor=actor,
                        command=command,
                        run=run,
                        state=formal.state,
                    )
                    capability = self._resume_capabilities.issue(
                        execution=execution,
                        state=formal.state,
                    )
                    with self._resume_capabilities.bind(capability):
                        raw = await self._run_graph_with_renewer(
                            execution=execution,
                            config=config,
                            graph_input=Command(resume=True),
                        )
                else:
                    raw = await self._run_graph_with_renewer(
                        execution=execution,
                        config=config,
                        graph_input=None,
                    )
            resumed_state = self._load_graph_state(raw)
            resumed_run = resumed_state.active_run
            if (
                resumed_run is None
                or resumed_run.run_id != execution.run_id
                or resumed_run.attempt_id != execution.attempt_id
                or resumed_run.run_status is not RunStatus.WAITING_ADMIN_APPROVAL
                or resumed_run.customer_confirmation_status is not CustomerConfirmationStatus.CONFIRMED
                or resumed_run.side_effect_authorization is not None
                or resumed_run.pending_action_id is None
            ):
                raise CustomerConfirmationError("customer confirmation resume result is invalid")
            response = CustomerConfirmationResponse(
                conversationId=execution.conversation_id,
                agentStatus=RunStatus.WAITING_ADMIN_APPROVAL.value,
                confirmationPrompt=command.confirmation_text,
                confirmationChallengeDigest=command.confirmation_challenge_digest,
                pendingActionId=resumed_run.pending_action_id,
            )
            observation_outcome = AttemptOutcome.WAITING
            self._record_successful_resume(
                observation,
                observation_started_ns,
            )
            return response
        except PreparedActionValidationError as caught:
            operation_error = caught
            observation_error = normalized_error_type(caught)
            raise CustomerConfirmationError(
                "prepared action evidence is invalid"
            ) from None
        except BaseException as caught:
            operation_error = caught
            observation_outcome = (
                AttemptOutcome.CANCELLED
                if isinstance(caught, asyncio.CancelledError)
                else AttemptOutcome.FAILED
            )
            observation_error = normalized_error_type(caught)
            raise
        finally:
            cleanup_error: BaseException | None = None
            if operation_error is not None:
                self._observability_runtime.record(
                    observation.scope if observation is not None else None,
                    HITLOperationRecordV1(
                        operation=OperationCode.HITL_CUSTOMER_RESUME,
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

    def _record_successful_resume(
        self,
        observation: AttemptObservationV1 | None,
        started_ns: int,
    ) -> None:
        scope = observation.scope if observation is not None else None
        duration_ms = max(0, (time.monotonic_ns() - started_ns) // 1_000_000)
        self._observability_runtime.record(
            scope,
            HITLOperationRecordV1(
                operation=OperationCode.HITL_CUSTOMER_RESUME,
                status=ObservabilityStatus.SUCCEEDED,
                duration_ms=duration_ms,
            ),
        )
        self._observability_runtime.record(
            scope,
            HITLOperationRecordV1(
                operation=OperationCode.HITL_ADMIN_INTERRUPT,
                status=ObservabilityStatus.WAITING,
                duration_ms=duration_ms,
            ),
        )

    async def _resolve_confirmation(
        self,
        actor: AuthenticatedUser,
        command: CustomerConfirmationCommand,
    ) -> _ResolvedConfirmation:
        try:
            async with self._unit_of_work.open(
                operation=UnitOfWorkOperation.AGENT_CUSTOMER_CONFIRMATION_RESOLVE
            ) as uow:
                run = await uow.store.customer_confirmation_run(
                    conversation_id=command.conversation_id,
                    subject_user_id=actor.user_id,
                )
                challenge = self._side_effect_policy.parse_exact_confirmation(run.confirmation_prompt)
                if challenge is None:
                    raise CustomerConfirmationStoreError("durable customer confirmation prompt is invalid")
        except CustomerConfirmationOwnershipError:
            raise CustomerConfirmationAccessDenied("customer confirmation access is denied") from None
        except CustomerConfirmationStoreError:
            raise CustomerConfirmationError("customer confirmation state is unavailable") from None
        if command.confirmation_text != run.confirmation_prompt:
            raise CustomerConfirmationError("customer confirmation does not match the waiting action")
        return _ResolvedConfirmation(
            run=run,
            action_type=challenge.action_type,
            target_order_no=challenge.target_order_no,
        )

    def _validate_frozen_confirmation(
        self,
        *,
        actor: AuthenticatedUser,
        command: CustomerConfirmationCommand,
        resolved: _ResolvedConfirmation,
        state: ConversationCheckpointState,
    ) -> None:
        active_run = state.active_run
        draft = active_run.action_draft if active_run is not None else None
        if draft is None:
            raise CustomerConfirmationError(
                "checkpoint durable action draft is unavailable"
            )
        try:
            expected_text = self._side_effect_policy.durable_confirmation_prompt(
                PolicyActionDraftSnapshot(**draft.model_dump(mode="python"))
            )
            expected_digest = (
                self._side_effect_policy.durable_confirmation_challenge_digest(
                    run_id=resolved.run.run_id,
                    subject_user_id=draft.subject_user_id,
                    action_type=draft.action_type,
                    target_order_id=draft.target_order_id,
                    target_order_no=draft.target_order_no,
                )
            )
        except SideEffectPolicyError:
            raise CustomerConfirmationError(
                "checkpoint customer confirmation evidence is invalid"
            ) from None
        if (
            actor.role != "CUSTOMER"
            or resolved.run.subject_user_id != actor.user_id
            or draft.subject_user_id != actor.user_id
            or resolved.action_type != draft.action_type
            or resolved.target_order_no != draft.target_order_no.upper()
            or command.confirmation_text != resolved.run.confirmation_prompt
            or command.confirmation_text != expected_text
            or draft.nonce_digest != expected_digest
            or command.confirmation_challenge_digest != expected_digest
        ):
            raise CustomerConfirmationError(
                "customer confirmation does not match the frozen action"
            )

    def _validate_formal_checkpoint(
        self,
        *,
        checkpoint_tuple: CheckpointTuple | None,
        execution: ExecutionScope,
        paused_attempt_id: str,
    ) -> _FormalCheckpoint:
        if checkpoint_tuple is None:
            raise CustomerConfirmationError("customer confirmation checkpoint is unavailable")
        values = checkpoint_tuple.checkpoint.get("channel_values")
        if not isinstance(values, Mapping):
            raise CustomerConfirmationError("checkpoint state is unavailable")
        if "__start__" in values:
            state = self._load_graph_state(values["__start__"])
        else:
            state = self._load_graph_state(values)
        active_run = state.active_run
        identity = state.conversation_identity
        if (
            execution.attempt_id == paused_attempt_id
            or identity.conversation_id != execution.conversation_id
            or identity.thread_id != execution.thread_id
            or identity.subject_user_id != execution.subject.user_id
            or active_run is None
            or active_run.run_id != execution.run_id
            or active_run.attempt_id != execution.attempt_id
            or active_run.side_effect_authorization is not None
        ):
            raise CustomerConfirmationError("checkpoint is not bound to the customer action run")
        interrupted: list[Interrupt] = []
        for _task_id, channel, value in checkpoint_tuple.pending_writes or ():
            if channel != INTERRUPT or type(value) is not tuple:
                continue
            interrupted.extend(item for item in value if isinstance(item, Interrupt))
        if (
            active_run.run_status is RunStatus.RUNNING
            and active_run.pending_action_id is None
            and not interrupted
        ):
            raise _CheckpointRecoveryRequired(
                "customer confirmation marker has no committed pause point"
            )
        if active_run.action_draft is None:
            raise CustomerConfirmationError(
                "checkpoint durable action draft is unavailable"
            )
        if active_run.run_status is RunStatus.RESUME_PENDING:
            if (
                active_run.customer_confirmation_status
                is not CustomerConfirmationStatus.CONFIRMED
                or active_run.pending_action_id is not None
                or interrupted
            ):
                raise CustomerConfirmationError(
                    "checkpoint customer ACK transition is invalid"
                )
            return _FormalCheckpoint(state=state, stage="CUSTOMER_ACKED")
        if active_run.run_status is RunStatus.WAITING_CUSTOMER_CONFIRMATION:
            expected_node = "customer_confirmation_gate:"
            expected_status = CustomerConfirmationStatus.PENDING
            stage: Literal["CUSTOMER_INTERRUPT", "ADMIN_INTERRUPT"] = (
                "CUSTOMER_INTERRUPT"
            )
            if active_run.pending_action_id is not None:
                raise CustomerConfirmationError(
                    "customer interrupt cannot reference a pending action"
                )
        elif active_run.run_status is RunStatus.WAITING_ADMIN_APPROVAL:
            expected_node = "admin_approval_gate:"
            expected_status = CustomerConfirmationStatus.CONFIRMED
            stage = "ADMIN_INTERRUPT"
            if active_run.pending_action_id is None:
                raise CustomerConfirmationError(
                    "admin interrupt requires a pending action"
                )
            if not interrupted:
                if active_run.customer_confirmation_status is not expected_status:
                    raise CustomerConfirmationError(
                        "admin approval state has invalid customer evidence"
                    )
                return _FormalCheckpoint(state=state, stage="ADMIN_READY")
        else:
            raise CustomerConfirmationError(
                "checkpoint is not a supported durable interrupt state"
            )
        if (
            active_run.customer_confirmation_status is not expected_status
            or len(interrupted) != 1
            or interrupted[0].resumable is not True
            or interrupted[0].when != "during"
            or interrupted[0].ns is None
            or len(interrupted[0].ns) != 1
            or not interrupted[0].ns[0].startswith(expected_node)
            or not isinstance(interrupted[0].value, Mapping)
            or interrupted[0].value.get("control") != "GENERIC_INTERRUPT_V1"
            or type(interrupted[0].value.get("control_id")) is not str
        ):
            if stage == "CUSTOMER_INTERRUPT" and not interrupted:
                raise _CheckpointRecoveryRequired(
                    "customer confirmation pause interrupt is incomplete"
                )
            raise CustomerConfirmationError(
                "checkpoint interrupt does not match the durable action stage"
            )
        return _FormalCheckpoint(state=state, stage=stage)

    async def _validate_current_order_and_confirmation(
        self,
        *,
        actor: AuthenticatedUser,
        command: CustomerConfirmationCommand,
        run: CustomerConfirmationRunSnapshot,
        state: ConversationCheckpointState,
    ) -> None:
        active_run = state.active_run
        assert active_run is not None and active_run.action_draft is not None
        try:
            async with self._unit_of_work.open(
                operation=UnitOfWorkOperation.AGENT_CUSTOMER_CONFIRMATION_REVALIDATE
            ) as uow:
                order = await uow.store.lock_customer_confirmation_order(
                    user_id=actor.user_id,
                    order_no=active_run.action_draft.target_order_no,
                )
                if order is None:
                    raise SideEffectPolicyError("customer confirmation target is unavailable")
                database_now = await uow.store.customer_confirmation_database_time()
                draft = PolicyActionDraftSnapshot(**active_run.action_draft.model_dump(mode="python"))
                self._side_effect_policy.validate_durable_customer_confirmation(
                    run_id=run.run_id,
                    user=actor,
                    order=order,
                    draft=draft,
                    confirmation_text=command.confirmation_text,
                    confirmation_challenge_digest=(command.confirmation_challenge_digest),
                    now=database_now,
                )
        except SideEffectPolicyError:
            raise CustomerConfirmationError("customer confirmation validation failed") from None

    async def _mark_confirmation_retryable(
        self,
        execution: ExecutionScope,
    ) -> None:
        async with self._unit_of_work.open(
            operation=UnitOfWorkOperation.AGENT_CUSTOMER_CONFIRMATION_RETRYABLE
        ) as uow:
            await uow.store.mark_customer_confirmation_retryable(
                scope=self._marker_scope(execution),
                conversation_id=execution.conversation_id,
                subject_user_id=execution.subject.user_id,
                run_id=execution.run_id,
            )

    async def _mark_admin_waiting(
        self,
        execution: ExecutionScope,
        *,
        pending_action_id: int,
    ) -> None:
        async with self._unit_of_work.open(
            operation=UnitOfWorkOperation.AGENT_CUSTOMER_CONFIRMATION_ADMIN_WAITING
        ) as uow:
            await uow.store.mark_admin_approval_waiting(
                scope=self._marker_scope(execution),
                conversation_id=execution.conversation_id,
                subject_user_id=execution.subject.user_id,
                run_id=execution.run_id,
                pending_action_id=pending_action_id,
            )

    def _marker_scope(self, execution: ExecutionScope) -> EffectWriteScope:
        fence = execution.lease.fence_token
        if type(fence) is not int or fence <= 0:
            raise CustomerConfirmationError(
                "customer confirmation marker requires a positive live fence"
            )
        execution.require_attempt_active()
        return EffectWriteScope(
            thread_id=execution.thread_id,
            attempt_id=execution.attempt_id,
            fence_version=fence,
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
                graph_result = await self._graph.ainvoke(graph_input, config)
            except asyncio.CancelledError:
                # A renewal failure records its authoritative error before it
                # cancels the graph.  External cancellation must still escape.
                if renewal_error is None:
                    raise
            except BaseException as caught:
                graph_error = caught
                execution.revoke_attempt()
            finally:
                stop_renewal.set()

        async def renew() -> None:
            nonlocal renewal_error
            try:
                while True:
                    async with asyncio.timeout(self._lease_renew_interval_seconds):
                        await stop_renewal.wait()
                    return
            except TimeoutError:
                while not stop_renewal.is_set():
                    try:
                        await self._durable_authority.renew_lease(
                            execution,
                            lease_duration=timedelta(seconds=self._lease_ttl_seconds),
                        )
                    except asyncio.CancelledError:
                        raise
                    except BaseException as caught:
                        renewal_error = caught
                        execution.revoke_attempt()
                        stop_renewal.set()
                        active_graph = graph_task
                        if active_graph is not None and not active_graph.done():
                            active_graph.cancel()
                        return
                    try:
                        async with asyncio.timeout(self._lease_renew_interval_seconds):
                            await stop_renewal.wait()
                    except TimeoutError:
                        continue
                    return

        try:
            async with asyncio.TaskGroup() as tasks:
                graph_task = tasks.create_task(
                    execute(),
                    name="customer-confirmation-graph",
                )
                tasks.create_task(
                    renew(),
                    name="customer-confirmation-lease-renewer",
                )
        except BaseException:
            execution.revoke_attempt()
            stop_renewal.set()
            raise
        finally:
            stop_renewal.set()
        if graph_error is not None:
            raise graph_error
        if renewal_error is not None:
            raise renewal_error
        if graph_result is None:
            raise CustomerConfirmationError("customer confirmation graph returned no state")
        return graph_result

    def _load_graph_state(
        self,
        raw: Mapping[str, object] | object,
    ) -> ConversationCheckpointState:
        if not isinstance(raw, Mapping):
            raise CustomerConfirmationError("checkpoint state is unavailable")
        required = (
            "schema_version",
            "conversation_identity",
            "memory",
            "active_run",
        )
        if not all(key in raw for key in required):
            raise CustomerConfirmationError("checkpoint state is incomplete")
        try:
            return load_checkpoint_state({key: raw[key] for key in required})
        except Exception:
            raise CustomerConfirmationError("checkpoint state is invalid") from None

    def _response(
        self,
        command: CustomerConfirmationCommand,
        resolved: _ResolvedConfirmation,
        *,
        pending_action_id: int,
    ) -> CustomerConfirmationResponse:
        return CustomerConfirmationResponse(
            conversationId=command.conversation_id,
            agentStatus=RunStatus.WAITING_ADMIN_APPROVAL.value,
            confirmationPrompt=resolved.run.confirmation_prompt,
            confirmationChallengeDigest=command.confirmation_challenge_digest,
            pendingActionId=pending_action_id,
        )


__all__ = [
    "CustomerConfirmationAccessDenied",
    "CustomerConfirmationApplicationService",
    "CustomerConfirmationCheckpointReaderPort",
    "CustomerConfirmationCommand",
    "CustomerConfirmationError",
    "CustomerConfirmationRetryable",
    "CustomerConfirmationGraphPort",
    "CustomerResumeCapabilityError",
    "CustomerResumeCapabilityProvider",
]
