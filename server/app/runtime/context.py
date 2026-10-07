from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol

from app.agent.thread_identity import ThreadIdentity, require_positive_identity
from app.runtime.uow import (
    ApplicationUnitOfWorkFactory,
    bind_application_unit_of_work_factory,
)


class RuntimeContextError(RuntimeError):
    """Runtime dependencies or execution identity are absent or inconsistent."""


class AttemptCapabilityRevoked(RuntimeContextError):
    """The application-owned authority to issue attempt writes was revoked."""


class AttemptWriteCapability:
    """Process-local, monotonic revocation gate shared by one attempt's writers."""

    __slots__ = ("_active",)

    def __init__(self) -> None:
        self._active = True

    def revoke(self) -> None:
        self._active = False

    def require_active(self) -> None:
        if not self._active:
            raise AttemptCapabilityRevoked("attempt write capability is revoked")


class RuntimeConfigReader(Protocol):
    async def read(self, unit_of_work: object) -> object: ...


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class ControlledResourceFactories:
    """Composition inputs for one attempt's controlled runtime resources.

    LLM/vector client factories were deliberately removed: they had zero
    production callers and would have handed out raw third-party clients that
    can outlive the attempt lifecycle. External clients must be reached only
    through explicit narrow ports owned by the application layer, never
    borrowed through the runtime context.
    """

    unit_of_work: ApplicationUnitOfWorkFactory[object]
    runtime_config: RuntimeConfigReader

    def bind(self, lifecycle: "_AttemptLifecycle") -> "_ControlledRuntimeResources":
        return _ControlledRuntimeResources(
            _unit_of_work=bind_application_unit_of_work_factory(
                self.unit_of_work,
                lifecycle,
            ),
            _runtime_config=_LifecycleBoundRuntimeConfigReader(
                self.runtime_config,
                lifecycle,
            ),
            _lifecycle=lifecycle,
        )


class _LifecycleBoundRuntimeConfigReader:
    def __init__(
        self,
        delegate: RuntimeConfigReader,
        lifecycle: "_AttemptLifecycle",
    ) -> None:
        self._delegate = delegate
        self._lifecycle = lifecycle

    async def read(self, unit_of_work: object) -> object:
        self._lifecycle.ensure_active()
        result = await self._delegate.read(unit_of_work)
        self._lifecycle.ensure_active()
        return result


@dataclass(frozen=True, slots=True)
class _ControlledRuntimeResources:
    _unit_of_work: ApplicationUnitOfWorkFactory[object]
    _runtime_config: RuntimeConfigReader
    _lifecycle: "_AttemptLifecycle" = field(repr=False, compare=False)

    @property
    def unit_of_work(self) -> ApplicationUnitOfWorkFactory[object]:
        self._lifecycle.ensure_available()
        return self._unit_of_work

    @property
    def runtime_config(self) -> RuntimeConfigReader:
        self._lifecycle.ensure_available()
        return self._runtime_config


@dataclass(frozen=True, slots=True)
class ActorIdentity:
    user_id: int
    username: str
    display_name: str
    role: str

    def __post_init__(self) -> None:
        require_positive_identity(self.user_id, field_name="actor_user_id")
        if not self.username or not self.display_name or not self.role:
            raise RuntimeContextError("actor identity fields must be non-empty")


@dataclass(frozen=True, slots=True)
class SubjectIdentity:
    user_id: int
    role_snapshot: str

    def __post_init__(self) -> None:
        require_positive_identity(self.user_id, field_name="subject_user_id")
        if not self.role_snapshot:
            raise RuntimeContextError("subject role snapshot must be non-empty")


@dataclass(frozen=True, slots=True)
class LeaseExecutionScope:
    owner_attempt_id: str
    fence_token: int | None = None

    def __post_init__(self) -> None:
        if not self.owner_attempt_id:
            raise RuntimeContextError("lease owner attempt must be non-empty")
        if self.fence_token is not None and (
            type(self.fence_token) is not int or self.fence_token < 0
        ):
            raise RuntimeContextError("fence token must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class ExecutionScope:
    conversation_id: int
    thread_id: str
    run_id: str
    attempt_id: str
    actor: ActorIdentity
    subject: SubjectIdentity
    lease: LeaseExecutionScope
    started_at: datetime
    attempt_capability: AttemptWriteCapability = field(
        default_factory=AttemptWriteCapability,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        ThreadIdentity.from_conversation_id(self.conversation_id).assert_matches(
            self.conversation_id,
            self.thread_id,
        )
        if not self.run_id or not self.attempt_id:
            raise RuntimeContextError("run and attempt identities must be non-empty")
        if self.lease.owner_attempt_id != self.attempt_id:
            raise RuntimeContextError("lease owner must match the current attempt")
        if self.started_at.tzinfo is None:
            raise RuntimeContextError("attempt start time must include a timezone")
        if self.actor.role == "CUSTOMER" and self.actor.user_id != self.subject.user_id:
            raise RuntimeContextError("customer actor must be the conversation subject")

    def require_attempt_active(self) -> None:
        self.attempt_capability.require_active()

    def revoke_attempt(self) -> None:
        self.attempt_capability.revoke()


@dataclass(frozen=True, slots=True)
class AgentRuntimeContext:
    _execution: ExecutionScope
    _resources: _ControlledRuntimeResources
    _clock: Clock
    _lifecycle: "_AttemptLifecycle" = field(default_factory=lambda: _AttemptLifecycle(), repr=False, compare=False)

    @property
    def execution(self) -> ExecutionScope:
        self._lifecycle.ensure_available()
        return self._execution

    @property
    def resources(self) -> _ControlledRuntimeResources:
        self._lifecycle.ensure_available()
        return self._resources

    @property
    def clock(self) -> Clock:
        self._lifecycle.ensure_available()
        return self._clock

    def ensure_active(self) -> None:
        self._lifecycle.ensure_active()
        self._execution.require_attempt_active()

    def revoke_attempt(self) -> None:
        self._execution.revoke_attempt()


class _AttemptLifecycle:
    def __init__(self) -> None:
        self._active = False
        self._used = False

    def activate(self) -> None:
        if self._used:
            raise RuntimeContextError("agent runtime context cannot be rebound after its lifecycle ended")
        self._used = True
        self._active = True

    def revoke(self) -> None:
        self._active = False

    def ensure_active(self) -> None:
        if not self._active:
            raise RuntimeContextError("agent runtime context is no longer active")

    def ensure_available(self) -> None:
        if self._used and not self._active:
            raise RuntimeContextError("agent runtime context is no longer active")


ResourceFactory = Callable[[], ControlledResourceFactories]
_CURRENT_RUNTIME: ContextVar[AgentRuntimeContext | None] = ContextVar(
    "agent_runtime_context",
    default=None,
)


class RuntimeContextProvider:
    """Build and bind one typed, non-locating runtime context per execution attempt."""

    def __init__(
        self,
        resource_factory: ResourceFactory | None = None,
        *,
        clock: Clock | None = None,
    ) -> None:
        self._resource_factory = resource_factory
        self._clock = clock or SystemClock()

    def resolve(
        self,
        *,
        conversation_id: int,
        thread_id: str,
        run_id: str,
        attempt_id: str,
        actor: ActorIdentity,
        subject: SubjectIdentity,
        fence_token: int | None = None,
        attempt_capability: AttemptWriteCapability | None = None,
        resource_factory: ResourceFactory | None = None,
    ) -> AgentRuntimeContext:
        execution = ExecutionScope(
            conversation_id=conversation_id,
            thread_id=thread_id,
            run_id=run_id,
            attempt_id=attempt_id,
            actor=actor,
            subject=subject,
            lease=LeaseExecutionScope(owner_attempt_id=attempt_id, fence_token=fence_token),
            started_at=self._clock.now(),
            attempt_capability=attempt_capability or AttemptWriteCapability(),
        )
        selected_factory = resource_factory or self._resource_factory
        if selected_factory is None:
            raise RuntimeContextError("controlled resource factory is required")
        lifecycle = _AttemptLifecycle()
        resources = selected_factory().bind(lifecycle)
        return AgentRuntimeContext(
            _execution=execution,
            _resources=resources,
            _clock=self._clock,
            _lifecycle=lifecycle,
        )

    def current(self) -> AgentRuntimeContext:
        runtime = _CURRENT_RUNTIME.get()
        if runtime is None:
            raise RuntimeContextError("agent runtime context is not bound")
        runtime.ensure_active()
        return runtime

    @contextmanager
    def bind(self, runtime: AgentRuntimeContext) -> Iterator[None]:
        runtime._lifecycle.activate()
        token: Token[AgentRuntimeContext | None] = _CURRENT_RUNTIME.set(runtime)
        try:
            yield
        finally:
            runtime.revoke_attempt()
            runtime._lifecycle.revoke()
            _CURRENT_RUNTIME.reset(token)
