from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextvars import ContextVar, Token
from dataclasses import dataclass
from enum import StrEnum
from inspect import isawaitable
from types import TracebackType
from typing import Any, Generic, Literal, Protocol, TypeVar, cast

from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.runtime.durable import DurableContentionError

_StoreT = TypeVar("_StoreT")
_StoreCovariantT = TypeVar("_StoreCovariantT", covariant=True)
_ResultT = TypeVar("_ResultT")


class UnitOfWorkState(StrEnum):
    OPEN = "OPEN"
    BODY_RUNNING = "BODY_RUNNING"
    COMMITTING = "COMMITTING"
    COMMITTED = "COMMITTED"
    ROLLED_BACK = "ROLLED_BACK"
    OUTCOME_UNKNOWN = "OUTCOME_UNKNOWN"
    CLOSED = "CLOSED"


class UnitOfWorkOperation(StrEnum):
    """Central typed allowlist for application UoW operation codes.

    Only these fixed codes may appear in domain exceptions, logs, or telemetry.
    Caller-supplied operation strings are normalized onto this allowlist before
    a transaction starts; unknown values map to the fixed safe code
    ``operation.unspecified`` so raw strings never leak into any sink.
    """

    APPLICATION_TRANSACTION = "application.transaction"
    AGENT_CONTEXT = "agent.context"
    AGENT_BLOCKED = "agent.blocked"
    AGENT_PLANNER_PERSIST = "agent.planner.persist"
    AGENT_PLANNER_CONFIG = "agent.planner.config"
    AGENT_TOOL_AUDIT = "agent.tool_audit"
    AGENT_TOOL_STAGE_AUDIT = "agent.tool_stage.audit"
    AGENT_RESPONSE_GUARD_AUDIT = "agent.response_guard.audit"
    AGENT_FINALIZE = "agent.finalize"
    AGENT_RETRIEVAL_AUDIT = "agent.retrieval.audit"
    AGENT_RETRIEVAL_KEYWORD = "agent.retrieval.keyword"
    AGENT_RETRIEVAL_RULES = "agent.retrieval.rules"
    AGENT_RETRIEVAL_CONFIG = "agent.retrieval.config"
    AGENT_RETRIEVAL_REVALIDATE = "agent.retrieval.revalidate"
    AGENT_ANSWER_CONFIG = "agent.answer.config"
    AGENT_ANSWER_REVALIDATE = "agent.answer.revalidate"
    AGENT_R2_ACTION_PREPARE = "agent.r2_action_prepare"
    AGENT_CONFIRMATION_READ = "agent.confirmation.read"
    AGENT_CUSTOMER_CONFIRMATION_ADMISSION = "agent.customer_confirmation.admission"
    AGENT_CUSTOMER_CONFIRMATION_DATABASE_TIME = "agent.customer_confirmation.database_time"
    AGENT_CUSTOMER_CONFIRMATION_FREEZE = "agent.customer_confirmation.freeze"
    AGENT_CUSTOMER_CONFIRMATION_RESOLVE = "agent.customer_confirmation.resolve"
    AGENT_CUSTOMER_CONFIRMATION_REVALIDATE = "agent.customer_confirmation.revalidate"
    AGENT_CUSTOMER_CONFIRMATION_RESUMED = "agent.customer_confirmation.resumed"
    AGENT_CUSTOMER_CONFIRMATION_RETRYABLE = "agent.customer_confirmation.retryable"
    AGENT_CUSTOMER_CONFIRMATION_ADMIN_WAITING = "agent.customer_confirmation.admin_waiting"
    AGENT_ADMIN_ACTION_RESOLVE = "agent.admin_action.resolve"
    AGENT_ADMIN_ACTION_DECIDE = "agent.admin_action.decide"
    AGENT_ADMIN_RESUME_VALIDATE = "agent.admin_resume.validate"
    AGENT_ADMIN_RESUME_MARK = "agent.admin_resume.mark"
    AGENT_ADMIN_RECONCILE_SCAN = "agent.admin_reconcile.scan"
    AGENT_BUSINESS_EXECUTE = "agent.business_execute"
    AGENT_BUSINESS_EXECUTION_VERIFY = "agent.business_execute.verify"
    AGENT_ORDER_READ = "agent.order.read"
    AGENT_ORDERS_READ = "agent.orders.read"
    AGENT_PRODUCT_READ = "agent.product.read"
    AGENT_PRODUCT_SOURCES = "agent.product.sources"
    CHAT_CONVERSATION_CREATE = "chat.conversation.create"
    CHAT_CONVERSATION_REQUIRE_OWNED = "chat.conversation.require_owned"
    ATTEMPT_REGISTER = "attempt.register"
    RUN_BEGIN = "run.begin"
    LEASE_ACQUIRE = "lease.acquire"
    LEASE_RENEW = "lease.renew"
    LEASE_RELEASE = "lease.release"
    LEASE_READ = "lease.read"
    LEASE_REQUIRE_LIVE = "lease.require_live"
    PUBLICATION_PUBLISH = "publication.publish"
    PUBLICATION_READ = "publication.read"
    EFFECT_MESSAGE = "effect.message"
    EFFECT_AUDIT = "effect.audit"
    EFFECT_ACTION_PREPARE = "effect.action_prepare"
    UNSPECIFIED = "operation.unspecified"


def normalize_uow_operation(operation: str) -> UnitOfWorkOperation:
    """Map any caller-supplied operation string onto the fixed allowlist."""

    try:
        return UnitOfWorkOperation(operation)
    except ValueError:
        return UnitOfWorkOperation.UNSPECIFIED


class CommitOutcomeUnknown(RuntimeError):
    """A commit was dispatched but its outcome cannot safely be claimed."""

    retriable = False

    def __init__(self, *, operation: str) -> None:
        self.operation = normalize_uow_operation(operation).value
        super().__init__(f"database commit outcome is unknown for {self.operation}")


class UnitOfWorkClosedError(RuntimeError):
    """A transaction-bound store escaped its owning UoW."""


class LifecycleCapabilityGuard(Protocol):
    def ensure_active(self) -> None: ...


class TransactionBoundStoreGuard:
    def __init__(
        self,
        lifecycle_guard: LifecycleCapabilityGuard | None = None,
    ) -> None:
        self._lifecycle_guard = lifecycle_guard
        self._active = True

    def ensure_active(self) -> None:
        if self._lifecycle_guard is not None:
            self._lifecycle_guard.ensure_active()
        if not self._active:
            raise UnitOfWorkClosedError("transaction-bound store is no longer active")

    def revoke(self) -> None:
        self._active = False

    async def flush(self, session: AsyncSession) -> None:
        self.ensure_active()
        await session.flush()
        self.ensure_active()


class TransactionBoundStoreFactory(Protocol[_StoreCovariantT]):
    def __call__(
        self,
        session: AsyncSession,
        guard: TransactionBoundStoreGuard,
    ) -> _StoreCovariantT: ...


class ApplicationUnitOfWork(Protocol[_StoreCovariantT]):
    @property
    def store(self) -> _StoreCovariantT: ...

    @property
    def state(self) -> UnitOfWorkState: ...

    @property
    def outcome(self) -> UnitOfWorkState | None: ...

    async def __aenter__(self) -> ApplicationUnitOfWork[_StoreCovariantT]: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]: ...


class ApplicationUnitOfWorkFactory(Protocol[_StoreCovariantT]):
    def open(
        self,
        *,
        operation: str = "application.transaction",
    ) -> ApplicationUnitOfWork[_StoreCovariantT]: ...


_ACTIVE_DB_TRANSACTIONS: ContextVar[int] = ContextVar(
    "application_active_db_transactions",
    default=0,
)


def active_transaction_count() -> int:
    """Return the active DB transaction count for the current async context."""

    return _ACTIVE_DB_TRANSACTIONS.get()


def _bind_active_transaction() -> Token[int]:
    return _ACTIVE_DB_TRANSACTIONS.set(active_transaction_count() + 1)


def require_no_active_transaction(operation: str) -> None:
    if active_transaction_count() != 0:
        raise RuntimeError(f"{operation} cannot run while an application DB transaction is active")


class SqlAlchemyApplicationUnitOfWork(Generic[_StoreT]):
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        store_factory: TransactionBoundStoreFactory[_StoreT],
        *,
        operation: str,
        close_timeout_seconds: float,
        lifecycle_guard: LifecycleCapabilityGuard | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._store_factory = store_factory
        self._operation = operation
        self._close_timeout_seconds = close_timeout_seconds
        self._lifecycle_guard = lifecycle_guard
        self._session: AsyncSession | None = None
        self._guard: TransactionBoundStoreGuard | None = None
        self._transaction_token: Token[int] | None = None
        self._store: _StoreT | None = None
        self.state = UnitOfWorkState.OPEN
        self.outcome: UnitOfWorkState | None = None

    @property
    def store(self) -> _StoreT:
        if self._store is None:
            raise UnitOfWorkClosedError("transaction-bound store is not available")
        return self._store

    async def __aenter__(self) -> SqlAlchemyApplicationUnitOfWork[_StoreT]:
        if self.state is not UnitOfWorkState.OPEN:
            raise RuntimeError("a unit of work instance can be entered only once")
        session: AsyncSession | None = None
        session_factory = self._session_factory
        store_factory = self._store_factory
        try:
            self._ensure_lifecycle_active()
            session = session_factory()
            self._session = session
            self._ensure_lifecycle_active()
            await session.begin()
            self._ensure_lifecycle_active()
            self._transaction_token = _bind_active_transaction()
            self._guard = TransactionBoundStoreGuard(self._lifecycle_guard)
            self._ensure_lifecycle_active()
            self._store = store_factory(session, self._guard)
            self._ensure_lifecycle_active()
        except BaseException:
            await self._abort_partial_enter(session)
            del session, session_factory, store_factory
            raise
        self.state = UnitOfWorkState.BODY_RUNNING
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        del exc_type, traceback
        session = self._require_session()
        if exc is not None:
            rollback_confirmed = await self._rollback_without_masking(session)
            must_sanitize_unknown = isinstance(exc, DBAPIError) and not rollback_confirmed
            await self._close_and_revoke(session, preserve_exception=True)
            del session
            if must_sanitize_unknown:
                del exc
                _raise_commit_outcome_unknown(self._operation)
            return False

        try:
            self._ensure_lifecycle_active()
        except BaseException:
            await self._rollback_without_masking(session)
            await self._close_and_revoke(session, preserve_exception=True)
            raise

        self.state = UnitOfWorkState.COMMITTING
        cancellation: asyncio.CancelledError | None = None
        unknown = False
        try:
            await session.commit()
        except asyncio.CancelledError as caught:
            self.outcome = UnitOfWorkState.OUTCOME_UNKNOWN
            cancellation = caught
        except BaseException:
            self.outcome = UnitOfWorkState.OUTCOME_UNKNOWN
            unknown = True
        else:
            self.outcome = UnitOfWorkState.COMMITTED
            self.state = UnitOfWorkState.COMMITTED

        await self._close_and_revoke(
            session,
            preserve_exception=cancellation is not None or unknown,
        )
        del session
        if cancellation is not None:
            raise cancellation
        if unknown:
            _raise_commit_outcome_unknown(self._operation)
        return False

    async def _abort_partial_enter(self, session: AsyncSession | None) -> None:
        if self._guard is not None:
            self._guard.revoke()
        self._reset_transaction_counter()
        if session is not None:
            should_rollback = False
            try:
                should_rollback = bool(session.in_transaction())
            except BaseException:
                should_rollback = True
            if should_rollback:
                await self._rollback_without_masking(session)
            try:
                await self._bounded_close(session)
            except BaseException:
                pass
        self._session = None
        self._store = None
        self._guard = None
        self._session_factory = cast(Any, None)
        self._store_factory = cast(Any, None)
        self.state = UnitOfWorkState.CLOSED

    async def _rollback_without_masking(self, session: AsyncSession) -> bool:
        try:
            await session.rollback()
        except BaseException:
            self.outcome = UnitOfWorkState.OUTCOME_UNKNOWN
            self.state = UnitOfWorkState.OUTCOME_UNKNOWN
            return False
        else:
            self.outcome = UnitOfWorkState.ROLLED_BACK
            self.state = UnitOfWorkState.ROLLED_BACK
            return True

    async def _close_and_revoke(
        self,
        session: AsyncSession,
        *,
        preserve_exception: bool,
    ) -> None:
        if self._guard is not None:
            self._guard.revoke()
        self._reset_transaction_counter()
        close_error: BaseException | None = None
        try:
            await self._bounded_close(session)
        except BaseException as caught:
            close_error = caught
        finally:
            self._session = None
            self._store = None
            self._guard = None
            self._session_factory = cast(Any, None)
            self._store_factory = cast(Any, None)
            self.state = UnitOfWorkState.CLOSED
        if close_error is not None and not preserve_exception:
            raise close_error

    async def _bounded_close(self, session: AsyncSession) -> None:
        async with asyncio.timeout(self._close_timeout_seconds):
            await session.close()

    def _require_session(self) -> AsyncSession:
        if self._session is None:
            raise RuntimeError("unit of work was not entered")
        return self._session

    def _ensure_lifecycle_active(self) -> None:
        if self._lifecycle_guard is not None:
            self._lifecycle_guard.ensure_active()

    def _reset_transaction_counter(self) -> None:
        if self._transaction_token is not None:
            _ACTIVE_DB_TRANSACTIONS.reset(self._transaction_token)
            self._transaction_token = None


def _raise_commit_outcome_unknown(operation: str) -> None:
    sanitized = CommitOutcomeUnknown(operation=operation)
    try:
        raise sanitized
    except CommitOutcomeUnknown:
        # While the original DBAPIError is still the active exception (the
        # body-failure path inside __aexit__), a raise would implicitly chain
        # it as __context__, and ``raise ... from None`` only suppresses the
        # display of that link without clearing it. Strip the link here and
        # re-raise bare: a bare re-raise does not re-run implicit chaining, so
        # the propagated exception keeps __context__ and __cause__ as None and
        # holds no reference to the original exception, session, SQL, or
        # parameters.
        sanitized.__context__ = None
        raise


class SqlAlchemyApplicationUnitOfWorkFactory(Generic[_StoreT]):
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        store_factory: TransactionBoundStoreFactory[_StoreT],
        *,
        close_timeout_seconds: float = 1.0,
    ) -> None:
        if close_timeout_seconds <= 0:
            raise ValueError("close_timeout_seconds must be positive")
        self._session_factory = session_factory
        self._store_factory = store_factory
        self._close_timeout_seconds = close_timeout_seconds

    def open(
        self,
        *,
        operation: str = "application.transaction",
    ) -> SqlAlchemyApplicationUnitOfWork[_StoreT]:
        return self._open_with_lifecycle_guard(
            operation=operation,
            lifecycle_guard=None,
        )

    def _open_with_lifecycle_guard(
        self,
        *,
        operation: str,
        lifecycle_guard: LifecycleCapabilityGuard | None,
    ) -> SqlAlchemyApplicationUnitOfWork[_StoreT]:
        return SqlAlchemyApplicationUnitOfWork(
            self._session_factory,
            self._store_factory,
            operation=normalize_uow_operation(operation).value,
            close_timeout_seconds=self._close_timeout_seconds,
            lifecycle_guard=lifecycle_guard,
        )


class _RevocableStoreCapability(Generic[_StoreT]):
    def __init__(
        self,
        target: _StoreT,
        lifecycle_guard: LifecycleCapabilityGuard,
    ) -> None:
        self._target = target
        self._lifecycle_guard = lifecycle_guard

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        self._lifecycle_guard.ensure_active()
        attribute = getattr(self._target, name)
        if not callable(attribute):
            return attribute

        def guarded_call(*args: object, **kwargs: object) -> object:
            self._lifecycle_guard.ensure_active()
            result = attribute(*args, **kwargs)
            if isawaitable(result):

                async def guarded_await() -> object:
                    value = await result
                    self._lifecycle_guard.ensure_active()
                    return value

                return guarded_await()
            self._lifecycle_guard.ensure_active()
            return result

        return guarded_call


class _LifecycleBoundApplicationUnitOfWork(Generic[_StoreT]):
    def __init__(
        self,
        delegate: ApplicationUnitOfWork[_StoreT],
        lifecycle_guard: LifecycleCapabilityGuard,
    ) -> None:
        self._delegate = delegate
        self._lifecycle_guard = lifecycle_guard
        self._store: _StoreT | None = None

    @property
    def store(self) -> _StoreT:
        self._lifecycle_guard.ensure_active()
        if self._store is None:
            raise UnitOfWorkClosedError("transaction-bound store is not available")
        return self._store

    @property
    def state(self) -> UnitOfWorkState:
        return self._delegate.state

    @property
    def outcome(self) -> UnitOfWorkState | None:
        return self._delegate.outcome

    async def __aenter__(self) -> _LifecycleBoundApplicationUnitOfWork[_StoreT]:
        self._lifecycle_guard.ensure_active()
        entered = await self._delegate.__aenter__()
        try:
            self._lifecycle_guard.ensure_active()
        except BaseException as lifecycle_error:
            await self._delegate.__aexit__(
                type(lifecycle_error),
                lifecycle_error,
                lifecycle_error.__traceback__,
            )
            raise
        self._store = cast(
            _StoreT,
            _RevocableStoreCapability(entered.store, self._lifecycle_guard),
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        if exc is None:
            try:
                self._lifecycle_guard.ensure_active()
            except BaseException as lifecycle_error:
                try:
                    await self._delegate.__aexit__(
                        type(lifecycle_error),
                        lifecycle_error,
                        lifecycle_error.__traceback__,
                    )
                finally:
                    self._store = None
                raise
        try:
            return await self._delegate.__aexit__(exc_type, exc, traceback)
        finally:
            self._store = None


class _LifecycleBoundApplicationUnitOfWorkFactory(Generic[_StoreT]):
    def __init__(
        self,
        delegate: ApplicationUnitOfWorkFactory[_StoreT],
        lifecycle_guard: LifecycleCapabilityGuard,
    ) -> None:
        self._delegate = delegate
        self._lifecycle_guard = lifecycle_guard

    def open(
        self,
        *,
        operation: str = "application.transaction",
    ) -> ApplicationUnitOfWork[_StoreT]:
        self._lifecycle_guard.ensure_active()
        normalized_operation = normalize_uow_operation(operation).value
        opened: ApplicationUnitOfWork[_StoreT]
        if isinstance(self._delegate, SqlAlchemyApplicationUnitOfWorkFactory):
            opened = self._delegate._open_with_lifecycle_guard(
                operation=normalized_operation,
                lifecycle_guard=self._lifecycle_guard,
            )
        else:
            opened = self._delegate.open(operation=normalized_operation)
        return _LifecycleBoundApplicationUnitOfWork(
            opened,
            self._lifecycle_guard,
        )


def bind_application_unit_of_work_factory(
    factory: ApplicationUnitOfWorkFactory[Any],
    lifecycle_guard: LifecycleCapabilityGuard,
) -> ApplicationUnitOfWorkFactory[Any]:
    return _LifecycleBoundApplicationUnitOfWorkFactory(
        factory,
        lifecycle_guard,
    )


@dataclass(frozen=True, slots=True)
class SafeMysqlContention:
    operation: str
    attempt_count: int
    mysql_error_code: int


class MysqlContentionClassifier(Protocol):
    def classify(
        self,
        exc: DBAPIError,
        *,
        operation: str,
        attempt_count: int,
    ) -> SafeMysqlContention | None: ...


@dataclass(frozen=True, slots=True)
class ApplicationRetryPolicy:
    max_attempts: int = 3
    initial_backoff_seconds: float = 0.01
    maximum_backoff_seconds: float = 0.05

    def __post_init__(self) -> None:
        if type(self.max_attempts) is not int or self.max_attempts <= 0:
            raise ValueError("max_attempts must be a positive integer")
        if self.initial_backoff_seconds < 0:
            raise ValueError("initial_backoff_seconds must be non-negative")
        if self.maximum_backoff_seconds < self.initial_backoff_seconds:
            raise ValueError("maximum_backoff_seconds must be at least initial_backoff_seconds")


class ApplicationTransactionCoordinator:
    """Own bounded retries; every retry opens a fresh application UoW."""

    def __init__(
        self,
        classifier: MysqlContentionClassifier,
        *,
        policy: ApplicationRetryPolicy | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._classifier = classifier
        self._policy = policy or ApplicationRetryPolicy()
        self._sleep = sleep

    async def run(
        self,
        operation: str,
        factory: ApplicationUnitOfWorkFactory[_StoreT],
        work: Callable[[_StoreT], Awaitable[_ResultT]],
    ) -> _ResultT:
        operation_code = normalize_uow_operation(operation).value
        for attempt_number in range(1, self._policy.max_attempts + 1):
            contention: SafeMysqlContention | None = None
            unknown_operation: str | None = None
            unit_of_work: ApplicationUnitOfWork[_StoreT] | None = None
            try:
                async with factory.open(operation=operation_code) as opened:
                    unit_of_work = opened
                    result = await work(opened.store)
                return result
            except CommitOutcomeUnknown as caught:
                unknown_operation = caught.operation
            except DBAPIError as exc:
                outcome = unit_of_work.outcome if unit_of_work is not None else None
                contention = self.retry_after_rollback(
                    exc,
                    operation=operation_code,
                    attempt_count=attempt_number,
                    outcome=outcome,
                )
                if contention is None:
                    raise
            if unknown_operation is not None:
                del factory, work, unit_of_work, opened
                _raise_commit_outcome_unknown(unknown_operation)
            if contention is None:
                raise AssertionError("contention classification lost its safe fields")
            if contention.attempt_count == self._policy.max_attempts:
                raise DurableContentionError(
                    operation=contention.operation,
                    attempt_count=contention.attempt_count,
                    mysql_error_code=contention.mysql_error_code,
                )
            await self.wait_before_retry(contention)
        raise AssertionError("bounded application retry loop did not return or raise")

    @property
    def max_attempts(self) -> int:
        return self._policy.max_attempts

    def retry_after_rollback(
        self,
        exc: DBAPIError,
        *,
        operation: str,
        attempt_count: int,
        outcome: UnitOfWorkState | None,
    ) -> SafeMysqlContention | None:
        if outcome is not UnitOfWorkState.ROLLED_BACK:
            return None
        contention = self._classifier.classify(
            exc,
            operation=operation,
            attempt_count=attempt_count,
        )
        if contention is None:
            return None
        return contention

    async def wait_before_retry(self, contention: SafeMysqlContention) -> None:
        await self._sleep(self._backoff_seconds(contention.attempt_count))

    def _backoff_seconds(self, failed_attempt_number: int) -> float:
        delay = self._policy.initial_backoff_seconds * (2.0 ** (failed_attempt_number - 1))
        return min(delay, self._policy.maximum_backoff_seconds)
