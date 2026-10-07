"""Managed process lifecycle for the AsyncSqliteSaver provider.

This module owns storage initialization, readiness and shutdown. It does not
expose the saver or compile the Agent graph.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol, cast

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.runtime.checkpoint_process_lock import (
    CheckpointProcessLockError,
    OperatingSystemCheckpointProcessLock,
)
from app.runtime.checkpoint_security import (
    CheckpointFilesystemSecurityPort,
    CheckpointProcessLockPort,
    CheckpointSettingsView,
    CheckpointStorageError,
    revalidate_checkpoint_location,
    validate_checkpoint_storage,
)

DEFAULT_CHECKPOINT_SHUTDOWN_TIMEOUT_SECONDS = 5.0


class CheckpointRuntimeError(RuntimeError):
    """Stable, content-free lifecycle failure."""

    __slots__ = ("reason",)

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"checkpoint runtime failed: {reason}")


class CheckpointRuntimeState(str, Enum):
    NEW = "NEW"
    STARTING = "STARTING"
    READY = "READY"
    FAILED = "FAILED"
    DISABLED = "DISABLED"
    STOPPING = "STOPPING"
    SHUTDOWN_FAILED = "SHUTDOWN_FAILED"
    STOPPED = "STOPPED"


@dataclass(frozen=True, slots=True)
class CheckpointRuntimeStatus:
    state: CheckpointRuntimeState
    required: bool
    reason_code: str | None = None
    graph_wired: bool = False

    @property
    def provider_ready(self) -> bool:
        return self.state is CheckpointRuntimeState.READY

    @property
    def ready_for_requests(self) -> bool:
        if not self.required and self.state is CheckpointRuntimeState.DISABLED:
            return True
        return self.provider_ready and self.graph_checkpointer_wired

    @property
    def graph_checkpointer_wired(self) -> bool:
        return self.graph_wired

    @property
    def durable_resume_available(self) -> bool:
        return False

    def to_health_payload(self) -> dict[str, object]:
        return {
            "required": self.required,
            "state": self.state.value,
            "providerReady": self.provider_ready,
            "graphCheckpointerWired": self.graph_checkpointer_wired,
            "durableResumeAvailable": self.durable_resume_available,
            "reasonCode": self.reason_code,
        }


class CheckpointSaverLifecycle(Protocol):
    async def setup(self) -> None: ...


class CheckpointRuntimeLifecycle(Protocol):
    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    def status(self) -> CheckpointRuntimeStatus: ...

    def composition_handle(self) -> CheckpointRuntimeCompositionHandle: ...


class CheckpointRuntimeCompositionHandle(Protocol):
    """Narrow lifespan-only access to the already managed saver."""

    def checkpointer(self) -> CheckpointSaverLifecycle | None: ...

    def mark_graph_checkpointer_wired(self) -> None: ...


class CheckpointRuntimeFactory(Protocol):
    def __call__(
        self, settings: CheckpointSettingsView
    ) -> CheckpointRuntimeLifecycle: ...


class CheckpointSaverContextFactory(Protocol):
    def __call__(
        self, canonical_path: Path
    ) -> AbstractAsyncContextManager[CheckpointSaverLifecycle]: ...


@asynccontextmanager
async def _real_saver_context(
    canonical_path: Path,
) -> AsyncIterator[CheckpointSaverLifecycle]:
    async with AsyncSqliteSaver.from_conn_string(str(canonical_path)) as saver:
        yield saver


class DisabledCheckpointRuntime:
    """Explicit optional-backend state with zero filesystem side effects."""

    __slots__ = ()

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    def status(self) -> CheckpointRuntimeStatus:
        return CheckpointRuntimeStatus(
            state=CheckpointRuntimeState.DISABLED,
            required=False,
        )

    def composition_handle(self) -> CheckpointRuntimeCompositionHandle:
        return _DisabledCompositionHandle()


class _DisabledCompositionHandle:
    __slots__ = ()

    def checkpointer(self) -> None:
        return None

    def mark_graph_checkpointer_wired(self) -> None:
        raise CheckpointRuntimeError("CHECKPOINTER_DISABLED") from None


class _ManagedCompositionHandle:
    __slots__ = ("_runtime",)

    def __init__(self, runtime: ManagedCheckpointRuntime) -> None:
        self._runtime = runtime

    def checkpointer(self) -> CheckpointSaverLifecycle:
        return self._runtime._composition_checkpointer()

    def mark_graph_checkpointer_wired(self) -> None:
        self._runtime._mark_graph_checkpointer_wired()


class ManagedCheckpointRuntime:
    """Own exactly one saver context and one process lock for this process."""

    def __init__(
        self,
        settings: CheckpointSettingsView,
        *,
        repository_root: Path,
        security: CheckpointFilesystemSecurityPort | None,
        environ: Mapping[str, str] | None,
        process_lock_factory: Callable[[], CheckpointProcessLockPort],
        saver_context_factory: CheckpointSaverContextFactory,
        shutdown_timeout_seconds: float,
    ) -> None:
        if shutdown_timeout_seconds <= 0:
            raise ValueError("shutdown timeout must be positive")
        self._settings = settings
        self._repository_root = repository_root
        self._security = security
        self._environ = environ
        self._process_lock_factory = process_lock_factory
        self._saver_context_factory = saver_context_factory
        self._shutdown_timeout_seconds = shutdown_timeout_seconds
        self._state = CheckpointRuntimeState.NEW
        self._reason_code: str | None = None
        self._lock: CheckpointProcessLockPort | None = None
        self._saver_context: (
            AbstractAsyncContextManager[CheckpointSaverLifecycle] | None
        ) = None
        self._saver_context_entered = False
        self._saver: CheckpointSaverLifecycle | None = None
        self._close_task: asyncio.Task[bool | None] | None = None
        self._graph_wired = False

    async def start(self) -> None:
        if self._state is CheckpointRuntimeState.READY:
            return
        if self._state is not CheckpointRuntimeState.NEW:
            _raise_runtime_error("INVALID_START_STATE")

        self._state = CheckpointRuntimeState.STARTING
        self._reason_code = None
        failure_reason: str | None = None
        cancelled = False
        try:
            attestation = validate_checkpoint_storage(
                self._settings,
                security=self._security,
                environ=self._environ,
                repository_root=self._repository_root,
            )
            if attestation is None:  # pragma: no cover - builder invariant
                raise CheckpointStorageError("ATTESTATION_REQUIRED")
            location = revalidate_checkpoint_location(
                self._settings,
                attestation,
                security=self._security,
                environ=self._environ,
                repository_root=self._repository_root,
            )
            process_lock = self._process_lock_factory()
            process_lock.acquire(location)
            self._lock = process_lock

            saver_context = self._saver_context_factory(location.canonical_path)
            self._saver_context = saver_context
            saver = await saver_context.__aenter__()
            self._saver_context_entered = True
            self._saver = saver
            await saver.setup()
        except asyncio.CancelledError:
            cancelled = True
        except CheckpointStorageError as exc:
            failure_reason = exc.reason
        except CheckpointProcessLockError:
            failure_reason = "PROCESS_LOCK_UNAVAILABLE"
        except Exception:
            failure_reason = (
                "SAVER_SETUP_FAILED"
                if self._saver is not None
                else "SAVER_CREATE_FAILED"
            )

        if cancelled or failure_reason is not None:
            cleanup_reason = await self._cleanup_after_failed_start()
            self._state = CheckpointRuntimeState.FAILED
            self._reason_code = cleanup_reason or failure_reason or "START_CANCELLED"
            if cancelled:
                raise asyncio.CancelledError
            _raise_runtime_error(self._reason_code)

        self._state = CheckpointRuntimeState.READY

    async def stop(self) -> None:
        if self._state is CheckpointRuntimeState.STOPPED:
            return
        if self._state is CheckpointRuntimeState.NEW:
            self._state = CheckpointRuntimeState.STOPPED
            return
        if self._state is CheckpointRuntimeState.FAILED and not self._has_resources():
            return
        if self._state not in {
            CheckpointRuntimeState.READY,
            CheckpointRuntimeState.FAILED,
            CheckpointRuntimeState.SHUTDOWN_FAILED,
        }:
            _raise_runtime_error("INVALID_STOP_STATE")

        self._state = CheckpointRuntimeState.STOPPING
        failure_reason = await self._close_saver()
        if failure_reason is not None:
            self._state = CheckpointRuntimeState.SHUTDOWN_FAILED
            self._reason_code = failure_reason
            _raise_runtime_error(failure_reason)
        lock_reason = self._release_lock()
        if lock_reason is not None:
            self._state = CheckpointRuntimeState.SHUTDOWN_FAILED
            self._reason_code = lock_reason
            _raise_runtime_error(lock_reason)
        self._state = CheckpointRuntimeState.STOPPED
        self._reason_code = None

    def status(self) -> CheckpointRuntimeStatus:
        return CheckpointRuntimeStatus(
            state=self._state,
            required=True,
            reason_code=self._reason_code,
            graph_wired=self._graph_wired,
        )

    def composition_handle(self) -> CheckpointRuntimeCompositionHandle:
        return _ManagedCompositionHandle(self)

    def _composition_checkpointer(self) -> CheckpointSaverLifecycle:
        if self._state is not CheckpointRuntimeState.READY or self._saver is None:
            _raise_runtime_error("CHECKPOINTER_NOT_READY")
        return cast(CheckpointSaverLifecycle, self._saver)

    def _mark_graph_checkpointer_wired(self) -> None:
        if self._state is not CheckpointRuntimeState.READY or self._saver is None:
            _raise_runtime_error("CHECKPOINTER_NOT_READY")
        self._graph_wired = True

    def _has_resources(self) -> bool:
        return (
            self._lock is not None
            or self._saver_context is not None
            or self._close_task is not None
        )

    async def _cleanup_after_failed_start(self) -> str | None:
        close_reason = await self._close_saver()
        if close_reason is not None:
            return f"STARTUP_{close_reason}"
        return self._release_lock()

    async def _close_saver(self) -> str | None:
        saver_context = self._saver_context
        saver_context_entered = self._saver_context_entered
        if saver_context is None or not saver_context_entered:
            return None
        close_task = self._close_task
        if close_task is None or close_task.cancelled() or (
            close_task.done() and close_task.exception() is not None
        ):
            close_task = asyncio.create_task(
                saver_context.__aexit__(None, None, None),
                name="checkpoint-saver-close",
            )
            self._close_task = close_task
        try:
            await asyncio.wait_for(
                asyncio.shield(close_task),
                timeout=self._shutdown_timeout_seconds,
            )
        except TimeoutError:
            return "SAVER_CLOSE_TIMEOUT"
        except asyncio.CancelledError:
            return "SAVER_CLOSE_CANCELLED"
        except Exception:
            return "SAVER_CLOSE_FAILED"
        self._close_task = None
        self._saver_context = None
        self._saver_context_entered = False
        self._saver = None
        return None

    def _release_lock(self) -> str | None:
        process_lock = self._lock
        if process_lock is None:
            return None
        try:
            process_lock.release()
        except CheckpointProcessLockError:
            return "PROCESS_LOCK_RELEASE_FAILED"
        except Exception:
            return "PROCESS_LOCK_RELEASE_FAILED"
        self._lock = None
        return None


def build_checkpoint_runtime(
    settings: CheckpointSettingsView,
    *,
    repository_root: Path | None = None,
    security: CheckpointFilesystemSecurityPort | None = None,
    environ: Mapping[str, str] | None = None,
    process_lock_factory: Callable[[], CheckpointProcessLockPort] = (
        OperatingSystemCheckpointProcessLock
    ),
    saver_context_factory: CheckpointSaverContextFactory = _real_saver_context,
    shutdown_timeout_seconds: float = DEFAULT_CHECKPOINT_SHUTDOWN_TIMEOUT_SECONDS,
) -> CheckpointRuntimeLifecycle:
    if not settings.checkpoint_required:
        return DisabledCheckpointRuntime()
    root = repository_root or Path(__file__).resolve().parents[3]
    return ManagedCheckpointRuntime(
        settings,
        repository_root=root,
        security=security,
        environ=environ,
        process_lock_factory=process_lock_factory,
        saver_context_factory=saver_context_factory,
        shutdown_timeout_seconds=shutdown_timeout_seconds,
    )


def _raise_runtime_error(reason: str) -> None:
    raise CheckpointRuntimeError(reason) from None


__all__ = [
    "CheckpointRuntimeError",
    "CheckpointRuntimeFactory",
    "CheckpointRuntimeLifecycle",
    "CheckpointRuntimeCompositionHandle",
    "CheckpointRuntimeState",
    "CheckpointRuntimeStatus",
    "DisabledCheckpointRuntime",
    "ManagedCheckpointRuntime",
    "build_checkpoint_runtime",
]
