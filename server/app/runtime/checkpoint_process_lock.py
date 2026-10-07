"""Operating-system process lock for one canonical checkpoint database.

The sidecar file is only a stable inode/handle on which the operating system
holds an advisory lock.  Its mere existence never grants or denies authority.
"""

from __future__ import annotations

import errno
import importlib
import os
from pathlib import Path
from typing import BinaryIO, Protocol, cast

from app.runtime.checkpoint_security import ValidatedCheckpointLocation


class _FcntlModule(Protocol):
    LOCK_EX: int
    LOCK_NB: int
    LOCK_UN: int

    def flock(self, descriptor: int, operation: int) -> None: ...


class CheckpointProcessLockError(RuntimeError):
    """Content-free process-lock diagnostic."""

    __slots__ = ("reason",)

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"checkpoint process lock failed: {reason}")


def lock_sidecar_path(canonical_path: Path) -> Path:
    """Return the deterministic lock inode bound to a canonical DB path."""

    return canonical_path.with_name(f"{canonical_path.name}.lock")


class OperatingSystemCheckpointProcessLock:
    """Exclusive non-blocking advisory lock implemented by the host OS."""

    __slots__ = ("_stream",)

    def __init__(self) -> None:
        self._stream: BinaryIO | None = None

    def acquire(self, location: ValidatedCheckpointLocation) -> None:
        if self._stream is not None:
            raise CheckpointProcessLockError("ALREADY_ACQUIRED")

        descriptor: int | None = None
        stream: BinaryIO | None = None
        reason: str | None = None
        try:
            descriptor = os.open(
                lock_sidecar_path(location.canonical_path),
                os.O_CREAT | os.O_RDWR,
                0o600,
            )
            stream = os.fdopen(descriptor, "r+b", buffering=0)
            descriptor = None
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            _lock_stream(stream)
        except OSError as exc:
            reason = (
                "LOCK_HELD"
                if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}
                else "LOCK_ACQUIRE_FAILED"
            )

        if reason is not None:
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
            elif descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            raise CheckpointProcessLockError(reason) from None

        if stream is None:  # pragma: no cover - defensive invariant
            raise CheckpointProcessLockError("LOCK_ACQUIRE_FAILED")
        self._stream = stream

    def release(self) -> None:
        stream = self._stream
        if stream is None:
            return
        self._stream = None

        failed = False
        try:
            stream.seek(0)
            _unlock_stream(stream)
        except OSError:
            failed = True
        finally:
            try:
                stream.close()
            except OSError:
                failed = True
        if failed:
            raise CheckpointProcessLockError("LOCK_RELEASE_FAILED") from None


def _lock_stream(stream: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        return

    fcntl = cast(_FcntlModule, importlib.import_module("fcntl"))
    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_stream(stream: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        return

    fcntl = cast(_FcntlModule, importlib.import_module("fcntl"))
    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


__all__ = [
    "CheckpointProcessLockError",
    "OperatingSystemCheckpointProcessLock",
    "lock_sidecar_path",
]
