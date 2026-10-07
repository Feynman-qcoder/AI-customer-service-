"""Process-local non-queuing single-flight capabilities keyed by thread id."""

from __future__ import annotations

import asyncio
import secrets
from dataclasses import dataclass


class ThreadSingleFlightConflict(RuntimeError):
    status_code = 409


@dataclass(slots=True)
class ThreadSingleFlightCapability:
    _owner: PerThreadSingleFlight
    _thread_id: str
    _nonce: str
    _released: bool = False

    async def release(self) -> None:
        if self._released:
            return
        await self._owner._release(self._thread_id, self._nonce)
        self._released = True

    async def __aenter__(self) -> ThreadSingleFlightCapability:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        del exc_type, exc, traceback
        await self.release()


class PerThreadSingleFlight:
    """Acquire is deliberately non-waiting; there is no hidden per-thread queue."""

    def __init__(self) -> None:
        self._guard = asyncio.Lock()
        self._owners: dict[str, str] = {}

    async def acquire(self, thread_id: str) -> ThreadSingleFlightCapability:
        if type(thread_id) is not str or not thread_id or len(thread_id) > 128:
            raise ValueError("single-flight thread identity is invalid")
        nonce = secrets.token_hex(16)
        async with self._guard:
            if thread_id in self._owners:
                raise ThreadSingleFlightConflict(
                    "conversation already has an active execution"
                )
            self._owners[thread_id] = nonce
        return ThreadSingleFlightCapability(self, thread_id, nonce)

    async def _release(self, thread_id: str, nonce: str) -> None:
        async with self._guard:
            if self._owners.get(thread_id) != nonce:
                raise RuntimeError("single-flight capability no longer owns the thread")
            del self._owners[thread_id]

    async def active_count(self) -> int:
        async with self._guard:
            return len(self._owners)


__all__ = [
    "PerThreadSingleFlight",
    "ThreadSingleFlightCapability",
    "ThreadSingleFlightConflict",
]
