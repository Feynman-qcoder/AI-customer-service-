import re
from dataclasses import dataclass

_THREAD_PATTERN = re.compile(r"conversation-([1-9][0-9]*)\Z")
_RUN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,63}\Z")
_MAX_IDENTITY = (1 << 63) - 1


class ThreadIdentityError(ValueError):
    """Raised when an untrusted value cannot identify a canonical conversation thread."""


def require_positive_identity(value: object, *, field_name: str) -> int:
    """Accept only a canonical positive integer identity, never coercible lookalikes."""
    if type(value) is not int or not 0 < value <= _MAX_IDENTITY:
        raise ThreadIdentityError(f"{field_name} must be a positive 64-bit integer")
    return value


def derive_thread_id(conversation_id: object) -> str:
    canonical_id = require_positive_identity(conversation_id, field_name="conversation_id")
    return f"conversation-{canonical_id}"


def derive_checkpoint_namespace(run_id: object) -> str:
    """Return the server-owned physical checkpoint namespace for one logical run."""

    if type(run_id) is not str or _RUN_PATTERN.fullmatch(run_id) is None:
        raise ThreadIdentityError("run_id must be a canonical server identifier")
    return f"run:{run_id}"


def parse_thread_id(thread_id: object) -> int:
    if type(thread_id) is not str:
        raise ThreadIdentityError("thread_id must be a canonical string")
    match = _THREAD_PATTERN.fullmatch(thread_id)
    if match is None:
        raise ThreadIdentityError("thread_id must use the canonical conversation namespace")
    conversation_id = require_positive_identity(int(match.group(1)), field_name="conversation_id")
    if derive_thread_id(conversation_id) != thread_id:
        raise ThreadIdentityError("thread_id is not canonical")
    return conversation_id


def reject_client_thread_override(thread_id: object | None) -> None:
    """Clients never select checkpoint namespaces, even when a supplied value looks valid."""
    if thread_id is not None:
        raise ThreadIdentityError("client-supplied thread_id is forbidden")


@dataclass(frozen=True, slots=True)
class ThreadIdentity:
    conversation_id: int
    thread_id: str

    @classmethod
    def from_conversation_id(cls, conversation_id: object) -> "ThreadIdentity":
        canonical_id = require_positive_identity(conversation_id, field_name="conversation_id")
        return cls(conversation_id=canonical_id, thread_id=derive_thread_id(canonical_id))

    @classmethod
    def parse(cls, thread_id: object) -> "ThreadIdentity":
        conversation_id = parse_thread_id(thread_id)
        return cls(conversation_id=conversation_id, thread_id=derive_thread_id(conversation_id))

    def assert_matches(self, conversation_id: object, thread_id: object) -> None:
        if self.conversation_id != require_positive_identity(conversation_id, field_name="conversation_id"):
            raise ThreadIdentityError("conversation identity mismatch")
        if self.thread_id != thread_id or parse_thread_id(thread_id) != self.conversation_id:
            raise ThreadIdentityError("thread identity mismatch")
