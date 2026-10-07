from __future__ import annotations

from typing import Literal

TOKEN_COUNTER_VERSION_V1: Literal["UTF8_BYTES_CEIL_DIV_3_V1"] = (
    "UTF8_BYTES_CEIL_DIV_3_V1"
)
TOKEN_COUNTER_V1_ALGORITHM: Literal["ceil(utf8_byte_length / 3)"] = (
    "ceil(utf8_byte_length / 3)"
)


class MemoryTokenCounterError(ValueError):
    """Sanitized failure at the single versioned memory token boundary."""


class UnsupportedTokenCounterVersion(MemoryTokenCounterError):
    pass


def count_memory_tokens(
    text: str,
    *,
    version: str = TOKEN_COUNTER_VERSION_V1,
) -> int:
    """Count one text with the frozen V1 UTF-8 byte estimator."""

    if type(version) is not str or version != TOKEN_COUNTER_VERSION_V1:
        raise UnsupportedTokenCounterVersion("memory token counter version is unsupported")
    if type(text) is not str:
        raise MemoryTokenCounterError("memory token counter requires strict text")
    try:
        byte_length = len(text.encode("utf-8", errors="strict"))
    except UnicodeEncodeError:
        raise MemoryTokenCounterError("memory token counter rejected invalid text") from None
    return (byte_length + 2) // 3


__all__ = [
    "MemoryTokenCounterError",
    "TOKEN_COUNTER_V1_ALGORITHM",
    "TOKEN_COUNTER_VERSION_V1",
    "UnsupportedTokenCounterVersion",
    "count_memory_tokens",
]
