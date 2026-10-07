"""Versioned, deterministic sensitive-text classification for protected sinks.

The classifier is deliberately narrow: it recognizes a frozen set of high-risk
text shapes and returns only stable reason codes.  It does not transform input,
retain values, or claim comprehensive natural-language prompt-injection
detection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Protocol

SENSITIVE_TEXT_CLASSIFIER_VERSION_V1: Literal["SENSITIVE_TEXT_CLASSIFIER_V1"] = (
    "SENSITIVE_TEXT_CLASSIFIER_V1"
)


class SensitiveTextReason(StrEnum):
    CREDENTIAL = "SENSITIVE_CREDENTIAL"
    AUTH_TOKEN = "SENSITIVE_AUTH_TOKEN"
    URL_CREDENTIAL = "SENSITIVE_URL_CREDENTIAL"
    PHONE = "SENSITIVE_PHONE"
    ADDRESS = "SENSITIVE_ADDRESS"
    PROMPT_CONTROL = "SENSITIVE_PROMPT_CONTROL"
    APPROVAL_BYPASS = "SENSITIVE_APPROVAL_BYPASS"
    TOOL_PAYLOAD = "SENSITIVE_TOOL_PAYLOAD"
    UNKNOWN_STRUCTURE = "SENSITIVE_UNKNOWN_STRUCTURE"


class SensitiveTextClassifierError(ValueError):
    """A sanitized classifier contract failure with no source text."""

    __slots__ = ("reason",)

    def __init__(self, reason: str = "SENSITIVE_CLASSIFIER_FAILED") -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class SensitiveTextClassificationV1:
    classifier_version: Literal["SENSITIVE_TEXT_CLASSIFIER_V1"]
    is_sensitive: bool
    reason: SensitiveTextReason | None

    def __post_init__(self) -> None:
        if self.classifier_version != SENSITIVE_TEXT_CLASSIFIER_VERSION_V1:
            raise SensitiveTextClassifierError()
        if type(self.is_sensitive) is not bool:
            raise SensitiveTextClassifierError()
        if self.is_sensitive != (self.reason is not None):
            raise SensitiveTextClassifierError()


class SensitiveTextClassifierPort(Protocol):
    def classify(self, text: str) -> SensitiveTextClassificationV1: ...


_AUTH_TOKEN_PATTERNS = (
    re.compile(r"(?i)(?<![A-Za-z0-9])bearer[ \t]+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(
        r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{4,}"
        r"\.[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}(?![A-Za-z0-9_-])"
    ),
)
_URL_CREDENTIAL_PATTERN = re.compile(
    r"(?i)\b[a-z][a-z0-9+.-]{1,15}://[^\s/:@]+:[^\s/@]+@"
)
_CREDENTIAL_ASSIGNMENT_PATTERN = re.compile(
    r"(?ix)"
    r"(?<![A-Za-z0-9_])"
    r"[\"']?"
    r"(?:password|passwd|credential|secret|api[ _-]*key|authorization|"
    r"access[ _-]*token|refresh[ _-]*token|密码|密钥|授权|访问令牌|刷新令牌)"
    r"[\"']?\s*[:=：]\s*[\"']?[^\s,;}\]\"']+"
)
_PHONE_PATTERN = re.compile(r"(?<!\d)1[3-9](?:[ -]?\d){9}(?!\d)")
_CHINESE_ADDRESS_PATTERN = re.compile(
    r"(?:[\u4e00-\u9fff]{2,}(?:省|自治区))?"
    r"[\u4e00-\u9fff]{2,}市"
    r"[\u4e00-\u9fff]{1,}(?:区|县)"
    r"[\u4e00-\u9fff0-9]{1,40}(?:街道|路|街|巷)"
    r"[\u4e00-\u9fff0-9-]{0,30}(?:号|室|栋|单元)"
)
_ENGLISH_ADDRESS_PATTERN = re.compile(
    r"(?i)\b\d{1,6}\s+(?:[A-Za-z0-9.'-]+\s+){1,6}"
    r"(?:street|st|road|rd|avenue|ave|boulevard|blvd|lane|ln)\b"
)
_APPROVAL_BYPASS_PATTERNS = (
    re.compile(
        r"(?i)\b(?:bypass|skip|evade|circumvent)\b.{0,48}"
        r"\b(?:approval|authorization|policy|safety|security)\b"
    ),
    re.compile(r"(?:绕过|跳过|规避).{0,24}(?:审批|授权|策略|规则|安全门|安全检查)"),
)
_TOOL_PAYLOAD_PATTERNS = (
    re.compile(r"(?i)\btool[ _-]*(?:call|calls|payload|arguments?|args)\b"),
    re.compile(r"(?:工具调用|调用工具|工具载荷|工具负载|工具参数|调用参数)"),
)
_PROMPT_CONTROL_PATTERNS = (
    re.compile(
        r"(?i)\b(?:ignore|disregard|override|replace|reveal|leak)\b.{0,80}"
        r"\b(?:system[ _-]*prompt|previous[ _-]*(?:instructions|rules)|"
        r"prior[ _-]*(?:instructions|rules)|instructions|rules)\b"
    ),
    re.compile(
        r"(?:忽略|无视|覆盖|改写|替换|泄露|显示).{0,40}"
        r"(?:系统提示词|系统提示|隐藏指令|之前的指令|先前规则|系统规则|审批规则)"
    ),
)


class SensitiveTextClassifierV1:
    """Frozen stateless classifier safe to share across concurrent calls."""

    __slots__ = ()

    def classify(self, text: str) -> SensitiveTextClassificationV1:
        if type(text) is not str:
            raise SensitiveTextClassifierError()
        reason = _classify_sensitive_reason(text)
        return SensitiveTextClassificationV1(
            classifier_version=SENSITIVE_TEXT_CLASSIFIER_VERSION_V1,
            is_sensitive=reason is not None,
            reason=reason,
        )


def _classify_sensitive_reason(text: str) -> SensitiveTextReason | None:
    if any(pattern.search(text) is not None for pattern in _AUTH_TOKEN_PATTERNS):
        return SensitiveTextReason.AUTH_TOKEN
    if _URL_CREDENTIAL_PATTERN.search(text) is not None:
        return SensitiveTextReason.URL_CREDENTIAL
    if _CREDENTIAL_ASSIGNMENT_PATTERN.search(text) is not None:
        return SensitiveTextReason.CREDENTIAL
    if _PHONE_PATTERN.search(text) is not None:
        return SensitiveTextReason.PHONE
    if (
        _CHINESE_ADDRESS_PATTERN.search(text) is not None
        or _ENGLISH_ADDRESS_PATTERN.search(text) is not None
    ):
        return SensitiveTextReason.ADDRESS
    if any(pattern.search(text) is not None for pattern in _APPROVAL_BYPASS_PATTERNS):
        return SensitiveTextReason.APPROVAL_BYPASS
    if any(pattern.search(text) is not None for pattern in _TOOL_PAYLOAD_PATTERNS):
        return SensitiveTextReason.TOOL_PAYLOAD
    if any(pattern.search(text) is not None for pattern in _PROMPT_CONTROL_PATTERNS):
        return SensitiveTextReason.PROMPT_CONTROL
    stripped = text.strip()
    if (
        (stripped.startswith("{") and stripped.endswith("}"))
        or (stripped.startswith("[") and stripped.endswith("]"))
        or stripped.casefold().startswith("```json")
    ):
        return SensitiveTextReason.UNKNOWN_STRUCTURE
    return None


__all__ = [
    "SENSITIVE_TEXT_CLASSIFIER_VERSION_V1",
    "SensitiveTextClassificationV1",
    "SensitiveTextClassifierError",
    "SensitiveTextClassifierPort",
    "SensitiveTextClassifierV1",
    "SensitiveTextReason",
]
