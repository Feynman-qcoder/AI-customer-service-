from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from app.memory import ContextPackageV1
from app.schemas.agent import AgentPlan


class GroundedAnswer(BaseModel):
    answer: str
    confidence_level: str
    need_human: bool
    cited_candidate_ids: list[str] = []


class LLMModelFamilyV1(StrEnum):
    OPENAI_COMPATIBLE = "OPENAI_COMPATIBLE"
    MOCK = "MOCK"


class LLMInvocationFailureCodeV1(StrEnum):
    TIMEOUT = "TIMEOUT"
    UNAVAILABLE = "UNAVAILABLE"
    CONFIGURATION = "CONFIGURATION"
    UNKNOWN = "UNKNOWN"


class LLMInvocationOutcomeV1(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    INVALID_RESPONSE = "INVALID_RESPONSE"


@dataclass(frozen=True, slots=True)
class LLMTokenUsageV1:
    prompt_tokens: int
    completion_tokens: int

    def __post_init__(self) -> None:
        if (
            type(self.prompt_tokens) is not int
            or self.prompt_tokens < 0
            or type(self.completion_tokens) is not int
            or self.completion_tokens < 0
        ):
            raise ValueError("LLM token usage is invalid")


_T = TypeVar("_T", covariant=True)


@dataclass(frozen=True, slots=True)
class LLMInvocationResultV1(Generic[_T]):
    value: _T
    duration_ms: int
    retry_count: int
    model_family: LLMModelFamilyV1
    usage: LLMTokenUsageV1 | None = None
    outcome: LLMInvocationOutcomeV1 = LLMInvocationOutcomeV1.SUCCEEDED

    def __post_init__(self) -> None:
        if type(self.duration_ms) is not int or self.duration_ms < 0:
            raise ValueError("LLM duration is invalid")
        if type(self.retry_count) is not int or self.retry_count < 0:
            raise ValueError("LLM retry count is invalid")
        if not isinstance(self.model_family, LLMModelFamilyV1):
            raise ValueError("LLM model family is invalid")
        if self.usage is not None and type(self.usage) is not LLMTokenUsageV1:
            raise ValueError("LLM token usage is invalid")
        if not isinstance(self.outcome, LLMInvocationOutcomeV1):
            raise ValueError("LLM invocation outcome is invalid")


@dataclass(frozen=True, slots=True)
class LLMInvocationFailureV1:
    duration_ms: int
    retry_count: int
    model_family: LLMModelFamilyV1
    code: LLMInvocationFailureCodeV1

    def __post_init__(self) -> None:
        if type(self.duration_ms) is not int or self.duration_ms < 0:
            raise ValueError("LLM duration is invalid")
        if type(self.retry_count) is not int or self.retry_count < 0:
            raise ValueError("LLM retry count is invalid")
        if not isinstance(self.model_family, LLMModelFamilyV1):
            raise ValueError("LLM model family is invalid")
        if not isinstance(self.code, LLMInvocationFailureCodeV1):
            raise ValueError("LLM failure code is invalid")


@dataclass(slots=True)
class LLMInvocationCaptureV1:
    results: list[LLMInvocationResultV1[object]]
    failures: list[LLMInvocationFailureV1]


_CURRENT_INVOCATION_CAPTURE: ContextVar[LLMInvocationCaptureV1 | None] = ContextVar(
    "llm_invocation_capture",
    default=None,
)


@contextmanager
def bind_llm_invocation_capture() -> Iterator[LLMInvocationCaptureV1]:
    capture = LLMInvocationCaptureV1(results=[], failures=[])
    token: Token[LLMInvocationCaptureV1 | None] = _CURRENT_INVOCATION_CAPTURE.set(capture)
    try:
        yield capture
    finally:
        _CURRENT_INVOCATION_CAPTURE.reset(token)


def current_llm_invocation_capture() -> LLMInvocationCaptureV1 | None:
    return _CURRENT_INVOCATION_CAPTURE.get()


def failure_from_provider_error(error: LLMProviderError) -> LLMInvocationFailureV1:
    if error.error_type == "MISSING_API_KEY":
        code = LLMInvocationFailureCodeV1.CONFIGURATION
    elif "TIMEOUT" in error.error_type.upper() or "TIMEOUT" in type(error).__name__.upper():
        code = LLMInvocationFailureCodeV1.TIMEOUT
    elif error.error_type == "PROVIDER_CALL_FAILED":
        code = LLMInvocationFailureCodeV1.UNAVAILABLE
    else:
        code = LLMInvocationFailureCodeV1.UNKNOWN
    return LLMInvocationFailureV1(
        duration_ms=error.duration_ms,
        retry_count=error.retry_count,
        model_family=error.model_family,
        code=code,
    )


class LLMProviderError(RuntimeError):
    def __init__(
        self,
        message: str,
        error_type: str = "LLM_PROVIDER_ERROR",
        *,
        duration_ms: int = 0,
        retry_count: int = 0,
        model_family: LLMModelFamilyV1 = LLMModelFamilyV1.OPENAI_COMPATIBLE,
    ) -> None:
        self.error_type = error_type
        self.duration_ms = duration_ms
        self.retry_count = retry_count
        self.model_family = model_family
        super().__init__(message)


class LLMClient(Protocol):
    async def plan(self, context: ContextPackageV1) -> AgentPlan | None:
        pass

    async def answer(
        self,
        context: ContextPackageV1,
        evidence: str,
        draft_answer: str,
    ) -> GroundedAnswer | None:
        pass


@runtime_checkable
class ObservedLLMClient(Protocol):
    """Optional metering extension implemented by production LLM providers.

    The base LLM port intentionally remains compatible with narrow deterministic
    adapters used by security tests and local callers.  When an adapter does not
    implement this extension, the application still measures the real wall-clock
    duration and keeps usage unknown instead of manufacturing token counts.
    """

    async def plan_observed(
        self,
        context: ContextPackageV1,
    ) -> LLMInvocationResultV1[AgentPlan | None]: ...

    async def answer_observed(
        self,
        context: ContextPackageV1,
        evidence: str,
        draft_answer: str,
    ) -> LLMInvocationResultV1[GroundedAnswer | None]: ...


async def invoke_plan_observed(
    client: LLMClient,
    context: ContextPackageV1,
    *,
    fallback_model_family: LLMModelFamilyV1,
) -> LLMInvocationResultV1[AgentPlan | None]:
    if isinstance(client, ObservedLLMClient):
        return await client.plan_observed(context)
    started = time.monotonic_ns()
    value = await client.plan(context)
    return LLMInvocationResultV1(
        value=value,
        duration_ms=max(0, (time.monotonic_ns() - started) // 1_000_000),
        retry_count=0,
        model_family=fallback_model_family,
        usage=None,
    )


async def invoke_answer_observed(
    client: LLMClient,
    context: ContextPackageV1,
    evidence: str,
    draft_answer: str,
    *,
    fallback_model_family: LLMModelFamilyV1,
) -> LLMInvocationResultV1[GroundedAnswer | None]:
    if isinstance(client, ObservedLLMClient):
        return await client.answer_observed(context, evidence, draft_answer)
    started = time.monotonic_ns()
    value = await client.answer(context, evidence, draft_answer)
    return LLMInvocationResultV1(
        value=value,
        duration_ms=max(0, (time.monotonic_ns() - started) // 1_000_000),
        retry_count=0,
        model_family=fallback_model_family,
        usage=None,
    )


__all__ = [
    "GroundedAnswer",
    "LLMClient",
    "LLMInvocationResultV1",
    "LLMInvocationCaptureV1",
    "LLMInvocationFailureCodeV1",
    "LLMInvocationFailureV1",
    "LLMInvocationOutcomeV1",
    "LLMModelFamilyV1",
    "LLMProviderError",
    "LLMTokenUsageV1",
    "ObservedLLMClient",
    "bind_llm_invocation_capture",
    "current_llm_invocation_capture",
    "failure_from_provider_error",
    "invoke_answer_observed",
    "invoke_plan_observed",
]
