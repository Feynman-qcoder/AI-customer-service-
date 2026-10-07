import asyncio
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import ValidationError

from app.core.config import settings
from app.llm.base import (
    GroundedAnswer,
    LLMInvocationOutcomeV1,
    LLMInvocationResultV1,
    LLMModelFamilyV1,
    LLMProviderError,
    LLMTokenUsageV1,
)
from app.memory import ContextPackageV1, ContextPurpose
from app.schemas.agent import AgentPlan

logger = logging.getLogger(__name__)

_AGENT_PLAN_FIELDS = frozenset(
    {
        "intent",
        "goal",
        "order_reference",
        "product_reference",
        "required_tools",
        "action_type",
        "risk_level",
        "requires_confirmation",
        "missing_information",
        "decision_reason",
    }
)
_ORDER_REFERENCE_FIELDS = frozenset(
    {"order_no", "ordinal_index", "product_keyword", "latest", "list_all"}
)
_GROUNDED_ANSWER_FIELDS = frozenset(
    {"answer", "confidence_level", "need_human", "cited_candidate_ids"}
)
_ALLOWED_ACTION_TYPES = frozenset(
    {None, "ORDER_CANCELLATION", "REFUND", "CREATE_SUPPORT_TICKET"}
)
_ALLOWED_CONFIDENCE_LEVELS = frozenset({"HIGH", "MEDIUM", "LOW"})
_SAFE_FIELD_NAMES = (
    _AGENT_PLAN_FIELDS | _ORDER_REFERENCE_FIELDS | _GROUNDED_ANSWER_FIELDS
)
_SAFE_FINISH_REASONS = frozenset(
    {
        "stop",
        "length",
        "content_filter",
        "tool_calls",
        "insufficient_system_resource",
        "aborted",
    }
)


@dataclass(frozen=True, slots=True)
class _ChatResultV1:
    content: str
    finish_reason: object
    duration_ms: int
    retry_count: int
    usage: LLMTokenUsageV1 | None


class _StructuredOutputValidationError(ValueError):
    def __init__(self, *issues: tuple[str, str]) -> None:
        self.issues = issues or (("$", "invalid_structure"),)
        super().__init__("structured output validation failed")


class OpenAICompatibleLLMClient:
    def __init__(self, temperature: float) -> None:
        self.temperature = temperature

    async def plan(self, context: ContextPackageV1) -> AgentPlan | None:
        messages = _metered_messages(context, ContextPurpose.PLANNER)
        payload = _structured_payload(messages=messages, temperature=self.temperature)
        started = time.monotonic_ns()
        result = _coerce_chat_result(await self._chat(payload), started_ns=started)
        return _validated_structured_value(
            result,
            schema="AgentPlan",
            validator=_validate_agent_plan,
        )

    async def answer(
        self,
        context: ContextPackageV1,
        evidence: str,
        draft_answer: str,
    ) -> GroundedAnswer | None:
        if evidence or draft_answer:
            raise LLMProviderError(
                "unmetered answer inputs are rejected",
                "UNMETERED_CONTEXT_REJECTED",
            )
        messages = _metered_messages(context, ContextPurpose.ANSWER)
        payload = _structured_payload(
            messages=messages,
            temperature=min(self.temperature, 0.4),
        )
        started = time.monotonic_ns()
        result = _coerce_chat_result(await self._chat(payload), started_ns=started)
        return _validated_structured_value(
            result,
            schema="GroundedAnswer",
            validator=lambda content: _validate_grounded_answer(content, context),
        )

    async def plan_observed(
        self,
        context: ContextPackageV1,
    ) -> LLMInvocationResultV1[AgentPlan | None]:
        messages = _metered_messages(context, ContextPurpose.PLANNER)
        payload = _structured_payload(messages=messages, temperature=self.temperature)
        result = await self._chat_observed(payload)
        value = _validated_structured_value(
            result,
            schema="AgentPlan",
            validator=_validate_agent_plan,
        )
        return LLMInvocationResultV1(
            value=value,
            duration_ms=result.duration_ms,
            retry_count=result.retry_count,
            model_family=LLMModelFamilyV1.OPENAI_COMPATIBLE,
            usage=result.usage,
            outcome=(
                LLMInvocationOutcomeV1.SUCCEEDED
                if value is not None
                else LLMInvocationOutcomeV1.INVALID_RESPONSE
            ),
        )

    async def answer_observed(
        self,
        context: ContextPackageV1,
        evidence: str,
        draft_answer: str,
    ) -> LLMInvocationResultV1[GroundedAnswer | None]:
        if evidence or draft_answer:
            raise LLMProviderError(
                "unmetered answer inputs are rejected",
                "UNMETERED_CONTEXT_REJECTED",
            )
        messages = _metered_messages(context, ContextPurpose.ANSWER)
        payload = _structured_payload(
            messages=messages,
            temperature=min(self.temperature, 0.4),
        )
        result = await self._chat_observed(payload)
        value = _validated_structured_value(
            result,
            schema="GroundedAnswer",
            validator=lambda content: _validate_grounded_answer(content, context),
        )
        return LLMInvocationResultV1(
            value=value,
            duration_ms=result.duration_ms,
            retry_count=result.retry_count,
            model_family=LLMModelFamilyV1.OPENAI_COMPATIBLE,
            usage=result.usage,
            outcome=(
                LLMInvocationOutcomeV1.SUCCEEDED
                if value is not None
                else LLMInvocationOutcomeV1.INVALID_RESPONSE
            ),
        )

    async def _chat(self, payload: dict[str, Any]) -> _ChatResultV1:
        return await self._chat_observed(payload)

    async def _chat_observed(self, payload: dict[str, Any]) -> _ChatResultV1:
        started = time.monotonic_ns()
        if not settings.llm_api_key:
            raise LLMProviderError(
                "LLM_API_KEY is missing",
                "MISSING_API_KEY",
                duration_ms=_duration_ms(started),
            )
        try:
            async with httpx.AsyncClient(
                timeout=settings.llm_request_timeout_seconds
            ) as client:
                response = await client.post(
                    self._endpoint(),
                    headers={"Authorization": f"Bearer {settings.llm_api_key}"},
                    json=payload,
                )
                response.raise_for_status()
            body = response.json()
            return _ChatResultV1(
                content=_provider_content(body),
                finish_reason=_provider_finish_reason(body),
                duration_ms=_duration_ms(started),
                retry_count=0,
                usage=_provider_usage(body),
            )
        except asyncio.CancelledError:
            raise
        except httpx.TimeoutException:
            logger.warning("LLM request timeout; provider call will degrade")
            raise LLMProviderError(
                "provider request timed out",
                "PROVIDER_TIMEOUT",
                duration_ms=_duration_ms(started),
                retry_count=0,
            ) from None
        except httpx.HTTPStatusError as error:
            logger.warning(
                "LLM provider returned HTTP %s; provider call will degrade",
                error.response.status_code,
            )
            raise LLMProviderError(
                "provider returned an unsuccessful status",
                "PROVIDER_CALL_FAILED",
                duration_ms=_duration_ms(started),
                retry_count=0,
            ) from None
        except Exception as error:
            logger.warning(
                "LLM provider call failed with %s; provider call will degrade",
                type(error).__name__,
            )
            raise LLMProviderError(
                type(error).__name__,
                "PROVIDER_CALL_FAILED",
                duration_ms=_duration_ms(started),
                retry_count=0,
            ) from None

    def _endpoint(self) -> str:
        base_url = settings.llm_base_url.rstrip("/")
        if base_url.endswith(("/v1", "/v4")):
            return f"{base_url}/chat/completions"
        return f"{base_url}/v1/chat/completions"


def _structured_payload(
    *,
    messages: list[dict[str, str]],
    temperature: float,
) -> dict[str, Any]:
    return {
        "model": settings.llm_model_name,
        "temperature": temperature,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "max_tokens": settings.llm_max_completion_tokens,
        "thinking": {"type": "disabled"},
    }


def _coerce_chat_result(
    result: _ChatResultV1 | str,
    *,
    started_ns: int,
) -> _ChatResultV1:
    if isinstance(result, _ChatResultV1):
        return result
    return _ChatResultV1(
        content=result,
        finish_reason="stop",
        duration_ms=_duration_ms(started_ns),
        retry_count=0,
        usage=None,
    )


def _validated_structured_value[StructuredT](
    result: _ChatResultV1,
    *,
    schema: str,
    validator: Callable[[str], StructuredT],
) -> StructuredT | None:
    if type(result.finish_reason) is not str or result.finish_reason != "stop":
        _log_finish_reason_rejection(schema=schema, result=result)
        return None
    try:
        value = validator(result.content)
    except (TypeError, ValueError) as error:
        _log_validation_failure(
            schema=schema,
            error=error,
            finish_reason=result.finish_reason,
            duration_ms=result.duration_ms,
            retry_count=result.retry_count,
            usage=result.usage,
        )
        return None
    _log_structured_success(schema=schema, result=result)
    return value


def _json_loads(content: str) -> dict[str, Any]:
    clean = content.strip()
    if clean.startswith("```"):
        clean = clean.strip("`").removeprefix("json").strip()
    value = json.loads(clean)
    if isinstance(value, dict) and set(value) == {"AgentPlan"} and isinstance(value["AgentPlan"], dict):
        value = value["AgentPlan"]
    if isinstance(value, dict) and set(value) == {"GroundedAnswer"} and isinstance(value["GroundedAnswer"], dict):
        value = value["GroundedAnswer"]
    if not isinstance(value, dict):
        raise ValueError("LLM returned non-object JSON")
    return value


def _validate_agent_plan(content: str) -> AgentPlan:
    value = _json_loads(content)
    _require_exact_fields(value, _AGENT_PLAN_FIELDS)
    order_reference = value.get("order_reference")
    if isinstance(order_reference, dict):
        _require_exact_fields(order_reference, _ORDER_REFERENCE_FIELDS)
    if value.get("action_type") not in _ALLOWED_ACTION_TYPES:
        raise _StructuredOutputValidationError(
            ("action_type", "literal_error")
        )
    return AgentPlan.model_validate(value, strict=True)


def _validate_grounded_answer(
    content: str,
    context: ContextPackageV1,
) -> GroundedAnswer:
    value = _json_loads(content)
    _require_exact_fields(value, _GROUNDED_ANSWER_FIELDS)
    grounded = GroundedAnswer.model_validate(value, strict=True)
    if not grounded.answer.strip():
        raise _StructuredOutputValidationError(("answer", "string_too_short"))
    if grounded.confidence_level not in _ALLOWED_CONFIDENCE_LEVELS:
        raise _StructuredOutputValidationError(
            ("confidence_level", "literal_error")
        )
    controlled_ids = context.controlled_evidence_ids()
    if any(
        candidate_id not in controlled_ids
        for candidate_id in grounded.cited_candidate_ids
    ):
        raise _StructuredOutputValidationError(
            ("cited_candidate_ids", "unknown_evidence_id")
        )
    return grounded


def _require_exact_fields(
    value: dict[str, Any],
    expected: frozenset[str],
) -> None:
    actual = set(value)
    issues = [
        (field, "missing_field")
        for field in sorted(expected - actual)
    ]
    if actual - expected:
        issues.append(("$", "extra_fields"))
    if issues:
        raise _StructuredOutputValidationError(*issues)


def _log_validation_failure(
    *,
    schema: str,
    error: Exception,
    finish_reason: object,
    duration_ms: int,
    retry_count: int,
    usage: LLMTokenUsageV1 | None,
) -> None:
    issues = _safe_validation_issues(error)
    logger.warning(
        "LLM structured output validation failed "
        "schema=%s error_count=%s field_paths=%s error_types=%s "
        "finish_reason=%s duration_ms=%s retry_count=%s "
        "prompt_tokens=%s completion_tokens=%s outcome=%s",
        schema,
        len(issues),
        ",".join(path for path, _error_type in issues),
        ",".join(error_type for _path, error_type in issues),
        _safe_finish_reason(finish_reason),
        duration_ms,
        retry_count,
        usage.prompt_tokens if usage is not None else None,
        usage.completion_tokens if usage is not None else None,
        LLMInvocationOutcomeV1.INVALID_RESPONSE.value,
    )


def _log_finish_reason_rejection(*, schema: str, result: _ChatResultV1) -> None:
    logger.warning(
        "LLM structured output rejected schema=%s finish_reason=%s "
        "duration_ms=%s retry_count=%s prompt_tokens=%s completion_tokens=%s outcome=%s",
        schema,
        _safe_finish_reason(result.finish_reason),
        result.duration_ms,
        result.retry_count,
        result.usage.prompt_tokens if result.usage is not None else None,
        result.usage.completion_tokens if result.usage is not None else None,
        LLMInvocationOutcomeV1.INVALID_RESPONSE.value,
    )


def _log_structured_success(*, schema: str, result: _ChatResultV1) -> None:
    logger.info(
        "LLM structured output accepted schema=%s finish_reason=%s "
        "duration_ms=%s retry_count=%s prompt_tokens=%s completion_tokens=%s outcome=%s",
        schema,
        _safe_finish_reason(result.finish_reason),
        result.duration_ms,
        result.retry_count,
        result.usage.prompt_tokens if result.usage is not None else None,
        result.usage.completion_tokens if result.usage is not None else None,
        LLMInvocationOutcomeV1.SUCCEEDED.value,
    )


def _safe_finish_reason(finish_reason: object) -> str:
    if type(finish_reason) is str and finish_reason in _SAFE_FINISH_REASONS:
        return finish_reason
    return "invalid"


def _safe_validation_issues(error: Exception) -> tuple[tuple[str, str], ...]:
    if isinstance(error, _StructuredOutputValidationError):
        return error.issues
    if isinstance(error, ValidationError):
        issues: list[tuple[str, str]] = []
        for detail in error.errors(
            include_url=False,
            include_context=False,
            include_input=False,
        ):
            location = detail.get("loc", ())
            path = _safe_field_path(location if isinstance(location, tuple) else ())
            error_type = detail.get("type")
            issues.append(
                (
                    path,
                    error_type if isinstance(error_type, str) else "validation_error",
                )
            )
        return tuple(issues) or (("$", "validation_error"),)
    return (("$", type(error).__name__),)


def _safe_field_path(location: tuple[object, ...]) -> str:
    parts: list[str] = []
    for part in location:
        if isinstance(part, str) and part in _SAFE_FIELD_NAMES:
            parts.append(part)
        elif type(part) is int:
            parts.append("[*]")
        else:
            parts.append("*")
    return ".".join(parts) or "$"


def _provider_choice(body: object) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise TypeError("provider response envelope is invalid")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise TypeError("provider response choices are invalid")
    first = choices[0]
    if not isinstance(first, dict):
        raise TypeError("provider response choice is invalid")
    return first


def _provider_content(body: object) -> str:
    first = _provider_choice(body)
    message = first.get("message")
    if not isinstance(message, dict):
        raise TypeError("provider response message is invalid")
    content = message.get("content")
    if not isinstance(content, str):
        raise TypeError("provider response content is invalid")
    return content


def _provider_finish_reason(body: object) -> object:
    return _provider_choice(body).get("finish_reason")


def _duration_ms(started_ns: int) -> int:
    return max(0, (time.monotonic_ns() - started_ns) // 1_000_000)


def _provider_usage(body: object) -> LLMTokenUsageV1 | None:
    if not isinstance(body, dict):
        return None
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return None
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if (
        type(prompt) is not int
        or prompt < 0
        or type(completion) is not int
        or completion < 0
    ):
        return None
    return LLMTokenUsageV1(prompt_tokens=prompt, completion_tokens=completion)


def _metered_messages(
    context: ContextPackageV1,
    purpose: ContextPurpose,
) -> list[dict[str, str]]:
    if not isinstance(context, ContextPackageV1):
        raise LLMProviderError(
            "versioned context package is required",
            "CONTEXT_PACKAGE_REQUIRED",
        )
    if context.purpose is not purpose:
        raise LLMProviderError(
            "context package purpose does not match the model operation",
            "CONTEXT_PACKAGE_PURPOSE_MISMATCH",
        )
    return [message.model_dump(mode="json") for message in context.model_messages()]
