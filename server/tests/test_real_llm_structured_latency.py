from __future__ import annotations

import asyncio
import json
import logging
from hashlib import sha256
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

import app.llm.openai_compatible_client as openai_client_module
from app.agent.planner import build_agent_plan
from app.agent.routing import build_rule_based_plan
from app.agent.state import PlanSnapshot, ResponseMeta, new_conversation_state, start_new_run
from app.core.config import Settings, settings
from app.llm.base import LLMInvocationOutcomeV1, LLMProviderError
from app.llm.openai_compatible_client import OpenAICompatibleLLMClient
from app.memory import (
    ContextPurpose,
    ControlledEvidenceV1,
    SystemPolicyCatalogV1,
    build_single_turn_context_package,
)
from app.runtime.model_config import EffectiveModelRuntimeConfig
from app.schemas.agent import AgentPlan
from app.services.customer_agent_application import AgentService

_RUNTIME = EffectiveModelRuntimeConfig(
    temperature=0.2,
    top_k=5,
    min_retrieval_score=0.35,
    mock_enabled=False,
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_MISSING_FINISH_REASON = object()


class _CountingPlannerClient:
    def __init__(self) -> None:
        self.plan_calls = 0

    async def plan(self, _context: object) -> None:
        self.plan_calls += 1
        return None

    async def answer(self, _context: object, _evidence: str, _draft: str) -> None:
        raise AssertionError("planner test must not call answer")


def _planner_context(question: str = "safe question"):
    return build_single_turn_context_package(
        purpose=ContextPurpose.PLANNER,
        question=question,
    )


def _answer_context(*, evidence_ids: tuple[str, ...] = ()):
    return build_single_turn_context_package(
        purpose=ContextPurpose.ANSWER,
        question="safe question",
        evidence=tuple(
            ControlledEvidenceV1(evidence_id=evidence_id, content=f"evidence for {evidence_id}")
            for evidence_id in evidence_ids
        ),
        draft_answer="deterministic draft",
    )


def _install_transport(
    monkeypatch: pytest.MonkeyPatch,
    handler: Any,
) -> list[httpx.Request]:
    requests: list[httpx.Request] = []

    def recording_handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    transport = httpx.MockTransport(recording_handler)
    async_client = httpx.AsyncClient
    monkeypatch.setattr(
        openai_client_module.httpx,
        "AsyncClient",
        lambda **kwargs: async_client(transport=transport, **kwargs),
    )
    monkeypatch.setattr(openai_client_module.settings, "llm_api_key", "unit-test-key")
    return requests


def _provider_response(
    content: str,
    *,
    prompt_tokens: int = 20,
    completion_tokens: int = 30,
    status_code: int = 200,
    finish_reason: object = "stop",
    reasoning_content: str = "provider reasoning must never be consumed or logged",
) -> httpx.Response:
    choice: dict[str, Any] = {
        "message": {
            "content": content,
            "reasoning_content": reasoning_content,
        }
    }
    if finish_reason is not _MISSING_FINISH_REASON:
        choice["finish_reason"] = finish_reason
    return httpx.Response(
        status_code,
        json={
            "choices": [choice],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        },
    )


def _valid_plan_json() -> str:
    return json.dumps(
        {
            "intent": "PRODUCT_QUERY",
            "goal": "查询商品库存",
            "order_reference": None,
            "product_reference": "H100",
            "required_tools": ["get_product_information"],
            "action_type": None,
            "risk_level": "LOW",
            "requires_confirmation": False,
            "missing_information": [],
            "decision_reason": "读取商品信息",
        },
        ensure_ascii=False,
    )


def _valid_answer_json(*, answer: str = "validated answer", cited: list[str] | None = None) -> str:
    return json.dumps(
        {
            "answer": answer,
            "confidence_level": "HIGH",
            "need_human": False,
            "cited_candidate_ids": cited or [],
        },
        ensure_ascii=False,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "question",
    [
        "暖风杯 H100 还有库存吗？",
        "商品拆封后还能退货吗？",
        "偏远地区物流为什么不能给出固定天数？",
        "我的最近订单",
        "H100-PRO 还有库存吗？",
        "H100 和 C20 哪个有库存？",
        "帮我取消最近订单",
        "帮我退一下最近订单",
        "你好",
    ],
)
async def test_all_current_production_routes_skip_external_planner(
    monkeypatch: pytest.MonkeyPatch,
    question: str,
) -> None:
    client = _CountingPlannerClient()
    monkeypatch.setattr("app.agent.planner.create_llm_client", lambda _runtime: client)

    plan = await build_agent_plan(_RUNTIME, question, context=_planner_context(question))

    assert plan == build_rule_based_plan(question)
    assert client.plan_calls == 0


def test_system_policies_publish_complete_compact_json_contracts() -> None:
    catalog = SystemPolicyCatalogV1()
    planner = catalog.policy_for(ContextPurpose.PLANNER)
    answer = catalog.policy_for(ContextPurpose.ANSWER)

    assert '"intent"' in planner
    assert '"order_reference"' in planner
    assert '"order_no"' in planner
    assert '"ordinal_index"' in planner
    assert '"product_keyword"' in planner
    assert '"latest"' in planner
    assert '"list_all"' in planner
    assert "ORDER_QUERY|SHIPPING_QUERY|PRODUCT_QUERY|KNOWLEDGE_QUERY" in planner
    assert "只输出一个 JSON 对象" in planner
    assert '"confidence_level":"HIGH|MEDIUM|LOW"' in answer
    assert '"cited_candidate_ids":[]' in answer
    assert "只输出一个 JSON 对象" in answer
    assert "不得输出 Markdown" in answer
    assert "不得输出隐藏推理" in answer


def test_llm_bounds_are_validated_and_capped() -> None:
    configured = Settings(
        _env_file=None,
        llm_request_timeout_seconds=12,
        llm_max_completion_tokens=512,
    )

    assert configured.llm_request_timeout_seconds == 12
    assert configured.llm_max_completion_tokens == 512
    for field, value in (
        ("llm_request_timeout_seconds", 0),
        ("llm_request_timeout_seconds", 13),
        ("llm_request_timeout_seconds", True),
        ("llm_max_completion_tokens", 0),
        ("llm_max_completion_tokens", 513),
        ("llm_max_completion_tokens", False),
    ):
        with pytest.raises(ValidationError):
            Settings(_env_file=None, **{field: value})


@pytest.mark.asyncio
async def test_valid_agent_plan_json_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_transport(monkeypatch, lambda _request: _provider_response(_valid_plan_json()))

    result = await OpenAICompatibleLLMClient(temperature=0.2).plan_observed(_planner_context())

    assert result.outcome is LLMInvocationOutcomeV1.SUCCEEDED
    assert result.value == AgentPlan.model_validate_json(_valid_plan_json(), strict=True)
    assert result.retry_count == 0


@pytest.mark.asyncio
async def test_all_structured_paths_disable_provider_thinking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        system = payload["messages"][0]["content"]
        content = _valid_plan_json() if "规划节点" in system else _valid_answer_json()
        return _provider_response(content)

    requests = _install_transport(monkeypatch, respond)
    client = OpenAICompatibleLLMClient(temperature=0.2)

    assert await client.plan(_planner_context()) is not None
    assert await client.answer(_answer_context(), "", "") is not None
    assert (await client.plan_observed(_planner_context())).value is not None
    assert (await client.answer_observed(_answer_context(), "", "")).value is not None

    assert len(requests) == 4
    for request in requests:
        payload = json.loads(request.content)
        assert payload["thinking"] == {"type": "disabled"}
        assert payload["response_format"] == {"type": "json_object"}
        assert payload["max_tokens"] == 512


@pytest.mark.asyncio
async def test_valid_grounded_answer_replaces_the_draft(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = _install_transport(
        monkeypatch,
        lambda _request: _provider_response(
            _valid_answer_json(answer="validated grounded answer", cited=["kb:1"])
        ),
    )
    client = OpenAICompatibleLLMClient(temperature=0.2)
    monkeypatch.setattr(
        "app.services.customer_agent_application.create_llm_client",
        lambda _runtime: client,
    )
    service = AgentService()

    async def load_config(_operation: object) -> EffectiveModelRuntimeConfig:
        return _RUNTIME

    monkeypatch.setattr(service, "_load_runtime_config", load_config)
    answer = await service._polish_answer_with_llm(  # noqa: SLF001
        "safe question",
        "deterministic draft",
        context=_answer_context(evidence_ids=("kb:1",)),
    )

    assert answer == "validated grounded answer"
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_length_grounded_answer_is_rejected_before_json_parsing(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    raw_content = "raw answer Bearer private-value 13800000000"
    raw_reasoning = "raw reasoning Bearer private-reasoning 13900000000"
    requests = _install_transport(
        monkeypatch,
        lambda _request: _provider_response(
            _valid_answer_json(answer=raw_content),
            finish_reason="length",
            reasoning_content=raw_reasoning,
        ),
    )
    parse_calls = 0

    def forbidden_json_parse(_content: str) -> dict[str, Any]:
        nonlocal parse_calls
        parse_calls += 1
        raise AssertionError("truncated content reached JSON parsing")

    monkeypatch.setattr(openai_client_module, "_json_loads", forbidden_json_parse)
    caplog.set_level(logging.WARNING)

    result = await OpenAICompatibleLLMClient(temperature=0.2).answer_observed(
        _answer_context(),
        "",
        "",
    )

    assert result.value is None
    assert result.outcome is LLMInvocationOutcomeV1.INVALID_RESPONSE
    assert parse_calls == 0
    assert len(requests) == 1
    assert "schema=GroundedAnswer" in caplog.text
    assert "finish_reason=length" in caplog.text
    assert "outcome=INVALID_RESPONSE" in caplog.text
    assert "duration_ms=" in caplog.text
    assert "retry_count=0" in caplog.text
    assert "prompt_tokens=20" in caplog.text
    assert "completion_tokens=30" in caplog.text
    assert raw_content not in caplog.text
    assert raw_reasoning not in caplog.text


@pytest.mark.asyncio
async def test_length_agent_plan_is_rejected_before_json_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = _install_transport(
        monkeypatch,
        lambda _request: _provider_response(_valid_plan_json(), finish_reason="length"),
    )
    parse_calls = 0

    def forbidden_json_parse(_content: str) -> dict[str, Any]:
        nonlocal parse_calls
        parse_calls += 1
        raise AssertionError("truncated content reached JSON parsing")

    monkeypatch.setattr(openai_client_module, "_json_loads", forbidden_json_parse)

    result = await OpenAICompatibleLLMClient(temperature=0.2).plan_observed(
        _planner_context()
    )

    assert result.value is None
    assert result.outcome is LLMInvocationOutcomeV1.INVALID_RESPONSE
    assert parse_calls == 0
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("finish_reason", "safe_finish_reason"),
    [
        ("content_filter", "content_filter"),
        ("tool_calls", "tool_calls"),
        ("insufficient_system_resource", "insufficient_system_resource"),
        ("aborted", "aborted"),
        (_MISSING_FINISH_REASON, "invalid"),
        (None, "invalid"),
        (7, "invalid"),
        ("unknown-provider-value", "invalid"),
    ],
    ids=[
        "content-filter",
        "tool-calls",
        "insufficient-system-resource",
        "aborted",
        "missing",
        "null",
        "non-string",
        "unknown",
    ],
)
async def test_non_stop_finish_reasons_fail_closed_before_json_parsing(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    finish_reason: object,
    safe_finish_reason: str,
) -> None:
    requests = _install_transport(
        monkeypatch,
        lambda _request: _provider_response(
            _valid_answer_json(),
            finish_reason=finish_reason,
        ),
    )
    parse_calls = 0

    def forbidden_json_parse(_content: str) -> dict[str, Any]:
        nonlocal parse_calls
        parse_calls += 1
        raise AssertionError("non-stop content reached JSON parsing")

    monkeypatch.setattr(openai_client_module, "_json_loads", forbidden_json_parse)
    caplog.set_level(logging.WARNING)

    result = await OpenAICompatibleLLMClient(temperature=0.2).answer_observed(
        _answer_context(),
        "",
        "",
    )

    assert result.value is None
    assert result.outcome is LLMInvocationOutcomeV1.INVALID_RESPONSE
    assert parse_calls == 0
    assert len(requests) == 1
    assert f"finish_reason={safe_finish_reason}" in caplog.text


@pytest.mark.asyncio
async def test_application_falls_back_to_draft_on_length_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = _install_transport(
        monkeypatch,
        lambda _request: _provider_response(
            _valid_answer_json(answer="must not replace deterministic draft"),
            finish_reason="length",
        ),
    )
    client = OpenAICompatibleLLMClient(temperature=0.2)
    monkeypatch.setattr(
        "app.services.customer_agent_application.create_llm_client",
        lambda _runtime: client,
    )
    service = AgentService()

    async def load_config(_operation: object) -> EffectiveModelRuntimeConfig:
        return _RUNTIME

    monkeypatch.setattr(service, "_load_runtime_config", load_config)

    answer = await service._polish_answer_with_llm(  # noqa: SLF001
        "safe question",
        "deterministic draft",
        context=_answer_context(),
    )

    assert answer == "deterministic draft"
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_reasoning_content_is_ignored_and_never_logged_or_returned(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    reasoning_marker = "private reasoning Bearer hidden-value 13700000000"
    _install_transport(
        monkeypatch,
        lambda _request: _provider_response(
            _valid_answer_json(),
            finish_reason="stop",
            reasoning_content=reasoning_marker,
        ),
    )
    caplog.set_level(logging.INFO)

    result = await OpenAICompatibleLLMClient(temperature=0.2).answer_observed(
        _answer_context(),
        "",
        "",
    )

    assert result.value is not None
    assert result.outcome is LLMInvocationOutcomeV1.SUCCEEDED
    assert reasoning_marker not in repr(result)
    assert reasoning_marker not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        '{"answer":"missing fields"}',
        (
            '{"answer":"","confidence_level":"HIGH",'
            '"need_human":false,"cited_candidate_ids":[]}'
        ),
        (
            '{"answer":"wrong type","confidence_level":7,'
            '"need_human":false,"cited_candidate_ids":[]}'
        ),
        (
            '{"answer":"unknown citation","confidence_level":"HIGH",'
            '"need_human":false,"cited_candidate_ids":["kb:unknown"]}'
        ),
    ],
)
async def test_invalid_grounded_answer_contracts_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    content: str,
) -> None:
    _install_transport(monkeypatch, lambda _request: _provider_response(content))

    result = await OpenAICompatibleLLMClient(temperature=0.2).answer_observed(
        _answer_context(evidence_ids=("kb:1",)),
        "",
        "",
    )

    assert result.value is None
    assert result.outcome is LLMInvocationOutcomeV1.INVALID_RESPONSE


@pytest.mark.asyncio
async def test_json_fence_compatibility_is_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    fenced = f"```json\n{_valid_answer_json()}\n```"
    _install_transport(monkeypatch, lambda _request: _provider_response(fenced))

    result = await OpenAICompatibleLLMClient(temperature=0.2).answer_observed(
        _answer_context(),
        "",
        "",
    )

    assert result.value is not None
    assert result.outcome is LLMInvocationOutcomeV1.SUCCEEDED


@pytest.mark.asyncio
async def test_payload_is_json_bounded_and_uses_single_provider_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = _install_transport(
        monkeypatch,
        lambda _request: _provider_response(_valid_answer_json(), completion_tokens=512),
    )

    result = await OpenAICompatibleLLMClient(temperature=0.2).answer_observed(
        _answer_context(),
        "",
        "",
    )

    assert result.value is not None
    assert result.usage is not None and result.usage.completion_tokens <= 512
    assert result.retry_count == 0
    assert len(requests) == 1
    payload = json.loads(requests[0].content)
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["max_tokens"] == 512
    assert payload["thinking"] == {"type": "disabled"}


@pytest.mark.asyncio
async def test_timeout_has_one_attempt_and_never_logs_provider_text(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "provider timeout Bearer raw-secret-content"

    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout(secret, request=request)

    requests = _install_transport(monkeypatch, timeout)
    caplog.set_level(logging.WARNING)

    with pytest.raises(LLMProviderError) as caught:
        await OpenAICompatibleLLMClient(temperature=0.2).answer_observed(
            _answer_context(),
            "",
            "",
        )

    assert caught.value.error_type == "PROVIDER_TIMEOUT"
    assert caught.value.retry_count == 0
    assert len(requests) == 1
    assert secret not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [400, 429, 500, 503])
async def test_http_errors_are_not_retried(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
) -> None:
    requests = _install_transport(
        monkeypatch,
        lambda request: httpx.Response(status_code, request=request, text="provider secret"),
    )

    with pytest.raises(LLMProviderError) as caught:
        await OpenAICompatibleLLMClient(temperature=0.2).answer_observed(
            _answer_context(),
            "",
            "",
        )

    assert caught.value.error_type == "PROVIDER_CALL_FAILED"
    assert caught.value.retry_count == 0
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_cancellation_propagates_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    def cancel(_request: httpx.Request) -> httpx.Response:
        raise asyncio.CancelledError

    requests = _install_transport(monkeypatch, cancel)

    with pytest.raises(asyncio.CancelledError):
        await OpenAICompatibleLLMClient(temperature=0.2).answer_observed(
            _answer_context(),
            "",
            "",
        )

    assert len(requests) == 1


@pytest.mark.asyncio
async def test_validation_diagnostics_are_structured_and_never_log_raw_content(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "raw answer Bearer secret 13800000000"
    _install_transport(
        monkeypatch,
        lambda _request: _provider_response(json.dumps({"answer": secret})),
    )
    caplog.set_level(logging.WARNING)

    result = await OpenAICompatibleLLMClient(temperature=0.2).answer_observed(
        _answer_context(),
        "",
        "",
    )

    assert result.value is None
    assert "schema=GroundedAnswer" in caplog.text
    assert "error_count=" in caplog.text
    assert "field_paths=" in caplog.text
    assert "error_types=" in caplog.text
    assert "duration_ms=" in caplog.text
    assert "retry_count=0" in caplog.text
    assert "completion_tokens=30" in caplog.text
    assert secret not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "question",
    [
        "暖风杯 H100 还有库存吗？",
        "商品拆封后还能退货吗？",
        "偏远地区物流为什么不能给出固定天数？",
        "我的最近订单",
    ],
)
async def test_current_read_only_routes_call_answer_at_most_once(
    monkeypatch: pytest.MonkeyPatch,
    question: str,
) -> None:
    plan = build_rule_based_plan(question)
    assert plan.intent != "CLARIFICATION"
    assert plan.risk_level == "LOW"
    state = start_new_run(
        new_conversation_state(
            conversation_id=7,
            subject_user_id=1,
            subject_role_snapshot="CUSTOMER",
        ),
        run_id="run-read-only-answer",
        attempt_id="attempt-read-only-answer",
        question=question,
    )
    assert state.active_run is not None
    state = state.model_copy(
        update={
            "active_run": state.active_run.model_copy(
                update={
                    "plan": PlanSnapshot.model_validate(plan.model_dump(mode="python")),
                    "response_meta": ResponseMeta(
                        sources=[],
                        retrieval_score=0.0,
                        confidence_level="LOW",
                        need_human=False,
                    ),
                    "draft_answer": "deterministic read-only answer",
                }
            )
        }
    )
    calls = 0

    async def assemble(*_args: object, **_kwargs: object):
        return _answer_context()

    async def polish(*_args: object, **_kwargs: object) -> str:
        nonlocal calls
        calls += 1
        return "model answer"

    service = AgentService()
    monkeypatch.setattr(service, "_assemble_context_package", assemble)
    monkeypatch.setattr(service, "_polish_answer_with_llm", polish)

    result = await service._workflow_answer_polisher(  # noqa: SLF001
        state,
        object(),  # type: ignore[arg-type]
    )

    assert result.active_run is not None
    assert result.active_run.final_answer == "model answer"
    assert calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "plan",
    [
        build_rule_based_plan("H100-PRO 还有库存吗？"),
        build_rule_based_plan("H100 和 C20 哪个有库存？"),
        AgentPlan(
            intent="CREATE_TICKET",
            goal="需要人工处理",
            required_tools=["create_support_ticket"],
            action_type="CREATE_TICKET",
            risk_level="HIGH",
            requires_confirmation=True,
            missing_information=[],
            decision_reason="高风险等待点",
        ),
    ],
)
async def test_clarification_and_high_risk_answers_skip_llm(
    monkeypatch: pytest.MonkeyPatch,
    plan: AgentPlan,
) -> None:
    service = AgentService()
    state = start_new_run(
        new_conversation_state(
            conversation_id=7,
            subject_user_id=1,
            subject_role_snapshot="CUSTOMER",
        ),
        run_id="run-deterministic-answer",
        attempt_id="attempt-deterministic-answer",
        question=plan.goal,
    )
    assert state.active_run is not None
    draft = "deterministic clarification or approval answer"
    state = state.model_copy(
        update={
            "active_run": state.active_run.model_copy(
                update={
                    "plan": PlanSnapshot.model_validate(plan.model_dump(mode="python")),
                    "response_meta": ResponseMeta(
                        sources=[],
                        retrieval_score=0.0,
                        confidence_level="LOW",
                        need_human=plan.risk_level != "LOW",
                    ),
                    "draft_answer": draft,
                }
            )
        }
    )
    calls = 0

    async def assemble(*_args: object, **_kwargs: object):
        return _answer_context()

    async def polish(*_args: object, **_kwargs: object) -> str:
        nonlocal calls
        calls += 1
        return "model answer"

    monkeypatch.setattr(service, "_assemble_context_package", assemble)
    monkeypatch.setattr(service, "_polish_answer_with_llm", polish)

    result = await service._workflow_answer_polisher(state, object())  # type: ignore[arg-type]  # noqa: SLF001

    assert result.active_run is not None
    assert result.active_run.final_answer == draft
    assert calls == 0


@pytest.mark.asyncio
async def test_one_read_only_turn_uses_at_most_one_provider_call_and_stays_under_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        system = payload["messages"][0]["content"]
        content = _valid_plan_json() if "规划节点" in system else _valid_answer_json()
        return _provider_response(content)

    requests = _install_transport(monkeypatch, respond)
    client = OpenAICompatibleLLMClient(temperature=0.2)
    monkeypatch.setattr("app.agent.planner.create_llm_client", lambda _runtime: client)

    await build_agent_plan(
        _RUNTIME,
        "暖风杯 H100 还有库存吗？",
        context=_planner_context("暖风杯 H100 还有库存吗？"),
    )
    grounded = await client.answer_observed(_answer_context(), "", "")

    assert grounded.value is not None
    assert len(requests) == 1
    assert len(requests) * settings.llm_request_timeout_seconds < 20


def test_frontend_twenty_second_timeout_file_is_byte_identical() -> None:
    api_path = _REPOSITORY_ROOT / "web/src/api.ts"

    assert sha256(api_path.read_bytes()).hexdigest() == (
        "4e5bb63abe1679ee8fabbf3bf29630acaaacccae56900037c942247286df6787"
    )


def test_specialty_suite_never_contains_real_provider_credentials() -> None:
    source = Path(__file__).read_text(encoding="utf-8")

    assert "api." + "deepseek.com" not in source
    assert "sk" + "-" not in source
