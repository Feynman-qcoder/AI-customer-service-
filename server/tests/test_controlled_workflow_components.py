from decimal import Decimal

import pytest

from app.agent.planner import _constrain_plan
from app.agent.routing import build_rule_based_plan
from app.core.config import settings
from app.embeddings.factory import create_embedding_client
from app.embeddings.mock_embedding import MockEmbeddingClient
from app.embeddings.openai_compatible_embedding import OpenAICompatibleEmbeddingClient
from app.llm.factory import create_llm_client
from app.llm.mock_client import MockLLMClient
from app.llm.openai_compatible_client import OpenAICompatibleLLMClient
from app.runtime.model_config import EffectiveModelRuntimeConfig
from app.schemas.admin_agent import ModelConfigRequest
from app.schemas.agent import AgentPlan, OrderReference
from app.services.model_runtime_config_service import ModelRuntimeConfigService


def _plan(required_tools: list[str], risk_level: str = "LOW", requires_confirmation: bool = False) -> AgentPlan:
    return AgentPlan(
        intent="KNOWLEDGE_QUERY",
        goal="测试",
        order_reference=None,
        product_reference=None,
        required_tools=required_tools,
        action_type=None,
        risk_level=risk_level,  # type: ignore[arg-type]
        requires_confirmation=requires_confirmation,
        missing_information=[],
        decision_reason="test",
    )


def test_policy_guard_rejects_llm_unknown_tool() -> None:
    rule_plan = _plan(["search_knowledge_base"])
    llm_plan = _plan(["drop_database"])

    constrained = _constrain_plan(llm_plan, rule_plan)

    assert constrained.required_tools == ["search_knowledge_base"]


def test_policy_guard_preserves_high_risk_rule_plan() -> None:
    rule_plan = _plan(["request_refund"], risk_level="HIGH", requires_confirmation=True)
    llm_plan = _plan(["search_knowledge_base"])

    constrained = _constrain_plan(llm_plan, rule_plan)

    assert constrained.required_tools == ["request_refund"]
    assert constrained.risk_level == "HIGH"
    assert constrained.requires_confirmation is True


def test_policy_guard_never_allows_llm_to_override_clarification() -> None:
    rule_plan = AgentPlan(
        intent="CLARIFICATION",
        goal="H100-PRO 还有库存吗？",
        order_reference=None,
        product_reference=None,
        required_tools=[],
        action_type=None,
        risk_level="LOW",
        requires_confirmation=False,
        missing_information=["product_reference"],
        decision_reason="商品引用无效，需要澄清。",
    )
    llm_plan = AgentPlan(
        intent="PRODUCT_QUERY",
        goal=rule_plan.goal,
        order_reference=None,
        product_reference="H100",
        required_tools=["get_product_information"],
        action_type=None,
        risk_level="LOW",
        requires_confirmation=False,
        missing_information=[],
        decision_reason="LLM guessed a product.",
    )

    assert _constrain_plan(llm_plan, rule_plan) == rule_plan


def test_policy_guard_never_allows_llm_to_target_order_for_generic_policy() -> None:
    rule_plan = build_rule_based_plan("偏远地区物流时效是多少？")
    llm_plan = AgentPlan(
        intent="ORDER_QUERY",
        goal=rule_plan.goal,
        order_reference=OrderReference(latest=True),
        product_reference="P9",
        required_tools=["get_order_detail"],
        action_type=None,
        risk_level="LOW",
        requires_confirmation=False,
        missing_information=[],
        decision_reason="LLM guessed historical targets.",
    )

    assert _constrain_plan(llm_plan, rule_plan) == rule_plan


def test_policy_guard_never_allows_llm_to_override_recent_order_route() -> None:
    rule_plan = build_rule_based_plan("我的最近订单")
    llm_plan = AgentPlan(
        intent="KNOWLEDGE_QUERY",
        goal=rule_plan.goal,
        order_reference=None,
        product_reference=None,
        required_tools=["search_knowledge_base"],
        action_type=None,
        risk_level="LOW",
        requires_confirmation=False,
        missing_information=[],
        decision_reason="LLM ignored deterministic latest-order semantics.",
    )

    assert _constrain_plan(llm_plan, rule_plan) == rule_plan


def test_llm_factory_uses_mock_without_network_when_mock_enabled() -> None:
    runtime = EffectiveModelRuntimeConfig(
        temperature=0.2,
        top_k=5,
        min_retrieval_score=0.35,
        mock_enabled=True,
    )

    assert isinstance(create_llm_client(runtime), MockLLMClient)


def test_llm_factory_uses_openai_compatible_client_when_mock_disabled() -> None:
    runtime = EffectiveModelRuntimeConfig(
        temperature=0.2,
        top_k=5,
        min_retrieval_score=0.35,
        mock_enabled=False,
    )

    assert isinstance(create_llm_client(runtime), OpenAICompatibleLLMClient)


def test_embedding_factory_switches_by_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "embedding_mock_enabled", True)
    assert isinstance(create_embedding_client(), MockEmbeddingClient)

    monkeypatch.setattr(settings, "embedding_mock_enabled", False)
    assert isinstance(create_embedding_client(), OpenAICompatibleEmbeddingClient)


def test_model_config_schema_validates_ranges() -> None:
    request = ModelConfigRequest(temperature=0.2, topK=5, minRetrievalScore=0.35, mockEnabled=True)

    assert request.topK == 5

    with pytest.raises(ValueError):
        ModelConfigRequest(temperature=3, topK=5, minRetrievalScore=0.35, mockEnabled=True)


@pytest.mark.asyncio
async def test_runtime_config_falls_back_to_settings_without_session() -> None:
    runtime = await ModelRuntimeConfigService().get_effective(object())  # type: ignore[arg-type]

    assert runtime.top_k == settings.rag_top_k
    assert Decimal(str(runtime.min_retrieval_score)) == Decimal(str(settings.rag_min_retrieval_score))
