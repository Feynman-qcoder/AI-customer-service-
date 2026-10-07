from app.agent.routing import build_rule_based_plan
from app.agent.tools.bindings import planner_executable_tool_names, production_tool_binding
from app.agent.tools.registry import EffectPhase
from app.llm import LLMProviderError, create_llm_client
from app.llm.base import (
    LLMInvocationOutcomeV1,
    LLMInvocationResultV1,
    LLMModelFamilyV1,
    current_llm_invocation_capture,
    failure_from_provider_error,
    invoke_plan_observed,
)
from app.memory import (
    ContextPackageV1,
    ContextPurpose,
    bound_context_package,
    build_single_turn_context_package,
)
from app.runtime.model_config import EffectiveModelRuntimeConfig
from app.runtime.uow import require_no_active_transaction
from app.schemas.agent import AgentPlan


async def build_agent_plan(
    runtime: EffectiveModelRuntimeConfig,
    question: str,
    context: ContextPackageV1 | None = None,
    *,
    invocation_results: list[LLMInvocationResultV1[AgentPlan | None]] | None = None,
) -> AgentPlan:
    rule_plan = build_rule_based_plan(question)
    if _requires_deterministic_rule_plan(rule_plan):
        return rule_plan
    client = create_llm_client(runtime)
    invocation_outcome: LLMInvocationOutcomeV1 | None = None
    metered_context = (
        context
        or bound_context_package(ContextPurpose.PLANNER)
        or build_single_turn_context_package(
            purpose=ContextPurpose.PLANNER,
            question=question,
        )
    )
    try:
        require_no_active_transaction("planner LLM call")
        capture = current_llm_invocation_capture()
        if invocation_results is None and capture is None:
            llm_plan = await client.plan(metered_context)
        else:
            invocation = await invoke_plan_observed(
                client,
                metered_context,
                fallback_model_family=(
                    LLMModelFamilyV1.MOCK
                    if runtime.mock_enabled
                    else LLMModelFamilyV1.OPENAI_COMPATIBLE
                ),
            )
            invocation_outcome = invocation.outcome
            if invocation_results is not None:
                invocation_results.append(invocation)
            if capture is not None:
                capture.results.append(invocation)
            llm_plan = invocation.value
    except LLMProviderError as error:
        if capture is not None:
            capture.failures.append(failure_from_provider_error(error))
        return rule_plan.model_copy(update={"decision_reason": f"{rule_plan.decision_reason}（LLM规划失败，规则兜底）"})
    if llm_plan is None:
        suffix = (
            "LLM响应无效，规则兜底"
            if invocation_outcome is LLMInvocationOutcomeV1.INVALID_RESPONSE
            else "Mock规划"
        )
        return rule_plan.model_copy(
            update={"decision_reason": f"{rule_plan.decision_reason}（{suffix}）"}
        )
    return _constrain_plan(llm_plan, rule_plan)


def _constrain_plan(llm_plan: AgentPlan, rule_plan: AgentPlan) -> AgentPlan:
    if _requires_deterministic_rule_plan(rule_plan):
        return rule_plan
    if any(tool not in planner_executable_tool_names() for tool in llm_plan.required_tools):
        return rule_plan
    if (
        llm_plan.order_reference != rule_plan.order_reference
        or llm_plan.product_reference != rule_plan.product_reference
        or any(tool not in rule_plan.required_tools for tool in llm_plan.required_tools)
    ):
        return rule_plan
    if rule_plan.risk_level in {"HIGH", "FORBIDDEN"}:
        return rule_plan.model_copy(update={"decision_reason": f"{rule_plan.decision_reason}（规则安全层覆盖LLM规划）"})
    if _requests_side_effect_authority(llm_plan) or llm_plan.risk_level != "LOW" or llm_plan.requires_confirmation:
        return rule_plan
    return llm_plan


def _requires_deterministic_rule_plan(rule_plan: AgentPlan) -> bool:
    if rule_plan.intent == "CLARIFICATION":
        return True
    if (
        rule_plan.risk_level in {"HIGH", "FORBIDDEN"}
        or rule_plan.requires_confirmation
        or rule_plan.action_type is not None
    ):
        return True
    if rule_plan.order_reference is not None or rule_plan.product_reference is not None:
        return True
    if rule_plan.required_tools and all(
        production_tool_binding(tool_name) is not None
        for tool_name in rule_plan.required_tools
    ):
        return True
    return (
        rule_plan.intent == "KNOWLEDGE_QUERY"
        and rule_plan.required_tools == ["search_knowledge_base"]
    )


def _requests_side_effect_authority(plan: AgentPlan) -> bool:
    if plan.intent in {"CANCEL_ORDER", "REFUND_REQUEST"} or plan.action_type is not None:
        return True
    for tool_name in plan.required_tools:
        binding = production_tool_binding(tool_name)
        if binding is None or binding.definition.policy.effect_phase is not EffectPhase.READ_ONLY:
            return True
    return False
