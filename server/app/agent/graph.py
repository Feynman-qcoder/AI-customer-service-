from collections.abc import Callable, Mapping
from typing import cast

from app.agent.state import (
    AgentState,
    ConversationCheckpointState,
    RunStatus,
    load_checkpoint_state,
    validate_state_json_round_trip,
)

_FORBIDDEN_ANSWER = "这个请求可能涉及越权或不安全操作，我不能直接执行，建议转人工处理。"


def apply_input_guard(state: ConversationCheckpointState) -> ConversationCheckpointState:
    active_run = state.active_run
    if active_run is None:
        raise ValueError("input guard requires an active run")
    question = active_run.effective_question
    if "忽略系统规则" not in question and "取消所有订单" not in question:
        return state
    guarded = active_run.model_copy(
        update={
            "risk_level": "FORBIDDEN",
            "decision_reason": "输入疑似 Prompt Injection 或批量越权操作。",
            "blocked": True,
        }
    )
    return state.model_copy(update={"active_run": guarded})


def apply_response_guard(state: ConversationCheckpointState) -> ConversationCheckpointState:
    active_run = state.active_run
    if active_run is None:
        raise ValueError("response guard requires an active run")
    answer = active_run.final_answer or "已完成处理。"
    if active_run.risk_level == "FORBIDDEN":
        answer = _FORBIDDEN_ANSWER
    guarded = active_run.model_copy(update={"final_answer": answer})
    return state.model_copy(update={"active_run": guarded})


def input_guard(raw_state: AgentState) -> AgentState:
    state = load_checkpoint_state(cast(Mapping[str, object], raw_state))
    return validate_state_json_round_trip(apply_input_guard(state)).to_agent_state()


def finalize_response(raw_state: AgentState) -> AgentState:
    state = load_checkpoint_state(cast(Mapping[str, object], raw_state))
    return validate_state_json_round_trip(apply_response_guard(state)).to_agent_state()


def run_response_guard_graph(state: AgentState) -> AgentState:
    """Run the isolated guard contract without compiling a second graph."""

    return finalize_response(input_guard(state))


def run_fallback_graph(state: AgentState, handler: Callable[[AgentState], AgentState]) -> AgentState:
    guarded = input_guard(state)
    loaded = load_checkpoint_state(cast(Mapping[str, object], guarded))
    if loaded.active_run is None:
        raise ValueError("fallback graph requires an active run")
    if loaded.active_run.risk_level != "FORBIDDEN":
        guarded = handler(guarded)
    result = finalize_response(guarded)
    loaded_result = load_checkpoint_state(cast(Mapping[str, object], result))
    if loaded_result.active_run is not None and loaded_result.active_run.risk_level == "FORBIDDEN":
        rejected = loaded_result.active_run.model_copy(update={"run_status": RunStatus.REJECTED})
        rejected_state = loaded_result.model_copy(update={"active_run": rejected})
        return validate_state_json_round_trip(rejected_state).to_agent_state()
    return result
