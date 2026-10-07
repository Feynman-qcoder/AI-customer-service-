import asyncio
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, cast

from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.checkpoint.base import BaseCheckpointSaver

from app.agent.state import (
    AgentState,
    ConversationCheckpointState,
    load_checkpoint_state,
    validate_state_json_round_trip,
)
from app.observability import (
    NodeOperationRecordV1,
    ObservabilityStatus,
    OperationCode,
    SpanOperation,
)
from app.runtime.context import AgentRuntimeContext, RuntimeContextProvider
from app.runtime.observability_runtime import (
    ObservabilityRuntime,
    normalized_error_type,
)


class ContextResolverPort(Protocol):
    async def __call__(
        self,
        state: ConversationCheckpointState,
        runtime: AgentRuntimeContext,
    ) -> ConversationCheckpointState: ...


class WorkingMemoryLoadPort(Protocol):
    async def __call__(
        self,
        state: ConversationCheckpointState,
        runtime: AgentRuntimeContext,
    ) -> ConversationCheckpointState: ...


class InputGuardPort(Protocol):
    async def __call__(
        self,
        state: ConversationCheckpointState,
        runtime: AgentRuntimeContext,
    ) -> ConversationCheckpointState: ...


class BlockedResponsePort(Protocol):
    async def __call__(
        self,
        state: ConversationCheckpointState,
        runtime: AgentRuntimeContext,
    ) -> ConversationCheckpointState: ...


class PlannerPort(Protocol):
    async def __call__(
        self,
        state: ConversationCheckpointState,
        runtime: AgentRuntimeContext,
    ) -> ConversationCheckpointState: ...


class ToolOrRetrievalPort(Protocol):
    async def __call__(
        self,
        state: ConversationCheckpointState,
        runtime: AgentRuntimeContext,
    ) -> ConversationCheckpointState: ...


class CustomerConfirmationGatePort(Protocol):
    async def __call__(
        self,
        state: ConversationCheckpointState,
        runtime: AgentRuntimeContext,
    ) -> ConversationCheckpointState: ...


class DurableActionPreparePort(Protocol):
    async def __call__(
        self,
        state: ConversationCheckpointState,
        runtime: AgentRuntimeContext,
    ) -> ConversationCheckpointState: ...


class PreparedActionValidationStagePort(Protocol):
    async def __call__(
        self,
        state: ConversationCheckpointState,
        runtime: AgentRuntimeContext,
    ) -> ConversationCheckpointState: ...


class AdminDecisionResumeStagePort(Protocol):
    async def __call__(
        self,
        state: ConversationCheckpointState,
        runtime: AgentRuntimeContext,
    ) -> ConversationCheckpointState: ...


class DurableBusinessExecutePort(Protocol):
    async def __call__(
        self,
        state: ConversationCheckpointState,
        runtime: AgentRuntimeContext,
    ) -> ConversationCheckpointState: ...


class WorkingMemoryUpdatePort(Protocol):
    async def __call__(
        self,
        state: ConversationCheckpointState,
        runtime: AgentRuntimeContext,
    ) -> ConversationCheckpointState: ...


class AnswerPolisherPort(Protocol):
    async def __call__(
        self,
        state: ConversationCheckpointState,
        runtime: AgentRuntimeContext,
    ) -> ConversationCheckpointState: ...


class ResponseGuardrailPort(Protocol):
    async def __call__(
        self,
        state: ConversationCheckpointState,
        runtime: AgentRuntimeContext,
    ) -> ConversationCheckpointState: ...


class AuditFinalizePort(Protocol):
    async def __call__(
        self,
        state: ConversationCheckpointState,
        runtime: AgentRuntimeContext,
    ) -> ConversationCheckpointState: ...


class AsyncAgentGraph(Protocol):
    async def ainvoke(
        self,
        state: AgentState,
        config: RunnableConfig | None = None,
    ) -> AgentState: ...


@dataclass(frozen=True, slots=True)
class CustomerWorkflowPorts:
    working_memory_load: WorkingMemoryLoadPort
    context_resolver: ContextResolverPort
    input_guard: InputGuardPort
    blocked_response: BlockedResponsePort
    planner: PlannerPort
    tool_or_retrieval: ToolOrRetrievalPort
    working_memory_update: WorkingMemoryUpdatePort
    answer_polisher: AnswerPolisherPort
    response_guardrail: ResponseGuardrailPort
    audit_finalize: AuditFinalizePort
    customer_confirmation_gate: CustomerConfirmationGatePort | None = None
    durable_action_prepare: DurableActionPreparePort | None = None
    prepared_action_validation: PreparedActionValidationStagePort | None = None
    admin_decision_resume: AdminDecisionResumeStagePort | None = None
    durable_business_execute: DurableBusinessExecutePort | None = None


WorkflowStagePort = (
    WorkingMemoryLoadPort
    | ContextResolverPort
    | InputGuardPort
    | BlockedResponsePort
    | PlannerPort
    | ToolOrRetrievalPort
    | CustomerConfirmationGatePort
    | DurableActionPreparePort
    | PreparedActionValidationStagePort
    | AdminDecisionResumeStagePort
    | DurableBusinessExecutePort
    | WorkingMemoryUpdatePort
    | AnswerPolisherPort
    | ResponseGuardrailPort
    | AuditFinalizePort
)


class CustomerWorkflowNodes:
    """Checkpoint-safe graph nodes backed only by explicit application ports."""

    def __init__(
        self,
        runtime_provider: RuntimeContextProvider,
        ports: CustomerWorkflowPorts,
        observability_runtime: ObservabilityRuntime | None = None,
    ) -> None:
        self._runtime_provider = runtime_provider
        self._ports = ports
        self._observability_runtime = observability_runtime

    async def _invoke(
        self,
        operation: OperationCode,
        port: WorkflowStagePort,
        raw_state: AgentState,
    ) -> AgentState:
        return await self._observe_node(operation, port, raw_state)

    async def _observe_node(
        self,
        operation: OperationCode,
        port: WorkflowStagePort,
        raw_state: AgentState,
    ) -> AgentState:
        runtime = self._runtime_provider.current()
        telemetry = self._observability_runtime
        scope = (
            telemetry.scope_for(runtime.execution.attempt_id)
            if telemetry is not None
            else None
        )
        started_ns = time.monotonic_ns()
        try:
            state = load_checkpoint_state(cast(Mapping[str, object], raw_state))
            self._assert_runtime_matches_state(state, runtime)
            manager = (
                telemetry.span(scope, operation=SpanOperation.NODE)
                if telemetry is not None
                else None
            )
            if manager is None:
                updated = await port(state, runtime)
                self._assert_runtime_matches_state(updated, runtime)
                validated = validate_state_json_round_trip(updated)
            else:
                with manager:
                    updated = await port(state, runtime)
                    self._assert_runtime_matches_state(updated, runtime)
                    validated = validate_state_json_round_trip(updated)
        except BaseException as error:
            if telemetry is not None:
                telemetry.record(
                    scope,
                    NodeOperationRecordV1(
                        operation=operation,
                        status=(
                            ObservabilityStatus.CANCELLED
                            if isinstance(error, asyncio.CancelledError)
                            else ObservabilityStatus.FAILED
                        ),
                        duration_ms=max(
                            0,
                            (time.monotonic_ns() - started_ns) // 1_000_000,
                        ),
                        error_type=normalized_error_type(error),
                    ),
                )
            raise
        if telemetry is not None:
            telemetry.record(
                scope,
                NodeOperationRecordV1(
                    operation=operation,
                    status=ObservabilityStatus.SUCCEEDED,
                    duration_ms=max(
                        0,
                        (time.monotonic_ns() - started_ns) // 1_000_000,
                    ),
                ),
            )
        return validated.to_agent_state()

    def _assert_runtime_matches_state(
        self,
        state: ConversationCheckpointState,
        runtime: AgentRuntimeContext,
    ) -> None:
        active_run = state.active_run
        if active_run is None:
            raise ValueError("workflow requires an active run")
        identity = state.conversation_identity
        execution = runtime.execution
        if (
            identity.conversation_id != execution.conversation_id
            or identity.thread_id != execution.thread_id
            or identity.subject_user_id != execution.subject.user_id
            or active_run.run_id != execution.run_id
            or active_run.attempt_id != execution.attempt_id
        ):
            raise ValueError("runtime execution scope does not match stable state")

    async def context_resolver(self, state: AgentState) -> AgentState:
        return await self._invoke(OperationCode.NODE_CONTEXT_RESOLVER, self._ports.context_resolver, state)

    async def working_memory_load(self, state: AgentState) -> AgentState:
        return await self._invoke(OperationCode.NODE_WORKING_MEMORY_LOAD, self._ports.working_memory_load, state)

    async def input_guard(self, state: AgentState) -> AgentState:
        return await self._invoke(OperationCode.NODE_INPUT_GUARD, self._ports.input_guard, state)

    async def blocked_response(self, state: AgentState) -> AgentState:
        return await self._invoke(OperationCode.NODE_BLOCKED_RESPONSE, self._ports.blocked_response, state)

    async def planner(self, state: AgentState) -> AgentState:
        return await self._invoke(OperationCode.NODE_PLANNER, self._ports.planner, state)

    async def tool_or_retrieval(self, state: AgentState) -> AgentState:
        return await self._invoke(OperationCode.NODE_TOOL_OR_RETRIEVAL, self._ports.tool_or_retrieval, state)

    async def customer_confirmation_gate(self, state: AgentState) -> AgentState:
        from langgraph.types import interrupt

        # The predecessor has already returned a frozen WAITING state, so the
        # checkpointer commits that state before this distinct node pauses.
        resumed = interrupt({"kind": "CUSTOMER_CONFIRMATION_V1"})
        if resumed is not True:
            raise ValueError("customer confirmation gate requires the registered ACK")
        gate = self._ports.customer_confirmation_gate
        if gate is None:
            raise ValueError("customer confirmation gate application port is unavailable")
        return await self._invoke(OperationCode.NODE_CUSTOMER_CONFIRMATION_GATE, gate, state)

    async def durable_action_prepare(self, state: AgentState) -> AgentState:
        prepare = self._ports.durable_action_prepare
        if prepare is None:
            raise ValueError("durable action prepare application port is unavailable")
        return await self._invoke(OperationCode.NODE_ACTION_PREPARE, prepare, state)

    async def admin_approval_gate(self, raw_state: AgentState) -> AgentState:
        from langgraph.types import interrupt

        from app.agent.state import CustomerConfirmationStatus, RunStatus

        runtime = self._runtime_provider.current()
        if runtime.execution.actor.role == "CUSTOMER":
            validator = self._ports.prepared_action_validation
            if validator is None:
                raise ValueError("prepared action validation port is unavailable")
            validated_state = await self._invoke(OperationCode.NODE_ADMIN_APPROVAL_GATE, validator, raw_state)
        elif runtime.execution.actor.role == "ADMIN":
            validated_state = raw_state
        else:
            raise ValueError("admin approval gate requires customer or admin authority")
        state = load_checkpoint_state(
            cast(Mapping[str, object], validated_state)
        )
        self._assert_runtime_matches_state(state, runtime)
        active_run = state.active_run
        if (
            active_run is None
            or active_run.run_status is not RunStatus.WAITING_ADMIN_APPROVAL
            or active_run.customer_confirmation_status
            is not CustomerConfirmationStatus.CONFIRMED
            or active_run.action_draft is None
            or active_run.pending_action_id is None
            or active_run.side_effect_authorization is not None
        ):
            raise ValueError("admin approval gate is not at a durable pause point")
        resumed = interrupt({"kind": "ADMIN_APPROVAL_V1"})
        if resumed is not True:
            raise ValueError("admin approval gate requires the registered decision")
        resume = self._ports.admin_decision_resume
        if resume is None:
            raise ValueError("admin decision resume application port is unavailable")
        return await self._invoke(OperationCode.NODE_ADMIN_APPROVAL_GATE, resume, validated_state)

    async def working_memory_update(self, state: AgentState) -> AgentState:
        return await self._invoke(OperationCode.NODE_WORKING_MEMORY_UPDATE, self._ports.working_memory_update, state)

    async def durable_business_execute(self, state: AgentState) -> AgentState:
        execute = self._ports.durable_business_execute
        if execute is None:
            raise ValueError("durable business execution port is unavailable")
        return await self._invoke(OperationCode.NODE_BUSINESS_EXECUTE, execute, state)

    async def answer_polisher(self, state: AgentState) -> AgentState:
        return await self._invoke(OperationCode.NODE_ANSWER_POLISHER, self._ports.answer_polisher, state)

    async def response_guardrail(self, state: AgentState) -> AgentState:
        return await self._invoke(OperationCode.NODE_RESPONSE_GUARDRAIL, self._ports.response_guardrail, state)

    async def audit_finalize(self, state: AgentState) -> AgentState:
        return await self._invoke(OperationCode.NODE_AUDIT_FINALIZE, self._ports.audit_finalize, state)

    def route_after_input_guard(self, raw_state: AgentState) -> str:
        state = load_checkpoint_state(cast(Mapping[str, object], raw_state))
        if state.active_run is None:
            raise ValueError("workflow requires an active run")
        return "blocked_response" if state.active_run.blocked else "planner"

    def route_after_tool_or_retrieval(self, raw_state: AgentState) -> str:
        from app.agent.state import RunStatus

        state = load_checkpoint_state(cast(Mapping[str, object], raw_state))
        if state.active_run is None:
            raise ValueError("workflow requires an active run")
        if state.active_run.run_status is RunStatus.WAITING_CUSTOMER_CONFIRMATION:
            return "customer_confirmation_gate"
        return "working_memory_update"

    def route_after_admin_approval(self, raw_state: AgentState) -> str:
        from app.agent.state import ApprovalDecision, RunStatus

        state = load_checkpoint_state(cast(Mapping[str, object], raw_state))
        active_run = state.active_run
        if active_run is None:
            raise ValueError("workflow requires an active run")
        if (
            active_run.run_status is RunStatus.RESUME_PENDING
            and active_run.approval_decision is ApprovalDecision.APPROVED
        ):
            return "durable_business_execute"
        if (
            active_run.run_status is RunStatus.REJECTED
            and active_run.approval_decision is ApprovalDecision.REJECTED
        ):
            return "end"
        raise ValueError("admin decision did not produce a supported route")


def build_customer_service_graph(
    nodes: CustomerWorkflowNodes,
    *,
    checkpointer: BaseCheckpointSaver[Any] | None,
) -> AsyncAgentGraph:
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(AgentState)
    graph.add_node("working_memory_load", nodes.working_memory_load)
    graph.add_node("context_resolver", nodes.context_resolver)
    graph.add_node("input_guard", nodes.input_guard)
    graph.add_node("blocked_response", nodes.blocked_response)
    graph.add_node("planner", nodes.planner)
    graph.add_node("tool_or_retrieval_executor", nodes.tool_or_retrieval)
    graph.add_node(
        "customer_confirmation_gate",
        nodes.customer_confirmation_gate,
    )
    graph.add_node("durable_action_prepare", nodes.durable_action_prepare)
    graph.add_node(
        "admin_approval_gate",
        RunnableLambda[AgentState, AgentState](nodes.admin_approval_gate),
    )
    graph.add_node(
        "durable_business_execute",
        nodes.durable_business_execute,
    )
    graph.add_node("working_memory_update", nodes.working_memory_update)
    graph.add_node("answer_polisher", nodes.answer_polisher)
    graph.add_node("response_guardrail", nodes.response_guardrail)
    graph.add_node("audit_finalize", nodes.audit_finalize)
    graph.add_edge(START, "working_memory_load")
    graph.add_edge("working_memory_load", "context_resolver")
    graph.add_edge("context_resolver", "input_guard")
    graph.add_conditional_edges(
        "input_guard",
        nodes.route_after_input_guard,
        {"blocked_response": "blocked_response", "planner": "planner"},
    )
    graph.add_edge("blocked_response", END)
    graph.add_edge("planner", "tool_or_retrieval_executor")
    graph.add_conditional_edges(
        "tool_or_retrieval_executor",
        nodes.route_after_tool_or_retrieval,
        {
            "customer_confirmation_gate": "customer_confirmation_gate",
            "working_memory_update": "working_memory_update",
        },
    )
    graph.add_edge("customer_confirmation_gate", "durable_action_prepare")
    graph.add_edge("durable_action_prepare", "admin_approval_gate")
    graph.add_conditional_edges(
        "admin_approval_gate",
        nodes.route_after_admin_approval,
        {
            "durable_business_execute": "durable_business_execute",
            "end": END,
        },
    )
    graph.add_edge("durable_business_execute", END)
    graph.add_edge("working_memory_update", "answer_polisher")
    graph.add_edge("answer_polisher", "response_guardrail")
    graph.add_edge("response_guardrail", "audit_finalize")
    graph.add_edge("audit_finalize", END)
    return cast(AsyncAgentGraph, graph.compile(checkpointer=checkpointer))
