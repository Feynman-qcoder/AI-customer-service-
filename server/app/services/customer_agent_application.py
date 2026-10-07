from __future__ import annotations

import asyncio
import re
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Literal, cast
from uuid import uuid4

from app.agent.graph import apply_input_guard, apply_response_guard
from app.agent.planner import build_agent_plan
from app.agent.routing import (
    build_rule_based_plan,
    classify_explicit_action,
    extract_product_reference,
    has_explicit_product_reference,
)
from app.agent.state import (
    ActionDraftSnapshot as StateActionDraftSnapshot,
)
from app.agent.state import (
    ActiveRunConflictError,
    ActiveRunState,
    ApprovalDecision,
    ConversationCheckpointState,
    ConversationMemoryState,
    CustomerConfirmationStatus,
    EffectIdentity,
    MemoryProvenanceRecord,
    PlanSnapshot,
    ResponseMeta,
    RetrievalChannel,
    RetrievalEvidence,
    RunStatus,
    ToolResultSnapshot,
    ToolResultStatus,
    load_checkpoint_state,
    new_conversation_state,
    start_new_run,
)
from app.agent.thread_identity import ThreadIdentity, derive_checkpoint_namespace
from app.agent.tools.bindings import PRODUCTION_TOOL_BINDINGS, ProductionToolName
from app.agent.tools.executor import ToolAuditPort, ToolExecutor, tool_executor
from app.agent.tools.registry import (
    TOOL_REGISTRY,
    EffectPhase,
    GetOrderDetailArgs,
    GetProductInformationArgs,
    ListMyOrdersArgs,
    RequestOrderCancellationArgs,
    RequestRefundArgs,
    SearchKnowledgeBaseArgs,
)
from app.agent.workflow import (
    AsyncAgentGraph,
    CustomerWorkflowNodes,
    CustomerWorkflowPorts,
)
from app.core.config import settings
from app.core.security import AuthenticatedUser
from app.db.session import session_factory
from app.llm import LLMProviderError, create_llm_client
from app.llm.base import (
    LLMInvocationCaptureV1,
    LLMInvocationFailureCodeV1,
    LLMInvocationFailureV1,
    LLMInvocationOutcomeV1,
    LLMInvocationResultV1,
    LLMModelFamilyV1,
    bind_llm_invocation_capture,
    failure_from_provider_error,
    invoke_answer_observed,
)
from app.memory import (
    TOKEN_COUNTER_VERSION_V1,
    AnswerCurrentTurnV1,
    AuthorizedOrderReadResultV1,
    AuthorizedProductReadResultV1,
    ContextAssemblerV1,
    ContextAssemblyRequestV1,
    ContextBudgetConfigV1,
    ContextPackageV1,
    ContextPartitionBudgetExceeded,
    ContextPartitionName,
    ContextPurpose,
    ContextRollingSummaryV1,
    ContextWorkingMemoryV1,
    ControlledEvidenceV1,
    ConversationMemoryScopeV1,
    CurrentQuestionGuardPort,
    CurrentQuestionMeasurementV1,
    CurrentQuestionRequestGuard,
    LoadedWorkingMemoryV1,
    PlannerCurrentTurnV1,
    RecentSummaryApplicationPort,
    SummaryRefreshCommandV1,
    SummaryRefreshStatus,
    WorkingMemoryApplicationPort,
    WorkingMemoryPromotionCommandV1,
    bind_context_package,
    bound_context_package,
    build_single_turn_context_package,
)
from app.observability import (
    ActionCode,
    ActionOperationRecordV1,
    AttemptOutcome,
    AttemptResumeKind,
    AttemptScopeV1,
    HITLOperationRecordV1,
    LLMOperationRecordV1,
    MemoryOperationRecordV1,
    ModelFamily,
    NoOpMetricsRecorder,
    NoOpObservability,
    NormalizedErrorCode,
    NormalizedErrorTypeV1,
    ObservabilityStatus,
    OperationCode,
    RetrievalOperationRecordV1,
    RevalidationOperationRecordV1,
    RiskCode,
    ToolCode,
    ToolOperationRecordV1,
)
from app.observability import (
    RetrievalChannel as ObservabilityRetrievalChannel,
)
from app.repositories.agent_workflow_repository import (
    ActionPrepareStorePort,
    AgentWorkflowTransactionStore,
    BlockedRunStorePort,
    ConfirmationStorePort,
    ConversationContextStorePort,
    OrderReadStorePort,
    OrderSnapshot,
    PlannerPersistenceStorePort,
    ProductReadStorePort,
    ProductSnapshot,
    RetrievalStorePort,
    RuntimeConfigStorePort,
    SqlAlchemyAgentWorkflowStore,
    ToolAuditStorePort,
    ToolCallRecord,
    WorkflowAuditStorePort,
)
from app.runtime.business_execution import (
    BusinessExecutionApplicationPort,
    BusinessExecutionOutcome,
)
from app.runtime.context import (
    ActorIdentity,
    AgentRuntimeContext,
    AttemptWriteCapability,
    ControlledResourceFactories,
    ExecutionScope,
    LeaseExecutionScope,
    RuntimeConfigReader,
    RuntimeContextProvider,
    SubjectIdentity,
)
from app.runtime.durable import (
    DURABLE_INTERRUPT_CONFIRMATION_MODE,
    EffectWriteResult,
    EffectWriteScope,
    MessagePurpose,
)
from app.runtime.model_config import EffectiveModelRuntimeConfig
from app.runtime.observability_runtime import (
    AttemptObservationV1,
    ObservabilityRuntime,
    normalized_error_type,
)
from app.runtime.single_flight import PerThreadSingleFlight
from app.runtime.uow import (
    ApplicationUnitOfWorkFactory,
    SqlAlchemyApplicationUnitOfWorkFactory,
    UnitOfWorkOperation,
    require_no_active_transaction,
)
from app.schemas.agent import AgentPlan
from app.schemas.chat import ChatResponse, SourceReference
from app.schemas.retrieval import RetrievalCandidate, RetrievalQueryContext, RetrievalResult
from app.services.admin_decision_application import (
    AdminDecision,
    AdminDecisionApplicationService,
    AdminDecisionReconciler,
    AdminDecisionResumeCapabilityProvider,
)
from app.services.agent_response_formatter import AgentResponseFormatter
from app.services.customer_confirmation_application import (
    CustomerConfirmationApplicationService,
    CustomerResumeCapabilityProvider,
)
from app.services.durable_runtime_service import (
    DurableRuntimeAuthorityService,
    ReplaySafeEffectService,
)
from app.services.knowledge_service import knowledge_service
from app.services.prepared_action_validation import PreparedActionValidationPort
from app.services.side_effect_policy_service import (
    POLICY_VERSION,
    ConfirmationChallenge,
    SideEffectAuthorization,
    SideEffectPolicyError,
    SideEffectPolicyService,
    ToolAuthorizationContext,
    side_effect_policy_service,
)
from app.services.side_effect_policy_service import (
    ActionDraftSnapshot as PolicyActionDraftSnapshot,
)

R2_IDEMPOTENCY_MODE = "FENCED_REPLAY_SAFE_EFFECT_V1"


@dataclass(frozen=True, slots=True)
class ActionTargetResolution:
    order: OrderSnapshot | None
    candidates: tuple[OrderSnapshot, ...] = ()


@dataclass(frozen=True, slots=True)
class AgentAnswerOutcome:
    response: ChatResponse
    authorized_reads: tuple[OrderSnapshot | ProductSnapshot, ...] = ()


@dataclass(frozen=True, slots=True)
class DurableCustomerConfirmationOutcome:
    response: ChatResponse
    action_draft: PolicyActionDraftSnapshot


@dataclass(frozen=True, slots=True)
class LegacyContextCandidate:
    kind: str
    value: str


@dataclass(frozen=True, slots=True)
class _StructuredMemoryContextResolution:
    effective_question: str
    allow_legacy_fallback: bool


@dataclass(frozen=True, slots=True)
class AgentApplicationUnitOfWorks:
    context: ApplicationUnitOfWorkFactory[ConversationContextStorePort]
    blocked_run: ApplicationUnitOfWorkFactory[BlockedRunStorePort]
    planner: ApplicationUnitOfWorkFactory[PlannerPersistenceStorePort]
    workflow_audit: ApplicationUnitOfWorkFactory[WorkflowAuditStorePort]
    confirmation: ApplicationUnitOfWorkFactory[ConfirmationStorePort]
    orders: ApplicationUnitOfWorkFactory[OrderReadStorePort]
    action_prepare: ApplicationUnitOfWorkFactory[ActionPrepareStorePort]
    products: ApplicationUnitOfWorkFactory[ProductReadStorePort]
    runtime_config: ApplicationUnitOfWorkFactory[RuntimeConfigStorePort]
    retrieval: ApplicationUnitOfWorkFactory[RetrievalStorePort]
    tool_audit: ApplicationUnitOfWorkFactory[ToolAuditStorePort]

    @classmethod
    def from_transaction_store_factory(
        cls,
        factory: ApplicationUnitOfWorkFactory[AgentWorkflowTransactionStore],
    ) -> AgentApplicationUnitOfWorks:
        return cls(
            context=factory,
            blocked_run=factory,
            planner=factory,
            workflow_audit=factory,
            confirmation=factory,
            orders=factory,
            action_prepare=factory,
            products=factory,
            runtime_config=factory,
            retrieval=factory,
            tool_audit=factory,
        )


class _UnusedRuntimeConfigReader(RuntimeConfigReader):
    async def read(self, unit_of_work: object) -> object:
        del unit_of_work
        raise RuntimeError("workflow runtime config must be loaded through its application port")


class _ToolAuditApplication(ToolAuditPort):
    def __init__(
        self,
        unit_of_work: ApplicationUnitOfWorkFactory[ToolAuditStorePort],
        observability_runtime: ObservabilityRuntime,
        runtime_context_provider: RuntimeContextProvider,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._observability_runtime = observability_runtime
        self._runtime_context_provider = runtime_context_provider

    async def record(
        self,
        *,
        run_id: str,
        subject_user_id: int,
        tool_name: str,
        redacted_arguments: dict[str, object],
        result_summary: str,
        success: bool,
        retry_count: int,
        duration_ms: int,
    ) -> None:
        async with self._unit_of_work.open(operation="agent.tool_audit") as uow:
            await uow.store.record_tool_call(
                ToolCallRecord(
                    run_id=run_id,
                    subject_user_id=subject_user_id,
                    tool_name=tool_name,
                    redacted_arguments=redacted_arguments,
                    result_summary=result_summary,
                    success=success,
                    retry_count=retry_count,
                    duration_ms=duration_ms,
                )
            )
        try:
            attempt_id = self._runtime_context_provider.current().execution.attempt_id
        except Exception:
            return
        scope = self._observability_runtime.scope_for(attempt_id)
        if scope is None:
            return
        definition = TOOL_REGISTRY.get(tool_name)
        if definition is None:
            return
        self._observability_runtime.record(
            scope,
            ToolOperationRecordV1(
                status=(
                    ObservabilityStatus.SUCCEEDED
                    if success
                    else ObservabilityStatus.FAILED
                ),
                duration_ms=duration_ms,
                retry_count=retry_count,
                tool_name=ToolCode(tool_name),
                risk_level=RiskCode(definition.policy.risk_level),
            ),
        )
        if tool_name == ToolCode.SEARCH_KNOWLEDGE_BASE.value:
            self._observability_runtime.record(
                scope,
                RetrievalOperationRecordV1(
                    status=(
                        ObservabilityStatus.SUCCEEDED
                        if success
                        else ObservabilityStatus.FAILED
                    ),
                    duration_ms=duration_ms,
                    retry_count=retry_count,
                    channel=ObservabilityRetrievalChannel.FUSED,
                    error_type=(
                        None
                        if success
                        else NormalizedErrorTypeV1(
                            NormalizedErrorCode.UNAVAILABLE
                        )
                    ),
                ),
            )


class AgentService(AgentResponseFormatter):
    def __init__(
        self,
        unit_of_works: AgentApplicationUnitOfWorks | None = None,
        executor: ToolExecutor | None = None,
        side_effect_policy: SideEffectPolicyService | None = None,
        *,
        runtime_context_provider: RuntimeContextProvider | None = None,
        durable_authority: DurableRuntimeAuthorityService | None = None,
        replay_safe_effects: ReplaySafeEffectService | None = None,
        working_memory: WorkingMemoryApplicationPort | None = None,
        recent_summary: RecentSummaryApplicationPort | None = None,
        context_assembler: ContextAssemblerV1 | None = None,
        current_question_guard: CurrentQuestionGuardPort | None = None,
        single_flight: PerThreadSingleFlight | None = None,
        lease_ttl_seconds: int | None = None,
        lease_renew_interval_seconds: int | None = None,
        durable_customer_interrupt_enabled: bool | None = None,
        customer_resume_capabilities: CustomerResumeCapabilityProvider | None = None,
        prepared_action_validation: PreparedActionValidationPort | None = None,
        admin_resume_capabilities: AdminDecisionResumeCapabilityProvider | None = None,
        business_execution: BusinessExecutionApplicationPort | None = None,
        observability_runtime: ObservabilityRuntime | None = None,
    ) -> None:
        if unit_of_works is None:
            transaction_store_factory = cast(
                ApplicationUnitOfWorkFactory[AgentWorkflowTransactionStore],
                SqlAlchemyApplicationUnitOfWorkFactory(
                    session_factory(),
                    SqlAlchemyAgentWorkflowStore,
                ),
            )
            unit_of_works = AgentApplicationUnitOfWorks.from_transaction_store_factory(transaction_store_factory)
        self._unit_of_works = unit_of_works
        self._tool_executor = executor or tool_executor
        self._side_effect_policy = side_effect_policy or side_effect_policy_service
        self._runtime_context_provider = runtime_context_provider or RuntimeContextProvider()
        self._durable_authority = durable_authority
        self._replay_safe_effects = replay_safe_effects
        self._working_memory = working_memory
        self._recent_summary = recent_summary
        self._context_assembler = context_assembler or ContextAssemblerV1(ContextBudgetConfigV1())
        self._current_question_guard = current_question_guard or CurrentQuestionRequestGuard(self._context_assembler)
        self._single_flight = single_flight
        self._lease_ttl_seconds = lease_ttl_seconds or settings.checkpoint_lease_ttl_seconds
        self._lease_renew_interval_seconds = (
            lease_renew_interval_seconds or settings.checkpoint_lease_renew_interval_seconds
        )
        self._durable_customer_interrupt_enabled = (
            settings.durable_customer_interrupt_enabled
            if durable_customer_interrupt_enabled is None
            else durable_customer_interrupt_enabled
        )
        self._customer_resume_capabilities = customer_resume_capabilities or CustomerResumeCapabilityProvider()
        self._prepared_action_validation = prepared_action_validation
        self._admin_resume_capabilities = (
            admin_resume_capabilities or AdminDecisionResumeCapabilityProvider()
        )
        self._business_execution = business_execution
        self._observability_runtime = observability_runtime or ObservabilityRuntime(
            NoOpObservability(),
            NoOpMetricsRecorder(),
        )
        self._tool_audit = _ToolAuditApplication(
            self._unit_of_works.tool_audit,
            self._observability_runtime,
            self._runtime_context_provider,
        )
        self._customer_service_graph: AsyncAgentGraph | None = None
        self._customer_confirmation_application: CustomerConfirmationApplicationService | None = None
        self._admin_decision_application: AdminDecisionApplicationService | None = None
        self._admin_decision_reconciler: AdminDecisionReconciler | None = None

    @property
    def current_question_guard(self) -> CurrentQuestionGuardPort:
        return self._current_question_guard

    @property
    def observability_runtime(self) -> ObservabilityRuntime:
        return self._observability_runtime

    def validate_current_question(self, question: str) -> CurrentQuestionMeasurementV1:
        return self._current_question_guard.validate(question)

    def workflow_nodes(self) -> CustomerWorkflowNodes:
        """Return graph nodes backed by this service's explicit stage ports."""

        ports = CustomerWorkflowPorts(
            working_memory_load=self._workflow_working_memory_load,
            context_resolver=self._workflow_context_resolver,
            input_guard=self._workflow_input_guard,
            blocked_response=self._workflow_blocked_response,
            planner=self._workflow_planner,
            tool_or_retrieval=self._workflow_tool_or_retrieval_executor,
            customer_confirmation_gate=self._workflow_customer_confirmation_gate,
            durable_action_prepare=self._workflow_durable_action_prepare,
            prepared_action_validation=self._workflow_prepared_action_validation,
            admin_decision_resume=self._workflow_admin_decision_resume,
            durable_business_execute=self._workflow_durable_business_execute,
            working_memory_update=self._workflow_working_memory_update,
            answer_polisher=self._workflow_answer_polisher,
            response_guardrail=self._workflow_response_guardrail,
            audit_finalize=self._workflow_audit_finalize,
        )
        return CustomerWorkflowNodes(
            self._runtime_context_provider,
            ports,
            self._observability_runtime,
        )

    def bind_compiled_graph(self, graph: AsyncAgentGraph) -> None:
        """Composition-only one-time graph binding; graph compilation is external."""

        if self._customer_service_graph is not None:
            raise RuntimeError("AgentService compiled graph is already bound")
        self._customer_service_graph = graph

    @property
    def customer_confirmation_application(
        self,
    ) -> CustomerConfirmationApplicationService | None:
        return self._customer_confirmation_application

    def bind_customer_confirmation_application(
        self,
        application: CustomerConfirmationApplicationService,
    ) -> None:
        if self._customer_confirmation_application is not None:
            raise RuntimeError("customer confirmation application is already bound")
        self._customer_confirmation_application = application

    @property
    def admin_decision_application(
        self,
    ) -> AdminDecisionApplicationService | None:
        return self._admin_decision_application

    @property
    def admin_decision_reconciler(self) -> AdminDecisionReconciler | None:
        return self._admin_decision_reconciler

    def bind_admin_decision_application(
        self,
        application: AdminDecisionApplicationService,
        reconciler: AdminDecisionReconciler,
    ) -> None:
        if self._admin_decision_application is not None:
            raise RuntimeError("admin decision application is already bound")
        self._admin_decision_application = application
        self._admin_decision_reconciler = reconciler

    async def chat(
        self,
        user: AuthenticatedUser,
        conversation_id: int,
        question: str,
    ) -> ChatResponse:
        self.validate_current_question(question)
        if (
            self._customer_service_graph is None
            or self._durable_authority is None
            or self._replay_safe_effects is None
            or self._single_flight is None
        ):
            raise RuntimeError("AgentService durable composition is not initialized")
        identity = ThreadIdentity.from_conversation_id(conversation_id)
        run_id = "run_" + uuid4().hex
        attempt_id = "attempt_" + uuid4().hex
        actor = ActorIdentity(
            user_id=user.user_id,
            username=user.username,
            display_name=user.name,
            role=user.role,
        )
        subject = SubjectIdentity(user_id=user.user_id, role_snapshot=user.role)
        attempt_capability = AttemptWriteCapability()
        started_at = datetime.now(UTC)
        unfenced_execution = ExecutionScope(
            conversation_id=identity.conversation_id,
            thread_id=identity.thread_id,
            run_id=run_id,
            attempt_id=attempt_id,
            actor=actor,
            subject=subject,
            lease=LeaseExecutionScope(owner_attempt_id=attempt_id),
            started_at=started_at,
            attempt_capability=attempt_capability,
        )
        single_flight = await self._single_flight.acquire(identity.thread_id)
        lease_acquired = False
        fenced_execution: ExecutionScope | None = None
        operation_error: BaseException | None = None
        observation: AttemptObservationV1 | None = None
        observation_outcome = AttemptOutcome.FAILED
        observation_error: NormalizedErrorTypeV1 | None = None
        try:
            if self._durable_customer_interrupt_enabled:
                async with self._unit_of_works.confirmation.open(
                    operation=UnitOfWorkOperation.AGENT_CUSTOMER_CONFIRMATION_ADMISSION
                ) as uow:
                    if await uow.store.has_active_customer_confirmation(
                        conversation_id=identity.conversation_id,
                        subject_user_id=user.user_id,
                    ):
                        raise ActiveRunConflictError("the conversation already has a customer confirmation run")
            await self._durable_authority.begin_run(unfenced_execution)
            await self._durable_authority.register_attempt(unfenced_execution)
            grant = await self._durable_authority.acquire_lease(
                unfenced_execution,
                lease_duration=timedelta(seconds=self._lease_ttl_seconds),
            )
            lease_acquired = True
            fenced_execution = ExecutionScope(
                conversation_id=identity.conversation_id,
                thread_id=identity.thread_id,
                run_id=run_id,
                attempt_id=attempt_id,
                actor=actor,
                subject=subject,
                lease=LeaseExecutionScope(
                    owner_attempt_id=attempt_id,
                    fence_token=grant.fence_version,
                ),
                started_at=started_at,
                attempt_capability=attempt_capability,
            )
            observation = self._observability_runtime.begin(
                fenced_execution,
                resume=AttemptResumeKind.FRESH,
            )
            await self._refresh_summary_once(fenced_execution)
            user_message = await self._replay_safe_effects.write_message(
                fenced_execution,
                identity=EffectIdentity(
                    run_id=run_id,
                    node_name="chat_entry",
                    purpose=MessagePurpose.USER_INPUT.value,
                    sequence=0,
                ),
                purpose=MessagePurpose.USER_INPUT,
                role="USER",
                content=question,
            )
            checkpoint = start_new_run(
                new_conversation_state(
                    conversation_id=identity.conversation_id,
                    subject_user_id=user.user_id,
                    subject_role_snapshot=user.role,
                ),
                run_id=run_id,
                attempt_id=attempt_id,
                question=question,
            )
            assert checkpoint.active_run is not None
            checkpoint = checkpoint.model_copy(
                update={
                    "active_run": checkpoint.active_run.model_copy(
                        update={"current_user_message_id": user_message.target_id}
                    )
                }
            )
            runtime = self._runtime_context_provider.resolve(
                conversation_id=identity.conversation_id,
                thread_id=identity.thread_id,
                run_id=run_id,
                attempt_id=attempt_id,
                actor=actor,
                subject=subject,
                fence_token=grant.fence_version,
                attempt_capability=attempt_capability,
                resource_factory=lambda: ControlledResourceFactories(
                    unit_of_work=self._unit_of_works.context,
                    runtime_config=_UnusedRuntimeConfigReader(),
                ),
            )
            with self._runtime_context_provider.bind(runtime):
                response = await self._run_graph_and_final_effect(
                    checkpoint=checkpoint,
                    execution=fenced_execution,
                )
            observation_outcome = (
                AttemptOutcome.WAITING
                if response.agentStatus is not None
                and response.agentStatus.startswith("WAITING_")
                else AttemptOutcome.SUCCEEDED
            )
            if observation_outcome is AttemptOutcome.WAITING:
                self._observability_runtime.record(
                    observation.scope if observation is not None else None,
                    HITLOperationRecordV1(
                        operation=OperationCode.HITL_CUSTOMER_INTERRUPT,
                        status=ObservabilityStatus.WAITING,
                        duration_ms=max(
                            0,
                            (time.monotonic_ns() - observation.started_ns)
                            // 1_000_000
                            if observation is not None
                            else 0,
                        ),
                    ),
                )
            return response
        except BaseException as caught:
            operation_error = caught
            observation_outcome = (
                AttemptOutcome.CANCELLED
                if isinstance(caught, asyncio.CancelledError)
                else AttemptOutcome.FAILED
            )
            observation_error = normalized_error_type(caught)
            raise
        finally:
            cleanup_error: BaseException | None = None
            if fenced_execution is not None:
                fenced_execution.revoke_attempt()
            else:
                unfenced_execution.revoke_attempt()
            try:
                if lease_acquired and fenced_execution is not None:
                    await self._durable_authority.release_lease(fenced_execution)
            except BaseException as caught:
                cleanup_error = caught
            finally:
                try:
                    await single_flight.release()
                except BaseException as caught:
                    if cleanup_error is None:
                        cleanup_error = caught
            if cleanup_error is not None and operation_error is None:
                observation_outcome = AttemptOutcome.FAILED
                observation_error = normalized_error_type(cleanup_error)
            self._observability_runtime.finish(
                observation,
                outcome=observation_outcome,
                error_type=observation_error,
            )
            if cleanup_error is not None and operation_error is None:
                raise cleanup_error

    async def _refresh_summary_once(self, execution: ExecutionScope) -> None:
        recent_summary = self._recent_summary
        if recent_summary is None:
            return
        fence_version = execution.lease.fence_token
        if fence_version is None:
            raise RuntimeError("summary refresh requires a fenced execution")
        started_ns = time.monotonic_ns()
        try:
            result = await recent_summary.refresh(
                SummaryRefreshCommandV1(
                    scope=ConversationMemoryScopeV1(
                        conversation_id=execution.conversation_id,
                        subject_user_id=execution.subject.user_id,
                    ),
                    thread_id=execution.thread_id,
                    run_id=execution.run_id,
                    attempt_id=execution.attempt_id,
                    fence_version=fence_version,
                )
            )
        except BaseException as error:
            self._record_memory_operation(
                execution,
                operation=OperationCode.MEMORY_SUMMARY,
                status=(
                    ObservabilityStatus.CANCELLED
                    if isinstance(error, asyncio.CancelledError)
                    else ObservabilityStatus.FAILED
                ),
                started_ns=started_ns,
                error=error,
            )
            raise
        refresh_status = (
            result.status
            if result is not None
            else SummaryRefreshStatus.NOT_TRIGGERED
        )
        self._record_memory_operation(
            execution,
            operation=(
                OperationCode.MEMORY_RECENT_FALLBACK
                if refresh_status is SummaryRefreshStatus.FALLBACK
                else OperationCode.MEMORY_SUMMARY
            ),
            status=(
                ObservabilityStatus.FAILED
                if refresh_status is SummaryRefreshStatus.FALLBACK
                else ObservabilityStatus.SUCCEEDED
            ),
            started_ns=started_ns,
        )

    async def _run_graph_and_final_effect(
        self,
        *,
        checkpoint: ConversationCheckpointState,
        execution: ExecutionScope,
    ) -> ChatResponse:
        graph = self._customer_service_graph
        effects = self._replay_safe_effects
        authority = self._durable_authority
        if graph is None or effects is None or authority is None:
            raise RuntimeError("AgentService durable composition is not initialized")
        stop_renewal = asyncio.Event()

        async def execute() -> ChatResponse:
            try:
                raw_state = await graph.ainvoke(
                    checkpoint.to_agent_state(),
                    config={
                        "configurable": {
                            "thread_id": execution.thread_id,
                            "checkpoint_ns": derive_checkpoint_namespace(execution.run_id),
                        }
                    },
                )
                raw_mapping = cast(dict[str, object], raw_state)
                final_state = load_checkpoint_state(
                    {
                        key: raw_mapping[key]
                        for key in (
                            "schema_version",
                            "conversation_identity",
                            "memory",
                            "active_run",
                        )
                        if key in raw_mapping
                    }
                )
                response = self._chat_response_from_state(final_state)
                active_run = self._require_active_run(final_state)
                purpose = (
                    MessagePurpose.CONFIRMATION_CHALLENGE
                    if self._durable_customer_interrupt_enabled
                    and active_run.run_status
                    is RunStatus.WAITING_CUSTOMER_CONFIRMATION
                    else (
                        MessagePurpose.BLOCKED_RESPONSE
                        if active_run.blocked or active_run.run_status is RunStatus.REJECTED
                        else MessagePurpose.FINAL_ANSWER
                    )
                )
                await effects.write_message(
                    execution,
                    identity=EffectIdentity(
                        run_id=execution.run_id,
                        node_name="chat_exit",
                        purpose=purpose.value,
                        sequence=0,
                    ),
                    purpose=purpose,
                    role="ASSISTANT",
                    content=response.answer,
                    sources_json=response.model_dump_json(include={"sources"}),
                    retrieval_score=Decimal(str(response.retrievalScore)),
                    confidence_level=response.confidenceLevel,
                    need_human=response.needHuman,
                )
                return response
            finally:
                stop_renewal.set()

        async def renew() -> None:
            interval = self._lease_renew_interval_seconds
            while True:
                try:
                    async with asyncio.timeout(interval):
                        await stop_renewal.wait()
                except TimeoutError:
                    try:
                        await authority.renew_lease(
                            execution,
                            lease_duration=timedelta(seconds=self._lease_ttl_seconds),
                        )
                    except Exception:
                        # Renewal failure is confirmed: revoke this attempt's
                        # write capability synchronously BEFORE the failure
                        # reaches the supervisor. A graph that runs in the
                        # pre-cancellation scheduling window (the database lease
                        # may still be live for its remaining TTL) must not be
                        # able to pass the local gate for checkpoint writes or
                        # ACTION_PREPARE.
                        execution.revoke_attempt()
                        raise
                    continue
                return

        # Both tasks are application-owned: retain their handles so revocation
        # always precedes cancellation and both tasks are joined below.
        loop = asyncio.get_running_loop()
        graph_task = loop.create_task(execute(), name="customer-agent-graph")
        renewal_task = loop.create_task(
            renew(),
            name="customer-agent-lease-renewer",
        )
        try:
            await asyncio.wait(
                {graph_task, renewal_task},
                return_when=asyncio.FIRST_EXCEPTION,
            )
            for task in (renewal_task, graph_task):
                if task.done() and not task.cancelled():
                    error = task.exception()
                    if error is not None:
                        raise error
            return await graph_task
        except BaseException:
            # Revocation is deliberately synchronous and precedes cancellation.
            # A graph that temporarily swallows CancelledError therefore cannot
            # use either the checkpoint or MySQL effect authority afterwards.
            execution.revoke_attempt()
            stop_renewal.set()
            for task in (graph_task, renewal_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(graph_task, renewal_task, return_exceptions=True)
            raise
        finally:
            stop_renewal.set()

    async def _workflow_working_memory_load(
        self,
        state: ConversationCheckpointState,
        runtime: AgentRuntimeContext,
    ) -> ConversationCheckpointState:
        runtime.ensure_active()
        started_ns = time.monotonic_ns()
        try:
            loaded = await self._require_working_memory().load(self._memory_scope(state))
        except BaseException as error:
            self._record_memory_operation(
                runtime.execution,
                operation=OperationCode.MEMORY_LOAD,
                status=(
                    ObservabilityStatus.CANCELLED
                    if isinstance(error, asyncio.CancelledError)
                    else ObservabilityStatus.FAILED
                ),
                started_ns=started_ns,
                error=error,
            )
            raise
        self._record_memory_operation(
            runtime.execution,
            operation=OperationCode.MEMORY_LOAD,
            status=ObservabilityStatus.SUCCEEDED,
            started_ns=started_ns,
        )
        return self._replace_working_memory(state, loaded)

    async def _workflow_context_resolver(
        self,
        state: ConversationCheckpointState,
        runtime: AgentRuntimeContext,
    ) -> ConversationCheckpointState:
        active_run = self._require_active_run(state)
        structured = await self._resolve_structured_memory_context_resolution(
            runtime,
            state.memory,
            active_run.question,
        )
        if structured.effective_question != active_run.question or not structured.allow_legacy_fallback:
            return state.model_copy(
                update={
                    "active_run": active_run.model_copy(update={"effective_question": structured.effective_question})
                }
            )
        # Production composition always injects the 6.3 recent/summary port.
        # Multi-turn model context is assembled later by ContextAssembler and
        # must never be converted back into deterministic action authority by
        # parsing assistant prose.  The legacy branch below remains only for
        # isolated pre-6.4 test doubles that do not provide that port.
        if self._recent_summary is not None:
            return state
        conversation_id = state.conversation_identity.conversation_id
        async with self._unit_of_works.context.open(operation="agent.context") as uow:
            await uow.store.assert_conversation_owned(
                conversation_id=conversation_id,
                subject_user_id=state.conversation_identity.subject_user_id,
            )
            messages = await uow.store.recent_assistant_messages(conversation_id, 6)
        candidate = self._legacy_context_candidate(messages, active_run.question)
        if candidate is not None:
            effective_question = await self._revalidate_legacy_context_candidate(
                runtime,
                candidate,
                active_run.question,
            )
        else:
            effective_question = self._resolve_conversation_context(
                messages,
                active_run.question,
            )
        return state.model_copy(
            update={"active_run": active_run.model_copy(update={"effective_question": effective_question})}
        )

    async def _workflow_input_guard(
        self,
        state: ConversationCheckpointState,
        runtime: AgentRuntimeContext,
    ) -> ConversationCheckpointState:
        runtime.ensure_active()
        return apply_input_guard(state)

    async def _workflow_blocked_response(
        self,
        state: ConversationCheckpointState,
        runtime: AgentRuntimeContext,
    ) -> ConversationCheckpointState:
        active_run = self._require_active_run(state)
        identity = state.conversation_identity
        final_answer = "这个请求可能涉及越权或不安全操作，我不能直接执行，建议转人工处理。"
        async with self._unit_of_works.blocked_run.open(operation="agent.blocked") as uow:
            config = await uow.store.runtime_config()
            await uow.store.persist_blocked_run(
                run_id=active_run.run_id,
                thread_id=identity.thread_id,
                conversation_id=identity.conversation_id,
                user_id=identity.subject_user_id,
                started_at=self._database_time(runtime.execution.started_at),
                completed_at=self._database_time(runtime.clock.now()),
                question=self._compact(active_run.effective_question),
                final_answer=final_answer,
                model_name=self._runtime_model_name(config),
                config_version=self._runtime_config_version(config),
            )
        updated = active_run.model_copy(
            update={
                "run_status": RunStatus.REJECTED,
                "final_answer": final_answer,
                "response_meta": ResponseMeta(sources=[], retrieval_score=0.0, confidence_level="LOW", need_human=True),
            }
        )
        return state.model_copy(update={"active_run": updated})

    async def _workflow_planner(
        self,
        state: ConversationCheckpointState,
        runtime: AgentRuntimeContext,
    ) -> ConversationCheckpointState:
        active_run = self._require_active_run(state)
        identity = state.conversation_identity
        config = await self._load_runtime_config(UnitOfWorkOperation.AGENT_PLANNER_CONFIG)
        context = await self._assemble_context_package(
            state,
            PlannerCurrentTurnV1(question=active_run.question),
        )
        with bind_context_package(context):
            execution = getattr(runtime, "execution", None)
            scope = (
                self._observability_runtime.scope_for(execution.attempt_id)
                if isinstance(execution, ExecutionScope)
                else None
            )
            if scope is None:
                plan = await build_agent_plan(config, active_run.effective_question)
            else:
                with bind_llm_invocation_capture() as invocation_capture:
                    plan = await build_agent_plan(config, active_run.effective_question)
                self._record_llm_capture(
                    scope,
                    OperationCode.LLM_PLANNER,
                    invocation_capture,
                )
        async with self._unit_of_works.planner.open(operation="agent.planner.persist") as uow:
            await uow.store.assert_conversation_owned(
                conversation_id=identity.conversation_id,
                subject_user_id=identity.subject_user_id,
            )
            current_config = await uow.store.runtime_config()
            if current_config != config:
                raise RuntimeError("model runtime configuration changed during planning")
            await uow.store.persist_planned_run(
                run_id=active_run.run_id,
                thread_id=identity.thread_id,
                conversation_id=identity.conversation_id,
                user_id=identity.subject_user_id,
                started_at=self._database_time(runtime.execution.started_at),
                question=self._compact(active_run.effective_question),
                intent=plan.intent,
                risk_level=plan.risk_level,
                model_name=self._runtime_model_name(config),
                config_version=self._runtime_config_version(config),
            )
        snapshot = PlanSnapshot.model_validate(plan.model_dump(mode="python"))
        updated = active_run.model_copy(
            update={
                "plan": snapshot,
                "intent": plan.intent,
                "risk_level": plan.risk_level,
                "selected_tools": list(plan.required_tools),
                "decision_reason": plan.decision_reason,
            }
        )
        return state.model_copy(update={"active_run": updated})

    async def _workflow_tool_or_retrieval_executor(
        self,
        state: ConversationCheckpointState,
        runtime: AgentRuntimeContext,
    ) -> ConversationCheckpointState:
        active_run = self._require_active_run(state)
        if active_run.plan is None:
            raise ValueError("tool stage requires a stable plan snapshot")
        plan = self._agent_plan_from_snapshot(active_run.plan)
        answer_outcome = await self._answer_with_tools(
            self._authenticated_actor(runtime),
            state.conversation_identity.conversation_id,
            active_run.run_id,
            plan,
            active_run.effective_question,
        )
        if isinstance(answer_outcome, DurableCustomerConfirmationOutcome):
            answer = answer_outcome.response
            draft = StateActionDraftSnapshot.model_validate(asdict(answer_outcome.action_draft))
            async with self._unit_of_works.confirmation.open(
                operation=UnitOfWorkOperation.AGENT_CUSTOMER_CONFIRMATION_FREEZE
            ) as uow:
                await uow.store.mark_customer_confirmation_waiting(
                    scope=self._confirmation_effect_scope(runtime.execution),
                    conversation_id=state.conversation_identity.conversation_id,
                    subject_user_id=state.conversation_identity.subject_user_id,
                    run_id=active_run.run_id,
                    confirmation_prompt=answer.answer,
                )
            response_meta = self._response_meta(answer)
            waiting = active_run.model_copy(
                update={
                    "run_status": RunStatus.WAITING_CUSTOMER_CONFIRMATION,
                    "action_draft": draft,
                    "side_effect_authorization": None,
                    "pending_action_id": None,
                    "customer_confirmation_status": CustomerConfirmationStatus.PENDING,
                    "draft_answer": answer.answer,
                    "final_answer": answer.answer,
                    "response_meta": response_meta,
                    "retrieval_evidence": list(response_meta.sources),
                    "retrieval_score": response_meta.retrieval_score,
                    "tool_results": [],
                }
            )
            return state.model_copy(update={"active_run": waiting})
        if isinstance(answer_outcome, AgentAnswerOutcome):
            answer = answer_outcome.response
            tool_results = self._memory_tool_results(
                answer_outcome.authorized_reads,
                observed_at=runtime.clock.now(),
            )
        else:
            answer = answer_outcome
            tool_results = []
        async with self._unit_of_works.workflow_audit.open(operation="agent.tool_stage.audit") as uow:
            await uow.store.assert_conversation_owned(
                conversation_id=state.conversation_identity.conversation_id,
                subject_user_id=state.conversation_identity.subject_user_id,
            )
            await uow.store.record_step(
                active_run.run_id,
                "tool_executor",
                f"intent={plan.intent}",
                f"confidence={answer.confidenceLevel}, need_human={answer.needHuman}",
                "COMPLETED",
            )
        response_meta = self._response_meta(answer)
        updated = active_run.model_copy(
            update={
                "draft_answer": answer.answer,
                "response_meta": response_meta,
                "retrieval_evidence": list(response_meta.sources),
                "retrieval_score": response_meta.retrieval_score,
                "tool_results": tool_results,
            }
        )
        return state.model_copy(update={"active_run": updated})

    async def _workflow_customer_confirmation_gate(
        self,
        state: ConversationCheckpointState,
        runtime: AgentRuntimeContext,
    ) -> ConversationCheckpointState:
        runtime.ensure_active()
        active_run = self._require_active_run(state)
        if (
            active_run.run_status is not RunStatus.WAITING_CUSTOMER_CONFIRMATION
            or active_run.customer_confirmation_status is not CustomerConfirmationStatus.PENDING
            or active_run.action_draft is None
            or active_run.side_effect_authorization is not None
            or active_run.pending_action_id is not None
        ):
            raise ActiveRunConflictError("customer confirmation gate is not at a resumable pause point")
        self._customer_resume_capabilities.consume(
            execution=runtime.execution,
            state=state,
        )
        resumed = active_run.model_copy(
            update={
                "run_status": RunStatus.RESUME_PENDING,
                "customer_confirmation_status": CustomerConfirmationStatus.CONFIRMED,
            }
        )
        return state.model_copy(update={"active_run": resumed})

    async def _workflow_durable_action_prepare(
        self,
        state: ConversationCheckpointState,
        runtime: AgentRuntimeContext,
    ) -> ConversationCheckpointState:
        runtime.ensure_active()
        active_run = self._require_active_run(state)
        draft = active_run.action_draft
        execution = runtime.execution
        if (
            not self._durable_customer_interrupt_enabled
            or active_run.run_status is not RunStatus.RESUME_PENDING
            or active_run.customer_confirmation_status
            is not CustomerConfirmationStatus.CONFIRMED
            or draft is None
            or active_run.pending_action_id is not None
            or active_run.side_effect_authorization is not None
            or draft.subject_user_id != execution.subject.user_id
            or draft.policy_version != POLICY_VERSION
        ):
            raise ActiveRunConflictError(
                "durable action prepare is not at a confirmed customer pause point"
            )
        effects = self._replay_safe_effects
        if effects is None:
            raise RuntimeError("durable action prepare requires replay-safe effects")
        marker_scope = self._confirmation_effect_scope(execution)
        async with self._unit_of_works.confirmation.open(
            operation=UnitOfWorkOperation.AGENT_CUSTOMER_CONFIRMATION_RESUMED
        ) as uow:
            await uow.store.mark_customer_confirmation_resumed(
                scope=marker_scope,
                conversation_id=execution.conversation_id,
                subject_user_id=execution.subject.user_id,
                run_id=execution.run_id,
            )
        async def write(
            authorization: SideEffectAuthorization | None,
            validated_order_status: str,
        ) -> EffectWriteResult:
            return await effects.write_action_prepare(
                execution,
                identity=EffectIdentity(
                    run_id=execution.run_id,
                    node_name="durable_action_prepare",
                    purpose=EffectPhase.ACTION_PREPARE.value,
                    sequence=0,
                ),
                authorization=authorization,
                logical_action_id=draft.logical_action_id,
                action_type=draft.action_type,
                target_order_id=draft.target_order_id,
                target_order_no=draft.target_order_no,
                validated_order_status=validated_order_status,
                action_payload={"reason": draft.reason_code},
                risk_level="HIGH",
                confirmation_mode=DURABLE_INTERRUPT_CONFIRMATION_MODE,
                customer_confirmation_challenge_digest=draft.nonce_digest,
                draft_revision=draft.draft_revision,
                draft_expires_at=draft.expires_at,
            )

        action_started_ns = time.monotonic_ns()
        try:
            result = await effects.replay_durable_action_prepare(
                execution,
                identity=EffectIdentity(
                    run_id=execution.run_id,
                    node_name="durable_action_prepare",
                    purpose=EffectPhase.ACTION_PREPARE.value,
                    sequence=0,
                ),
                logical_action_id=draft.logical_action_id,
                action_type=draft.action_type,
                target_order_id=draft.target_order_id,
                target_order_no=draft.target_order_no,
                action_payload={"reason": draft.reason_code},
                risk_level="HIGH",
                customer_confirmation_challenge_digest=draft.nonce_digest,
                draft_revision=draft.draft_revision,
                draft_expires_at=draft.expires_at,
            )
            if result is None:
                revalidation_started_ns = time.monotonic_ns()
                try:
                    current, authorization = await self._authorize_durable_action_prepare(
                        execution=execution,
                        draft=draft,
                    )
                except BaseException as error:
                    self._record_revalidation_operation(
                        execution,
                        operation=OperationCode.REVALIDATION_POLICY,
                        status=ObservabilityStatus.FAILED,
                        action_type=draft.action_type,
                        started_ns=revalidation_started_ns,
                        error=error,
                    )
                    raise
                self._record_revalidation_operation(
                    execution,
                    operation=OperationCode.REVALIDATION_POLICY,
                    status=ObservabilityStatus.SUCCEEDED,
                    action_type=draft.action_type,
                    started_ns=revalidation_started_ns,
                )
                result = await write(authorization, current.status)
        except BaseException as error:
            self._record_action_operation(
                execution,
                operation=OperationCode.ACTION_PREPARE,
                status=(
                    ObservabilityStatus.CANCELLED
                    if isinstance(error, asyncio.CancelledError)
                    else ObservabilityStatus.FAILED
                ),
                action_type=draft.action_type,
                started_ns=action_started_ns,
                error=error,
            )
            raise
        self._record_action_operation(
            execution,
            operation=OperationCode.ACTION_PREPARE,
            status=ObservabilityStatus.SUCCEEDED,
            action_type=draft.action_type,
            started_ns=action_started_ns,
        )
        async with self._unit_of_works.confirmation.open(
            operation=UnitOfWorkOperation.AGENT_CUSTOMER_CONFIRMATION_ADMIN_WAITING
        ) as uow:
            await uow.store.mark_admin_approval_waiting(
                scope=marker_scope,
                conversation_id=execution.conversation_id,
                subject_user_id=execution.subject.user_id,
                run_id=execution.run_id,
                pending_action_id=result.target_id,
            )
        waiting = active_run.model_copy(
            update={
                "run_status": RunStatus.WAITING_ADMIN_APPROVAL,
                "pending_action_id": result.target_id,
                "side_effect_authorization": None,
            }
        )
        return state.model_copy(update={"active_run": waiting})

    async def _workflow_prepared_action_validation(
        self,
        state: ConversationCheckpointState,
        runtime: AgentRuntimeContext,
    ) -> ConversationCheckpointState:
        runtime.ensure_active()
        active_run = self._require_active_run(state)
        draft = active_run.action_draft
        pending_action_id = active_run.pending_action_id
        validator = self._prepared_action_validation
        if (
            validator is None
            or active_run.run_status is not RunStatus.WAITING_ADMIN_APPROVAL
            or active_run.customer_confirmation_status
            is not CustomerConfirmationStatus.CONFIRMED
            or draft is None
            or pending_action_id is None
            or active_run.side_effect_authorization is not None
        ):
            raise ActiveRunConflictError(
                "prepared action is not at a validated admin pause point"
            )
        started_ns = time.monotonic_ns()
        try:
            await validator.validate(
                execution=runtime.execution,
                draft=draft,
                pending_action_id=pending_action_id,
            )
        except BaseException as error:
            self._record_revalidation_operation(
                runtime.execution,
                operation=OperationCode.REVALIDATION_POLICY,
                status=(
                    ObservabilityStatus.CANCELLED
                    if isinstance(error, asyncio.CancelledError)
                    else ObservabilityStatus.FAILED
                ),
                action_type=draft.action_type,
                started_ns=started_ns,
                error=error,
            )
            raise
        self._record_revalidation_operation(
            runtime.execution,
            operation=OperationCode.REVALIDATION_POLICY,
            status=ObservabilityStatus.SUCCEEDED,
            action_type=draft.action_type,
            started_ns=started_ns,
        )
        runtime.ensure_active()
        return state

    async def _workflow_admin_decision_resume(
        self,
        state: ConversationCheckpointState,
        runtime: AgentRuntimeContext,
    ) -> ConversationCheckpointState:
        runtime.ensure_active()
        active_run = self._require_active_run(state)
        if (
            runtime.execution.actor.role != "ADMIN"
            or active_run.run_status is not RunStatus.WAITING_ADMIN_APPROVAL
            or active_run.customer_confirmation_status
            is not CustomerConfirmationStatus.CONFIRMED
            or active_run.action_draft is None
            or active_run.pending_action_id is None
            or active_run.approval_decision is not None
            or active_run.side_effect_authorization is not None
        ):
            raise ActiveRunConflictError(
                "admin decision is not at a resumable pause point"
            )
        decision = self._admin_resume_capabilities.consume(
            execution=runtime.execution,
            state=state,
        )
        if decision.decision is AdminDecision.APPROVE:
            run_status = RunStatus.RESUME_PENDING
            approval = ApprovalDecision.APPROVED
        else:
            run_status = RunStatus.REJECTED
            approval = ApprovalDecision.REJECTED
        resumed = active_run.model_copy(
            update={
                "run_status": run_status,
                "approval_decision": approval,
                "decision_reason": decision.reason_code,
                "side_effect_authorization": None,
            }
        )
        runtime.ensure_active()
        return state.model_copy(update={"active_run": resumed})

    async def _workflow_durable_business_execute(
        self,
        state: ConversationCheckpointState,
        runtime: AgentRuntimeContext,
    ) -> ConversationCheckpointState:
        runtime.ensure_active()
        active_run = self._require_active_run(state)
        business_execution = self._business_execution
        if (
            business_execution is None
            or runtime.execution.actor.role != "ADMIN"
            or active_run.run_status is not RunStatus.RESUME_PENDING
            or active_run.approval_decision is not ApprovalDecision.APPROVED
            or active_run.customer_confirmation_status
            is not CustomerConfirmationStatus.CONFIRMED
            or active_run.action_draft is None
            or active_run.pending_action_id is None
            or active_run.side_effect_authorization is not None
        ):
            raise ActiveRunConflictError(
                "durable business execution is not at an approved action boundary"
            )
        started_ns = time.monotonic_ns()
        try:
            result = await business_execution.execute(
                execution=runtime.execution,
                draft=active_run.action_draft,
                pending_action_id=active_run.pending_action_id,
            )
        except BaseException as error:
            self._record_action_operation(
                runtime.execution,
                operation=OperationCode.ACTION_EXECUTION_CLAIM,
                status=(
                    ObservabilityStatus.CANCELLED
                    if isinstance(error, asyncio.CancelledError)
                    else ObservabilityStatus.FAILED
                ),
                action_type=active_run.action_draft.action_type,
                started_ns=started_ns,
                error=error,
            )
            self._record_action_operation(
                runtime.execution,
                operation=OperationCode.ACTION_BUSINESS_EXECUTE,
                status=(
                    ObservabilityStatus.CANCELLED
                    if isinstance(error, asyncio.CancelledError)
                    else ObservabilityStatus.FAILED
                ),
                action_type=active_run.action_draft.action_type,
                started_ns=started_ns,
                error=error,
            )
            raise
        self._record_action_operation(
            runtime.execution,
            operation=OperationCode.ACTION_EXECUTION_CLAIM,
            status=ObservabilityStatus.SUCCEEDED,
            action_type=active_run.action_draft.action_type,
            started_ns=started_ns,
        )
        self._record_action_operation(
            runtime.execution,
            operation=OperationCode.ACTION_BUSINESS_EXECUTE,
            status=(
                ObservabilityStatus.SUCCEEDED
                if result.outcome is BusinessExecutionOutcome.EXECUTED
                else ObservabilityStatus.STALE
            ),
            action_type=active_run.action_draft.action_type,
            started_ns=started_ns,
        )
        if result.outcome is BusinessExecutionOutcome.EXECUTED:
            approval = ApprovalDecision.APPROVED
            final_answer = "已按管理员决定提交业务处置。"
        else:
            approval = ApprovalDecision.STALE
            final_answer = "业务事实已变化，本次动作未执行。"
        completed = active_run.model_copy(
            update={
                "run_status": RunStatus.COMPLETED,
                "approval_decision": approval,
                "decision_reason": result.reason_code.value,
                "side_effect_authorization": None,
                "final_answer": final_answer,
            }
        )
        runtime.ensure_active()
        return state.model_copy(update={"active_run": completed})

    async def _load_durable_action_prepare_order(
        self,
        *,
        execution: ExecutionScope,
        draft: StateActionDraftSnapshot,
        validate_policy: bool,
    ) -> OrderSnapshot:
        execution.require_attempt_active()
        user = AuthenticatedUser(
            execution.actor.user_id,
            execution.actor.username,
            execution.actor.display_name,
            execution.actor.role,
        )
        async with self._unit_of_works.confirmation.open(
            operation=UnitOfWorkOperation.AGENT_CUSTOMER_CONFIRMATION_REVALIDATE
        ) as uow:
            order = await uow.store.lock_customer_confirmation_order(
                user_id=execution.subject.user_id,
                order_no=draft.target_order_no,
            )
            if (
                order is None
                or order.id != draft.target_order_id
                or order.order_no.upper() != draft.target_order_no.upper()
                or order.user_id != execution.subject.user_id
            ):
                raise SideEffectPolicyError(
                    "durable action target is unavailable or no longer owned"
                )
            if validate_policy:
                database_now = await uow.store.customer_confirmation_database_time()
                policy_draft = PolicyActionDraftSnapshot(
                    **draft.model_dump(mode="python")
                )
                self._side_effect_policy.validate_durable_customer_confirmation(
                    run_id=execution.run_id,
                    user=user,
                    order=order,
                    draft=policy_draft,
                    confirmation_text=self._side_effect_policy.durable_confirmation_prompt(
                        policy_draft
                    ),
                    confirmation_challenge_digest=draft.nonce_digest,
                    now=database_now,
                )
        execution.require_attempt_active()
        return order

    async def _authorize_durable_action_prepare(
        self,
        *,
        execution: ExecutionScope,
        draft: StateActionDraftSnapshot,
    ) -> tuple[OrderSnapshot, SideEffectAuthorization]:
        order = await self._load_durable_action_prepare_order(
            execution=execution,
            draft=draft,
            validate_policy=True,
        )
        async with self._unit_of_works.confirmation.open(
            operation=UnitOfWorkOperation.AGENT_CUSTOMER_CONFIRMATION_DATABASE_TIME
        ) as uow:
            database_now = await uow.store.customer_confirmation_database_time()
        policy_draft = PolicyActionDraftSnapshot(**draft.model_dump(mode="python"))
        authorization = self._side_effect_policy.authorize_action_prepare(
            execution.run_id,
            AuthenticatedUser(
                execution.actor.user_id,
                execution.actor.username,
                execution.actor.display_name,
                execution.actor.role,
            ),
            policy_draft,
            draft.action_type,
            draft.target_order_no,
            now=database_now,
        )
        return order, authorization

    async def _workflow_working_memory_update(
        self,
        state: ConversationCheckpointState,
        runtime: AgentRuntimeContext,
    ) -> ConversationCheckpointState:
        runtime.ensure_active()
        active_run = self._require_active_run(state)
        if active_run.blocked:
            return state
        if active_run.current_user_message_id is None:
            raise ValueError("working memory update requires the current user message")
        if (
            has_explicit_product_reference(active_run.question)
            and extract_product_reference(active_run.question) is None
        ):
            return state
        authorized_reads = await self._revalidate_memory_tool_results(
            runtime,
            active_run.tool_results,
            source_message_id=active_run.current_user_message_id,
        )
        started_ns = time.monotonic_ns()
        try:
            loaded = await self._require_working_memory().update(
                self._memory_scope(state),
                WorkingMemoryPromotionCommandV1(
                    current_input_message_id=active_run.current_user_message_id,
                    authorized_read_results=authorized_reads,
                ),
            )
        except BaseException as error:
            self._record_memory_operation(
                runtime.execution,
                operation=OperationCode.MEMORY_UPDATE,
                status=(
                    ObservabilityStatus.CANCELLED
                    if isinstance(error, asyncio.CancelledError)
                    else ObservabilityStatus.FAILED
                ),
                started_ns=started_ns,
                error=error,
            )
            raise
        self._record_memory_operation(
            runtime.execution,
            operation=OperationCode.MEMORY_UPDATE,
            status=ObservabilityStatus.SUCCEEDED,
            started_ns=started_ns,
        )
        return self._replace_working_memory(state, loaded)

    async def _workflow_answer_polisher(
        self,
        state: ConversationCheckpointState,
        runtime: AgentRuntimeContext,
    ) -> ConversationCheckpointState:
        del runtime
        active_run = self._require_active_run(state)
        if active_run.plan is None or active_run.response_meta is None or active_run.draft_answer is None:
            raise ValueError("answer stage requires plan, response metadata, and draft")
        plan = self._agent_plan_from_snapshot(active_run.plan)
        if self._requires_deterministic_answer(plan, active_run.draft_answer):
            answer = active_run.draft_answer
        else:
            try:
                context = await self._assemble_context_package(
                    state,
                    AnswerCurrentTurnV1(
                        question=active_run.question,
                        evidence=self._controlled_answer_evidence(active_run),
                        draft_answer=active_run.draft_answer,
                    ),
                )
            except ContextPartitionBudgetExceeded as error:
                if error.partition is not ContextPartitionName.CURRENT_TURN:
                    raise
                answer = active_run.draft_answer
            else:
                with bind_context_package(context):
                    answer = await self._polish_answer_with_llm(
                        active_run.question,
                        active_run.draft_answer,
                    )
        return state.model_copy(update={"active_run": active_run.model_copy(update={"final_answer": answer})})

    def _requires_deterministic_answer(
        self,
        plan: AgentPlan,
        draft_answer: str,
    ) -> bool:
        return bool(
            plan.intent in {"CLARIFICATION", "CANCEL_ORDER", "REFUND_REQUEST"}
            or plan.risk_level != "LOW"
            or plan.requires_confirmation
            or plan.action_type is not None
            or self._side_effect_policy.parse_confirmation_prompt(draft_answer)
            is not None
        )

    async def _workflow_response_guardrail(
        self,
        state: ConversationCheckpointState,
        runtime: AgentRuntimeContext,
    ) -> ConversationCheckpointState:
        del runtime
        active_run = self._require_active_run(state)
        if active_run.final_answer is None:
            raise ValueError("response guard requires a final answer candidate")
        guarded = apply_response_guard(state)
        guarded_run = self._require_active_run(guarded)
        async with self._unit_of_works.workflow_audit.open(operation="agent.response_guard.audit") as uow:
            await uow.store.assert_conversation_owned(
                conversation_id=state.conversation_identity.conversation_id,
                subject_user_id=state.conversation_identity.subject_user_id,
            )
            await uow.store.record_step(
                active_run.run_id,
                "response_guardrail",
                "ANSWER_CANDIDATE_CLASSIFIED",
                "ANSWER_ALLOWED" if guarded_run.final_answer else "ANSWER_REJECTED",
                "COMPLETED",
            )
        return guarded

    async def _workflow_audit_finalize(
        self,
        state: ConversationCheckpointState,
        runtime: AgentRuntimeContext,
    ) -> ConversationCheckpointState:
        active_run = self._require_active_run(state)
        if active_run.response_meta is None or active_run.final_answer is None:
            raise ValueError("audit finalize requires response metadata and final answer")
        response = self._chat_response_from_state(state)
        async with self._unit_of_works.workflow_audit.open(operation="agent.finalize") as uow:
            await uow.store.assert_conversation_owned(
                conversation_id=state.conversation_identity.conversation_id,
                subject_user_id=state.conversation_identity.subject_user_id,
            )
            await uow.store.finalize_run(
                run_id=active_run.run_id,
                conversation_id=state.conversation_identity.conversation_id,
                answer=active_run.final_answer,
                sources_json=response.model_dump_json(include={"sources"}),
                retrieval_score=response.retrievalScore,
                confidence_level=response.confidenceLevel,
                need_human=response.needHuman,
                completed_at=self._database_time(runtime.clock.now()),
            )
        return state.model_copy(
            update={"active_run": active_run.model_copy(update={"run_status": RunStatus.COMPLETED})}
        )

    async def _answer_with_tools(
        self,
        user: AuthenticatedUser,
        conversation_id: int,
        run_id: str,
        plan: AgentPlan,
        question: str,
    ) -> ChatResponse | AgentAnswerOutcome | DurableCustomerConfirmationOutcome:
        if plan.intent == "CLARIFICATION":
            return self._plain_response(
                conversation_id,
                "请只提供一个有效商品编号，例如 H100、C20 或 P9。",
            )
        exact_confirmation = self._side_effect_policy.parse_exact_confirmation(question)
        expected_confirmation: ConfirmationChallenge | None = None
        if exact_confirmation is not None or self._side_effect_policy.is_bare_confirmation(question):
            expected_confirmation = await self._load_expected_confirmation(conversation_id)
        if self._side_effect_policy.is_bare_confirmation(question):
            if expected_confirmation is not None:
                return self._plain_response(conversation_id, expected_confirmation.prompt)
            return self._plain_response(
                conversation_id,
                "请同时提供明确动作和完整订单号，例如：确认退款 ORD12345678。",
            )

        action_type = (
            exact_confirmation.action_type if exact_confirmation is not None else classify_explicit_action(question)
        )
        if action_type is not None:
            return await self._handle_action_candidate(
                user=user,
                conversation_id=conversation_id,
                run_id=run_id,
                plan=build_rule_based_plan(question),
                action_type=action_type,
                exact_confirmation=exact_confirmation,
                expected_confirmation=expected_confirmation,
            )

        if self._plan_requests_side_effect(plan):
            plan = build_rule_based_plan(question)
        if plan.order_reference and plan.order_reference.product_keyword:
            plan = build_rule_based_plan(question)
        if plan.intent in {"SHIPPING_QUERY", "ORDER_QUERY"}:
            reference = plan.order_reference
            product_filter = self._runtime_order_product_filter(plan)
            has_explicit_order_target = bool(
                reference
                and (
                    reference.order_no
                    or reference.ordinal_index is not None
                )
            )
            if (
                plan.product_reference is not None
                and product_filter is None
                and not has_explicit_order_target
            ):
                return self._plain_response(
                    conversation_id,
                    "请只提供一个有效商品编号，例如 H100、C20 或 P9。",
                )
            if plan.order_reference and plan.order_reference.list_all:

                async def list_orders(args: ListMyOrdersArgs) -> tuple[OrderSnapshot, ...]:
                    return await self._resolve_orders_by_args(user, args)

                orders = cast(
                    tuple[OrderSnapshot, ...],
                    await self._tool_executor.execute_bound(
                        run_id,
                        user,
                        PRODUCTION_TOOL_BINDINGS[ProductionToolName.LIST_MY_ORDERS],
                        {"limit": 20},
                        list_orders,
                        lambda resolved: f"resolved {len(resolved)} orders",
                        self._tool_audit,
                    ),
                )
                if not orders:
                    return self._plain_response(conversation_id, "我这边暂时没有查到您的已下单商品。")
                return self._plain_response(conversation_id, self._order_list_answer(orders))
            if product_filter and not (reference and reference.latest):
                matched = await self._resolve_orders_by_args(
                    user,
                    ListMyOrdersArgs(product_keyword=product_filter, limit=20),
                )
                if len(matched) > 1:
                    return self._plain_response(conversation_id, self._multiple_order_answer(matched))

            async def get_order(args: GetOrderDetailArgs) -> OrderSnapshot | None:
                return await self._resolve_order_by_args(user, args)

            order = cast(
                OrderSnapshot | None,
                await self._tool_executor.execute_bound(
                    run_id,
                    user,
                    PRODUCTION_TOOL_BINDINGS[ProductionToolName.GET_ORDER_DETAIL],
                    self._order_tool_args(plan),
                    get_order,
                    lambda resolved: "resolved order" if resolved else "not found",
                    self._tool_audit,
                ),
            )
            if order is None:
                return self._plain_response(
                    conversation_id,
                    "我没有定位到对应订单。您可以说“最近订单”“第二个订单”，或直接提供订单号。",
                )
            return AgentAnswerOutcome(
                response=self._plain_response(
                    conversation_id,
                    self._order_answer(order, question),
                ),
                authorized_reads=(order,),
            )

        if plan.intent == "PRODUCT_QUERY" and plan.order_reference:

            async def order_product(args: GetOrderDetailArgs) -> OrderSnapshot | None:
                return await self._resolve_order_by_args(user, args)

            order = cast(
                OrderSnapshot | None,
                await self._tool_executor.execute_bound(
                    run_id,
                    user,
                    PRODUCTION_TOOL_BINDINGS[ProductionToolName.GET_ORDER_DETAIL],
                    self._order_tool_args(plan),
                    order_product,
                    lambda resolved: "resolved order" if resolved else "not found",
                    self._tool_audit,
                ),
            )
            if order is not None:
                sources = await self._product_sources(order.product)
                return AgentAnswerOutcome(
                    response=self._plain_response(
                        conversation_id,
                        f"我查到订单 {order.order_no} 对应的商品是「{order.product.product_name}」。"
                        + self._product_answer(order.product, include_name=False),
                        sources=sources,
                    ),
                    authorized_reads=(order,),
                )
        if plan.intent == "PRODUCT_QUERY" and plan.product_reference:

            async def get_product(args: GetProductInformationArgs) -> ProductSnapshot | None:
                return await self._resolve_product(args.product_keyword)

            product = cast(
                ProductSnapshot | None,
                await self._tool_executor.execute_bound(
                    run_id,
                    user,
                    PRODUCTION_TOOL_BINDINGS[ProductionToolName.GET_PRODUCT_INFORMATION],
                    {"product_keyword": plan.product_reference},
                    get_product,
                    lambda resolved: "resolved product" if resolved else "not found",
                    self._tool_audit,
                ),
            )
            if product is not None:
                return AgentAnswerOutcome(
                    response=self._plain_response(
                        conversation_id,
                        self._product_answer(product),
                        sources=await self._product_sources(product),
                    ),
                    authorized_reads=(product,),
                )

        retrieval_context = await self._build_retrieval_context(user, plan)

        async def retrieve(args: SearchKnowledgeBaseArgs) -> RetrievalResult:
            return await self._retrieve_knowledge(args.query, args.limit, retrieval_context)

        retrieval_result = cast(
            RetrievalResult,
            await self._tool_executor.execute_bound(
                run_id,
                user,
                PRODUCTION_TOOL_BINDINGS[ProductionToolName.SEARCH_KNOWLEDGE_BASE],
                {"query": question, "limit": 5},
                retrieve,
                lambda resolved: f"resolved {len(resolved.candidates)} retrieval candidates",
                self._tool_audit,
            ),
        )
        await self._persist_optional_retrieval_trace(
            conversation_id=conversation_id,
            subject_user_id=user.user_id,
            run_id=run_id,
            retrieval_result=retrieval_result,
        )
        order_for_answer = await self._resolve_order(user, plan) if plan.order_reference else None
        candidates = retrieval_result.candidates
        sources = self._source_references(candidates, question, order_for_answer)
        if candidates:
            best = candidates[0]
            response = ChatResponse(
                conversationId=conversation_id,
                answer=self._knowledge_answer(candidates, order_for_answer, question),
                sources=sources,
                retrievalScore=best.rerank_score or best.fused_score or best.original_score,
                confidenceLevel="MEDIUM" if (best.rerank_score or 0) >= 0.5 else "LOW",
                needHuman=False,
            )
            return (
                AgentAnswerOutcome(response=response, authorized_reads=(order_for_answer,))
                if order_for_answer is not None
                else response
            )
        response = self._plain_response(
            conversation_id,
            "这个问题我暂时没有足够依据直接回答。为了避免编造规则，建议转人工客服处理。",
            confidence_level="LOW",
            need_human=True,
        )
        return (
            AgentAnswerOutcome(response=response, authorized_reads=(order_for_answer,))
            if order_for_answer is not None
            else response
        )

    def _record_memory_operation(
        self,
        execution: ExecutionScope,
        *,
        operation: OperationCode,
        status: ObservabilityStatus,
        started_ns: int,
        error: BaseException | None = None,
    ) -> None:
        scope = self._observability_runtime.scope_for(execution.attempt_id)
        if scope is None:
            return
        self._observability_runtime.record(
            scope,
            MemoryOperationRecordV1(
                operation=operation,
                status=status,
                duration_ms=max(0, (time.monotonic_ns() - started_ns) // 1_000_000),
                error_type=(
                    normalized_error_type(error) if error is not None else None
                ),
            ),
        )

    def _record_revalidation_operation(
        self,
        execution: ExecutionScope,
        *,
        operation: OperationCode,
        status: ObservabilityStatus,
        action_type: str,
        started_ns: int,
        error: BaseException | None = None,
    ) -> None:
        scope = self._observability_runtime.scope_for(execution.attempt_id)
        if scope is None:
            return
        self._observability_runtime.record(
            scope,
            RevalidationOperationRecordV1(
                operation=operation,
                status=status,
                duration_ms=max(0, (time.monotonic_ns() - started_ns) // 1_000_000),
                action_type=self._observability_action_code(action_type),
                risk_level=RiskCode.HIGH,
                error_type=(
                    normalized_error_type(error) if error is not None else None
                ),
            ),
        )

    def _record_action_operation(
        self,
        execution: ExecutionScope,
        *,
        operation: OperationCode,
        status: ObservabilityStatus,
        action_type: str,
        started_ns: int,
        error: BaseException | None = None,
    ) -> None:
        scope = self._observability_runtime.scope_for(execution.attempt_id)
        if scope is None:
            return
        self._observability_runtime.record(
            scope,
            ActionOperationRecordV1(
                operation=operation,
                status=status,
                duration_ms=max(0, (time.monotonic_ns() - started_ns) // 1_000_000),
                retry_count=0,
                action_type=self._observability_action_code(action_type),
                risk_level=RiskCode.HIGH,
                error_type=(
                    normalized_error_type(error) if error is not None else None
                ),
            ),
        )

    @staticmethod
    def _observability_action_code(action_type: str) -> ActionCode:
        return {
            "ORDER_CANCELLATION": ActionCode.CANCEL_ORDER,
            "CANCEL_ORDER": ActionCode.CANCEL_ORDER,
            "REFUND": ActionCode.REQUEST_REFUND,
            "REFUND_REQUEST": ActionCode.REQUEST_REFUND,
        }.get(action_type, ActionCode.UNKNOWN)

    async def _persist_optional_retrieval_trace(
        self,
        *,
        conversation_id: int,
        subject_user_id: int,
        run_id: str,
        retrieval_result: RetrievalResult,
    ) -> None:
        async with self._unit_of_works.retrieval.open(
            operation=UnitOfWorkOperation.AGENT_RETRIEVAL_AUDIT
        ) as uow:
            await uow.store.assert_conversation_owned(
                conversation_id=conversation_id,
                subject_user_id=subject_user_id,
            )
        try:
            async with self._unit_of_works.retrieval.open(
                operation="agent.retrieval.audit"
            ) as uow:
                await uow.store.record_retrieval_trace(
                    run_id,
                    retrieval_result.candidates,
                    list(retrieval_result.diagnostics),
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            # Optional diagnostics use a distinct short UoW. Owner/authority
            # failures happen above and are intentionally never swallowed.
            return

    async def _handle_action_candidate(
        self,
        *,
        user: AuthenticatedUser,
        conversation_id: int,
        run_id: str,
        plan: AgentPlan,
        action_type: str,
        exact_confirmation: ConfirmationChallenge | None,
        expected_confirmation: ConfirmationChallenge | None,
    ) -> ChatResponse | DurableCustomerConfirmationOutcome:
        resolution = await self._resolve_action_target(user, plan)
        if resolution.candidates:
            return self._plain_response(
                conversation_id,
                self._multiple_action_target_answer(resolution.candidates),
            )
        order = resolution.order
        if order is None:
            return self._plain_response(
                conversation_id,
                "我无法验证要处理的订单。请提供您本人订单的完整订单号，或说明是最近订单/第几个订单。",
            )
        try:
            if self._durable_customer_interrupt_enabled:
                async with self._unit_of_works.confirmation.open(
                    operation=UnitOfWorkOperation.AGENT_CUSTOMER_CONFIRMATION_DATABASE_TIME
                ) as uow:
                    draft_time = await uow.store.customer_confirmation_database_time()
                draft = self._side_effect_policy.create_durable_action_draft(
                    run_id,
                    user,
                    order,
                    action_type,
                    now=draft_time,
                )
            else:
                draft = self._side_effect_policy.create_action_draft(
                    run_id,
                    user,
                    order,
                    action_type,
                )
        except SideEffectPolicyError:
            return self._plain_response(
                conversation_id,
                "当前订单状态或权限不满足该动作的安全条件，未创建任何处理请求。",
            )
        if self._durable_customer_interrupt_enabled:
            prompt = self._side_effect_policy.durable_confirmation_prompt(draft)
            return DurableCustomerConfirmationOutcome(
                response=self._plain_response(
                    conversation_id,
                    prompt,
                    confidence_level="HIGH",
                    need_human=True,
                ),
                action_draft=draft,
            )
        is_matching_confirmation = (
            exact_confirmation is not None
            and expected_confirmation is not None
            and expected_confirmation.matches(
                exact_confirmation.action_type,
                exact_confirmation.target_order_no,
            )
        )

        if not is_matching_confirmation:
            return self._plain_response(
                conversation_id,
                self._side_effect_policy.confirmation_prompt(draft),
            )
        assert exact_confirmation is not None
        authorization = self._side_effect_policy.authorize_action_prepare(
            run_id,
            user,
            draft,
            exact_confirmation.action_type,
            exact_confirmation.target_order_no,
        )
        action_tool = (
            ProductionToolName.REQUEST_ORDER_CANCELLATION
            if action_type == "ORDER_CANCELLATION"
            else ProductionToolName.REQUEST_REFUND
        )

        async def prepare(args: RequestOrderCancellationArgs | RequestRefundArgs) -> int:
            return await self._prepare_r2_action(
                user=user,
                run_id=run_id,
                order=order,
                action_type=action_type,
                args=args,
                authorization=authorization,
                logical_action_id=draft.logical_action_id,
            )

        await self._tool_executor.execute_bound(
            run_id,
            user,
            PRODUCTION_TOOL_BINDINGS[action_tool],
            {"order_no": order.order_no, "reason": draft.reason_code},
            prepare,
            lambda request_id: f"created pending approval request {request_id}",
            self._tool_audit,
            authorization=authorization,
            target_order_id=order.id,
            logical_action_id=draft.logical_action_id,
        )
        return self._plain_response(
            conversation_id,
            self._action_request_answer(order, action_type),
        )

    async def _prepare_r2_action(
        self,
        *,
        user: AuthenticatedUser,
        run_id: str,
        order: OrderSnapshot,
        action_type: str,
        args: RequestOrderCancellationArgs | RequestRefundArgs,
        authorization: SideEffectAuthorization,
        logical_action_id: str,
    ) -> int:
        if args.order_no.upper() != order.order_no.upper():
            raise SideEffectPolicyError("action arguments do not match resolved order")
        # Pre-UoW revocation gate: a revoked attempt (renewal failure,
        # cancellation, or any prior failure cleanup) must be rejected BEFORE
        # the order-validation UoW is opened, so neither the order lock nor
        # the action UoW can be reached from a dead attempt.
        execution = self._runtime_context_provider.current().execution
        execution.require_attempt_active()
        if (
            execution.run_id != run_id
            or execution.actor.user_id != user.user_id
            or execution.subject.user_id != user.user_id
        ):
            raise SideEffectPolicyError("action execution scope does not match the authenticated request")
        async with self._unit_of_works.action_prepare.open(operation="agent.r2_action_prepare") as uow:
            current = await uow.store.lock_order(user_id=user.user_id, order_no=order.order_no)
            if current is None:
                raise SideEffectPolicyError("action target is no longer owned by the authenticated subject")
            self._side_effect_policy.verify_tool_authorization(
                authorization,
                ToolAuthorizationContext(
                    run_id=run_id,
                    logical_action_id=logical_action_id,
                    subject_user_id=user.user_id,
                    tool_name=(
                        "request_order_cancellation" if action_type == "ORDER_CANCELLATION" else "request_refund"
                    ),
                    action_type=action_type,
                    target_order_id=current.id,
                    target_order_no=current.order_no.upper(),
                    effect_phase=EffectPhase.ACTION_PREPARE,
                    policy_version=POLICY_VERSION,
                ),
            )
            self._side_effect_policy.assert_action_allowed(user, current, action_type)
            execution.require_attempt_active()
        effects = self._replay_safe_effects
        if effects is None:
            raise RuntimeError("R2 action prepare requires replay-safe effect composition")
        result = await effects.write_action_prepare(
            execution,
            identity=EffectIdentity(
                run_id=run_id,
                node_name="r2_action_prepare",
                purpose=EffectPhase.ACTION_PREPARE.value,
                sequence=0,
            ),
            authorization=authorization,
            logical_action_id=logical_action_id,
            action_type=action_type,
            target_order_id=current.id,
            target_order_no=current.order_no,
            validated_order_status=current.status,
            action_payload={"reason": args.reason},
            risk_level="HIGH",
        )
        return result.target_id

    def _confirmation_effect_scope(self, execution: ExecutionScope) -> EffectWriteScope:
        fence = execution.lease.fence_token
        if type(fence) is not int or fence <= 0:
            raise SideEffectPolicyError(
                "customer confirmation marker requires a positive live fence"
            )
        execution.require_attempt_active()
        return EffectWriteScope(
            thread_id=execution.thread_id,
            attempt_id=execution.attempt_id,
            fence_version=fence,
        )

    async def _retrieve_knowledge(
        self,
        query: str,
        limit: int,
        context: RetrievalQueryContext,
    ) -> RetrievalResult:
        config = await self._load_runtime_config(UnitOfWorkOperation.AGENT_RETRIEVAL_CONFIG)

        async def keyword_loader(value: str, requested_limit: int) -> list[RetrievalCandidate]:
            async with self._unit_of_works.retrieval.open(operation="agent.retrieval.keyword") as uow:
                return await uow.store.keyword_recall(value, requested_limit)

        async def rule_loader(
            value: str,
            requested_limit: int,
            query_context: RetrievalQueryContext,
        ) -> list[RetrievalCandidate]:
            async with self._unit_of_works.retrieval.open(operation="agent.retrieval.rules") as uow:
                return await uow.store.structured_rule_recall(value, requested_limit, query_context)

        result = await knowledge_service.retrieve_with_ports(
            query,
            runtime=config,
            keyword_loader=keyword_loader,
            rule_loader=rule_loader,
            limit=limit,
            context=context,
        )
        current_config = await self._load_runtime_config(UnitOfWorkOperation.AGENT_RETRIEVAL_REVALIDATE)
        if current_config != config:
            raise RuntimeError("model runtime configuration changed during retrieval")
        return result

    async def _assemble_context_package(
        self,
        state: ConversationCheckpointState,
        current_turn: PlannerCurrentTurnV1 | AnswerCurrentTurnV1,
    ) -> ContextPackageV1:
        purpose = ContextPurpose.ANSWER if isinstance(current_turn, AnswerCurrentTurnV1) else ContextPurpose.PLANNER
        recent_summary = self._recent_summary
        if recent_summary is None:
            return build_single_turn_context_package(
                purpose=purpose,
                question=current_turn.question,
                evidence=(current_turn.evidence if isinstance(current_turn, AnswerCurrentTurnV1) else ()),
                draft_answer=(current_turn.draft_answer if isinstance(current_turn, AnswerCurrentTurnV1) else None),
                assembler=self._context_assembler,
            )
        active_run = self._require_active_run(state)
        if active_run.current_user_message_id is None:
            raise ValueError("context assembly requires the current user message")
        memory_started_ns = time.monotonic_ns()
        try:
            snapshot = await recent_summary.read_context(
                self._memory_scope(state),
                before_message_id=active_run.current_user_message_id,
            )
        except BaseException as error:
            self._record_current_memory_operation(
                operation=OperationCode.MEMORY_LOAD,
                status=(
                    ObservabilityStatus.CANCELLED
                    if isinstance(error, asyncio.CancelledError)
                    else ObservabilityStatus.FAILED
                ),
                started_ns=memory_started_ns,
                error=error,
            )
            raise
        self._record_current_memory_operation(
            operation=OperationCode.MEMORY_LOAD,
            status=ObservabilityStatus.SUCCEEDED,
            started_ns=memory_started_ns,
        )
        rolling_summary = (
            ContextRollingSummaryV1(
                summary=snapshot.rolling_summary,
                summary_text=snapshot.summary_text,
            )
            if snapshot.rolling_summary is not None and snapshot.summary_text is not None
            else None
        )
        request = ContextAssemblyRequestV1(
            purpose=purpose,
            rolling_summary=rolling_summary,
            working_memory=self._context_working_memory(state.memory),
            recent=snapshot.recent,
            current_turn=current_turn,
            token_counter_version=TOKEN_COUNTER_VERSION_V1,
        )
        try:
            return self._context_assembler.assemble(request)
        except ContextPartitionBudgetExceeded as error:
            if error.partition is not ContextPartitionName.ROLLING_SUMMARY or rolling_summary is None:
                raise
        fallback_started_ns = time.monotonic_ns()
        try:
            recent_fallback = await recent_summary.read_recent_without_summary(
                self._memory_scope(state),
                before_message_id=active_run.current_user_message_id,
            )
        except BaseException as error:
            self._record_current_memory_operation(
                operation=OperationCode.MEMORY_RECENT_FALLBACK,
                status=(
                    ObservabilityStatus.CANCELLED
                    if isinstance(error, asyncio.CancelledError)
                    else ObservabilityStatus.FAILED
                ),
                started_ns=fallback_started_ns,
                error=error,
            )
            raise
        self._record_current_memory_operation(
            operation=OperationCode.MEMORY_RECENT_FALLBACK,
            status=ObservabilityStatus.SUCCEEDED,
            started_ns=fallback_started_ns,
        )
        return self._context_assembler.assemble(
            request.model_copy(
                update={
                    "rolling_summary": None,
                    "recent": recent_fallback,
                }
            )
        )

    def _record_current_memory_operation(
        self,
        *,
        operation: OperationCode,
        status: ObservabilityStatus,
        started_ns: int,
        error: BaseException | None = None,
    ) -> None:
        try:
            execution = self._runtime_context_provider.current().execution
        except Exception:
            return
        self._record_memory_operation(
            execution,
            operation=operation,
            status=status,
            started_ns=started_ns,
            error=error,
        )

    def _context_working_memory(
        self,
        memory: ConversationMemoryState,
    ) -> ContextWorkingMemoryV1 | None:
        if memory.memory_revision == 0:
            if any(
                value is not None
                for value in (
                    memory.active_order_no,
                    memory.active_product_code,
                    memory.current_issue,
                    memory.last_intent,
                )
            ):
                raise ValueError("working memory values require a positive revision")
            return None
        return ContextWorkingMemoryV1(
            active_order_no=memory.active_order_no,
            active_product_code=memory.active_product_code,
            current_issue=memory.current_issue,
            last_intent=memory.last_intent,
            memory_revision=memory.memory_revision,
        )

    def _controlled_answer_evidence(
        self,
        active_run: ActiveRunState,
    ) -> tuple[ControlledEvidenceV1, ...]:
        return tuple(
            ControlledEvidenceV1(
                evidence_id=item.chunk_ref,
                content=item.snippet,
            )
            for item in active_run.retrieval_evidence
        )

    async def _load_runtime_config(
        self,
        operation: UnitOfWorkOperation,
    ) -> EffectiveModelRuntimeConfig:
        async with self._unit_of_works.runtime_config.open(operation=operation) as uow:
            return await uow.store.runtime_config()

    async def _polish_answer_with_llm(
        self,
        question: str,
        draft_answer: str,
        *,
        context: ContextPackageV1 | None = None,
    ) -> str:
        metered_context = (
            context
            or bound_context_package(ContextPurpose.ANSWER)
            or build_single_turn_context_package(
                purpose=ContextPurpose.ANSWER,
                question=question,
                draft_answer=draft_answer,
                assembler=self._context_assembler,
            )
        )
        config = await self._load_runtime_config(UnitOfWorkOperation.AGENT_ANSWER_CONFIG)
        client = create_llm_client(config)
        scope = None
        try:
            runtime = self._runtime_context_provider.current()
        except Exception:
            runtime = None
        if runtime is not None:
            scope = self._observability_runtime.scope_for(runtime.execution.attempt_id)
        try:
            require_no_active_transaction("answer and HTTP LLM call")
            if scope is None:
                grounded = await client.answer(metered_context, "", "")
            else:
                invocation = await invoke_answer_observed(
                    client,
                    metered_context,
                    "",
                    "",
                    fallback_model_family=(
                        LLMModelFamilyV1.MOCK
                        if config.mock_enabled
                        else LLMModelFamilyV1.OPENAI_COMPATIBLE
                    ),
                )
                grounded = invocation.value
                self._record_llm_invocation(
                    scope,
                    OperationCode.LLM_ANSWER,
                    invocation,
                )
        except LLMProviderError as error:
            if scope is not None:
                self._record_llm_failure(
                    scope,
                    OperationCode.LLM_ANSWER,
                    failure_from_provider_error(error),
                )
            return draft_answer
        current_config = await self._load_runtime_config(UnitOfWorkOperation.AGENT_ANSWER_REVALIDATE)
        if current_config != config:
            raise RuntimeError("model runtime configuration changed during answer generation")
        if grounded is None:
            return draft_answer
        return grounded.answer or draft_answer

    def _record_llm_capture(
        self,
        scope: AttemptScopeV1,
        operation: OperationCode,
        capture: LLMInvocationCaptureV1,
    ) -> None:
        for result in capture.results:
            self._record_llm_invocation(scope, operation, result)
        for failure in capture.failures:
            self._record_llm_failure(scope, operation, failure)

    def _record_llm_invocation(
        self,
        scope: AttemptScopeV1,
        operation: OperationCode,
        result: LLMInvocationResultV1[object],
    ) -> None:
        usage = result.usage
        invalid_response = (
            result.outcome is LLMInvocationOutcomeV1.INVALID_RESPONSE
        )
        self._observability_runtime.record(
            scope,
            LLMOperationRecordV1(
                operation=operation,
                status=(
                    ObservabilityStatus.FAILED
                    if invalid_response
                    else ObservabilityStatus.SUCCEEDED
                ),
                duration_ms=result.duration_ms,
                retry_count=result.retry_count,
                model_family=ModelFamily(result.model_family.value),
                prompt_tokens=usage.prompt_tokens if usage is not None else None,
                completion_tokens=(
                    usage.completion_tokens if usage is not None else None
                ),
                error_type=(
                    NormalizedErrorTypeV1(
                        NormalizedErrorCode.INTEGRITY_FAILURE
                    )
                    if invalid_response
                    else None
                ),
            ),
        )

    def _record_llm_failure(
        self,
        scope: AttemptScopeV1,
        operation: OperationCode,
        failure: LLMInvocationFailureV1,
    ) -> None:
        error_code = {
            LLMInvocationFailureCodeV1.TIMEOUT: NormalizedErrorCode.TIMEOUT,
            LLMInvocationFailureCodeV1.UNAVAILABLE: NormalizedErrorCode.UNAVAILABLE,
            LLMInvocationFailureCodeV1.CONFIGURATION: NormalizedErrorCode.UNAVAILABLE,
            LLMInvocationFailureCodeV1.UNKNOWN: NormalizedErrorCode.UNKNOWN,
        }[failure.code]
        self._observability_runtime.record(
            scope,
            LLMOperationRecordV1(
                operation=operation,
                status=ObservabilityStatus.FAILED,
                duration_ms=failure.duration_ms,
                retry_count=failure.retry_count,
                model_family=ModelFamily(failure.model_family.value),
                error_type=NormalizedErrorTypeV1(error_code),
            ),
        )

    async def _load_expected_confirmation(
        self,
        conversation_id: int,
    ) -> ConfirmationChallenge | None:
        async with self._unit_of_works.confirmation.open(operation="agent.confirmation.read") as uow:
            content = await uow.store.latest_assistant_message(conversation_id)
        return self._side_effect_policy.parse_confirmation_prompt(content) if content else None

    async def _resolve_action_target(
        self,
        user: AuthenticatedUser,
        plan: AgentPlan,
    ) -> ActionTargetResolution:
        reference = plan.order_reference
        if reference is not None and reference.product_keyword:
            return ActionTargetResolution(order=None)
        if reference is not None and reference.order_no:
            return ActionTargetResolution(
                order=await self._resolve_order_by_args(
                    user,
                    GetOrderDetailArgs(order_no=reference.order_no),
                )
            )
        if reference is not None and reference.ordinal_index is not None:
            orders = await self._resolve_orders_by_args(user, ListMyOrdersArgs(limit=20))
            order = orders[reference.ordinal_index] if reference.ordinal_index < len(orders) else None
            return ActionTargetResolution(order=order)
        product_filter = self._runtime_order_product_filter(plan)
        if plan.product_reference is not None and product_filter is None:
            return ActionTargetResolution(order=None)
        if product_filter:
            orders = await self._resolve_orders_by_args(
                user,
                ListMyOrdersArgs(product_keyword=product_filter, limit=20),
            )
            if reference is not None and reference.latest:
                return ActionTargetResolution(order=orders[0] if orders else None)
            if len(orders) == 1:
                return ActionTargetResolution(order=orders[0])
            if len(orders) > 1:
                return ActionTargetResolution(order=None, candidates=orders)
            return ActionTargetResolution(order=None)
        if reference is not None and reference.latest:
            orders = await self._resolve_orders_by_args(user, ListMyOrdersArgs(limit=1))
            return ActionTargetResolution(order=orders[0] if orders else None)
        return ActionTargetResolution(order=None)

    async def _resolve_order(
        self,
        user: AuthenticatedUser,
        plan: AgentPlan,
    ) -> OrderSnapshot | None:
        return await self._resolve_order_by_args(
            user,
            GetOrderDetailArgs.model_validate(self._order_tool_args(plan)),
        )

    async def _resolve_order_by_args(
        self,
        user: AuthenticatedUser,
        args: GetOrderDetailArgs,
    ) -> OrderSnapshot | None:
        async with self._unit_of_works.orders.open(operation="agent.order.read") as uow:
            if args.order_no:
                return await uow.store.get_order(user_id=user.user_id, order_no=args.order_no)
            orders = await uow.store.list_orders(
                user_id=user.user_id,
                limit=20,
                product_keyword=args.product_keyword,
            )
        if not orders:
            return None
        if args.ordinal_index is not None:
            return orders[args.ordinal_index] if args.ordinal_index < len(orders) else None
        return orders[0]

    async def _resolve_orders_by_args(
        self,
        user: AuthenticatedUser,
        args: ListMyOrdersArgs,
    ) -> tuple[OrderSnapshot, ...]:
        async with self._unit_of_works.orders.open(operation="agent.orders.read") as uow:
            return await uow.store.list_orders(
                user_id=user.user_id,
                limit=args.limit,
                status=args.status,
                product_keyword=args.product_keyword,
            )

    async def _resolve_product(self, keyword: str) -> ProductSnapshot | None:
        async with self._unit_of_works.products.open(operation="agent.product.read") as uow:
            return await uow.store.get_product(keyword)

    async def _product_sources(self, product: ProductSnapshot) -> list[SourceReference]:
        async with self._unit_of_works.products.open(operation="agent.product.sources") as uow:
            return list(await uow.store.product_sources(product))

    def _memory_tool_results(
        self,
        authorized_reads: tuple[OrderSnapshot | ProductSnapshot, ...],
        *,
        observed_at: datetime,
    ) -> list[ToolResultSnapshot]:
        timestamp = self._rfc3339(observed_at)
        results: list[ToolResultSnapshot] = []
        for snapshot in authorized_reads:
            if type(snapshot) is OrderSnapshot:
                results.append(
                    ToolResultSnapshot(
                        tool_name="get_order_detail",
                        status=ToolResultStatus.SUCCEEDED,
                        result_ref=f"ORDER:{snapshot.id}:{snapshot.order_no}",
                        safe_metadata={
                            "owner_verified": True,
                            "status": "FOUND",
                        },
                        observed_at=timestamp,
                    )
                )
            elif type(snapshot) is ProductSnapshot:
                results.append(
                    ToolResultSnapshot(
                        tool_name="get_product_information",
                        status=ToolResultStatus.SUCCEEDED,
                        result_ref=(f"PRODUCT:{snapshot.id}:{snapshot.product_code}"),
                        safe_metadata={
                            "product_code": snapshot.product_code,
                            "status": "FOUND",
                        },
                        observed_at=timestamp,
                    )
                )
            else:
                raise ValueError("authorized memory read type is invalid")
        return results

    async def _revalidate_memory_tool_results(
        self,
        runtime: AgentRuntimeContext,
        tool_results: list[ToolResultSnapshot],
        *,
        source_message_id: int,
    ) -> tuple[AuthorizedOrderReadResultV1 | AuthorizedProductReadResultV1, ...]:
        actor = self._authenticated_actor(runtime)
        evidence: list[AuthorizedOrderReadResultV1 | AuthorizedProductReadResultV1] = []
        for result in tool_results:
            if result.tool_name == "get_order_detail":
                match = re.fullmatch(
                    r"ORDER:([1-9][0-9]*):([A-Za-z0-9][A-Za-z0-9_.-]{0,127})",
                    result.result_ref or "",
                )
                if (
                    result.status is not ToolResultStatus.SUCCEEDED
                    or dict(result.safe_metadata) != {"owner_verified": True, "status": "FOUND"}
                    or match is None
                ):
                    raise ValueError("working memory read evidence is invalid")
                order = await self._resolve_order_by_args(
                    actor,
                    GetOrderDetailArgs(order_no=match.group(2)),
                )
                if order is None or order.id != int(match.group(1)):
                    raise ValueError("working memory read evidence is invalid")
                evidence.append(
                    AuthorizedOrderReadResultV1(
                        subject_user_id=actor.user_id,
                        source_message_id=source_message_id,
                        order_id=order.id,
                        order_no=order.order_no,
                        product_id=order.product.id,
                        product_code=order.product.product_code,
                    )
                )
            elif result.tool_name == "get_product_information":
                match = re.fullmatch(
                    r"PRODUCT:([1-9][0-9]*):([A-Za-z0-9][A-Za-z0-9_.-]{0,127})",
                    result.result_ref or "",
                )
                metadata = dict(result.safe_metadata)
                if (
                    result.status is not ToolResultStatus.SUCCEEDED
                    or match is None
                    or metadata != {"product_code": match.group(2), "status": "FOUND"}
                ):
                    raise ValueError("working memory read evidence is invalid")
                product = await self._resolve_product(match.group(2))
                if product is None or product.id != int(match.group(1)) or product.product_code != match.group(2):
                    raise ValueError("working memory read evidence is invalid")
                evidence.append(
                    AuthorizedProductReadResultV1(
                        subject_user_id=actor.user_id,
                        source_message_id=source_message_id,
                        product_id=product.id,
                        product_code=product.product_code,
                    )
                )
        return tuple(evidence)

    async def _resolve_structured_memory_context(
        self,
        runtime: AgentRuntimeContext,
        memory: ConversationMemoryState,
        question: str,
    ) -> str:
        resolution = await self._resolve_structured_memory_context_resolution(
            runtime,
            memory,
            question,
        )
        return resolution.effective_question

    async def _resolve_structured_memory_context_resolution(
        self,
        runtime: AgentRuntimeContext,
        memory: ConversationMemoryState,
        question: str,
    ) -> _StructuredMemoryContextResolution:
        clean = question.strip()
        if self._has_explicit_order_identifier(clean) or has_explicit_product_reference(clean):
            return _StructuredMemoryContextResolution(
                effective_question=question,
                allow_legacy_fallback=False,
            )
        current_plan = build_rule_based_plan(clean)
        current_order_reference = current_plan.order_reference
        if current_order_reference is not None and any(
            (
                current_order_reference.order_no,
                current_order_reference.ordinal_index is not None,
                current_order_reference.latest,
                current_order_reference.list_all,
            )
        ):
            return _StructuredMemoryContextResolution(
                effective_question=question,
                allow_legacy_fallback=False,
            )
        target = self._structured_memory_reference_target(clean)
        if target == "AMBIGUOUS":
            return _StructuredMemoryContextResolution(
                effective_question=question,
                allow_legacy_fallback=False,
            )
        actor = self._authenticated_actor(runtime)
        if target == "ORDER" and memory.active_order_no:
            order = await self._resolve_order_by_args(
                actor,
                GetOrderDetailArgs(order_no=memory.active_order_no),
            )
            if order is not None:
                return _StructuredMemoryContextResolution(
                    effective_question=f"订单 {order.order_no} {clean}",
                    allow_legacy_fallback=False,
                )
            return _StructuredMemoryContextResolution(
                effective_question=question,
                allow_legacy_fallback=False,
            )
        if target == "PRODUCT" and memory.active_product_code:
            product = await self._resolve_product(memory.active_product_code)
            if product is not None and product.product_code == memory.active_product_code:
                return _StructuredMemoryContextResolution(
                    effective_question=f"商品 {product.product_code} {clean}",
                    allow_legacy_fallback=False,
                )
            return _StructuredMemoryContextResolution(
                effective_question=question,
                allow_legacy_fallback=False,
            )
        return _StructuredMemoryContextResolution(
            effective_question=question,
            allow_legacy_fallback=True,
        )

    async def _revalidate_legacy_context_candidate(
        self,
        runtime: AgentRuntimeContext,
        candidate: LegacyContextCandidate,
        question: str,
    ) -> str:
        actor = self._authenticated_actor(runtime)
        if candidate.kind == "ORDER":
            order = await self._resolve_order_by_args(
                actor,
                GetOrderDetailArgs(order_no=candidate.value),
            )
            return f"订单 {order.order_no} {question.strip()}" if order is not None else question
        if candidate.kind == "PRODUCT":
            product = await self._resolve_product(candidate.value)
            return f"商品 {product.product_code} {question.strip()}" if product is not None else question
        return question

    def _memory_scope(
        self,
        state: ConversationCheckpointState,
    ) -> ConversationMemoryScopeV1:
        identity = state.conversation_identity
        return ConversationMemoryScopeV1(
            conversation_id=identity.conversation_id,
            subject_user_id=identity.subject_user_id,
        )

    def _replace_working_memory(
        self,
        state: ConversationCheckpointState,
        loaded: LoadedWorkingMemoryV1,
    ) -> ConversationCheckpointState:
        persisted = loaded.memory
        existing = state.memory
        if persisted is None:
            memory = ConversationMemoryState(
                conversation_summary=existing.conversation_summary,
                summary_until_message_id=existing.summary_until_message_id,
                summary_revision=existing.summary_revision,
            )
        else:
            memory = ConversationMemoryState(
                active_order_no=persisted.active_order_no,
                active_product_code=persisted.active_product_code,
                current_issue=persisted.current_issue,
                last_intent=persisted.last_intent,
                provenance=[
                    MemoryProvenanceRecord(
                        field_name=item.field_name,
                        source_type=item.source_kind,
                        source_ref=item.source_reference,
                        observed_at=self._rfc3339(item.observed_at),
                        memory_revision=item.memory_revision,
                    )
                    for item in loaded.runtime_provenance
                ],
                memory_revision=persisted.memory_revision,
                conversation_summary=existing.conversation_summary,
                summary_until_message_id=existing.summary_until_message_id,
                summary_revision=existing.summary_revision,
            )
        return state.model_copy(update={"memory": memory})

    def _require_working_memory(self) -> WorkingMemoryApplicationPort:
        if self._working_memory is None:
            raise RuntimeError("AgentService working memory composition is not initialized")
        return self._working_memory

    def _rfc3339(self, value: datetime) -> str:
        current = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return current.astimezone(UTC).isoformat().replace("+00:00", "Z")

    async def _build_retrieval_context(
        self,
        user: AuthenticatedUser,
        plan: AgentPlan,
    ) -> RetrievalQueryContext:
        if plan.order_reference is None:
            return RetrievalQueryContext()
        order = await self._resolve_order(user, plan)
        if order is None:
            return RetrievalQueryContext()
        signed_days = (datetime.now() - order.signed_at).days if order.signed_at else None
        return RetrievalQueryContext(
            product_category=order.product.category,
            order_status=order.status,
            payment_status="PAID" if order.paid_at else "UNPAID",
            shipment_status=_shipment_status(order.status),
            signed_days=signed_days,
            has_specific_order=True,
        )

    def _resolve_conversation_context(self, messages: tuple[str, ...], question: str) -> str:
        del messages
        return question

    def _legacy_context_candidate(
        self,
        messages: tuple[str, ...],
        question: str,
    ) -> LegacyContextCandidate | None:
        if re.search(r"ORD[0-9A-Z]{8,}", question, flags=re.IGNORECASE):
            return None
        if not messages:
            return None
        index = self._short_choice_index(question)
        if index is not None:
            resolved = self._resolve_order_choice_by_index(messages, question, index)
            match = re.search(r"ORD[0-9A-Z]+", resolved, flags=re.IGNORECASE)
            return LegacyContextCandidate(kind="ORDER", value=match.group(0).upper()) if match is not None else None
        clean = question.strip()
        has_current_product = has_explicit_product_reference(clean)
        if self._order_context_follow_up(clean) and not has_current_product:
            order_no = self._latest_order_no(messages)
            if order_no:
                return LegacyContextCandidate(kind="ORDER", value=order_no)
        if self._product_context_follow_up(clean) and not has_current_product:
            product_name = self._latest_product_name(messages)
            if product_name:
                return LegacyContextCandidate(kind="PRODUCT", value=product_name)
        return None

    def _resolve_order_choice_by_index(
        self,
        messages: tuple[str, ...],
        question: str,
        index: int,
    ) -> str:
        for content in messages:
            if "多笔订单" not in content:
                continue
            order_nos = re.findall(r"订单\s+(ORD[0-9A-Z]+)", content, flags=re.IGNORECASE)
            if index < len(order_nos):
                return f"订单 {order_nos[index].upper()} 物流到哪里了"
        return question

    def _latest_order_no(self, messages: tuple[str, ...]) -> str | None:
        for content in messages:
            match = re.search(r"我查到订单\s+(ORD[0-9A-Z]+)", content, flags=re.IGNORECASE)
            if match:
                return str(match.group(1)).upper()
        for content in messages:
            order_nos = re.findall(r"订单\s+(ORD[0-9A-Z]+)", content, flags=re.IGNORECASE)
            if len(order_nos) == 1:
                return str(order_nos[0]).upper()
        return None

    def _latest_product_name(self, messages: tuple[str, ...]) -> str | None:
        for content in messages:
            match = re.search(r"「([^」]+)」", content)
            if match:
                return str(match.group(1))
        return None

    def _order_context_follow_up(self, question: str) -> bool:
        if not 0 < len(question) <= 24:
            return False
        return self._structured_memory_reference_target(question) == "ORDER"

    def _product_context_follow_up(self, question: str) -> bool:
        if not 0 < len(question) <= 18:
            return False
        return self._structured_memory_reference_target(question) == "PRODUCT"

    def _short_choice_index(self, question: str) -> int | None:
        clean = question.strip()
        if clean.isdigit():
            value = int(clean)
            return value - 1 if value > 0 else None
        return {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4}.get(clean)

    def _references_prior_object(self, question: str) -> bool:
        return any(token in question for token in ("它", "这个", "那个", "这单", "那单", "该订单"))

    def _has_explicit_order_identifier(self, question: str) -> bool:
        return (
            re.search(
                r"(?<![A-Z0-9])ORD[0-9A-Z]{8,252}(?![A-Z0-9])",
                question,
                flags=re.IGNORECASE,
            )
            is not None
        )

    def _structured_memory_reference_target(
        self,
        question: str,
    ) -> Literal["ORDER", "PRODUCT", "AMBIGUOUS", "NONE"]:
        explicit_product_cues = (
            "这个商品",
            "那个商品",
            "该商品",
        )
        explicit_order_cues = (
            "这个订单",
            "那个订单",
            "这单",
            "那单",
            "该订单",
        )
        has_explicit_product = any(
            cue in question for cue in explicit_product_cues
        )
        has_explicit_order = any(
            cue in question for cue in explicit_order_cues
        )
        if has_explicit_product and has_explicit_order:
            return "AMBIGUOUS"
        if has_explicit_product:
            return "PRODUCT"
        if has_explicit_order:
            return "ORDER"
        if "订单状态" in question:
            return "ORDER"

        product_cues = (
            "价格",
            "多少钱",
            "库存",
            "还有货",
            "在售",
            "参数",
            "介绍",
            "商品资料",
            "商品信息",
        )
        order_cues = (
            "订单状态",
            "物流",
            "快递",
            "发货",
            "出库",
            "包裹",
            "没动静",
            "取消",
        )
        has_product_cue = any(cue in question for cue in product_cues)
        has_order_cue = any(cue in question for cue in order_cues)
        has_neutral_reference = any(
            token in question for token in ("它", "这个", "那个")
        )
        if has_neutral_reference:
            if has_product_cue and has_order_cue:
                return "AMBIGUOUS"
            if has_product_cue:
                return "PRODUCT"
            if has_order_cue:
                return "ORDER"
            return "AMBIGUOUS"

        current_plan = build_rule_based_plan(question)
        if (
            current_plan.intent == "KNOWLEDGE_QUERY"
            and current_plan.order_reference is None
            and current_plan.product_reference is None
            and current_plan.required_tools == ["search_knowledge_base"]
        ):
            return "NONE"
        if has_product_cue and has_order_cue:
            return "AMBIGUOUS"
        if has_product_cue:
            return "PRODUCT"
        if has_order_cue:
            return "ORDER"
        return "NONE"

    def _plan_requests_side_effect(self, plan: AgentPlan) -> bool:
        if plan.intent in {"CANCEL_ORDER", "REFUND_REQUEST"} or plan.action_type is not None:
            return True
        for tool_name in plan.required_tools:
            definition = TOOL_REGISTRY.get(tool_name)
            if definition is None or definition.policy.effect_phase is not EffectPhase.READ_ONLY:
                return True
        return False

    def _order_tool_args(self, plan: AgentPlan) -> dict[str, object]:
        reference = plan.order_reference
        order_no = reference.order_no if reference is not None else None
        ordinal_index = reference.ordinal_index if reference is not None else None
        has_explicit_order_target = bool(
            order_no or ordinal_index is not None
        )
        product_filter = (
            None
            if has_explicit_order_target
            else self._runtime_order_product_filter(plan)
        )
        return {
            "order_no": order_no,
            "ordinal_index": ordinal_index,
            "product_keyword": product_filter,
            "latest": bool(reference and reference.latest)
            or not any(
                [
                    order_no,
                    ordinal_index is not None,
                    product_filter,
                ]
            ),
        }

    def _runtime_order_product_filter(self, plan: AgentPlan) -> str | None:
        product_reference = plan.product_reference
        if product_reference is None:
            return None
        resolved = extract_product_reference(product_reference)
        return product_reference if resolved == product_reference else None

    def _require_active_run(self, state: ConversationCheckpointState) -> ActiveRunState:
        if state.active_run is None:
            raise ValueError("agent workflow requires an active run")
        return state.active_run

    def _authenticated_actor(self, runtime: AgentRuntimeContext) -> AuthenticatedUser:
        actor = runtime.execution.actor
        return AuthenticatedUser(
            user_id=actor.user_id,
            username=actor.username,
            name=actor.display_name,
            role=actor.role,
        )

    def _agent_plan_from_snapshot(self, snapshot: PlanSnapshot) -> AgentPlan:
        payload = snapshot.model_dump(mode="python")
        payload["requires_confirmation"] = payload.pop("confirmation_required")
        return AgentPlan.model_validate(payload)

    def _response_meta(self, response: ChatResponse) -> ResponseMeta:
        sources = [
            RetrievalEvidence(
                document_id=source.documentId,
                chunk_ref=f"document:{source.documentId}:{index}",
                file_name=source.fileName,
                snippet=source.snippet,
                score=source.score,
                channel=RetrievalChannel.RERANKED,
            )
            for index, source in enumerate(response.sources)
        ]
        return ResponseMeta(
            sources=sources,
            retrieval_score=response.retrievalScore,
            confidence_level=response.confidenceLevel,
            need_human=response.needHuman,
            ticket_id=response.ticketId,
        )

    def _chat_response_from_state(self, state: ConversationCheckpointState) -> ChatResponse:
        active_run = self._require_active_run(state)
        if active_run.final_answer is None or active_run.response_meta is None:
            raise ValueError("stable state does not contain a completed response")
        meta = active_run.response_meta
        durable_confirmation = (
            getattr(self, "_durable_customer_interrupt_enabled", False)
            and active_run.action_draft is not None
            and active_run.run_status
            in {
                RunStatus.WAITING_CUSTOMER_CONFIRMATION,
                RunStatus.RESUME_PENDING,
            }
        )
        confirmation_prompt = None
        confirmation_digest = None
        if durable_confirmation:
            assert active_run.action_draft is not None
            confirmation_prompt = self._side_effect_policy.durable_confirmation_prompt(
                PolicyActionDraftSnapshot(**active_run.action_draft.model_dump(mode="python"))
            )
            confirmation_digest = active_run.action_draft.nonce_digest
        return ChatResponse(
            conversationId=state.conversation_identity.conversation_id,
            answer=active_run.final_answer,
            sources=[
                SourceReference(
                    documentId=source.document_id,
                    fileName=source.file_name,
                    snippet=source.snippet,
                    score=source.score,
                )
                for source in meta.sources
            ],
            retrievalScore=meta.retrieval_score,
            confidenceLevel=meta.confidence_level,
            needHuman=meta.need_human,
            ticketId=meta.ticket_id,
            agentStatus=active_run.run_status.value if durable_confirmation else None,
            confirmationPrompt=confirmation_prompt,
            confirmationChallengeDigest=confirmation_digest,
        )

    def _database_time(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value
        return value.astimezone(UTC).replace(tzinfo=None)

    def _compact(self, value: str, limit: int = 240) -> str:
        normalized = " ".join(value.split())
        return normalized if len(normalized) <= limit else normalized[: limit - 3] + "..."

    def _runtime_model_name(self, runtime: EffectiveModelRuntimeConfig) -> str:
        return "mock-llm" if runtime.mock_enabled else settings.llm_model_name

    def _runtime_config_version(self, runtime: EffectiveModelRuntimeConfig) -> str:
        return (
            f"temperature={runtime.temperature};"
            f"top_k={runtime.top_k};"
            f"min_score={runtime.min_retrieval_score};"
            f"mock={runtime.mock_enabled}"
        )


def _shipment_status(order_status: str) -> str:
    if order_status in {"SHIPPED", "IN_TRANSIT", "SIGNED"}:
        return "SHIPPED"
    return "UNSHIPPED"
