"""Process-lifetime composition root for the customer Agent runtime."""

from __future__ import annotations

from typing import Any, cast

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agent.checkpoint_projection import (
    AgentCheckpointStateProjector,
    AgentPendingWriteProjector,
    build_tool_metadata_policy_catalog,
)
from app.agent.workflow import build_customer_service_graph
from app.core.config import Settings
from app.db.session import session_factory
from app.memory import (
    ContextAssemblerV1,
    ContextBudgetConfigV1,
    ConversationMemoryStorePort,
    CurrentQuestionRequestGuard,
    DeterministicSummaryGeneratorV1,
    MemoryBudgetConfigV1,
    build_rolling_summary_protection,
)
from app.observability import (
    LocalJsonTelemetrySink,
    ProtectedMetricsRecorder,
    ProtectedObservability,
    build_metric_protection,
    build_observability_protection,
)
from app.repositories.admin_decision_repository import (
    SqlAlchemyAdminDecisionSecurityAuditStore,
    SqlAlchemyAdminDecisionStore,
)
from app.repositories.agent_workflow_repository import (
    AgentWorkflowTransactionStore,
    ConfirmationStorePort,
    SqlAlchemyAgentWorkflowStore,
)
from app.repositories.business_execution_repository import (
    SqlAlchemyBusinessExecutionStore,
)
from app.repositories.checkpoint_content_source_repository import (
    SqlAlchemyCheckpointContentSourceStore,
)
from app.repositories.conversation_memory_repository import (
    SqlAlchemyConversationMemoryStore,
)
from app.repositories.durable_runtime_repository import SqlAlchemyDurableRuntimeStore
from app.repositories.mysql_transaction_retry import MysqlTransactionErrorClassifier
from app.runtime.admin_decision import AdminDecisionStorePort
from app.runtime.admin_decision_audit import (
    AdminDecisionSecurityAuditStorePort,
    build_admin_decision_audit_protection,
)
from app.runtime.business_execution import (
    BusinessExecutionStorePort,
    build_business_execution_audit_protection,
)
from app.runtime.checkpoint_runtime import CheckpointRuntimeCompositionHandle
from app.runtime.content_source import ContentSourceAuthorityStorePort
from app.runtime.context import RuntimeContextProvider
from app.runtime.durable import (
    ActionPrepareStorePort,
    AttemptLeaseStorePort,
    AuditEffectStorePort,
    LogicalRunStorePort,
    MessageEffectStorePort,
    PreparedActionValidationStorePort,
    PublicationStorePort,
)
from app.runtime.observability_runtime import ObservabilityRuntime
from app.runtime.safe_checkpoint_bridge import (
    AsyncCheckpointSaverPort,
    ProtectedCheckpointBridge,
)
from app.runtime.single_flight import PerThreadSingleFlight
from app.runtime.uow import (
    ApplicationTransactionCoordinator,
    ApplicationUnitOfWorkFactory,
    SqlAlchemyApplicationUnitOfWorkFactory,
)
from app.services.admin_decision_application import (
    AdminDecisionApplicationService,
    AdminDecisionGraphPort,
    AdminDecisionReconciler,
    AdminDecisionResumeCapabilityProvider,
    AdminDecisionSecurityAuditApplication,
)
from app.services.business_execution_application import (
    DurableBusinessExecutionApplication,
)
from app.services.checkpoint_content_source_service import ContentSourceAuthorityService
from app.services.customer_agent_application import AgentApplicationUnitOfWorks, AgentService
from app.services.customer_confirmation_application import (
    CustomerConfirmationApplicationService,
    CustomerConfirmationCheckpointReaderPort,
    CustomerConfirmationGraphPort,
    CustomerResumeCapabilityProvider,
)
from app.services.durable_runtime_service import (
    DurableRuntimeAuthorityService,
    ReplaySafeEffectService,
)
from app.services.prepared_action_validation import PreparedActionValidationService
from app.services.recent_summary_application import RecentMessagesRollingSummaryService
from app.services.side_effect_policy_service import side_effect_policy_service
from app.services.working_memory_application import WorkingMemoryApplicationService


def _build_local_observability_runtime() -> ObservabilityRuntime:
    """Create the process-local protected telemetry runtime."""

    local_telemetry_sink = LocalJsonTelemetrySink()
    return ObservabilityRuntime(
        ProtectedObservability(
            protection=build_observability_protection(),
            sink=local_telemetry_sink,
        ),
        ProtectedMetricsRecorder(
            protection=build_metric_protection(),
            sink=local_telemetry_sink,
        ),
    )


def build_managed_agent_service(
    settings: Settings,
    checkpoint_runtime: CheckpointRuntimeCompositionHandle,
    *,
    database_session_factory: async_sessionmaker[AsyncSession] | None = None,
) -> AgentService:
    """Create one bridge, compiled graph, and AgentService for one lifespan."""

    maker = database_session_factory or session_factory()
    workflow_uow = cast(
        ApplicationUnitOfWorkFactory[AgentWorkflowTransactionStore],
        SqlAlchemyApplicationUnitOfWorkFactory(
            maker,
            SqlAlchemyAgentWorkflowStore,
        ),
    )
    durable_uow_base = SqlAlchemyApplicationUnitOfWorkFactory(
        maker,
        SqlAlchemyDurableRuntimeStore,
    )
    source_uow = cast(
        ApplicationUnitOfWorkFactory[ContentSourceAuthorityStorePort],
        SqlAlchemyApplicationUnitOfWorkFactory(
            maker,
            SqlAlchemyCheckpointContentSourceStore,
        ),
    )
    memory_uow = cast(
        ApplicationUnitOfWorkFactory[ConversationMemoryStorePort],
        SqlAlchemyApplicationUnitOfWorkFactory(
            maker,
            SqlAlchemyConversationMemoryStore,
        ),
    )
    admin_decision_uow = cast(
        ApplicationUnitOfWorkFactory[AdminDecisionStorePort],
        SqlAlchemyApplicationUnitOfWorkFactory(
            maker,
            SqlAlchemyAdminDecisionStore,
        ),
    )
    admin_decision_audit_uow = cast(
        ApplicationUnitOfWorkFactory[AdminDecisionSecurityAuditStorePort],
        SqlAlchemyApplicationUnitOfWorkFactory(
            maker,
            SqlAlchemyAdminDecisionSecurityAuditStore,
        ),
    )
    business_execution_uow = cast(
        ApplicationUnitOfWorkFactory[BusinessExecutionStorePort],
        SqlAlchemyApplicationUnitOfWorkFactory(
            maker,
            SqlAlchemyBusinessExecutionStore,
        ),
    )
    transactions = ApplicationTransactionCoordinator(MysqlTransactionErrorClassifier())
    attempt_uow = cast(
        ApplicationUnitOfWorkFactory[AttemptLeaseStorePort],
        durable_uow_base,
    )
    durable_authority = DurableRuntimeAuthorityService(
        attempt_uow,
        cast(ApplicationUnitOfWorkFactory[PublicationStorePort], durable_uow_base),
        transactions,
        run_uow=cast(
            ApplicationUnitOfWorkFactory[LogicalRunStorePort],
            durable_uow_base,
        ),
    )
    effects = ReplaySafeEffectService(
        cast(ApplicationUnitOfWorkFactory[MessageEffectStorePort], durable_uow_base),
        cast(ApplicationUnitOfWorkFactory[AuditEffectStorePort], durable_uow_base),
        cast(ApplicationUnitOfWorkFactory[ActionPrepareStorePort], durable_uow_base),
        side_effect_policy_service,
        transactions,
    )
    prepared_action_validation = PreparedActionValidationService(
        cast(
            ApplicationUnitOfWorkFactory[PreparedActionValidationStorePort],
            durable_uow_base,
        ),
        transactions,
    )
    content_sources = ContentSourceAuthorityService(source_uow, transactions)
    context_assembler = ContextAssemblerV1(
        ContextBudgetConfigV1(
            total_token_budget=settings.memory_context_total_token_budget,
            system_token_budget=settings.memory_context_system_token_budget,
            summary_token_budget=settings.memory_context_summary_token_budget,
            working_token_budget=settings.memory_context_working_token_budget,
            recent_token_budget=settings.memory_context_recent_token_budget,
            current_token_budget=settings.memory_context_current_token_budget,
        )
    )
    effective_summary_content_budget = min(
        settings.memory_summary_token_budget,
        context_assembler.summary_content_token_budget,
    )
    if effective_summary_content_budget <= 0:
        raise RuntimeError("summary partition leaves no room for summary content")
    recent_summary = RecentMessagesRollingSummaryService(
        memory_uow=memory_uow,
        transactions=transactions,
        summary_generator=DeterministicSummaryGeneratorV1(content_token_budget=effective_summary_content_budget),
        data_protection=build_rolling_summary_protection(),
        content_sources=content_sources,
        config=MemoryBudgetConfigV1(
            recent_message_limit=settings.memory_recent_message_limit,
            recent_token_budget=settings.memory_recent_token_budget,
            summary_trigger_message_count=settings.memory_summary_trigger_message_count,
            summary_trigger_token_budget=settings.memory_summary_trigger_token_budget,
            summary_token_budget=effective_summary_content_budget,
        ),
    )
    runtime_contexts = RuntimeContextProvider()
    single_flight = PerThreadSingleFlight()
    observability_runtime = _build_local_observability_runtime()
    resume_capabilities = CustomerResumeCapabilityProvider()
    admin_resume_capabilities = AdminDecisionResumeCapabilityProvider()
    business_execution = DurableBusinessExecutionApplication(
        business_execution_uow,
        transactions,
        build_business_execution_audit_protection(),
    )
    agent = AgentService(
        unit_of_works=AgentApplicationUnitOfWorks.from_transaction_store_factory(workflow_uow),
        runtime_context_provider=runtime_contexts,
        durable_authority=durable_authority,
        replay_safe_effects=effects,
        working_memory=WorkingMemoryApplicationService(memory_uow),
        recent_summary=recent_summary,
        context_assembler=context_assembler,
        current_question_guard=CurrentQuestionRequestGuard(context_assembler),
        single_flight=single_flight,
        lease_ttl_seconds=settings.checkpoint_lease_ttl_seconds,
        lease_renew_interval_seconds=(settings.checkpoint_lease_renew_interval_seconds),
        durable_customer_interrupt_enabled=(settings.durable_customer_interrupt_enabled),
        customer_resume_capabilities=resume_capabilities,
        prepared_action_validation=prepared_action_validation,
        admin_resume_capabilities=admin_resume_capabilities,
        business_execution=business_execution,
        observability_runtime=observability_runtime,
    )
    managed_saver = checkpoint_runtime.checkpointer()
    checkpoint_reader: CustomerConfirmationCheckpointReaderPort | None = None
    if managed_saver is None:
        if settings.durable_customer_interrupt_enabled:
            raise RuntimeError("durable customer confirmation requires the checkpoint saver")
        if settings.checkpoint_required:
            raise RuntimeError("required checkpoint saver is unavailable")
        graph = build_customer_service_graph(
            agent.workflow_nodes(),
            checkpointer=None,
        )
    else:
        metadata_catalog = build_tool_metadata_policy_catalog()
        bridge = ProtectedCheckpointBridge(
            saver=cast(AsyncCheckpointSaverPort, managed_saver),
            runtime_contexts=runtime_contexts,
            content_sources=content_sources,
            durable_authority=durable_authority,
            state_projector=AgentCheckpointStateProjector(metadata_catalog=metadata_catalog),
            pending_projector=AgentPendingWriteProjector(metadata_catalog=metadata_catalog),
        )
        graph = build_customer_service_graph(
            agent.workflow_nodes(),
            checkpointer=cast(Any, bridge),
        )
        checkpoint_reader = cast(CustomerConfirmationCheckpointReaderPort, bridge)
        checkpoint_runtime.mark_graph_checkpointer_wired()
    agent.bind_compiled_graph(graph)
    customer_confirmation = CustomerConfirmationApplicationService(
        unit_of_work=cast(
            ApplicationUnitOfWorkFactory[ConfirmationStorePort],
            workflow_uow,
        ),
        durable_authority=durable_authority,
        runtime_contexts=runtime_contexts,
        graph=cast(CustomerConfirmationGraphPort, graph),
        checkpoint_reader=checkpoint_reader,
        single_flight=single_flight,
        resume_capabilities=resume_capabilities,
        side_effect_policy=side_effect_policy_service,
        prepared_action_validation=prepared_action_validation,
        lease_ttl_seconds=settings.checkpoint_lease_ttl_seconds,
        lease_renew_interval_seconds=(settings.checkpoint_lease_renew_interval_seconds),
        enabled=settings.durable_customer_interrupt_enabled,
        observability_runtime=observability_runtime,
    )
    agent.bind_customer_confirmation_application(customer_confirmation)
    admin_decision = AdminDecisionApplicationService(
        unit_of_work=admin_decision_uow,
        durable_authority=durable_authority,
        runtime_contexts=runtime_contexts,
        graph=cast(AdminDecisionGraphPort, graph),
        checkpoint_reader=checkpoint_reader,
        single_flight=single_flight,
        resume_capabilities=admin_resume_capabilities,
        security_audit=AdminDecisionSecurityAuditApplication(
            unit_of_work=admin_decision_audit_uow,
            data_protection=build_admin_decision_audit_protection(),
        ),
        business_execution=business_execution,
        lease_ttl_seconds=settings.checkpoint_lease_ttl_seconds,
        lease_renew_interval_seconds=(
            settings.checkpoint_lease_renew_interval_seconds
        ),
        enabled=settings.durable_customer_interrupt_enabled,
        observability_runtime=observability_runtime,
    )
    agent.bind_admin_decision_application(
        admin_decision,
        AdminDecisionReconciler(admin_decision),
    )
    return agent


__all__ = ["build_managed_agent_service"]
