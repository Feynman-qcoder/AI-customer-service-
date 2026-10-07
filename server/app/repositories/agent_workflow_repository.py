from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Protocol

from sqlalchemy import literal_column, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy.sql.elements import ColumnElement

from app.agent.tools.registry import TOOL_REGISTRY
from app.core.config import settings
from app.db.models import (
    AfterSaleRule,
    AfterSaleRuleCondition,
    AgentActionRequest,
    AgentRetrievalTrace,
    AgentRun,
    AgentRunAttempt,
    AgentStep,
    AgentThreadExecution,
    AgentToolCall,
    ChatConversation,
    ChatMessage,
    CustomerOrder,
    KbChunk,
    KbDocument,
    ModelRuntimeConfig,
    ProductCatalog,
)
from app.rag.retrieval_config import extract_retrieval_keywords as _extract_keywords
from app.runtime.data_protection import DataProtectionPort
from app.runtime.durable import EffectWriteScope
from app.runtime.model_config import EffectiveModelRuntimeConfig
from app.runtime.tool_retrieval_telemetry import (
    OptionalTelemetryContractError,
    RetrievalDiagnosticStatusV1,
    RetrievalErrorTypeV1,
    RetrievalSourceTypeV1,
    RetrievalTraceEventV1,
    RetrievalTraceKindV1,
    ToolAuditEventV1,
    ToolNameV1,
    build_retrieval_trace_protection,
    build_tool_audit_protection,
    protect_retrieval_trace_event,
    protect_tool_audit_event,
)
from app.runtime.uow import TransactionBoundStoreGuard
from app.schemas.chat import SourceReference
from app.schemas.retrieval import (
    RetrievalCandidate,
    RetrievalChannelDiagnostic,
    RetrievalQueryContext,
)

_DB_NOW: ColumnElement[datetime] = literal_column("CURRENT_TIMESTAMP(6)")


@dataclass(frozen=True, slots=True)
class ProductSnapshot:
    id: int
    product_code: str
    product_name: str
    category: str
    sale_status: str
    price: Decimal
    stock_quantity: int
    dispatch_rule: str
    after_sale_rule: str


@dataclass(frozen=True, slots=True)
class OrderSnapshot:
    id: int
    order_no: str
    user_id: int
    product_id: int
    quantity: int
    amount: Decimal
    status: str
    paid_at: datetime | None
    expected_ship_at: datetime | None
    shipped_at: datetime | None
    signed_at: datetime | None
    created_at: datetime
    product: ProductSnapshot


@dataclass(frozen=True, slots=True)
class ToolCallRecord:
    run_id: str
    subject_user_id: int
    tool_name: str
    redacted_arguments: dict[str, object]
    result_summary: str
    success: bool
    retry_count: int
    duration_ms: int


@dataclass(frozen=True, slots=True)
class ConversationOwnershipSnapshot:
    conversation_id: int
    owner_user_id: int | None


@dataclass(frozen=True, slots=True)
class CustomerConfirmationRunSnapshot:
    conversation_id: int
    thread_id: str
    run_id: str
    subject_user_id: int
    status: str
    confirmation_prompt: str
    paused_attempt_id: str
    pending_action_id: int | None = None


class CustomerConfirmationStoreError(RuntimeError):
    """Sanitized persistence invariant failure for the resume boundary."""


class CustomerConfirmationOwnershipError(CustomerConfirmationStoreError):
    """The authenticated customer does not own the requested conversation."""


class ChatConversationPreparationStorePort(Protocol):
    async def create_chat_conversation(
        self,
        *,
        owner_user_id: int,
        conversation_no: str,
        title: str,
        created_at: datetime,
    ) -> int: ...

    async def conversation_ownership(
        self,
        conversation_id: int,
    ) -> ConversationOwnershipSnapshot | None: ...


class ConversationContextStorePort(Protocol):
    async def assert_conversation_owned(
        self,
        *,
        conversation_id: int,
        subject_user_id: int,
    ) -> None: ...

    async def recent_assistant_messages(self, conversation_id: int, limit: int) -> tuple[str, ...]: ...


class BlockedRunStorePort(Protocol):
    async def runtime_config(self) -> EffectiveModelRuntimeConfig: ...

    async def persist_blocked_run(
        self,
        *,
        run_id: str,
        thread_id: str,
        conversation_id: int,
        user_id: int,
        started_at: datetime,
        completed_at: datetime,
        question: str,
        final_answer: str,
        model_name: str,
        config_version: str,
    ) -> None: ...


class PlannerPersistenceStorePort(Protocol):
    async def assert_conversation_owned(
        self,
        *,
        conversation_id: int,
        subject_user_id: int,
    ) -> None: ...

    async def runtime_config(self) -> EffectiveModelRuntimeConfig: ...

    async def persist_planned_run(
        self,
        *,
        run_id: str,
        thread_id: str,
        conversation_id: int,
        user_id: int,
        started_at: datetime,
        question: str,
        intent: str,
        risk_level: str,
        model_name: str,
        config_version: str,
    ) -> None: ...


class WorkflowAuditStorePort(Protocol):
    async def assert_conversation_owned(
        self,
        *,
        conversation_id: int,
        subject_user_id: int,
    ) -> None: ...

    async def record_step(
        self,
        run_id: str,
        node_name: str,
        input_summary: str | None,
        output_summary: str | None,
        status: str,
    ) -> None: ...

    async def finalize_run(
        self,
        *,
        run_id: str,
        conversation_id: int,
        answer: str,
        sources_json: str,
        retrieval_score: float,
        confidence_level: str,
        need_human: bool,
        completed_at: datetime,
    ) -> None: ...


class ConfirmationStorePort(Protocol):
    async def latest_assistant_message(self, conversation_id: int) -> str | None: ...

    async def has_active_customer_confirmation(
        self,
        *,
        conversation_id: int,
        subject_user_id: int,
    ) -> bool: ...

    async def customer_confirmation_run(
        self,
        *,
        conversation_id: int,
        subject_user_id: int,
    ) -> CustomerConfirmationRunSnapshot: ...

    async def mark_customer_confirmation_waiting(
        self,
        *,
        scope: EffectWriteScope,
        conversation_id: int,
        subject_user_id: int,
        run_id: str,
        confirmation_prompt: str,
    ) -> None: ...

    async def mark_customer_confirmation_resumed(
        self,
        *,
        scope: EffectWriteScope,
        conversation_id: int,
        subject_user_id: int,
        run_id: str,
    ) -> None: ...

    async def mark_customer_confirmation_retryable(
        self,
        *,
        scope: EffectWriteScope,
        conversation_id: int,
        subject_user_id: int,
        run_id: str,
    ) -> None: ...

    async def mark_admin_approval_waiting(
        self,
        *,
        scope: EffectWriteScope,
        conversation_id: int,
        subject_user_id: int,
        run_id: str,
        pending_action_id: int,
    ) -> None: ...

    async def lock_customer_confirmation_order(
        self,
        *,
        user_id: int,
        order_no: str,
    ) -> OrderSnapshot | None: ...

    async def customer_confirmation_database_time(self) -> datetime: ...


class OrderReadStorePort(Protocol):
    async def list_orders(
        self,
        *,
        user_id: int,
        limit: int,
        status: str | None = None,
        product_keyword: str | None = None,
    ) -> tuple[OrderSnapshot, ...]: ...

    async def get_order(self, *, user_id: int, order_no: str) -> OrderSnapshot | None: ...


class ActionPrepareStorePort(Protocol):
    async def lock_order(self, *, user_id: int, order_no: str) -> OrderSnapshot | None: ...

    async def create_r2_action_request(
        self,
        *,
        run_id: str,
        action_type: str,
        target_order_id: int,
        target_order_no: str,
        subject_user_id: int,
        reason: str,
        idempotency_key: str,
        created_at: datetime,
    ) -> int: ...


class ProductReadStorePort(Protocol):
    async def get_product(self, keyword: str) -> ProductSnapshot | None: ...

    async def product_sources(self, product: ProductSnapshot) -> tuple[SourceReference, ...]: ...


class RuntimeConfigStorePort(Protocol):
    async def runtime_config(self) -> EffectiveModelRuntimeConfig: ...


class RetrievalStorePort(Protocol):
    async def assert_conversation_owned(
        self,
        *,
        conversation_id: int,
        subject_user_id: int,
    ) -> None: ...

    async def keyword_recall(self, query: str, limit: int) -> list[RetrievalCandidate]: ...

    async def structured_rule_recall(
        self,
        query: str,
        limit: int,
        context: RetrievalQueryContext,
    ) -> list[RetrievalCandidate]: ...

    async def record_retrieval_trace(
        self,
        run_id: str,
        candidates: list[RetrievalCandidate],
        diagnostics: list[RetrievalChannelDiagnostic],
    ) -> None: ...


class ToolAuditStorePort(Protocol):
    async def record_tool_call(self, record: ToolCallRecord) -> None: ...


class AgentWorkflowTransactionStore(
    ConversationContextStorePort,
    BlockedRunStorePort,
    PlannerPersistenceStorePort,
    WorkflowAuditStorePort,
    ConfirmationStorePort,
    OrderReadStorePort,
    ActionPrepareStorePort,
    ProductReadStorePort,
    RuntimeConfigStorePort,
    RetrievalStorePort,
    ToolAuditStorePort,
    Protocol,
):
    """Concrete-adapter wiring surface; application consumers receive only narrow views."""


class SqlAlchemyAgentWorkflowStore:
    """Transaction-bound persistence adapter for the customer chat application use case."""

    def __init__(
        self,
        session: AsyncSession,
        guard: TransactionBoundStoreGuard,
        *,
        tool_audit_protection: DataProtectionPort | None = None,
        retrieval_trace_protection: DataProtectionPort | None = None,
    ) -> None:
        self._session = session
        self._guard = guard
        self._tool_audit_protection = (
            tool_audit_protection or build_tool_audit_protection()
        )
        self._retrieval_trace_protection = (
            retrieval_trace_protection or build_retrieval_trace_protection()
        )

    async def create_chat_conversation(
        self,
        *,
        owner_user_id: int,
        conversation_no: str,
        title: str,
        created_at: datetime,
    ) -> int:
        self._guard.ensure_active()
        row = ChatConversation(
            user_id=owner_user_id,
            conversation_no=conversation_no,
            title=title,
            status="ACTIVE",
            created_at=created_at,
            updated_at=created_at,
        )
        self._session.add(row)
        await self._guard.flush(self._session)
        return row.id

    async def conversation_ownership(
        self,
        conversation_id: int,
    ) -> ConversationOwnershipSnapshot | None:
        self._guard.ensure_active()
        row = await self._session.get(ChatConversation, conversation_id)
        if row is None:
            return None
        return ConversationOwnershipSnapshot(
            conversation_id=row.id,
            owner_user_id=row.user_id,
        )

    async def assert_conversation_owned(
        self,
        *,
        conversation_id: int,
        subject_user_id: int,
    ) -> None:
        self._guard.ensure_active()
        owned_id = await self._session.scalar(
            select(ChatConversation.id).where(
                ChatConversation.id == conversation_id,
                ChatConversation.user_id == subject_user_id,
                ChatConversation.status == "ACTIVE",
            )
        )
        if owned_id is None:
            raise ValueError("conversation ownership changed before agent execution")

    async def runtime_config(self) -> EffectiveModelRuntimeConfig:
        self._guard.ensure_active()
        row = await self._session.get(ModelRuntimeConfig, 1)
        if row is None:
            row = ModelRuntimeConfig(
                id=1,
                temperature=Decimal(str(settings.llm_temperature)),
                top_k=settings.rag_top_k,
                min_retrieval_score=Decimal(str(settings.rag_min_retrieval_score)),
                mock_enabled=settings.llm_mock_enabled,
                updated_at=datetime.now(),
            )
            self._session.add(row)
            await self._guard.flush(self._session)
        return EffectiveModelRuntimeConfig(
            temperature=float(row.temperature),
            top_k=int(row.top_k),
            min_retrieval_score=float(row.min_retrieval_score),
            mock_enabled=bool(row.mock_enabled),
        )

    async def recent_assistant_messages(self, conversation_id: int, limit: int) -> tuple[str, ...]:
        self._guard.ensure_active()
        rows = (
            await self._session.scalars(
                select(ChatMessage.content)
                .where(ChatMessage.conversation_id == conversation_id, ChatMessage.role == "ASSISTANT")
                .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
                .limit(limit)
            )
        ).all()
        return tuple(str(value) for value in rows)

    async def persist_blocked_run(
        self,
        *,
        run_id: str,
        thread_id: str,
        conversation_id: int,
        user_id: int,
        started_at: datetime,
        completed_at: datetime,
        question: str,
        final_answer: str,
        model_name: str,
        config_version: str,
    ) -> None:
        self._guard.ensure_active()
        del started_at
        run = await self._require_existing_run(
            run_id=run_id,
            thread_id=thread_id,
            conversation_id=conversation_id,
            user_id=user_id,
        )
        run.status = "BLOCKED"
        run.intent = "CLARIFICATION"
        run.risk_level = "FORBIDDEN"
        run.completed_at = completed_at
        run.final_answer = final_answer
        run.model_name = model_name
        run.config_version = config_version
        run.prompt_version = "controlled-workflow-v1"
        await self.record_step(
            run_id,
            "input_guard",
            "CURRENT_TURN_CLASSIFIED",
            "INPUT_REJECTED",
            "BLOCKED",
        )

    async def persist_planned_run(
        self,
        *,
        run_id: str,
        thread_id: str,
        conversation_id: int,
        user_id: int,
        started_at: datetime,
        question: str,
        intent: str,
        risk_level: str,
        model_name: str,
        config_version: str,
    ) -> None:
        self._guard.ensure_active()
        del started_at
        run = await self._require_existing_run(
            run_id=run_id,
            thread_id=thread_id,
            conversation_id=conversation_id,
            user_id=user_id,
        )
        run.status = "RUNNING"
        run.intent = intent
        run.risk_level = risk_level
        run.model_name = model_name
        run.config_version = config_version
        run.prompt_version = "controlled-workflow-v1"
        await self.record_step(
            run_id,
            "planner",
            "CURRENT_TURN_CLASSIFIED",
            f"intent={intent}, risk={risk_level}",
            "COMPLETED",
        )

    async def record_step(
        self,
        run_id: str,
        node_name: str,
        input_summary: str | None,
        output_summary: str | None,
        status: str,
    ) -> None:
        self._guard.ensure_active()
        self._session.add(
            AgentStep(
                run_id=run_id,
                node_name=node_name,
                input_summary=input_summary,
                output_summary=output_summary,
                status=status,
                created_at=datetime.now(),
            )
        )

    async def finalize_run(
        self,
        *,
        run_id: str,
        conversation_id: int,
        answer: str,
        sources_json: str,
        retrieval_score: float,
        confidence_level: str,
        need_human: bool,
        completed_at: datetime,
    ) -> None:
        self._guard.ensure_active()
        del sources_json, retrieval_score, confidence_level, need_human
        run = await self._session.scalar(
            select(AgentRun)
            .where(
                AgentRun.run_id == run_id,
                AgentRun.conversation_id == conversation_id,
            )
            .with_for_update()
        )
        if run is None:
            raise ValueError("finalize requires an existing logical run")
        run.status = "COMPLETED"
        run.completed_at = completed_at
        run.final_answer = answer

    async def _require_existing_run(
        self,
        *,
        run_id: str,
        thread_id: str,
        conversation_id: int,
        user_id: int,
    ) -> AgentRun:
        run = await self._session.scalar(select(AgentRun).where(AgentRun.run_id == run_id).with_for_update())
        if run is None:
            raise ValueError("workflow persistence requires an existing logical run")
        if run.thread_id != thread_id or run.conversation_id != conversation_id or run.user_id != user_id:
            raise ValueError("workflow run identity changed before persistence")
        return run

    async def latest_assistant_message(self, conversation_id: int) -> str | None:
        messages = await self.recent_assistant_messages(conversation_id, 1)
        return messages[0] if messages else None

    async def has_active_customer_confirmation(
        self,
        *,
        conversation_id: int,
        subject_user_id: int,
    ) -> bool:
        await self._assert_customer_confirmation_owner(
            conversation_id=conversation_id,
            subject_user_id=subject_user_id,
        )
        run_id = await self._session.scalar(
            select(AgentRun.run_id)
            .where(
                AgentRun.conversation_id == conversation_id,
                AgentRun.user_id == subject_user_id,
                AgentRun.status.in_(
                    (
                        "WAITING_CUSTOMER_CONFIRMATION",
                        "RESUME_PENDING",
                        "WAITING_ADMIN_APPROVAL",
                    )
                ),
            )
            .limit(1)
        )
        return run_id is not None

    async def customer_confirmation_run(
        self,
        *,
        conversation_id: int,
        subject_user_id: int,
    ) -> CustomerConfirmationRunSnapshot:
        await self._assert_customer_confirmation_owner(
            conversation_id=conversation_id,
            subject_user_id=subject_user_id,
        )
        runs = list(
            (
                await self._session.scalars(
                    select(AgentRun)
                    .where(
                        AgentRun.conversation_id == conversation_id,
                        AgentRun.user_id == subject_user_id,
                        AgentRun.status.in_(
                            (
                                "WAITING_CUSTOMER_CONFIRMATION",
                                "RESUME_PENDING",
                                "WAITING_ADMIN_APPROVAL",
                            )
                        ),
                    )
                    .order_by(AgentRun.id.asc())
                    .limit(2)
                )
            ).all()
        )
        if len(runs) != 1:
            raise CustomerConfirmationStoreError("conversation does not have one durable customer confirmation run")
        run = runs[0]
        prompt = run.final_answer
        if not isinstance(prompt, str) or not prompt:
            raise CustomerConfirmationStoreError("durable customer confirmation metadata is incomplete")
        paused_attempt_id = await self._session.scalar(
            select(AgentRunAttempt.attempt_id)
            .where(AgentRunAttempt.run_id == run.run_id)
            .order_by(AgentRunAttempt.id.asc())
            .limit(1)
        )
        if not isinstance(paused_attempt_id, str) or not paused_attempt_id:
            raise CustomerConfirmationStoreError("durable customer confirmation attempt is missing")
        pending_action_id: int | None = None
        if run.status == "WAITING_ADMIN_APPROVAL":
            action_ids = list(
                (
                    await self._session.scalars(
                        select(AgentActionRequest.id).where(
                            AgentActionRequest.run_id == run.run_id,
                            AgentActionRequest.created_by == subject_user_id,
                            AgentActionRequest.confirmation_mode == "DURABLE_INTERRUPT",
                            AgentActionRequest.status == "PENDING",
                            AgentActionRequest.resume_status == "WAITING_ADMIN_DECISION",
                        )
                    )
                ).all()
            )
            if len(action_ids) != 1:
                raise CustomerConfirmationStoreError(
                    "admin approval wait does not have one durable pending action"
                )
            pending_action_id = int(action_ids[0])
        return CustomerConfirmationRunSnapshot(
            conversation_id=run.conversation_id,
            thread_id=run.thread_id,
            run_id=run.run_id,
            subject_user_id=run.user_id,
            status=run.status,
            confirmation_prompt=prompt,
            paused_attempt_id=paused_attempt_id,
            pending_action_id=pending_action_id,
        )

    async def mark_customer_confirmation_waiting(
        self,
        *,
        scope: EffectWriteScope,
        conversation_id: int,
        subject_user_id: int,
        run_id: str,
        confirmation_prompt: str,
    ) -> None:
        await self._assert_customer_confirmation_owner(
            conversation_id=conversation_id,
            subject_user_id=subject_user_id,
        )
        await self._require_live_confirmation_authority(
            scope=scope,
            conversation_id=conversation_id,
            subject_user_id=subject_user_id,
            run_id=run_id,
        )
        run = await self._session.scalar(
            select(AgentRun)
            .where(
                AgentRun.run_id == run_id,
                AgentRun.conversation_id == conversation_id,
                AgentRun.user_id == subject_user_id,
            )
            .with_for_update()
        )
        if run is None or run.status not in {
            "RUNNING",
            "WAITING_CUSTOMER_CONFIRMATION",
        }:
            raise CustomerConfirmationStoreError("logical run cannot enter customer confirmation wait")
        if run.status == "WAITING_CUSTOMER_CONFIRMATION" and (run.final_answer != confirmation_prompt):
            raise CustomerConfirmationStoreError("customer confirmation replay changed its frozen prompt")
        run.status = "WAITING_CUSTOMER_CONFIRMATION"
        run.final_answer = confirmation_prompt

    async def mark_customer_confirmation_resumed(
        self,
        *,
        scope: EffectWriteScope,
        conversation_id: int,
        subject_user_id: int,
        run_id: str,
    ) -> None:
        await self._assert_customer_confirmation_owner(
            conversation_id=conversation_id,
            subject_user_id=subject_user_id,
        )
        await self._require_live_confirmation_authority(
            scope=scope,
            conversation_id=conversation_id,
            subject_user_id=subject_user_id,
            run_id=run_id,
        )
        run = await self._session.scalar(
            select(AgentRun)
            .where(
                AgentRun.run_id == run_id,
                AgentRun.conversation_id == conversation_id,
                AgentRun.user_id == subject_user_id,
            )
            .with_for_update()
        )
        if run is None or run.status not in {
            "WAITING_CUSTOMER_CONFIRMATION",
            "RESUME_PENDING",
        }:
            raise CustomerConfirmationStoreError("logical run is not waiting for customer confirmation")
        run.status = "RESUME_PENDING"

    async def mark_customer_confirmation_retryable(
        self,
        *,
        scope: EffectWriteScope,
        conversation_id: int,
        subject_user_id: int,
        run_id: str,
    ) -> None:
        await self._assert_customer_confirmation_owner(
            conversation_id=conversation_id,
            subject_user_id=subject_user_id,
        )
        await self._require_live_confirmation_authority(
            scope=scope,
            conversation_id=conversation_id,
            subject_user_id=subject_user_id,
            run_id=run_id,
        )
        run = await self._session.scalar(
            select(AgentRun)
            .where(
                AgentRun.run_id == run_id,
                AgentRun.conversation_id == conversation_id,
                AgentRun.user_id == subject_user_id,
            )
            .with_for_update()
        )
        if run is None or run.status not in {
            "WAITING_CUSTOMER_CONFIRMATION",
            "RESUME_PENDING",
        }:
            raise CustomerConfirmationStoreError(
                "logical run cannot be marked retryable from its current state"
            )
        run.status = "FAILED"
        run.completed_at = await self._session.scalar(select(_DB_NOW))

    async def mark_admin_approval_waiting(
        self,
        *,
        scope: EffectWriteScope,
        conversation_id: int,
        subject_user_id: int,
        run_id: str,
        pending_action_id: int,
    ) -> None:
        await self._assert_customer_confirmation_owner(
            conversation_id=conversation_id,
            subject_user_id=subject_user_id,
        )
        await self._require_live_confirmation_authority(
            scope=scope,
            conversation_id=conversation_id,
            subject_user_id=subject_user_id,
            run_id=run_id,
        )
        action = await self._session.scalar(
            select(AgentActionRequest)
            .where(
                AgentActionRequest.id == pending_action_id,
                AgentActionRequest.run_id == run_id,
                AgentActionRequest.created_by == subject_user_id,
                AgentActionRequest.confirmation_mode == "DURABLE_INTERRUPT",
                AgentActionRequest.status == "PENDING",
                AgentActionRequest.resume_status == "WAITING_ADMIN_DECISION",
            )
            .with_for_update()
        )
        if action is None:
            raise CustomerConfirmationStoreError(
                "admin approval marker does not match a durable pending action"
            )
        run = await self._session.scalar(
            select(AgentRun)
            .where(
                AgentRun.run_id == run_id,
                AgentRun.conversation_id == conversation_id,
                AgentRun.user_id == subject_user_id,
            )
            .with_for_update()
        )
        if run is None or run.status not in {
            "RESUME_PENDING",
            "WAITING_ADMIN_APPROVAL",
        }:
            raise CustomerConfirmationStoreError(
                "logical run cannot enter admin approval wait"
            )
        run.status = "WAITING_ADMIN_APPROVAL"

    async def lock_customer_confirmation_order(
        self,
        *,
        user_id: int,
        order_no: str,
    ) -> OrderSnapshot | None:
        return await self.lock_order(user_id=user_id, order_no=order_no)

    async def customer_confirmation_database_time(self) -> datetime:
        self._guard.ensure_active()
        value = (
            await self._session.execute(text("SELECT CURRENT_TIMESTAMP(6)"))
        ).scalar_one()
        if not isinstance(value, datetime):
            raise CustomerConfirmationStoreError("customer confirmation database time is unavailable")
        return value

    async def _assert_customer_confirmation_owner(
        self,
        *,
        conversation_id: int,
        subject_user_id: int,
    ) -> None:
        self._guard.ensure_active()
        owner = await self._session.scalar(
            select(ChatConversation.user_id).where(
                ChatConversation.id == conversation_id,
                ChatConversation.status == "ACTIVE",
            )
        )
        if owner is None or owner != subject_user_id:
            raise CustomerConfirmationOwnershipError("customer confirmation conversation access is denied")

    async def _require_live_confirmation_authority(
        self,
        *,
        scope: EffectWriteScope,
        conversation_id: int,
        subject_user_id: int,
        run_id: str,
    ) -> AgentRunAttempt:
        self._guard.ensure_active()
        attempt = await self._session.scalar(
            select(AgentRunAttempt)
            .where(
                AgentRunAttempt.attempt_id == scope.attempt_id,
                AgentRunAttempt.run_id == run_id,
                AgentRunAttempt.thread_id == scope.thread_id,
                AgentRunAttempt.conversation_id == conversation_id,
                AgentRunAttempt.fence_version == scope.fence_version,
            )
        )
        if (
            attempt is None
            or attempt.actor_user_id != subject_user_id
            or attempt.subject_user_id != subject_user_id
            or attempt.actor_role != "CUSTOMER"
            or attempt.service_principal is not None
        ):
            raise CustomerConfirmationStoreError(
                "customer confirmation marker authority is invalid"
            )
        execution = await self._session.scalar(
            select(AgentThreadExecution)
            .where(
                AgentThreadExecution.thread_id == scope.thread_id,
                AgentThreadExecution.conversation_id == conversation_id,
                AgentThreadExecution.owner_attempt_id == scope.attempt_id,
                AgentThreadExecution.fence_version == scope.fence_version,
                AgentThreadExecution.lease_expires_at.is_not(None),
                AgentThreadExecution.lease_expires_at > _DB_NOW,
            )
            .with_for_update()
        )
        if execution is None:
            raise CustomerConfirmationStoreError(
                "customer confirmation marker lost live lease authority"
            )
        return attempt

    async def list_orders(
        self,
        *,
        user_id: int,
        limit: int,
        status: str | None = None,
        product_keyword: str | None = None,
    ) -> tuple[OrderSnapshot, ...]:
        self._guard.ensure_active()
        query = (
            select(CustomerOrder).options(selectinload(CustomerOrder.product)).where(CustomerOrder.user_id == user_id)
        )
        if status:
            query = query.where(CustomerOrder.status == status)
        if product_keyword:
            query = query.join(CustomerOrder.product).where(
                or_(
                    ProductCatalog.product_code == product_keyword,
                    ProductCatalog.product_name.contains(
                        product_keyword,
                        autoescape=True,
                    ),
                )
            )
        rows = list((await self._session.scalars(query.order_by(CustomerOrder.created_at.desc()).limit(limit))).all())
        return tuple(_order_snapshot(row) for row in rows)

    async def get_order(self, *, user_id: int, order_no: str) -> OrderSnapshot | None:
        self._guard.ensure_active()
        row = await self._session.scalar(
            select(CustomerOrder)
            .options(selectinload(CustomerOrder.product))
            .where(CustomerOrder.user_id == user_id, CustomerOrder.order_no == order_no)
        )
        return _order_snapshot(row) if row is not None else None

    async def lock_order(self, *, user_id: int, order_no: str) -> OrderSnapshot | None:
        self._guard.ensure_active()
        row = await self._session.scalar(
            select(CustomerOrder)
            .options(selectinload(CustomerOrder.product))
            .where(CustomerOrder.user_id == user_id, CustomerOrder.order_no == order_no)
            .with_for_update()
        )
        return _order_snapshot(row) if row is not None else None

    async def get_product(self, keyword: str) -> ProductSnapshot | None:
        self._guard.ensure_active()
        row = await self._session.scalar(
            select(ProductCatalog)
            .where(ProductCatalog.product_name.like(f"%{keyword}%") | ProductCatalog.product_code.like(f"%{keyword}%"))
            .order_by(ProductCatalog.id.asc())
            .limit(1)
        )
        return _product_snapshot(row) if row is not None else None

    async def product_sources(self, product: ProductSnapshot) -> tuple[SourceReference, ...]:
        self._guard.ensure_active()
        candidates = [product.product_code, product.product_name.replace(" ", "")]
        conditions = [KbDocument.original_name.like(f"%{value}%") for value in candidates if value]
        if not conditions:
            return ()
        row = (
            await self._session.execute(
                select(KbDocument, KbChunk)
                .join(KbChunk, KbChunk.document_id == KbDocument.id)
                .where(KbDocument.status.in_(["READY", "COMPLETED"]), or_(*conditions))
                .order_by(KbChunk.chunk_index.asc())
                .limit(1)
            )
        ).first()
        if row is None:
            return ()
        document, chunk = row
        return (
            SourceReference(
                documentId=int(document.id),
                fileName=str(document.original_name),
                snippet=str(chunk.content)[:260],
                score=1.0,
            ),
        )

    async def create_r2_action_request(
        self,
        *,
        run_id: str,
        action_type: str,
        target_order_id: int,
        target_order_no: str,
        subject_user_id: int,
        reason: str,
        idempotency_key: str,
        created_at: datetime,
    ) -> int:
        self._guard.ensure_active()
        order = await self._session.scalar(
            select(CustomerOrder)
            .where(
                CustomerOrder.id == target_order_id,
                CustomerOrder.order_no == target_order_no,
                CustomerOrder.user_id == subject_user_id,
            )
            .with_for_update()
        )
        if order is None:
            raise ValueError("action target is no longer owned by the authenticated subject")
        existing = await self._session.scalar(
            select(AgentActionRequest).where(
                AgentActionRequest.action_type == action_type,
                AgentActionRequest.target_order_id == target_order_id,
                AgentActionRequest.created_by == subject_user_id,
                AgentActionRequest.idempotency_key == idempotency_key,
            )
        )
        if existing is not None:
            return int(existing.id)
        request = AgentActionRequest(
            run_id=run_id,
            action_type=action_type,
            target_order_id=target_order_id,
            action_payload_json=json.dumps({"reason": reason}, ensure_ascii=False),
            risk_level="HIGH",
            status="PENDING",
            logical_action_id=None,
            confirmation_mode="R2_STATELESS_COMPAT",
            resume_status="NOT_APPLICABLE",
            idempotency_key=idempotency_key,
            created_by=subject_user_id,
            created_at=created_at,
        )
        self._session.add(request)
        await self._guard.flush(self._session)
        return int(request.id)

    async def keyword_recall(self, query: str, limit: int) -> list[RetrievalCandidate]:
        self._guard.ensure_active()
        keywords = _extract_keywords(query)
        if not keywords:
            return []
        conditions = [KbChunk.content.like(f"%{keyword}%") for keyword in keywords]
        rows = (
            await self._session.execute(
                select(KbChunk, KbDocument)
                .join(KbDocument, KbDocument.id == KbChunk.document_id)
                .where(KbDocument.status == "READY", or_(*conditions))
                .order_by(KbDocument.updated_at.desc(), KbChunk.chunk_index.asc())
                .limit(limit * 2)
            )
        ).all()
        candidates: list[RetrievalCandidate] = []
        for chunk, document in rows:
            matched_terms = [keyword for keyword in keywords if keyword in chunk.content]
            if not matched_terms:
                continue
            score = min(1.0, 0.35 + 0.15 * len(matched_terms))
            candidates.append(
                RetrievalCandidate(
                    candidate_id=f"chunk:{chunk.id}",
                    source_type="keyword",
                    content=chunk.content,
                    document_id=str(document.id),
                    chunk_id=str(chunk.id),
                    rule_id=None,
                    metadata={
                        "file_name": document.original_name,
                        "matched_terms": matched_terms,
                        "keyword_score": score,
                    },
                    original_score=score,
                )
            )
        return sorted(candidates, key=lambda item: item.original_score, reverse=True)[:limit]

    async def structured_rule_recall(
        self,
        query: str,
        limit: int,
        context: RetrievalQueryContext,
    ) -> list[RetrievalCandidate]:
        self._guard.ensure_active()
        now = datetime.now()
        after_sale_type = context.after_sale_type or _extract_after_sale_type(query)
        query_stmt = (
            select(AfterSaleRule, AfterSaleRuleCondition)
            .join(AfterSaleRuleCondition, AfterSaleRuleCondition.rule_id == AfterSaleRule.id)
            .where(
                AfterSaleRule.status == "ACTIVE",
                AfterSaleRule.effective_from <= now,
                or_(AfterSaleRule.effective_to.is_(None), AfterSaleRule.effective_to > now),
            )
            .order_by(AfterSaleRule.priority.desc(), AfterSaleRule.effective_from.desc())
            .limit(limit * 3)
        )
        if after_sale_type:
            query_stmt = query_stmt.where(AfterSaleRuleCondition.after_sale_type.in_([after_sale_type, "ANY"]))
        rows = (await self._session.execute(query_stmt)).all()
        candidates: list[RetrievalCandidate] = []
        for rule, condition in rows:
            matched, matched_conditions = _condition_matches(condition, context)
            if not matched:
                continue
            score = min(1.0, 0.55 + 0.08 * len(matched_conditions) + rule.priority / 100)
            candidates.append(
                RetrievalCandidate(
                    candidate_id=f"rule:{rule.id}",
                    source_type="structured_rule",
                    content=rule.content,
                    document_id=None,
                    chunk_id=None,
                    rule_id=str(rule.id),
                    metadata={
                        "rule_code": rule.rule_code,
                        "rule_title": rule.title,
                        "rule_version": rule.version.version_code,
                        "effective_from": rule.effective_from.isoformat(),
                        "effective_to": rule.effective_to.isoformat() if rule.effective_to else None,
                        "matched_conditions": matched_conditions,
                        "priority": rule.priority,
                    },
                    original_score=score,
                )
            )
        return candidates[:limit]

    async def record_retrieval_trace(
        self,
        run_id: str,
        candidates: list[RetrievalCandidate],
        diagnostics: list[RetrievalChannelDiagnostic],
    ) -> None:
        self._guard.ensure_active()
        try:
            events = _retrieval_trace_events(
                run_id=run_id,
                candidates=candidates,
                diagnostics=diagnostics,
            )
            writes = tuple(
                protect_retrieval_trace_event(
                    event,
                    self._retrieval_trace_protection,
                )
                for event in events
            )
        except Exception:
            return
        for write in writes:
            self._session.add(
                AgentRetrievalTrace(
                    run_id=write.run_id,
                    candidate_id=write.candidate_id,
                    source_type=write.source_type,
                    document_id=write.document_id,
                    chunk_id=write.chunk_id,
                    rule_id=write.rule_id,
                    original_score=Decimal(str(write.original_score)),
                    fused_score=(
                        Decimal(str(write.fused_score))
                        if write.fused_score is not None
                        else None
                    ),
                    rerank_score=(
                        Decimal(str(write.rerank_score))
                        if write.rerank_score is not None
                        else None
                    ),
                    selected=write.selected,
                    decision_reason=write.decision_reason,
                    metadata_json=write.metadata_json,
                    created_at=datetime.now(),
                )
            )

    async def record_tool_call(self, record: ToolCallRecord) -> None:
        self._guard.ensure_active()
        owned_run_id = await self._session.scalar(
            select(AgentRun.run_id)
            .join(ChatConversation, ChatConversation.id == AgentRun.conversation_id)
            .where(
                AgentRun.run_id == record.run_id,
                AgentRun.user_id == record.subject_user_id,
                ChatConversation.user_id == record.subject_user_id,
                ChatConversation.status == "ACTIVE",
            )
        )
        if owned_run_id is None:
            raise ValueError("conversation ownership changed before tool audit persistence")
        try:
            event = ToolAuditEventV1(
                run_id=record.run_id,
                subject_user_id=record.subject_user_id,
                tool_name=ToolNameV1(record.tool_name),
                arguments_json=_closed_tool_arguments_json(record),
                result_summary=record.result_summary,
                success=record.success,
                retry_count=record.retry_count,
                duration_ms=record.duration_ms,
            )
            write = protect_tool_audit_event(
                event,
                self._tool_audit_protection,
            )
        except Exception:
            return
        self._session.add(
            AgentToolCall(
                run_id=write.run_id,
                tool_name=write.tool_name,
                redacted_arguments_json=write.redacted_arguments_json,
                result_summary=write.result_summary,
                success=write.success,
                retry_count=write.retry_count,
                duration_ms=write.duration_ms,
            )
        )


_RETRIEVAL_METADATA_KEYS: dict[str, frozenset[str]] = {
    "keyword": frozenset({"file_name", "matched_terms", "keyword_score"}),
    "dense_vector": frozenset({"file_name", "payload"}),
    "structured_rule": frozenset(
        {
            "rule_code",
            "rule_title",
            "rule_version",
            "effective_from",
            "effective_to",
            "matched_conditions",
            "priority",
        }
    ),
}


def _closed_tool_arguments_json(record: ToolCallRecord) -> str:
    definition = TOOL_REGISTRY.get(record.tool_name)
    if definition is None or type(record.redacted_arguments) is not dict:
        raise OptionalTelemetryContractError("tool audit arguments are invalid")
    if any(type(key) is not str for key in record.redacted_arguments):
        raise OptionalTelemetryContractError("tool audit arguments are invalid")
    allowed = frozenset(definition.argument_model.model_fields)
    if not set(record.redacted_arguments).issubset(allowed):
        raise OptionalTelemetryContractError("tool audit arguments are invalid")
    try:
        normalized = definition.argument_model.model_validate(
            record.redacted_arguments
        ).model_dump(mode="json")
        return json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError):
        raise OptionalTelemetryContractError(
            "tool audit arguments are invalid"
        ) from None


def _retrieval_trace_events(
    *,
    run_id: str,
    candidates: list[RetrievalCandidate],
    diagnostics: list[RetrievalChannelDiagnostic],
) -> tuple[RetrievalTraceEventV1, ...]:
    events: list[RetrievalTraceEventV1] = []
    for diagnostic in diagnostics:
        if type(diagnostic) is not RetrievalChannelDiagnostic:
            raise OptionalTelemetryContractError(
                "retrieval diagnostic is invalid"
            )
        status = RetrievalDiagnosticStatusV1(diagnostic.status)
        if status is RetrievalDiagnosticStatusV1.OK:
            continue
        source = RetrievalSourceTypeV1(diagnostic.channel)
        error_type = RetrievalErrorTypeV1(
            diagnostic.error_type or RetrievalErrorTypeV1.UNKNOWN.value
        )
        events.append(
            RetrievalTraceEventV1(
                run_id=run_id,
                kind=RetrievalTraceKindV1.DIAGNOSTIC,
                candidate_id=f"diagnostic:{source.value}",
                source_type=source,
                document_id=None,
                chunk_id=None,
                rule_id=None,
                original_score=0.0,
                fused_score=None,
                rerank_score=None,
                selected=False,
                decision_reason=f"{status.value}:{error_type.value}",
                metadata_json="{}",
                diagnostic_status=status,
                error_type=error_type,
            )
        )
    selected_ids = {candidate.candidate_id for candidate in candidates[:3]}
    for candidate in candidates:
        if type(candidate) is not RetrievalCandidate:
            raise OptionalTelemetryContractError("retrieval candidate is invalid")
        events.append(
            RetrievalTraceEventV1(
                run_id=run_id,
                kind=RetrievalTraceKindV1.CANDIDATE,
                candidate_id=candidate.candidate_id,
                source_type=RetrievalSourceTypeV1(candidate.source_type),
                document_id=candidate.document_id,
                chunk_id=candidate.chunk_id,
                rule_id=candidate.rule_id,
                original_score=candidate.original_score,
                fused_score=candidate.fused_score,
                rerank_score=candidate.rerank_score,
                selected=candidate.candidate_id in selected_ids,
                decision_reason=candidate.decision_reason,
                metadata_json=_closed_retrieval_metadata_json(candidate),
            )
        )
    return tuple(events)


def _closed_retrieval_metadata_json(candidate: RetrievalCandidate) -> str:
    metadata = candidate.metadata
    if type(metadata) is not dict or any(type(key) is not str for key in metadata):
        raise OptionalTelemetryContractError("retrieval metadata is invalid")
    allowed = _RETRIEVAL_METADATA_KEYS[candidate.source_type]
    if not set(metadata).issubset(allowed):
        raise OptionalTelemetryContractError("retrieval metadata is invalid")
    for key, value in metadata.items():
        if key in {
            "file_name",
            "rule_code",
            "rule_title",
            "rule_version",
            "effective_from",
        } and type(value) is not str:
            raise OptionalTelemetryContractError("retrieval metadata is invalid")
        if key == "effective_to" and value is not None and type(value) is not str:
            raise OptionalTelemetryContractError("retrieval metadata is invalid")
        if key in {"matched_terms", "matched_conditions"} and (
            type(value) is not list
            or any(type(item) is not str for item in value)
        ):
            raise OptionalTelemetryContractError("retrieval metadata is invalid")
        if key == "keyword_score" and type(value) not in {int, float}:
            raise OptionalTelemetryContractError("retrieval metadata is invalid")
        if key == "priority" and type(value) is not int:
            raise OptionalTelemetryContractError("retrieval metadata is invalid")
        if key == "payload":
            if (
                type(value) is not dict
                or set(value) != {"document_id", "chunk_id"}
                or any(type(item) not in {int, str} for item in value.values())
            ):
                raise OptionalTelemetryContractError("retrieval metadata is invalid")
    try:
        return json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError):
        raise OptionalTelemetryContractError("retrieval metadata is invalid") from None


def _product_snapshot(row: ProductCatalog) -> ProductSnapshot:
    return ProductSnapshot(
        id=int(row.id),
        product_code=row.product_code,
        product_name=row.product_name,
        category=row.category,
        sale_status=row.sale_status,
        price=row.price,
        stock_quantity=row.stock_quantity,
        dispatch_rule=row.dispatch_rule,
        after_sale_rule=row.after_sale_rule,
    )


def _order_snapshot(row: CustomerOrder) -> OrderSnapshot:
    return OrderSnapshot(
        id=int(row.id),
        order_no=row.order_no,
        user_id=int(row.user_id),
        product_id=int(row.product_id),
        quantity=row.quantity,
        amount=row.amount,
        status=row.status,
        paid_at=row.paid_at,
        expected_ship_at=row.expected_ship_at,
        shipped_at=row.shipped_at,
        signed_at=row.signed_at,
        created_at=row.created_at,
        product=_product_snapshot(row.product),
    )


def _extract_after_sale_type(query: str) -> str | None:
    if any(term in query for term in ["运费", "邮费", "包邮", "谁承担", "承担费用", "退货费"]):
        return "RETURN_FREIGHT"
    if any(term in query for term in ["破损", "损坏", "裂", "坏了", "质量"]):
        return "QUALITY_DAMAGE"
    if any(term in query for term in ["拆封", "退货", "退吗", "能退"]):
        return "RETURN"
    if any(term in query for term in ["退款", "退钱"]):
        return "REFUND"
    if "换货" in query:
        return "EXCHANGE"
    return None


def _condition_matches(
    condition: AfterSaleRuleCondition,
    context: RetrievalQueryContext,
) -> tuple[bool, list[str]]:
    matched: list[str] = []
    for field in ["product_category", "order_status", "payment_status", "shipment_status", "after_sale_type"]:
        expected = getattr(condition, field)
        actual = getattr(context, field)
        if expected is None or expected == "ANY":
            continue
        if actual is None:
            if context.has_specific_order:
                return False, matched
            matched.append(f"{field}={expected}")
            continue
        if expected != actual:
            return False, matched
        matched.append(f"{field}={expected}")
    if condition.signed_within_days is not None:
        if context.signed_days is None:
            if context.has_specific_order:
                return False, matched
            matched.append(f"signed_within_days<={condition.signed_within_days}")
            return True, matched
        if context.signed_days > condition.signed_within_days:
            return False, matched
        matched.append(f"signed_within_days<={condition.signed_within_days}")
    return True, matched
