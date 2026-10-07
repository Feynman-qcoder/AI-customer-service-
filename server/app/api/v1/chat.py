import logging

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.state import ActiveRunConflictError
from app.core.config import settings
from app.core.exceptions import AppError
from app.core.security import AuthenticatedUser, current_user
from app.db.session import get_session
from app.memory import (
    ContextAssemblerError,
    ContextAssemblerV1,
    ContextBudgetConfigV1,
    CurrentQuestionRequestGuard,
)
from app.runtime.external_ports import ChatRateLimitPort
from app.runtime.single_flight import ThreadSingleFlightConflict
from app.runtime.uow import require_no_active_transaction
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    CustomerConfirmationRequest,
    CustomerConfirmationResponse,
)
from app.schemas.common import ApiResponse
from app.services.agent_service import AgentService
from app.services.conversation_service import ChatConversationApplicationService
from app.services.customer_confirmation_application import (
    CustomerConfirmationApplicationService,
    CustomerConfirmationCommand,
    CustomerConfirmationError,
)
from app.services.redis_runtime_service import redis_runtime_service

router = APIRouter(tags=["chat"])
# Test-only override seam.  Production composition is process-lifetime state
# owned by FastAPI lifespan; no AgentService is constructed at module import.
agent_service: AgentService | None = None
customer_confirmation_application: CustomerConfirmationApplicationService | None = None
chat_conversation_application = ChatConversationApplicationService()
chat_rate_limiter: ChatRateLimitPort = redis_runtime_service
current_question_guard = CurrentQuestionRequestGuard(
    ContextAssemblerV1(
        ContextBudgetConfigV1(
            total_token_budget=settings.memory_context_total_token_budget,
            system_token_budget=settings.memory_context_system_token_budget,
            summary_token_budget=settings.memory_context_summary_token_budget,
            working_token_budget=settings.memory_context_working_token_budget,
            recent_token_budget=settings.memory_context_recent_token_budget,
            current_token_budget=settings.memory_context_current_token_budget,
        )
    )
)
logger = logging.getLogger(__name__)


@router.post("/chat")
async def chat(
    payload: ChatRequest,
    http_request: Request,
    user: AuthenticatedUser = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> ApiResponse[ChatResponse]:
    actor = AuthenticatedUser(
        user_id=user.user_id,
        username=user.username,
        name=user.name,
        role=user.role,
    )
    try:
        if session.in_transaction():
            await session.rollback()
    finally:
        await session.close()

    service = agent_service or getattr(http_request.app.state, "agent_service", None)
    if service is None or not callable(getattr(service, "chat", None)):
        raise AppError("Agent 服务尚未就绪", 503)
    guard = getattr(service, "current_question_guard", current_question_guard)
    try:
        guard.validate(payload.question)
    except ContextAssemblerError as exc:
        raise AppError("问题超过当前上下文预算，请缩短后重试", 422) from exc

    require_no_active_transaction("chat Redis rate limit")
    try:
        if not await chat_rate_limiter.allow_chat_request(actor.user_id):
            raise AppError("请求过于频繁，请稍后再试", 429)
    except AppError:
        raise
    except Exception:
        logger.exception("chat rate limiter degraded because Redis is unavailable")
    conversation_id = await chat_conversation_application.prepare(
        actor=actor,
        conversation_id=payload.conversationId,
        title="用户客服会话",
    )
    try:
        response = await service.chat(actor, conversation_id, payload.question)
    except (ActiveRunConflictError, ThreadSingleFlightConflict) as exc:
        raise AppError("当前会话正在处理中，请稍后重试", 409) from exc
    return ApiResponse.ok(response)


@router.post("/chat/customer-confirmation")
async def customer_confirmation(
    payload: CustomerConfirmationRequest,
    http_request: Request,
    user: AuthenticatedUser = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> ApiResponse[CustomerConfirmationResponse]:
    actor = AuthenticatedUser(
        user_id=user.user_id,
        username=user.username,
        name=user.name,
        role=user.role,
    )
    try:
        if session.in_transaction():
            await session.rollback()
    finally:
        await session.close()
    application = customer_confirmation_application or getattr(
        http_request.app.state,
        "customer_confirmation_application",
        None,
    )
    if application is None or not application.available:
        raise AppError("客户确认恢复服务尚未就绪", 503)
    try:
        result = await application.resume(
            actor,
            CustomerConfirmationCommand(
                conversation_id=payload.conversationId,
                confirmation_text=payload.confirmationText,
                confirmation_challenge_digest=(payload.confirmationChallengeDigest),
            ),
        )
    except ThreadSingleFlightConflict as exc:
        raise AppError("当前会话正在处理中，请稍后重试", 409) from exc
    except CustomerConfirmationError as exc:
        raise AppError("客户确认未被接受", exc.status_code) from exc
    return ApiResponse.ok(result)


@router.post("/agent/chat")
async def agent_chat(
    payload: ChatRequest,
    http_request: Request,
    user: AuthenticatedUser = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> ApiResponse[ChatResponse]:
    return await chat(payload, http_request, user, session)
