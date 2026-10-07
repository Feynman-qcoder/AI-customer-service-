from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import AuthenticatedUser, current_user
from app.db.session import get_session
from app.schemas.common import ApiResponse, PageResult
from app.schemas.conversation import (
    ConversationResponse,
    ConversationSummaryResponse,
    CreateConversationRequest,
    MessageResponse,
)
from app.services.conversation_service import ConversationService

router = APIRouter(prefix="/conversations", tags=["conversations"])
service = ConversationService()


@router.get("")
async def list_conversations(
    page: Annotated[int, Query(ge=1)] = 1,
    size: Annotated[int, Query(ge=1, le=100)] = 50,
    user: AuthenticatedUser = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> ApiResponse[PageResult[ConversationSummaryResponse]]:
    return ApiResponse.ok(await service.list_owned(session, user, page=page, size=size))


@router.post("")
async def create_conversation(
    request: CreateConversationRequest | None = None,
    user: AuthenticatedUser = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> ApiResponse[ConversationResponse]:
    return ApiResponse.ok(await service.create(session, user, request.title if request else None))


@router.get("/{conversation_id}/messages")
async def messages(
    conversation_id: int,
    user: AuthenticatedUser = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> ApiResponse[list[MessageResponse]]:
    return ApiResponse.ok(await service.messages(session, user, conversation_id))


@router.delete("/{conversation_id}/messages")
async def clear_messages(
    conversation_id: int,
    user: AuthenticatedUser = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> ApiResponse[None]:
    await service.clear_messages(session, user, conversation_id)
    return ApiResponse.ok(None)
