from datetime import datetime
from typing import cast
from uuid import uuid4

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ForbiddenError, NotFoundError
from app.core.security import AuthenticatedUser
from app.core.timezone import to_asia_shanghai
from app.db.models import ChatConversation, ChatMessage
from app.db.session import session_factory
from app.repositories.agent_workflow_repository import (
    ChatConversationPreparationStorePort,
    SqlAlchemyAgentWorkflowStore,
)
from app.repositories.conversation_repository import ConversationRepository, ConversationSummaryRecord
from app.runtime.uow import ApplicationUnitOfWorkFactory, SqlAlchemyApplicationUnitOfWorkFactory
from app.schemas.common import PageResult
from app.schemas.conversation import ConversationResponse, ConversationSummaryResponse, MessageResponse


def conversation_response(row: ChatConversation) -> ConversationResponse:
    return ConversationResponse(
        id=row.id,
        conversationNo=row.conversation_no,
        title=row.title,
        status=row.status,
        createdAt=row.created_at,
        updatedAt=row.updated_at,
    )


def message_response(row: ChatMessage) -> MessageResponse:
    return MessageResponse(
        id=row.id,
        role=row.role,
        content=row.content,
        sourcesJson=row.sources_json,
        retrievalScore=float(row.retrieval_score) if row.retrieval_score is not None else None,
        confidenceLevel=row.confidence_level,
        needHuman=row.need_human,
        createdAt=row.created_at,
    )


def conversation_summary_response(row: ConversationSummaryRecord) -> ConversationSummaryResponse:
    return ConversationSummaryResponse(
        id=row.id,
        conversationNo=row.conversation_no,
        title=row.title,
        status=row.status,
        createdAt=to_asia_shanghai(row.created_at),
        updatedAt=to_asia_shanghai(row.updated_at),
        messageCount=row.message_count,
        lastMessageAt=to_asia_shanghai(row.last_message_at) if row.last_message_at is not None else None,
    )


class ConversationService:
    def __init__(self, repository: ConversationRepository | None = None) -> None:
        self._repository = repository or ConversationRepository()

    async def list_owned(
        self,
        session: AsyncSession,
        user: AuthenticatedUser,
        *,
        page: int,
        size: int,
    ) -> PageResult[ConversationSummaryResponse]:
        rows, total = await self._repository.list_owned(
            session,
            owner_user_id=user.user_id,
            page=page,
            size=size,
        )
        return PageResult(
            page=page,
            size=size,
            total=total,
            records=[conversation_summary_response(row) for row in rows],
        )

    async def create(self, session: AsyncSession, user: AuthenticatedUser, title: str | None) -> ConversationResponse:
        now = datetime.now()
        row = ChatConversation(
            user_id=user.user_id,
            conversation_no="CV" + uuid4().hex[:16].upper(),
            title=title or "用户客服会话",
            status="ACTIVE",
            created_at=now,
            updated_at=now,
        )
        session.add(row)
        await session.commit()
        await session.refresh(row)
        return conversation_response(row)

    async def require_owned(
        self, session: AsyncSession, user: AuthenticatedUser, conversation_id: int
    ) -> ChatConversation:
        row = await session.get(ChatConversation, conversation_id)
        if row is None:
            raise NotFoundError("会话不存在")
        if row.user_id != user.user_id and user.role != "ADMIN":
            raise ForbiddenError("不能访问其他用户的会话")
        return row

    async def messages(
        self, session: AsyncSession, user: AuthenticatedUser, conversation_id: int
    ) -> list[MessageResponse]:
        await self.require_owned(session, user, conversation_id)
        rows = (
            (
                await session.execute(
                    select(ChatMessage)
                    .where(ChatMessage.conversation_id == conversation_id)
                    .order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc())
                )
            )
            .scalars()
            .all()
        )
        return [message_response(row) for row in rows]

    async def clear_messages(self, session: AsyncSession, user: AuthenticatedUser, conversation_id: int) -> None:
        await self.require_owned(session, user, conversation_id)
        await session.execute(delete(ChatMessage).where(ChatMessage.conversation_id == conversation_id))
        await session.commit()


class ChatConversationApplicationService:
    """Prepare one chat conversation in an independent, short application UoW."""

    def __init__(
        self,
        unit_of_work: ApplicationUnitOfWorkFactory[ChatConversationPreparationStorePort] | None = None,
    ) -> None:
        if unit_of_work is None:
            unit_of_work = cast(
                ApplicationUnitOfWorkFactory[ChatConversationPreparationStorePort],
                SqlAlchemyApplicationUnitOfWorkFactory(
                    session_factory(),
                    SqlAlchemyAgentWorkflowStore,
                ),
            )
        self._unit_of_work = unit_of_work

    async def prepare(
        self,
        *,
        actor: AuthenticatedUser,
        conversation_id: int | None,
        title: str,
    ) -> int:
        if conversation_id is None:
            now = datetime.now()
            async with self._unit_of_work.open(operation="chat.conversation.create") as uow:
                created_id = await uow.store.create_chat_conversation(
                    owner_user_id=actor.user_id,
                    conversation_no="CV" + uuid4().hex[:16].upper(),
                    title=title,
                    created_at=now,
                )
            return created_id

        async with self._unit_of_work.open(operation="chat.conversation.require_owned") as uow:
            ownership = await uow.store.conversation_ownership(conversation_id)
            if ownership is None:
                raise NotFoundError("会话不存在")
            if ownership.owner_user_id != actor.user_id and actor.role != "ADMIN":
                raise ForbiddenError("不能访问其他用户的会话")
        return conversation_id
