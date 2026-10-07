from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import cast

from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ChatConversation, ChatMessage


@dataclass(frozen=True, slots=True)
class ConversationSummaryRecord:
    id: int
    conversation_no: str
    title: str
    status: str
    created_at: datetime
    updated_at: datetime
    message_count: int
    last_message_at: datetime | None


class ConversationRepository:
    async def list_owned(
        self,
        session: AsyncSession,
        *,
        owner_user_id: int,
        page: int,
        size: int,
    ) -> tuple[list[ConversationSummaryRecord], int]:
        message_stats = (
            select(
                ChatMessage.conversation_id.label("conversation_id"),
                func.count(ChatMessage.id).label("message_count"),
                func.max(ChatMessage.created_at).label("last_message_at"),
            )
            .group_by(ChatMessage.conversation_id)
            .subquery()
        )
        is_empty = message_stats.c.last_message_at.is_(None)
        empty_created_at = case(
            (is_empty, ChatConversation.created_at),
            else_=None,
        )
        statement = (
            select(
                ChatConversation.id.label("id"),
                ChatConversation.conversation_no.label("conversation_no"),
                ChatConversation.title.label("title"),
                ChatConversation.status.label("status"),
                ChatConversation.created_at.label("created_at"),
                ChatConversation.updated_at.label("updated_at"),
                func.coalesce(message_stats.c.message_count, 0).label("message_count"),
                message_stats.c.last_message_at.label("last_message_at"),
            )
            .outerjoin(
                message_stats,
                message_stats.c.conversation_id == ChatConversation.id,
            )
            .where(ChatConversation.user_id == owner_user_id)
            .order_by(
                case((is_empty, 1), else_=0).asc(),
                message_stats.c.last_message_at.desc(),
                empty_created_at.desc(),
                ChatConversation.id.desc(),
            )
            .offset((page - 1) * size)
            .limit(size)
        )
        mappings = (await session.execute(statement)).mappings().all()
        records = [
            ConversationSummaryRecord(
                id=int(row["id"]),
                conversation_no=cast(str, row["conversation_no"]),
                title=cast(str, row["title"]),
                status=cast(str, row["status"]),
                created_at=cast(datetime, row["created_at"]),
                updated_at=cast(datetime, row["updated_at"]),
                message_count=int(row["message_count"]),
                last_message_at=cast(datetime | None, row["last_message_at"]),
            )
            for row in mappings
        ]
        total = int(
            await session.scalar(
                select(func.count(ChatConversation.id)).where(
                    ChatConversation.user_id == owner_user_id
                )
            )
            or 0
        )
        return records, total
