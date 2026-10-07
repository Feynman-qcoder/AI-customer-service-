from datetime import datetime

from pydantic import BaseModel


class CreateConversationRequest(BaseModel):
    title: str | None = None


class ConversationResponse(BaseModel):
    id: int
    conversationNo: str
    title: str
    status: str
    createdAt: datetime
    updatedAt: datetime


class ConversationSummaryResponse(BaseModel):
    id: int
    conversationNo: str
    title: str
    status: str
    createdAt: datetime
    updatedAt: datetime
    messageCount: int
    lastMessageAt: datetime | None = None


class MessageResponse(BaseModel):
    id: int
    role: str
    content: str
    sourcesJson: str | None = None
    retrievalScore: float | None = None
    confidenceLevel: str | None = None
    needHuman: bool
    createdAt: datetime
