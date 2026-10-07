from datetime import datetime

from pydantic import BaseModel


class CreateTicketRequest(BaseModel):
    conversationId: int
    description: str
    category: str
    contact: str | None = None


class TicketResponse(BaseModel):
    id: int
    ticketNo: str
    conversationId: int
    category: str
    status: str
    description: str
    contact: str | None = None
    handlingNote: str | None = None
    resolution: str | None = None
    lockVersion: int
    createdAt: datetime
    updatedAt: datetime
