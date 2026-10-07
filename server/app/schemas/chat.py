from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, StringConstraints

MAX_CONVERSATION_ID = (1 << 63) - 1
PositiveConversationId = Annotated[
    StrictInt,
    Field(gt=0, le=MAX_CONVERSATION_ID),
]


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversationId: PositiveConversationId | None = None
    question: str


class SourceReference(BaseModel):
    documentId: int
    fileName: str
    snippet: str
    score: float


class ChatResponse(BaseModel):
    conversationId: int
    answer: str
    sources: list[SourceReference]
    retrievalScore: float
    confidenceLevel: str
    needHuman: bool
    ticketId: int | None = None
    agentStatus: str | None = None
    confirmationPrompt: str | None = None
    confirmationChallengeDigest: str | None = None


ConfirmationText = Annotated[
    StrictStr,
    StringConstraints(min_length=1, max_length=256),
]
Sha256Digest = Annotated[
    StrictStr,
    StringConstraints(pattern=r"^[0-9a-f]{64}$"),
]


class CustomerConfirmationRequest(BaseModel):
    """The only client-controlled fields accepted at the durable resume edge."""

    model_config = ConfigDict(extra="forbid")

    conversationId: PositiveConversationId
    confirmationText: ConfirmationText
    confirmationChallengeDigest: Sha256Digest


class CustomerConfirmationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversationId: PositiveConversationId
    agentStatus: str
    confirmationPrompt: ConfirmationText
    confirmationChallengeDigest: Sha256Digest
    pendingActionId: PositiveConversationId | None = None
