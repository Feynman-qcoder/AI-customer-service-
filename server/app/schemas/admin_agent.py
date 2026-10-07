from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, StrictInt


class AgentActionResponse(BaseModel):
    id: int
    runId: str
    actionType: str
    targetOrderId: int | None
    actionPayloadJson: str
    riskLevel: str
    status: str
    idempotencyKey: str
    lockVersion: int
    createdBy: int
    approvedBy: int | None = None
    approvalNote: str | None = None
    logicalActionId: str | None = None
    confirmationMode: str
    customerConfirmedActorId: int | None = None
    customerConfirmedAt: datetime | None = None
    customerConfirmationChallengeDigest: str | None = None
    adminDecision: str | None = None
    adminDecidedActorId: int | None = None
    adminDecidedAt: datetime | None = None
    adminReasonCode: str | None = None
    resumeStatus: str
    decisionAvailable: bool
    executionResultCode: str | None = None
    executionErrorType: str | None = None
    executionErrorSummary: str | None = None
    legacyOriginalStatus: str | None = None
    createdAt: datetime
    approvedAt: datetime | None = None
    executedAt: datetime | None = None


class AgentToolCallResponse(BaseModel):
    id: int
    runId: str
    toolName: str
    redactedArgumentsJson: str
    resultSummary: str | None
    success: bool
    retryCount: int
    durationMs: int
    createdAt: datetime


class AgentStepResponse(BaseModel):
    id: int
    runId: str
    nodeName: str
    inputSummary: str | None
    outputSummary: str | None
    status: str
    durationMs: int
    errorSummary: str | None
    createdAt: datetime


class AgentRunResponse(BaseModel):
    id: int
    runId: str
    threadId: str
    conversationId: int
    userId: int
    status: str
    intent: str | None
    riskLevel: str | None
    startedAt: datetime
    completedAt: datetime | None
    finalAnswer: str | None
    errorType: str | None
    requestId: str
    modelName: str | None = None
    configVersion: str | None = None
    promptVersion: str | None = None
    providerLatencyMs: int | None = None
    promptTokens: int | None = None
    completionTokens: int | None = None
    toolCallCount: int = 0
    pendingActionCount: int = 0


class AgentRunDetailResponse(BaseModel):
    run: AgentRunResponse
    steps: list[AgentStepResponse]
    toolCalls: list[AgentToolCallResponse]
    actionRequests: list[AgentActionResponse]


class ApproveActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lockVersion: StrictInt = Field(ge=0)
    approvalNote: str | None = Field(default=None, max_length=512)


class RejectActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lockVersion: StrictInt = Field(ge=0)
    approvalNote: str = Field(min_length=1, max_length=512)


class ModelConfigRequest(BaseModel):
    temperature: float = Field(ge=0, le=2)
    topK: int = Field(ge=1, le=20)
    minRetrievalScore: float = Field(ge=0, le=1)
    mockEnabled: bool
