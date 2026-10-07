from fastapi import APIRouter, Depends, Request
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import AppError, NotFoundError
from app.core.security import AuthenticatedUser, current_user, require_admin
from app.db.models import AgentActionRequest, AgentRun, AgentStep, AgentToolCall
from app.db.session import get_session
from app.runtime.admin_decision import is_admin_decision_available
from app.schemas.admin_agent import (
    AgentActionResponse,
    AgentRunDetailResponse,
    AgentRunResponse,
    AgentStepResponse,
    AgentToolCallResponse,
    ApproveActionRequest,
    RejectActionRequest,
)
from app.schemas.common import ApiResponse, PageResult
from app.services.admin_decision_application import (
    AdminDecision,
    AdminDecisionApplicationService,
    AdminDecisionCommand,
    AdminDecisionError,
)

router = APIRouter(tags=["admin-agent"])
actions_router = APIRouter(prefix="/admin/agent/actions")
agent_runs_router = APIRouter(prefix="/admin/agent/runs")
# Test-only override seam. Production composition is lifespan-owned app state.
admin_decision_application: AdminDecisionApplicationService | None = None


def action_response(row: AgentActionRequest) -> AgentActionResponse:
    return AgentActionResponse(
        id=row.id,
        runId=row.run_id,
        actionType=row.action_type,
        targetOrderId=row.target_order_id,
        actionPayloadJson=row.action_payload_json,
        riskLevel=row.risk_level,
        status=row.status,
        idempotencyKey=row.idempotency_key,
        lockVersion=row.lock_version,
        createdBy=row.created_by,
        approvedBy=row.approved_by,
        approvalNote=row.approval_note,
        logicalActionId=row.logical_action_id,
        confirmationMode=row.confirmation_mode,
        customerConfirmedActorId=row.customer_confirmed_actor_id,
        customerConfirmedAt=row.customer_confirmed_at,
        customerConfirmationChallengeDigest=(
            row.customer_confirmation_challenge_digest
        ),
        adminDecision=row.admin_decision,
        adminDecidedActorId=row.admin_decided_actor_id,
        adminDecidedAt=row.admin_decided_at,
        adminReasonCode=row.admin_reason_code,
        resumeStatus=row.resume_status,
        decisionAvailable=is_admin_decision_available(
            status=row.status,
            confirmation_mode=row.confirmation_mode,
            resume_status=row.resume_status,
            admin_decision=row.admin_decision,
        ),
        executionResultCode=row.execution_result_code,
        executionErrorType=row.execution_error_type,
        executionErrorSummary=row.execution_error_summary,
        legacyOriginalStatus=row.legacy_original_status,
        createdAt=row.created_at,
        approvedAt=row.approved_at,
        executedAt=row.executed_at,
    )


def tool_call_response(row: AgentToolCall) -> AgentToolCallResponse:
    return AgentToolCallResponse(
        id=row.id,
        runId=row.run_id,
        toolName=row.tool_name,
        redactedArgumentsJson=row.redacted_arguments_json,
        resultSummary=row.result_summary,
        success=row.success,
        retryCount=row.retry_count,
        durationMs=row.duration_ms,
        createdAt=row.created_at,
    )


def step_response(row: AgentStep) -> AgentStepResponse:
    return AgentStepResponse(
        id=row.id,
        runId=row.run_id,
        nodeName=row.node_name,
        inputSummary=row.input_summary,
        outputSummary=row.output_summary,
        status=row.status,
        durationMs=row.duration_ms,
        errorSummary=row.error_summary,
        createdAt=row.created_at,
    )


def run_response(row: AgentRun, tool_call_count: int = 0, pending_action_count: int = 0) -> AgentRunResponse:
    return AgentRunResponse(
        id=row.id,
        runId=row.run_id,
        threadId=row.thread_id,
        conversationId=row.conversation_id,
        userId=row.user_id,
        status=row.status,
        intent=row.intent,
        riskLevel=row.risk_level,
        startedAt=row.started_at,
        completedAt=row.completed_at,
        finalAnswer=row.final_answer,
        errorType=row.error_type,
        requestId=row.request_id,
        modelName=row.model_name,
        configVersion=row.config_version,
        promptVersion=row.prompt_version,
        providerLatencyMs=row.provider_latency_ms,
        promptTokens=row.prompt_tokens,
        completionTokens=row.completion_tokens,
        toolCallCount=tool_call_count,
        pendingActionCount=pending_action_count,
    )


@agent_runs_router.get("")
async def list_runs(
    page: int = 1,
    size: int = 20,
    status: str | None = None,
    intent: str | None = None,
    _admin: AuthenticatedUser = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
) -> ApiResponse[PageResult[AgentRunResponse]]:
    page = max(page, 1)
    size = min(max(size, 1), 100)
    query = select(AgentRun)
    count_query = select(func.count()).select_from(AgentRun)
    if status:
        query = query.where(AgentRun.status == status)
        count_query = count_query.where(AgentRun.status == status)
    if intent:
        query = query.where(AgentRun.intent == intent)
        count_query = count_query.where(AgentRun.intent == intent)
    total = int((await session.execute(count_query)).scalar_one())
    rows = (
        (await session.execute(query.order_by(AgentRun.started_at.desc()).offset((page - 1) * size).limit(size)))
        .scalars()
        .all()
    )
    if not rows:
        return ApiResponse.ok(PageResult(page=page, size=size, total=total, records=[]))

    run_ids = [row.run_id for row in rows]
    tool_count_rows = (
        await session.execute(
            select(AgentToolCall.run_id, func.count())
            .where(AgentToolCall.run_id.in_(run_ids))
            .group_by(AgentToolCall.run_id)
        )
    ).tuples()
    tool_counts: dict[str, int] = {run_id: int(count) for run_id, count in tool_count_rows}
    pending_action_count_rows = (
        await session.execute(
            select(AgentActionRequest.run_id, func.count())
            .where(AgentActionRequest.run_id.in_(run_ids), AgentActionRequest.status == "PENDING")
            .group_by(AgentActionRequest.run_id)
        )
    ).tuples()
    pending_action_counts: dict[str, int] = (
        {run_id: int(count) for run_id, count in pending_action_count_rows}
    )
    records = [
        run_response(
            row,
            tool_call_count=int(tool_counts.get(row.run_id, 0)),
            pending_action_count=int(pending_action_counts.get(row.run_id, 0)),
        )
        for row in rows
    ]
    return ApiResponse.ok(PageResult(page=page, size=size, total=total, records=records))


@agent_runs_router.get("/{run_id}")
async def get_run(
    run_id: str,
    _admin: AuthenticatedUser = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
) -> ApiResponse[AgentRunDetailResponse]:
    run = (await session.execute(select(AgentRun).where(AgentRun.run_id == run_id))).scalar_one_or_none()
    if run is None:
        raise NotFoundError("Agent运行记录不存在")
    steps = (
        (
            await session.execute(
                select(AgentStep).where(AgentStep.run_id == run_id).order_by(AgentStep.created_at.asc())
            )
        )
        .scalars()
        .all()
    )
    tool_calls = (
        (
            await session.execute(
                select(AgentToolCall).where(AgentToolCall.run_id == run_id).order_by(AgentToolCall.created_at.asc())
            )
        )
        .scalars()
        .all()
    )
    actions = (
        (
            await session.execute(
                select(AgentActionRequest)
                .where(AgentActionRequest.run_id == run_id)
                .order_by(AgentActionRequest.created_at.asc())
            )
        )
        .scalars()
        .all()
    )
    pending_actions = sum(1 for action in actions if action.status == "PENDING")
    return ApiResponse.ok(
        AgentRunDetailResponse(
            run=run_response(run, tool_call_count=len(tool_calls), pending_action_count=pending_actions),
            steps=[step_response(row) for row in steps],
            toolCalls=[tool_call_response(row) for row in tool_calls],
            actionRequests=[action_response(row) for row in actions],
        )
    )


@actions_router.get("")
async def list_actions(
    status: str | None = None,
    _admin: AuthenticatedUser = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
) -> ApiResponse[list[AgentActionResponse]]:
    query = select(AgentActionRequest).order_by(AgentActionRequest.created_at.desc())
    if status:
        query = query.where(AgentActionRequest.status == status)
    rows = (await session.execute(query.limit(100))).scalars().all()
    return ApiResponse.ok([action_response(row) for row in rows])


@actions_router.get("/{action_id}")
async def get_action(
    action_id: int,
    _admin: AuthenticatedUser = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
) -> ApiResponse[AgentActionResponse]:
    row = await session.get(AgentActionRequest, action_id)
    if row is None:
        raise NotFoundError("审批请求不存在")
    return ApiResponse.ok(action_response(row))


async def _submit_admin_decision(
    *,
    action_id: int,
    lock_version: int,
    approval_note: str | None,
    decision: AdminDecision,
    http_request: Request,
    admin: AuthenticatedUser,
) -> ApiResponse[AgentActionResponse]:
    application = admin_decision_application or getattr(
        http_request.app.state,
        "admin_decision_application",
        None,
    )
    if application is None:
        raise AppError("管理员决定服务尚未就绪", 503)
    try:
        result = await application.decide_and_resume(
            admin,
            AdminDecisionCommand(
                action_id=action_id,
                lock_version=lock_version,
                decision=decision,
                approval_note=approval_note,
            ),
        )
    except AdminDecisionError as exc:
        raise AppError("管理员决定未被接受", exc.status_code) from exc
    return ApiResponse.ok(result)


async def approve_action(
    action_id: int,
    request: ApproveActionRequest,
    http_request: Request,
    admin: AuthenticatedUser,
    session: AsyncSession | None = None,
) -> ApiResponse[AgentActionResponse]:
    del session
    return await _submit_admin_decision(
        action_id=action_id,
        lock_version=request.lockVersion,
        approval_note=request.approvalNote,
        decision=AdminDecision.APPROVE,
        http_request=http_request,
        admin=admin,
    )


async def reject_action(
    action_id: int,
    request: RejectActionRequest,
    http_request: Request,
    admin: AuthenticatedUser,
    session: AsyncSession | None = None,
) -> ApiResponse[AgentActionResponse]:
    del session
    return await _submit_admin_decision(
        action_id=action_id,
        lock_version=request.lockVersion,
        approval_note=request.approvalNote,
        decision=AdminDecision.REJECT,
        http_request=http_request,
        admin=admin,
    )


@actions_router.post("/{action_id}/approve")
async def approve_action_endpoint(
    action_id: int,
    request: ApproveActionRequest,
    http_request: Request,
    actor: AuthenticatedUser = Depends(current_user),
) -> ApiResponse[AgentActionResponse]:
    return await approve_action(
        action_id,
        request,
        http_request,
        actor,
    )


@actions_router.post("/{action_id}/reject")
async def reject_action_endpoint(
    action_id: int,
    request: RejectActionRequest,
    http_request: Request,
    actor: AuthenticatedUser = Depends(current_user),
) -> ApiResponse[AgentActionResponse]:
    return await reject_action(
        action_id,
        request,
        http_request,
        actor,
    )

router.include_router(actions_router)
router.include_router(agent_runs_router)
