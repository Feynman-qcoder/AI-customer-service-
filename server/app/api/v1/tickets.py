from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import AppError, NotFoundError
from app.core.security import AuthenticatedUser, current_user, require_admin
from app.db.models import ChatConversation, SupportTicket, TicketOperationLog
from app.db.session import get_session
from app.schemas.common import ApiResponse, PageResult
from app.schemas.ticket import CreateTicketRequest, TicketResponse

router = APIRouter(tags=["tickets"])


def ticket_response(row: SupportTicket) -> TicketResponse:
    return TicketResponse(
        id=row.id,
        ticketNo=row.ticket_no,
        conversationId=row.conversation_id,
        category=row.category,
        status=row.status,
        description=row.description,
        contact=row.contact,
        handlingNote=row.handling_note,
        resolution=row.resolution,
        lockVersion=row.lock_version,
        createdAt=row.created_at,
        updatedAt=row.updated_at,
    )


@router.post("/tickets")
async def create_ticket(
    request: CreateTicketRequest,
    user: AuthenticatedUser = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> ApiResponse[TicketResponse]:
    conversation = await session.get(ChatConversation, request.conversationId)
    if conversation is None or conversation.user_id != user.user_id:
        raise NotFoundError("会话不存在")
    now = datetime.now()
    row = SupportTicket(
        user_id=user.user_id,
        ticket_no="TK" + now.strftime("%Y%m%d%H%M%S%f")[:17],
        conversation_id=request.conversationId,
        category=request.category,
        description=request.description,
        contact=request.contact,
        status="OPEN",
        created_at=now,
        updated_at=now,
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return ApiResponse.ok(ticket_response(row))


@router.get("/tickets")
async def my_tickets(
    page: int = Query(default=1, ge=1),
    size: int = Query(default=20, ge=1, le=100),
    status: str | None = None,
    user: AuthenticatedUser = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> ApiResponse[PageResult[TicketResponse]]:
    query = select(SupportTicket).where(SupportTicket.user_id == user.user_id)
    count_query = select(func.count()).select_from(SupportTicket).where(SupportTicket.user_id == user.user_id)
    if status:
        query = query.where(SupportTicket.status == status)
        count_query = count_query.where(SupportTicket.status == status)
    total = int((await session.execute(count_query)).scalar_one())
    rows = (
        (await session.execute(query.order_by(SupportTicket.created_at.desc()).offset((page - 1) * size).limit(size)))
        .scalars()
        .all()
    )
    return ApiResponse.ok(PageResult(page=page, size=size, total=total, records=[ticket_response(row) for row in rows]))


@router.get("/admin/tickets")
async def admin_tickets(
    page: int = Query(default=1, ge=1),
    size: int = Query(default=20, ge=1, le=100),
    status: str | None = None,
    _admin: AuthenticatedUser = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
) -> ApiResponse[PageResult[TicketResponse]]:
    query = select(SupportTicket)
    count_query = select(func.count()).select_from(SupportTicket)
    if status:
        query = query.where(SupportTicket.status == status)
        count_query = count_query.where(SupportTicket.status == status)
    total = int((await session.execute(count_query)).scalar_one())
    rows = (
        (await session.execute(query.order_by(SupportTicket.created_at.desc()).offset((page - 1) * size).limit(size)))
        .scalars()
        .all()
    )
    return ApiResponse.ok(PageResult(page=page, size=size, total=total, records=[ticket_response(row) for row in rows]))


@router.patch("/admin/tickets/{ticket_id}/status")
async def update_ticket_status(
    ticket_id: int,
    request: dict[str, object],
    admin: AuthenticatedUser = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
) -> ApiResponse[TicketResponse]:
    row = await session.get(SupportTicket, ticket_id)
    if row is None:
        raise NotFoundError("工单不存在")
    lock_value = request.get("lockVersion", -1)
    lock_version = int(lock_value) if isinstance(lock_value, int | str) else -1
    if lock_version != row.lock_version:
        raise AppError("工单已被其他人修改", 409)
    status_value = request.get("status")
    if not isinstance(status_value, str) or not status_value:
        raise AppError("工单状态不能为空", 400)
    previous = row.status
    row.status = status_value
    row.handling_note = str(request.get("handlingNote") or "")
    if request.get("resolution"):
        row.resolution = str(request["resolution"])
    row.lock_version += 1
    row.updated_at = datetime.now()
    session.add(
        TicketOperationLog(
            ticket_id=row.id,
            operator_id=admin.user_id,
            previous_status=previous,
            next_status=row.status,
            operation_note=row.handling_note,
            created_at=datetime.now(),
        )
    )
    await session.commit()
    await session.refresh(row)
    return ApiResponse.ok(ticket_response(row))
