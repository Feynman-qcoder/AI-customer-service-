from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import AuthenticatedUser, current_user, require_admin
from app.db.session import get_session
from app.schemas.commerce import CreateOrderRequest, OrderResponse, ProductResponse, UpdateOrderStatusRequest
from app.schemas.common import ApiResponse, PageResult
from app.services.commerce_service import CommerceService

router = APIRouter(tags=["commerce"])
service = CommerceService()


@router.get("/products")
async def products(
    page: int = Query(default=1, ge=1),
    size: int = Query(default=20, ge=1, le=100),
    keyword: str | None = None,
    session: AsyncSession = Depends(get_session),
) -> ApiResponse[PageResult[ProductResponse]]:
    return ApiResponse.ok(await service.list_products(session, page, size, keyword))


@router.get("/orders")
async def my_orders(
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, ge=1, le=100),
    status: str | None = None,
    user: AuthenticatedUser = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> ApiResponse[PageResult[OrderResponse]]:
    return ApiResponse.ok(await service.list_my_orders(session, user, page, size, status))


@router.post("/orders")
async def create_order(
    request: CreateOrderRequest,
    user: AuthenticatedUser = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> ApiResponse[OrderResponse]:
    return ApiResponse.ok(
        await service.create_order(
            session,
            user,
            request.productId,
            request.quantity,
            request.receiverName,
            request.receiverPhone,
            request.receiverAddress,
            request.remark,
        )
    )


@router.get("/admin/products")
async def admin_products(
    page: int = Query(default=1, ge=1),
    size: int = Query(default=20, ge=1, le=100),
    keyword: str | None = None,
    _admin: AuthenticatedUser = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
) -> ApiResponse[PageResult[ProductResponse]]:
    return ApiResponse.ok(await service.list_products(session, page, size, keyword))


@router.get("/admin/orders")
async def admin_orders(
    page: int = Query(default=1, ge=1),
    size: int = Query(default=20, ge=1, le=100),
    status: str | None = None,
    keyword: str | None = None,
    _admin: AuthenticatedUser = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
) -> ApiResponse[PageResult[OrderResponse]]:
    return ApiResponse.ok(await service.list_all_orders(session, page, size, status, keyword))


@router.patch("/admin/orders/{order_id}/status")
async def update_order_status(
    order_id: int,
    request: UpdateOrderStatusRequest,
    _admin: AuthenticatedUser = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
) -> ApiResponse[OrderResponse]:
    return ApiResponse.ok(
        await service.update_order_status(
            session,
            order_id,
            request.status,
            request.carrier,
            request.trackingNo,
            request.location,
            request.eventNote,
        )
    )
