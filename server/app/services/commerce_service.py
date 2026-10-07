from datetime import datetime, timedelta
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.exceptions import ForbiddenError, NotFoundError
from app.core.security import AuthenticatedUser
from app.core.timezone import to_asia_shanghai, utc_now_naive
from app.db.models import CustomerOrder, ProductCatalog, ShipmentEvent
from app.schemas.commerce import OrderResponse, ProductResponse, ShipmentEventResponse
from app.schemas.common import PageResult


def product_response(product: ProductCatalog) -> ProductResponse:
    return ProductResponse(
        id=product.id,
        productCode=product.product_code,
        productName=product.product_name,
        category=product.category,
        saleStatus=product.sale_status,
        price=product.price,
        stockQuantity=product.stock_quantity,
        dispatchRule=product.dispatch_rule,
        afterSaleRule=product.after_sale_rule,
        createdAt=to_asia_shanghai(product.created_at),
        updatedAt=to_asia_shanghai(product.updated_at),
    )


def shipment_response(event: ShipmentEvent) -> ShipmentEventResponse:
    return ShipmentEventResponse(
        id=event.id,
        carrier=event.carrier,
        trackingNo=event.tracking_no,
        status=event.status,
        location=event.location,
        eventNote=event.event_note,
        eventTime=to_asia_shanghai(event.event_time),
    )


async def order_response(session: AsyncSession, order: CustomerOrder) -> OrderResponse:
    events = (
        (
            await session.execute(
                select(ShipmentEvent)
                .where(ShipmentEvent.order_id == order.id)
                .order_by(ShipmentEvent.event_time.desc())
            )
        )
        .scalars()
        .all()
    )
    return OrderResponse(
        id=order.id,
        orderNo=order.order_no,
        userId=order.user_id,
        product=product_response(order.product),
        quantity=order.quantity,
        amount=order.amount,
        status=order.status,
        paidAt=to_asia_shanghai(order.paid_at) if order.paid_at is not None else None,
        expectedShipAt=to_asia_shanghai(order.expected_ship_at) if order.expected_ship_at is not None else None,
        shippedAt=to_asia_shanghai(order.shipped_at) if order.shipped_at is not None else None,
        signedAt=to_asia_shanghai(order.signed_at) if order.signed_at is not None else None,
        receiverName=order.receiver_name,
        receiverPhone=order.receiver_phone,
        receiverAddress=order.receiver_address,
        remark=order.remark,
        shipmentEvents=[shipment_response(event) for event in events],
        createdAt=to_asia_shanghai(order.created_at),
        updatedAt=to_asia_shanghai(order.updated_at),
    )


class CommerceService:
    async def list_products(
        self, session: AsyncSession, page: int, size: int, keyword: str | None
    ) -> PageResult[ProductResponse]:
        query = select(ProductCatalog)
        count_query = select(func.count()).select_from(ProductCatalog)
        if keyword:
            like = f"%{keyword}%"
            query = query.where(ProductCatalog.product_name.like(like) | ProductCatalog.product_code.like(like))
            count_query = count_query.where(
                ProductCatalog.product_name.like(like) | ProductCatalog.product_code.like(like)
            )
        total = int((await session.execute(count_query)).scalar_one())
        rows = (
            (
                await session.execute(
                    query.order_by(ProductCatalog.updated_at.desc()).offset((page - 1) * size).limit(size)
                )
            )
            .scalars()
            .all()
        )
        return PageResult(page=page, size=size, total=total, records=[product_response(row) for row in rows])

    async def list_my_orders(
        self,
        session: AsyncSession,
        user: AuthenticatedUser,
        page: int,
        size: int,
        status: str | None,
    ) -> PageResult[OrderResponse]:
        query = (
            select(CustomerOrder)
            .options(selectinload(CustomerOrder.product))
            .where(CustomerOrder.user_id == user.user_id)
        )
        count_query = select(func.count()).select_from(CustomerOrder).where(CustomerOrder.user_id == user.user_id)
        if status:
            query = query.where(CustomerOrder.status == status)
            count_query = count_query.where(CustomerOrder.status == status)
        total = int((await session.execute(count_query)).scalar_one())
        orders = (
            (
                await session.execute(
                    query.order_by(CustomerOrder.created_at.desc()).offset((page - 1) * size).limit(size)
                )
            )
            .scalars()
            .all()
        )
        return PageResult(
            page=page, size=size, total=total, records=[await order_response(session, order) for order in orders]
        )

    async def list_all_orders(
        self,
        session: AsyncSession,
        page: int,
        size: int,
        status: str | None,
        keyword: str | None,
    ) -> PageResult[OrderResponse]:
        query = select(CustomerOrder).options(selectinload(CustomerOrder.product))
        count_query = select(func.count()).select_from(CustomerOrder)
        if status:
            query = query.where(CustomerOrder.status == status)
            count_query = count_query.where(CustomerOrder.status == status)
        if keyword:
            like = f"%{keyword}%"
            query = query.where(CustomerOrder.order_no.like(like))
            count_query = count_query.where(CustomerOrder.order_no.like(like))
        total = int((await session.execute(count_query)).scalar_one())
        orders = (
            (
                await session.execute(
                    query.order_by(CustomerOrder.created_at.desc()).offset((page - 1) * size).limit(size)
                )
            )
            .scalars()
            .all()
        )
        return PageResult(
            page=page, size=size, total=total, records=[await order_response(session, order) for order in orders]
        )

    async def create_order(
        self,
        session: AsyncSession,
        user: AuthenticatedUser,
        product_id: int,
        quantity: int,
        receiver_name: str,
        receiver_phone: str,
        receiver_address: str,
        remark: str | None,
    ) -> OrderResponse:
        product = await session.get(ProductCatalog, product_id)
        if product is None:
            raise NotFoundError("商品不存在")
        if product.sale_status != "ON_SALE":
            raise ForbiddenError("商品当前不可下单")
        if product.stock_quantity < quantity:
            raise ForbiddenError("商品库存不足")
        now = utc_now_naive()
        product.stock_quantity -= quantity
        product.updated_at = now
        order = CustomerOrder(
            order_no="ORD" + now.strftime("%Y%m%d%H%M%S%f")[:17],
            user_id=user.user_id,
            product_id=product.id,
            quantity=quantity,
            amount=Decimal(product.price) * Decimal(quantity),
            status="WAITING_SHIPMENT",
            paid_at=now,
            expected_ship_at=self._expected_ship_at(product, now),
            receiver_name=receiver_name,
            receiver_phone=receiver_phone,
            receiver_address=receiver_address,
            remark=remark,
            created_at=now,
            updated_at=now,
        )
        session.add(order)
        await session.flush()
        session.add(
            ShipmentEvent(
                order_id=order.id,
                status="CREATED",
                location="系统",
                event_note="订单已创建并支付，等待仓库处理",
                event_time=now,
                created_at=now,
            )
        )
        await session.commit()
        await session.refresh(order, attribute_names=["product"])
        return await order_response(session, order)

    async def update_order_status(
        self,
        session: AsyncSession,
        order_id: int,
        status: str,
        carrier: str | None,
        tracking_no: str | None,
        location: str | None,
        event_note: str | None,
    ) -> OrderResponse:
        order = await session.get(CustomerOrder, order_id, options=[selectinload(CustomerOrder.product)])
        if order is None:
            raise NotFoundError("订单不存在")
        now = utc_now_naive()
        order.status = status
        order.updated_at = now
        if status in {"SHIPPED", "IN_TRANSIT"} and order.shipped_at is None:
            order.shipped_at = now
        if status == "SIGNED" and order.signed_at is None:
            order.signed_at = now
        session.add(
            ShipmentEvent(
                order_id=order.id,
                carrier=carrier,
                tracking_no=tracking_no,
                status=self._shipment_status(status),
                location=location,
                event_note=event_note or self._default_shipment_note(status),
                event_time=now,
                created_at=now,
            )
        )
        await session.commit()
        return await order_response(session, order)

    def _expected_ship_at(self, product: ProductCatalog, now: datetime) -> datetime:
        if product.product_code == "H100":
            return now + timedelta(hours=48)
        if product.product_code == "P9":
            return now + timedelta(hours=24)
        return now + timedelta(hours=8)

    def _shipment_status(self, order_status: str) -> str:
        return {
            "SHIPPED": "PICKED_UP",
            "IN_TRANSIT": "IN_TRANSIT",
            "SIGNED": "DELIVERED",
        }.get(order_status, "CREATED")

    def _default_shipment_note(self, order_status: str) -> str:
        return {
            "SHIPPED": "包裹已交给快递",
            "IN_TRANSIT": "包裹运输中",
            "SIGNED": "订单已签收",
        }.get(order_status, "订单状态已更新")
