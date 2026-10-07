from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel


class ProductResponse(BaseModel):
    id: int
    productCode: str
    productName: str
    category: str
    saleStatus: str
    price: Decimal
    stockQuantity: int
    dispatchRule: str
    afterSaleRule: str
    createdAt: datetime
    updatedAt: datetime


class ShipmentEventResponse(BaseModel):
    id: int
    carrier: str | None = None
    trackingNo: str | None = None
    status: str
    location: str | None = None
    eventNote: str
    eventTime: datetime


class OrderResponse(BaseModel):
    id: int
    orderNo: str
    userId: int
    product: ProductResponse
    quantity: int
    amount: Decimal
    status: str
    paidAt: datetime | None = None
    expectedShipAt: datetime | None = None
    shippedAt: datetime | None = None
    signedAt: datetime | None = None
    receiverName: str
    receiverPhone: str
    receiverAddress: str
    remark: str | None = None
    shipmentEvents: list[ShipmentEventResponse]
    createdAt: datetime
    updatedAt: datetime


class CreateOrderRequest(BaseModel):
    productId: int
    quantity: int
    receiverName: str
    receiverPhone: str
    receiverAddress: str
    remark: str | None = None


class UpdateOrderStatusRequest(BaseModel):
    status: str
    carrier: str | None = None
    trackingNo: str | None = None
    location: str | None = None
    eventNote: str | None = None
