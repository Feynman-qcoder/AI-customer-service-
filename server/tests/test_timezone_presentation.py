from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import AuthenticatedUser
from app.core.timezone import format_asia_shanghai, to_asia_shanghai, utc_now_naive
from app.db.models import CustomerOrder, ProductCatalog, ShipmentEvent
from app.services import commerce_service as commerce_module
from app.services.commerce_service import CommerceService, order_response

UTC_SAMPLE = datetime(2026, 10, 7, 10, 13)
SHANGHAI_SAMPLE = datetime(2026, 10, 7, 18, 13, tzinfo=timezone(timedelta(hours=8)))


class _Scalars:
    def __init__(self, values: list[ShipmentEvent]) -> None:
        self._values = values

    def scalars(self) -> "_Scalars":
        return self

    def all(self) -> list[ShipmentEvent]:
        return self._values


class _ResponseSession:
    def __init__(self, events: list[ShipmentEvent]) -> None:
        self.events = events

    async def execute(self, _statement: object) -> _Scalars:
        return _Scalars(self.events)


class _WriteSession(_ResponseSession):
    def __init__(self, product: ProductCatalog, order: CustomerOrder | None = None) -> None:
        super().__init__([])
        self.product = product
        self.order = order
        self.added: list[object] = []

    async def get(self, model: type[object], _identity: int, **_kwargs: object) -> object | None:
        if model is ProductCatalog:
            return self.product
        if model is CustomerOrder:
            return self.order
        return None

    def add(self, value: object) -> None:
        self.added.append(value)
        if isinstance(value, CustomerOrder):
            self.order = value
        if isinstance(value, ShipmentEvent):
            self.events.append(value)

    async def flush(self) -> None:
        if self.order is not None and self.order.id is None:
            self.order.id = 100

    async def commit(self) -> None:
        for index, event in enumerate(self.events, start=200):
            if event.id is None:
                event.id = index

    async def refresh(self, order: CustomerOrder, **_kwargs: object) -> None:
        order.product = self.product


def _product() -> ProductCatalog:
    return ProductCatalog(
        id=1,
        product_code="C20",
        product_name="轻氧洗面巾 C20",
        category="个护",
        sale_status="ON_SALE",
        price=Decimal("39.90"),
        stock_quantity=20,
        dispatch_rule="现货 8 小时内发货。",
        after_sale_rule="质量问题可申请售后。",
        created_at=UTC_SAMPLE,
        updated_at=UTC_SAMPLE,
    )


def _order(product: ProductCatalog) -> CustomerOrder:
    order = CustomerOrder(
        id=10,
        order_no="ORD20261007021302001",
        user_id=1,
        product_id=product.id,
        quantity=1,
        amount=product.price,
        status="WAITING_SHIPMENT",
        paid_at=UTC_SAMPLE,
        expected_ship_at=UTC_SAMPLE,
        shipped_at=UTC_SAMPLE,
        signed_at=UTC_SAMPLE,
        receiver_name="演示用户",
        receiver_phone="13800000000",
        receiver_address="演示地址",
        created_at=UTC_SAMPLE,
        updated_at=UTC_SAMPLE,
    )
    order.product = product
    return order


def _event(order: CustomerOrder) -> ShipmentEvent:
    return ShipmentEvent(
        id=20,
        order_id=order.id,
        status="CREATED",
        location="系统",
        event_note="等待处理",
        event_time=UTC_SAMPLE,
        created_at=UTC_SAMPLE,
    )


def test_naive_database_datetime_is_explicitly_interpreted_as_utc() -> None:
    converted = to_asia_shanghai(UTC_SAMPLE)

    assert converted == SHANGHAI_SAMPLE
    assert converted.isoformat() == "2026-10-07T18:13:00+08:00"


def test_aware_utc_datetime_converts_by_the_same_instant() -> None:
    converted = to_asia_shanghai(UTC_SAMPLE.replace(tzinfo=UTC))

    assert converted == SHANGHAI_SAMPLE
    assert converted.astimezone(UTC) == UTC_SAMPLE.replace(tzinfo=UTC)


def test_already_shanghai_aware_datetime_is_not_double_converted() -> None:
    converted = to_asia_shanghai(SHANGHAI_SAMPLE)

    assert converted == SHANGHAI_SAMPLE


def test_none_chat_time_retains_unsynchronised_copy() -> None:
    assert format_asia_shanghai(None) == "暂未同步"


@pytest.mark.asyncio
async def test_order_response_converts_all_presentation_fields_without_mutating_orm_values() -> None:
    product = _product()
    order = _order(product)
    event = _event(order)
    session = cast(AsyncSession, _ResponseSession([event]))

    response = await order_response(session, order)

    order_times = (
        response.paidAt,
        response.expectedShipAt,
        response.shippedAt,
        response.signedAt,
        response.createdAt,
        response.updatedAt,
    )
    assert all(value == SHANGHAI_SAMPLE for value in order_times)
    assert response.product.createdAt == SHANGHAI_SAMPLE
    assert response.product.updatedAt == SHANGHAI_SAMPLE
    assert response.shipmentEvents[0].eventTime == SHANGHAI_SAMPLE
    assert response.model_dump(mode="json")["expectedShipAt"] == "2026-10-07T18:13:00+08:00"

    assert order.paid_at is UTC_SAMPLE
    assert order.expected_ship_at is UTC_SAMPLE
    assert order.shipped_at is UTC_SAMPLE
    assert order.signed_at is UTC_SAMPLE
    assert order.created_at is UTC_SAMPLE
    assert order.updated_at is UTC_SAMPLE
    assert product.created_at is UTC_SAMPLE
    assert product.updated_at is UTC_SAMPLE
    assert event.event_time is UTC_SAMPLE


def test_database_value_and_api_value_represent_the_same_instant() -> None:
    api_value = to_asia_shanghai(UTC_SAMPLE)

    assert api_value.astimezone(UTC).replace(tzinfo=None) == UTC_SAMPLE


def test_durable_database_clock_authority_is_not_routed_through_presentation_time() -> None:
    server_root = Path(__file__).resolve().parents[1]
    authority_paths = (
        server_root / "app/repositories/durable_runtime_repository.py",
        server_root / "app/repositories/admin_decision_repository.py",
        server_root / "app/repositories/business_execution_repository.py",
    )
    sources = [path.read_text(encoding="utf-8") for path in authority_paths]

    assert all("app.core.timezone" not in source for source in sources)
    assert all('literal_column("CURRENT_TIMESTAMP(6)")' in source for source in sources)
    assert "TIMESTAMPADD(MICROSECOND" in sources[0]


def test_demo_seed_uses_the_utc_naive_write_clock_for_future_persistent_rows() -> None:
    seed_source = (Path(__file__).resolve().parents[1] / "scripts/seed_demo.py").read_text(encoding="utf-8")

    assert "datetime.now" not in seed_source
    assert "utc_now_naive()" in seed_source


@pytest.mark.parametrize("view_name", ["CustomerChat.vue", "AdminPanel.vue"])
def test_customer_and_admin_order_views_render_the_api_wall_clock_directly(view_name: str) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    source = (repository_root / "web/src/views" / view_name).read_text(encoding="utf-8")

    assert "formatDate(row.expectedShipAt)" in source or "formatDate(order.expectedShipAt)" in source
    assert "return value ? value.replace('T', ' ').slice(0, 16) : '暂未同步'" in source


def test_utc_now_naive_is_host_timezone_independent(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Clock(datetime):
        @classmethod
        def now(cls, tz: Any = None) -> "_Clock":
            assert tz is UTC
            return cls(2026, 10, 7, 10, 13, tzinfo=UTC)

    import app.core.timezone as timezone_module

    monkeypatch.setattr(timezone_module, "datetime", _Clock)

    assert utc_now_naive() == UTC_SAMPLE
    assert utc_now_naive().tzinfo is None


@pytest.mark.asyncio
async def test_create_order_persists_only_utc_naive_times(monkeypatch: pytest.MonkeyPatch) -> None:
    product = _product()
    session = _WriteSession(product)
    monkeypatch.setattr(commerce_module, "utc_now_naive", lambda: UTC_SAMPLE)

    response = await CommerceService().create_order(
        cast(AsyncSession, session),
        AuthenticatedUser(user_id=1, username="customer", name="演示用户", role="CUSTOMER"),
        product.id,
        1,
        "演示用户",
        "13800000000",
        "演示地址",
        None,
    )

    assert session.order is not None
    assert session.order.paid_at == UTC_SAMPLE
    assert session.order.created_at == UTC_SAMPLE
    assert session.order.updated_at == UTC_SAMPLE
    assert session.order.expected_ship_at == UTC_SAMPLE + timedelta(hours=8)
    assert all(
        value.tzinfo is None
        for value in (
            session.order.paid_at,
            session.order.created_at,
            session.order.updated_at,
            session.order.expected_ship_at,
        )
        if value is not None
    )
    event = next(value for value in session.added if isinstance(value, ShipmentEvent))
    assert event.event_time == UTC_SAMPLE
    assert event.created_at == UTC_SAMPLE
    assert event.event_time.tzinfo is None
    assert event.created_at.tzinfo is None
    assert response.createdAt.utcoffset() == timedelta(hours=8)


@pytest.mark.asyncio
async def test_status_update_persists_order_and_shipment_times_as_utc_naive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product = _product()
    order = _order(product)
    order.shipped_at = None
    session = _WriteSession(product, order)
    monkeypatch.setattr(commerce_module, "utc_now_naive", lambda: UTC_SAMPLE)

    response = await CommerceService().update_order_status(
        cast(AsyncSession, session),
        order.id,
        "SHIPPED",
        "演示快递",
        "TRACKING-1",
        "上海",
        None,
    )

    assert order.updated_at == UTC_SAMPLE
    assert order.shipped_at == UTC_SAMPLE
    assert order.updated_at.tzinfo is None
    assert order.shipped_at.tzinfo is None
    event = next(value for value in session.added if isinstance(value, ShipmentEvent))
    assert event.event_time == UTC_SAMPLE
    assert event.created_at == UTC_SAMPLE
    assert event.event_time.tzinfo is None
    assert event.created_at.tzinfo is None
    assert response.shippedAt == SHANGHAI_SAMPLE
