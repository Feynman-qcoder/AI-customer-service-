import asyncio
import hashlib
from datetime import timedelta
from decimal import Decimal
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.security import hash_password
from app.core.timezone import utc_now_naive
from app.db.models import (
    AfterSaleRule,
    AfterSaleRuleCondition,
    AfterSaleRuleVersion,
    CustomerOrder,
    KbChunk,
    KbDocument,
    ProductCatalog,
    ShipmentEvent,
    UserAccount,
)
from app.db.session import dispose_engine, session_factory


async def main() -> None:
    factory = session_factory()
    async with factory() as session:
        for username, password, name, role in [
            (settings.demo_customer_username, settings.demo_customer_password, "演示用户", "CUSTOMER"),
            (settings.demo_admin_username, settings.demo_admin_password, "后台管理员", "ADMIN"),
        ]:
            existing_user = (
                await session.execute(select(UserAccount).where(UserAccount.username == username))
            ).scalar_one_or_none()
            if existing_user is None:
                session.add(
                    UserAccount(
                        username=username,
                        password_hash=hash_password(password),
                        display_name=name,
                        role=role,
                        status="ACTIVE",
                        created_at=utc_now_naive(),
                        updated_at=utc_now_naive(),
                    )
                )
            elif existing_user.password_hash == password or not existing_user.password_hash.startswith("$2"):
                existing_user.password_hash = hash_password(password)
                existing_user.updated_at = utc_now_naive()
        await session.flush()
        products = [
            (
                "C20",
                "轻氧洗面巾 C20",
                "个护耗材",
                Decimal("39.90"),
                318,
                "工作日 18 点前付款通常当天出库，偏远地区以物流时效为准。",
                "个护耗材拆封后通常不支持无理由退货；质量问题可转人工核实。",
            ),
            (
                "H100",
                "暖风杯 H100",
                "小家电",
                Decimal("199.00"),
                40,
                "现货订单通常在付款后 48 小时内发货，预售或活动高峰可能顺延。",
                "未影响二次销售可申请退货；质量问题可提供照片或视频凭证申请换货。",
            ),
            (
                "P9",
                "云感靠枕 P9",
                "居家纺织品",
                Decimal("129.00"),
                85,
                "现货订单通常 24 小时内发货，定制颜色以页面预计时间为准。",
                "未清洗、未明显使用且包装完整时可提交退货申请。",
            ),
        ]
        for code, name, category, price, stock, dispatch, after_sale in products:
            existing_product = (
                await session.execute(select(ProductCatalog).where(ProductCatalog.product_code == code))
            ).scalar_one_or_none()
            if existing_product is None:
                session.add(
                    ProductCatalog(
                        product_code=code,
                        product_name=name,
                        category=category,
                        sale_status="ON_SALE",
                        price=price,
                        stock_quantity=stock,
                        dispatch_rule=dispatch,
                        after_sale_rule=after_sale,
                        created_at=utc_now_naive(),
                        updated_at=utc_now_naive(),
                    )
                )
            else:
                existing_product.product_name = name
                existing_product.category = category
                existing_product.sale_status = "ON_SALE"
                existing_product.price = price
                existing_product.stock_quantity = stock
                existing_product.dispatch_rule = dispatch
                existing_product.after_sale_rule = after_sale
                existing_product.updated_at = utc_now_naive()
        await _seed_policy_document(session)
        await _seed_structured_rules(session)
        await session.commit()

        user = (
            await session.execute(select(UserAccount).where(UserAccount.username == settings.demo_customer_username))
        ).scalar_one()
        cup = (await session.execute(select(ProductCatalog).where(ProductCatalog.product_code == "H100"))).scalar_one()
        towel = (await session.execute(select(ProductCatalog).where(ProductCatalog.product_code == "C20"))).scalar_one()
        if (
            await session.execute(select(CustomerOrder).where(CustomerOrder.user_id == user.id).limit(1))
        ).scalar_one_or_none() is None:
            now = utc_now_naive()
            for index, product in enumerate([towel, cup, cup], start=1):
                order = CustomerOrder(
                    order_no=f"ORD{now.strftime('%Y%m%d%H%M%S')}{index:03d}",
                    user_id=user.id,
                    product_id=product.id,
                    quantity=1,
                    amount=product.price,
                    status="WAITING_SHIPMENT",
                    paid_at=now - timedelta(hours=index),
                    expected_ship_at=now + timedelta(hours=8 * index),
                    receiver_name="张同学",
                    receiver_phone="13800000001",
                    receiver_address="上海市浦东新区演示路 100 号",
                    remark="演示订单",
                    created_at=now - timedelta(hours=index),
                    updated_at=now - timedelta(hours=index),
                )
                session.add(order)
                await session.flush()
                session.add(
                    ShipmentEvent(
                        order_id=order.id,
                        status="CREATED",
                        location="系统",
                        event_note="订单已创建并支付，等待仓库处理",
                        event_time=order.created_at,
                        created_at=order.created_at,
                    )
                )
            await session.commit()


async def _seed_policy_document(session: AsyncSession) -> None:
    filename = "demo-after-sale-policy.md"
    content = """售后与退换货政策
本文件用于演示企业知识库问答，不代表任何真实企业政策。

破损商品处理：
用户收到商品后发现破损、裂痕、漏液、无法正常使用，建议先保留商品、外包装、快递面单，并拍摄清晰照片或视频。客服可先引导用户提交售后申请；如破损情况明显，可以进入换货或补发流程。若缺少凭证，需要转人工进一步核实。

拆封后退货：
商品拆封不等于一定不能退。若商品未明显使用、配件齐全、包装没有严重损坏，且不影响二次销售，可以提交退货申请。若已经明显使用、配件缺失、包装严重破损，或属于个人护理类已影响二次销售的场景，需要转人工判断。

退款处理：
退款通常按原支付路径退回。未发货订单审核通过后，一般进入退款处理中；已发货或已签收订单，需要先完成退货或售后审核，再处理退款。

退货运费：
非质量问题、用户主观原因退货时，退回运费通常由用户承担；若商品存在质量问题、破损、错发或漏发，经审核属实后，退回运费由商家承担或补贴。活动包邮订单如发生部分退货，是否扣减运费以活动规则和人工审核为准。

质量问题换货：
商品存在演示规则中的质量问题时，可以申请换货。用户需要提供问题描述、照片或视频凭证，客服确认后进入售后审核。
"""
    now = utc_now_naive()
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    existing_document = (
        await session.execute(select(KbDocument).where(KbDocument.original_name == filename))
    ).scalar_one_or_none()
    if existing_document is not None:
        existing_document.file_size = len(content.encode("utf-8"))
        existing_document.file_sha256 = digest
        existing_document.chunk_count = 1
        existing_document.status = "READY"
        existing_document.updated_at = now
        existing_chunk = (
            await session.execute(select(KbChunk).where(KbChunk.document_id == existing_document.id).limit(1))
        ).scalar_one_or_none()
        if existing_chunk is not None:
            existing_chunk.content = content
            existing_chunk.content_hash = digest
            existing_chunk.char_count = len(content)
            existing_chunk.updated_at = now
        else:
            session.add(
                KbChunk(
                    document_id=existing_document.id,
                    chunk_index=0,
                    content=content,
                    content_hash=digest,
                    char_count=len(content),
                    vector_point_id=uuid4().hex,
                    created_at=now,
                    updated_at=now,
                )
            )
        return
    document = KbDocument(
        original_name=filename,
        storage_name=filename,
        storage_path=f"seed://{filename}",
        file_type="md",
        file_size=len(content.encode("utf-8")),
        file_sha256=digest,
        uploaded_by=None,
        status="READY",
        chunk_count=1,
        created_at=now,
        updated_at=now,
    )
    session.add(document)
    await session.flush()
    session.add(
        KbChunk(
            document_id=document.id,
            chunk_index=0,
            content=content,
            content_hash=digest,
            char_count=len(content),
            vector_point_id=uuid4().hex,
            created_at=now,
            updated_at=now,
        )
    )


async def _seed_structured_rules(session: AsyncSession) -> None:
    version = (
        await session.execute(select(AfterSaleRuleVersion).where(AfterSaleRuleVersion.version_code == "AS-2026-07"))
    ).scalar_one_or_none()
    if version is None:
        now = utc_now_naive()
        version = AfterSaleRuleVersion(
            version_code="AS-2026-07",
            description="演示售后规则版本",
            effective_from=now - timedelta(days=30),
            effective_to=None,
            status="ACTIVE",
            created_at=now,
        )
        session.add(version)
        await session.flush()

    rules = [
        (
            "AS-DAMAGE-001",
            "签收破损处理",
            "收到商品破损时，请先保留商品、外包装和快递面单，并拍摄照片或视频。若订单已签收且仍在售后期内，可提交质量/破损售后申请，客服审核后可换货、补发或转人工处理。",
            "QUALITY_DAMAGE",
            90,
            "ANY",
            "SIGNED",
            7,
        ),
        (
            "AS-RETURN-001",
            "拆封后退货判断",
            "商品拆封后仍可能退货。只要未明显使用、配件齐全、包装没有严重损坏，并且不影响二次销售，可以提交退货申请；如已明显使用、配件缺失或包装严重破损，需要转人工判断。",
            "RETURN",
            80,
            "ANY",
            "SIGNED",
            7,
        ),
        (
            "AS-REFUND-001",
            "退款路径说明",
            "退款通常按原支付路径退回。未发货订单审核通过后进入退款处理中；已发货或已签收订单需要先完成退货或售后审核，再处理退款。",
            "REFUND",
            70,
            "ANY",
            "ANY",
            None,
        ),
        (
            "AS-FREIGHT-001",
            "退货运费承担",
            "退货运费需要先区分原因：非质量问题或用户主观原因退货时，退回运费通常由用户承担；若商品质量问题、破损、错发或漏发经审核属实，退回运费由商家承担或补贴。活动包邮订单如发生部分退货，是否扣减运费以活动规则和人工审核为准。",
            "RETURN_FREIGHT",
            85,
            "ANY",
            "ANY",
            None,
        ),
    ]
    for code, title, content, after_sale_type, priority, category, order_status, signed_days in rules:
        existing_rule = (
            await session.execute(select(AfterSaleRule).where(AfterSaleRule.rule_code == code))
        ).scalar_one_or_none()
        if existing_rule is not None:
            existing_rule.version_id = version.id
            existing_rule.title = title
            existing_rule.content = content
            existing_rule.after_sale_type = after_sale_type
            existing_rule.priority = priority
            existing_rule.effective_from = utc_now_naive() - timedelta(days=30)
            existing_rule.effective_to = None
            existing_rule.status = "ACTIVE"
            existing_rule.updated_at = utc_now_naive()
            existing_condition = (
                await session.execute(
                    select(AfterSaleRuleCondition).where(AfterSaleRuleCondition.rule_id == existing_rule.id).limit(1)
                )
            ).scalar_one_or_none()
            if existing_condition is not None:
                existing_condition.product_category = None if category == "ANY" else category
                existing_condition.order_status = None if order_status == "ANY" else order_status
                existing_condition.payment_status = None
                existing_condition.shipment_status = None
                existing_condition.signed_within_days = signed_days
                existing_condition.after_sale_type = after_sale_type
            continue
        now = utc_now_naive()
        rule = AfterSaleRule(
            rule_code=code,
            version_id=version.id,
            title=title,
            content=content,
            after_sale_type=after_sale_type,
            priority=priority,
            effective_from=now - timedelta(days=30),
            effective_to=None,
            status="ACTIVE",
            created_at=now,
            updated_at=now,
        )
        session.add(rule)
        await session.flush()
        session.add(
            AfterSaleRuleCondition(
                rule_id=rule.id,
                product_category=None if category == "ANY" else category,
                order_status=None if order_status == "ANY" else order_status,
                payment_status=None,
                shipment_status=None,
                signed_within_days=signed_days,
                after_sale_type=after_sale_type,
                created_at=now,
            )
        )


if __name__ == "__main__":
    async def run() -> None:
        try:
            await main()
        finally:
            await dispose_engine()

    asyncio.run(run())
