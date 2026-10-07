from datetime import datetime
from decimal import Decimal

import pytest

from app.db.models import CustomerOrder, ProductCatalog
from app.schemas.retrieval import RetrievalCandidate
from app.services.agent_service import AgentService


def _signed_pillow_order() -> CustomerOrder:
    product = ProductCatalog(
        id=2,
        product_code="P9",
        product_name="云感靠枕 P9",
        category="居家纺织品",
        sale_status="ON_SALE",
        price=Decimal("129.00"),
        stock_quantity=85,
        dispatch_rule="现货订单通常 24 小时内发货，定制颜色以页面预计时间为准。",
        after_sale_rule="未清洗、未明显使用且包装完整时可提交退货申请。",
    )
    order = CustomerOrder(
        id=13,
        order_no="ORD202607140003",
        user_id=1,
        product_id=2,
        quantity=1,
        amount=Decimal("129.00"),
        status="SIGNED",
        paid_at=datetime(2026, 7, 14, 9, 0),
        expected_ship_at=datetime(2026, 7, 17, 9, 0),
        shipped_at=datetime(2026, 7, 17, 9, 0),
        signed_at=datetime(2026, 7, 18, 9, 0),
        receiver_name="演示用户",
        receiver_phone="13800000000",
        receiver_address="演示地址",
    )
    order.product = product
    return order


@pytest.mark.parametrize(
    ("path", "question"),
    [
        ("single", "订单 ORD202607140003 什么时候发货？"),
        ("recent", "我的最近订单什么时候发货？"),
        ("second", "第二个订单什么时候发货？"),
    ],
)
def test_selected_order_answers_present_utc_naive_time_as_beijing_time(path: str, question: str) -> None:
    del path
    service = AgentService()
    order = _signed_pillow_order()
    order.expected_ship_at = datetime(2026, 10, 7, 10, 13)

    answer = service._order_answer(order, question)  # noqa: SLF001

    assert "2026-10-07 18:13（北京时间）" in answer
    assert "2026-10-07 10:13" not in answer


def test_order_list_and_multiple_order_answers_present_beijing_time() -> None:
    service = AgentService()
    order = _signed_pillow_order()
    order.expected_ship_at = datetime(2026, 10, 7, 10, 13)

    list_answer = service._order_list_answer([order])  # noqa: SLF001
    multiple_answer = service._multiple_order_answer([order])  # noqa: SLF001

    assert "2026-10-07 18:13（北京时间）" in list_answer
    assert "2026-10-07 18:13（北京时间）" in multiple_answer
    assert "2026-10-07 10:13" not in list_answer
    assert "2026-10-07 10:13" not in multiple_answer


def _knowledge_candidate(
    candidate_id: str,
    file_name: str,
    content: str,
    *,
    score: float = 0.8,
) -> RetrievalCandidate:
    return RetrievalCandidate(
        candidate_id=candidate_id,
        source_type="keyword",
        content=content,
        document_id=candidate_id.removeprefix("chunk:"),
        chunk_id=candidate_id.removeprefix("chunk:"),
        rule_id=None,
        metadata={"file_name": file_name},
        original_score=score,
        rerank_score=score,
    )


def test_after_sale_answer_is_customer_facing_not_raw_kb_dump() -> None:
    service = AgentService()
    answer = service._customer_after_sale_answer(  # noqa: SLF001
        _signed_pillow_order(),
        "我要退货退款",
        "退款处理说明 退款通常按原支付路径退回，不建议客服承诺具体到账分钟数。演示规则中...",
    )

    assert "云感靠枕 P9" in answer
    assert "原支付路径" in answer
    assert "退款处理说明" not in answer
    assert "不建议客服承诺" not in answer
    assert "演示规则" not in answer
    assert "#" not in answer


def test_damage_answer_is_clear_and_actionable() -> None:
    service = AgentService()
    answer = service._customer_after_sale_answer(_signed_pillow_order(), "商品包装破损")  # noqa: SLF001

    assert "外包装" in answer
    assert "照片或视频" in answer
    assert "售后申请" in answer


def test_order_bound_sources_do_not_include_other_product_documents() -> None:
    service = AgentService()
    order = _signed_pillow_order()
    candidates = [
        RetrievalCandidate(
            candidate_id="chunk:1",
            source_type="keyword",
            content="C20 个护耗材拆封后通常不支持无理由退货",
            document_id="1",
            chunk_id="1",
            rule_id=None,
            metadata={"file_name": "商品资料-轻氧洗面巾C20.md"},
            original_score=0.9,
        ),
        RetrievalCandidate(
            candidate_id="chunk:2",
            source_type="keyword",
            content="P9 未清洗、未明显使用且包装完整时可提交退货申请",
            document_id="2",
            chunk_id="2",
            rule_id=None,
            metadata={"file_name": "商品资料-云感靠枕P9.md"},
            original_score=0.8,
        ),
    ]

    sources = service._source_references(candidates, "订单 ORD202607140003 怎么退货", order)  # noqa: SLF001

    assert [source.fileName for source in sources] == ["商品资料-云感靠枕P9.md"]


def test_generic_unboxing_answer_distinguishes_return_categories() -> None:
    service = AgentService()
    candidates = [
        _knowledge_candidate(
            "chunk:10",
            "退换货政策.md",
            "普通非特殊商品仅拆开外包装且未使用时可申请退货；小家电可合理开箱检查；"
            "个护、食品、耗材拆封后一般不支持无理由退货，质量问题可申请售后。",
        )
    ]

    answer = service._knowledge_answer(candidates, question="商品拆封后还能退货吗？")  # noqa: SLF001

    assert "普通非特殊商品" in answer
    assert "小家电" in answer
    assert "合理开箱检查" in answer
    assert "个护、食品或耗材" in answer
    assert "不支持无理由退货" in answer
    assert "质量问题" in answer


def test_generic_unboxing_sources_exclude_product_docs_and_refund_timing() -> None:
    service = AgentService()
    candidates = [
        RetrievalCandidate(
            candidate_id="rule:10",
            source_type="structured_rule",
            content="通用售后规则",
            document_id=None,
            chunk_id=None,
            rule_id="10",
            metadata={"rule_title": "通用退货规则"},
            original_score=0.9,
            rerank_score=0.9,
        ),
        _knowledge_candidate("chunk:11", "退换货政策.md", "拆封后需按品类判断。"),
        _knowledge_candidate("chunk:12", "商品资料-暖风杯H100.md", "H100 可合理开箱检查。"),
        _knowledge_candidate("chunk:13", "商品资料-轻氧洗面巾C20.md", "C20 拆封后不支持无理由退货。"),
        _knowledge_candidate("chunk:14", "商品资料-云感靠枕P9.md", "P9 未清洗时可申请退货。"),
        _knowledge_candidate("chunk:15", "退款处理说明.md", "退款审核后原路退回。"),
    ]

    sources = service._source_references(candidates, "商品拆封后还能退货吗？")  # noqa: SLF001

    assert [source.fileName for source in sources] == ["售后规则：通用退货规则", "退换货政策.md"]


@pytest.mark.parametrize(
    ("question", "file_name", "content", "expected_terms"),
    [
        (
            "C20 拆封后还能退货吗？",
            "商品资料-轻氧洗面巾C20.md",
            "C20 属于个护耗材，拆封后通常不支持无理由退货；质量问题可提供照片和订单号申请售后。",
            ("不支持无理由退货", "质量问题", "照片", "订单号"),
        ),
        (
            "H100 拆封后还能退货吗？",
            "商品资料-暖风杯H100.md",
            "H100 可以合理开箱检查，未长时间通电、无划痕、配件齐全且包装完整时可申请退货。",
            ("合理开箱检查", "未长时间通电", "无明显划痕", "配件齐全", "包装完整"),
        ),
        (
            "P9 拆封后还能退货吗？",
            "商品资料-云感靠枕P9.md",
            "P9 未清洗、未明显使用、无污渍、无异味且吊牌和包装完整时可申请退货。",
            ("未清洗", "未明显使用", "无污渍", "无异味", "吊牌和包装完整"),
        ),
    ],
)
def test_product_specific_unboxing_answer_uses_matching_policy(
    question: str,
    file_name: str,
    content: str,
    expected_terms: tuple[str, ...],
) -> None:
    service = AgentService()
    candidates = [
        RetrievalCandidate(
            candidate_id="rule:20",
            source_type="structured_rule",
            content="通用结构化退货规则",
            document_id=None,
            chunk_id=None,
            rule_id="20",
            metadata={"rule_title": "通用退货规则"},
            original_score=0.95,
            rerank_score=0.95,
        ),
        _knowledge_candidate("chunk:20", file_name, content, score=0.8),
    ]

    answer = service._knowledge_answer(candidates, question=question)  # noqa: SLF001

    assert all(term in answer for term in expected_terms)


def test_remote_area_logistics_answer_and_sources_are_grounded_in_logistics_document() -> None:
    service = AgentService()
    candidates = [
        _knowledge_candidate("chunk:30", "退换货政策.md", "商品退货需要保持包装完整。", score=0.9),
        _knowledge_candidate(
            "chunk:31",
            "发货与物流规则.md",
            "发货后普通地区通常 2 到 5 天送达；偏远地区、节假日、大促期间可能延迟。",
            score=0.8,
        ),
    ]
    question = "偏远地区物流时效是多少？"

    answer = service._knowledge_answer(candidates, question=question)  # noqa: SLF001
    sources = service._source_references(candidates, question)  # noqa: SLF001

    assert "普通地区通常 2 到 5 天送达" in answer
    assert "偏远地区" in answer
    assert "可能延迟" in answer
    assert "没有统一" in answer
    assert "精确天数" in answer
    assert [source.fileName for source in sources] == ["发货与物流规则.md"]


@pytest.mark.parametrize(
    "question",
    [
        "这个订单能退吗？",
        "这个订单可以退吗？",
        "这个订单是否可退？",
        "这个订单能否退？",
        "这个订单可不可以退？",
        "这个订单还能退吗？",
        "该订单能退吗？",
    ],
)
def test_waiting_shipment_refund_eligibility_uses_known_order_status_without_creating_action(
    question: str,
) -> None:
    service = AgentService()
    order = _signed_pillow_order()
    order.status = "WAITING_SHIPMENT"

    answer = service._customer_after_sale_answer(order, question)  # noqa: SLF001

    assert "待发货" in answer
    assert "尚未发货" in answer
    assert "可以发起退款申请" in answer
    assert "需要明确确认" in answer
    assert "需要审核" in answer
    assert "本次询问不会创建退款申请" in answer
    assert "预计发货" not in answer


def test_shipped_refund_eligibility_remains_conditional_and_read_only() -> None:
    service = AgentService()
    order = _signed_pillow_order()
    order.status = "IN_TRANSIT"

    answer = service._customer_after_sale_answer(order, "这个订单能退吗？")  # noqa: SLF001

    assert "运输中" in answer
    assert "需要结合" in answer
    assert "退货" in answer or "拒收" in answer
    assert "审核" in answer
    assert "不会创建退款申请" in answer
