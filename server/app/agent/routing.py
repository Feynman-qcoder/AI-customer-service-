import re
from dataclasses import dataclass
from typing import Literal

from app.schemas.agent import AgentPlan, OrderReference

ORDER_NO_PATTERN = re.compile(r"(ORD[0-9A-Z]{8,})", re.IGNORECASE)
_REFUND_ELIGIBILITY_PATTERN = re.compile(
    r"(?:这个订单|那个订单|该订单|这笔订单|当前订单|这单|那单|"
    r"(?:订单\s*)?ORD[0-9A-Z]{8,})"
    r"[^。！？?\n]{0,24}"
    r"(?:能不能|可不可以|是否(?:还)?可|能否|还(?:能|可)|可以|能|可)"
    r"退(?:款|货)?(?:吗|么)?(?:[？?])?$",
    re.IGNORECASE,
)

_PRODUCT_ALIASES_BY_SKU: dict[str, tuple[str, ...]] = {
    "H100": ("暖风杯", "杯"),
    "C20": ("轻氧洗面巾", "洗面巾", "洗脸巾", "洁面巾"),
    "P9": ("云感靠枕", "靠枕", "枕头"),
}
_PRODUCT_SKU_ALTERNATION = "|".join(map(re.escape, _PRODUCT_ALIASES_BY_SKU))
_EXPLICIT_PRODUCT_SKU_PATTERN = re.compile(
    rf"(?<![A-Za-z0-9._-])(?:{_PRODUCT_SKU_ALTERNATION})(?![A-Za-z0-9._-])",
    re.IGNORECASE,
)
_BARE_EXTENDED_PRODUCT_IDENTIFIER_PATTERN = re.compile(
    rf"(?<![A-Za-z0-9._-])(?:{_PRODUCT_SKU_ALTERNATION})"
    r"[A-Za-z0-9._-]+(?![A-Za-z0-9._-])",
    re.IGNORECASE,
)
_PREFIXED_PRODUCT_IDENTIFIER_PATTERN = re.compile(
    r"(?:商品(?:编号|编码)|SKU)\s*[:：]?\s*"
    r"(?P<identifier>[A-Z0-9][A-Z0-9._-]{0,63})(?![A-Z0-9._-])",
    re.IGNORECASE,
)
_PRODUCT_ALIAS_LOOKUP: tuple[tuple[str, str], ...] = tuple(
    sorted(
        (
            (alias, sku)
            for sku, aliases in _PRODUCT_ALIASES_BY_SKU.items()
            for alias in aliases
        ),
        key=lambda item: len(item[0]),
        reverse=True,
    )
)


@dataclass(frozen=True, slots=True)
class _ProductReferenceResolution:
    status: Literal["NONE", "RESOLVED", "EXPLICIT_UNKNOWN", "AMBIGUOUS"]
    product_reference: str | None = None


def _resolve_product_reference(question: str) -> _ProductReferenceResolution:
    prefixed_identifiers = {
        match.group("identifier").upper()
        for match in _PREFIXED_PRODUCT_IDENTIFIER_PATTERN.finditer(question)
    }
    if prefixed_identifiers - _PRODUCT_ALIASES_BY_SKU.keys():
        return _ProductReferenceResolution(status="EXPLICIT_UNKNOWN")
    if _BARE_EXTENDED_PRODUCT_IDENTIFIER_PATTERN.search(question) is not None:
        return _ProductReferenceResolution(status="EXPLICIT_UNKNOWN")

    sku_targets = set(prefixed_identifiers)
    sku_targets.update(
        match.group(0).upper()
        for match in _EXPLICIT_PRODUCT_SKU_PATTERN.finditer(question)
    )
    if len(sku_targets) > 1:
        return _ProductReferenceResolution(status="AMBIGUOUS")
    if sku_targets:
        return _ProductReferenceResolution(
            status="RESOLVED",
            product_reference=next(iter(sku_targets)),
        )

    alias_targets = {
        sku
        for alias, sku in _PRODUCT_ALIAS_LOOKUP
        if alias in question
    }
    if len(alias_targets) > 1:
        return _ProductReferenceResolution(status="AMBIGUOUS")
    if alias_targets:
        return _ProductReferenceResolution(
            status="RESOLVED",
            product_reference=next(iter(alias_targets)),
        )
    return _ProductReferenceResolution(status="NONE")


def extract_product_reference(question: str) -> str | None:
    return _resolve_product_reference(question).product_reference


def has_explicit_product_reference(question: str) -> bool:
    return _resolve_product_reference(question).status != "NONE"


def is_refund_eligibility_question(question: str) -> bool:
    """Return whether an order-scoped question asks only about refund eligibility."""

    return _REFUND_ELIGIBILITY_PATTERN.search(question.strip()) is not None


def _has_product_intent(
    question: str,
    resolution: _ProductReferenceResolution,
) -> bool:
    if resolution.status != "NONE":
        return True
    return any(
        term in question
        for term in ("商品", "库存", "价格", "多少钱", "介绍", "资料", "参数")
    )


def build_rule_based_plan(question: str) -> AgentPlan:
    clean = question.strip()
    product_resolution = _resolve_product_reference(clean)
    if product_resolution.status in {"EXPLICIT_UNKNOWN", "AMBIGUOUS"}:
        return AgentPlan(
            intent="CLARIFICATION",
            goal=clean,
            order_reference=None,
            product_reference=None,
            required_tools=[],
            action_type=None,
            risk_level="LOW",
            requires_confirmation=False,
            missing_information=["product_reference"],
            decision_reason="商品引用未知或存在多个目标，需要用户明确一个规范商品编号。",
        )
    order_ref = _extract_order_reference(clean, product_resolution)
    action_type = classify_explicit_action(clean, order_ref)
    if action_type == "ORDER_CANCELLATION":
        return AgentPlan(
            intent="CANCEL_ORDER",
            goal=clean,
            order_reference=order_ref or OrderReference(latest="最近" in clean),
            product_reference=None,
            required_tools=["get_order_detail", "request_order_cancellation"],
            action_type="ORDER_CANCELLATION",
            risk_level="HIGH",
            requires_confirmation=True,
            missing_information=[] if order_ref else ["order_reference"],
            decision_reason="用户表达了取消订单意图，属于高风险有副作用操作。",
        )
    if action_type == "REFUND":
        product_ref = product_resolution.product_reference
        return AgentPlan(
            intent="REFUND_REQUEST",
            goal=clean,
            order_reference=order_ref,
            product_reference=product_ref,
            required_tools=["get_order_detail", "request_refund"],
            action_type="REFUND",
            risk_level="HIGH",
            requires_confirmation=True,
            missing_information=[]
            if order_ref or product_ref
            else ["order_reference"],
            decision_reason="用户表达了退款意图，必须经过确认和管理员审批。",
        )
    if (
        not (order_ref and order_ref.order_no)
        and not _is_recent_order_query(clean)
        and _is_order_list_query(clean)
    ):
        return AgentPlan(
            intent="ORDER_QUERY",
            goal=clean,
            order_reference=OrderReference(list_all=True),
            product_reference=None,
            required_tools=["list_my_orders"],
            action_type=None,
            risk_level="LOW",
            requires_confirmation=False,
            missing_information=[],
            decision_reason="用户希望查看自己已下单商品或订单列表，优先查询订单列表工具。",
        )
    product_ref = product_resolution.product_reference
    if order_ref and order_ref.latest and _is_recent_order_query(clean):
        shipping_progress = _is_shipping_query(clean) and not _is_shipping_rule_query(clean)
        return AgentPlan(
            intent="SHIPPING_QUERY" if shipping_progress else "ORDER_QUERY",
            goal=clean,
            order_reference=order_ref,
            product_reference=None,
            required_tools=["get_order_detail"],
            action_type=None,
            risk_level="LOW",
            requires_confirmation=False,
            missing_information=[],
            decision_reason=(
                "用户查询本人最近订单的实时物流进度，读取 owner-scoped 最近一笔订单。"
                if shipping_progress
                else "用户明确查询本人最近订单，读取 owner-scoped 最近一笔订单。"
            ),
        )
    if order_ref and order_ref.order_no and _is_shipping_rule_query(clean):
        return AgentPlan(
            intent="ORDER_QUERY",
            goal=clean,
            order_reference=order_ref,
            product_reference=None,
            required_tools=["get_order_detail"],
            action_type=None,
            risk_level="LOW",
            requires_confirmation=False,
            missing_information=[],
            decision_reason="用户提供了明确订单号并咨询发货规则，读取该订单商品规则和订单状态。",
        )
    if order_ref and order_ref.order_no and _is_shipping_query(clean):
        return AgentPlan(
            intent="SHIPPING_QUERY",
            goal=clean,
            order_reference=order_ref,
            product_reference=product_ref,
            required_tools=["get_order_detail"],
            action_type=None,
            risk_level="LOW",
            requires_confirmation=False,
            missing_information=[],
            decision_reason="用户提供了明确订单号并咨询物流或发货，查询该订单真实状态。",
        )
    if order_ref and order_ref.order_no and _is_product_info_query(clean):
        return AgentPlan(
            intent="PRODUCT_QUERY",
            goal=clean,
            order_reference=order_ref,
            product_reference=product_ref,
            required_tools=["get_order_detail", "get_product_information"],
            action_type=None,
            risk_level="LOW",
            requires_confirmation=False,
            missing_information=[],
            decision_reason="用户基于明确订单咨询商品资料，先查订单再读取该商品资料。",
        )
    if order_ref and order_ref.order_no and _is_after_sale_policy_query(clean):
        return AgentPlan(
            intent="KNOWLEDGE_QUERY",
            goal=clean,
            order_reference=order_ref,
            product_reference=product_ref,
            required_tools=["search_knowledge_base"],
            action_type=None,
            risk_level="LOW",
            requires_confirmation=False,
            missing_information=[],
            decision_reason="用户提供了明确订单号并咨询售后规则，结合订单状态检索售后规则。",
        )
    if order_ref and order_ref.order_no:
        return AgentPlan(
            intent="ORDER_QUERY",
            goal=clean,
            order_reference=order_ref,
            product_reference=None,
            required_tools=["get_order_detail"],
            action_type=None,
            risk_level="LOW",
            requires_confirmation=False,
            missing_information=[],
            decision_reason="用户提供了明确订单号，优先查询该订单真实状态。",
        )
    if product_ref and _is_product_rule_query(clean):
        return AgentPlan(
            intent="PRODUCT_QUERY",
            goal=clean,
            order_reference=None,
            product_reference=product_ref,
            required_tools=["get_product_information"],
            action_type=None,
            risk_level="LOW",
            requires_confirmation=False,
            missing_information=[],
            decision_reason="用户咨询商品规则或商品基础信息，读取商品资料表。",
        )
    if _is_after_sale_policy_query(clean):
        return AgentPlan(
            intent="KNOWLEDGE_QUERY",
            goal=clean,
            order_reference=order_ref,
            product_reference=product_ref,
            required_tools=["search_knowledge_base"],
            action_type=None,
            risk_level="LOW",
            requires_confirmation=False,
            missing_information=[],
            decision_reason="用户咨询退换货、破损、售后规则，进入知识库和结构化规则检索。",
        )
    if product_ref is None and _is_generic_shipping_policy_query(clean):
        return AgentPlan(
            intent="KNOWLEDGE_QUERY",
            goal=clean,
            order_reference=None,
            product_reference=None,
            required_tools=["search_knowledge_base"],
            action_type=None,
            risk_level="LOW",
            requires_confirmation=False,
            missing_information=[],
            decision_reason="用户咨询通用物流或配送政策，不读取历史订单。",
        )
    if _is_shipping_query(clean):
        return AgentPlan(
            intent="SHIPPING_QUERY",
            goal=clean,
            order_reference=order_ref,
            product_reference=product_ref,
            required_tools=["get_order_detail"],
            action_type=None,
            risk_level="LOW",
            requires_confirmation=False,
            missing_information=[],
            decision_reason="问题涉及订单物流或发货，优先使用订单工具。",
        )
    if _has_product_intent(clean, product_resolution):
        return AgentPlan(
            intent="PRODUCT_QUERY",
            goal=clean,
            order_reference=None if product_ref is not None else order_ref,
            product_reference=product_ref,
            required_tools=["get_product_information"],
            action_type=None,
            risk_level="LOW",
            requires_confirmation=False,
            missing_information=[],
            decision_reason="问题涉及商品资料，使用商品信息工具。",
        )
    return AgentPlan(
        intent="KNOWLEDGE_QUERY",
        goal=clean,
        order_reference=order_ref,
        product_reference=product_ref,
        required_tools=["search_knowledge_base"],
        action_type=None,
        risk_level="LOW",
        requires_confirmation=False,
        missing_information=[],
        decision_reason="未命中明确业务操作，进入知识库检索。",
    )


def classify_explicit_action(question: str, order_ref: OrderReference | None = None) -> str | None:
    resolved_ref = order_ref if order_ref is not None else _extract_order_reference(question)
    if _is_cancellation_action_request(question, resolved_ref):
        return "ORDER_CANCELLATION"
    if _is_refund_action_request(question, resolved_ref):
        return "REFUND"
    return None


def _is_cancellation_action_request(question: str, order_ref: OrderReference | None) -> bool:
    if not any(word in question for word in ["取消", "撤销"]):
        return False
    if any(
        word in question
        for word in [
            "不要取消",
            "不想取消",
            "别取消",
            "暂不取消",
            "不用取消",
            "无需取消",
            "不要撤销",
            "不想撤销",
        ]
    ):
        return False
    explicit_action_words = [
        "我要取消",
        "我想取消",
        "帮我取消",
        "请取消",
        "申请取消",
        "直接取消",
        "批量取消",
        "取消我",
        "取消这单",
        "帮我撤销",
        "我要撤销",
        "确认取消订单",
    ]
    if any(word in question for word in explicit_action_words):
        return True
    policy_words = ["如何", "怎么", "流程", "多久", "几天", "规则", "说明", "能", "可以", "吗", "是否", "？", "?"]
    if any(word in question for word in policy_words):
        return False
    return order_ref is not None


def _is_refund_action_request(question: str, order_ref: OrderReference | None) -> bool:
    if any(word in question for word in ["查询", "列出", "列出来", "所有用户"]):
        return False
    if any(
        word in question
        for word in [
            "不要退款",
            "不想退款",
            "不想要退款",
            "不需要退款",
            "无需退款",
            "别退款",
            "别给我退款",
            "暂不退款",
            "不用退款",
            "不要退钱",
        ]
    ):
        return False
    if not any(word in question for word in ["退款", "退钱", "退一下", "不想要"]):
        return False
    policy_words = [
        "如何",
        "怎么",
        "流程",
        "多久",
        "几天",
        "一般",
        "规则",
        "说明",
        "方式",
        "能",
        "可以",
        "吗",
        "是否",
        "？",
        "?",
    ]
    explicit_action_words = [
        "我要",
        "想要",
        "不想要",
        "帮我",
        "申请",
        "办理",
        "退一下",
        "给我退",
        "这单",
        "直接退款",
        "确认退款",
    ]
    if any(word in question for word in policy_words) and not any(word in question for word in explicit_action_words):
        return False
    if question.strip() in {"退一下", "退款", "退钱"}:
        return False
    return order_ref is not None or any(word in question for word in explicit_action_words)


def _is_after_sale_policy_query(question: str) -> bool:
    if is_refund_eligibility_question(question):
        return True
    terms = [
        "退货",
        "退款",
        "退钱",
        "售后",
        "破损",
        "损坏",
        "包装",
        "拆封",
        "换货",
        "能不能退",
        "怎么退",
        "漏液",
        "污渍",
        "坏了",
        "质量问题",
        "要拍什么",
        "还能退",
        "清洗",
    ]
    return any(term in question for term in terms)


def _is_shipping_query(question: str) -> bool:
    terms = ["物流", "快递", "发货", "到哪", "到哪里", "什么时候到", "包裹", "没动静", "出库"]
    return any(term in question for term in terms)


def _is_generic_shipping_policy_query(question: str) -> bool:
    if any(
        term in question
        for term in (
            "订单",
            "这单",
            "那单",
            "该订单",
            "包裹",
            "没动静",
            "到哪",
            "到哪里",
            "什么时候到",
        )
    ):
        return False
    policy_terms = (
        "物流时效",
        "配送时效",
        "配送范围",
        "物流规则",
        "配送规则",
        "发货规则",
        "发货时效",
        "出库规则",
        "偏远地区",
    )
    if any(term in question for term in policy_terms):
        return True
    return "一般" in question and "发货" in question and any(
        term in question for term in ("几天", "多久", "多长时间")
    )


def _is_shipping_rule_query(question: str) -> bool:
    terms = ["发货规则", "发货时效", "出库规则"]
    return any(term in question for term in terms)


def _is_product_rule_query(question: str) -> bool:
    terms = [
        "发货规则",
        "发货时效",
        "出库规则",
        "多久出库",
        "售后规则",
        "库存",
        "价格",
        "多少钱",
        "还有货",
        "在售",
        "商品资料",
        "介绍",
        "参数",
        "分类",
    ]
    return any(term in question for term in terms)


def _is_product_info_query(question: str) -> bool:
    terms = ["商品资料", "介绍", "参数", "分类", "这个商品", "商品信息", "卖点"]
    return any(term in question for term in terms)


def _extract_order_reference(
    question: str,
    product_resolution: _ProductReferenceResolution | None = None,
) -> OrderReference | None:
    match = ORDER_NO_PATTERN.search(question)
    if match:
        return OrderReference(order_no=match.group(1).upper())
    ordinal = _extract_ordinal_index(question)
    if ordinal is not None:
        return OrderReference(ordinal_index=ordinal)
    for keyword, index in [("第一个", 0), ("第一单", 0), ("第二个", 1), ("第二单", 1), ("第三个", 2), ("第三单", 2)]:
        if keyword in question:
            return OrderReference(ordinal_index=index)
    product = (
        product_resolution.product_reference
        if product_resolution is not None
        else extract_product_reference(question)
    )
    has_recent_reference = any(
        term in question for term in ("最近", "最新", "刚买", "刚下单")
    )
    has_order_context = product is not None or any(
        term in question for term in ("订单", "下单", "买的", "购买", "这单", "那单")
    )
    if has_recent_reference and has_order_context:
        return OrderReference(latest=True)
    return None


def _is_recent_order_query(question: str) -> bool:
    return any(
        phrase in question
        for phrase in (
            "最近订单",
            "最新订单",
            "最近一笔订单",
            "最新一笔订单",
            "最后一笔订单",
        )
    )


def _is_order_list_query(question: str) -> bool:
    order_words = ["订单", "下单", "买的", "购买", "商品"]
    list_words = ["所有", "全部", "列表", "列出", "查询", "查看", "分别", "哪些", "已经下单"]
    return any(word in question for word in order_words) and any(word in question for word in list_words)


def _extract_ordinal_index(question: str) -> int | None:
    match = re.search(r"第\s*(\d+)\s*(个|单|笔|条|件|号)?", question)
    if match:
        value = int(match.group(1))
        return value - 1 if value > 0 else None
    chinese_digits = {
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
        "十一": 11,
        "十二": 12,
        "十三": 13,
        "十四": 14,
        "十五": 15,
        "十六": 16,
        "十七": 17,
        "十八": 18,
        "十九": 19,
        "二十": 20,
    }
    for word, value in sorted(chinese_digits.items(), key=lambda item: len(item[0]), reverse=True):
        if f"第{word}" in question:
            return value - 1
    return None
