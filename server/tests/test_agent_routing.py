import pytest

from app.agent.routing import (
    build_rule_based_plan,
    extract_product_reference,
    has_explicit_product_reference,
    is_refund_eligibility_question,
)


def test_shipping_question_with_product_keyword_routes_to_order_tool() -> None:
    plan = build_rule_based_plan("我的洗脸巾物流到哪里了")

    assert plan.intent == "SHIPPING_QUERY"
    assert plan.order_reference is None
    assert plan.product_reference == "C20"
    assert plan.required_tools == ["get_order_detail"]
    assert plan.risk_level == "LOW"
    assert plan.requires_confirmation is False


def test_recent_product_shipping_question_keeps_product_keyword() -> None:
    plan = build_rule_based_plan("我刚买的杯子什么时候发货？")

    assert plan.intent == "SHIPPING_QUERY"
    assert plan.order_reference is not None
    assert plan.order_reference.product_keyword is None
    assert plan.order_reference.latest is True
    assert plan.product_reference == "H100"
    assert plan.required_tools == ["get_order_detail"]


@pytest.mark.parametrize(
    "question",
    [
        "偏远地区物流时效是多少？",
        "一般几天发货？",
        "物流时效规则是什么？",
        "配送范围有哪些？",
    ],
)
def test_generic_shipping_policy_routes_to_knowledge_without_order_target(
    question: str,
) -> None:
    plan = build_rule_based_plan(question)

    assert plan.intent == "KNOWLEDGE_QUERY"
    assert plan.order_reference is None
    assert plan.product_reference is None
    assert plan.required_tools == ["search_knowledge_base"]


@pytest.mark.parametrize("question", ["我的最近订单", "最近一笔订单"])
def test_recent_order_routes_to_owner_scoped_order_detail(question: str) -> None:
    plan = build_rule_based_plan(question)

    assert plan.intent == "ORDER_QUERY"
    assert plan.order_reference is not None
    assert plan.order_reference.latest is True
    assert plan.order_reference.product_keyword is None
    assert plan.product_reference is None
    assert plan.required_tools == ["get_order_detail"]


@pytest.mark.parametrize("question", ["最近订单物流到哪里了", "最新订单快递到哪了"])
def test_recent_order_shipping_progress_routes_to_shipping_query(question: str) -> None:
    plan = build_rule_based_plan(question)

    assert plan.intent == "SHIPPING_QUERY"
    assert plan.order_reference is not None
    assert plan.order_reference.latest is True
    assert plan.required_tools == ["get_order_detail"]
    assert plan.risk_level == "LOW"
    assert plan.requires_confirmation is False


def test_recent_order_shipping_rule_remains_order_query() -> None:
    plan = build_rule_based_plan("最近订单发货规则")

    assert plan.intent == "ORDER_QUERY"
    assert plan.order_reference is not None
    assert plan.order_reference.latest is True
    assert plan.required_tools == ["get_order_detail"]
    assert plan.risk_level == "LOW"
    assert plan.requires_confirmation is False


@pytest.mark.parametrize(
    ("question", "expected_sku", "expected_latest"),
    [
        ("我的洗脸巾物流到哪里了", "C20", False),
        ("我刚买的杯子什么时候发货？", "H100", True),
        ("C20 订单物流到哪了？", "C20", False),
    ],
)
def test_order_filter_sku_is_only_persisted_as_product_reference(
    question: str,
    expected_sku: str,
    expected_latest: bool,
) -> None:
    plan = build_rule_based_plan(question)

    assert plan.intent == "SHIPPING_QUERY"
    assert plan.product_reference == expected_sku
    assert plan.order_reference is None or plan.order_reference.product_keyword is None
    assert bool(plan.order_reference and plan.order_reference.latest) is expected_latest


def test_generic_after_sale_question_has_no_persisted_target() -> None:
    plan = build_rule_based_plan("商品拆封后还能退货吗？")

    assert plan.intent == "KNOWLEDGE_QUERY"
    assert plan.order_reference is None
    assert plan.product_reference is None


def test_explicit_product_follow_up_has_only_canonical_product_reference() -> None:
    plan = build_rule_based_plan("商品 P9 这个商品拆封后还能退货吗？")

    assert plan.intent == "KNOWLEDGE_QUERY"
    assert plan.product_reference == "P9"
    assert plan.order_reference is None


def test_ordinal_order_reference_is_extracted() -> None:
    plan = build_rule_based_plan("第三个商品物流到哪里了")

    assert plan.intent == "SHIPPING_QUERY"
    assert plan.order_reference is not None
    assert plan.order_reference.ordinal_index == 2


def test_numeric_and_chinese_ordinal_order_reference_is_extracted() -> None:
    numeric = build_rule_based_plan("第13个订单物流到哪里了")
    chinese = build_rule_based_plan("第十三个订单物流到哪里了")

    assert numeric.order_reference is not None
    assert numeric.order_reference.ordinal_index == 12
    assert chinese.order_reference is not None
    assert chinese.order_reference.ordinal_index == 12


def test_all_orders_question_routes_to_order_list_tool() -> None:
    plan = build_rule_based_plan("查询所有订单")

    assert plan.intent == "ORDER_QUERY"
    assert plan.order_reference is not None
    assert plan.order_reference.list_all is True
    assert plan.required_tools == ["list_my_orders"]


def test_explicit_order_no_routes_to_order_query() -> None:
    plan = build_rule_based_plan("ORD202607140003")

    assert plan.intent == "ORDER_QUERY"
    assert plan.order_reference is not None
    assert plan.order_reference.order_no == "ORD202607140003"
    assert plan.required_tools == ["get_order_detail"]


def test_explicit_order_no_with_shipping_rule_uses_order_tool() -> None:
    plan = build_rule_based_plan("订单 ORD202607140003 发货规则")

    assert plan.intent == "ORDER_QUERY"
    assert plan.order_reference is not None
    assert plan.order_reference.order_no == "ORD202607140003"
    assert plan.required_tools == ["get_order_detail"]


def test_product_code_with_shipping_rule_uses_product_tool() -> None:
    plan = build_rule_based_plan("C20 发货规则")

    assert plan.intent == "PRODUCT_QUERY"
    assert plan.order_reference is None
    assert plan.product_reference == "C20"
    assert plan.required_tools == ["get_product_information"]


def test_product_name_with_shipping_rule_uses_product_tool() -> None:
    plan = build_rule_based_plan("云感靠枕 P9 发货规则")

    assert plan.intent == "PRODUCT_QUERY"
    assert plan.order_reference is None
    assert plan.product_reference == "P9"
    assert plan.required_tools == ["get_product_information"]


def test_order_product_intro_uses_product_query_with_order_tool() -> None:
    plan = build_rule_based_plan("介绍一下订单 ORD202607140003 这个商品")

    assert plan.intent == "PRODUCT_QUERY"
    assert plan.order_reference is not None
    assert plan.order_reference.order_no == "ORD202607140003"
    assert plan.required_tools == ["get_order_detail", "get_product_information"]


def test_product_intro_question_uses_product_tool() -> None:
    plan = build_rule_based_plan("介绍一下暖风杯 H100")

    assert plan.intent == "PRODUCT_QUERY"
    assert plan.product_reference == "H100"
    assert plan.required_tools == ["get_product_information"]


def test_order_product_material_question_uses_order_product_tool() -> None:
    plan = build_rule_based_plan("我要这个订单 ORD202607140003 的商品资料")

    assert plan.intent == "PRODUCT_QUERY"
    assert plan.order_reference is not None
    assert plan.order_reference.order_no == "ORD202607140003"
    assert plan.required_tools == ["get_order_detail", "get_product_information"]


def test_product_material_follow_up_without_order_falls_back_to_knowledge() -> None:
    plan = build_rule_based_plan("我要这个的商品资料")

    assert plan.intent == "PRODUCT_QUERY"
    assert plan.order_reference is None
    assert plan.product_reference is None


def test_product_quality_problem_routes_to_knowledge() -> None:
    plan = build_rule_based_plan("洗脸巾包装破损怎么办")

    assert plan.intent == "KNOWLEDGE_QUERY"
    assert plan.product_reference == "C20"
    assert plan.required_tools == ["search_knowledge_base"]


@pytest.mark.parametrize(
    ("question", "expected_sku"),
    [
        ("暖风杯 H100 还有库存吗？", "H100"),
        ("H100 还有库存吗？", "H100"),
        ("轻氧洗面巾 C20 有库存吗？", "C20"),
        ("C20 库存多少？", "C20"),
        ("云感靠枕 P9 还有库存吗？", "P9"),
        ("P9 库存多少？", "P9"),
        ("暖风杯还有库存吗？", "H100"),
        ("洗面巾还有库存吗？", "C20"),
        ("靠枕还有库存吗？", "P9"),
        ("暖风杯 h100 还有库存吗？", "H100"),
        ("暖风杯 C20 还有库存吗？", "C20"),
    ],
)
def test_product_reference_is_canonical_sku(
    question: str,
    expected_sku: str,
) -> None:
    plan = build_rule_based_plan(question)

    assert plan.intent == "PRODUCT_QUERY"
    assert plan.product_reference == expected_sku
    assert plan.order_reference is None
    assert plan.required_tools == ["get_product_information"]
    assert plan.action_type is None
    assert plan.risk_level == "LOW"
    assert plan.requires_confirmation is False
    assert extract_product_reference(question) == expected_sku
    assert has_explicit_product_reference(question) is True


def test_unknown_product_name_does_not_invent_sku() -> None:
    plan = build_rule_based_plan("神秘水壶还有库存吗？")

    assert plan.intent == "PRODUCT_QUERY"
    assert plan.product_reference is None
    assert plan.action_type is None
    assert plan.requires_confirmation is False
    assert extract_product_reference("神秘水壶还有库存吗？") is None
    assert has_explicit_product_reference("神秘水壶还有库存吗？") is False


def test_prefixed_unknown_product_is_explicit_without_inventing_sku() -> None:
    question = "商品编号 UNKNOWN999 还有库存吗？"

    assert extract_product_reference(question) is None
    assert has_explicit_product_reference(question) is True


@pytest.mark.parametrize(
    "question",
    [
        "商品编号 H100-PRO 还有库存吗？",
        "SKU C20.2 还有库存吗？",
        "商品编码 P9_TEST 还有库存吗？",
        "商品编号 X200 的杯子还有库存吗？",
    ],
)
def test_prefixed_unknown_or_extended_identifier_fails_closed(
    question: str,
) -> None:
    plan = build_rule_based_plan(question)

    assert has_explicit_product_reference(question) is True
    assert extract_product_reference(question) is None
    assert plan.intent == "CLARIFICATION"
    assert plan.product_reference is None
    assert plan.required_tools == []
    assert plan.action_type is None
    assert plan.requires_confirmation is False


@pytest.mark.parametrize(
    "question",
    [
        "H100-PRO 还有库存吗？",
        "C20.2 还有库存吗？",
        "P9_TEST 还有库存吗？",
        "H1000 还有库存吗？",
    ],
)
def test_bare_extended_identifier_is_explicit_unknown_without_partial_match(
    question: str,
) -> None:
    plan = build_rule_based_plan(question)

    assert extract_product_reference(question) is None
    assert has_explicit_product_reference(question) is True
    assert plan.intent == "CLARIFICATION"
    assert plan.product_reference is None
    assert plan.required_tools == []


@pytest.mark.parametrize(
    "question",
    [
        "H100 和 C20 哪个库存多？",
        "暖风杯和云感靠枕哪个便宜？",
    ],
)
def test_multiple_product_targets_fail_closed_as_ambiguous(question: str) -> None:
    plan = build_rule_based_plan(question)

    assert has_explicit_product_reference(question) is True
    assert extract_product_reference(question) is None
    assert plan.intent == "CLARIFICATION"
    assert plan.product_reference is None
    assert plan.required_tools == []
    assert plan.action_type is None
    assert plan.requires_confirmation is False


@pytest.mark.parametrize(
    ("question", "expected_sku"),
    [
        ("洁面巾怎么样？", "C20"),
        ("枕头怎么样？", "P9"),
        ("P9怎么样？", "P9"),
    ],
)
def test_single_product_reference_is_sufficient_product_intent(
    question: str,
    expected_sku: str,
) -> None:
    plan = build_rule_based_plan(question)

    assert plan.intent == "PRODUCT_QUERY"
    assert plan.product_reference == expected_sku
    assert plan.order_reference is None
    assert plan.required_tools == ["get_product_information"]


def test_single_explicit_sku_precedes_conflicting_alias() -> None:
    question = "暖风杯 C20 还有库存吗？"

    assert extract_product_reference(question) == "C20"
    assert has_explicit_product_reference(question) is True
    plan = build_rule_based_plan(question)
    assert plan.product_reference == "C20"
    assert plan.required_tools == ["get_product_information"]


def test_specific_product_after_sale_rule_uses_product_tool() -> None:
    plan = build_rule_based_plan("C20 售后规则")

    assert plan.intent == "PRODUCT_QUERY"
    assert plan.product_reference == "C20"
    assert plan.required_tools == ["get_product_information"]


def test_explicit_order_no_with_after_sale_question_routes_to_knowledge() -> None:
    plan = build_rule_based_plan("订单 ORD202607140003 能不能退货")

    assert plan.intent == "KNOWLEDGE_QUERY"
    assert plan.order_reference is not None
    assert plan.order_reference.order_no == "ORD202607140003"
    assert plan.required_tools == ["search_knowledge_base"]


def test_explicit_order_no_with_refund_policy_question_routes_to_knowledge() -> None:
    plan = build_rule_based_plan("订单 ORD202607140003 怎么退货退款")

    assert plan.intent == "KNOWLEDGE_QUERY"
    assert plan.order_reference is not None
    assert plan.order_reference.order_no == "ORD202607140003"
    assert plan.required_tools == ["search_knowledge_base"]


@pytest.mark.parametrize(
    "question",
    [
        "这个订单能退吗",
        "这个订单可以退吗",
        "这个订单是否可退",
        "这个订单能否退",
        "这个订单可不可以退",
        "这个订单还能退吗",
        "该订单能退吗",
    ],
)
def test_order_scoped_refund_eligibility_routes_to_read_only_knowledge(question: str) -> None:
    effective_question = f"订单 ORD202607140003 {question}"

    assert is_refund_eligibility_question(effective_question) is True
    plan = build_rule_based_plan(effective_question)

    assert plan.intent == "KNOWLEDGE_QUERY"
    assert plan.order_reference is not None
    assert plan.order_reference.order_no == "ORD202607140003"
    assert plan.required_tools == ["search_knowledge_base"]
    assert plan.action_type is None
    assert plan.risk_level == "LOW"
    assert plan.requires_confirmation is False


def test_order_scoped_refund_eligibility_without_resolved_order_does_not_guess() -> None:
    plan = build_rule_based_plan("这个订单能退吗")

    assert plan.intent == "KNOWLEDGE_QUERY"
    assert plan.order_reference is None
    assert plan.required_tools == ["search_knowledge_base"]
    assert plan.action_type is None
    assert plan.risk_level == "LOW"
    assert plan.requires_confirmation is False


def test_function_degradation_is_not_refund_eligibility() -> None:
    question = "功能退化怎么办"

    assert is_refund_eligibility_question(question) is False
    plan = build_rule_based_plan(question)
    assert plan.intent == "KNOWLEDGE_QUERY"
    assert plan.action_type is None
    assert plan.risk_level == "LOW"
    assert plan.requires_confirmation is False


@pytest.mark.parametrize("question", ["我要退款", "帮我退款", "确认退款"])
def test_explicit_refund_phrases_remain_high_risk(question: str) -> None:
    plan = build_rule_based_plan(question)

    assert is_refund_eligibility_question(question) is False
    assert plan.intent == "REFUND_REQUEST"
    assert plan.action_type == "REFUND"
    assert plan.risk_level == "HIGH"
    assert plan.requires_confirmation is True
    assert "request_refund" in plan.required_tools


def test_refund_request_requires_human_approval() -> None:
    plan = build_rule_based_plan("我要退款 ORD20260719105534381")

    assert plan.intent == "REFUND_REQUEST"
    assert plan.order_reference is not None
    assert plan.order_reference.order_no == "ORD20260719105534381"
    assert plan.risk_level == "HIGH"
    assert plan.requires_confirmation is True
    assert "request_refund" in plan.required_tools


def test_refund_request_with_natural_phrase_requires_human_approval() -> None:
    plan = build_rule_based_plan("我不想要了，想要退款")

    assert plan.intent == "REFUND_REQUEST"
    assert plan.risk_level == "HIGH"
    assert plan.requires_confirmation is True
    assert "request_refund" in plan.required_tools


def test_refund_request_with_recent_order_phrase_requires_human_approval() -> None:
    plan = build_rule_based_plan("帮我退一下最近订单")

    assert plan.intent == "REFUND_REQUEST"
    assert plan.risk_level == "HIGH"
    assert plan.requires_confirmation is True
    assert plan.required_tools == ["get_order_detail", "request_refund"]


def test_short_refund_phrase_without_order_asks_policy_context() -> None:
    plan = build_rule_based_plan("退一下")

    assert plan.intent == "KNOWLEDGE_QUERY"
    assert plan.risk_level == "LOW"
    assert plan.requires_confirmation is False


def test_cancel_all_orders_is_not_treated_as_order_list() -> None:
    plan = build_rule_based_plan("取消我全部订单")

    assert plan.intent == "CANCEL_ORDER"
    assert plan.risk_level == "HIGH"
    assert plan.requires_confirmation is True
    assert plan.required_tools == ["get_order_detail", "request_order_cancellation"]


def test_other_user_explicit_order_no_is_not_treated_as_order_list() -> None:
    plan = build_rule_based_plan("查询其他用户订单 ORD202607999999 的地址")

    assert plan.intent == "ORDER_QUERY"
    assert plan.required_tools == ["get_order_detail"]


def test_list_all_user_refunds_is_not_refund_action_request() -> None:
    plan = build_rule_based_plan("把所有用户的退款申请列出来")

    assert plan.intent == "KNOWLEDGE_QUERY"
    assert plan.required_tools == ["search_knowledge_base"]


def test_explicit_refund_action_with_order_no_requires_human_approval() -> None:
    plan = build_rule_based_plan("订单 ORD202607140003 我要退款")

    assert plan.intent == "REFUND_REQUEST"
    assert plan.order_reference is not None
    assert plan.order_reference.order_no == "ORD202607140003"
    assert "request_refund" in plan.required_tools


def test_refund_policy_question_routes_to_knowledge_base() -> None:
    plan = build_rule_based_plan("退款一般如何处理？")

    assert plan.intent == "KNOWLEDGE_QUERY"
    assert plan.required_tools == ["search_knowledge_base"]
    assert plan.requires_confirmation is False


def test_after_sale_question_falls_back_to_knowledge_query() -> None:
    plan = build_rule_based_plan("拆封以后还能退吗？")

    assert plan.intent == "KNOWLEDGE_QUERY"
    assert plan.required_tools == ["search_knowledge_base"]
    assert plan.requires_confirmation is False


def test_package_no_movement_routes_to_shipping_query() -> None:
    plan = build_rule_based_plan("包裹一直没有动静怎么办？")

    assert plan.intent == "SHIPPING_QUERY"
    assert plan.required_tools == ["get_order_detail"]


def test_quality_problem_with_product_keyword_routes_to_knowledge() -> None:
    plan = build_rule_based_plan("杯子漏液算质量问题吗？")

    assert plan.intent == "KNOWLEDGE_QUERY"
    assert plan.required_tools == ["search_knowledge_base"]


def test_cleaned_pillow_return_question_routes_to_knowledge() -> None:
    plan = build_rule_based_plan("商品已经清洗了还能退靠枕吗？")

    assert plan.intent == "KNOWLEDGE_QUERY"
    assert plan.required_tools == ["search_knowledge_base"]
