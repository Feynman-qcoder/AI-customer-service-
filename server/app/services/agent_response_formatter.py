from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal
from typing import Protocol

from app.agent.routing import is_refund_eligibility_question
from app.core.timezone import format_asia_shanghai
from app.schemas.chat import ChatResponse, SourceReference
from app.schemas.retrieval import RetrievalCandidate


class ProductView(Protocol):
    @property
    def id(self) -> int: ...

    @property
    def product_code(self) -> str: ...

    @property
    def product_name(self) -> str: ...

    @property
    def category(self) -> str: ...

    @property
    def sale_status(self) -> str: ...

    @property
    def price(self) -> Decimal: ...

    @property
    def stock_quantity(self) -> int: ...

    @property
    def dispatch_rule(self) -> str: ...

    @property
    def after_sale_rule(self) -> str: ...


class OrderView(Protocol):
    @property
    def id(self) -> int: ...

    @property
    def order_no(self) -> str: ...

    @property
    def user_id(self) -> int: ...

    @property
    def product_id(self) -> int: ...

    @property
    def quantity(self) -> int: ...

    @property
    def amount(self) -> Decimal: ...

    @property
    def status(self) -> str: ...

    @property
    def paid_at(self) -> datetime | None: ...

    @property
    def expected_ship_at(self) -> datetime | None: ...

    @property
    def signed_at(self) -> datetime | None: ...

    @property
    def created_at(self) -> datetime: ...

    @property
    def product(self) -> ProductView: ...


class AgentResponseFormatter:
    def _order_answer(self, order: OrderView, question: str) -> str:
        lead = f"我查到订单 {order.order_no} 是「{order.product.product_name}」。"
        if self._is_shipping_rule_question(question):
            return (
                f"{lead}这个商品的发货规则是：{order.product.dispatch_rule}"
                f"这单当前状态：{self._order_status_label(order.status)}，"
                f"预计发货时间：{self._format_time(order.expected_ship_at)}。"
            )
        if order.status in {"PAID", "WAITING_SHIPMENT"}:
            if any(word in question for word in ["物流", "快递", "到哪", "到哪里"]):
                return f"{lead}这单还没有进入物流运输，预计发货时间是 {self._format_time(order.expected_ship_at)}。"
            return f"{lead}当前还未发货，预计发货时间是 {self._format_time(order.expected_ship_at)}。"
        if order.status in {"SHIPPED", "IN_TRANSIT"}:
            return f"{lead}这单已发货，当前状态是 {order.status}。"
        if order.status == "SIGNED":
            return f"{lead}这单已签收。如需售后，可以继续描述商品问题。"
        return f"{lead}当前状态是 {self._order_status_label(order.status)}。"

    def _product_answer(self, product: ProductView, include_name: bool = True) -> str:
        prefix = f"「{product.product_name}」" if include_name else ""
        return (
            f"{prefix}商品编码 {product.product_code}，分类：{product.category}，"
            f"当前状态：{self._product_status_label(product.sale_status)}，库存 {product.stock_quantity} 件，"
            f"售价 {product.price} 元。"
            f"发货规则：{self._clean_sentence(product.dispatch_rule)}。"
            f"售后规则：{self._clean_sentence(product.after_sale_rule)}。"
        )

    def _action_request_answer(self, order: OrderView, action_type: str) -> str:
        if action_type == "REFUND":
            if order.status in {"PAID", "WAITING_SHIPMENT"}:
                detail = "这单还没有发货，客服审核通过后会按原支付路径处理。"
            else:
                detail = "客服会结合商品状态、物流和凭证进行售后审核。"
            return (
                f"已为您提交退款申请，订单号 {order.order_no}，商品是「{order.product.product_name}」。"
                f"{detail}您可以在当前页面继续补充原因或凭证，后台客服会尽快处理。"
            )
        return (
            f"已为您登记取消订单申请，订单号 {order.order_no}，商品是「{order.product.product_name}」。"
            "客服审核通过后会更新订单状态；如果订单已经进入发货流程，可能需要转为售后处理。"
        )

    def _order_list_answer(self, orders: Sequence[OrderView]) -> str:
        lines = ["我查到您已下单的商品如下，按下单时间从近到远排列："]
        for index, order in enumerate(orders, start=1):
            lines.append(
                f"{index}. 「{order.product.product_name}」x {order.quantity}，"
                f"订单号 {order.order_no}，状态：{self._order_status_label(order.status)}，"
                f"预计发货：{self._format_time(order.expected_ship_at)}。"
            )
        lines.append("您可以继续问“第几个订单物流到哪里了”，也可以直接按商品名或订单号查询。")
        return "\n".join(lines)

    def _multiple_order_answer(self, orders: Sequence[OrderView]) -> str:
        lines = ["我查到这个商品有多笔订单，先不替您默认选某一单："]
        for index, order in enumerate(orders, start=1):
            lines.append(
                f"{index}. 订单 {order.order_no}，商品「{order.product.product_name}」，"
                f"状态：{self._order_status_label(order.status)}，预计发货：{self._format_time(order.expected_ship_at)}，"
                "暂无物流。"
            )
        lines.append("请直接发订单号，或说“第几个订单”，我再按那一单查询。")
        return "\n".join(lines)

    def _multiple_action_target_answer(self, orders: Sequence[OrderView]) -> str:
        lines = ["我找到多笔订单，暂不替您选择："]
        for index, order in enumerate(orders, start=1):
            lines.append(f"{index}. 订单 {order.order_no}，商品「{order.product.product_name}」")
        lines.append("请提供要处理的完整订单号后重新提交动作请求。")
        return "\n".join(lines)

    def _source_references(
        self,
        candidates: list[RetrievalCandidate],
        question: str = "",
        order: OrderView | None = None,
    ) -> list[SourceReference]:
        self._include_order_refund_eligibility_evidence(candidates, question, order)
        sources: list[SourceReference] = []
        logistics_question = self._is_logistics_timing_question(question)
        for candidate in candidates:
            if candidate.source_type == "structured_rule":
                if logistics_question:
                    continue
                sources.append(
                    SourceReference(
                        documentId=0,
                        fileName=f"售后规则：{candidate.metadata.get('rule_title', '结构化规则')}",
                        snippet=candidate.content[:260],
                        score=candidate.rerank_score or candidate.fused_score or candidate.original_score,
                    )
                )
                continue
            if candidate.document_id is None:
                continue
            file_name = str(candidate.metadata.get("file_name", "knowledge"))
            if logistics_question and "发货与物流规则" not in file_name:
                continue
            if not self._source_matches_question(file_name, question, order):
                continue
            sources.append(
                SourceReference(
                    documentId=int(candidate.document_id),
                    fileName=file_name,
                    snippet=candidate.content[:260],
                    score=candidate.rerank_score or candidate.fused_score or candidate.original_score,
                )
            )
        return sources

    def _include_order_refund_eligibility_evidence(
        self,
        candidates: list[RetrievalCandidate],
        question: str,
        order: OrderView | None,
    ) -> None:
        if candidates or order is None or not is_refund_eligibility_question(question):
            return
        product_rule = self._clean_sentence(order.product.after_sale_rule)
        if not product_rule:
            return
        candidates.append(
            RetrievalCandidate(
                candidate_id=f"order-product-policy:{order.product.id}",
                source_type="structured_rule",
                content=product_rule,
                document_id=None,
                chunk_id=None,
                rule_id=f"product-catalog:{order.product.id}",
                metadata={
                    "rule_title": f"{order.product.product_name} 商品售后规则",
                    "evidence_origin": "OWNER_SCOPED_PRODUCT_CATALOG",
                },
                original_score=0.0,
                decision_reason="使用 owner-scoped 订单关联商品的真实售后规则作为低置信度受控证据",
            )
        )

    def _knowledge_answer(
        self,
        candidates: list[RetrievalCandidate],
        order: OrderView | None = None,
        question: str = "",
    ) -> str:
        structured = [candidate for candidate in candidates if candidate.source_type == "structured_rule"]
        if order is not None and self._is_after_sale_rule_question(question):
            return self._customer_after_sale_answer(order, question, structured[0].content if structured else None)
        if self._is_logistics_timing_question(question) or self._is_after_sale_rule_question(question):
            return self._customer_knowledge_answer(candidates, question)
        if structured:
            return structured[0].content
        return self._customer_knowledge_answer(candidates, question)

    def _customer_after_sale_answer(
        self,
        order: OrderView,
        question: str,
        structured_content: str | None = None,
    ) -> str:
        product_name = order.product.product_name
        product_rule = self._clean_sentence(order.product.after_sale_rule)
        normalized = question.strip()
        if any(term in normalized for term in ["破损", "损坏", "包装", "坏了", "质量", "裂", "漏"]):
            return (
                f"您这单是「{product_name}」。如果收到商品破损，请先保留商品、外包装和快递面单，"
                "再拍摄清晰照片或视频作为凭证。"
                f"{product_rule}。如果破损影响使用，可以提交售后申请；"
                "凭证不足、责任不清或包装损坏比较严重时，建议转人工复核。"
            )
        if is_refund_eligibility_question(normalized) or any(
            term in normalized for term in ["退款", "退钱", "钱退", "到账"]
        ):
            status_label = self._order_status_label(order.status)
            if order.status in {"PAID", "WAITING_SHIPMENT"}:
                return (
                    f"您这单是「{product_name}」，当前状态为{status_label}，尚未发货，可以发起退款申请。"
                    "退款申请需要明确确认后才会提交，并且需要审核；本次询问不会创建退款申请，"
                    "只是在判断是否可申请。"
                )
            if order.status in {"SHIPPED", "IN_TRANSIT", "SIGNED"}:
                return (
                    f"您这单是「{product_name}」，当前状态为{status_label}，已经进入发货或签收阶段。"
                    "是否可以退款需要结合拒收或退货进度、商品状态和售后凭证审核判断；"
                    "审核通过后通常按原支付路径退款。如需正式申请，仍要由您明确确认。"
                    "本次询问不会创建退款申请。"
                )
            return (
                f"您这单是「{product_name}」，当前状态为{status_label}。如果您想退款，"
                "还需要结合退货原因和商品情况判断。"
                f"{product_rule}。审核通过后通常会按原支付路径退款；"
                "正式申请需要您明确确认并经过审核；本次询问不会创建退款申请。"
            )
        if "换货" in normalized:
            return (
                f"您这单是「{product_name}」。如果是质量问题、收到破损或无法正常使用，可以提交换货或补发售后申请。"
                f"建议先准备问题照片或视频，方便客服审核。{product_rule}。"
            )
        if any(term in normalized for term in ["拆封", "打开", "开封"]):
            product_answer = self._product_unboxing_answer(order.product.product_code, product_name)
            if product_answer is not None:
                return product_answer
            return (
                f"您这单是「{product_name}」。拆封不一定等于不能退，关键要看是否影响二次销售。"
                f"{product_rule}。如果商品未明显使用、配件齐全且包装没有严重损坏，可以提交退货申请；"
                "如果已经清洗、明显使用或包装严重破损，建议转人工复核。"
            )
        if structured_content:
            customer_rule = self._customer_policy_summary(structured_content, normalized)
            if customer_rule:
                return f"您这单是「{product_name}」。{product_rule}。{customer_rule}"
        return (
            f"您这单是「{product_name}」。{product_rule}。"
            "如果您要退货或退款，可以继续补充商品是否拆封、是否使用、包装是否完整，"
            "我会先帮您判断是否适合直接提交售后申请。"
        )

    def _customer_knowledge_answer(self, candidates: list[RetrievalCandidate], question: str) -> str:
        normalized = question.strip()
        if self._is_logistics_timing_question(normalized):
            logistics_candidate = next(
                (
                    candidate
                    for candidate in candidates
                    if "发货与物流规则" in str(candidate.metadata.get("file_name", ""))
                ),
                None,
            )
            if logistics_candidate is not None:
                return (
                    "发货后，普通地区通常 2 到 5 天送达；偏远地区、节假日或大促期间可能延迟，"
                    "因此偏远地区没有统一的精确天数承诺，具体请以订单页和承运商物流更新为准。"
                )
            return "当前检索结果没有找到可核实的物流时效规则，建议提供订单号后再查询具体进度。"
        if any(term in normalized for term in ["退货", "拆封", "换货", "怎么退", "能不能退"]):
            product_code = self._question_product_code(normalized)
            if product_code is not None and self._has_product_policy_candidate(candidates, product_code):
                product_answer = self._product_unboxing_answer(product_code)
                if product_answer is not None:
                    return product_answer
        if any(term in normalized for term in ["破损", "损坏", "包装", "坏了", "质量", "裂", "漏"]):
            return (
                "收到商品破损时，请先保留商品、外包装和快递面单，并拍摄清晰照片或视频。"
                "如果破损影响使用，可以提交售后申请；凭证不足或情况复杂时，建议转人工复核。"
            )
        if any(term in normalized for term in ["退款", "退钱", "钱退", "到账"]):
            return (
                "退款通常会按原支付路径退回。未发货订单一般先提交退款申请，审核通过后进入退款处理；"
                "已发货或已签收订单通常需要先完成拒收、退货或售后审核，再处理退款。"
            )
        if any(term in normalized for term in ["退货", "拆封", "换货", "怎么退", "能不能退"]):
            if any(term in normalized for term in ["拆封", "打开", "开封"]):
                return (
                    "拆封后能否退货需要按品类区分：普通非特殊商品仅拆开外包装、未使用、配件齐全且"
                    "包装完整时，可以提交退货申请；小家电和带电产品可以合理开箱检查，但明显使用、"
                    "有划痕或缺少配件时需要人工确认；个护、食品或耗材拆封后通常不支持无理由退货，"
                    "如有质量问题可以提供售后凭证申请处理。"
                )
            return (
                "退货主要看商品是否影响二次销售。商品未明显使用、配件齐全、包装没有严重损坏时，"
                "通常可以提交退货申请；质量问题或收到破损时，可以按售后流程申请换货、补发或人工复核。"
            )
        return self._knowledge_excerpt(self._best_customer_candidate(candidates, question).content)

    def _customer_policy_summary(self, content: str, question: str) -> str:
        if any(term in question for term in ["退款", "退钱", "钱退", "到账"]):
            return "审核通过后通常会按原支付路径退款；如果订单已发货或已签收，一般需要先完成拒收、退货或售后审核。"
        if any(term in question for term in ["破损", "损坏", "包装", "坏了", "质量"]):
            return "如果商品破损影响使用，请准备照片或视频凭证后提交售后申请。"
        if any(term in question for term in ["退货", "拆封", "换货", "怎么退", "能不能退"]):
            return "退货时主要核对商品是否明显使用、配件是否齐全，以及包装是否严重损坏。"
        return self._knowledge_excerpt(content, max_chars=120)

    def _best_customer_candidate(
        self,
        candidates: list[RetrievalCandidate],
        question: str,
    ) -> RetrievalCandidate:
        preferred: list[str] = []
        if self._is_logistics_timing_question(question):
            preferred = ["发货与物流规则"]
        elif any(term in question for term in ["破损", "损坏", "包装", "坏了", "质量"]):
            preferred = ["商品损坏", "售后与退换货", "退换货政策"]
        elif any(term in question for term in ["退货", "拆封", "换货"]):
            preferred = ["退换货政策", "售后与退换货", "商品损坏"]
        elif any(term in question for term in ["退款", "退钱", "到账"]):
            preferred = ["退款处理"]
        for keyword in preferred:
            for candidate in candidates:
                file_name = str(candidate.metadata.get("file_name", ""))
                if keyword in file_name or keyword in candidate.content[:80]:
                    return candidate
        return candidates[0]

    def _source_matches_question(
        self,
        file_name: str,
        question: str,
        order: OrderView | None = None,
    ) -> bool:
        if self._is_logistics_timing_question(question):
            if file_name.startswith("商品资料-"):
                product_code = self._question_product_code(question)
                return product_code is not None and self._product_source_matches(file_name, product_code)
            return "发货与物流规则" in file_name
        if self._is_unboxing_return_question(question) and "退款处理" in file_name:
            return False
        if not file_name.startswith("商品资料-"):
            return True
        if order is not None:
            file_product_code = self._product_code_from_source_name(file_name)
            if file_product_code is not None:
                return file_product_code == order.product.product_code
            return order.product.product_code in file_name or order.product.product_name.replace(
                " ", ""
            ) in file_name.replace(" ", "")
        product_code = self._question_product_code(question)
        return product_code is not None and self._product_source_matches(file_name, product_code)

    def _knowledge_excerpt(self, content: str, max_chars: int = 260) -> str:
        lines: list[str] = []
        for line in content.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            clean = line.strip().lstrip("#").strip()
            if not clean or clean.startswith("本文件用于演示") or "不建议客服承诺" in clean or "演示规则" in clean:
                continue
            lines.append(clean)
        return " ".join(lines)[:max_chars]

    def _plain_response(
        self,
        conversation_id: int,
        answer: str,
        confidence_level: str = "HIGH",
        need_human: bool = False,
        sources: list[SourceReference] | None = None,
    ) -> ChatResponse:
        return ChatResponse(
            conversationId=conversation_id,
            answer=answer,
            sources=sources or [],
            retrievalScore=0,
            confidenceLevel=confidence_level,
            needHuman=need_human,
        )

    def _order_status_label(self, status: str) -> str:
        return {
            "PENDING_PAYMENT": "待付款",
            "PAID": "已付款",
            "WAITING_SHIPMENT": "待发货",
            "SHIPPED": "已发货",
            "IN_TRANSIT": "运输中",
            "SIGNED": "已签收",
            "REFUNDING": "退款中",
            "REFUNDED": "已退款",
            "CANCELLED": "已取消",
        }.get(status, status)

    def _product_status_label(self, status: str) -> str:
        return {"ON_SALE": "在售", "OUT_OF_STOCK": "缺货", "OFF_SHELF": "下架"}.get(status, status)

    def _product_code_from_source_name(self, file_name: str) -> str | None:
        upper = file_name.upper()
        for code in ["H100", "C20", "P9"]:
            if code in upper:
                return code
        return None

    def _is_after_sale_rule_question(self, question: str) -> bool:
        if is_refund_eligibility_question(question):
            return True
        after_sale_terms = [
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
        ]
        return any(
            term in question
            for term in after_sale_terms
        )

    def _is_shipping_rule_question(self, question: str) -> bool:
        return any(term in question for term in ["发货规则", "发货时效", "出库规则", "多久发货", "什么时候发货"])

    def _is_logistics_timing_question(self, question: str) -> bool:
        return any(term in question for term in ["物流时效", "偏远地区", "配送", "送达"])

    def _is_unboxing_return_question(self, question: str) -> bool:
        return any(term in question for term in ["拆封", "打开", "开封"])

    def _question_product_code(self, question: str) -> str | None:
        upper_question = question.upper()
        explicit_codes = [code for code in ["H100", "C20", "P9"] if code in upper_question]
        if len(explicit_codes) == 1:
            return explicit_codes[0]
        if explicit_codes:
            return None
        if any(term in question for term in ["洗脸巾", "洗面巾", "洁面巾"]):
            return "C20"
        if "暖风杯" in question:
            return "H100"
        if any(term in question for term in ["靠枕", "枕头"]):
            return "P9"
        return None

    def _product_source_matches(self, file_name: str, product_code: str) -> bool:
        source_code = self._product_code_from_source_name(file_name)
        return source_code == product_code

    def _has_product_policy_candidate(
        self,
        candidates: list[RetrievalCandidate],
        product_code: str,
    ) -> bool:
        return any(
            self._product_source_matches(str(candidate.metadata.get("file_name", "")), product_code)
            for candidate in candidates
            if candidate.document_id is not None
        )

    def _product_unboxing_answer(self, product_code: str, product_name: str | None = None) -> str | None:
        display_name = product_name or {
            "C20": "轻氧洗面巾 C20",
            "H100": "暖风杯 H100",
            "P9": "云感靠枕 P9",
        }.get(product_code, product_code)
        if product_code == "C20":
            return (
                f"「{display_name}」属于个护耗材，拆封后通常不支持无理由退货。"
                "如果存在质量问题，可以提供商品照片和订单号申请售后处理。"
            )
        if product_code == "H100":
            return (
                f"「{display_name}」可以合理开箱检查。仅拆封且未长时间通电使用、无明显划痕、"
                "配件齐全、包装完整时，可以提交退货申请；已经明显使用或缺少配件时需要人工确认。"
            )
        if product_code == "P9":
            return (
                f"「{display_name}」支持开箱检查。未清洗、未明显使用、无污渍、无异味、"
                "吊牌和包装完整时，可以提交退货申请。"
            )
        return None

    def _clean_sentence(self, value: str) -> str:
        return value.strip().rstrip("。.!！")

    def _format_time(self, value: datetime | None) -> str:
        return format_asia_shanghai(value)
