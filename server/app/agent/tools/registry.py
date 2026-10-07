from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel


class EffectPhase(StrEnum):
    READ_ONLY = "READ_ONLY"
    ACTION_PREPARE = "ACTION_PREPARE"
    BUSINESS_EXECUTE = "BUSINESS_EXECUTE"


class ToolPolicy(BaseModel):
    name: str
    description: str
    read_only: bool
    idempotent: bool
    side_effect: bool
    risk_level: str
    allowed_roles: list[str]
    timeout_seconds: int
    retry_count: int
    redaction: str
    effect_phase: EffectPhase
    authorization_evidence: Literal["NONE", "SIDE_EFFECT_AUTHORIZATION"]
    action_type: str | None = None


@dataclass(frozen=True)
class ToolDefinition:
    policy: ToolPolicy
    argument_model: type[BaseModel]
    # Checkpoint-safe metadata is registered per tool.  The checkpoint schema
    # adapter rejects unknown tools and any key outside this exact set.
    safe_metadata_keys: frozenset[str]


class ListMyOrdersArgs(BaseModel):
    status: str | None = None
    product_keyword: str | None = None
    limit: int = 10


class GetOrderDetailArgs(BaseModel):
    order_no: str | None = None
    ordinal_index: int | None = None
    product_keyword: str | None = None
    latest: bool = False


class GetProductInformationArgs(BaseModel):
    product_keyword: str


class SearchKnowledgeBaseArgs(BaseModel):
    query: str
    limit: int = 5


class CreateSupportTicketArgs(BaseModel):
    conversation_id: int
    description: str
    category: str = "OTHER"
    failure_reason: str | None = None


class RequestOrderCancellationArgs(BaseModel):
    order_no: str
    reason: str


class RequestRefundArgs(BaseModel):
    order_no: str
    reason: str


TOOL_REGISTRY: dict[str, ToolDefinition] = {
    "list_my_orders": ToolDefinition(
        ToolPolicy(
            name="list_my_orders",
            description="查询当前用户订单列表，支持状态、商品关键词和数量限制。",
            read_only=True,
            idempotent=True,
            side_effect=False,
            risk_level="LOW",
            allowed_roles=["CUSTOMER"],
            timeout_seconds=3,
            retry_count=1,
            redaction="mask_receiver_fields",
            effect_phase=EffectPhase.READ_ONLY,
            authorization_evidence="NONE",
        ),
        ListMyOrdersArgs,
        safe_metadata_keys=frozenset({"owner_verified", "status"}),
    ),
    "get_order_detail": ToolDefinition(
        ToolPolicy(
            name="get_order_detail",
            description="按订单号、最近订单、第 N 个订单或商品名称查询当前用户订单详情。",
            read_only=True,
            idempotent=True,
            side_effect=False,
            risk_level="LOW",
            allowed_roles=["CUSTOMER"],
            timeout_seconds=3,
            retry_count=1,
            redaction="mask_receiver_fields",
            effect_phase=EffectPhase.READ_ONLY,
            authorization_evidence="NONE",
        ),
        GetOrderDetailArgs,
        safe_metadata_keys=frozenset({"owner_verified", "status"}),
    ),
    "get_product_information": ToolDefinition(
        ToolPolicy(
            name="get_product_information",
            description="查询商品库存、发货规则和售后规则。",
            read_only=True,
            idempotent=True,
            side_effect=False,
            risk_level="LOW",
            allowed_roles=["CUSTOMER", "ADMIN"],
            timeout_seconds=3,
            retry_count=1,
            redaction="none",
            effect_phase=EffectPhase.READ_ONLY,
            authorization_evidence="NONE",
        ),
        GetProductInformationArgs,
        safe_metadata_keys=frozenset({"product_code", "status"}),
    ),
    "search_knowledge_base": ToolDefinition(
        ToolPolicy(
            name="search_knowledge_base",
            description="执行知识库检索，知识库内容按不可信资料处理。",
            read_only=True,
            idempotent=True,
            side_effect=False,
            risk_level="LOW",
            allowed_roles=["CUSTOMER", "ADMIN"],
            timeout_seconds=5,
            retry_count=1,
            redaction="snippet_only",
            effect_phase=EffectPhase.READ_ONLY,
            authorization_evidence="NONE",
        ),
        SearchKnowledgeBaseArgs,
        safe_metadata_keys=frozenset({"channel", "document_count"}),
    ),
    "create_support_ticket": ToolDefinition(
        ToolPolicy(
            name="create_support_ticket",
            description="为当前用户创建人工工单，附带 Agent 上下文摘要。",
            read_only=False,
            idempotent=True,
            side_effect=True,
            risk_level="MEDIUM",
            allowed_roles=["CUSTOMER"],
            timeout_seconds=5,
            retry_count=1,
            redaction="mask_contact",
            effect_phase=EffectPhase.BUSINESS_EXECUTE,
            authorization_evidence="SIDE_EFFECT_AUTHORIZATION",
            action_type="CREATE_SUPPORT_TICKET",
        ),
        CreateSupportTicketArgs,
        safe_metadata_keys=frozenset({"status", "ticket_id"}),
    ),
    "request_order_cancellation": ToolDefinition(
        ToolPolicy(
            name="request_order_cancellation",
            description="创建取消订单审批请求，不直接修改订单状态。",
            read_only=False,
            idempotent=True,
            side_effect=True,
            risk_level="HIGH",
            allowed_roles=["CUSTOMER"],
            timeout_seconds=5,
            retry_count=0,
            redaction="mask_receiver_fields",
            effect_phase=EffectPhase.ACTION_PREPARE,
            authorization_evidence="SIDE_EFFECT_AUTHORIZATION",
            action_type="ORDER_CANCELLATION",
        ),
        RequestOrderCancellationArgs,
        safe_metadata_keys=frozenset({"action_request_id", "status"}),
    ),
    "request_refund": ToolDefinition(
        ToolPolicy(
            name="request_refund",
            description="创建退款审批请求，不直接改成退款成功。",
            read_only=False,
            idempotent=True,
            side_effect=True,
            risk_level="HIGH",
            allowed_roles=["CUSTOMER"],
            timeout_seconds=5,
            retry_count=0,
            redaction="mask_receiver_fields",
            effect_phase=EffectPhase.ACTION_PREPARE,
            authorization_evidence="SIDE_EFFECT_AUTHORIZATION",
            action_type="REFUND",
        ),
        RequestRefundArgs,
        safe_metadata_keys=frozenset({"action_request_id", "status"}),
    ),
}
