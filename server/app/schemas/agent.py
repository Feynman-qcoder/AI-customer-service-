from typing import Literal

from pydantic import BaseModel, Field

IntentType = Literal[
    "ORDER_QUERY",
    "SHIPPING_QUERY",
    "PRODUCT_QUERY",
    "KNOWLEDGE_QUERY",
    "CANCEL_ORDER",
    "REFUND_REQUEST",
    "CREATE_TICKET",
    "CLARIFICATION",
]
RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "FORBIDDEN"]


class OrderReference(BaseModel):
    order_no: str | None = None
    ordinal_index: int | None = Field(default=None, ge=0)
    product_keyword: str | None = None
    latest: bool = False
    list_all: bool = False


class AgentPlan(BaseModel):
    intent: IntentType
    goal: str
    order_reference: OrderReference | None = None
    product_reference: str | None = None
    required_tools: list[str]
    action_type: str | None = None
    risk_level: RiskLevel
    requires_confirmation: bool
    missing_information: list[str]
    decision_reason: str
