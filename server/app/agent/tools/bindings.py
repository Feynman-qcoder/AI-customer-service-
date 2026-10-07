from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from app.agent.tools.registry import TOOL_REGISTRY, ToolDefinition


class ProductionToolName(StrEnum):
    LIST_MY_ORDERS = "list_my_orders"
    GET_ORDER_DETAIL = "get_order_detail"
    GET_PRODUCT_INFORMATION = "get_product_information"
    SEARCH_KNOWLEDGE_BASE = "search_knowledge_base"
    REQUEST_ORDER_CANCELLATION = "request_order_cancellation"
    REQUEST_REFUND = "request_refund"


@dataclass(frozen=True)
class ProductionToolBinding:
    """A canonical production call-site binding to one registered definition."""

    name: ProductionToolName
    definition: ToolDefinition


def _build_bindings() -> dict[ProductionToolName, ProductionToolBinding]:
    bindings: dict[ProductionToolName, ProductionToolBinding] = {}
    for name in ProductionToolName:
        definition = TOOL_REGISTRY.get(name.value)
        if definition is None or definition.policy.name != name.value:
            raise RuntimeError(f"production tool binding is not registered: {name.value}")
        bindings[name] = ProductionToolBinding(name=name, definition=definition)
    return bindings


PRODUCTION_TOOL_BINDINGS = _build_bindings()


def planner_executable_tool_names() -> frozenset[str]:
    return frozenset(name.value for name in PRODUCTION_TOOL_BINDINGS)


def registered_only_tool_names() -> frozenset[str]:
    return frozenset(TOOL_REGISTRY).difference(planner_executable_tool_names())


def production_tool_binding(name: str | ProductionToolName) -> ProductionToolBinding | None:
    try:
        normalized = name if isinstance(name, ProductionToolName) else ProductionToolName(name)
    except ValueError:
        return None
    return PRODUCTION_TOOL_BINDINGS[normalized]


def require_canonical_production_binding(binding: ProductionToolBinding) -> ToolDefinition:
    canonical = PRODUCTION_TOOL_BINDINGS.get(binding.name)
    if canonical is not binding or canonical.definition is not TOOL_REGISTRY.get(binding.name.value):
        raise ValueError("non-canonical production tool binding")
    return canonical.definition
