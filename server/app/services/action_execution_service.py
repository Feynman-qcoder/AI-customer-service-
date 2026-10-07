from dataclasses import dataclass


class ActionExecutionError(ValueError):
    pass


@dataclass(frozen=True)
class OrderActionTransition:
    next_status: str
    summary: str
    restore_stock: bool = False


def resolve_order_action_transition(action_type: str, current_status: str) -> OrderActionTransition:
    if action_type == "ORDER_CANCELLATION":
        return _resolve_cancellation(current_status)
    if action_type == "REFUND":
        return _resolve_refund(current_status)
    raise ActionExecutionError(f"不支持的审批动作类型：{action_type}")


def _resolve_cancellation(current_status: str) -> OrderActionTransition:
    if current_status in {"PAID", "WAITING_SHIPMENT"}:
        return OrderActionTransition(
            next_status="CANCELLED",
            summary="订单未发货，审批通过后已取消订单并回补库存。",
            restore_stock=True,
        )
    if current_status in {"CANCELLED", "REFUND_PENDING", "REFUNDED"}:
        raise ActionExecutionError("订单已经取消或进入退款流程，不能重复取消。")
    raise ActionExecutionError("订单已进入发货或签收流程，不能直接取消，请改走退款/售后流程。")


def _resolve_refund(current_status: str) -> OrderActionTransition:
    if current_status in {"PAID", "WAITING_SHIPMENT"}:
        return OrderActionTransition(
            next_status="REFUND_PENDING",
            summary="订单未发货，审批通过后已进入退款处理流程。",
        )
    if current_status in {"SHIPPED", "IN_TRANSIT", "SIGNED"}:
        return OrderActionTransition(
            next_status="AFTER_SALE_REVIEWING",
            summary="订单已发货或签收，审批通过后已进入售后退款审核流程。",
        )
    if current_status in {"CANCELLED", "REFUND_PENDING", "REFUNDED", "AFTER_SALE_REVIEWING"}:
        raise ActionExecutionError("订单已经取消、退款或进入售后流程，不能重复发起退款。")
    raise ActionExecutionError(f"当前订单状态 {current_status} 暂不支持退款动作。")
