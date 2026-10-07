import pytest

from app.services.action_execution_service import ActionExecutionError, resolve_order_action_transition


def test_cancel_before_shipping_restores_stock() -> None:
    transition = resolve_order_action_transition("ORDER_CANCELLATION", "WAITING_SHIPMENT")

    assert transition.next_status == "CANCELLED"
    assert transition.restore_stock is True


def test_cancel_after_shipping_is_rejected() -> None:
    with pytest.raises(ActionExecutionError):
        resolve_order_action_transition("ORDER_CANCELLATION", "IN_TRANSIT")


def test_refund_signed_order_enters_after_sale_review() -> None:
    transition = resolve_order_action_transition("REFUND", "SIGNED")

    assert transition.next_status == "AFTER_SALE_REVIEWING"
    assert transition.restore_stock is False
