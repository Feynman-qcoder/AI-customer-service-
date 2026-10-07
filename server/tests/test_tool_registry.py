import pytest

from app.agent.tools.executor import ToolExecutionError, tool_executor
from app.agent.tools.registry import TOOL_REGISTRY, GetOrderDetailArgs, RequestRefundArgs
from app.core.security import AuthenticatedUser


class RecordingAudit:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    async def record(
        self,
        *,
        run_id: str,
        subject_user_id: int,
        tool_name: str,
        redacted_arguments: dict[str, object],
        result_summary: str,
        success: bool,
        retry_count: int,
        duration_ms: int,
    ) -> None:
        self.records.append(
            {
                "run_id": run_id,
                "subject_user_id": subject_user_id,
                "tool_name": tool_name,
                "redacted_arguments": redacted_arguments,
                "result_summary": result_summary,
                "success": success,
                "retry_count": retry_count,
                "duration_ms": duration_ms,
            }
        )


def test_read_only_order_lookup_policy_is_low_risk() -> None:
    definition = TOOL_REGISTRY["get_order_detail"]

    assert definition.argument_model is GetOrderDetailArgs
    assert definition.policy.read_only is True
    assert definition.policy.side_effect is False
    assert definition.policy.risk_level == "LOW"
    assert definition.policy.allowed_roles == ["CUSTOMER"]


def test_refund_request_policy_requires_side_effect_tracking() -> None:
    definition = TOOL_REGISTRY["request_refund"]

    assert definition.argument_model is RequestRefundArgs
    assert definition.policy.read_only is False
    assert definition.policy.side_effect is True
    assert definition.policy.risk_level == "HIGH"
    assert definition.policy.retry_count == 0


@pytest.mark.asyncio
async def test_tool_executor_validates_arguments_and_records_success() -> None:
    audit = RecordingAudit()
    user = AuthenticatedUser(user_id=1, username="user", name="用户", role="CUSTOMER")

    result = await tool_executor.execute(
        "run_1",
        user,
        "get_order_detail",
        {"order_no": "ORD202607140003"},
        lambda args: {"order_no": args.order_no},
        lambda value: f"resolved {value['order_no']}",
        audit,
    )

    assert result == {"order_no": "ORD202607140003"}
    assert len(audit.records) == 1
    assert audit.records[0]["success"] is True


@pytest.mark.asyncio
async def test_tool_executor_rejects_disallowed_role() -> None:
    audit = RecordingAudit()
    admin = AuthenticatedUser(user_id=2, username="admin", name="管理员", role="ADMIN")

    with pytest.raises(ToolExecutionError):
        await tool_executor.execute(
            "run_1",
            admin,
            "get_order_detail",
            {"order_no": "ORD202607140003"},
            lambda args: args,
            lambda _value: "should not run",
            audit,
        )


@pytest.mark.asyncio
async def test_tool_executor_records_validation_failure() -> None:
    audit = RecordingAudit()
    user = AuthenticatedUser(user_id=1, username="user", name="用户", role="CUSTOMER")

    with pytest.raises(ToolExecutionError):
        await tool_executor.execute(
            "run_1",
            user,
            "get_product_information",
            {},
            lambda args: args,
            lambda _value: "should not run",
            audit,
        )

    assert len(audit.records) == 1
    assert audit.records[0]["success"] is False
