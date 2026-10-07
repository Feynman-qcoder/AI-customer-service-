"""initial python agent schema

Revision ID: 0001_initial_python_agent_schema
Revises:
Create Date: 2026-07-23
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0001_initial_python_agent_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "user_account",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("username", sa.String(64), nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("display_name", sa.String(128), nullable=False),
        sa.Column("role", sa.String(32), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("last_login_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("username", name="uk_user_account_username"),
    )
    op.create_table(
        "product_catalog",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("product_code", sa.String(64), nullable=False),
        sa.Column("product_name", sa.String(128), nullable=False),
        sa.Column("category", sa.String(64), nullable=False),
        sa.Column("sale_status", sa.String(32), nullable=False),
        sa.Column("price", sa.Numeric(10, 2), nullable=False),
        sa.Column("stock_quantity", sa.Integer(), nullable=False),
        sa.Column("dispatch_rule", sa.String(255), nullable=False),
        sa.Column("after_sale_rule", sa.String(512), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("product_code", name="uk_product_catalog_code"),
    )
    op.create_table(
        "chat_conversation",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.BigInteger(), sa.ForeignKey("user_account.id"), nullable=True),
        sa.Column("conversation_no", sa.String(64), nullable=False),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("conversation_no", name="uk_chat_conversation_no"),
    )
    op.create_table(
        "customer_order",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("order_no", sa.String(64), nullable=False),
        sa.Column("user_id", sa.BigInteger(), sa.ForeignKey("user_account.id"), nullable=False),
        sa.Column("product_id", sa.BigInteger(), sa.ForeignKey("product_catalog.id"), nullable=False),
        sa.Column("quantity", sa.Integer(), nullable=False),
        sa.Column("amount", sa.Numeric(10, 2), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("paid_at", sa.DateTime(), nullable=True),
        sa.Column("expected_ship_at", sa.DateTime(), nullable=True),
        sa.Column("shipped_at", sa.DateTime(), nullable=True),
        sa.Column("signed_at", sa.DateTime(), nullable=True),
        sa.Column("receiver_name", sa.String(64), nullable=False),
        sa.Column("receiver_phone", sa.String(32), nullable=False),
        sa.Column("receiver_address", sa.String(255), nullable=False),
        sa.Column("remark", sa.String(255), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("order_no", name="uk_customer_order_no"),
    )
    op.create_table(
        "shipment_event",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("order_id", sa.BigInteger(), sa.ForeignKey("customer_order.id"), nullable=False),
        sa.Column("carrier", sa.String(64), nullable=True),
        sa.Column("tracking_no", sa.String(64), nullable=True),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("location", sa.String(128), nullable=True),
        sa.Column("event_note", sa.String(255), nullable=False),
        sa.Column("event_time", sa.DateTime(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_table(
        "chat_message",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("conversation_id", sa.BigInteger(), sa.ForeignKey("chat_conversation.id"), nullable=False),
        sa.Column("role", sa.String(32), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("sources_json", sa.Text(), nullable=True),
        sa.Column("retrieval_score", sa.Numeric(8, 4), nullable=True),
        sa.Column("confidence_level", sa.String(32), nullable=True),
        sa.Column("need_human", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_table(
        "kb_document",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("original_name", sa.String(512), nullable=False),
        sa.Column("storage_name", sa.String(128), nullable=False),
        sa.Column("storage_path", sa.String(1024), nullable=False),
        sa.Column("file_type", sa.String(32), nullable=False),
        sa.Column("file_size", sa.BigInteger(), nullable=False),
        sa.Column("file_sha256", sa.String(64), nullable=True),
        sa.Column("uploaded_by", sa.BigInteger(), sa.ForeignKey("user_account.id"), nullable=True),
        sa.Column("lock_version", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("failure_reason", sa.String(512), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("file_sha256", name="uk_kb_document_sha256"),
    )
    op.create_table(
        "document_processing_task",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("document_id", sa.BigInteger(), sa.ForeignKey("kb_document.id"), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_retry_count", sa.Integer(), nullable=False, server_default="3"),
        sa.Column("next_retry_at", sa.DateTime(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("error_message", sa.String(1000), nullable=True),
        sa.Column("lock_version", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("document_id", name="uk_document_processing_task_document"),
    )
    op.create_table(
        "kb_chunk",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("document_id", sa.BigInteger(), sa.ForeignKey("kb_document.id"), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("char_count", sa.Integer(), nullable=False),
        sa.Column("vector_point_id", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("document_id", "chunk_index", name="uk_kb_chunk_document_index"),
    )
    op.create_table(
        "chat_message_source",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("message_id", sa.BigInteger(), sa.ForeignKey("chat_message.id"), nullable=False),
        sa.Column("document_id", sa.BigInteger(), sa.ForeignKey("kb_document.id"), nullable=False),
        sa.Column("chunk_id", sa.BigInteger(), sa.ForeignKey("kb_chunk.id"), nullable=True),
        sa.Column("rank_no", sa.Integer(), nullable=False),
        sa.Column("retrieval_score", sa.Numeric(8, 4), nullable=False),
        sa.Column("snippet_snapshot", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_table(
        "support_ticket",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.BigInteger(), sa.ForeignKey("user_account.id"), nullable=True),
        sa.Column("ticket_no", sa.String(64), nullable=False),
        sa.Column("conversation_id", sa.BigInteger(), sa.ForeignKey("chat_conversation.id"), nullable=False),
        sa.Column("category", sa.String(32), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("contact", sa.String(255), nullable=True),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("handler_id", sa.BigInteger(), sa.ForeignKey("user_account.id"), nullable=True),
        sa.Column("priority", sa.String(32), nullable=False, server_default="NORMAL"),
        sa.Column("handling_note", sa.Text(), nullable=True),
        sa.Column("resolution", sa.Text(), nullable=True),
        sa.Column("resolved_at", sa.DateTime(), nullable=True),
        sa.Column("lock_version", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("ticket_no", name="uk_support_ticket_no"),
    )
    op.create_table(
        "ticket_operation_log",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("ticket_id", sa.BigInteger(), sa.ForeignKey("support_ticket.id"), nullable=False),
        sa.Column("operator_id", sa.BigInteger(), sa.ForeignKey("user_account.id"), nullable=False),
        sa.Column("previous_status", sa.String(32), nullable=True),
        sa.Column("next_status", sa.String(32), nullable=False),
        sa.Column("operation_note", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_table(
        "model_runtime_config",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("temperature", sa.Numeric(3, 2), nullable=False),
        sa.Column("top_k", sa.Integer(), nullable=False),
        sa.Column("min_retrieval_score", sa.Numeric(4, 3), nullable=False, server_default="0.350"),
        sa.Column("mock_enabled", sa.Boolean(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_table(
        "agent_run",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.String(64), nullable=False),
        sa.Column("thread_id", sa.String(128), nullable=False),
        sa.Column("conversation_id", sa.BigInteger(), sa.ForeignKey("chat_conversation.id"), nullable=False),
        sa.Column("user_id", sa.BigInteger(), sa.ForeignKey("user_account.id"), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("intent", sa.String(64), nullable=True),
        sa.Column("risk_level", sa.String(32), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=False),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("final_answer", sa.Text(), nullable=True),
        sa.Column("error_type", sa.String(64), nullable=True),
        sa.Column("request_id", sa.String(64), nullable=False),
        sa.UniqueConstraint("run_id", name="uk_agent_run_run_id"),
    )
    op.create_table(
        "agent_step",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.String(64), nullable=False),
        sa.Column("node_name", sa.String(64), nullable=False),
        sa.Column("input_summary", sa.Text(), nullable=True),
        sa.Column("output_summary", sa.Text(), nullable=True),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_summary", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_table(
        "agent_tool_call",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.String(64), nullable=False),
        sa.Column("tool_name", sa.String(128), nullable=False),
        sa.Column("redacted_arguments_json", sa.Text(), nullable=False),
        sa.Column("result_summary", sa.Text(), nullable=True),
        sa.Column("success", sa.Boolean(), nullable=False),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("duration_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_table(
        "agent_action_request",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.String(64), nullable=False),
        sa.Column("action_type", sa.String(64), nullable=False),
        sa.Column("target_order_id", sa.BigInteger(), sa.ForeignKey("customer_order.id"), nullable=True),
        sa.Column("action_payload_json", sa.Text(), nullable=False),
        sa.Column("risk_level", sa.String(32), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("idempotency_key", sa.String(128), nullable=False),
        sa.Column("lock_version", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_by", sa.BigInteger(), sa.ForeignKey("user_account.id"), nullable=False),
        sa.Column("approved_by", sa.BigInteger(), sa.ForeignKey("user_account.id"), nullable=True),
        sa.Column("approval_note", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("approved_at", sa.DateTime(), nullable=True),
        sa.Column("executed_at", sa.DateTime(), nullable=True),
        sa.UniqueConstraint("idempotency_key", name="uk_agent_action_request_idempotency"),
    )
    op.create_table(
        "agent_feedback",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.String(64), nullable=False),
        sa.Column("rating", sa.String(32), nullable=False),
        sa.Column("reason", sa.String(512), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )

    op.create_index("idx_customer_order_user_created", "customer_order", ["user_id", "created_at"])
    op.create_index("idx_shipment_event_order_time", "shipment_event", ["order_id", "event_time"])
    op.create_index("idx_agent_run_thread", "agent_run", ["thread_id"])
    op.create_index("idx_agent_step_run_created", "agent_step", ["run_id", "created_at"])
    op.create_index("idx_agent_tool_call_run_created", "agent_tool_call", ["run_id", "created_at"])
    op.create_index("idx_agent_action_status_created", "agent_action_request", ["status", "created_at"])


def downgrade() -> None:
    for table_name in [
        "agent_feedback",
        "agent_action_request",
        "agent_tool_call",
        "agent_step",
        "agent_run",
        "model_runtime_config",
        "ticket_operation_log",
        "support_ticket",
        "chat_message_source",
        "kb_chunk",
        "document_processing_task",
        "kb_document",
        "chat_message",
        "shipment_event",
        "customer_order",
        "chat_conversation",
        "product_catalog",
        "user_account",
    ]:
        op.drop_table(table_name)
