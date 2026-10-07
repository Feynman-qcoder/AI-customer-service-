"""hybrid retrieval structured rules

Revision ID: 0002_hybrid_retrieval_rules
Revises: 0001_initial_python_agent_schema
Create Date: 2026-07-23
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0002_hybrid_retrieval_rules"
down_revision: str | None = "0001_initial_python_agent_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "after_sale_rule_version",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("version_code", sa.String(64), nullable=False),
        sa.Column("description", sa.String(255), nullable=False),
        sa.Column("effective_from", sa.DateTime(), nullable=False),
        sa.Column("effective_to", sa.DateTime(), nullable=True),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("version_code", name="uk_after_sale_rule_version_code"),
    )
    op.create_table(
        "after_sale_rule",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("rule_code", sa.String(64), nullable=False),
        sa.Column("version_id", sa.BigInteger(), sa.ForeignKey("after_sale_rule_version.id"), nullable=False),
        sa.Column("title", sa.String(128), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("after_sale_type", sa.String(32), nullable=False),
        sa.Column("priority", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("effective_from", sa.DateTime(), nullable=False),
        sa.Column("effective_to", sa.DateTime(), nullable=True),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("rule_code", name="uk_after_sale_rule_code"),
    )
    op.create_table(
        "after_sale_rule_condition",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("rule_id", sa.BigInteger(), sa.ForeignKey("after_sale_rule.id"), nullable=False),
        sa.Column("product_category", sa.String(64), nullable=True),
        sa.Column("order_status", sa.String(32), nullable=True),
        sa.Column("payment_status", sa.String(32), nullable=True),
        sa.Column("shipment_status", sa.String(32), nullable=True),
        sa.Column("signed_within_days", sa.Integer(), nullable=True),
        sa.Column("after_sale_type", sa.String(32), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_table(
        "agent_retrieval_trace",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.String(64), nullable=False),
        sa.Column("candidate_id", sa.String(128), nullable=False),
        sa.Column("source_type", sa.String(32), nullable=False),
        sa.Column("document_id", sa.String(64), nullable=True),
        sa.Column("chunk_id", sa.String(64), nullable=True),
        sa.Column("rule_id", sa.String(64), nullable=True),
        sa.Column("original_score", sa.Numeric(10, 6), nullable=False),
        sa.Column("fused_score", sa.Numeric(10, 6), nullable=True),
        sa.Column("rerank_score", sa.Numeric(10, 6), nullable=True),
        sa.Column("selected", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("decision_reason", sa.String(255), nullable=True),
        sa.Column("metadata_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index(
        "idx_after_sale_rule_effective",
        "after_sale_rule",
        ["status", "effective_from", "effective_to"],
    )
    op.create_index(
        "idx_after_sale_rule_condition_lookup",
        "after_sale_rule_condition",
        ["after_sale_type", "order_status"],
    )
    op.create_index("idx_agent_retrieval_trace_run", "agent_retrieval_trace", ["run_id", "created_at"])


def downgrade() -> None:
    op.drop_table("agent_retrieval_trace")
    op.drop_table("after_sale_rule_condition")
    op.drop_table("after_sale_rule")
    op.drop_table("after_sale_rule_version")
