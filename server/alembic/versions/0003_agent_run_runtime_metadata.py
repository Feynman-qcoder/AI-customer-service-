"""add agent run runtime metadata

Revision ID: 0003_agent_run_runtime_metadata
Revises: 0002_hybrid_retrieval_rules
Create Date: 2026-07-25 15:10:00
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0003_agent_run_runtime_metadata"
down_revision: str | None = "0002_hybrid_retrieval_rules"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("agent_run", sa.Column("model_name", sa.String(length=128), nullable=True))
    op.add_column("agent_run", sa.Column("config_version", sa.String(length=64), nullable=True))
    op.add_column("agent_run", sa.Column("prompt_version", sa.String(length=64), nullable=True))
    op.add_column("agent_run", sa.Column("provider_latency_ms", sa.Integer(), nullable=True))
    op.add_column("agent_run", sa.Column("prompt_tokens", sa.Integer(), nullable=True))
    op.add_column("agent_run", sa.Column("completion_tokens", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("agent_run", "completion_tokens")
    op.drop_column("agent_run", "prompt_tokens")
    op.drop_column("agent_run", "provider_latency_ms")
    op.drop_column("agent_run", "prompt_version")
    op.drop_column("agent_run", "config_version")
    op.drop_column("agent_run", "model_name")
