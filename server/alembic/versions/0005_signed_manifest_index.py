"""allow signed opaque checkpoint write indexes

Revision ID: 0005_signed_manifest_index
Revises: 0004_durable_runtime
Create Date: 2026-09-23
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0005_signed_manifest_index"
down_revision: str | None = "0004_durable_runtime"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_constraint(
        "manifest_write_index_nonnegative",
        "agent_checkpoint_write_manifest",
        type_="check",
    )


def downgrade() -> None:
    raise RuntimeError(
        "0005 cannot be safely downgraded because signed write indexes may already exist"
    )
