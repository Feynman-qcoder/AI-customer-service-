"""add durable runtime authority and replay-safe effect schema

Revision ID: 0004_durable_runtime
Revises: 0003_agent_run_runtime_metadata
Create Date: 2026-09-22
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import mysql

from alembic import op

revision: str = "0004_durable_runtime"
down_revision: str | None = "0003_agent_run_runtime_metadata"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_NOW = sa.text("CURRENT_TIMESTAMP(6)")


def upgrade() -> None:
    op.create_table(
        "agent_run_attempt",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("attempt_id", sa.String(64), nullable=False),
        sa.Column("run_id", sa.String(64), nullable=False),
        sa.Column("thread_id", sa.String(128), nullable=False),
        sa.Column("conversation_id", sa.BigInteger(), nullable=False),
        sa.Column("actor_user_id", sa.BigInteger(), nullable=True),
        sa.Column("actor_role", sa.String(32), nullable=False),
        sa.Column("subject_user_id", sa.BigInteger(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("fence_version", sa.BigInteger(), nullable=True),
        sa.Column("started_at", mysql.DATETIME(fsp=6), nullable=False),
        sa.Column("completed_at", mysql.DATETIME(fsp=6), nullable=True),
        sa.Column("error_type", sa.String(64), nullable=True),
        sa.Column("created_at", mysql.DATETIME(fsp=6), nullable=False, server_default=_NOW),
        sa.Column("updated_at", mysql.DATETIME(fsp=6), nullable=False, server_default=_NOW),
        sa.ForeignKeyConstraint(["run_id"], ["agent_run.run_id"], name="fk_attempt_run"),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["chat_conversation.id"],
            name="fk_attempt_conversation",
        ),
        sa.ForeignKeyConstraint(["actor_user_id"], ["user_account.id"], name="fk_attempt_actor"),
        sa.ForeignKeyConstraint(["subject_user_id"], ["user_account.id"], name="fk_attempt_subject"),
        sa.UniqueConstraint("attempt_id", name="uk_agent_run_attempt_id"),
        sa.UniqueConstraint("attempt_id", "run_id", name="uk_agent_run_attempt_run"),
        sa.CheckConstraint(
            "fence_version IS NULL OR fence_version > 0",
            name="attempt_fence_positive",
        ),
        sa.CheckConstraint(
            "status IN ('REGISTERED', 'ACTIVE', 'RELEASED', 'COMPLETED', 'FAILED')",
            name="attempt_status_allowed",
        ),
    )
    op.create_index(
        "idx_agent_attempt_run_started",
        "agent_run_attempt",
        ["run_id", "started_at"],
    )
    op.create_index(
        "idx_agent_attempt_thread_status",
        "agent_run_attempt",
        ["thread_id", "status"],
    )

    op.create_table(
        "agent_thread_execution",
        sa.Column("thread_id", sa.String(128), primary_key=True),
        sa.Column("conversation_id", sa.BigInteger(), nullable=False),
        sa.Column("owner_attempt_id", sa.String(64), nullable=True),
        sa.Column("fence_version", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("lease_expires_at", mysql.DATETIME(fsp=6), nullable=True),
        sa.Column("created_at", mysql.DATETIME(fsp=6), nullable=False, server_default=_NOW),
        sa.Column("updated_at", mysql.DATETIME(fsp=6), nullable=False, server_default=_NOW),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["chat_conversation.id"],
            name="fk_thread_conversation",
        ),
        sa.ForeignKeyConstraint(
            ["owner_attempt_id"],
            ["agent_run_attempt.attempt_id"],
            name="fk_thread_owner_attempt",
        ),
        sa.UniqueConstraint("conversation_id", name="uq_agent_thread_execution_conversation_id"),
        sa.CheckConstraint(
            "fence_version >= 0",
            name="thread_fence_nonnegative",
        ),
        sa.CheckConstraint(
            "(owner_attempt_id IS NULL AND lease_expires_at IS NULL) "
            "OR (owner_attempt_id IS NOT NULL AND lease_expires_at IS NOT NULL)",
            name="thread_lease_complete",
        ),
    )

    op.create_table(
        "agent_checkpoint_publication",
        sa.Column("thread_id", sa.String(128), primary_key=True),
        sa.Column("conversation_id", sa.BigInteger(), nullable=False),
        sa.Column("publication_version", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("logical_namespace", sa.String(128), nullable=True),
        sa.Column("physical_namespace", sa.String(255), nullable=True),
        sa.Column("checkpoint_id", sa.String(128), nullable=True),
        sa.Column("checkpoint_digest", sa.String(64), nullable=True),
        sa.Column("previous_logical_namespace", sa.String(128), nullable=True),
        sa.Column("previous_physical_namespace", sa.String(255), nullable=True),
        sa.Column("previous_checkpoint_id", sa.String(128), nullable=True),
        sa.Column("manifest_root", sa.String(64), nullable=True),
        sa.Column("manifest_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", mysql.DATETIME(fsp=6), nullable=False, server_default=_NOW),
        sa.Column("updated_at", mysql.DATETIME(fsp=6), nullable=False, server_default=_NOW),
        sa.ForeignKeyConstraint(
            ["thread_id"],
            ["agent_thread_execution.thread_id"],
            name="fk_publication_thread",
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["chat_conversation.id"],
            name="fk_publication_conversation",
        ),
        sa.UniqueConstraint("conversation_id", name="uq_agent_checkpoint_publication_conversation_id"),
        sa.CheckConstraint(
            "publication_version >= 0",
            name="pub_version_nonnegative",
        ),
        sa.CheckConstraint(
            "manifest_count >= 0",
            name="pub_manifest_count_nonnegative",
        ),
        sa.CheckConstraint(
            "("
            "publication_version = 0 AND logical_namespace IS NULL "
            "AND physical_namespace IS NULL AND checkpoint_id IS NULL "
            "AND checkpoint_digest IS NULL AND manifest_root IS NULL AND manifest_count = 0"
            ") OR ("
            "publication_version > 0 AND logical_namespace IS NOT NULL "
            "AND physical_namespace IS NOT NULL AND checkpoint_id IS NOT NULL "
            "AND checkpoint_digest IS NOT NULL AND manifest_root IS NOT NULL"
            ")",
            name="pub_pointer_complete",
        ),
        sa.CheckConstraint(
            "("
            "previous_logical_namespace IS NULL AND previous_physical_namespace IS NULL "
            "AND previous_checkpoint_id IS NULL"
            ") OR ("
            "previous_logical_namespace IS NOT NULL AND previous_physical_namespace IS NOT NULL "
            "AND previous_checkpoint_id IS NOT NULL"
            ")",
            name="pub_previous_complete",
        ),
        sa.CheckConstraint(
            "checkpoint_digest IS NULL OR CHAR_LENGTH(checkpoint_digest) = 64",
            name="pub_checkpoint_digest_length",
        ),
        sa.CheckConstraint(
            "manifest_root IS NULL OR CHAR_LENGTH(manifest_root) = 64",
            name="pub_manifest_root_length",
        ),
    )

    op.create_table(
        "agent_checkpoint_write_manifest",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("thread_id", sa.String(128), nullable=False),
        sa.Column("publication_version", sa.BigInteger(), nullable=False),
        sa.Column("logical_namespace", sa.String(128), nullable=False),
        sa.Column("physical_namespace", sa.String(255), nullable=False),
        sa.Column("checkpoint_id", sa.String(128), nullable=False),
        sa.Column("task_id", sa.String(128), nullable=False),
        sa.Column("write_index", sa.Integer(), nullable=False),
        sa.Column("channel", sa.String(128), nullable=False),
        sa.Column("content_digest", sa.String(64), nullable=False),
        sa.Column("created_at", mysql.DATETIME(fsp=6), nullable=False, server_default=_NOW),
        sa.ForeignKeyConstraint(
            ["thread_id"],
            ["agent_checkpoint_publication.thread_id"],
            name="fk_manifest_publication",
        ),
        sa.UniqueConstraint(
            "thread_id",
            "publication_version",
            "task_id",
            "write_index",
            name="uk_checkpoint_manifest_item",
        ),
        sa.CheckConstraint(
            "publication_version > 0",
            name="manifest_version_positive",
        ),
        sa.CheckConstraint(
            "write_index >= 0",
            name="manifest_write_index_nonnegative",
        ),
        sa.CheckConstraint(
            "CHAR_LENGTH(content_digest) = 64",
            name="manifest_digest_length",
        ),
    )
    op.create_index(
        "idx_checkpoint_manifest_publication",
        "agent_checkpoint_write_manifest",
        ["thread_id", "publication_version"],
    )

    op.create_table(
        "agent_effect",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.String(64), nullable=False),
        sa.Column("attempt_id", sa.String(64), nullable=False),
        sa.Column("node_name", sa.String(64), nullable=False),
        sa.Column("purpose", sa.String(64), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("effect_type", sa.String(32), nullable=False),
        sa.Column("idempotency_key", sa.String(64), nullable=False),
        sa.Column("payload_digest", sa.String(64), nullable=False),
        sa.Column("created_at", mysql.DATETIME(fsp=6), nullable=False, server_default=_NOW),
        sa.ForeignKeyConstraint(
            ["attempt_id", "run_id"],
            ["agent_run_attempt.attempt_id", "agent_run_attempt.run_id"],
            name="fk_agent_effect_attempt_run",
        ),
        sa.UniqueConstraint(
            "run_id",
            "node_name",
            "purpose",
            "sequence",
            name="uk_agent_effect_identity",
        ),
        sa.UniqueConstraint("idempotency_key", name="uk_agent_effect_idempotency"),
        sa.CheckConstraint("sequence >= 0", name="effect_sequence_nonnegative"),
        sa.CheckConstraint(
            "effect_type IN ('CHAT_MESSAGE', 'ACTION_PREPARE', 'LOCAL_AUDIT')",
            name="effect_type_allowed",
        ),
        sa.CheckConstraint(
            "CHAR_LENGTH(idempotency_key) = 64",
            name="effect_key_length",
        ),
        sa.CheckConstraint(
            "CHAR_LENGTH(payload_digest) = 64",
            name="effect_digest_length",
        ),
    )
    op.create_index("idx_agent_effect_attempt", "agent_effect", ["attempt_id", "created_at"])

    op.add_column("chat_message", sa.Column("source_run_id", sa.String(64), nullable=True))
    op.add_column("chat_message", sa.Column("source_attempt_id", sa.String(64), nullable=True))
    op.add_column("chat_message", sa.Column("message_purpose", sa.String(64), nullable=True))
    op.add_column("chat_message", sa.Column("message_sequence", sa.Integer(), nullable=True))
    op.add_column("chat_message", sa.Column("message_idempotency_key", sa.String(64), nullable=True))
    op.add_column("chat_message", sa.Column("effect_id", sa.BigInteger(), nullable=True))
    op.create_foreign_key(
        "fk_chat_message_attempt_run",
        "chat_message",
        "agent_run_attempt",
        ["source_attempt_id", "source_run_id"],
        ["attempt_id", "run_id"],
    )
    op.create_foreign_key(
        "fk_chat_message_effect",
        "chat_message",
        "agent_effect",
        ["effect_id"],
        ["id"],
    )
    op.create_unique_constraint(
        "uk_chat_message_idempotency",
        "chat_message",
        ["message_idempotency_key"],
    )
    op.create_unique_constraint("uk_chat_message_effect", "chat_message", ["effect_id"])
    op.create_check_constraint(
        "message_sequence_nonnegative",
        "chat_message",
        "message_sequence IS NULL OR message_sequence >= 0",
    )
    op.create_check_constraint(
        "chat_message_effect_complete",
        "chat_message",
        "("
        "effect_id IS NULL AND source_run_id IS NULL AND source_attempt_id IS NULL "
        "AND message_purpose IS NULL AND message_sequence IS NULL "
        "AND message_idempotency_key IS NULL"
        ") OR ("
        "effect_id IS NOT NULL AND source_run_id IS NOT NULL AND source_attempt_id IS NOT NULL "
        "AND message_purpose IS NOT NULL AND message_sequence IS NOT NULL "
        "AND message_idempotency_key IS NOT NULL"
        ")",
    )

    op.add_column("agent_step", sa.Column("attempt_id", sa.String(64), nullable=True))
    op.add_column("agent_step", sa.Column("effect_id", sa.BigInteger(), nullable=True))
    op.add_column("agent_step", sa.Column("effect_purpose", sa.String(64), nullable=True))
    op.add_column("agent_step", sa.Column("effect_sequence", sa.Integer(), nullable=True))
    op.add_column("agent_step", sa.Column("effect_idempotency_key", sa.String(64), nullable=True))
    op.create_foreign_key(
        "fk_agent_step_attempt_run",
        "agent_step",
        "agent_run_attempt",
        ["attempt_id", "run_id"],
        ["attempt_id", "run_id"],
    )
    op.create_foreign_key(
        "fk_agent_step_effect",
        "agent_step",
        "agent_effect",
        ["effect_id"],
        ["id"],
    )
    op.create_unique_constraint("uk_agent_step_effect", "agent_step", ["effect_id"])
    op.create_unique_constraint(
        "uk_agent_step_effect_idempotency",
        "agent_step",
        ["effect_idempotency_key"],
    )
    op.create_check_constraint(
        "agent_step_effect_sequence_nonnegative",
        "agent_step",
        "effect_sequence IS NULL OR effect_sequence >= 0",
    )
    op.create_check_constraint(
        "agent_step_effect_complete",
        "agent_step",
        "("
        "effect_id IS NULL AND attempt_id IS NULL AND effect_purpose IS NULL "
        "AND effect_sequence IS NULL AND effect_idempotency_key IS NULL"
        ") OR ("
        "effect_id IS NOT NULL AND attempt_id IS NOT NULL AND effect_purpose IS NOT NULL "
        "AND effect_sequence IS NOT NULL AND effect_idempotency_key IS NOT NULL"
        ")",
    )

    op.add_column("agent_action_request", sa.Column("attempt_id", sa.String(64), nullable=True))
    op.add_column("agent_action_request", sa.Column("effect_id", sa.BigInteger(), nullable=True))
    op.add_column("agent_action_request", sa.Column("effect_node_name", sa.String(64), nullable=True))
    op.add_column("agent_action_request", sa.Column("effect_purpose", sa.String(64), nullable=True))
    op.add_column("agent_action_request", sa.Column("effect_sequence", sa.Integer(), nullable=True))
    op.add_column(
        "agent_action_request",
        sa.Column("effect_idempotency_key", sa.String(64), nullable=True),
    )
    op.create_foreign_key(
        "fk_agent_action_attempt_run",
        "agent_action_request",
        "agent_run_attempt",
        ["attempt_id", "run_id"],
        ["attempt_id", "run_id"],
    )
    op.create_foreign_key(
        "fk_agent_action_effect",
        "agent_action_request",
        "agent_effect",
        ["effect_id"],
        ["id"],
    )
    op.create_unique_constraint(
        "uk_agent_action_effect",
        "agent_action_request",
        ["effect_id"],
    )
    op.create_unique_constraint(
        "uk_agent_action_effect_idempotency",
        "agent_action_request",
        ["effect_idempotency_key"],
    )
    op.create_check_constraint(
        "agent_action_effect_sequence_nonnegative",
        "agent_action_request",
        "effect_sequence IS NULL OR effect_sequence >= 0",
    )
    op.create_check_constraint(
        "agent_action_effect_complete",
        "agent_action_request",
        "("
        "effect_id IS NULL AND attempt_id IS NULL AND effect_node_name IS NULL "
        "AND effect_purpose IS NULL AND effect_sequence IS NULL "
        "AND effect_idempotency_key IS NULL"
        ") OR ("
        "effect_id IS NOT NULL AND attempt_id IS NOT NULL AND effect_node_name IS NOT NULL "
        "AND effect_purpose IS NOT NULL AND effect_sequence IS NOT NULL "
        "AND effect_idempotency_key IS NOT NULL"
        ")",
    )


def downgrade() -> None:
    raise RuntimeError(
        "downgrade is intentionally prohibited because it would discard "
        "attempt, lease high-water, publication, manifest, and effect evidence; "
        "disable the feature and apply a reviewed forward migration instead."
    )
