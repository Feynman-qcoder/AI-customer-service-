"""add immutable checkpoint content source revisions and publication holds

Revision ID: 0006_checkpoint_content_source_revision
Revises: 0005_signed_manifest_index
Create Date: 2026-09-28
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import mysql

from alembic import op

revision: str = "0006_checkpoint_content_source_revision"
down_revision: str | None = "0005_signed_manifest_index"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_NOW = sa.text("CURRENT_TIMESTAMP(6)")


def upgrade() -> None:
    # Alembic creates ``version_num`` as VARCHAR(32).  The frozen descriptive
    # revision identifier is longer, so widen this metadata column before
    # Alembic records the successful 0006 step.  This is a lossless expansion.
    op.alter_column(
        "alembic_version",
        "version_num",
        existing_type=sa.String(32),
        type_=sa.String(64),
        existing_nullable=False,
    )
    op.create_table(
        "agent_content_source_revision",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("source_kind", sa.String(64, collation="ascii_bin"), nullable=False),
        sa.Column("source_record_id", sa.String(130, collation="ascii_bin"), nullable=False),
        sa.Column("source_revision", sa.BigInteger(), nullable=False),
        sa.Column("content_role", sa.String(64, collation="ascii_bin"), nullable=False),
        sa.Column("content_schema_version", sa.Integer(), nullable=False),
        sa.Column("normalization_version", sa.String(32, collation="ascii_bin"), nullable=False),
        sa.Column("content_sha256", sa.String(64, collation="ascii_bin"), nullable=False),
        sa.Column("raw_content_utf8", mysql.LONGBLOB(), nullable=False),
        sa.Column("conversation_id", sa.BigInteger(), nullable=False),
        sa.Column("subject_user_id", sa.BigInteger(), nullable=False),
        sa.Column("producing_principal_kind", sa.String(16, collation="ascii_bin"), nullable=False),
        sa.Column("producing_actor_id", sa.BigInteger(), nullable=True),
        sa.Column(
            "producing_service_principal",
            sa.String(128, collation="ascii_bin"),
            nullable=True,
        ),
        sa.Column("run_id", sa.String(64), nullable=False),
        sa.Column("producing_attempt_id", sa.String(64), nullable=False),
        sa.Column("created_at", mysql.DATETIME(fsp=6), nullable=False, server_default=_NOW),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["chat_conversation.id"],
            name="fk_content_source_conversation",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["subject_user_id"],
            ["user_account.id"],
            name="fk_content_source_subject",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["producing_actor_id"],
            ["user_account.id"],
            name="fk_content_source_actor",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["producing_attempt_id", "run_id"],
            ["agent_run_attempt.attempt_id", "agent_run_attempt.run_id"],
            name="fk_content_source_attempt_run",
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint(
            "source_kind",
            "source_record_id",
            "source_revision",
            name="uk_content_source_exact_revision",
        ),
        sa.UniqueConstraint(
            "source_kind",
            "source_record_id",
            "source_revision",
            "content_role",
            "content_sha256",
            name="uk_content_source_hold_binding",
        ),
        sa.CheckConstraint("source_revision > 0", name="content_source_revision_positive"),
        sa.CheckConstraint(
            "content_schema_version > 0",
            name="content_source_schema_version_positive",
        ),
        sa.CheckConstraint(
            "source_kind IN ('CHAT_MESSAGE', 'AGENT_AUDIT_CONTENT', 'ACTION_RECORD', 'RAG_DOCUMENT')",
            name="content_source_kind_allowed",
        ),
        sa.CheckConstraint(
            "content_role IN ('QUESTION', 'EFFECTIVE_QUESTION', 'PLAN_GOAL', 'PLAN_REASON', "
            "'PLAN_MISSING_INFORMATION', 'CURRENT_ISSUE', 'CONVERSATION_SUMMARY', "
            "'RETRIEVAL_FILE_NAME', 'RETRIEVAL_SNIPPET', 'TOOL_CONTENT', 'DRAFT_ANSWER', "
            "'FINAL_ANSWER', 'ERROR_DETAIL')",
            name="content_source_role_allowed",
        ),
        sa.CheckConstraint(
            "normalization_version = 'RAW_UTF8_V1'",
            name="content_source_normalization_allowed",
        ),
        sa.CheckConstraint(
            "CHAR_LENGTH(content_sha256) = 64",
            name="content_source_digest_length",
        ),
        sa.CheckConstraint(
            "(producing_principal_kind = 'USER' AND producing_actor_id IS NOT NULL "
            "AND producing_service_principal IS NULL) OR "
            "(producing_principal_kind = 'SERVICE' AND producing_actor_id IS NULL "
            "AND producing_service_principal IS NOT NULL)",
            name="content_source_principal_complete",
        ),
    )
    op.create_index(
        "idx_content_source_conversation",
        "agent_content_source_revision",
        ["conversation_id", "subject_user_id"],
    )
    op.create_index(
        "idx_content_source_run_attempt",
        "agent_content_source_revision",
        ["run_id", "producing_attempt_id"],
    )

    op.create_table(
        "agent_checkpoint_content_reference",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("thread_id", sa.String(128), nullable=False),
        sa.Column("publication_version", sa.BigInteger(), nullable=False),
        sa.Column("reference_ordinal", sa.Integer(), nullable=False),
        sa.Column("projection_slot", sa.String(255, collation="ascii_bin"), nullable=False),
        sa.Column("content_role", sa.String(64, collation="ascii_bin"), nullable=False),
        sa.Column("source_kind", sa.String(64, collation="ascii_bin"), nullable=False),
        sa.Column("source_record_id", sa.String(130, collation="ascii_bin"), nullable=False),
        sa.Column("source_revision", sa.BigInteger(), nullable=False),
        sa.Column("content_sha256", sa.String(64, collation="ascii_bin"), nullable=False),
        sa.Column("created_at", mysql.DATETIME(fsp=6), nullable=False, server_default=_NOW),
        sa.ForeignKeyConstraint(
            ["thread_id"],
            ["agent_checkpoint_publication.thread_id"],
            name="fk_checkpoint_content_ref_publication",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            [
                "source_kind",
                "source_record_id",
                "source_revision",
                "content_role",
                "content_sha256",
            ],
            [
                "agent_content_source_revision.source_kind",
                "agent_content_source_revision.source_record_id",
                "agent_content_source_revision.source_revision",
                "agent_content_source_revision.content_role",
                "agent_content_source_revision.content_sha256",
            ],
            name="fk_checkpoint_content_ref_exact_source",
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint(
            "thread_id",
            "publication_version",
            "projection_slot",
            name="uk_checkpoint_content_ref_slot",
        ),
        sa.UniqueConstraint(
            "thread_id",
            "publication_version",
            "content_role",
            name="uk_checkpoint_content_ref_role",
        ),
        sa.UniqueConstraint(
            "thread_id",
            "publication_version",
            "reference_ordinal",
            name="uk_checkpoint_content_ref_ordinal",
        ),
        sa.CheckConstraint(
            "publication_version > 0",
            name="checkpoint_content_ref_version_positive",
        ),
        sa.CheckConstraint(
            "reference_ordinal >= 0",
            name="checkpoint_content_ref_ordinal_nonnegative",
        ),
        sa.CheckConstraint(
            "source_revision > 0",
            name="checkpoint_content_ref_revision_positive",
        ),
        sa.CheckConstraint(
            "CHAR_LENGTH(content_sha256) = 64",
            name="checkpoint_content_ref_digest_length",
        ),
    )
    op.create_index(
        "idx_checkpoint_content_ref_publication",
        "agent_checkpoint_content_reference",
        ["thread_id", "publication_version"],
    )
    op.create_index(
        "idx_checkpoint_content_ref_source",
        "agent_checkpoint_content_reference",
        ["source_kind", "source_record_id", "source_revision"],
    )

    op.execute(
        "CREATE TRIGGER trg_agent_content_source_revision_append_only "
        "BEFORE UPDATE ON agent_content_source_revision FOR EACH ROW "
        "SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = "
        "'agent_content_source_revision is append-only'"
    )


def downgrade() -> None:
    raise RuntimeError(
        "0006 cannot be safely downgraded because exact source revisions and "
        "publication retention holds may already be recovery authority"
    )
