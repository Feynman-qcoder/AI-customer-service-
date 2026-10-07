"""add owner-scoped conversation working memory and rolling summary references

Revision ID: 0011_conversation_layered_memory
Revises: 0010_service_checkpoint_source
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import mysql

from alembic import op

revision: str = "0011_conversation_layered_memory"
down_revision: str | None = "0010_service_checkpoint_source"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _memory_provenance_check(column: str) -> str:
    return (
        f"{column} IS NULL OR ("
        f"JSON_TYPE({column}) = 'OBJECT' AND JSON_LENGTH({column}) = 3 "
        f"AND JSON_UNQUOTE(JSON_EXTRACT({column}, '$.source_kind')) "
        "IN ('CURRENT_INPUT', 'AUTHORIZED_READ_RESULT') "
        f"AND JSON_TYPE(JSON_EXTRACT({column}, '$.source_reference')) = 'STRING' "
        f"AND CHAR_LENGTH(JSON_UNQUOTE(JSON_EXTRACT({column}, '$.source_reference'))) "
        "BETWEEN 1 AND 256 "
        f"AND REGEXP_LIKE(JSON_UNQUOTE(JSON_EXTRACT({column}, '$.source_reference')), "
        "'^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$', 'c') "
        f"AND JSON_TYPE(JSON_EXTRACT({column}, '$.source_message_id')) = 'INTEGER' "
        f"AND CAST(JSON_UNQUOTE(JSON_EXTRACT({column}, '$.source_message_id')) "
        "AS DECIMAL(20, 0)) BETWEEN 1 AND 9223372036854775807)"
    )


def _prepare_service_source_trigger_replacement() -> str:
    bind = op.get_bind()
    trigger_row = bind.execute(
        sa.text(
            "SHOW CREATE TRIGGER trg_agent_content_source_revision_validate_insert"
        )
    ).one()
    original_statement = str(trigger_row[2])
    old_fragment = "'FINAL_ANSWER', 'ERROR_DETAIL'"
    new_fragment = "'FINAL_ANSWER', 'CONVERSATION_SUMMARY', 'ERROR_DETAIL'"
    if original_statement.count(old_fragment) != 1:
        raise RuntimeError(
            "0011 cannot prove the exact 0010 content-source trigger shape"
        )
    replacement = original_statement.replace(old_fragment, new_fragment, 1)
    replacement = re.sub(
        r"^CREATE\s+DEFINER=`[^`]+`@`[^`]+`\s+",
        "CREATE ",
        replacement,
        count=1,
        flags=re.IGNORECASE,
    )
    return replacement


def _replace_service_source_trigger(replacement: str) -> None:
    op.execute("DROP TRIGGER trg_agent_content_source_revision_validate_insert")
    op.execute(replacement)


def upgrade() -> None:
    service_source_trigger_replacement = (
        _prepare_service_source_trigger_replacement()
    )

    op.create_unique_constraint(
        "uk_chat_conversation_owner_scope",
        "chat_conversation",
        ["id", "user_id"],
    )
    op.create_unique_constraint(
        "uk_chat_message_conversation_cursor",
        "chat_message",
        ["id", "conversation_id"],
    )
    op.create_unique_constraint(
        "uk_content_source_memory_binding",
        "agent_content_source_revision",
        [
            "source_kind",
            "source_record_id",
            "source_revision",
            "content_role",
            "content_sha256",
            "conversation_id",
            "subject_user_id",
        ],
    )

    op.create_table(
        "conversation_working_memory",
        sa.Column("conversation_id", sa.BigInteger(), nullable=False),
        sa.Column("subject_user_id", sa.BigInteger(), nullable=False),
        sa.Column("active_order_no", sa.String(length=256), nullable=True),
        sa.Column("active_order_no_provenance", mysql.JSON(), nullable=True),
        sa.Column("active_product_code", sa.String(length=256), nullable=True),
        sa.Column("active_product_code_provenance", mysql.JSON(), nullable=True),
        sa.Column("current_issue", sa.String(length=4000), nullable=True),
        sa.Column("current_issue_provenance", mysql.JSON(), nullable=True),
        sa.Column(
            "last_intent",
            sa.String(length=64, collation="ascii_bin"),
            nullable=True,
        ),
        sa.Column("last_intent_provenance", mysql.JSON(), nullable=True),
        sa.Column("memory_revision", sa.BigInteger(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP(6)"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP(6)"),
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id", "subject_user_id"],
            ["chat_conversation.id", "chat_conversation.user_id"],
            name="fk_working_memory_conversation_owner",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["subject_user_id"],
            ["user_account.id"],
            name="fk_working_memory_subject",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("conversation_id"),
        sa.CheckConstraint(
            "memory_revision > 0",
            name="working_memory_revision_positive",
        ),
        sa.CheckConstraint(
            "last_intent IS NULL OR "
            "REGEXP_LIKE(last_intent, '^[A-Z][A-Z0-9_]{0,63}$', 'c')",
            name="working_memory_last_intent_normalized",
        ),
        sa.CheckConstraint(
            "(active_order_no IS NULL AND active_order_no_provenance IS NULL) OR "
            "(active_order_no IS NOT NULL AND active_order_no_provenance IS NOT NULL)",
            name="working_memory_active_order_pair",
        ),
        sa.CheckConstraint(
            "(active_product_code IS NULL AND active_product_code_provenance IS NULL) OR "
            "(active_product_code IS NOT NULL AND active_product_code_provenance IS NOT NULL)",
            name="working_memory_active_product_pair",
        ),
        sa.CheckConstraint(
            "(current_issue IS NULL AND current_issue_provenance IS NULL) OR "
            "(current_issue IS NOT NULL AND current_issue_provenance IS NOT NULL)",
            name="working_memory_current_issue_pair",
        ),
        sa.CheckConstraint(
            "(last_intent IS NULL AND last_intent_provenance IS NULL) OR "
            "(last_intent IS NOT NULL AND last_intent_provenance IS NOT NULL)",
            name="working_memory_last_intent_pair",
        ),
        sa.CheckConstraint(
            _memory_provenance_check("active_order_no_provenance"),
            name="working_memory_active_order_provenance_shape",
        ),
        sa.CheckConstraint(
            _memory_provenance_check("active_product_code_provenance"),
            name="working_memory_active_product_provenance_shape",
        ),
        sa.CheckConstraint(
            _memory_provenance_check("current_issue_provenance"),
            name="working_memory_current_issue_provenance_shape",
        ),
        sa.CheckConstraint(
            _memory_provenance_check("last_intent_provenance"),
            name="working_memory_last_intent_provenance_shape",
        ),
    )
    op.create_index(
        "idx_working_memory_subject_conversation",
        "conversation_working_memory",
        ["subject_user_id", "conversation_id"],
        unique=False,
    )

    op.create_table(
        "conversation_rolling_summary",
        sa.Column("conversation_id", sa.BigInteger(), nullable=False),
        sa.Column("subject_user_id", sa.BigInteger(), nullable=False),
        sa.Column(
            "source_kind",
            sa.String(length=64, collation="ascii_bin"),
            nullable=False,
        ),
        sa.Column(
            "source_record_id",
            sa.String(length=130, collation="ascii_bin"),
            nullable=False,
        ),
        sa.Column("source_revision", sa.BigInteger(), nullable=False),
        sa.Column(
            "content_role",
            sa.String(length=64, collation="ascii_bin"),
            nullable=False,
        ),
        sa.Column("content_schema_version", sa.Integer(), nullable=False),
        sa.Column(
            "normalization_version",
            sa.String(length=32, collation="ascii_bin"),
            nullable=False,
        ),
        sa.Column(
            "content_sha256",
            sa.String(length=64, collation="ascii_bin"),
            nullable=False,
        ),
        sa.Column("summary_until_message_id", sa.BigInteger(), nullable=False),
        sa.Column("summary_revision", sa.BigInteger(), nullable=False),
        sa.Column(
            "token_counter_version",
            sa.String(length=64, collation="ascii_bin"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP(6)"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP(6)"),
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id", "subject_user_id"],
            ["chat_conversation.id", "chat_conversation.user_id"],
            name="fk_rolling_summary_conversation_owner",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["subject_user_id"],
            ["user_account.id"],
            name="fk_rolling_summary_subject",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["summary_until_message_id", "conversation_id"],
            ["chat_message.id", "chat_message.conversation_id"],
            name="fk_rolling_summary_cursor_conversation",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            [
                "source_kind",
                "source_record_id",
                "source_revision",
                "content_role",
                "content_sha256",
                "conversation_id",
                "subject_user_id",
            ],
            [
                "agent_content_source_revision.source_kind",
                "agent_content_source_revision.source_record_id",
                "agent_content_source_revision.source_revision",
                "agent_content_source_revision.content_role",
                "agent_content_source_revision.content_sha256",
                "agent_content_source_revision.conversation_id",
                "agent_content_source_revision.subject_user_id",
            ],
            name="fk_rolling_summary_exact_source",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("conversation_id"),
        sa.CheckConstraint(
            "summary_revision > 0",
            name="rolling_summary_revision_positive",
        ),
        sa.CheckConstraint(
            "source_revision > 0",
            name="rolling_summary_source_revision_positive",
        ),
        sa.CheckConstraint(
            "source_kind = 'AGENT_AUDIT_CONTENT' AND "
            "content_role = 'CONVERSATION_SUMMARY' AND content_schema_version = 1 AND "
            "normalization_version = 'RAW_UTF8_V1'",
            name="rolling_summary_source_role_v1",
        ),
        sa.CheckConstraint(
            "REGEXP_LIKE(content_sha256, '^[0-9a-f]{64}$', 'c')",
            name="rolling_summary_digest_lower_hex",
        ),
        sa.CheckConstraint(
            "summary_until_message_id > 0",
            name="rolling_summary_cursor_positive",
        ),
        sa.CheckConstraint(
            "token_counter_version = 'UTF8_BYTES_CEIL_DIV_3_V1'",
            name="rolling_summary_counter_version_v1",
        ),
    )
    op.create_index(
        "idx_rolling_summary_subject_conversation",
        "conversation_rolling_summary",
        ["subject_user_id", "conversation_id"],
        unique=False,
    )
    op.create_index(
        "idx_rolling_summary_cursor",
        "conversation_rolling_summary",
        ["summary_until_message_id"],
        unique=False,
    )

    _replace_service_source_trigger(service_source_trigger_replacement)


def downgrade() -> None:
    raise RuntimeError(
        "0011 is forward-only; removing conversation memory would destroy persisted context"
    )
