"""add durable HITL evidence and classify legacy action requests

Revision ID: 0012_durable_hitl_action_schema
Revises: 0011_conversation_layered_memory
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import mysql

from alembic import op

revision: str = "0012_durable_hitl_action_schema"
down_revision: str | None = "0011_conversation_layered_memory"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_OLD_STATUSES = "'PENDING', 'APPROVING', 'REJECTING', 'EXECUTED', 'REJECTED'"


def _invalid_count(bind: sa.Connection, predicate: str, joins: str = "") -> int:
    value = bind.execute(
        sa.text(
            "SELECT COUNT(*) FROM agent_action_request a "
            f"{joins} WHERE {predicate}"
        )
    ).scalar_one()
    return int(value or 0)


def _require_clean_preflight(
    bind: sa.Connection,
    *,
    predicate: str,
    reason: str,
    joins: str = "",
) -> None:
    if _invalid_count(bind, predicate, joins) != 0:
        raise RuntimeError(f"0012 preflight rejected {reason}")


def _preflight_existing_actions(bind: sa.Connection) -> None:
    """Validate every legacy row before the first schema DDL.

    MySQL DDL is non-transactional.  The preflight therefore proves that all
    legacy rows can be classified without guessing before any new column,
    constraint, index, foreign key, or trigger is created.
    """

    _require_clean_preflight(
        bind,
        predicate=f"a.status NOT IN ({_OLD_STATUSES})",
        reason="an unknown legacy action status",
    )
    _require_clean_preflight(
        bind,
        predicate=(
            "NOT ((a.effect_id IS NULL AND a.attempt_id IS NULL "
            "AND a.effect_node_name IS NULL AND a.effect_purpose IS NULL "
            "AND a.effect_sequence IS NULL AND a.effect_idempotency_key IS NULL) OR "
            "(a.effect_id IS NOT NULL AND a.attempt_id IS NOT NULL "
            "AND a.effect_node_name IS NOT NULL AND a.effect_purpose IS NOT NULL "
            "AND a.effect_sequence IS NOT NULL AND a.effect_idempotency_key IS NOT NULL))"
        ),
        reason="an incomplete effect linkage tuple",
    )
    _require_clean_preflight(
        bind,
        predicate=(
            "a.effect_id IS NOT NULL AND (e.id IS NULL "
            "OR NOT (e.run_id <=> a.run_id) "
            "OR NOT (e.attempt_id <=> a.attempt_id) "
            "OR NOT (e.node_name <=> a.effect_node_name) "
            "OR NOT (e.purpose <=> a.effect_purpose) "
            "OR NOT (e.sequence <=> a.effect_sequence) "
            "OR NOT (e.idempotency_key <=> a.effect_idempotency_key) "
            "OR e.effect_type <> 'ACTION_PREPARE')"
        ),
        joins="LEFT JOIN agent_effect e ON e.id = a.effect_id",
        reason="a semantically corrupt effect linkage",
    )
    _require_clean_preflight(
        bind,
        predicate=(
            "a.status IN ('PENDING', 'APPROVING', 'REJECTING') AND "
            "(a.approved_by IS NOT NULL OR a.approved_at IS NOT NULL "
            "OR a.approval_note IS NOT NULL OR a.executed_at IS NOT NULL)"
        ),
        reason="unproven authority or terminal evidence on an open legacy action",
    )
    _require_clean_preflight(
        bind,
        predicate=(
            "a.status IN ('EXECUTED', 'REJECTED') AND "
            "((a.approved_by IS NULL) <> (a.approved_at IS NULL))"
        ),
        reason="a partial legacy administrator evidence tuple",
    )
    _require_clean_preflight(
        bind,
        predicate="a.status = 'REJECTED' AND a.executed_at IS NOT NULL",
        reason="execution evidence on a rejected legacy action",
    )
    _require_clean_preflight(
        bind,
        predicate="creator.id IS NULL OR creator.role <> 'CUSTOMER'",
        joins="LEFT JOIN user_account creator ON creator.id = a.created_by",
        reason="a non-customer legacy action creator",
    )
    _require_clean_preflight(
        bind,
        predicate=(
            "a.approved_by IS NOT NULL AND "
            "(decider.id IS NULL OR decider.role <> 'ADMIN')"
        ),
        joins="LEFT JOIN user_account decider ON decider.id = a.approved_by",
        reason="a non-admin legacy decision actor",
    )


def _add_columns() -> None:
    op.add_column(
        "agent_action_request",
        sa.Column(
            "logical_action_id",
            sa.String(length=64, collation="ascii_bin"),
            nullable=True,
        ),
    )
    op.add_column(
        "agent_action_request",
        sa.Column(
            "confirmation_mode",
            sa.String(length=32, collation="ascii_bin"),
            nullable=True,
        ),
    )
    op.add_column(
        "agent_action_request",
        sa.Column("customer_confirmed_actor_id", sa.BigInteger(), nullable=True),
    )
    op.add_column(
        "agent_action_request",
        sa.Column("customer_confirmed_at", mysql.DATETIME(fsp=6), nullable=True),
    )
    op.add_column(
        "agent_action_request",
        sa.Column(
            "customer_confirmation_challenge_digest",
            sa.String(length=64, collation="ascii_bin"),
            nullable=True,
        ),
    )
    op.add_column(
        "agent_action_request",
        sa.Column(
            "admin_decision",
            sa.String(length=16, collation="ascii_bin"),
            nullable=True,
        ),
    )
    op.add_column(
        "agent_action_request",
        sa.Column("admin_decided_actor_id", sa.BigInteger(), nullable=True),
    )
    op.add_column(
        "agent_action_request",
        sa.Column("admin_decided_at", mysql.DATETIME(fsp=6), nullable=True),
    )
    op.add_column(
        "agent_action_request",
        sa.Column(
            "admin_reason_code",
            sa.String(length=64, collation="ascii_bin"),
            nullable=True,
        ),
    )
    op.add_column(
        "agent_action_request",
        sa.Column(
            "resume_status",
            sa.String(length=32, collation="ascii_bin"),
            nullable=True,
        ),
    )
    op.add_column(
        "agent_action_request",
        sa.Column(
            "execution_result_code",
            sa.String(length=64, collation="ascii_bin"),
            nullable=True,
        ),
    )
    op.add_column(
        "agent_action_request",
        sa.Column(
            "execution_error_type",
            sa.String(length=64, collation="ascii_bin"),
            nullable=True,
        ),
    )
    op.add_column(
        "agent_action_request",
        sa.Column("execution_error_summary", sa.String(length=512), nullable=True),
    )
    op.add_column(
        "agent_action_request",
        sa.Column(
            "legacy_original_status",
            sa.String(length=32, collation="ascii_bin"),
            nullable=True,
        ),
    )


def _backfill_legacy_rows() -> None:
    op.execute(
        "UPDATE agent_action_request SET "
        "legacy_original_status = status, "
        "confirmation_mode = 'LEGACY_UNVERIFIED', "
        "resume_status = CASE "
        "WHEN status IN ('PENDING', 'APPROVING', 'REJECTING') THEN 'LEGACY_BLOCKED' "
        "ELSE 'COMPLETED' END, "
        "admin_decision = CASE "
        "WHEN status = 'EXECUTED' AND approved_by IS NOT NULL THEN 'APPROVE' "
        "WHEN status = 'REJECTED' AND approved_by IS NOT NULL THEN 'REJECT' "
        "ELSE NULL END, "
        "admin_decided_actor_id = CASE "
        "WHEN status IN ('EXECUTED', 'REJECTED') AND approved_by IS NOT NULL "
        "THEN approved_by ELSE NULL END, "
        "admin_decided_at = CASE "
        "WHEN status IN ('EXECUTED', 'REJECTED') AND approved_by IS NOT NULL "
        "THEN approved_at ELSE NULL END, "
        "admin_reason_code = CASE "
        "WHEN status = 'EXECUTED' AND approved_by IS NOT NULL THEN 'LEGACY_EXECUTED' "
        "WHEN status = 'REJECTED' AND approved_by IS NOT NULL THEN 'LEGACY_REJECTED' "
        "ELSE NULL END"
    )
    op.execute(
        "UPDATE agent_action_request SET status = 'LEGACY_REVIEW_REQUIRED' "
        "WHERE status IN ('PENDING', 'APPROVING', 'REJECTING')"
    )
    op.alter_column(
        "agent_action_request",
        "confirmation_mode",
        existing_type=sa.String(length=32, collation="ascii_bin"),
        nullable=False,
    )
    op.alter_column(
        "agent_action_request",
        "resume_status",
        existing_type=sa.String(length=32, collation="ascii_bin"),
        nullable=False,
    )


def _add_constraints() -> None:
    op.create_unique_constraint(
        "uk_agent_action_logical_action",
        "agent_action_request",
        ["logical_action_id"],
    )
    op.create_foreign_key(
        "fk_agent_action_customer_confirmer",
        "agent_action_request",
        "user_account",
        ["customer_confirmed_actor_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_foreign_key(
        "fk_agent_action_admin_decider",
        "agent_action_request",
        "user_account",
        ["admin_decided_actor_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    checks = (
        (
            "action_confirmation_mode_closed",
            "confirmation_mode IN "
            "('LEGACY_UNVERIFIED', 'R2_STATELESS_COMPAT', 'DURABLE_INTERRUPT')",
        ),
        (
            "action_resume_status_closed",
            "resume_status IN ('NOT_APPLICABLE', 'WAITING_ADMIN_DECISION', "
            "'RESUME_PENDING', 'RESUMED', 'COMPLETED', 'FAILED_RETRYABLE', "
            "'LEGACY_BLOCKED')",
        ),
        (
            "action_customer_confirmation_tuple",
            "(customer_confirmed_actor_id IS NULL AND customer_confirmed_at IS NULL "
            "AND customer_confirmation_challenge_digest IS NULL) OR "
            "(customer_confirmed_actor_id IS NOT NULL AND customer_confirmed_at IS NOT NULL "
            "AND customer_confirmation_challenge_digest IS NOT NULL)",
        ),
        (
            "action_customer_digest_shape",
            "customer_confirmation_challenge_digest IS NULL OR "
            "REGEXP_LIKE(customer_confirmation_challenge_digest, '^[0-9a-f]{64}$', 'c')",
        ),
        (
            "action_logical_identity_shape",
            "logical_action_id IS NULL OR "
            "REGEXP_LIKE(logical_action_id, '^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$', 'c')",
        ),
        (
            "action_confirmation_mode_binding",
            "(confirmation_mode = 'DURABLE_INTERRUPT' "
            "AND logical_action_id IS NOT NULL "
            "AND customer_confirmed_actor_id IS NOT NULL "
            "AND customer_confirmed_actor_id = created_by) OR "
            "(confirmation_mode IN ('LEGACY_UNVERIFIED', 'R2_STATELESS_COMPAT') "
            "AND logical_action_id IS NULL AND customer_confirmed_actor_id IS NULL)",
        ),
        (
            "action_admin_decision_tuple",
            "(admin_decision IS NULL AND admin_decided_actor_id IS NULL "
            "AND admin_decided_at IS NULL AND admin_reason_code IS NULL) OR "
            "(admin_decision IS NOT NULL AND admin_decided_actor_id IS NOT NULL "
            "AND admin_decided_at IS NOT NULL AND admin_reason_code IS NOT NULL)",
        ),
        (
            "action_admin_decision_closed",
            "admin_decision IS NULL OR admin_decision IN ('APPROVE', 'REJECT')",
        ),
        (
            "action_admin_reason_shape",
            "admin_reason_code IS NULL OR "
            "REGEXP_LIKE(admin_reason_code, '^[A-Z][A-Z0-9_]{0,63}$', 'c')",
        ),
        (
            "action_status_closed",
            "status IN ('LEGACY_REVIEW_REQUIRED', 'PENDING', 'APPROVING', "
            "'APPROVED', 'REJECTING', 'REJECTED', 'EXECUTING', 'EXECUTED', "
            "'STALE', 'FAILED', 'FAILED_RETRYABLE')",
        ),
        (
            "action_code_shapes",
            "(execution_result_code IS NULL OR "
            "REGEXP_LIKE(execution_result_code, '^[A-Z][A-Z0-9_]{0,63}$', 'c')) "
            "AND (execution_error_type IS NULL OR "
            "REGEXP_LIKE(execution_error_type, '^[A-Z][A-Z0-9_]{0,63}$', 'c'))",
        ),
        (
            "action_execution_outcome",
            "((execution_error_type IS NULL AND execution_error_summary IS NULL) OR "
            "(execution_error_type IS NOT NULL AND execution_error_summary IS NOT NULL "
            "AND status IN ('FAILED', 'FAILED_RETRYABLE'))) AND "
            "(execution_result_code IS NULL OR status = 'EXECUTED') AND "
            "NOT (execution_result_code IS NOT NULL AND execution_error_type IS NOT NULL) AND "
            "(executed_at IS NULL OR status = 'EXECUTED') AND "
            "(status NOT IN ('FAILED', 'FAILED_RETRYABLE') OR "
            "execution_error_type IS NOT NULL)",
        ),
        (
            "action_terminal_admin_binding",
            "confirmation_mode = 'LEGACY_UNVERIFIED' OR "
            "status NOT IN ('EXECUTED', 'REJECTED') OR "
            "(status = 'EXECUTED' AND admin_decision = 'APPROVE') OR "
            "(status = 'REJECTED' AND admin_decision = 'REJECT')",
        ),
        (
            "action_resume_mode_binding",
            "(confirmation_mode = 'R2_STATELESS_COMPAT' "
            "AND resume_status = 'NOT_APPLICABLE') OR "
            "(confirmation_mode = 'LEGACY_UNVERIFIED' "
            "AND resume_status IN ('LEGACY_BLOCKED', 'COMPLETED')) OR "
            "(confirmation_mode = 'DURABLE_INTERRUPT' "
            "AND resume_status IN ('WAITING_ADMIN_DECISION', 'RESUME_PENDING', "
            "'RESUMED', 'COMPLETED', 'FAILED_RETRYABLE'))",
        ),
        (
            "action_legacy_classification",
            "(confirmation_mode = 'LEGACY_UNVERIFIED' "
            "AND legacy_original_status IN "
            "('PENDING', 'APPROVING', 'REJECTING', 'EXECUTED', 'REJECTED') "
            "AND ((status = 'LEGACY_REVIEW_REQUIRED' "
            "AND legacy_original_status IN ('PENDING', 'APPROVING', 'REJECTING') "
            "AND resume_status = 'LEGACY_BLOCKED' AND admin_decision IS NULL) OR "
            "(status = legacy_original_status "
            "AND legacy_original_status IN ('EXECUTED', 'REJECTED') "
            "AND resume_status = 'COMPLETED'))) OR "
            "(confirmation_mode <> 'LEGACY_UNVERIFIED' "
            "AND legacy_original_status IS NULL "
            "AND status <> 'LEGACY_REVIEW_REQUIRED')",
        ),
    )
    for name, expression in checks:
        op.create_check_constraint(name, "agent_action_request", expression)


def _create_authority_triggers() -> None:
    for operation, reference in (("INSERT", "NEW"), ("UPDATE", "NEW")):
        trigger_name = f"trg_agent_action_authority_{operation.lower()}"
        op.execute(
            f"""
            CREATE TRIGGER {trigger_name}
            BEFORE {operation} ON agent_action_request FOR EACH ROW
            BEGIN
                DECLARE v_creator_role VARCHAR(32);
                DECLARE v_customer_role VARCHAR(32);
                DECLARE v_admin_role VARCHAR(32);

                SELECT role INTO v_creator_role
                  FROM user_account WHERE id = {reference}.created_by;
                IF v_creator_role <> 'CUSTOMER' THEN
                    SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT =
                        'action request creator authority is invalid';
                END IF;

                IF {reference}.customer_confirmed_actor_id IS NOT NULL THEN
                    SELECT role INTO v_customer_role
                      FROM user_account
                     WHERE id = {reference}.customer_confirmed_actor_id;
                    IF v_customer_role <> 'CUSTOMER'
                       OR {reference}.customer_confirmed_actor_id <> {reference}.created_by THEN
                        SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT =
                            'customer confirmation authority is invalid';
                    END IF;
                END IF;

                IF {reference}.admin_decided_actor_id IS NOT NULL THEN
                    SELECT role INTO v_admin_role
                      FROM user_account
                     WHERE id = {reference}.admin_decided_actor_id;
                    IF v_admin_role <> 'ADMIN' THEN
                        SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT =
                            'administrator decision authority is invalid';
                    END IF;
                END IF;
            END
            """
        )


def upgrade() -> None:
    bind = op.get_bind()
    _preflight_existing_actions(bind)
    _add_columns()
    _backfill_legacy_rows()
    _add_constraints()
    _create_authority_triggers()


def downgrade() -> None:
    raise RuntimeError(
        "0012 is forward-only; removing durable action evidence would destroy "
        "confirmation, decision, resume, execution, and legacy-classification facts"
    )
