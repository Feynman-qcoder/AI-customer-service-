"""add immutable durable ACTION_PREPARE order-status evidence

Revision ID: 0013_prepare_status_evidence
Revises: 0012_durable_hitl_action_schema
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0013_prepare_status_evidence"
down_revision: str | None = "0012_durable_hitl_action_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_ERROR_TYPE = "PREPARE_EVIDENCE_UNRECOVERABLE"
_ERROR_SUMMARY = "Durable prepare evidence is unavailable; create a new action."


def _invalid_count(bind: sa.Connection, predicate: str, joins: str = "") -> int:
    value = bind.execute(
        sa.text(
            "SELECT COUNT(*) FROM agent_action_request a "
            f"{joins} WHERE a.confirmation_mode = 'DURABLE_INTERRUPT' "
            f"AND ({predicate})"
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
        raise RuntimeError(f"0013 preflight rejected {reason}")


def _preflight_historical_durable_actions(bind: sa.Connection) -> None:
    """Prove every 0012 durable row can only be failed closed, before DDL."""

    _require_clean_preflight(
        bind,
        predicate="a.lock_version < 0 OR a.lock_version >= 2147483647",
        reason="a durable action lock version that cannot be incremented safely",
    )
    _require_clean_preflight(
        bind,
        predicate=(
            "a.status <> 'PENDING' OR "
            "a.resume_status <> 'WAITING_ADMIN_DECISION' OR "
            "a.logical_action_id IS NULL OR "
            "a.idempotency_key <> CONCAT('durable:', a.logical_action_id) OR "
            "a.action_type NOT IN ('REFUND', 'ORDER_CANCELLATION') OR "
            "a.risk_level <> 'HIGH' OR "
            "a.legacy_original_status IS NOT NULL OR "
            "a.customer_confirmed_actor_id IS NULL OR "
            "a.customer_confirmed_actor_id <> a.created_by OR "
            "a.customer_confirmed_at IS NULL OR "
            "a.customer_confirmation_challenge_digest IS NULL OR "
            "NOT REGEXP_LIKE(a.customer_confirmation_challenge_digest, '^[0-9a-f]{64}$', 'c') OR "
            "a.admin_decision IS NOT NULL OR a.admin_decided_actor_id IS NOT NULL OR "
            "a.admin_decided_at IS NOT NULL OR a.admin_reason_code IS NOT NULL OR "
            "a.approved_by IS NOT NULL OR a.approved_at IS NOT NULL OR "
            "a.approval_note IS NOT NULL OR a.executed_at IS NOT NULL OR "
            "a.execution_result_code IS NOT NULL OR a.execution_error_type IS NOT NULL OR "
            "a.execution_error_summary IS NOT NULL OR "
            "NOT JSON_VALID(a.action_payload_json) OR "
            "JSON_TYPE(CAST(a.action_payload_json AS JSON)) <> 'OBJECT' OR "
            "JSON_LENGTH(CAST(a.action_payload_json AS JSON)) <> 1 OR "
            "JSON_TYPE(JSON_EXTRACT(a.action_payload_json, '$.reason')) <> 'STRING'"
        ),
        reason="an unsupported durable action state or evidence tuple",
    )
    _require_clean_preflight(
        bind,
        predicate=(
            "a.effect_id IS NULL OR a.attempt_id IS NULL OR "
            "a.effect_node_name IS NULL OR a.effect_purpose IS NULL OR "
            "a.effect_sequence IS NULL OR a.effect_idempotency_key IS NULL OR "
            "e.id IS NULL OR e.effect_type <> 'ACTION_PREPARE' OR "
            "e.node_name <> 'durable_action_prepare' OR e.purpose <> 'ACTION_PREPARE' OR "
            "e.sequence <> 0 OR NOT (e.run_id <=> a.run_id) OR "
            "NOT (e.attempt_id <=> a.attempt_id) OR "
            "NOT (e.node_name <=> a.effect_node_name) OR "
            "NOT (e.purpose <=> a.effect_purpose) OR "
            "NOT (e.sequence <=> a.effect_sequence) OR "
            "NOT (e.idempotency_key <=> a.effect_idempotency_key) OR "
            "NOT REGEXP_LIKE(e.payload_digest, '^[0-9a-f]{64}$', 'c') OR "
            "e.idempotency_key <> SHA2(CONCAT("
            "'{\"node_name\":', JSON_QUOTE(e.node_name), "
            "',\"purpose\":', JSON_QUOTE(e.purpose), "
            "',\"run_id\":', JSON_QUOTE(e.run_id), "
            "',\"sequence\":', CAST(e.sequence AS CHAR), '}'), 256)"
        ),
        joins="LEFT JOIN agent_effect e ON e.id = a.effect_id",
        reason="an orphaned or non-canonical durable effect linkage",
    )
    _require_clean_preflight(
        bind,
        predicate=(
            "origin.attempt_id IS NULL OR origin.run_id <> a.run_id OR "
            "origin.actor_role <> 'CUSTOMER' OR origin.service_principal IS NOT NULL OR "
            "origin.actor_user_id IS NULL OR origin.actor_user_id <> a.created_by OR "
            "origin.subject_user_id <> a.created_by OR "
            "run.run_id IS NULL OR run.user_id <> a.created_by OR "
            "run.conversation_id <> origin.conversation_id OR "
            "run.thread_id <> origin.thread_id OR "
            "run.status NOT IN ('RESUME_PENDING', 'WAITING_ADMIN_APPROVAL') OR "
            "run.error_type IS NOT NULL OR run.completed_at IS NOT NULL OR "
            "target.id IS NULL OR target.user_id <> a.created_by"
        ),
        joins=(
            "LEFT JOIN agent_run_attempt origin "
            "ON origin.attempt_id = a.attempt_id AND origin.run_id = a.run_id "
            "LEFT JOIN agent_run run ON run.run_id = a.run_id "
            "LEFT JOIN customer_order target ON target.id = a.target_order_id"
        ),
        reason="durable actor, subject, run, attempt, or target authority mismatch",
    )


def _add_status_evidence_column() -> None:
    op.add_column(
        "agent_action_request",
        sa.Column(
            "prepared_order_status",
            sa.String(length=32, collation="ascii_bin"),
            nullable=True,
        ),
    )


def _fail_closed_historical_rows() -> None:
    escaped_summary = _ERROR_SUMMARY.replace("'", "''")
    op.execute(
        "UPDATE agent_action_request SET "
        "status = 'FAILED', resume_status = 'COMPLETED', "
        f"execution_error_type = '{_ERROR_TYPE}', "
        f"execution_error_summary = '{escaped_summary}', "
        "lock_version = lock_version + 1 "
        "WHERE confirmation_mode = 'DURABLE_INTERRUPT'"
    )
    op.execute(
        "UPDATE agent_run r INNER JOIN agent_action_request a ON a.run_id = r.run_id SET "
        "r.status = 'FAILED', r.error_type = 'PREPARE_EVIDENCE_UNRECOVERABLE', "
        "r.completed_at = CURRENT_TIMESTAMP(6) "
        "WHERE a.confirmation_mode = 'DURABLE_INTERRUPT' "
        "AND a.prepared_order_status IS NULL "
        "AND r.status IN ('RESUME_PENDING', 'WAITING_ADMIN_APPROVAL')"
    )


def _add_status_evidence_constraint() -> None:
    escaped_summary = _ERROR_SUMMARY.replace("'", "''")
    op.create_check_constraint(
        "action_prepared_status_mode_binding",
        "agent_action_request",
        "(confirmation_mode = 'DURABLE_INTERRUPT' AND (("
        "prepared_order_status IS NOT NULL AND "
        "REGEXP_LIKE(prepared_order_status, '^[A-Z][A-Z0-9_]{0,31}$', 'c')"
        ") OR ("
        "prepared_order_status IS NULL AND status = 'FAILED' "
        "AND resume_status = 'COMPLETED' "
        f"AND execution_error_type = '{_ERROR_TYPE}' "
        f"AND execution_error_summary = '{escaped_summary}' "
        "AND execution_result_code IS NULL AND admin_decision IS NULL "
        "AND admin_decided_actor_id IS NULL AND admin_decided_at IS NULL "
        "AND admin_reason_code IS NULL"
        "))) OR (confirmation_mode IN ('R2_STATELESS_COMPAT', 'LEGACY_UNVERIFIED') "
        "AND prepared_order_status IS NULL)",
    )


def _create_immutable_status_trigger() -> None:
    op.execute(
        """
        CREATE TRIGGER trg_agent_action_prepared_status_immutable
        BEFORE UPDATE ON agent_action_request FOR EACH ROW
        BEGIN
            IF NOT (OLD.prepared_order_status <=> NEW.prepared_order_status) THEN
                SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT =
                    'prepared order status evidence is immutable';
            END IF;
        END
        """
    )


def upgrade() -> None:
    bind = op.get_bind()
    _preflight_historical_durable_actions(bind)
    _add_status_evidence_column()
    _fail_closed_historical_rows()
    _add_status_evidence_constraint()
    _create_immutable_status_trigger()


def downgrade() -> None:
    raise RuntimeError(
        "0013 is forward-only; removing immutable ACTION_PREPARE evidence would "
        "make durable administrative review unverifiable"
    )
