"""repair content-source origin authority and V1 digest guards

Revision ID: 0008_checkpoint_content_source_authority_repair
Revises: 0007_checkpoint_content_source_authority_guards
Create Date: 2026-09-28
"""

from __future__ import annotations

import hashlib
import re
import struct
from collections.abc import Mapping, Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0008_checkpoint_content_source_authority_repair"
down_revision: str | None = "0007_checkpoint_content_source_authority_guards"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_DIGEST_DOMAIN_TAG = b"dianshang-agent/content-source-sha256/v1\x00"
_CHAT_MESSAGE_ROLE_PURPOSE: Mapping[str, tuple[str, str]] = {
    "QUESTION": ("USER", "USER_INPUT"),
    "EFFECTIVE_QUESTION": ("USER", "USER_INPUT"),
    "CURRENT_ISSUE": ("USER", "USER_INPUT"),
    "FINAL_ANSWER": ("ASSISTANT", "FINAL_ANSWER"),
}


def _lp64(data: bytes) -> bytes:
    return struct.pack(">Q", len(data)) + data


def _content_digest(row: Mapping[str, object], content: str) -> str:
    preimage = _DIGEST_DOMAIN_TAG
    preimage += _lp64(str(row["source_kind"]).encode("ascii"))
    preimage += _lp64(str(row["content_role"]).encode("ascii"))
    preimage += struct.pack(">Q", int(row["content_schema_version"]))
    preimage += _lp64(str(row["normalization_version"]).encode("ascii"))
    preimage += _lp64(content.encode("utf-8"))
    return hashlib.sha256(preimage).hexdigest()


def _digest_sql(row: str) -> str:
    return (
        "LOWER(SHA2(CONCAT("
        "_binary'dianshang-agent/content-source-sha256/v1', 0x00, "
        f"UNHEX(LPAD(HEX(OCTET_LENGTH(CONVERT({row}.source_kind USING binary))), 16, '0')), "
        f"CONVERT({row}.source_kind USING binary), "
        f"UNHEX(LPAD(HEX(OCTET_LENGTH(CONVERT({row}.content_role USING binary))), 16, '0')), "
        f"CONVERT({row}.content_role USING binary), "
        f"UNHEX(LPAD(HEX({row}.content_schema_version), 16, '0')), "
        f"UNHEX(LPAD(HEX(OCTET_LENGTH(CONVERT({row}.normalization_version USING binary))), 16, '0')), "
        f"CONVERT({row}.normalization_version USING binary), "
        f"UNHEX(LPAD(HEX(OCTET_LENGTH({row}.raw_content_utf8)), 16, '0')), "
        f"{row}.raw_content_utf8), 256))"
    )


def _slot_role_is_valid(slot: object, role: object) -> bool:
    if not isinstance(slot, str) or not isinstance(role, str):
        return False
    fixed = {
        "memory.current_issue": "CURRENT_ISSUE",
        "memory.conversation_summary": "CONVERSATION_SUMMARY",
        "active_run.question": "QUESTION",
        "active_run.effective_question": "EFFECTIVE_QUESTION",
        "active_run.decision_reason": "PLAN_REASON",
        "active_run.draft_answer": "DRAFT_ANSWER",
        "active_run.final_answer": "FINAL_ANSWER",
        "active_run.error_summary": "ERROR_DETAIL",
        "active_run.plan.goal": "PLAN_GOAL",
        "active_run.plan.decision_reason": "PLAN_REASON",
    }
    if slot in fixed:
        return fixed[slot] == role
    indexed = (
        (r"^active_run\.plan\.missing_information\[[0-9]+\]$", "PLAN_MISSING_INFORMATION"),
        (r"^active_run\.retrieval_evidence\[[0-9]+\]\.file_name$", "RETRIEVAL_FILE_NAME"),
        (r"^active_run\.retrieval_evidence\[[0-9]+\]\.snippet$", "RETRIEVAL_SNIPPET"),
        (r"^active_run\.response_meta\.sources\[[0-9]+\]\.file_name$", "RETRIEVAL_FILE_NAME"),
        (r"^active_run\.response_meta\.sources\[[0-9]+\]\.snippet$", "RETRIEVAL_SNIPPET"),
    )
    return any(expected == role and re.fullmatch(pattern, slot) for pattern, expected in indexed)


def _fail(message: str) -> None:
    raise RuntimeError(f"0008 authority preflight rejected existing data: {message}")


def _preflight_attempts(bind: sa.Connection) -> None:
    rows = bind.execute(
        sa.text(
            "SELECT a.attempt_id, a.actor_user_id, a.actor_role, a.service_principal, "
            "a.subject_user_id, c.user_id AS conversation_owner_id "
            "FROM agent_run_attempt a "
            "LEFT JOIN chat_conversation c ON c.id = a.conversation_id"
        )
    ).mappings()
    for row in rows:
        if (
            row["conversation_owner_id"] is None
            or row["subject_user_id"] != row["conversation_owner_id"]
        ):
            _fail("attempt conversation has no exact owner/subject authority")
        role = row["actor_role"]
        actor_id = row["actor_user_id"]
        service = row["service_principal"]
        if role == "SYSTEM":
            if actor_id is not None or service != "CHECKPOINT_RUNTIME":
                _fail("SYSTEM current execution actor is incomplete")
        elif role in {"CUSTOMER", "ADMIN"}:
            if actor_id is None or service is not None:
                _fail("user current execution actor is incomplete")
            if role == "CUSTOMER" and actor_id != row["subject_user_id"]:
                _fail("customer attempt actor does not match the conversation subject")
        else:
            _fail("attempt current execution actor role is not registered")


def _preflight_sources(bind: sa.Connection) -> None:
    rows = bind.execute(
        sa.text(
            "SELECT s.*, m.id AS origin_id, m.conversation_id AS origin_conversation_id, "
            "m.role AS origin_role, m.content AS origin_content, "
            "m.source_run_id AS origin_run_id, m.source_attempt_id AS origin_attempt_id, "
            "m.message_purpose AS origin_purpose, c.user_id AS conversation_owner_id, "
            "a.conversation_id AS attempt_conversation_id, "
            "a.subject_user_id AS attempt_subject_user_id, "
            "a.actor_user_id AS attempt_actor_user_id, a.actor_role AS attempt_actor_role, "
            "a.service_principal AS attempt_service_principal, "
            "e.run_id AS effect_run_id, e.attempt_id AS effect_attempt_id, "
            "e.purpose AS effect_purpose, e.effect_type AS effect_type "
            "FROM agent_content_source_revision s "
            "LEFT JOIN chat_message m ON m.id = s.origin_chat_message_id "
            "LEFT JOIN chat_conversation c ON c.id = s.conversation_id "
            "LEFT JOIN agent_run_attempt a "
            "ON a.attempt_id = s.producing_attempt_id AND a.run_id = s.run_id "
            "LEFT JOIN agent_effect e ON e.id = m.effect_id"
        )
    ).mappings()
    for row in rows:
        record_id = str(row["source_record_id"])
        if (
            row["source_kind"] != "CHAT_MESSAGE"
            or not record_id.isdecimal()
            or record_id.startswith("0")
            or int(record_id) <= 0
            or int(row["source_revision"]) != 1
            or row["origin_id"] is None
            or int(row["origin_id"]) != int(record_id)
        ):
            _fail("source kind/identity lacks a typed CHAT_MESSAGE origin")
        if int(row["content_schema_version"]) != 1:
            _fail("source content schema version is not registered")
        digest = str(row["content_sha256"])
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            _fail("source digest is not lowercase hexadecimal SHA-256")
        raw = bytes(row["raw_content_utf8"])
        try:
            raw_text = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            _fail("source bytes are not strict UTF-8")
        if raw_text != row["origin_content"] or _content_digest(row, raw_text) != digest:
            _fail("source bytes/digest disagree with the exact origin")
        expected = _CHAT_MESSAGE_ROLE_PURPOSE.get(str(row["content_role"]))
        if expected is None or expected != (row["origin_role"], row["origin_purpose"]):
            _fail("source role has no matching ChatMessage role/purpose authority")
        if (
            row["conversation_owner_id"] is None
            or row["origin_conversation_id"] != row["conversation_id"]
            or row["conversation_owner_id"] != row["subject_user_id"]
            or row["attempt_conversation_id"] != row["conversation_id"]
            or row["attempt_subject_user_id"] != row["subject_user_id"]
            or row["origin_run_id"] != row["run_id"]
            or row["origin_attempt_id"] != row["producing_attempt_id"]
            or row["effect_run_id"] != row["run_id"]
            or row["effect_attempt_id"] != row["producing_attempt_id"]
            or row["effect_purpose"] != row["origin_purpose"]
            or row["effect_type"] != "CHAT_MESSAGE"
        ):
            _fail("source origin/attempt/effect domain binding is incomplete")
        if row["origin_role"] == "ASSISTANT":
            if (
                row["producing_principal_kind"] != "SERVICE"
                or row["producing_actor_id"] is not None
                or row["producing_service_principal"] != "CHECKPOINT_RUNTIME"
            ):
                _fail("source SERVICE principal disagrees with its immutable origin")
        elif (
            row["attempt_actor_role"] != "CUSTOMER"
            or row["attempt_actor_user_id"] != row["conversation_owner_id"]
            or row["producing_principal_kind"] != "USER"
            or row["producing_actor_id"] != row["attempt_actor_user_id"]
            or row["producing_service_principal"] is not None
        ):
            _fail("source USER principal disagrees with its immutable origin")


def _preflight_holds(bind: sa.Connection) -> None:
    rows = bind.execute(
        sa.text(
            "SELECT h.*, p.publication_version AS current_publication_version, "
            "p.conversation_id AS publication_conversation_id, "
            "x.conversation_id AS execution_conversation_id, c.user_id AS conversation_owner_id, "
            "s.conversation_id AS source_conversation_id, s.subject_user_id AS source_subject_user_id, "
            "s.content_role AS source_content_role, s.content_sha256 AS source_content_sha256 "
            "FROM agent_checkpoint_content_reference h "
            "LEFT JOIN agent_checkpoint_publication p ON p.thread_id = h.thread_id "
            "LEFT JOIN agent_thread_execution x ON x.thread_id = h.thread_id "
            "LEFT JOIN chat_conversation c ON c.id = p.conversation_id "
            "LEFT JOIN agent_content_source_revision s "
            "ON s.source_kind = h.source_kind "
            "AND s.source_record_id = h.source_record_id "
            "AND s.source_revision = h.source_revision"
        )
    ).mappings()
    for row in rows:
        if (
            row["current_publication_version"] is None
            or int(row["publication_version"]) != int(row["current_publication_version"])
            or row["conversation_owner_id"] is None
            or row["publication_conversation_id"] != row["execution_conversation_id"]
            or row["source_conversation_id"] != row["publication_conversation_id"]
            or row["source_subject_user_id"] != row["conversation_owner_id"]
            or row["source_content_role"] != row["content_role"]
            or row["source_content_sha256"] != row["content_sha256"]
            or not _slot_role_is_valid(row["projection_slot"], row["content_role"])
        ):
            _fail("publication hold has a stale, cross-domain, or malformed binding")


def _preflight() -> None:
    bind = op.get_bind()
    _preflight_attempts(bind)
    _preflight_sources(bind)
    _preflight_holds(bind)


def _drop_checks_referencing_column(table: str, column: str) -> None:
    """Drop the deployed check, including MySQL-truncated convention names."""

    rows = op.get_bind().execute(
        sa.text(
            "SELECT tc.constraint_name "
            "FROM information_schema.table_constraints tc "
            "JOIN information_schema.check_constraints cc "
            "  ON cc.constraint_schema = tc.constraint_schema "
            " AND cc.constraint_name = tc.constraint_name "
            "WHERE tc.constraint_schema = DATABASE() "
            "  AND tc.table_name = :table "
            "  AND tc.constraint_type = 'CHECK' "
            "  AND LOWER(cc.check_clause) LIKE :column_pattern"
        ),
        {"table": table, "column_pattern": f"%{column.lower()}%"},
    )
    for (constraint_name,) in rows:
        op.drop_constraint(op.f(str(constraint_name)), table, type_="check")


def _create_source_insert_trigger() -> None:
    op.execute(
        f"""
        CREATE TRIGGER trg_agent_content_source_revision_validate_insert
        BEFORE INSERT ON agent_content_source_revision FOR EACH ROW
        BEGIN
            DECLARE v_missing INTEGER DEFAULT 0;
            DECLARE v_conversation BIGINT;
            DECLARE v_role VARCHAR(32);
            DECLARE v_content LONGTEXT;
            DECLARE v_run VARCHAR(64);
            DECLARE v_attempt VARCHAR(64);
            DECLARE v_purpose VARCHAR(64);
            DECLARE v_owner BIGINT;
            DECLARE v_attempt_conversation BIGINT;
            DECLARE v_attempt_subject BIGINT;
            DECLARE v_attempt_actor BIGINT;
            DECLARE v_attempt_role VARCHAR(32);
            DECLARE v_attempt_service VARCHAR(64);
            DECLARE v_expected_digest CHAR(64);
            DECLARE CONTINUE HANDLER FOR NOT FOUND SET v_missing = 1;

            SELECT m.conversation_id, m.role, m.content, m.source_run_id,
                   m.source_attempt_id, m.message_purpose, c.user_id,
                   a.conversation_id, a.subject_user_id, a.actor_user_id,
                   a.actor_role, a.service_principal
              INTO v_conversation, v_role, v_content, v_run, v_attempt,
                   v_purpose, v_owner, v_attempt_conversation, v_attempt_subject,
                   v_attempt_actor, v_attempt_role, v_attempt_service
              FROM chat_message m
              JOIN chat_conversation c ON c.id = m.conversation_id
              JOIN agent_run_attempt a
                ON a.attempt_id = m.source_attempt_id
               AND a.run_id = m.source_run_id
              JOIN agent_effect e
                ON e.id = m.effect_id
               AND e.run_id = m.source_run_id
               AND e.attempt_id = m.source_attempt_id
               AND e.purpose = m.message_purpose
               AND e.effect_type = 'CHAT_MESSAGE'
             WHERE m.id = NEW.origin_chat_message_id
             LIMIT 1;

            IF v_missing = 1 OR v_owner IS NULL THEN
                SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'content source origin missing';
            END IF;
            IF NOT (NEW.source_kind <=> 'CHAT_MESSAGE')
               OR NOT (NEW.source_record_id <=> CAST(NEW.origin_chat_message_id AS CHAR))
               OR NOT (NEW.source_revision <=> 1) THEN
                SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'content source origin identity invalid';
            END IF;
            IF NOT (v_conversation <=> NEW.conversation_id)
               OR NOT (v_attempt_conversation <=> NEW.conversation_id)
               OR NOT (v_owner <=> NEW.subject_user_id)
               OR NOT (v_attempt_subject <=> NEW.subject_user_id)
               OR NOT (v_run <=> NEW.run_id)
               OR NOT (v_attempt <=> NEW.producing_attempt_id) THEN
                SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'content source origin domain mismatch';
            END IF;
            IF OCTET_LENGTH(v_content) <> OCTET_LENGTH(NEW.raw_content_utf8)
               OR NOT (CONVERT(v_content USING binary) <=> NEW.raw_content_utf8) THEN
                SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'content source origin bytes mismatch';
            END IF;
            IF v_attempt_role = 'SYSTEM' THEN
                IF v_attempt_actor IS NOT NULL
                   OR NOT (v_attempt_service <=> 'CHECKPOINT_RUNTIME') THEN
                    SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'content source execution actor invalid';
                END IF;
            ELSEIF v_attempt_role IN ('CUSTOMER', 'ADMIN') THEN
                IF v_attempt_actor IS NULL OR v_attempt_service IS NOT NULL THEN
                    SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'content source execution actor invalid';
                END IF;
            ELSE
                SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'content source execution actor invalid';
            END IF;

            IF v_role = 'ASSISTANT' AND v_purpose = 'FINAL_ANSWER' THEN
                IF NOT (NEW.content_role <=> 'FINAL_ANSWER')
                   OR NOT (NEW.producing_principal_kind <=> 'SERVICE')
                   OR NEW.producing_actor_id IS NOT NULL
                   OR NOT (NEW.producing_service_principal <=> 'CHECKPOINT_RUNTIME') THEN
                    SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'content source service principal mismatch';
                END IF;
            ELSEIF v_role = 'USER' AND v_purpose = 'USER_INPUT' THEN
                IF NEW.content_role NOT IN ('QUESTION', 'EFFECTIVE_QUESTION', 'CURRENT_ISSUE')
                   OR NOT (NEW.producing_principal_kind <=> 'USER')
                   OR v_attempt_role <> 'CUSTOMER'
                   OR NOT (v_attempt_actor <=> v_owner)
                   OR NOT (v_attempt_actor <=> NEW.producing_actor_id)
                   OR NEW.producing_service_principal IS NOT NULL THEN
                    SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'content source user principal mismatch';
                END IF;
            ELSE
                SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'content source role purpose mismatch';
            END IF;

            IF NEW.content_schema_version IS NULL
               OR NEW.content_schema_version <> 1 THEN
                SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'content source schema version unsupported';
            END IF;
            SET v_expected_digest = {_digest_sql("NEW")};
            IF NOT (NEW.content_sha256 <=> v_expected_digest) THEN
                SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'content source digest mismatch';
            END IF;
        END
        """
    )


def _create_hold_insert_trigger() -> None:
    op.execute(
        """
        CREATE TRIGGER trg_agent_checkpoint_content_reference_validate_insert
        BEFORE INSERT ON agent_checkpoint_content_reference FOR EACH ROW
        BEGIN
            DECLARE v_missing INTEGER DEFAULT 0;
            DECLARE v_publication_version BIGINT;
            DECLARE v_publication_conversation BIGINT;
            DECLARE v_execution_conversation BIGINT;
            DECLARE v_owner BIGINT;
            DECLARE v_source_conversation BIGINT;
            DECLARE v_source_subject BIGINT;
            DECLARE v_origin BIGINT;
            DECLARE CONTINUE HANDLER FOR NOT FOUND SET v_missing = 1;

            SELECT p.publication_version, p.conversation_id, x.conversation_id,
                   c.user_id, s.conversation_id, s.subject_user_id,
                   s.origin_chat_message_id
              INTO v_publication_version, v_publication_conversation,
                   v_execution_conversation, v_owner, v_source_conversation,
                   v_source_subject, v_origin
              FROM agent_checkpoint_publication p
              JOIN agent_thread_execution x ON x.thread_id = p.thread_id
              JOIN chat_conversation c ON c.id = p.conversation_id
              JOIN agent_content_source_revision s
                ON s.source_kind = NEW.source_kind
               AND s.source_record_id = NEW.source_record_id
               AND s.source_revision = NEW.source_revision
               AND s.content_role = NEW.content_role
               AND s.content_sha256 = NEW.content_sha256
             WHERE p.thread_id = NEW.thread_id
             LIMIT 1;

            IF v_missing = 1 OR v_origin IS NULL OR v_owner IS NULL THEN
                SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'publication hold source domain missing';
            END IF;
            IF NOT (NEW.publication_version <=> v_publication_version) THEN
                SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'publication hold version mismatch';
            END IF;
            IF NOT (v_publication_conversation <=> v_execution_conversation)
               OR NOT (v_source_conversation <=> v_publication_conversation)
               OR NOT (v_source_subject <=> v_owner) THEN
                SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'publication hold authority mismatch';
            END IF;
        END
        """
    )


def _create_immutability_triggers() -> None:
    op.execute(
        """
        CREATE TRIGGER trg_agent_run_attempt_actor_identity_immutable
        BEFORE UPDATE ON agent_run_attempt FOR EACH ROW
        BEGIN
            IF NOT (NEW.attempt_id <=> OLD.attempt_id)
               OR NOT (NEW.run_id <=> OLD.run_id)
               OR NOT (NEW.thread_id <=> OLD.thread_id)
               OR NOT (NEW.conversation_id <=> OLD.conversation_id)
               OR NOT (NEW.actor_user_id <=> OLD.actor_user_id)
               OR NOT (NEW.actor_role <=> OLD.actor_role)
               OR NOT (NEW.service_principal <=> OLD.service_principal)
               OR NOT (NEW.subject_user_id <=> OLD.subject_user_id) THEN
                SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT =
                    'agent_run_attempt actor identity is immutable';
            END IF;
        END
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_chat_message_content_origin_immutable
        BEFORE UPDATE ON chat_message FOR EACH ROW
        BEGIN
            IF EXISTS (
                SELECT 1 FROM agent_content_source_revision s
                 WHERE s.origin_chat_message_id = OLD.id
            ) AND (
                NOT (NEW.conversation_id <=> OLD.conversation_id)
                OR NOT (NEW.role <=> OLD.role)
                OR NOT (NEW.content <=> OLD.content)
                OR NOT (NEW.source_run_id <=> OLD.source_run_id)
                OR NOT (NEW.source_attempt_id <=> OLD.source_attempt_id)
                OR NOT (NEW.message_purpose <=> OLD.message_purpose)
                OR NOT (NEW.effect_id <=> OLD.effect_id)
            ) THEN
                SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT =
                    'referenced chat_message origin authority is immutable';
            END IF;
        END
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_agent_effect_content_origin_immutable
        BEFORE UPDATE ON agent_effect FOR EACH ROW
        BEGIN
            IF EXISTS (
                SELECT 1
                  FROM chat_message m
                  JOIN agent_content_source_revision s
                    ON s.origin_chat_message_id = m.id
                 WHERE m.effect_id = OLD.id
            ) AND (
                NOT (NEW.run_id <=> OLD.run_id)
                OR NOT (NEW.attempt_id <=> OLD.attempt_id)
                OR NOT (NEW.node_name <=> OLD.node_name)
                OR NOT (NEW.purpose <=> OLD.purpose)
                OR NOT (NEW.sequence <=> OLD.sequence)
                OR NOT (NEW.effect_type <=> OLD.effect_type)
                OR NOT (NEW.idempotency_key <=> OLD.idempotency_key)
                OR NOT (NEW.payload_digest <=> OLD.payload_digest)
            ) THEN
                SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT =
                    'referenced agent_effect origin authority is immutable';
            END IF;
        END
        """
    )


def upgrade() -> None:
    _preflight()

    _drop_checks_referencing_column("agent_run_attempt", "service_principal")
    op.create_check_constraint(
        "attempt_service_principal_authority",
        "agent_run_attempt",
        "(actor_role = 'SYSTEM' AND actor_user_id IS NULL "
        "AND service_principal = 'CHECKPOINT_RUNTIME') OR "
        "(actor_role IN ('CUSTOMER', 'ADMIN') AND actor_user_id IS NOT NULL "
        "AND service_principal IS NULL)",
    )
    _drop_checks_referencing_column(
        "agent_content_source_revision",
        "content_schema_version",
    )
    op.create_check_constraint(
        "content_source_schema_version_v1",
        "agent_content_source_revision",
        "content_schema_version = 1",
    )

    for trigger in (
        "trg_agent_content_source_revision_validate_insert",
        "trg_agent_checkpoint_content_reference_validate_insert",
        "trg_agent_run_attempt_actor_identity_immutable",
        "trg_chat_message_content_origin_immutable",
        "trg_agent_effect_content_origin_immutable",
    ):
        op.execute(f"DROP TRIGGER IF EXISTS {trigger}")
    _create_source_insert_trigger()
    _create_hold_insert_trigger()
    _create_immutability_triggers()


def downgrade() -> None:
    raise RuntimeError(
        "0008 cannot be safely downgraded because it repairs content-source "
        "authority, digest validation, and immutable origin guarantees"
    )
