"""add origin authority and immutable publication-hold guards

Revision ID: 0007_checkpoint_content_source_authority_guards
Revises: 0006_checkpoint_content_source_revision
Create Date: 2026-09-28
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Mapping, Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0007_checkpoint_content_source_authority_guards"
down_revision: str | None = "0006_checkpoint_content_source_revision"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_DIGEST_DOMAIN_TAG = b"dianshang-agent/content-source-sha256/v1\x00"
_CHAT_MESSAGE_ROLE_PURPOSE: Mapping[str, tuple[str, str]] = {
    "QUESTION": ("USER", "USER_INPUT"),
    "EFFECTIVE_QUESTION": ("USER", "USER_INPUT"),
    "CURRENT_ISSUE": ("USER", "USER_INPUT"),
    "FINAL_ANSWER": ("ASSISTANT", "FINAL_ANSWER"),
}

_SLOT_ROLE_CHECK = (
    "(projection_slot = 'memory.current_issue' AND content_role = 'CURRENT_ISSUE') OR "
    "(projection_slot = 'memory.conversation_summary' AND content_role = 'CONVERSATION_SUMMARY') OR "
    "(projection_slot = 'active_run.question' AND content_role = 'QUESTION') OR "
    "(projection_slot = 'active_run.effective_question' AND content_role = 'EFFECTIVE_QUESTION') OR "
    "(projection_slot = 'active_run.decision_reason' AND content_role = 'PLAN_REASON') OR "
    "(projection_slot = 'active_run.draft_answer' AND content_role = 'DRAFT_ANSWER') OR "
    "(projection_slot = 'active_run.final_answer' AND content_role = 'FINAL_ANSWER') OR "
    "(projection_slot = 'active_run.error_summary' AND content_role = 'ERROR_DETAIL') OR "
    "(projection_slot = 'active_run.plan.goal' AND content_role = 'PLAN_GOAL') OR "
    "(projection_slot = 'active_run.plan.decision_reason' AND content_role = 'PLAN_REASON') OR "
    "(REGEXP_LIKE(projection_slot, '^active_run[.]plan[.]missing_information\\\\[[0-9]+\\\\]$', 'c') "
    "AND content_role = 'PLAN_MISSING_INFORMATION') OR "
    "(REGEXP_LIKE(projection_slot, '^active_run[.]retrieval_evidence\\\\[[0-9]+\\\\][.]file_name$', 'c') "
    "AND content_role = 'RETRIEVAL_FILE_NAME') OR "
    "(REGEXP_LIKE(projection_slot, '^active_run[.]retrieval_evidence\\\\[[0-9]+\\\\][.]snippet$', 'c') "
    "AND content_role = 'RETRIEVAL_SNIPPET') OR "
    "(REGEXP_LIKE(projection_slot, '^active_run[.]response_meta[.]sources\\\\[[0-9]+\\\\][.]file_name$', 'c') "
    "AND content_role = 'RETRIEVAL_FILE_NAME') OR "
    "(REGEXP_LIKE(projection_slot, '^active_run[.]response_meta[.]sources\\\\[[0-9]+\\\\][.]snippet$', 'c') "
    "AND content_role = 'RETRIEVAL_SNIPPET')"
)


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
    import re

    indexed = (
        (r"^active_run\.plan\.missing_information\[[0-9]+\]$", "PLAN_MISSING_INFORMATION"),
        (r"^active_run\.retrieval_evidence\[[0-9]+\]\.file_name$", "RETRIEVAL_FILE_NAME"),
        (r"^active_run\.retrieval_evidence\[[0-9]+\]\.snippet$", "RETRIEVAL_SNIPPET"),
        (r"^active_run\.response_meta\.sources\[[0-9]+\]\.file_name$", "RETRIEVAL_FILE_NAME"),
        (r"^active_run\.response_meta\.sources\[[0-9]+\]\.snippet$", "RETRIEVAL_SNIPPET"),
    )
    return any(expected == role and re.fullmatch(pattern, slot) for pattern, expected in indexed)


def _fail(message: str) -> None:
    raise RuntimeError(f"0007 authority preflight rejected existing data: {message}")


def _preflight_existing_rows() -> None:
    bind = op.get_bind()
    attempt_rows = bind.execute(
        sa.text(
            "SELECT a.attempt_id, a.run_id, a.conversation_id, a.actor_user_id, "
            "a.actor_role, a.subject_user_id, c.user_id AS conversation_owner_id "
            "FROM agent_run_attempt a "
            "LEFT JOIN chat_conversation c ON c.id = a.conversation_id"
        )
    ).mappings()
    for row in attempt_rows:
        if (
            row["conversation_owner_id"] is None
            or row["subject_user_id"] != row["conversation_owner_id"]
        ):
            _fail("attempt conversation has no exact owner/subject authority")
        actor_role = row["actor_role"]
        actor_user_id = row["actor_user_id"]
        if actor_role == "SYSTEM":
            _fail("pre-0007 SYSTEM attempt has no immutable service identity")
        if actor_role not in {"CUSTOMER", "ADMIN"} or actor_user_id is None:
            _fail("attempt current execution actor is incomplete")
        if actor_role == "CUSTOMER" and actor_user_id != row["subject_user_id"]:
            _fail("customer attempt actor does not match the conversation subject")

    source_rows = bind.execute(
        sa.text(
            "SELECT s.*, m.id AS origin_id, m.conversation_id AS origin_conversation_id, "
            "m.role AS origin_role, m.content AS origin_content, "
            "m.source_run_id AS origin_run_id, m.source_attempt_id AS origin_attempt_id, "
            "m.message_purpose AS origin_purpose, m.effect_id AS origin_effect_id, "
            "c.user_id AS conversation_owner_id, a.conversation_id AS attempt_conversation_id, "
            "a.subject_user_id AS attempt_subject_user_id, a.actor_user_id AS attempt_actor_user_id, "
            "a.actor_role AS attempt_actor_role, e.run_id AS effect_run_id, "
            "e.attempt_id AS effect_attempt_id, e.purpose AS effect_purpose, "
            "e.effect_type AS effect_type "
            "FROM agent_content_source_revision s "
            "LEFT JOIN chat_message m "
            "ON m.id = CAST(s.source_record_id AS UNSIGNED) "
            "LEFT JOIN chat_conversation c ON c.id = s.conversation_id "
            "LEFT JOIN agent_run_attempt a "
            "ON a.attempt_id = s.producing_attempt_id AND a.run_id = s.run_id "
            "LEFT JOIN agent_effect e ON e.id = m.effect_id"
        )
    ).mappings()
    for row in source_rows:
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
        expected_message = _CHAT_MESSAGE_ROLE_PURPOSE.get(str(row["content_role"]))
        if expected_message is None or expected_message != (
            row["origin_role"],
            row["origin_purpose"],
        ):
            _fail("source role has no matching ChatMessage role/purpose authority")
        if (
            row["origin_conversation_id"] != row["conversation_id"]
            or row["conversation_owner_id"] is None
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
            row["producing_principal_kind"] != "USER"
            or row["producing_actor_id"] != row["attempt_actor_user_id"]
            or row["attempt_actor_role"] != "CUSTOMER"
            or row["producing_service_principal"] is not None
        ):
            _fail("source USER principal disagrees with its immutable origin")

    hold_rows = bind.execute(
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
    for row in hold_rows:
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
            _fail("publication hold has a future, cross-domain, or malformed binding")


def upgrade() -> None:
    _preflight_existing_rows()

    op.add_column(
        "agent_run_attempt",
        sa.Column("service_principal", sa.String(64, collation="ascii_bin"), nullable=True),
    )
    op.create_check_constraint(
        "attempt_service_principal_authority",
        "agent_run_attempt",
        "(actor_role = 'SYSTEM' AND actor_user_id IS NULL "
        "AND service_principal = 'CHECKPOINT_RUNTIME') OR "
        "(actor_role IN ('CUSTOMER', 'ADMIN') AND actor_user_id IS NOT NULL "
        "AND service_principal IS NULL)",
    )

    op.add_column(
        "agent_content_source_revision",
        sa.Column("origin_chat_message_id", sa.BigInteger(), nullable=True),
    )
    op.execute(
        """
        CREATE TRIGGER trg_agent_content_source_revision_0007_backfill_guard
        BEFORE UPDATE ON agent_content_source_revision FOR EACH ROW
        BEGIN
            IF OLD.origin_chat_message_id IS NOT NULL
               OR NEW.origin_chat_message_id <> CAST(OLD.source_record_id AS UNSIGNED)
               OR NOT (NEW.id <=> OLD.id)
               OR NOT (NEW.source_kind <=> OLD.source_kind)
               OR NOT (NEW.source_record_id <=> OLD.source_record_id)
               OR NOT (NEW.source_revision <=> OLD.source_revision)
               OR NOT (NEW.content_role <=> OLD.content_role)
               OR NOT (NEW.content_schema_version <=> OLD.content_schema_version)
               OR NOT (NEW.normalization_version <=> OLD.normalization_version)
               OR NOT (NEW.content_sha256 <=> OLD.content_sha256)
               OR NOT (NEW.raw_content_utf8 <=> OLD.raw_content_utf8)
               OR NOT (NEW.conversation_id <=> OLD.conversation_id)
               OR NOT (NEW.subject_user_id <=> OLD.subject_user_id)
               OR NOT (NEW.producing_principal_kind <=> OLD.producing_principal_kind)
               OR NOT (NEW.producing_actor_id <=> OLD.producing_actor_id)
               OR NOT (NEW.producing_service_principal <=> OLD.producing_service_principal)
               OR NOT (NEW.run_id <=> OLD.run_id)
               OR NOT (NEW.producing_attempt_id <=> OLD.producing_attempt_id)
               OR NOT (NEW.created_at <=> OLD.created_at) THEN
                SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT =
                    '0007 source origin backfill changed an immutable field';
            END IF;
        END
        """
    )
    op.execute("DROP TRIGGER trg_agent_content_source_revision_append_only")
    op.execute(
        "UPDATE agent_content_source_revision "
        "SET origin_chat_message_id = CAST(source_record_id AS UNSIGNED)"
    )
    op.execute(
        "CREATE TRIGGER trg_agent_content_source_revision_append_only "
        "BEFORE UPDATE ON agent_content_source_revision FOR EACH ROW "
        "SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = "
        "'agent_content_source_revision is append-only'"
    )
    op.execute(
        "DROP TRIGGER trg_agent_content_source_revision_0007_backfill_guard"
    )
    op.alter_column(
        "agent_content_source_revision",
        "origin_chat_message_id",
        existing_type=sa.BigInteger(),
        nullable=False,
    )
    op.create_foreign_key(
        "fk_content_source_origin_chat_message",
        "agent_content_source_revision",
        "chat_message",
        ["origin_chat_message_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_index(
        "idx_content_source_origin_chat_message",
        "agent_content_source_revision",
        ["origin_chat_message_id"],
    )
    op.drop_constraint(
        "content_source_digest_length",
        "agent_content_source_revision",
        type_="check",
    )
    op.create_check_constraint(
        "content_source_digest_lower_hex",
        "agent_content_source_revision",
        "REGEXP_LIKE(content_sha256, '^[0-9a-f]{64}$', 'c')",
    )
    op.drop_constraint(
        "content_source_schema_version_positive",
        "agent_content_source_revision",
        type_="check",
    )
    op.create_check_constraint(
        "content_source_schema_version_v1",
        "agent_content_source_revision",
        "content_schema_version = 1",
    )
    op.create_check_constraint(
        "content_source_chat_origin_complete",
        "agent_content_source_revision",
        "source_kind = 'CHAT_MESSAGE' "
        "AND source_record_id = CAST(origin_chat_message_id AS CHAR) "
        "AND source_revision = 1",
    )
    op.drop_constraint(
        "checkpoint_content_ref_digest_length",
        "agent_checkpoint_content_reference",
        type_="check",
    )
    op.create_check_constraint(
        "checkpoint_content_ref_digest_lower_hex",
        "agent_checkpoint_content_reference",
        "REGEXP_LIKE(content_sha256, '^[0-9a-f]{64}$', 'c')",
    )
    op.create_check_constraint(
        "checkpoint_content_ref_slot_role_allowed",
        "agent_checkpoint_content_reference",
        _SLOT_ROLE_CHECK,
    )

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
    op.execute(
        "CREATE TRIGGER trg_agent_checkpoint_content_reference_no_update "
        "BEFORE UPDATE ON agent_checkpoint_content_reference FOR EACH ROW "
        "SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = "
        "'agent_checkpoint_content_reference is immutable'"
    )
    op.execute(
        "CREATE TRIGGER trg_agent_checkpoint_content_reference_no_delete "
        "BEFORE DELETE ON agent_checkpoint_content_reference FOR EACH ROW "
        "SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = "
        "'publication holds cannot be deleted before authority release'"
    )
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


def downgrade() -> None:
    raise RuntimeError(
        "0007 cannot be safely downgraded because origin authority and immutable "
        "publication holds protect recoverable content"
    )
