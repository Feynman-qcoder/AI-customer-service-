"""close nullable authority, owner, and effect-message safety gaps

Revision ID: 0009_persistence_safety_closeout
Revises: 0008_checkpoint_content_source_authority_repair
Create Date: 2026-09-28
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
from collections.abc import Mapping, Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0009_persistence_safety_closeout"
down_revision: str | None = "0008_checkpoint_content_source_authority_repair"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_CONTENT_DIGEST_DOMAIN = b"dianshang-agent/content-source-sha256/v1\x00"
_CHAT_MESSAGE_ROLE_PURPOSE: Mapping[str, tuple[str, str]] = {
    "QUESTION": ("USER", "USER_INPUT"),
    "EFFECTIVE_QUESTION": ("USER", "USER_INPUT"),
    "CURRENT_ISSUE": ("USER", "USER_INPUT"),
    "FINAL_ANSWER": ("ASSISTANT", "FINAL_ANSWER"),
}


def _lp64(data: bytes) -> bytes:
    return struct.pack(">Q", len(data)) + data


def _content_digest(row: Mapping[str, object], content: str) -> str:
    preimage = _CONTENT_DIGEST_DOMAIN
    preimage += _lp64(str(row["source_kind"]).encode("ascii"))
    preimage += _lp64(str(row["content_role"]).encode("ascii"))
    preimage += struct.pack(">Q", int(row["content_schema_version"]))
    preimage += _lp64(str(row["normalization_version"]).encode("ascii"))
    preimage += _lp64(content.encode("utf-8"))
    return hashlib.sha256(preimage).hexdigest()


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _message_payload_digest(row: Mapping[str, object]) -> str:
    retrieval_score = row["origin_retrieval_score"]
    return _canonical_digest(
        {
            "confidence_level": row["origin_confidence_level"],
            "content": row["origin_content"],
            "conversation_id": int(row["origin_conversation_id"]),
            "need_human": bool(row["origin_need_human"]),
            "retrieval_score": (
                str(retrieval_score) if retrieval_score is not None else None
            ),
            "role": row["origin_role"],
            "sources_json": row["origin_sources_json"],
        }
    )


def _effect_identity_digest(row: Mapping[str, object]) -> str:
    return _canonical_digest(
        {
            "node_name": row["effect_node_name"],
            "purpose": row["effect_purpose"],
            "run_id": row["effect_run_id"],
            "sequence": int(row["effect_sequence"]),
        }
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
        (
            r"^active_run\.plan\.missing_information\[[0-9]+\]$",
            "PLAN_MISSING_INFORMATION",
        ),
        (
            r"^active_run\.retrieval_evidence\[[0-9]+\]\.file_name$",
            "RETRIEVAL_FILE_NAME",
        ),
        (
            r"^active_run\.retrieval_evidence\[[0-9]+\]\.snippet$",
            "RETRIEVAL_SNIPPET",
        ),
        (
            r"^active_run\.response_meta\.sources\[[0-9]+\]\.file_name$",
            "RETRIEVAL_FILE_NAME",
        ),
        (
            r"^active_run\.response_meta\.sources\[[0-9]+\]\.snippet$",
            "RETRIEVAL_SNIPPET",
        ),
    )
    return any(
        expected == role and re.fullmatch(pattern, slot)
        for pattern, expected in indexed
    )


def _fail(message: str) -> None:
    raise RuntimeError(f"0009 safety preflight rejected existing data: {message}")


def _preflight_attempts(bind: sa.Connection) -> None:
    rows = bind.execute(
        sa.text(
            "SELECT a.attempt_id, a.actor_user_id, a.actor_role, "
            "a.service_principal, a.subject_user_id, "
            "c.user_id AS conversation_owner_id "
            "FROM agent_run_attempt a "
            "LEFT JOIN chat_conversation c ON c.id = a.conversation_id"
        )
    ).mappings()
    for row in rows:
        owner = row["conversation_owner_id"]
        role = row["actor_role"]
        actor_id = row["actor_user_id"]
        service = row["service_principal"]
        if owner is None or row["subject_user_id"] != owner:
            _fail("attempt conversation has no exact non-null owner/subject authority")
        if role == "SYSTEM":
            if actor_id is not None or service != "CHECKPOINT_RUNTIME":
                _fail("SYSTEM attempt has nullable or unregistered service authority")
        elif role in {"CUSTOMER", "ADMIN"}:
            if actor_id is None or service is not None:
                _fail("user attempt has incomplete actor authority")
            if role == "CUSTOMER" and actor_id != owner:
                _fail("customer attempt actor does not match conversation owner")
        else:
            _fail("attempt actor role is null or unregistered")


def _preflight_sources(bind: sa.Connection) -> None:
    rows = bind.execute(
        sa.text(
            "SELECT s.*, m.id AS origin_id, "
            "m.conversation_id AS origin_conversation_id, "
            "m.role AS origin_role, m.content AS origin_content, "
            "m.sources_json AS origin_sources_json, "
            "m.retrieval_score AS origin_retrieval_score, "
            "m.confidence_level AS origin_confidence_level, "
            "m.need_human AS origin_need_human, "
            "m.source_run_id AS origin_run_id, "
            "m.source_attempt_id AS origin_attempt_id, "
            "m.message_purpose AS origin_purpose, "
            "m.message_sequence AS origin_sequence, "
            "m.message_idempotency_key AS origin_idempotency_key, "
            "m.effect_id AS origin_effect_id, "
            "c.user_id AS conversation_owner_id, "
            "a.conversation_id AS attempt_conversation_id, "
            "a.subject_user_id AS attempt_subject_user_id, "
            "a.actor_user_id AS attempt_actor_user_id, "
            "a.actor_role AS attempt_actor_role, "
            "a.service_principal AS attempt_service_principal, "
            "e.id AS effect_id, e.run_id AS effect_run_id, "
            "e.attempt_id AS effect_attempt_id, "
            "e.node_name AS effect_node_name, "
            "e.purpose AS effect_purpose, e.sequence AS effect_sequence, "
            "e.effect_type AS effect_type, "
            "e.idempotency_key AS effect_idempotency_key, "
            "e.payload_digest AS effect_payload_digest "
            "FROM agent_content_source_revision s "
            "LEFT JOIN chat_message m ON m.id = s.origin_chat_message_id "
            "LEFT JOIN chat_conversation c ON c.id = s.conversation_id "
            "LEFT JOIN agent_run_attempt a "
            "ON a.attempt_id = s.producing_attempt_id AND a.run_id = s.run_id "
            "LEFT JOIN agent_effect e ON e.id = m.effect_id"
        )
    ).mappings()
    for row in rows:
        required = (
            "source_kind",
            "source_record_id",
            "source_revision",
            "content_role",
            "content_schema_version",
            "normalization_version",
            "content_sha256",
            "raw_content_utf8",
            "conversation_id",
            "subject_user_id",
            "producing_principal_kind",
            "run_id",
            "producing_attempt_id",
            "origin_chat_message_id",
            "origin_id",
            "origin_conversation_id",
            "origin_role",
            "origin_content",
            "origin_need_human",
            "origin_run_id",
            "origin_attempt_id",
            "origin_purpose",
            "origin_sequence",
            "origin_idempotency_key",
            "origin_effect_id",
            "conversation_owner_id",
            "attempt_conversation_id",
            "attempt_subject_user_id",
            "attempt_actor_role",
            "effect_id",
            "effect_run_id",
            "effect_attempt_id",
            "effect_node_name",
            "effect_purpose",
            "effect_sequence",
            "effect_type",
            "effect_idempotency_key",
            "effect_payload_digest",
        )
        if any(row[name] is None for name in required):
            _fail("source authority contains a null required binding")
        record_id = str(row["source_record_id"])
        if (
            row["source_kind"] != "CHAT_MESSAGE"
            or not record_id.isdecimal()
            or record_id.startswith("0")
            or int(record_id) <= 0
            or int(row["source_revision"]) != 1
            or int(row["origin_id"]) != int(record_id)
            or int(row["origin_chat_message_id"]) != int(record_id)
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
            _fail("source bytes/digest disagree with exact origin")
        expected = _CHAT_MESSAGE_ROLE_PURPOSE.get(str(row["content_role"]))
        if expected is None or expected != (row["origin_role"], row["origin_purpose"]):
            _fail("source role has no matching ChatMessage role/purpose authority")
        if (
            row["origin_conversation_id"] != row["conversation_id"]
            or row["conversation_owner_id"] != row["subject_user_id"]
            or row["attempt_conversation_id"] != row["conversation_id"]
            or row["attempt_subject_user_id"] != row["subject_user_id"]
            or row["origin_run_id"] != row["run_id"]
            or row["origin_attempt_id"] != row["producing_attempt_id"]
            or row["origin_effect_id"] != row["effect_id"]
            or row["effect_run_id"] != row["run_id"]
            or row["effect_attempt_id"] != row["producing_attempt_id"]
            or row["effect_purpose"] != row["origin_purpose"]
            or row["effect_type"] != "CHAT_MESSAGE"
            or row["effect_sequence"] != row["origin_sequence"]
            or row["effect_idempotency_key"] != row["origin_idempotency_key"]
            or row["effect_idempotency_key"] != _effect_identity_digest(row)
            or row["effect_payload_digest"] != _message_payload_digest(row)
        ):
            _fail("source origin/effect identity or payload digest is inconsistent")
        if row["origin_role"] == "ASSISTANT":
            if (
                row["producing_principal_kind"] != "SERVICE"
                or row["producing_actor_id"] is not None
                or row["producing_service_principal"] != "CHECKPOINT_RUNTIME"
            ):
                _fail("source SERVICE principal disagrees with immutable origin")
        elif (
            row["attempt_actor_role"] != "CUSTOMER"
            or row["attempt_actor_user_id"] != row["conversation_owner_id"]
            or row["producing_principal_kind"] != "USER"
            or row["producing_actor_id"] != row["attempt_actor_user_id"]
            or row["producing_service_principal"] is not None
        ):
            _fail("source USER principal disagrees with immutable origin")


def _preflight_holds(bind: sa.Connection) -> None:
    rows = bind.execute(
        sa.text(
            "SELECT h.*, p.publication_version AS current_publication_version, "
            "p.conversation_id AS publication_conversation_id, "
            "x.conversation_id AS execution_conversation_id, "
            "c.user_id AS conversation_owner_id, "
            "s.conversation_id AS source_conversation_id, "
            "s.subject_user_id AS source_subject_user_id, "
            "s.content_role AS source_content_role, "
            "s.content_sha256 AS source_content_sha256, "
            "s.origin_chat_message_id AS source_origin_id "
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
        required = (
            "thread_id",
            "publication_version",
            "reference_ordinal",
            "projection_slot",
            "content_role",
            "source_kind",
            "source_record_id",
            "source_revision",
            "content_sha256",
            "current_publication_version",
            "publication_conversation_id",
            "execution_conversation_id",
            "conversation_owner_id",
            "source_conversation_id",
            "source_subject_user_id",
            "source_content_role",
            "source_content_sha256",
            "source_origin_id",
        )
        if any(row[name] is None for name in required):
            _fail("publication hold contains a null authority binding")
        if (
            int(row["publication_version"])
            != int(row["current_publication_version"])
            or row["publication_conversation_id"]
            != row["execution_conversation_id"]
            or row["source_conversation_id"]
            != row["publication_conversation_id"]
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
    rows = list(
        op.get_bind().execute(
            sa.text(
                "SELECT tc.constraint_name "
                "FROM information_schema.table_constraints tc "
                "JOIN information_schema.check_constraints cc "
                "ON cc.constraint_schema = tc.constraint_schema "
                "AND cc.constraint_name = tc.constraint_name "
                "WHERE tc.constraint_schema = DATABASE() "
                "AND tc.table_name = :table "
                "AND tc.constraint_type = 'CHECK' "
                "AND LOWER(cc.check_clause) LIKE :column_pattern"
            ),
            {"table": table, "column_pattern": f"%{column.lower()}%"},
        )
    )
    for (constraint_name,) in rows:
        op.drop_constraint(op.f(str(constraint_name)), table, type_="check")


def _content_digest_sql(row: str) -> str:
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


def _json_nullable(value: str) -> str:
    return f"IF({value} IS NULL, 'null', JSON_QUOTE({value}))"


def _effect_identity_digest_sql() -> str:
    return (
        "LOWER(SHA2(CONCAT("
        "'{\"node_name\":', JSON_QUOTE(v_effect_node), "
        "',\"purpose\":', JSON_QUOTE(v_effect_purpose), "
        "',\"run_id\":', JSON_QUOTE(v_effect_run), "
        "',\"sequence\":', CAST(v_effect_sequence AS CHAR), '}'), 256))"
    )


def _message_payload_digest_sql() -> str:
    return (
        "LOWER(SHA2(CONCAT("
        "'{\"confidence_level\":', "
        f"{_json_nullable('v_confidence')}, "
        "',\"content\":', JSON_QUOTE(v_content), "
        "',\"conversation_id\":', CAST(v_conversation AS CHAR), "
        "',\"need_human\":', IF(v_need_human = 0, 'false', 'true'), "
        "',\"retrieval_score\":', "
        "IF(v_retrieval_score IS NULL, 'null', "
        "JSON_QUOTE(CAST(v_retrieval_score AS CHAR))), "
        "',\"role\":', JSON_QUOTE(v_role), "
        "',\"sources_json\":', "
        f"{_json_nullable('v_sources_json')}, '}}'), 256))"
    )


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
            DECLARE v_sources_json LONGTEXT;
            DECLARE v_retrieval_score DECIMAL(8,4);
            DECLARE v_confidence VARCHAR(32);
            DECLARE v_need_human BOOLEAN;
            DECLARE v_run VARCHAR(64);
            DECLARE v_attempt VARCHAR(64);
            DECLARE v_purpose VARCHAR(64);
            DECLARE v_message_sequence INTEGER;
            DECLARE v_message_key VARCHAR(64);
            DECLARE v_message_effect BIGINT;
            DECLARE v_owner BIGINT;
            DECLARE v_attempt_conversation BIGINT;
            DECLARE v_attempt_subject BIGINT;
            DECLARE v_attempt_actor BIGINT;
            DECLARE v_attempt_role VARCHAR(32);
            DECLARE v_attempt_service VARCHAR(64);
            DECLARE v_effect_id BIGINT;
            DECLARE v_effect_run VARCHAR(64);
            DECLARE v_effect_attempt VARCHAR(64);
            DECLARE v_effect_node VARCHAR(64);
            DECLARE v_effect_purpose VARCHAR(64);
            DECLARE v_effect_sequence INTEGER;
            DECLARE v_effect_type VARCHAR(32);
            DECLARE v_effect_key VARCHAR(64);
            DECLARE v_effect_payload_digest VARCHAR(64);
            DECLARE v_expected_effect_key CHAR(64);
            DECLARE v_expected_payload_digest CHAR(64);
            DECLARE v_expected_content_digest CHAR(64);
            DECLARE CONTINUE HANDLER FOR NOT FOUND SET v_missing = 1;

            SELECT m.conversation_id, m.role, m.content, m.sources_json,
                   m.retrieval_score, m.confidence_level, m.need_human,
                   m.source_run_id, m.source_attempt_id, m.message_purpose,
                   m.message_sequence, m.message_idempotency_key, m.effect_id,
                   c.user_id, a.conversation_id, a.subject_user_id,
                   a.actor_user_id, a.actor_role, a.service_principal,
                   e.id, e.run_id, e.attempt_id, e.node_name, e.purpose,
                   e.sequence, e.effect_type, e.idempotency_key, e.payload_digest
              INTO v_conversation, v_role, v_content, v_sources_json,
                   v_retrieval_score, v_confidence, v_need_human, v_run,
                   v_attempt, v_purpose, v_message_sequence, v_message_key,
                   v_message_effect, v_owner, v_attempt_conversation,
                   v_attempt_subject, v_attempt_actor, v_attempt_role,
                   v_attempt_service, v_effect_id, v_effect_run,
                   v_effect_attempt, v_effect_node, v_effect_purpose,
                   v_effect_sequence, v_effect_type, v_effect_key,
                   v_effect_payload_digest
              FROM chat_message m
              JOIN chat_conversation c ON c.id = m.conversation_id
              JOIN agent_run_attempt a
                ON a.attempt_id = m.source_attempt_id
               AND a.run_id = m.source_run_id
              JOIN agent_thread_execution x
                ON x.thread_id = a.thread_id
               AND x.conversation_id = a.conversation_id
              JOIN agent_effect e ON e.id = m.effect_id
             WHERE m.id = NEW.origin_chat_message_id
             LIMIT 1
             FOR UPDATE;

            IF v_missing = 1 OR v_owner IS NULL
               OR v_effect_id IS NULL OR v_message_effect IS NULL
               OR v_effect_node IS NULL OR v_effect_purpose IS NULL
               OR v_effect_sequence IS NULL OR v_effect_key IS NULL
               OR v_effect_payload_digest IS NULL THEN
                SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT =
                    'content source authority missing';
            END IF;
            IF NOT (NEW.source_kind <=> 'CHAT_MESSAGE')
               OR NOT (NEW.source_record_id <=> CAST(NEW.origin_chat_message_id AS CHAR))
               OR NOT (NEW.source_revision <=> 1) THEN
                SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT =
                    'content source origin identity invalid';
            END IF;
            IF NOT (v_conversation <=> NEW.conversation_id)
               OR NOT (v_attempt_conversation <=> NEW.conversation_id)
               OR NOT (v_owner <=> NEW.subject_user_id)
               OR NOT (v_attempt_subject <=> NEW.subject_user_id)
               OR NOT (v_run <=> NEW.run_id)
               OR NOT (v_attempt <=> NEW.producing_attempt_id)
               OR NOT (v_message_effect <=> v_effect_id)
               OR NOT (v_effect_run <=> v_run)
               OR NOT (v_effect_attempt <=> v_attempt)
               OR NOT (v_effect_purpose <=> v_purpose)
               OR NOT (v_effect_type <=> 'CHAT_MESSAGE')
               OR NOT (v_effect_sequence <=> v_message_sequence)
               OR NOT (v_effect_key <=> v_message_key) THEN
                SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT =
                    'content source origin/effect binding mismatch';
            END IF;
            SET v_expected_effect_key = {_effect_identity_digest_sql()};
            SET v_expected_payload_digest = {_message_payload_digest_sql()};
            IF NOT (v_effect_key <=> v_expected_effect_key)
               OR NOT (v_effect_payload_digest <=> v_expected_payload_digest) THEN
                SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT =
                    'content source effect identity or payload digest mismatch';
            END IF;
            IF OCTET_LENGTH(v_content) <> OCTET_LENGTH(NEW.raw_content_utf8)
               OR NOT (CONVERT(v_content USING binary) <=> NEW.raw_content_utf8) THEN
                SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT =
                    'content source origin bytes mismatch';
            END IF;
            IF v_attempt_role = 'SYSTEM' THEN
                IF v_attempt_actor IS NOT NULL
                   OR v_attempt_service IS NULL
                   OR NOT (v_attempt_service <=> 'CHECKPOINT_RUNTIME') THEN
                    SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT =
                        'content source execution actor invalid';
                END IF;
            ELSEIF v_attempt_role IN ('CUSTOMER', 'ADMIN') THEN
                IF v_attempt_actor IS NULL OR v_attempt_service IS NOT NULL THEN
                    SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT =
                        'content source execution actor invalid';
                END IF;
            ELSE
                SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT =
                    'content source execution actor invalid';
            END IF;

            IF v_role = 'ASSISTANT' AND v_purpose = 'FINAL_ANSWER' THEN
                IF NOT (NEW.content_role <=> 'FINAL_ANSWER')
                   OR NOT (NEW.producing_principal_kind <=> 'SERVICE')
                   OR NEW.producing_actor_id IS NOT NULL
                   OR NEW.producing_service_principal IS NULL
                   OR NOT (NEW.producing_service_principal <=> 'CHECKPOINT_RUNTIME') THEN
                    SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT =
                        'content source service principal mismatch';
                END IF;
            ELSEIF v_role = 'USER' AND v_purpose = 'USER_INPUT' THEN
                IF NEW.content_role NOT IN ('QUESTION', 'EFFECTIVE_QUESTION', 'CURRENT_ISSUE')
                   OR NOT (NEW.producing_principal_kind <=> 'USER')
                   OR v_attempt_role <> 'CUSTOMER'
                   OR NOT (v_attempt_actor <=> v_owner)
                   OR NOT (v_attempt_actor <=> NEW.producing_actor_id)
                   OR NEW.producing_service_principal IS NOT NULL THEN
                    SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT =
                        'content source user principal mismatch';
                END IF;
            ELSE
                SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT =
                    'content source role purpose mismatch';
            END IF;

            IF NEW.content_schema_version IS NULL
               OR NEW.content_schema_version <> 1 THEN
                SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT =
                    'content source schema version unsupported';
            END IF;
            SET v_expected_content_digest = {_content_digest_sql("NEW")};
            IF NOT (NEW.content_sha256 <=> v_expected_content_digest) THEN
                SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT =
                    'content source digest mismatch';
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

            IF NEW.thread_id IS NULL OR NEW.publication_version IS NULL
               OR NEW.projection_slot IS NULL OR NEW.content_role IS NULL
               OR NEW.source_kind IS NULL OR NEW.source_record_id IS NULL
               OR NEW.source_revision IS NULL OR NEW.content_sha256 IS NULL THEN
                SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT =
                    'publication hold authority is incomplete';
            END IF;

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
             LIMIT 1
             FOR UPDATE;

            IF v_missing = 1 OR v_origin IS NULL OR v_owner IS NULL THEN
                SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT =
                    'publication hold source domain missing';
            END IF;
            IF NOT (NEW.publication_version <=> v_publication_version) THEN
                SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT =
                    'publication hold version mismatch';
            END IF;
            IF NOT (v_publication_conversation <=> v_execution_conversation)
               OR NOT (v_source_conversation <=> v_publication_conversation)
               OR NOT (v_source_subject <=> v_owner) THEN
                SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT =
                    'publication hold authority mismatch';
            END IF;
        END
        """
    )


def _create_origin_immutability_triggers() -> None:
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
                OR NOT (NEW.sources_json <=> OLD.sources_json)
                OR NOT (NEW.retrieval_score <=> OLD.retrieval_score)
                OR NOT (NEW.confidence_level <=> OLD.confidence_level)
                OR NOT (NEW.need_human <=> OLD.need_human)
                OR NOT (NEW.source_run_id <=> OLD.source_run_id)
                OR NOT (NEW.source_attempt_id <=> OLD.source_attempt_id)
                OR NOT (NEW.message_purpose <=> OLD.message_purpose)
                OR NOT (NEW.message_sequence <=> OLD.message_sequence)
                OR NOT (NEW.message_idempotency_key <=> OLD.message_idempotency_key)
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
        CREATE TRIGGER trg_chat_conversation_content_owner_immutable
        BEFORE UPDATE ON chat_conversation FOR EACH ROW
        BEGIN
            IF NOT (NEW.user_id <=> OLD.user_id)
               AND EXISTS (
                   SELECT 1 FROM agent_content_source_revision s
                    WHERE s.conversation_id = OLD.id
               ) THEN
                SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT =
                    'content-source conversation owner is immutable';
            END IF;
        END
        """
    )


def upgrade() -> None:
    # MySQL DDL is non-transactional.  Validate every affected authority row
    # before the first DROP/CREATE so invalid stock leaves revision/schema intact.
    _preflight()

    _drop_checks_referencing_column("agent_run_attempt", "service_principal")
    op.create_check_constraint(
        "attempt_service_principal_authority",
        "agent_run_attempt",
        "actor_role IS NOT NULL AND ((actor_role = 'SYSTEM' "
        "AND actor_user_id IS NULL AND service_principal IS NOT NULL "
        "AND service_principal = 'CHECKPOINT_RUNTIME') OR "
        "(actor_role IN ('CUSTOMER', 'ADMIN') AND actor_user_id IS NOT NULL "
        "AND service_principal IS NULL))",
    )

    _drop_checks_referencing_column(
        "agent_content_source_revision",
        "producing_principal_kind",
    )
    op.create_check_constraint(
        "content_source_principal_complete",
        "agent_content_source_revision",
        "producing_principal_kind IS NOT NULL AND "
        "((producing_principal_kind = 'USER' "
        "AND producing_actor_id IS NOT NULL "
        "AND producing_service_principal IS NULL) OR "
        "(producing_principal_kind = 'SERVICE' "
        "AND producing_actor_id IS NULL "
        "AND producing_service_principal IS NOT NULL "
        "AND producing_service_principal = 'CHECKPOINT_RUNTIME'))",
    )

    for trigger in (
        "trg_agent_content_source_revision_validate_insert",
        "trg_agent_checkpoint_content_reference_validate_insert",
        "trg_chat_message_content_origin_immutable",
        "trg_chat_conversation_content_owner_immutable",
    ):
        op.execute(f"DROP TRIGGER IF EXISTS {trigger}")
    _create_source_insert_trigger()
    _create_hold_insert_trigger()
    _create_origin_immutability_triggers()


def downgrade() -> None:
    raise RuntimeError(
        "0009 cannot be safely downgraded because it closes nullable authority, "
        "owner locking, effect-message integrity, and retained-hold guarantees"
    )
