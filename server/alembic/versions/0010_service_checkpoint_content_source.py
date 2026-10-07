"""add the controlled CHECKPOINT_RUNTIME content-source branch

Revision ID: 0010_service_checkpoint_source
Revises: 0009_persistence_safety_closeout
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0010_service_checkpoint_source"
down_revision = "0009_persistence_safety_closeout"
branch_labels = None
depends_on = None


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


def _service_record_id_sql() -> str:
    return (
        "LOWER(SHA2(CONCAT("
        "_binary'dianshang-agent/checkpoint-service-source/v1', 0x00, "
        "CAST(NEW.conversation_id AS CHAR), 0x00, "
        "CONVERT(NEW.run_id USING binary), 0x00, "
        "CONVERT(NEW.content_role USING binary), 0x00, "
        "CONVERT(NEW.content_sha256 USING binary)), 256))"
    )


def _preflight() -> None:
    invalid = op.get_bind().scalar(
        sa.text(
            "SELECT COUNT(*) FROM agent_content_source_revision "
            "WHERE NOT (source_kind = 'CHAT_MESSAGE' "
            "AND origin_chat_message_id IS NOT NULL "
            "AND source_record_id = CAST(origin_chat_message_id AS CHAR) "
            "AND source_revision = 1)"
        )
    )
    if int(invalid or 0) != 0:
        raise RuntimeError(
            "0010 preflight rejected a source row without registered authority"
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
            DECLARE v_attempt_thread VARCHAR(128);
            DECLARE v_attempt_fence BIGINT;
            DECLARE v_execution_conversation BIGINT;
            DECLARE v_execution_owner VARCHAR(64);
            DECLARE v_execution_fence BIGINT;
            DECLARE v_execution_expiry DATETIME(6);
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
            DECLARE v_expected_service_record_id CHAR(64);
            DECLARE CONTINUE HANDLER FOR NOT FOUND SET v_missing = 1;

            IF NEW.source_kind <=> 'CHAT_MESSAGE' THEN
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
                IF NEW.origin_chat_message_id IS NULL
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

            ELSEIF NEW.source_kind <=> 'AGENT_AUDIT_CONTENT' THEN
                SELECT a.thread_id, a.conversation_id, a.subject_user_id,
                       a.fence_version, c.user_id, x.conversation_id,
                       x.owner_attempt_id, x.fence_version, x.lease_expires_at
                  INTO v_attempt_thread, v_attempt_conversation, v_attempt_subject,
                       v_attempt_fence, v_owner, v_execution_conversation,
                       v_execution_owner, v_execution_fence, v_execution_expiry
                  FROM agent_run_attempt a
                  JOIN agent_thread_execution x
                    ON x.thread_id = a.thread_id
                  JOIN chat_conversation c
                    ON c.id = a.conversation_id
                 WHERE a.attempt_id = NEW.producing_attempt_id
                   AND a.run_id = NEW.run_id
                 LIMIT 1
                 FOR UPDATE;
                IF v_missing = 1 OR v_owner IS NULL
                   OR v_attempt_fence IS NULL OR v_execution_expiry IS NULL THEN
                    SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT =
                        'service content source authority missing';
                END IF;
                IF NEW.origin_chat_message_id IS NOT NULL
                   OR NOT (NEW.source_revision <=> 1)
                   OR NOT (NEW.producing_principal_kind <=> 'SERVICE')
                   OR NEW.producing_actor_id IS NOT NULL
                   OR NOT (NEW.producing_service_principal <=> 'CHECKPOINT_RUNTIME')
                   OR NEW.content_role NOT IN (
                       'EFFECTIVE_QUESTION', 'PLAN_GOAL', 'PLAN_REASON',
                       'PLAN_MISSING_INFORMATION', 'CURRENT_ISSUE',
                       'RETRIEVAL_FILE_NAME', 'RETRIEVAL_SNIPPET', 'TOOL_CONTENT',
                       'DRAFT_ANSWER', 'FINAL_ANSWER', 'ERROR_DETAIL'
                   ) THEN
                    SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT =
                        'service content source branch invalid';
                END IF;
                IF NOT (v_attempt_conversation <=> NEW.conversation_id)
                   OR NOT (v_execution_conversation <=> NEW.conversation_id)
                   OR NOT (v_attempt_subject <=> NEW.subject_user_id)
                   OR NOT (v_owner <=> NEW.subject_user_id)
                   OR NOT (v_execution_owner <=> NEW.producing_attempt_id)
                   OR NOT (v_execution_fence <=> v_attempt_fence)
                   OR v_execution_expiry <= CURRENT_TIMESTAMP(6) THEN
                    SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT =
                        'service content source lost live execution authority';
                END IF;
                SET v_expected_service_record_id = {_service_record_id_sql()};
                IF NOT (
                    NEW.source_record_id <=> JSON_QUOTE(v_expected_service_record_id)
                ) THEN
                    SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT =
                        'service content source identity invalid';
                END IF;
            ELSE
                SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT =
                    'content source kind has no registered authority';
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


def upgrade() -> None:
    _preflight()
    op.execute("DROP TRIGGER IF EXISTS trg_agent_content_source_revision_validate_insert")
    op.drop_constraint(
        "content_source_chat_origin_complete",
        "agent_content_source_revision",
        type_="check",
    )
    op.alter_column(
        "agent_content_source_revision",
        "origin_chat_message_id",
        existing_type=sa.BigInteger(),
        nullable=True,
    )
    op.create_check_constraint(
        "content_source_chat_origin_complete",
        "agent_content_source_revision",
        "(source_kind = 'CHAT_MESSAGE' "
        "AND origin_chat_message_id IS NOT NULL "
        "AND source_record_id = CAST(origin_chat_message_id AS CHAR) "
        "AND source_revision = 1) OR "
        "(source_kind = 'AGENT_AUDIT_CONTENT' "
        "AND origin_chat_message_id IS NULL "
        "AND source_revision = 1 "
        "AND producing_principal_kind = 'SERVICE' "
        "AND producing_actor_id IS NULL "
        "AND producing_service_principal = 'CHECKPOINT_RUNTIME')",
    )
    _create_source_insert_trigger()


def downgrade() -> None:
    raise RuntimeError(
        "0010 cannot be downgraded without deleting valid service-produced source rows"
    )
