from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects import mysql
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class UserAccount(Base):
    __tablename__ = "user_account"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str] = mapped_column(String(128), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)


class ProductCatalog(Base):
    __tablename__ = "product_catalog"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    product_code: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    product_name: Mapped[str] = mapped_column(String(128), nullable=False)
    category: Mapped[str] = mapped_column(String(64), nullable=False)
    sale_status: Mapped[str] = mapped_column(String(32), nullable=False)
    price: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    stock_quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    dispatch_rule: Mapped[str] = mapped_column(String(255), nullable=False)
    after_sale_rule: Mapped[str] = mapped_column(String(512), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)


class CustomerOrder(Base):
    __tablename__ = "customer_order"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    order_no: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    user_id: Mapped[int] = mapped_column(ForeignKey("user_account.id"), nullable=False)
    product_id: Mapped[int] = mapped_column(ForeignKey("product_catalog.id"), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    amount: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    paid_at: Mapped[datetime | None] = mapped_column(DateTime)
    expected_ship_at: Mapped[datetime | None] = mapped_column(DateTime)
    shipped_at: Mapped[datetime | None] = mapped_column(DateTime)
    signed_at: Mapped[datetime | None] = mapped_column(DateTime)
    receiver_name: Mapped[str] = mapped_column(String(64), nullable=False)
    receiver_phone: Mapped[str] = mapped_column(String(32), nullable=False)
    receiver_address: Mapped[str] = mapped_column(String(255), nullable=False)
    remark: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)

    product: Mapped[ProductCatalog] = relationship(lazy="joined")


class ShipmentEvent(Base):
    __tablename__ = "shipment_event"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("customer_order.id"), nullable=False)
    carrier: Mapped[str | None] = mapped_column(String(64))
    tracking_no: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    location: Mapped[str | None] = mapped_column(String(128))
    event_note: Mapped[str] = mapped_column(String(255), nullable=False)
    event_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now)


class ChatConversation(Base):
    __tablename__ = "chat_conversation"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("user_account.id"))
    conversation_no: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)

    __table_args__ = (
        UniqueConstraint("id", "user_id", name="uk_chat_conversation_owner_scope"),
    )


class ChatMessage(Base):
    __tablename__ = "chat_message"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("chat_conversation.id"), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    sources_json: Mapped[str | None] = mapped_column(Text)
    retrieval_score: Mapped[Decimal | None] = mapped_column(Numeric(8, 4))
    confidence_level: Mapped[str | None] = mapped_column(String(32))
    need_human: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    source_run_id: Mapped[str | None] = mapped_column(String(64))
    source_attempt_id: Mapped[str | None] = mapped_column(String(64))
    message_purpose: Mapped[str | None] = mapped_column(String(64))
    message_sequence: Mapped[int | None] = mapped_column(Integer)
    message_idempotency_key: Mapped[str | None] = mapped_column(String(64))
    effect_id: Mapped[int | None] = mapped_column(ForeignKey("agent_effect.id", name="fk_chat_message_effect"))
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now)

    __table_args__ = (
        ForeignKeyConstraint(
            ["source_attempt_id", "source_run_id"],
            ["agent_run_attempt.attempt_id", "agent_run_attempt.run_id"],
            name="fk_chat_message_attempt_run",
        ),
        UniqueConstraint("id", "conversation_id", name="uk_chat_message_conversation_cursor"),
        UniqueConstraint("message_idempotency_key", name="uk_chat_message_idempotency"),
        UniqueConstraint("effect_id", name="uk_chat_message_effect"),
        CheckConstraint(
            "message_sequence IS NULL OR message_sequence >= 0",
            name="message_sequence_nonnegative",
        ),
        CheckConstraint(
            "("
            "effect_id IS NULL AND source_run_id IS NULL AND source_attempt_id IS NULL "
            "AND message_purpose IS NULL AND message_sequence IS NULL "
            "AND message_idempotency_key IS NULL"
            ") OR ("
            "effect_id IS NOT NULL AND source_run_id IS NOT NULL AND source_attempt_id IS NOT NULL "
            "AND message_purpose IS NOT NULL AND message_sequence IS NOT NULL "
            "AND message_idempotency_key IS NOT NULL"
            ")",
            name="chat_message_effect_complete",
        ),
    )


class KbDocument(Base):
    __tablename__ = "kb_document"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    original_name: Mapped[str] = mapped_column(String(512), nullable=False)
    storage_name: Mapped[str] = mapped_column(String(128), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    file_type: Mapped[str] = mapped_column(String(32), nullable=False)
    file_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    file_sha256: Mapped[str | None] = mapped_column(String(64), unique=True)
    uploaded_by: Mapped[int | None] = mapped_column(ForeignKey("user_account.id"))
    lock_version: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failure_reason: Mapped[str | None] = mapped_column(String(512))
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)


class DocumentProcessingTask(Base):
    __tablename__ = "document_processing_task"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("kb_document.id"), unique=True, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    next_retry_at: Mapped[datetime | None] = mapped_column(DateTime)
    started_at: Mapped[datetime | None] = mapped_column(DateTime)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)
    error_message: Mapped[str | None] = mapped_column(String(1000))
    lock_version: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)


class KbChunk(Base):
    __tablename__ = "kb_chunk"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("kb_document.id"), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False)
    vector_point_id: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)
    __table_args__ = (UniqueConstraint("document_id", "chunk_index", name="uk_kb_chunk_document_index"),)


class ChatMessageSource(Base):
    __tablename__ = "chat_message_source"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    message_id: Mapped[int] = mapped_column(ForeignKey("chat_message.id"), nullable=False)
    document_id: Mapped[int] = mapped_column(ForeignKey("kb_document.id"), nullable=False)
    chunk_id: Mapped[int | None] = mapped_column(ForeignKey("kb_chunk.id"))
    rank_no: Mapped[int] = mapped_column(Integer, nullable=False)
    retrieval_score: Mapped[Decimal] = mapped_column(Numeric(8, 4), nullable=False)
    snippet_snapshot: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now)


class SupportTicket(Base):
    __tablename__ = "support_ticket"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("user_account.id"))
    ticket_no: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("chat_conversation.id"), nullable=False)
    category: Mapped[str] = mapped_column(String(32), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    contact: Mapped[str | None] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    handler_id: Mapped[int | None] = mapped_column(ForeignKey("user_account.id"))
    priority: Mapped[str] = mapped_column(String(32), nullable=False, default="NORMAL")
    handling_note: Mapped[str | None] = mapped_column(Text)
    resolution: Mapped[str | None] = mapped_column(Text)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime)
    lock_version: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)


class TicketOperationLog(Base):
    __tablename__ = "ticket_operation_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    ticket_id: Mapped[int] = mapped_column(ForeignKey("support_ticket.id"), nullable=False)
    operator_id: Mapped[int] = mapped_column(ForeignKey("user_account.id"), nullable=False)
    previous_status: Mapped[str | None] = mapped_column(String(32))
    next_status: Mapped[str] = mapped_column(String(32), nullable=False)
    operation_note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now)


class ModelRuntimeConfig(Base):
    __tablename__ = "model_runtime_config"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    temperature: Mapped[Decimal] = mapped_column(Numeric(3, 2), nullable=False)
    top_k: Mapped[int] = mapped_column(Integer, nullable=False)
    min_retrieval_score: Mapped[Decimal] = mapped_column(Numeric(4, 3), nullable=False, default=Decimal("0.350"))
    mock_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now)


class AfterSaleRuleVersion(Base):
    __tablename__ = "after_sale_rule_version"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    version_code: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    description: Mapped[str] = mapped_column(String(255), nullable=False)
    effective_from: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    effective_to: Mapped[datetime | None] = mapped_column(DateTime)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now)


class AfterSaleRule(Base):
    __tablename__ = "after_sale_rule"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    rule_code: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    version_id: Mapped[int] = mapped_column(ForeignKey("after_sale_rule_version.id"), nullable=False)
    title: Mapped[str] = mapped_column(String(128), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    after_sale_type: Mapped[str] = mapped_column(String(32), nullable=False)
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    effective_from: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    effective_to: Mapped[datetime | None] = mapped_column(DateTime)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)

    version: Mapped[AfterSaleRuleVersion] = relationship(lazy="joined")


class AfterSaleRuleCondition(Base):
    __tablename__ = "after_sale_rule_condition"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    rule_id: Mapped[int] = mapped_column(ForeignKey("after_sale_rule.id"), nullable=False)
    product_category: Mapped[str | None] = mapped_column(String(64))
    order_status: Mapped[str | None] = mapped_column(String(32))
    payment_status: Mapped[str | None] = mapped_column(String(32))
    shipment_status: Mapped[str | None] = mapped_column(String(32))
    signed_within_days: Mapped[int | None] = mapped_column(Integer)
    after_sale_type: Mapped[str | None] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now)

    rule: Mapped[AfterSaleRule] = relationship(lazy="joined")


class AgentRun(Base):
    __tablename__ = "agent_run"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    thread_id: Mapped[str] = mapped_column(String(128), nullable=False)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("chat_conversation.id"), nullable=False)
    user_id: Mapped[int] = mapped_column(ForeignKey("user_account.id"), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    intent: Mapped[str | None] = mapped_column(String(64))
    risk_level: Mapped[str | None] = mapped_column(String(32))
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime)
    final_answer: Mapped[str | None] = mapped_column(Text)
    error_type: Mapped[str | None] = mapped_column(String(64))
    request_id: Mapped[str] = mapped_column(String(64), nullable=False)
    model_name: Mapped[str | None] = mapped_column(String(128))
    config_version: Mapped[str | None] = mapped_column(String(64))
    prompt_version: Mapped[str | None] = mapped_column(String(64))
    provider_latency_ms: Mapped[int | None] = mapped_column(Integer)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    completion_tokens: Mapped[int | None] = mapped_column(Integer)


class AgentRunAttempt(Base):
    __tablename__ = "agent_run_attempt"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    attempt_id: Mapped[str] = mapped_column(String(64), nullable=False)
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_run.run_id", name="fk_attempt_run"), nullable=False)
    thread_id: Mapped[str] = mapped_column(String(128), nullable=False)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("chat_conversation.id", name="fk_attempt_conversation"),
        nullable=False,
    )
    actor_user_id: Mapped[int | None] = mapped_column(ForeignKey("user_account.id", name="fk_attempt_actor"))
    actor_role: Mapped[str] = mapped_column(String(32), nullable=False)
    service_principal: Mapped[str | None] = mapped_column(
        String(64, collation="ascii_bin")
    )
    subject_user_id: Mapped[int] = mapped_column(
        ForeignKey("user_account.id", name="fk_attempt_subject"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    fence_version: Mapped[int | None] = mapped_column(BigInteger)
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime)
    error_type: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)

    __table_args__ = (
        UniqueConstraint("attempt_id", name="uk_agent_run_attempt_id"),
        UniqueConstraint("attempt_id", "run_id", name="uk_agent_run_attempt_run"),
        CheckConstraint(
            "fence_version IS NULL OR fence_version > 0",
            name="attempt_fence_positive",
        ),
        CheckConstraint(
            "status IN ('REGISTERED', 'ACTIVE', 'RELEASED', 'COMPLETED', 'FAILED')",
            name="attempt_status_allowed",
        ),
        CheckConstraint(
            "actor_role IS NOT NULL AND ((actor_role = 'SYSTEM' "
            "AND actor_user_id IS NULL AND service_principal IS NOT NULL "
            "AND service_principal = 'CHECKPOINT_RUNTIME') OR "
            "(actor_role IN ('CUSTOMER', 'ADMIN') AND actor_user_id IS NOT NULL "
            "AND service_principal IS NULL))",
            name="attempt_service_principal_authority",
        ),
    )


class AgentThreadExecution(Base):
    __tablename__ = "agent_thread_execution"

    thread_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("chat_conversation.id", name="fk_thread_conversation"),
        nullable=False,
        unique=True,
    )
    owner_attempt_id: Mapped[str | None] = mapped_column(
        ForeignKey("agent_run_attempt.attempt_id", name="fk_thread_owner_attempt")
    )
    fence_version: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)

    __table_args__ = (
        CheckConstraint("fence_version >= 0", name="thread_fence_nonnegative"),
        CheckConstraint(
            "(owner_attempt_id IS NULL AND lease_expires_at IS NULL) "
            "OR (owner_attempt_id IS NOT NULL AND lease_expires_at IS NOT NULL)",
            name="thread_lease_complete",
        ),
    )


class AgentCheckpointPublication(Base):
    __tablename__ = "agent_checkpoint_publication"

    thread_id: Mapped[str] = mapped_column(
        ForeignKey("agent_thread_execution.thread_id", name="fk_publication_thread"),
        primary_key=True,
    )
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("chat_conversation.id", name="fk_publication_conversation"),
        nullable=False,
        unique=True,
    )
    publication_version: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    logical_namespace: Mapped[str | None] = mapped_column(String(128))
    physical_namespace: Mapped[str | None] = mapped_column(String(255))
    checkpoint_id: Mapped[str | None] = mapped_column(String(128))
    checkpoint_digest: Mapped[str | None] = mapped_column(String(64))
    previous_logical_namespace: Mapped[str | None] = mapped_column(String(128))
    previous_physical_namespace: Mapped[str | None] = mapped_column(String(255))
    previous_checkpoint_id: Mapped[str | None] = mapped_column(String(128))
    manifest_root: Mapped[str | None] = mapped_column(String(64))
    manifest_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)

    __table_args__ = (
        CheckConstraint("publication_version >= 0", name="pub_version_nonnegative"),
        CheckConstraint("manifest_count >= 0", name="pub_manifest_count_nonnegative"),
        CheckConstraint(
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
        CheckConstraint(
            "("
            "previous_logical_namespace IS NULL AND previous_physical_namespace IS NULL "
            "AND previous_checkpoint_id IS NULL"
            ") OR ("
            "previous_logical_namespace IS NOT NULL AND previous_physical_namespace IS NOT NULL "
            "AND previous_checkpoint_id IS NOT NULL"
            ")",
            name="pub_previous_complete",
        ),
        CheckConstraint(
            "checkpoint_digest IS NULL OR CHAR_LENGTH(checkpoint_digest) = 64",
            name="pub_checkpoint_digest_length",
        ),
        CheckConstraint(
            "manifest_root IS NULL OR CHAR_LENGTH(manifest_root) = 64",
            name="pub_manifest_root_length",
        ),
    )


class AgentCheckpointWriteManifest(Base):
    __tablename__ = "agent_checkpoint_write_manifest"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    thread_id: Mapped[str] = mapped_column(
        ForeignKey("agent_checkpoint_publication.thread_id", name="fk_manifest_publication"),
        nullable=False,
    )
    publication_version: Mapped[int] = mapped_column(BigInteger, nullable=False)
    logical_namespace: Mapped[str] = mapped_column(String(128), nullable=False)
    physical_namespace: Mapped[str] = mapped_column(String(255), nullable=False)
    checkpoint_id: Mapped[str] = mapped_column(String(128), nullable=False)
    task_id: Mapped[str] = mapped_column(String(128), nullable=False)
    write_index: Mapped[int] = mapped_column(Integer, nullable=False)
    channel: Mapped[str] = mapped_column(String(128), nullable=False)
    content_digest: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now)

    __table_args__ = (
        UniqueConstraint(
            "thread_id",
            "publication_version",
            "task_id",
            "write_index",
            name="uk_checkpoint_manifest_item",
        ),
        CheckConstraint("publication_version > 0", name="manifest_version_positive"),
        CheckConstraint("CHAR_LENGTH(content_digest) = 64", name="manifest_digest_length"),
    )


class AgentContentSourceRevision(Base):
    __tablename__ = "agent_content_source_revision"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    source_kind: Mapped[str] = mapped_column(String(64, collation="ascii_bin"), nullable=False)
    source_record_id: Mapped[str] = mapped_column(String(130, collation="ascii_bin"), nullable=False)
    source_revision: Mapped[int] = mapped_column(BigInteger, nullable=False)
    content_role: Mapped[str] = mapped_column(String(64, collation="ascii_bin"), nullable=False)
    content_schema_version: Mapped[int] = mapped_column(Integer, nullable=False)
    normalization_version: Mapped[str] = mapped_column(String(32, collation="ascii_bin"), nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64, collation="ascii_bin"), nullable=False)
    raw_content_utf8: Mapped[bytes] = mapped_column(mysql.LONGBLOB, nullable=False)
    conversation_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    subject_user_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    producing_principal_kind: Mapped[str] = mapped_column(String(16, collation="ascii_bin"), nullable=False)
    producing_actor_id: Mapped[int | None] = mapped_column(BigInteger)
    producing_service_principal: Mapped[str | None] = mapped_column(
        String(128, collation="ascii_bin")
    )
    run_id: Mapped[str] = mapped_column(String(64), nullable=False)
    producing_attempt_id: Mapped[str] = mapped_column(String(64), nullable=False)
    origin_chat_message_id: Mapped[int | None] = mapped_column(BigInteger)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now)

    __table_args__ = (
        ForeignKeyConstraint(
            ["conversation_id"],
            ["chat_conversation.id"],
            name="fk_content_source_conversation",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["subject_user_id"],
            ["user_account.id"],
            name="fk_content_source_subject",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["producing_actor_id"],
            ["user_account.id"],
            name="fk_content_source_actor",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["producing_attempt_id", "run_id"],
            ["agent_run_attempt.attempt_id", "agent_run_attempt.run_id"],
            name="fk_content_source_attempt_run",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["origin_chat_message_id"],
            ["chat_message.id"],
            name="fk_content_source_origin_chat_message",
            ondelete="RESTRICT",
        ),
        UniqueConstraint(
            "source_kind",
            "source_record_id",
            "source_revision",
            name="uk_content_source_exact_revision",
        ),
        UniqueConstraint(
            "source_kind",
            "source_record_id",
            "source_revision",
            "content_role",
            "content_sha256",
            name="uk_content_source_hold_binding",
        ),
        UniqueConstraint(
            "source_kind",
            "source_record_id",
            "source_revision",
            "content_role",
            "content_sha256",
            "conversation_id",
            "subject_user_id",
            name="uk_content_source_memory_binding",
        ),
        Index(
            "idx_content_source_origin_chat_message",
            "origin_chat_message_id",
        ),
        CheckConstraint("source_revision > 0", name="content_source_revision_positive"),
        CheckConstraint(
            "content_schema_version = 1",
            name="content_source_schema_version_v1",
        ),
        CheckConstraint(
            "source_kind IN ('CHAT_MESSAGE', 'AGENT_AUDIT_CONTENT', 'ACTION_RECORD', 'RAG_DOCUMENT')",
            name="content_source_kind_allowed",
        ),
        CheckConstraint(
            "content_role IN ('QUESTION', 'EFFECTIVE_QUESTION', 'PLAN_GOAL', 'PLAN_REASON', "
            "'PLAN_MISSING_INFORMATION', 'CURRENT_ISSUE', 'CONVERSATION_SUMMARY', "
            "'RETRIEVAL_FILE_NAME', 'RETRIEVAL_SNIPPET', 'TOOL_CONTENT', 'DRAFT_ANSWER', "
            "'FINAL_ANSWER', 'ERROR_DETAIL')",
            name="content_source_role_allowed",
        ),
        CheckConstraint(
            "normalization_version = 'RAW_UTF8_V1'",
            name="content_source_normalization_allowed",
        ),
        CheckConstraint(
            "REGEXP_LIKE(content_sha256, '^[0-9a-f]{64}$', 'c')",
            name="content_source_digest_lower_hex",
        ),
        CheckConstraint(
            "producing_principal_kind IS NOT NULL AND "
            "((producing_principal_kind = 'USER' "
            "AND producing_actor_id IS NOT NULL "
            "AND producing_service_principal IS NULL) OR "
            "(producing_principal_kind = 'SERVICE' "
            "AND producing_actor_id IS NULL "
            "AND producing_service_principal IS NOT NULL "
            "AND producing_service_principal = 'CHECKPOINT_RUNTIME'))",
            name="content_source_principal_complete",
        ),
        CheckConstraint(
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
            name="content_source_chat_origin_complete",
        ),
    )


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


class ConversationWorkingMemory(Base):
    __tablename__ = "conversation_working_memory"

    conversation_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    subject_user_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    active_order_no: Mapped[str | None] = mapped_column(String(256))
    active_order_no_provenance: Mapped[dict[str, object] | None] = mapped_column(
        mysql.JSON(none_as_null=True)
    )
    active_product_code: Mapped[str | None] = mapped_column(String(256))
    active_product_code_provenance: Mapped[dict[str, object] | None] = mapped_column(
        mysql.JSON(none_as_null=True)
    )
    current_issue: Mapped[str | None] = mapped_column(String(4000))
    current_issue_provenance: Mapped[dict[str, object] | None] = mapped_column(
        mysql.JSON(none_as_null=True)
    )
    last_intent: Mapped[str | None] = mapped_column(String(64, collation="ascii_bin"))
    last_intent_provenance: Mapped[dict[str, object] | None] = mapped_column(
        mysql.JSON(none_as_null=True)
    )
    memory_revision: Mapped[int] = mapped_column(BigInteger, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.now,
        onupdate=datetime.now,
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["conversation_id", "subject_user_id"],
            ["chat_conversation.id", "chat_conversation.user_id"],
            name="fk_working_memory_conversation_owner",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["subject_user_id"],
            ["user_account.id"],
            name="fk_working_memory_subject",
            ondelete="RESTRICT",
        ),
        Index("idx_working_memory_subject_conversation", "subject_user_id", "conversation_id"),
        CheckConstraint("memory_revision > 0", name="working_memory_revision_positive"),
        CheckConstraint(
            "last_intent IS NULL OR "
            "REGEXP_LIKE(last_intent, '^[A-Z][A-Z0-9_]{0,63}$', 'c')",
            name="working_memory_last_intent_normalized",
        ),
        CheckConstraint(
            "(active_order_no IS NULL AND active_order_no_provenance IS NULL) OR "
            "(active_order_no IS NOT NULL AND active_order_no_provenance IS NOT NULL)",
            name="working_memory_active_order_pair",
        ),
        CheckConstraint(
            "(active_product_code IS NULL AND active_product_code_provenance IS NULL) OR "
            "(active_product_code IS NOT NULL AND active_product_code_provenance IS NOT NULL)",
            name="working_memory_active_product_pair",
        ),
        CheckConstraint(
            "(current_issue IS NULL AND current_issue_provenance IS NULL) OR "
            "(current_issue IS NOT NULL AND current_issue_provenance IS NOT NULL)",
            name="working_memory_current_issue_pair",
        ),
        CheckConstraint(
            "(last_intent IS NULL AND last_intent_provenance IS NULL) OR "
            "(last_intent IS NOT NULL AND last_intent_provenance IS NOT NULL)",
            name="working_memory_last_intent_pair",
        ),
        CheckConstraint(
            _memory_provenance_check("active_order_no_provenance"),
            name="working_memory_active_order_provenance_shape",
        ),
        CheckConstraint(
            _memory_provenance_check("active_product_code_provenance"),
            name="working_memory_active_product_provenance_shape",
        ),
        CheckConstraint(
            _memory_provenance_check("current_issue_provenance"),
            name="working_memory_current_issue_provenance_shape",
        ),
        CheckConstraint(
            _memory_provenance_check("last_intent_provenance"),
            name="working_memory_last_intent_provenance_shape",
        ),
    )


class ConversationRollingSummary(Base):
    __tablename__ = "conversation_rolling_summary"

    conversation_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    subject_user_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    source_kind: Mapped[str] = mapped_column(String(64, collation="ascii_bin"), nullable=False)
    source_record_id: Mapped[str] = mapped_column(String(130, collation="ascii_bin"), nullable=False)
    source_revision: Mapped[int] = mapped_column(BigInteger, nullable=False)
    content_role: Mapped[str] = mapped_column(String(64, collation="ascii_bin"), nullable=False)
    content_schema_version: Mapped[int] = mapped_column(Integer, nullable=False)
    normalization_version: Mapped[str] = mapped_column(String(32, collation="ascii_bin"), nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64, collation="ascii_bin"), nullable=False)
    summary_until_message_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    summary_revision: Mapped[int] = mapped_column(BigInteger, nullable=False)
    token_counter_version: Mapped[str] = mapped_column(
        String(64, collation="ascii_bin"),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.now,
        onupdate=datetime.now,
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["conversation_id", "subject_user_id"],
            ["chat_conversation.id", "chat_conversation.user_id"],
            name="fk_rolling_summary_conversation_owner",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["subject_user_id"],
            ["user_account.id"],
            name="fk_rolling_summary_subject",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["summary_until_message_id", "conversation_id"],
            ["chat_message.id", "chat_message.conversation_id"],
            name="fk_rolling_summary_cursor_conversation",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
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
        Index("idx_rolling_summary_subject_conversation", "subject_user_id", "conversation_id"),
        Index("idx_rolling_summary_cursor", "summary_until_message_id"),
        CheckConstraint("summary_revision > 0", name="rolling_summary_revision_positive"),
        CheckConstraint("source_revision > 0", name="rolling_summary_source_revision_positive"),
        CheckConstraint(
            "source_kind = 'AGENT_AUDIT_CONTENT' AND "
            "content_role = 'CONVERSATION_SUMMARY' AND content_schema_version = 1 AND "
            "normalization_version = 'RAW_UTF8_V1'",
            name="rolling_summary_source_role_v1",
        ),
        CheckConstraint(
            "REGEXP_LIKE(content_sha256, '^[0-9a-f]{64}$', 'c')",
            name="rolling_summary_digest_lower_hex",
        ),
        CheckConstraint(
            "summary_until_message_id > 0",
            name="rolling_summary_cursor_positive",
        ),
        CheckConstraint(
            "token_counter_version = 'UTF8_BYTES_CEIL_DIV_3_V1'",
            name="rolling_summary_counter_version_v1",
        ),
    )


class AgentCheckpointContentReference(Base):
    __tablename__ = "agent_checkpoint_content_reference"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    thread_id: Mapped[str] = mapped_column(String(128), nullable=False)
    publication_version: Mapped[int] = mapped_column(BigInteger, nullable=False)
    reference_ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    projection_slot: Mapped[str] = mapped_column(String(255, collation="ascii_bin"), nullable=False)
    content_role: Mapped[str] = mapped_column(String(64, collation="ascii_bin"), nullable=False)
    source_kind: Mapped[str] = mapped_column(String(64, collation="ascii_bin"), nullable=False)
    source_record_id: Mapped[str] = mapped_column(String(130, collation="ascii_bin"), nullable=False)
    source_revision: Mapped[int] = mapped_column(BigInteger, nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64, collation="ascii_bin"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now)

    __table_args__ = (
        ForeignKeyConstraint(
            ["thread_id"],
            ["agent_checkpoint_publication.thread_id"],
            name="fk_checkpoint_content_ref_publication",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
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
        UniqueConstraint(
            "thread_id",
            "publication_version",
            "projection_slot",
            name="uk_checkpoint_content_ref_slot",
        ),
        UniqueConstraint(
            "thread_id",
            "publication_version",
            "content_role",
            name="uk_checkpoint_content_ref_role",
        ),
        UniqueConstraint(
            "thread_id",
            "publication_version",
            "reference_ordinal",
            name="uk_checkpoint_content_ref_ordinal",
        ),
        CheckConstraint(
            "publication_version > 0",
            name="checkpoint_content_ref_version_positive",
        ),
        CheckConstraint(
            "reference_ordinal >= 0",
            name="checkpoint_content_ref_ordinal_nonnegative",
        ),
        CheckConstraint(
            "source_revision > 0",
            name="checkpoint_content_ref_revision_positive",
        ),
        CheckConstraint(
            "REGEXP_LIKE(content_sha256, '^[0-9a-f]{64}$', 'c')",
            name="checkpoint_content_ref_digest_lower_hex",
        ),
        CheckConstraint(
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
            "AND content_role = 'RETRIEVAL_SNIPPET')",
            name="checkpoint_content_ref_slot_role_allowed",
        ),
    )


class AgentEffect(Base):
    __tablename__ = "agent_effect"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False)
    attempt_id: Mapped[str] = mapped_column(String(64), nullable=False)
    node_name: Mapped[str] = mapped_column(String(64), nullable=False)
    purpose: Mapped[str] = mapped_column(String(64), nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    effect_type: Mapped[str] = mapped_column(String(32), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(64), nullable=False)
    payload_digest: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now)

    __table_args__ = (
        ForeignKeyConstraint(
            ["attempt_id", "run_id"],
            ["agent_run_attempt.attempt_id", "agent_run_attempt.run_id"],
            name="fk_agent_effect_attempt_run",
        ),
        UniqueConstraint(
            "run_id",
            "node_name",
            "purpose",
            "sequence",
            name="uk_agent_effect_identity",
        ),
        UniqueConstraint("idempotency_key", name="uk_agent_effect_idempotency"),
        CheckConstraint("sequence >= 0", name="effect_sequence_nonnegative"),
        CheckConstraint(
            "effect_type IN ('CHAT_MESSAGE', 'ACTION_PREPARE', 'LOCAL_AUDIT')",
            name="effect_type_allowed",
        ),
        CheckConstraint("CHAR_LENGTH(idempotency_key) = 64", name="effect_key_length"),
        CheckConstraint("CHAR_LENGTH(payload_digest) = 64", name="effect_digest_length"),
    )


class AgentStep(Base):
    __tablename__ = "agent_step"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False)
    node_name: Mapped[str] = mapped_column(String(64), nullable=False)
    input_summary: Mapped[str | None] = mapped_column(Text)
    output_summary: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_summary: Mapped[str | None] = mapped_column(Text)
    attempt_id: Mapped[str | None] = mapped_column(String(64))
    effect_id: Mapped[int | None] = mapped_column(ForeignKey("agent_effect.id", name="fk_agent_step_effect"))
    effect_purpose: Mapped[str | None] = mapped_column(String(64))
    effect_sequence: Mapped[int | None] = mapped_column(Integer)
    effect_idempotency_key: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now)

    __table_args__ = (
        ForeignKeyConstraint(
            ["attempt_id", "run_id"],
            ["agent_run_attempt.attempt_id", "agent_run_attempt.run_id"],
            name="fk_agent_step_attempt_run",
        ),
        UniqueConstraint("effect_id", name="uk_agent_step_effect"),
        UniqueConstraint("effect_idempotency_key", name="uk_agent_step_effect_idempotency"),
        CheckConstraint(
            "effect_sequence IS NULL OR effect_sequence >= 0",
            name="agent_step_effect_sequence_nonnegative",
        ),
        CheckConstraint(
            "("
            "effect_id IS NULL AND attempt_id IS NULL AND effect_purpose IS NULL "
            "AND effect_sequence IS NULL AND effect_idempotency_key IS NULL"
            ") OR ("
            "effect_id IS NOT NULL AND attempt_id IS NOT NULL AND effect_purpose IS NOT NULL "
            "AND effect_sequence IS NOT NULL AND effect_idempotency_key IS NOT NULL"
            ")",
            name="agent_step_effect_complete",
        ),
    )


class AgentToolCall(Base):
    __tablename__ = "agent_tool_call"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False)
    tool_name: Mapped[str] = mapped_column(String(128), nullable=False)
    redacted_arguments_json: Mapped[str] = mapped_column(Text, nullable=False)
    result_summary: Mapped[str | None] = mapped_column(Text)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now)


class AgentActionRequest(Base):
    __tablename__ = "agent_action_request"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False)
    action_type: Mapped[str] = mapped_column(String(64), nullable=False)
    target_order_id: Mapped[int | None] = mapped_column(ForeignKey("customer_order.id"))
    action_payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    lock_version: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_by: Mapped[int] = mapped_column(ForeignKey("user_account.id"), nullable=False)
    approved_by: Mapped[int | None] = mapped_column(ForeignKey("user_account.id"))
    approval_note: Mapped[str | None] = mapped_column(Text)
    logical_action_id: Mapped[str | None] = mapped_column(
        String(64, collation="ascii_bin")
    )
    confirmation_mode: Mapped[str] = mapped_column(
        String(32, collation="ascii_bin"),
        nullable=False,
        default="R2_STATELESS_COMPAT",
    )
    customer_confirmed_actor_id: Mapped[int | None] = mapped_column(
        ForeignKey(
            "user_account.id",
            name="fk_agent_action_customer_confirmer",
            ondelete="RESTRICT",
        )
    )
    customer_confirmed_at: Mapped[datetime | None] = mapped_column(DateTime)
    customer_confirmation_challenge_digest: Mapped[str | None] = mapped_column(
        String(64, collation="ascii_bin")
    )
    admin_decision: Mapped[str | None] = mapped_column(
        String(16, collation="ascii_bin")
    )
    admin_decided_actor_id: Mapped[int | None] = mapped_column(
        ForeignKey(
            "user_account.id",
            name="fk_agent_action_admin_decider",
            ondelete="RESTRICT",
        )
    )
    admin_decided_at: Mapped[datetime | None] = mapped_column(DateTime)
    admin_reason_code: Mapped[str | None] = mapped_column(
        String(64, collation="ascii_bin")
    )
    resume_status: Mapped[str] = mapped_column(
        String(32, collation="ascii_bin"),
        nullable=False,
        default="NOT_APPLICABLE",
    )
    execution_result_code: Mapped[str | None] = mapped_column(
        String(64, collation="ascii_bin")
    )
    execution_error_type: Mapped[str | None] = mapped_column(
        String(64, collation="ascii_bin")
    )
    execution_error_summary: Mapped[str | None] = mapped_column(String(512))
    legacy_original_status: Mapped[str | None] = mapped_column(
        String(32, collation="ascii_bin")
    )
    prepared_order_status: Mapped[str | None] = mapped_column(
        String(32, collation="ascii_bin")
    )
    attempt_id: Mapped[str | None] = mapped_column(String(64))
    effect_id: Mapped[int | None] = mapped_column(ForeignKey("agent_effect.id", name="fk_agent_action_effect"))
    effect_node_name: Mapped[str | None] = mapped_column(String(64))
    effect_purpose: Mapped[str | None] = mapped_column(String(64))
    effect_sequence: Mapped[int | None] = mapped_column(Integer)
    effect_idempotency_key: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime)
    executed_at: Mapped[datetime | None] = mapped_column(DateTime)

    __table_args__ = (
        ForeignKeyConstraint(
            ["attempt_id", "run_id"],
            ["agent_run_attempt.attempt_id", "agent_run_attempt.run_id"],
            name="fk_agent_action_attempt_run",
        ),
        UniqueConstraint(
            "logical_action_id",
            name="uk_agent_action_logical_action",
        ),
        UniqueConstraint("effect_id", name="uk_agent_action_effect"),
        UniqueConstraint("effect_idempotency_key", name="uk_agent_action_effect_idempotency"),
        CheckConstraint(
            "effect_sequence IS NULL OR effect_sequence >= 0",
            name="agent_action_effect_sequence_nonnegative",
        ),
        CheckConstraint(
            "("
            "effect_id IS NULL AND attempt_id IS NULL AND effect_node_name IS NULL "
            "AND effect_purpose IS NULL AND effect_sequence IS NULL "
            "AND effect_idempotency_key IS NULL"
            ") OR ("
            "effect_id IS NOT NULL AND attempt_id IS NOT NULL AND effect_node_name IS NOT NULL "
            "AND effect_purpose IS NOT NULL AND effect_sequence IS NOT NULL "
            "AND effect_idempotency_key IS NOT NULL"
            ")",
            name="agent_action_effect_complete",
        ),
        CheckConstraint(
            "confirmation_mode IN "
            "('LEGACY_UNVERIFIED', 'R2_STATELESS_COMPAT', 'DURABLE_INTERRUPT')",
            name="action_confirmation_mode_closed",
        ),
        CheckConstraint(
            "resume_status IN ('NOT_APPLICABLE', 'WAITING_ADMIN_DECISION', "
            "'RESUME_PENDING', 'RESUMED', 'COMPLETED', 'FAILED_RETRYABLE', "
            "'LEGACY_BLOCKED')",
            name="action_resume_status_closed",
        ),
        CheckConstraint(
            "(customer_confirmed_actor_id IS NULL AND customer_confirmed_at IS NULL "
            "AND customer_confirmation_challenge_digest IS NULL) OR "
            "(customer_confirmed_actor_id IS NOT NULL AND customer_confirmed_at IS NOT NULL "
            "AND customer_confirmation_challenge_digest IS NOT NULL)",
            name="action_customer_confirmation_tuple",
        ),
        CheckConstraint(
            "customer_confirmation_challenge_digest IS NULL OR "
            "REGEXP_LIKE(customer_confirmation_challenge_digest, '^[0-9a-f]{64}$', 'c')",
            name="action_customer_digest_shape",
        ),
        CheckConstraint(
            "logical_action_id IS NULL OR "
            "REGEXP_LIKE(logical_action_id, '^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$', 'c')",
            name="action_logical_identity_shape",
        ),
        CheckConstraint(
            "(confirmation_mode = 'DURABLE_INTERRUPT' "
            "AND logical_action_id IS NOT NULL "
            "AND customer_confirmed_actor_id IS NOT NULL "
            "AND customer_confirmed_actor_id = created_by) OR "
            "(confirmation_mode IN ('LEGACY_UNVERIFIED', 'R2_STATELESS_COMPAT') "
            "AND logical_action_id IS NULL AND customer_confirmed_actor_id IS NULL)",
            name="action_confirmation_mode_binding",
        ),
        CheckConstraint(
            "(admin_decision IS NULL AND admin_decided_actor_id IS NULL "
            "AND admin_decided_at IS NULL AND admin_reason_code IS NULL) OR "
            "(admin_decision IS NOT NULL AND admin_decided_actor_id IS NOT NULL "
            "AND admin_decided_at IS NOT NULL AND admin_reason_code IS NOT NULL)",
            name="action_admin_decision_tuple",
        ),
        CheckConstraint(
            "admin_decision IS NULL OR admin_decision IN ('APPROVE', 'REJECT')",
            name="action_admin_decision_closed",
        ),
        CheckConstraint(
            "admin_reason_code IS NULL OR "
            "REGEXP_LIKE(admin_reason_code, '^[A-Z][A-Z0-9_]{0,63}$', 'c')",
            name="action_admin_reason_shape",
        ),
        CheckConstraint(
            "status IN ('LEGACY_REVIEW_REQUIRED', 'PENDING', 'APPROVING', "
            "'APPROVED', 'REJECTING', 'REJECTED', 'EXECUTING', 'EXECUTED', "
            "'STALE', 'FAILED', 'FAILED_RETRYABLE')",
            name="action_status_closed",
        ),
        CheckConstraint(
            "(execution_result_code IS NULL OR "
            "REGEXP_LIKE(execution_result_code, '^[A-Z][A-Z0-9_]{0,63}$', 'c')) "
            "AND (execution_error_type IS NULL OR "
            "REGEXP_LIKE(execution_error_type, '^[A-Z][A-Z0-9_]{0,63}$', 'c'))",
            name="action_code_shapes",
        ),
        CheckConstraint(
            "((execution_error_type IS NULL AND execution_error_summary IS NULL) OR "
            "(execution_error_type IS NOT NULL AND execution_error_summary IS NOT NULL "
            "AND status IN ('FAILED', 'FAILED_RETRYABLE'))) AND "
            "(execution_result_code IS NULL OR status = 'EXECUTED') AND "
            "NOT (execution_result_code IS NOT NULL AND execution_error_type IS NOT NULL) AND "
            "(executed_at IS NULL OR status = 'EXECUTED') AND "
            "(status NOT IN ('FAILED', 'FAILED_RETRYABLE') OR "
            "execution_error_type IS NOT NULL)",
            name="action_execution_outcome",
        ),
        CheckConstraint(
            "confirmation_mode = 'LEGACY_UNVERIFIED' OR "
            "status NOT IN ('EXECUTED', 'REJECTED') OR "
            "(status = 'EXECUTED' AND admin_decision = 'APPROVE') OR "
            "(status = 'REJECTED' AND admin_decision = 'REJECT')",
            name="action_terminal_admin_binding",
        ),
        CheckConstraint(
            "(confirmation_mode = 'R2_STATELESS_COMPAT' "
            "AND resume_status = 'NOT_APPLICABLE') OR "
            "(confirmation_mode = 'LEGACY_UNVERIFIED' "
            "AND resume_status IN ('LEGACY_BLOCKED', 'COMPLETED')) OR "
            "(confirmation_mode = 'DURABLE_INTERRUPT' "
            "AND resume_status IN ('WAITING_ADMIN_DECISION', 'RESUME_PENDING', "
            "'RESUMED', 'COMPLETED', 'FAILED_RETRYABLE'))",
            name="action_resume_mode_binding",
        ),
        CheckConstraint(
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
            name="action_legacy_classification",
        ),
        CheckConstraint(
            "(confirmation_mode = 'DURABLE_INTERRUPT' AND (("
            "prepared_order_status IS NOT NULL AND "
            "REGEXP_LIKE(prepared_order_status, '^[A-Z][A-Z0-9_]{0,31}$', 'c')"
            ") OR (prepared_order_status IS NULL AND status = 'FAILED' "
            "AND resume_status = 'COMPLETED' "
            "AND execution_error_type = 'PREPARE_EVIDENCE_UNRECOVERABLE' "
            "AND execution_error_summary = "
            "'Durable prepare evidence is unavailable; create a new action.' "
            "AND execution_result_code IS NULL AND admin_decision IS NULL "
            "AND admin_decided_actor_id IS NULL AND admin_decided_at IS NULL "
            "AND admin_reason_code IS NULL))) OR "
            "(confirmation_mode IN ('R2_STATELESS_COMPAT', 'LEGACY_UNVERIFIED') "
            "AND prepared_order_status IS NULL)",
            name="action_prepared_status_mode_binding",
        ),
    )


class AgentFeedback(Base):
    __tablename__ = "agent_feedback"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False)
    rating: Mapped[str] = mapped_column(String(32), nullable=False)
    reason: Mapped[str | None] = mapped_column(String(512))
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now)


class AgentRetrievalTrace(Base):
    __tablename__ = "agent_retrieval_trace"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False)
    candidate_id: Mapped[str] = mapped_column(String(128), nullable=False)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    document_id: Mapped[str | None] = mapped_column(String(64))
    chunk_id: Mapped[str | None] = mapped_column(String(64))
    rule_id: Mapped[str | None] = mapped_column(String(64))
    original_score: Mapped[Decimal] = mapped_column(Numeric(10, 6), nullable=False)
    fused_score: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))
    rerank_score: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))
    selected: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    decision_reason: Mapped[str | None] = mapped_column(String(255))
    metadata_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now)


Index("idx_customer_order_user_created", CustomerOrder.user_id, CustomerOrder.created_at)
Index("idx_shipment_event_order_time", ShipmentEvent.order_id, ShipmentEvent.event_time)
Index("idx_after_sale_rule_effective", AfterSaleRule.status, AfterSaleRule.effective_from, AfterSaleRule.effective_to)
Index(
    "idx_after_sale_rule_condition_lookup",
    AfterSaleRuleCondition.after_sale_type,
    AfterSaleRuleCondition.order_status,
)
Index("idx_agent_run_thread", AgentRun.thread_id)
Index("idx_agent_attempt_run_started", AgentRunAttempt.run_id, AgentRunAttempt.started_at)
Index("idx_agent_attempt_thread_status", AgentRunAttempt.thread_id, AgentRunAttempt.status)
Index(
    "idx_checkpoint_manifest_publication",
    AgentCheckpointWriteManifest.thread_id,
    AgentCheckpointWriteManifest.publication_version,
)
Index(
    "idx_content_source_conversation",
    AgentContentSourceRevision.conversation_id,
    AgentContentSourceRevision.subject_user_id,
)
Index(
    "idx_content_source_run_attempt",
    AgentContentSourceRevision.run_id,
    AgentContentSourceRevision.producing_attempt_id,
)
Index(
    "idx_checkpoint_content_ref_publication",
    AgentCheckpointContentReference.thread_id,
    AgentCheckpointContentReference.publication_version,
)
Index(
    "idx_checkpoint_content_ref_source",
    AgentCheckpointContentReference.source_kind,
    AgentCheckpointContentReference.source_record_id,
    AgentCheckpointContentReference.source_revision,
)
Index("idx_agent_effect_attempt", AgentEffect.attempt_id, AgentEffect.created_at)
Index("idx_agent_step_run_created", AgentStep.run_id, AgentStep.created_at)
Index("idx_agent_tool_call_run_created", AgentToolCall.run_id, AgentToolCall.created_at)
Index("idx_agent_action_status_created", AgentActionRequest.status, AgentActionRequest.created_at)
Index("idx_agent_retrieval_trace_run", AgentRetrievalTrace.run_id, AgentRetrievalTrace.created_at)
