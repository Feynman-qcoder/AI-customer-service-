from __future__ import annotations

import re
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.runtime.checkpoint_security import (
    CheckpointFilesystemSecurityPort,
    ValidatedCheckpointLocation,
)
from app.runtime.checkpoint_security import (
    CheckpointStorageError as CheckpointStorageError,
)
from app.runtime.checkpoint_security import (
    validate_checkpoint_storage as _validate_checkpoint_storage,
)

# server/app/core/config.py -> parents[3] is the repository root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = PROJECT_ROOT / ".env"

_SUPPORTED_CHECKPOINT_BACKENDS = frozenset({"sqlite"})


class Settings(BaseSettings):
    """Runtime settings.

    Values resolve in this order: real process environment variables, then the
    repository root ``.env`` file, then the defaults declared below. Loading the
    ``.env`` file here means ``deploy/run-server-local.ps1`` and a bare
    ``uvicorn app.main:app`` observe the same configuration, so the backend can
    no longer silently fall back to the built-in defaults.
    """

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3307
    mysql_database: str = "smart_customer_service"
    mysql_username: str = "root"
    mysql_password: str = "change_me"

    redis_host: str = "127.0.0.1"
    redis_port: int = 6379
    redis_password: str = ""

    server_port: int = 18080
    jwt_secret: str = "dev-secret-change-me-at-least-32-chars"
    access_token_ttl_minutes: int = 30
    refresh_token_ttl_days: int = 7
    chat_rate_limit_per_minute: int = 20

    demo_admin_username: str = "admin"
    demo_admin_password: str = "local_admin_demo_only"
    demo_customer_username: str = "user"
    demo_customer_password: str = "local_customer_demo_only"

    llm_mock_enabled: bool = True
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com"
    llm_model_name: str = "deepseek-v4-flash"
    llm_temperature: float = 0.2
    llm_request_timeout_seconds: int = 12
    llm_max_completion_tokens: int = 512

    qdrant_host: str = "127.0.0.1"
    qdrant_port: int = 6333
    rag_top_k: int = 5
    rag_min_retrieval_score: float = 0.35
    document_storage_path: str = "./data/documents"
    embedding_mock_enabled: bool = True
    embedding_api_key: str = ""
    embedding_base_url: str = "https://api.openai.com"
    embedding_model_name: str = "text-embedding-3-small"
    embedding_dimension: int = 384

    cors_origins: list[str] = ["http://127.0.0.1:5173", "http://localhost:5173"]

    # Durable checkpoint storage fails closed when its location cannot be
    # verified. The current topology supports one SQLite backend, one host, one
    # worker; required=true fails closed at startup instead of silently
    # degrading to a checkpointer-less runtime. Retention only ever applies to
    # terminal states; WAITING/RESUME/EXECUTING states must never be removed
    # by TTL alone.
    checkpoint_backend: str = "sqlite"
    checkpoint_db_path: str = "./data/checkpoints/agent.sqlite"
    checkpoint_required: bool = True
    checkpoint_retention_days: int = 30
    checkpoint_worker_count: int = 1
    checkpoint_lease_ttl_seconds: int = 60
    checkpoint_lease_renew_interval_seconds: int = 15

    # Durable customer confirmation is opt-in for configurations that have the
    # required persistence and recovery guarantees.
    durable_customer_interrupt_enabled: bool = False

    # Recent-message and summary budgets are recorded in typed memory output.
    memory_recent_message_limit: int = 12
    memory_recent_token_budget: int = 2000
    memory_summary_trigger_message_count: int = 20
    memory_summary_trigger_token_budget: int = 4000
    memory_summary_token_budget: int = 1200

    # ContextAssembler partition budgets include
    # their label/role/trust/separator framing and never borrow from each other.
    memory_context_total_token_budget: int = 6000
    memory_context_system_token_budget: int = 1200
    memory_context_summary_token_budget: int = 1200
    memory_context_working_token_budget: int = 400
    memory_context_recent_token_budget: int = 2000
    memory_context_current_token_budget: int = 1200

    @field_validator(
        "memory_recent_message_limit",
        "memory_recent_token_budget",
        "memory_summary_trigger_message_count",
        "memory_summary_trigger_token_budget",
        "memory_summary_token_budget",
        "memory_context_total_token_budget",
        "memory_context_system_token_budget",
        "memory_context_summary_token_budget",
        "memory_context_working_token_budget",
        "memory_context_recent_token_budget",
        "memory_context_current_token_budget",
        mode="before",
    )
    @classmethod
    def _strict_positive_memory_budget(cls, value: object) -> int:
        if type(value) is int:
            parsed = value
        elif type(value) is str and re.fullmatch(r"[1-9][0-9]*", value) is not None:
            parsed = int(value)
        else:
            raise ValueError("memory budget settings must be strict positive integers")
        if parsed <= 0 or parsed > 9_223_372_036_854_775_807:
            raise ValueError("memory budget settings must be strict positive integers")
        return parsed

    @field_validator("llm_request_timeout_seconds", mode="before")
    @classmethod
    def _bounded_llm_request_timeout(cls, value: object) -> int:
        parsed = cls._strict_bounded_integer(value, setting="LLM_REQUEST_TIMEOUT_SECONDS")
        if parsed > 12:
            raise ValueError("LLM_REQUEST_TIMEOUT_SECONDS must be between 1 and 12")
        return parsed

    @field_validator("llm_max_completion_tokens", mode="before")
    @classmethod
    def _bounded_llm_completion_tokens(cls, value: object) -> int:
        parsed = cls._strict_bounded_integer(value, setting="LLM_MAX_COMPLETION_TOKENS")
        if parsed > 512:
            raise ValueError("LLM_MAX_COMPLETION_TOKENS must be between 1 and 512")
        return parsed

    @staticmethod
    def _strict_bounded_integer(value: object, *, setting: str) -> int:
        if type(value) is int:
            parsed = value
        elif type(value) is str and re.fullmatch(r"[1-9][0-9]*", value) is not None:
            parsed = int(value)
        else:
            raise ValueError(f"{setting} must be a strict positive integer")
        if parsed < 1:
            raise ValueError(f"{setting} must be a strict positive integer")
        return parsed

    @field_validator("jwt_secret")
    @classmethod
    def _require_long_secret(cls, value: str) -> str:
        if len(value) < 32:
            raise ValueError("JWT_SECRET must be at least 32 characters")
        return value

    @field_validator("document_storage_path", mode="after")
    @classmethod
    def _absolute_storage_path(cls, value: str) -> str:
        """Resolve relative storage paths against the repository root.

        Anchoring the path to the repository root keeps the API process and
        ingestion scripts on the same storage location.
        """
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return str(path)

    @field_validator("checkpoint_db_path", mode="after")
    @classmethod
    def _validate_checkpoint_db_path(cls, value: str) -> str:
        """Anchor and sanity-check the checkpoint database path.

        Settings performs pure shape validation only.  Canonical absolute path,
        symlink/reparse, mount and permission checks belong to the injected
        CheckpointFilesystemSecurityPort and happen before any storage I/O.
        """
        if not value or not value.strip():
            raise ValueError("CHECKPOINT_DB_PATH must not be empty when provided")
        if "://" in value:
            raise ValueError(
                "CHECKPOINT_DB_PATH must be a local filesystem path, not a URI; network shares are not supported in V1"
            )
        return value.strip()

    @model_validator(mode="after")
    def _validate_checkpoint_contract(self) -> Settings:
        """Pure configuration validation; no filesystem access happens here.

        These guards apply regardless of ``checkpoint_required``: an invalid
        backend, an impossible worker topology or a non-positive retention is
        always a configuration error. Storage path/permission validation with
        real I/O lives in :func:`validate_checkpoint_storage`, which runtime
        startup calls only when required=true.
        """
        if self.checkpoint_backend not in _SUPPORTED_CHECKPOINT_BACKENDS:
            raise ValueError(f"CHECKPOINT_BACKEND must be 'sqlite' in V1; got {self.checkpoint_backend!r}")
        if self.checkpoint_required and not self.checkpoint_db_path:
            raise ValueError("CHECKPOINT_DB_PATH is required when CHECKPOINT_REQUIRED=true")
        if self.checkpoint_retention_days < 1:
            raise ValueError("CHECKPOINT_RETENTION_DAYS must be a positive integer")
        if self.checkpoint_worker_count != 1:
            raise ValueError(f"V1 durable checkpoint supports exactly one worker; got {self.checkpoint_worker_count}")
        if not 0 < self.checkpoint_lease_ttl_seconds <= 86_400:
            raise ValueError("CHECKPOINT_LEASE_TTL_SECONDS must be between 1 and 86400")
        if not 0 < self.checkpoint_lease_renew_interval_seconds:
            raise ValueError("CHECKPOINT_LEASE_RENEW_INTERVAL_SECONDS must be positive")
        if self.checkpoint_lease_renew_interval_seconds * 3 > self.checkpoint_lease_ttl_seconds:
            raise ValueError(
                "CHECKPOINT_LEASE_RENEW_INTERVAL_SECONDS must not exceed one third of CHECKPOINT_LEASE_TTL_SECONDS"
            )
        return self

    @model_validator(mode="after")
    def _validate_context_budget_contract(self) -> Settings:
        partition_total = (
            self.memory_context_system_token_budget
            + self.memory_context_summary_token_budget
            + self.memory_context_working_token_budget
            + self.memory_context_recent_token_budget
            + self.memory_context_current_token_budget
        )
        if partition_total > self.memory_context_total_token_budget:
            raise ValueError("MEMORY_CONTEXT partition budgets must not exceed MEMORY_CONTEXT_TOTAL_TOKEN_BUDGET")
        return self

    @property
    def database_url(self) -> str:
        return (
            f"mysql+aiomysql://{self.mysql_username}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}?charset=utf8mb4"
        )

    @property
    def sync_database_url(self) -> str:
        return (
            f"mysql+pymysql://{self.mysql_username}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}?charset=utf8mb4"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()


def validate_checkpoint_storage(
    settings: Settings,
    *,
    security: CheckpointFilesystemSecurityPort | None = None,
    environ: Mapping[str, str] | None = None,
) -> ValidatedCheckpointLocation | None:
    """Compatibility entry point delegating all platform work to a narrow port."""

    return _validate_checkpoint_storage(
        settings,
        security=security,
        environ=environ,
        repository_root=PROJECT_ROOT,
    )
