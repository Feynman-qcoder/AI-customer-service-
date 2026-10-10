"""Service runtime bootstrap for the benchmark profiles.

Builds the REAL production composition (checkpoint runtime + managed agent
service) against the isolated stack, exactly the way ``app.main`` does in
production and the R4C acceptance tests do with real MySQL.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.composition import build_managed_agent_service
from app.core.config import Settings
from app.runtime.checkpoint_runtime import (
    CheckpointRuntimeLifecycle,
    build_checkpoint_runtime,
)
from app.services.customer_agent_application import AgentService


@dataclass(slots=True)
class ServiceRuntime:
    settings: Settings
    lifecycle: CheckpointRuntimeLifecycle
    service: AgentService
    session_maker: async_sessionmaker[AsyncSession]
    checkpoint_dir: Path
    actors: Any = None
    _engine: object = None

    async def stop(self) -> None:
        await self.lifecycle.stop()
        await self._engine.dispose()  # type: ignore[attr-defined]


def build_settings(
    *,
    checkpoint_dir: Path,
    durable_interrupt: bool = True,
    llm_mock_enabled: bool = True,
    embedding_mock_enabled: bool = True,
) -> Settings:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return Settings(
        _env_file=None,
        jwt_secret="benchmark-only-secret-at-least-32-characters",
        checkpoint_backend="sqlite",
        checkpoint_db_path=str(checkpoint_dir / "agent.sqlite"),
        checkpoint_required=True,
        checkpoint_worker_count=1,
        checkpoint_lease_ttl_seconds=30,
        checkpoint_lease_renew_interval_seconds=5,
        durable_customer_interrupt_enabled=durable_interrupt,
        llm_mock_enabled=llm_mock_enabled,
        embedding_mock_enabled=embedding_mock_enabled,
        rag_top_k=5,
        rag_min_retrieval_score=0.35,
    )


async def start_service_runtime(
    *,
    settings: Settings,
    work_dir: Path,
) -> ServiceRuntime:
    """Start the checkpoint runtime and build the managed agent service.

    ``repository_root`` must be the REAL repository: the checkpoint security
    ignores paths outside the repository and fails closed when ``git`` cannot
    verify a path inside the given root (a temp dir is not a git repo).
    """

    from app.db.session import require_session_factory_target
    from evals.benchmark.harness import resolve_actors

    repository_root = Path(__file__).resolve().parents[3]
    database_url = (
        f"mysql+aiomysql://{settings.mysql_username}:{settings.mysql_password}"
        f"@{settings.mysql_host}:{settings.mysql_port}/{settings.mysql_database}"
    )
    # No pool_pre_ping: aiomysql's async adapter ping signature is
    # incompatible with this SQLAlchemy version (the app factory agrees).
    engine = create_async_engine(database_url)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    require_session_factory_target(maker, database_url)
    lifecycle = build_checkpoint_runtime(
        settings,
        repository_root=repository_root,
        environ={"WEB_CONCURRENCY": "1"},
    )
    await lifecycle.start()
    service = build_managed_agent_service(
        settings,
        lifecycle.composition_handle(),
        database_session_factory=maker,
    )
    runtime = ServiceRuntime(
        settings=settings,
        lifecycle=lifecycle,
        service=service,
        session_maker=maker,
        checkpoint_dir=work_dir / "checkpoints",
        _engine=engine,
    )
    runtime.actors = await resolve_actors(maker)
    return runtime


__all__ = ["ServiceRuntime", "build_settings", "start_service_runtime"]
