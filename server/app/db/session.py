from collections.abc import AsyncIterator

from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings

_session_factory: async_sessionmaker[AsyncSession] | None = None
_engine: AsyncEngine | None = None


class SessionFactoryTargetMismatch(RuntimeError):
    """A test/runtime factory is not bound to the explicitly authorized database."""


def require_session_factory_target(
    factory: async_sessionmaker[AsyncSession],
    expected_database_url: str,
) -> None:
    """Fail before checkout/SQL when a factory targets a different database URL."""

    bind = factory.kw.get("bind")
    actual_url = getattr(bind, "url", None)
    try:
        expected_url = make_url(expected_database_url)
    except Exception:
        raise SessionFactoryTargetMismatch("expected database URL is invalid") from None
    if actual_url is None or actual_url != expected_url:
        raise SessionFactoryTargetMismatch(
            "session factory target does not match the explicitly authorized database"
        )


def session_factory() -> async_sessionmaker[AsyncSession]:
    global _engine, _session_factory
    if _session_factory is None:
        _engine = create_async_engine(settings.database_url)
        _session_factory = async_sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)
    return _session_factory


async def get_session() -> AsyncIterator[AsyncSession]:
    factory = session_factory()
    async with factory() as session:
        yield session


async def dispose_engine() -> None:
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None
