from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.api.v1 import conversations as conversations_api
from app.core.security import AuthenticatedUser, current_user
from app.db.models import ChatConversation, ChatMessage
from app.db.session import get_session

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CUSTOMER_CHAT_PATH = REPOSITORY_ROOT / "web" / "src" / "views" / "CustomerChat.vue"
CONVERSATION_SERVICE_PATH = REPOSITORY_ROOT / "server" / "app" / "services" / "conversation_service.py"


@dataclass
class HistoryHarness:
    client: AsyncClient
    sessions: async_sessionmaker[AsyncSession]
    actor: dict[str, AuthenticatedUser]


@pytest.fixture
async def history_harness() -> AsyncIterator[HistoryHarness]:
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
    )
    sessions = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.execute(
            text(
                """
                CREATE TABLE chat_conversation (
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER,
                    conversation_no VARCHAR(64) NOT NULL,
                    title VARCHAR(255) NOT NULL,
                    status VARCHAR(32) NOT NULL,
                    created_at DATETIME NOT NULL,
                    updated_at DATETIME NOT NULL
                )
                """
            )
        )
        await connection.execute(
            text(
                """
                CREATE TABLE chat_message (
                    id INTEGER PRIMARY KEY,
                    conversation_id INTEGER NOT NULL,
                    role VARCHAR(32) NOT NULL,
                    content TEXT NOT NULL,
                    sources_json TEXT,
                    retrieval_score NUMERIC(8, 4),
                    confidence_level VARCHAR(32),
                    need_human BOOLEAN NOT NULL DEFAULT 0,
                    source_run_id VARCHAR(64),
                    source_attempt_id VARCHAR(64),
                    message_purpose VARCHAR(64),
                    message_sequence INTEGER,
                    message_idempotency_key VARCHAR(64),
                    effect_id INTEGER,
                    created_at DATETIME NOT NULL
                )
                """
            )
        )

    async with sessions() as session:
        session.add_all(
            [
                _conversation(101, 1, "A-最近有消息", datetime(2026, 1, 1, 1, 0)),
                _conversation(102, 1, "A-较早有消息", datetime(2026, 1, 1, 2, 0)),
                _conversation(103, 1, "A-早期空会话", datetime(2026, 1, 1, 3, 0)),
                _conversation(104, 1, "A-同时间空会话-小ID", datetime(2026, 1, 1, 4, 0)),
                _conversation(105, 1, "A-同时间空会话-大ID", datetime(2026, 1, 1, 4, 0)),
                _conversation(201, 2, "B-私有会话", datetime(2026, 1, 1, 5, 0)),
            ]
        )
        session.add_all(
            [
                _message(1002, 101, "ASSISTANT", "fixture assistant", datetime(2026, 1, 2, 10, 0)),
                _message(1001, 101, "USER", "fixture user", datetime(2026, 1, 2, 10, 0)),
                _message(1101, 102, "USER", "fixture older", datetime(2026, 1, 2, 9, 0)),
                _message(2001, 201, "USER", "fixture private", datetime(2026, 1, 2, 11, 0)),
            ]
        )
        await session.commit()

    app = FastAPI()
    app.include_router(conversations_api.router, prefix="/api/v1")
    actor = {
        "value": AuthenticatedUser(user_id=1, username="customer-a", name="A", role="CUSTOMER")
    }

    async def override_session() -> AsyncIterator[AsyncSession]:
        async with sessions() as session:
            yield session

    async def override_user() -> AuthenticatedUser:
        return actor["value"]

    app.dependency_overrides[get_session] = override_session
    app.dependency_overrides[current_user] = override_user
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield HistoryHarness(client=client, sessions=sessions, actor=actor)
    await engine.dispose()


def _conversation(
    conversation_id: int,
    user_id: int,
    title: str,
    created_at: datetime,
) -> ChatConversation:
    return ChatConversation(
        id=conversation_id,
        user_id=user_id,
        conversation_no=f"CV{conversation_id:016d}",
        title=title,
        status="ACTIVE",
        created_at=created_at,
        updated_at=created_at,
    )


def _message(
    message_id: int,
    conversation_id: int,
    role: str,
    content: str,
    created_at: datetime,
) -> ChatMessage:
    return ChatMessage(
        id=message_id,
        conversation_id=conversation_id,
        role=role,
        content=content,
        sources_json=None,
        retrieval_score=None,
        confidence_level=None,
        need_human=False,
        source_run_id=None,
        source_attempt_id=None,
        message_purpose=None,
        message_sequence=None,
        message_idempotency_key=None,
        effect_id=None,
        created_at=created_at,
    )


async def _database_counts(sessions: async_sessionmaker[AsyncSession]) -> tuple[int, int]:
    async with sessions() as session:
        conversations = int(await session.scalar(select(func.count()).select_from(ChatConversation)) or 0)
        messages = int(await session.scalar(select(func.count()).select_from(ChatMessage)) or 0)
    return conversations, messages


@pytest.mark.asyncio
async def test_list_is_owner_scoped_and_read_only(history_harness: HistoryHarness) -> None:
    before = await _database_counts(history_harness.sessions)

    response = await history_harness.client.get("/api/v1/conversations")

    assert response.status_code == 200
    payload = response.json()["data"]
    assert payload["total"] == 5
    assert {record["id"] for record in payload["records"]} == {101, 102, 103, 104, 105}
    assert 201 not in {record["id"] for record in payload["records"]}
    assert await _database_counts(history_harness.sessions) == before


@pytest.mark.asyncio
async def test_second_user_never_sees_first_users_conversations(history_harness: HistoryHarness) -> None:
    history_harness.actor["value"] = AuthenticatedUser(
        user_id=2,
        username="customer-b",
        name="B",
        role="CUSTOMER",
    )

    response = await history_harness.client.get("/api/v1/conversations")

    assert response.status_code == 200
    payload = response.json()["data"]
    assert payload["total"] == 1
    assert [record["id"] for record in payload["records"]] == [201]


@pytest.mark.asyncio
async def test_admin_list_is_still_scoped_to_current_account(history_harness: HistoryHarness) -> None:
    history_harness.actor["value"] = AuthenticatedUser(
        user_id=1,
        username="admin-a",
        name="Admin A",
        role="ADMIN",
    )

    response = await history_harness.client.get("/api/v1/conversations")

    assert response.status_code == 200
    ids = [record["id"] for record in response.json()["data"]["records"]]
    assert ids == [101, 102, 105, 104, 103]
    assert 201 not in ids


@pytest.mark.asyncio
async def test_history_sorts_by_last_message_then_empty_creation_and_id(
    history_harness: HistoryHarness,
) -> None:
    response = await history_harness.client.get("/api/v1/conversations")

    assert response.status_code == 200
    records = response.json()["data"]["records"]
    assert [record["id"] for record in records] == [101, 102, 105, 104, 103]
    assert [record["messageCount"] for record in records] == [2, 1, 0, 0, 0]
    assert records[0]["lastMessageAt"].endswith("+08:00")
    assert records[2]["lastMessageAt"] is None
    assert set(records[0]) == {
        "id",
        "conversationNo",
        "title",
        "status",
        "createdAt",
        "updatedAt",
        "messageCount",
        "lastMessageAt",
    }


@pytest.mark.asyncio
async def test_history_pagination_and_boundaries(history_harness: HistoryHarness) -> None:
    first = await history_harness.client.get("/api/v1/conversations", params={"page": 1, "size": 2})
    second = await history_harness.client.get("/api/v1/conversations", params={"page": 2, "size": 2})
    third = await history_harness.client.get("/api/v1/conversations", params={"page": 3, "size": 2})

    assert first.status_code == second.status_code == third.status_code == 200
    assert first.json()["data"] == {
        "page": 1,
        "size": 2,
        "total": 5,
        "records": first.json()["data"]["records"],
    }
    assert [row["id"] for row in first.json()["data"]["records"]] == [101, 102]
    assert [row["id"] for row in second.json()["data"]["records"]] == [105, 104]
    assert [row["id"] for row in third.json()["data"]["records"]] == [103]
    for params in ({"page": 0}, {"size": 0}, {"size": 101}):
        invalid = await history_harness.client.get("/api/v1/conversations", params=params)
        assert invalid.status_code == 422


@pytest.mark.asyncio
async def test_message_order_is_created_at_then_id(history_harness: HistoryHarness) -> None:
    response = await history_harness.client.get("/api/v1/conversations/101/messages")

    assert response.status_code == 200
    assert [message["id"] for message in response.json()["data"]] == [1001, 1002]
    compact = "".join(CONVERSATION_SERVICE_PATH.read_text(encoding="utf-8").split())
    assert ".order_by(ChatMessage.created_at.asc(),ChatMessage.id.asc())" in compact


def test_frontend_mount_loads_history_before_any_creation() -> None:
    source = CUSTOMER_CHAT_PATH.read_text(encoding="utf-8")
    mounted_start = source.index("onMounted(async () => {")
    mounted_end = source.index("\n})", mounted_start) + len("\n})")
    mounted = source[mounted_start:mounted_end]

    assert "await initializeConversationHistory()" in mounted
    assert "api.post('/conversations'" not in mounted
    assert "api.get('/conversations'" in source


def test_frontend_history_failure_never_auto_creates() -> None:
    source = CUSTOMER_CHAT_PATH.read_text(encoding="utf-8")

    assert "historyLoadFailed" in source
    assert "if (historyLoadFailed.value) return" in source
    assert source.count("api.post('/conversations'") == 1


def test_frontend_only_empty_history_or_explicit_button_creates() -> None:
    source = CUSTOMER_CHAT_PATH.read_text(encoding="utf-8")

    assert '@click="createNewConversation"' in source
    assert "if (!conversationHistory.value.length)" in source
    assert "await createNewConversation()" in source
    assert ":disabled=" in source and "sending" in source


def test_frontend_preference_is_user_scoped_and_server_validated() -> None:
    source = CUSTOMER_CHAT_PATH.read_text(encoding="utf-8")

    assert "currentUser()?.userId" in source
    assert "conversation-preference" in source
    assert "Number.parseInt" in source
    assert ".find((conversation) => conversation.id === preferredId)" in source
    assert "localStorage.setItem" in source
    assert "JSON.stringify(messages" not in source


def test_frontend_switch_has_stale_request_guard_and_clears_transient_state() -> None:
    source = CUSTOMER_CHAT_PATH.read_text(encoding="utf-8")

    assert "messageRequestSequence" in source
    assert "requestSequence !== messageRequestSequence" in source
    assert "lastSources.value = []" in source
    assert "selectedSource.value = null" in source
    assert "if (sending.value) return" in source
