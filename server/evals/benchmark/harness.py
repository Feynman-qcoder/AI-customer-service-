"""Production-shaped harness shared by the benchmark profile runners.

Each case runs against the REAL ``AgentService`` composition (durable
interrupts, checkpoint bridge, replay-safe effects, working memory, rolling
summary) on the isolated MySQL/SQLite stack. Every case gets a fresh
conversation; multi-turn cases replay their USER history turns through the
real pipeline first so working memory and rolling summary are genuinely
populated by the system under test.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.security import AuthenticatedUser
from app.db.models import (
    AgentActionRequest,
    AgentEffect,
    AgentRun,
    AgentToolCall,
    ChatConversation,
    CustomerOrder,
    ProductCatalog,
)

# ---------------------------------------------------------------------------
# actors
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BenchmarkActors:
    customer: AuthenticatedUser
    victim: AuthenticatedUser
    admin: AuthenticatedUser


async def resolve_actors(maker: async_sessionmaker[AsyncSession]) -> BenchmarkActors:
    async with maker() as session:
        async def by_username(username: str) -> AuthenticatedUser:
            row = (
                await session.execute(
                    select(_user_model()).where(_user_model().username == username)
                )
            ).scalar_one()
            return AuthenticatedUser(
                user_id=int(row.id),
                username=str(row.username),
                name=str(row.display_name),
                role=str(row.role),
            )

        customer = await by_username("benchmark-customer")
        victim = await by_username("benchmark-victim")
        admin_row = (
            await session.execute(
                select(_user_model()).where(_user_model().role == "ADMIN")
            )
        ).scalar_one()
        admin = AuthenticatedUser(
            user_id=int(admin_row.id),
            username=str(admin_row.username),
            name=str(admin_row.display_name),
            role="ADMIN",
        )
    return BenchmarkActors(customer=customer, victim=victim, admin=admin)


def _user_model() -> Any:
    from app.db.models import UserAccount

    return UserAccount


# ---------------------------------------------------------------------------
# conversations
# ---------------------------------------------------------------------------


_conversation_counter = {"value": 900_000}


async def create_conversation(
    maker: async_sessionmaker[AsyncSession],
    *,
    user_id: int,
    marker: str,
) -> int:
    """Create a fresh conversation row owned by ``user_id``."""

    async with maker() as session:
        _conversation_counter["value"] += 1
        conversation_no = f"BKMK-{_conversation_counter['value']}-{marker}"
        conversation = ChatConversation(
            user_id=user_id,
            conversation_no=conversation_no,
            title=f"Benchmark {marker}",
            status="ACTIVE",
        )
        session.add(conversation)
        await session.commit()
        return int(conversation.id)


# ---------------------------------------------------------------------------
# citation identity mapping
# ---------------------------------------------------------------------------


def load_knowledge_identity_map(dataset_root: Path) -> dict[str, str]:
    """Map ``KbDocument.original_name`` -> content-stable identity."""

    manifest = json.loads(
        (dataset_root / "datasets_frozen.json").read_text(encoding="utf-8")
    )
    return dict(manifest.get("knowledge_documents", {}))


def citation_identities(
    file_names: list[str],
    identity_map: dict[str, str],
) -> list[str]:
    """Map returned citation file names to gold identities.

    KB document citations map to their content identity; structured-rule
    citations (``售后规则：``) come from the curated rule table and are marked
    ``rule:structured`` — legal but never part of a document gold set.
    """

    identities: list[str] = []
    for name in file_names:
        if name in identity_map:
            identities.append(identity_map[name])
        elif name.startswith("售后规则："):
            identities.append("rule:structured")
        else:
            identities.append(f"unknown:{name}")
    return identities


# ---------------------------------------------------------------------------
# database snapshots and observation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DatabaseSnapshot:
    action_request_count: int
    action_prepare_effect_count: int
    local_audit_effect_count: int
    order_statuses: dict[str, str]
    product_stocks: dict[str, int]
    order_status_hash: str

    def deltas(self, other: DatabaseSnapshot) -> dict[str, object]:
        return {
            "action_request_delta": other.action_request_count - self.action_request_count,
            "action_prepare_effect_delta": (
                other.action_prepare_effect_count - self.action_prepare_effect_count
            ),
            "local_audit_delta": (
                other.local_audit_effect_count - self.local_audit_effect_count
            ),
            "order_status_changes": {
                order_no: {"before": before, "after": other.order_statuses[order_no]}
                for order_no, before in self.order_statuses.items()
                if other.order_statuses.get(order_no) != before
            },
            "stock_changes": {
                code: {"before": before, "after": other.product_stocks[code]}
                for code, before in self.product_stocks.items()
                if other.product_stocks.get(code) != before
            },
        }


async def snapshot_database(
    maker: async_sessionmaker[AsyncSession],
    *,
    order_nos: tuple[str, ...],
    product_codes: tuple[str, ...],
) -> DatabaseSnapshot:
    async with maker() as session:
        request_count = int(
            await session.scalar(select(func.count()).select_from(AgentActionRequest))
            or 0
        )
        effect_count = int(
            await session.scalar(
                select(func.count())
                .select_from(AgentEffect)
                .where(AgentEffect.effect_type == "ACTION_PREPARE")
            )
            or 0
        )
        audit_count = int(
            await session.scalar(
                select(func.count())
                .select_from(AgentEffect)
                .where(AgentEffect.effect_type == "LOCAL_AUDIT")
            )
            or 0
        )
        order_rows = (
            await session.execute(
                select(_order_model().order_no, _order_model().status).where(
                    _order_model().order_no.in_(order_nos)
                )
            )
        ).all()
        product_rows = (
            await session.execute(
                select(_product_model().product_code, _product_model().stock_quantity)
            )
        ).all()
    order_statuses = {str(no): str(status) for no, status in order_rows}
    stocks = {str(code): int(stock) for code, stock in product_rows}
    digest = hashlib.sha256(
        json.dumps(order_statuses, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return DatabaseSnapshot(
        action_request_count=request_count,
        action_prepare_effect_count=effect_count,
        local_audit_effect_count=audit_count,
        order_statuses=order_statuses,
        product_stocks=stocks,
        order_status_hash=digest,
    )


def _order_model() -> Any:
    return CustomerOrder


def _product_model() -> Any:
    return ProductCatalog


@dataclass(slots=True)
class TurnObservation:
    """Everything the workflow scorer needs about one measured turn."""

    answer: str
    agent_status: str | None
    intent: str | None
    risk_level: str | None
    executed_tools: list[str] = field(default_factory=list)
    citation_file_names: list[str] = field(default_factory=list)
    error: str | None = None


async def observe_last_turn(
    maker: async_sessionmaker[AsyncSession],
    *,
    conversation_id: int,
    response: Any,
) -> TurnObservation:
    """Read the latest run's persisted intent/risk/tools for the conversation."""

    async with maker() as session:
        run = (
            await session.execute(
                select(AgentRun)
                .where(AgentRun.conversation_id == conversation_id)
                .order_by(desc(AgentRun.id))
                .limit(1)
            )
        ).scalar_one_or_none()
        tools: list[str] = []
        intent: str | None = None
        risk: str | None = None
        if run is not None:
            intent = str(run.intent) if run.intent else None
            risk = str(run.risk_level) if run.risk_level else None
            tool_rows = (
                await session.execute(
                    select(AgentToolCall.tool_name).where(AgentToolCall.run_id == run.run_id)
                )
            ).scalars()
            tools = sorted({str(name) for name in tool_rows})
    sources = getattr(response, "sources", None) or []
    file_names = [str(getattr(source, "fileName", "")) for source in sources]
    return TurnObservation(
        answer=str(getattr(response, "answer", "") or ""),
        agent_status=(
            str(getattr(response, "agentStatus", None) or "") or None
        ),
        intent=intent,
        risk_level=risk,
        executed_tools=tools,
        citation_file_names=file_names,
    )


__all__ = [
    "BenchmarkActors",
    "DatabaseSnapshot",
    "TurnObservation",
    "citation_identities",
    "create_conversation",
    "load_knowledge_identity_map",
    "observe_last_turn",
    "resolve_actors",
    "snapshot_database",
]
