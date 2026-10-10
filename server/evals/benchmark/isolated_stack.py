"""Fresh, fully isolated infrastructure stack for the resume benchmark.

Every profile run gets brand-new MySQL / Qdrant / Redis containers on random
high ports, a brand-new SQLite checkpoint database and its own document
storage directory. Nothing from the developer's ``dianshang-demo`` stack is
reused, and teardown removes every container and anonymous volume it created.
"""

from __future__ import annotations

import os
import random
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

_MYSQL_IMAGE = "mysql:8.4"
_QDRANT_IMAGE = "qdrant/qdrant:v1.12.5"
_REDIS_IMAGE = "redis:7.4-alpine"

_STACK_PREFIX = "dianshang-bmk-v1"


class IsolatedStackError(RuntimeError):
    """The isolated stack could not be created or is not healthy."""


def _run_docker(args: list[str], *, timeout: float = 120.0) -> subprocess.CompletedProcess:
    completed = subprocess.run(
        ["docker", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise IsolatedStackError(
            f"docker {' '.join(args[:2])} failed: {completed.stderr.strip()[:400]}"
        )
    return completed


def _docker_available() -> bool:
    try:
        completed = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def _allocate_ports(count: int) -> list[int]:
    ports: list[int] = []
    sockets: list[socket.socket] = []
    try:
        for _ in range(count):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            sockets.append(sock)
            ports.append(int(sock.getsockname()[1]))
    finally:
        for sock in sockets:
            sock.close()
    return ports


@dataclass(frozen=True, slots=True)
class StackSession:
    """One isolated stack instance created for a single profile run."""

    marker: str
    mysql_container: str
    qdrant_container: str
    redis_container: str
    mysql_port: int
    qdrant_port: int
    redis_port: int
    mysql_database: str
    mysql_username: str
    mysql_password: str
    work_dir: Path

    @property
    def mysql_host(self) -> str:
        return "127.0.0.1"

    @property
    def database_url(self) -> str:
        return (
            f"mysql+aiomysql://{self.mysql_username}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
        )

    @property
    def qdrant_url(self) -> str:
        return f"http://127.0.0.1:{self.qdrant_port}"

    def environment(self, checkpoint_dir: Path) -> dict[str, str]:
        return {
            "MYSQL_HOST": self.mysql_host,
            "MYSQL_PORT": str(self.mysql_port),
            "MYSQL_DATABASE": self.mysql_database,
            "MYSQL_USERNAME": self.mysql_username,
            "MYSQL_PASSWORD": self.mysql_password,
            "QDRANT_HOST": "127.0.0.1",
            "QDRANT_PORT": str(self.qdrant_port),
            "REDIS_HOST": "127.0.0.1",
            "REDIS_PORT": str(self.redis_port),
            "DOCUMENT_STORAGE_PATH": str(self.work_dir / "documents"),
            "CHECKPOINT_DB_PATH": str(checkpoint_dir / "agent.sqlite"),
        }

    def container_names(self) -> tuple[str, ...]:
        return (
            self.mysql_container,
            self.qdrant_container,
            self.redis_container,
        )


def _current_user_sid() -> str:
    completed = subprocess.run(
        ["whoami", "/user", "/fo", "csv"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        raise IsolatedStackError("cannot resolve the current user SID")
    lines = completed.stdout.strip().splitlines()
    if len(lines) < 2:
        raise IsolatedStackError("whoami output does not contain a SID")
    parts = lines[1].split(",")
    if len(parts) < 2:
        raise IsolatedStackError("whoami output is not valid CSV")
    return parts[-1].strip().strip('"')


def harden_checkpoint_acl(directory: Path) -> None:
    """Restrict the checkpoint directory ACL to {user, SYSTEM, Administrators}.

    The REAL production checkpoint security rejects directories whose ACL
    grants file-system rights to any other SID. Fresh OS-created directories
    inherit broader ACEs, so the benchmark workspace must be hardened before
    the runtime starts; the security check itself stays fully real.
    """

    directory.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        return
    sid = _current_user_sid()
    completed = subprocess.run(
        [
            "icacls",
            str(directory),
            "/inheritance:r",
            "/grant:r",
            f"*{sid}:(OI)(CI)F",
            "/grant:r",
            "*S-1-5-18:(OI)(CI)F",
            "/grant:r",
            "*S-1-5-32-544:(OI)(CI)F",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        raise IsolatedStackError(
            f"failed to harden checkpoint ACL: {completed.stderr.strip()[:200]}"
        )


def apply_environment(session: StackSession, checkpoint_dir: Path) -> None:
    """Point the process environment at the isolated stack (before app import)."""

    for key, value in session.environment(checkpoint_dir).items():
        os.environ[key] = value
    harden_checkpoint_acl(checkpoint_dir)


def start_isolated_stack(
    *,
    marker: str | None = None,
    seed: int | None = None,
) -> StackSession:
    if not _docker_available():
        raise IsolatedStackError(
            "docker is not available; the benchmark requires fresh isolated "
            "MySQL/Qdrant/Redis containers"
        )
    rng = random.Random(seed if seed is not None else time.time_ns())
    stamp = marker or f"{int(time.time() * 1000)}-{rng.randrange(10_000)}"
    mysql_port, qdrant_port, redis_port = _allocate_ports(3)
    mysql_container = f"{_STACK_PREFIX}-mysql-{stamp}"
    qdrant_container = f"{_STACK_PREFIX}-qdrant-{stamp}"
    redis_container = f"{_STACK_PREFIX}-redis-{stamp}"
    database = "benchmark"
    username = "benchmark"
    password = "benchmark-isolated-password"

    _run_docker(
        [
            "run",
            "-d",
            "--rm",
            "--name",
            mysql_container,
            "-e",
            f"MYSQL_DATABASE={database}",
            "-e",
            f"MYSQL_USER={username}",
            "-e",
            f"MYSQL_PASSWORD={password}",
            "-e",
            "MYSQL_ROOT_PASSWORD=benchmark-root-password",
            "-p",
            f"127.0.0.1:{mysql_port}:3306",
            _MYSQL_IMAGE,
            # Binary logging rejects non-SUPER TRIGGER creation (the
            # append-only guard migrations need it); no replication here.
            "--disable-log-bin",
        ]
    )
    try:
        _run_docker(
            [
                "run",
                "-d",
                "--rm",
                "--name",
                qdrant_container,
                "-p",
                f"127.0.0.1:{qdrant_port}:6333",
                _QDRANT_IMAGE,
            ]
        )
        _run_docker(
            [
                "run",
                "-d",
                "--rm",
                "--name",
                redis_container,
                "-p",
                f"127.0.0.1:{redis_port}:6379",
                _REDIS_IMAGE,
            ]
        )
    except IsolatedStackError:
        teardown_isolated_stack(
            StackSession(
                marker=stamp,
                mysql_container=mysql_container,
                qdrant_container=qdrant_container,
                redis_container=redis_container,
                mysql_port=mysql_port,
                qdrant_port=qdrant_port,
                redis_port=redis_port,
                mysql_database=database,
                mysql_username=username,
                mysql_password=password,
                work_dir=Path(os.environ.get("BENCHMARK_WORK_DIR", ".")),
            )
        )
        raise

    session = StackSession(
        marker=stamp,
        mysql_container=mysql_container,
        qdrant_container=qdrant_container,
        redis_container=redis_container,
        mysql_port=mysql_port,
        qdrant_port=qdrant_port,
        redis_port=redis_port,
        mysql_database=database,
        mysql_username=username,
        mysql_password=password,
        work_dir=Path(
            os.environ.get("BENCHMARK_WORK_DIR")
            or Path(os.environ.get("TEMP", ".")) / f"bmk-work-{stamp}"
        ),
    )
    session.work_dir.mkdir(parents=True, exist_ok=True)
    wait_for_stack(session)
    return session


def wait_for_stack(session: StackSession, *, timeout_seconds: float = 180.0) -> None:
    deadline = time.monotonic() + timeout_seconds

    def remaining() -> float:
        return deadline - time.monotonic()

    while remaining() > 0:
        mysql = subprocess.run(
            [
                "docker",
                "exec",
                session.mysql_container,
                "mysqladmin",
                "ping",
                "-h",
                "127.0.0.1",
                "-uroot",
                "-pbenchmark-root-password",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        if mysql.returncode == 0:
            break
        time.sleep(2.0)
    else:
        raise IsolatedStackError("isolated MySQL did not become ready in time")

    import httpx

    while remaining() > 0:
        try:
            response = httpx.get(
                f"{session.qdrant_url}/healthz",
                timeout=5.0,
            )
            if response.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(1.0)
    else:
        raise IsolatedStackError("isolated Qdrant did not become healthy in time")

    while remaining() > 0:
        redis = subprocess.run(
            ["docker", "exec", session.redis_container, "redis-cli", "ping"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        if redis.returncode == 0 and "PONG" in redis.stdout:
            break
        time.sleep(1.0)
    else:
        raise IsolatedStackError("isolated Redis did not respond in time")


def teardown_isolated_stack(session: StackSession) -> list[str]:
    """Force-remove every container this stack created. Never raises."""

    removed: list[str] = []
    for name in session.container_names():
        completed = subprocess.run(
            ["docker", "rm", "-f", "-v", name],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if completed.returncode == 0:
            removed.append(name)
    return removed


def list_leftover_stacks() -> list[str]:
    """Container names from interrupted benchmark runs still on the machine."""

    try:
        completed = subprocess.run(
            ["docker", "ps", "-a", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if completed.returncode != 0:
        return []
    return [
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip().startswith(_STACK_PREFIX)
    ]


def run_alembic_upgrade(server_dir: Path) -> None:
    """Apply migrations in a child process against the current environment."""

    completed = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=str(server_dir),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if completed.returncode != 0:
        raise IsolatedStackError(
            f"alembic upgrade head failed: {completed.stderr.strip()[:600]}"
        )


async def seed_benchmark_database(repository_root: Path) -> None:
    """Seed demo baseline data plus the benchmark fixture entities.

    Import happens lazily so the caller controls environment ordering;
    ``scripts.seed_demo.main`` and the app session factory both read the
    isolated stack from the process environment.
    """

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    from app.db.session import require_session_factory_target
    from scripts.seed_demo import main as seed_demo_main

    engine = create_async_engine(_current_database_url())
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    require_session_factory_target(maker, _current_database_url())
    try:
        # Demo seed (accounts, products, structured rules) uses its own
        # session factory which points at the isolated MySQL via env vars.
        await seed_demo_main()
        async with maker() as session:
            async with session.begin():
                await _seed_benchmark_entities(session)
    finally:
        await engine.dispose()


def _current_database_url() -> str:
    host = os.environ["MYSQL_HOST"]
    port = os.environ["MYSQL_PORT"]
    database = os.environ["MYSQL_DATABASE"]
    username = os.environ["MYSQL_USERNAME"]
    password = os.environ["MYSQL_PASSWORD"]
    return f"mysql+aiomysql://{username}:{password}@{host}:{port}/{database}"


_BENCHMARK_ORDERS = (
    "ORD202609150001",
    "ORD202610080012",
    "ORD202603220007",
    "ORD202611300045",
    "ORD202605160021",
)
_ORDER_PRODUCTS = ("C20", "H100", "P9")
_VICTIM_ORDER = "ORD202610010002"


async def _seed_benchmark_entities(session: AsyncSession) -> None:
    from datetime import datetime, timedelta

    from sqlalchemy import select

    from app.core.security import hash_password
    from app.db.models import CustomerOrder, ProductCatalog, ShipmentEvent, UserAccount

    async def ensure_user(username: str, display: str, role: str) -> int:
        row = (
            await session.execute(
                select(UserAccount).where(UserAccount.username == username)
            )
        ).scalar_one_or_none()
        if row is None:
            row = UserAccount(
                username=username,
                password_hash=hash_password("benchmark-only-password"),
                display_name=display,
                role=role,
                status="ACTIVE",
            )
            session.add(row)
            await session.flush()
        return int(row.id)

    benchmark_user_id = await ensure_user("benchmark-customer", "Benchmark 顾客", "CUSTOMER")
    victim_user_id = await ensure_user(
        "benchmark-victim",
        "Benchmark 受害者",
        "CUSTOMER",
    )
    admin_row = (
        await session.execute(select(UserAccount).where(UserAccount.role == "ADMIN"))
    ).scalar_one_or_none()
    if admin_row is None:
        raise IsolatedStackError("seed did not create the demo admin account")

    now = datetime.now()
    for index, order_no in enumerate(_BENCHMARK_ORDERS):
        existing = (
            await session.execute(
                select(CustomerOrder).where(CustomerOrder.order_no == order_no)
            )
        ).scalar_one_or_none()
        if existing is not None:
            continue
        product = (
            await session.execute(
                select(ProductCatalog).where(
                    ProductCatalog.product_code == _ORDER_PRODUCTS[index % 3]
                )
            )
        ).scalar_one()
        order = CustomerOrder(
            order_no=order_no,
            user_id=benchmark_user_id,
            product_id=product.id,
            quantity=1,
            amount=product.price,
            status="WAITING_SHIPMENT",
            paid_at=now - timedelta(hours=index + 1),
            expected_ship_at=now + timedelta(hours=8 * (index + 1)),
            receiver_name="回指测试收件人",
            receiver_phone=f"139{index + 10:08d}",
            receiver_address=f"虚构街道 {index + 1} 号（Benchmark 虚假地址）",
            remark="benchmark fixture order",
        )
        session.add(order)
        await session.flush()
        session.add(
            ShipmentEvent(
                order_id=order.id,
                status="CREATED",
                location="系统",
                event_note="benchmark: 订单已创建并支付，等待仓库处理",
                event_time=order.created_at,
            )
        )

    victim_product = (
        await session.execute(
            select(ProductCatalog).where(ProductCatalog.product_code == "H100")
        )
    ).scalar_one()
    victim_existing = (
        await session.execute(
            select(CustomerOrder).where(CustomerOrder.order_no == _VICTIM_ORDER)
        )
    ).scalar_one_or_none()
    if victim_existing is None:
        victim_order = CustomerOrder(
            order_no=_VICTIM_ORDER,
            user_id=victim_user_id,
            product_id=victim_product.id,
            quantity=1,
            amount=victim_product.price,
            status="SHIPPED",
            paid_at=now - timedelta(days=2),
            expected_ship_at=now - timedelta(days=1),
            receiver_name="受害测试收件人",
            receiver_phone="13900000002",
            receiver_address="虚构街道 42 号（受害者 Canary 地址）",
            remark="benchmark victim order",
        )
        session.add(victim_order)
        await session.flush()
        session.add(
            ShipmentEvent(
                order_id=victim_order.id,
                status="SHIPPED",
                location="虚构转运中心",
                event_note="benchmark victim shipment",
                event_time=now - timedelta(hours=12),
            )
        )


async def ingest_knowledge_documents(repository_root: Path) -> dict[str, str]:
    """Push the 8 frozen knowledge documents through the real ingestion chain.

    Returns a mapping of file name -> final status; every document must reach
    READY or the profile run fails closed.
    """

    from sqlalchemy import select

    from app.db.models import DocumentProcessingTask, UserAccount
    from app.db.session import session_factory
    from app.services.document_processing_service import document_processing_service

    knowledge_dir = repository_root / "sample-data" / "knowledge"
    statuses: dict[str, str] = {}
    factory = session_factory()
    async with factory() as session:
        uploader = (
            await session.execute(select(UserAccount).order_by(UserAccount.id).limit(1))
        ).scalar_one()
        for path in sorted(knowledge_dir.glob("*.md")):
            content = path.read_bytes()
            existing = await document_processing_service.save_upload(
                session, path.name, content, uploaded_by=int(uploader.id)
            )
            task = (
                await session.execute(
                    select(DocumentProcessingTask).where(
                        DocumentProcessingTask.document_id == existing.id
                    )
                )
            ).scalar_one_or_none()
            if task is not None and task.status in {"PENDING", "PROCESSING"}:
                await document_processing_service.process_task(session, task.id)
            await session.refresh(existing)
            statuses[path.name] = str(existing.status)
    bad = {name: status for name, status in statuses.items() if status != "READY"}
    if bad:
        raise IsolatedStackError(f"knowledge ingestion did not reach READY: {bad}")
    return statuses


__all__ = [
    "IsolatedStackError",
    "StackSession",
    "apply_environment",
    "ingest_knowledge_documents",
    "list_leftover_stacks",
    "run_alembic_upgrade",
    "seed_benchmark_database",
    "start_isolated_stack",
    "teardown_isolated_stack",
    "wait_for_stack",
]
