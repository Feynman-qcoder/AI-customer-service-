# ruff: noqa: E402
"""One-command local bootstrap for the 智服通 Agent backend.

Replaces the manual first-run sequence:

    1. ``alembic upgrade head``            -> create / upgrade the MySQL schema
    2. ``scripts/seed_demo.py``            -> accounts, products, orders, rules
    3. ``scripts/sync_demo_knowledge.py``  -> knowledge chunks + Qdrant vectors

Step 3 now reads the markdown files under ``sample-data/knowledge/`` and pushes
them through ``DocumentProcessingService`` -- the same chain the Web upload
endpoint uses -- so a bootstrapped document is indistinguishable from an
uploaded one: PENDING -> PROCESSING -> READY, chunked, embedded and upserted
into Qdrant. The files under ``sample-data/knowledge/`` are the source of truth.

Examples (run from the repository root, or from ``server/``)::

    .venv\\Scripts\\python.exe scripts\\bootstrap_local.py
    .venv\\Scripts\\python.exe scripts\\bootstrap_local.py --reprocess
    .venv\\Scripts\\python.exe scripts\\bootstrap_local.py --skip-migrate --skip-seed

The exit code is 0 only when every knowledge document reaches READY.
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SERVER_DIR.parent
DEFAULT_KNOWLEDGE_DIR = PROJECT_ROOT / "sample-data" / "knowledge"
KNOWLEDGE_SUFFIXES = {".md", ".markdown", ".txt"}

if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import AppError
from app.db.models import DocumentProcessingTask, KbDocument, UserAccount
from app.db.session import dispose_engine, session_factory
from app.services.document_processing_service import document_processing_service
from scripts.seed_demo import main as seed_main

UNCHANGED = "UNCHANGED"


def _run_migrations() -> None:
    """Run ``alembic upgrade head`` in a child process.

    A subprocess keeps Alembic's ``fileConfig`` logging setup from disabling the
    application loggers we need for readable ingestion errors, and mirrors the
    exact command a developer would type by hand.
    """
    print(f"    running: {Path(sys.executable).name} -m alembic upgrade head")
    completed = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=str(SERVER_DIR),
        check=False,
    )
    if completed.returncode != 0:
        raise AppError(f"alembic upgrade head 失败，退出码 {completed.returncode}", 500)


async def _resolve_uploader(session: AsyncSession) -> int:
    """Pick the account that owns bootstrapped documents.

    Knowledge documents have an ``uploaded_by`` foreign key, so attributing them
    to the demo administrator keeps the audit trail honest instead of leaving a
    dangling NULL.
    """
    for statement in (
        select(UserAccount).where(UserAccount.username == settings.demo_admin_username),
        select(UserAccount).order_by(UserAccount.id).limit(1),
    ):
        user = (await session.execute(statement)).scalar_one_or_none()
        if user is not None:
            return int(user.id)
    raise AppError("数据库中没有可用用户，无法归属知识库文档；请先执行 seed（默认会执行）", 500)


async def _ingest_document(session: AsyncSession, path: Path, uploaded_by: int, reprocess: bool) -> str:
    """Ingest one file and return its final status (or ``UNCHANGED``).

    Documents are keyed by ``original_name``: an unchanged file is skipped, a
    changed file is deleted (chunks, vectors and stored copy included) and
    re-uploaded, which makes repeated runs idempotent.
    """
    content = path.read_bytes()
    digest = sha256(content).hexdigest()
    existing = (
        await session.execute(select(KbDocument).where(KbDocument.original_name == path.name))
    ).scalar_one_or_none()

    if (
        existing is not None
        and not reprocess
        and existing.file_sha256 == digest
        and existing.status == "READY"
    ):
        return UNCHANGED

    if existing is not None:
        await document_processing_service.delete(session, existing.id)

    document = await document_processing_service.save_upload(
        session, path.name, content, uploaded_by=uploaded_by
    )
    task = (
        await session.execute(
            select(DocumentProcessingTask).where(DocumentProcessingTask.document_id == document.id)
        )
    ).scalar_one_or_none()
    if task is not None and task.status in {"PENDING", "PROCESSING"}:
        await document_processing_service.process_task(session, task.id)

    await session.refresh(document)
    return document.status


async def _ingest_knowledge(knowledge_dir: Path, reprocess: bool) -> int:
    if not knowledge_dir.is_dir():
        raise AppError(f"知识库目录不存在: {knowledge_dir}", 400)
    files = sorted(
        path for path in knowledge_dir.iterdir() if path.is_file() and path.suffix.lower() in KNOWLEDGE_SUFFIXES
    )
    if not files:
        print(f"[3/3] knowledge: no .md/.txt files under {knowledge_dir}")
        return 0

    print(f"[3/3] knowledge: ingesting {len(files)} file(s) from {knowledge_dir}")
    factory = session_factory()
    ready = unchanged = failed = 0
    async with factory() as session:
        uploaded_by = await _resolve_uploader(session)
        for path in files:
            status = await _ingest_document(session, path, uploaded_by, reprocess)
            if status == UNCHANGED:
                unchanged += 1
            elif status == "READY":
                ready += 1
            else:
                failed += 1
            print(f"    - {path.name}: {status}")

    print(
        f"[3/3] knowledge: ready={ready} unchanged={unchanged} failed={failed} "
        f"(storage={settings.document_storage_path})"
    )
    return 0 if failed == 0 else 1


async def main(
    *,
    knowledge_dir: Path,
    skip_migrate: bool = False,
    skip_seed: bool = False,
    skip_knowledge: bool = False,
    reprocess: bool = False,
) -> int:
    if skip_migrate:
        print("[1/3] migrations: skipped")
    else:
        print("[1/3] migrations: alembic upgrade head")
        _run_migrations()

    if skip_seed:
        print("[2/3] demo seed: skipped")
    else:
        print("[2/3] demo seed: accounts / products / orders / structured rules")
        await seed_main()

    if skip_knowledge:
        print("[3/3] knowledge: skipped")
        return 0

    return await _ingest_knowledge(knowledge_dir, reprocess)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap the 智服通 Agent local schema, demo data and knowledge base."
    )
    parser.add_argument(
        "--knowledge-dir",
        default=str(DEFAULT_KNOWLEDGE_DIR),
        help="Directory scanned for .md/.txt knowledge files (default: sample-data/knowledge).",
    )
    parser.add_argument("--skip-migrate", action="store_true", help="Skip 'alembic upgrade head'.")
    parser.add_argument("--skip-seed", action="store_true", help="Skip demo business data seeding.")
    parser.add_argument("--skip-knowledge", action="store_true", help="Skip knowledge ingestion.")
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Re-ingest documents even when the stored content hash is unchanged.",
    )
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    try:
        return await main(
            knowledge_dir=Path(str(args.knowledge_dir)),
            skip_migrate=bool(args.skip_migrate),
            skip_seed=bool(args.skip_seed),
            skip_knowledge=bool(args.skip_knowledge),
            reprocess=bool(args.reprocess),
        )
    finally:
        await dispose_engine()


if __name__ == "__main__":
    sys.exit(asyncio.run(_run(_parse_args(sys.argv[1:]))))
