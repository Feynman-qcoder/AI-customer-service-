"""CLI for RESUME_AGENT_BENCHMARK_V1.

Subcommands:
    validate   — re-verify the frozen datasets (contracts, SHA-256, dedup gate)
    workflow   — 120-case strict workflow profile (mock-mode isolated stack)
    rag-real   — 7-scheme retrieval ablation (needs a REAL embedding provider)
    security   — 60-case adversarial safety profile (mock-mode isolated stack)
    recovery   — 40-trial fault-injection profile (mock-mode isolated stack)
    llm-real   — 60 real provider calls (needs a REAL LLM provider)
    report     — aggregate evidence into benchmark_report.md + resume sentence
    verify     — re-hash an evidence directory and evaluate the hard gates

Every profile runs against a freshly created isolated MySQL/Qdrant/Redis
stack, refuses the demo database, forbids skips, and writes full evidence
outside the repository. Exit codes are non-zero whenever a dependency is
missing, mock state is wrong, the demo database is targeted, a case is
skipped, or evidence is incomplete.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from evals.benchmark.datasets import BENCHMARK_VERSION

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SERVER_DIR = _REPO_ROOT / "server"
_DATASET_ROOT = _SERVER_DIR / "evals" / "datasets" / "resume_benchmark_v1"

PROFILES_REQUIRING_MOCK = ("workflow", "security", "recovery")


class BenchmarkCLIError(RuntimeError):
    """The benchmark run cannot proceed."""


def _evidence_writer(profile: str, evidence_dir: Path | None):
    from evals.benchmark.evidence import EvidenceWriter, resolve_evidence_dir

    if evidence_dir is not None:
        evidence_dir.mkdir(parents=True, exist_ok=True)
        directory = evidence_dir
    else:
        directory = resolve_evidence_dir(profile=profile)
    return EvidenceWriter(directory=directory, profile=profile)


def _validate_mock_state(profile: str, *, llm_mock_enabled: bool, embedding_mock_enabled: bool) -> None:
    if profile in PROFILES_REQUIRING_MOCK:
        if not llm_mock_enabled or not embedding_mock_enabled:
            raise BenchmarkCLIError(
                f"{profile} profile must run with LLM_MOCK_ENABLED=true and "
                "EMBEDDING_MOCK_ENABLED=true (deterministic engineering chain)"
            )


def _settings_snapshot() -> dict[str, object]:
    from app.core.config import settings

    return {
        "llm_mock_enabled": settings.llm_mock_enabled,
        "embedding_mock_enabled": settings.embedding_mock_enabled,
        "llm_provider": "mock" if settings.llm_mock_enabled else settings.llm_base_url,
        "llm_model": settings.llm_model_name,
        "llm_temperature": settings.llm_temperature,
        "llm_max_completion_tokens": settings.llm_max_completion_tokens,
        "llm_request_timeout_seconds": settings.llm_request_timeout_seconds,
        "embedding_provider": (
            "mock" if settings.embedding_mock_enabled else settings.embedding_base_url
        ),
        "embedding_model": settings.embedding_model_name,
        "durable_customer_interrupt_enabled": settings.durable_customer_interrupt_enabled,
        "checkpoint_backend": settings.checkpoint_backend,
    }


async def _cmd_validate(args: argparse.Namespace) -> int:
    from evals.benchmark.generate_datasets import generate_and_freeze

    del args
    return generate_and_freeze(_REPO_ROOT, check_only=True)


async def _run_isolated_profile(
    profile: str,
    args: argparse.Namespace,
    *,
    llm_mock_enabled: bool,
    embedding_mock_enabled: bool,
) -> int:
    from evals.benchmark import isolated_stack
    from evals.benchmark.env_guard import assert_not_demo_database
    from evals.benchmark.evidence import EvidenceError

    marker = f"{profile}-{getattr(args, 'seed', os.environ.get('BENCHMARK_SEED', '20261008'))}"
    evidence = _evidence_writer(profile, getattr(args, "evidence_dir", None))
    evidence.log(f"benchmark={BENCHMARK_VERSION} profile={profile} marker={marker}")
    stack = None
    runtime = None
    removed_containers: list[str] = []
    exit_code = 0
    try:
        stack = isolated_stack.start_isolated_stack(marker=marker)
        isolated_stack.apply_environment(stack, stack.work_dir / "checkpoints")
        from app.core.config import settings

        assert_not_demo_database(
            mysql_host=settings.mysql_host, mysql_port=int(settings.mysql_port)
        )
        _validate_mock_state(
            profile,
            llm_mock_enabled=settings.llm_mock_enabled,
            embedding_mock_enabled=settings.embedding_mock_enabled,
        )
        evidence.log(
            f"isolated stack ready: mysql:{stack.mysql_port} "
            f"qdrant:{stack.qdrant_port} redis:{stack.redis_port}"
        )
        isolated_stack.run_alembic_upgrade(_SERVER_DIR)
        await isolated_stack.seed_benchmark_database(_REPO_ROOT)
        ingest_status = await isolated_stack.ingest_knowledge_documents(_REPO_ROOT)
        evidence.log(f"knowledge ingestion: {ingest_status}")

        from evals.benchmark.runtime import build_settings, start_service_runtime

        configured = build_settings(
            checkpoint_dir=stack.work_dir / "checkpoints",
            llm_mock_enabled=llm_mock_enabled,
            embedding_mock_enabled=embedding_mock_enabled,
        )
        runtime = await start_service_runtime(
            settings=configured, work_dir=stack.work_dir
        )
        evidence.log("service runtime started (durable interrupts enabled)")

        if profile == "workflow":
            from evals.benchmark.workflow_profile import run_workflow_profile

            summary = await run_workflow_profile(runtime, _DATASET_ROOT, evidence)
        elif profile == "security":
            from evals.benchmark.security_profile import run_security_profile

            summary = await run_security_profile(runtime, _DATASET_ROOT, evidence)
        elif profile == "recovery":
            from evals.benchmark.recovery_profile import run_recovery_profile

            summary = await run_recovery_profile(runtime, _DATASET_ROOT, evidence)
        else:  # pragma: no cover — guarded by caller
            raise BenchmarkCLIError(f"unknown profile: {profile}")

        skipped = _count_skips(summary)
        if skipped:
            evidence.log(f"SKIP DETECTED: {skipped}")
            exit_code = 3
        evidence.log(f"{profile} summary: {summary}")
    except Exception as caught:  # noqa: BLE001
        evidence.log(f"FATAL {type(caught).__name__}: {caught}")
        exit_code = 1
    finally:
        if runtime is not None:
            try:
                await runtime.stop()
            except Exception as caught:  # noqa: BLE001
                evidence.log(f"runtime stop failed: {caught}")
        if stack is not None:
            removed = isolated_stack.teardown_isolated_stack(stack)
            removed_containers.extend(removed)
            evidence.log(f"torn down containers: {removed}")
    try:
        evidence.finalize(
            repository_root=_REPO_ROOT,
            dataset_root=_DATASET_ROOT,
            stack=stack,
            settings_snapshot=_settings_snapshot(),
            run={"profile": profile, "exit_code": exit_code},
            profile_runs=[],
        )
    except EvidenceError as caught:
        print(f"EVIDENCE_ERROR: {caught}")
        return 4
    print(f"EVIDENCE_DIR={evidence.directory}")
    print(f"PROFILE={profile} EXIT_CODE={exit_code}")
    return exit_code


def _count_skips(summary: dict[str, object]) -> int:
    text = str(summary)
    return text.count("NOT_RUN") + text.count("SKIPPED")


async def _cmd_rag_real(args: argparse.Namespace) -> int:
    from evals.benchmark import isolated_stack
    from evals.benchmark.env_guard import (
        assert_not_demo_database,
        real_provider_status,
    )

    evidence = _evidence_writer("rag-real", args.evidence_dir)
    stack = None
    exit_code = 0
    try:
        stack = isolated_stack.start_isolated_stack(
            marker=f"rag-real-{args.seed}",
        )
        isolated_stack.apply_environment(stack, stack.work_dir / "checkpoints")
        from app.core.config import settings

        assert_not_demo_database(
            mysql_host=settings.mysql_host, mysql_port=int(settings.mysql_port)
        )
        if settings.llm_mock_enabled or settings.embedding_mock_enabled:
            # keep the LLM mock state as-is; the ablation only needs embeddings
            pass
        status = await real_provider_status(
            llm_mock_enabled=settings.embedding_mock_enabled,
            embedding_mock_enabled=settings.embedding_mock_enabled,
            llm_api_key=settings.llm_api_key,
            llm_base_url=settings.llm_base_url,
            llm_model_name=settings.llm_model_name,
            embedding_api_key=settings.embedding_api_key,
            embedding_base_url=settings.embedding_base_url,
            embedding_model_name=settings.embedding_model_name,
        )
        if not status.embedding_available:
            evidence.log(f"rag-real NOT_RUN: {status.reason}")
            from evals.benchmark.rag_profile import run_rag_real_profile

            summary = await run_rag_real_profile(
                None, _DATASET_ROOT, evidence, provider_status=status
            )
            evidence.log(f"rag-real summary: {summary}")
            exit_code = 2  # NOT_RUN: no score produced
        else:
            isolated_stack.run_alembic_upgrade(_SERVER_DIR)
            await isolated_stack.seed_benchmark_database(_REPO_ROOT)
            await isolated_stack.ingest_knowledge_documents(_REPO_ROOT)
            from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

            from evals.benchmark.rag_profile import run_rag_real_profile

            engine = create_async_engine(
                f"mysql+aiomysql://{settings.mysql_username}:{settings.mysql_password}"
                f"@{settings.mysql_host}:{settings.mysql_port}/{settings.mysql_database}"
            )
            maker = async_sessionmaker(engine)
            try:
                summary = await run_rag_real_profile(
                    maker, _DATASET_ROOT, evidence, provider_status=status
                )
                evidence.log(f"rag-real summary: {summary}")
            finally:
                await engine.dispose()
    except Exception as caught:  # noqa: BLE001
        evidence.log(f"FATAL {type(caught).__name__}: {caught}")
        exit_code = 1
    finally:
        if stack is not None:
            removed = isolated_stack.teardown_isolated_stack(stack)
            evidence.log(f"torn down containers: {removed}")
    evidence.finalize(
        repository_root=_REPO_ROOT,
        dataset_root=_DATASET_ROOT,
        stack=stack,
        settings_snapshot=_settings_snapshot(),
        run={"profile": "rag-real", "exit_code": exit_code},
        profile_runs=[],
    )
    print(f"EVIDENCE_DIR={evidence.directory}")
    print(f"PROFILE=rag-real EXIT_CODE={exit_code}")
    return exit_code


async def _cmd_llm_real(args: argparse.Namespace) -> int:
    from evals.benchmark import isolated_stack
    from evals.benchmark.env_guard import (
        assert_not_demo_database,
        real_provider_status,
    )

    evidence = _evidence_writer("llm-real", args.evidence_dir)
    stack = None
    runtime = None
    exit_code = 0
    try:
        stack = isolated_stack.start_isolated_stack(marker=f"llm-real-{args.seed}")
        isolated_stack.apply_environment(stack, stack.work_dir / "checkpoints")
        from app.core.config import settings

        assert_not_demo_database(
            mysql_host=settings.mysql_host, mysql_port=int(settings.mysql_port)
        )
        status = await real_provider_status(
            llm_mock_enabled=settings.llm_mock_enabled,
            embedding_mock_enabled=settings.embedding_mock_enabled,
            llm_api_key=settings.llm_api_key,
            llm_base_url=settings.llm_base_url,
            llm_model_name=settings.llm_model_name,
            embedding_api_key=settings.embedding_api_key,
            embedding_base_url=settings.embedding_base_url,
            embedding_model_name=settings.embedding_model_name,
        )
        if not status.llm_available:
            evidence.log(f"llm-real NOT_RUN: {status.reason}")
            from evals.benchmark.llm_profile import run_llm_real_profile

            summary = await run_llm_real_profile(
                None, _DATASET_ROOT, evidence, provider_status=status, seed=args.seed
            )
            evidence.log(f"llm-real summary: {summary}")
            exit_code = 2
        else:
            if settings.llm_mock_enabled or settings.embedding_mock_enabled:
                raise BenchmarkCLIError(
                    "llm-real requires LLM_MOCK_ENABLED=false and "
                    "EMBEDDING_MOCK_ENABLED=false in the environment"
                )
            isolated_stack.run_alembic_upgrade(_SERVER_DIR)
            await isolated_stack.seed_benchmark_database(_REPO_ROOT)
            await isolated_stack.ingest_knowledge_documents(_REPO_ROOT)
            from evals.benchmark.runtime import build_settings, start_service_runtime

            configured = build_settings(
                checkpoint_dir=stack.work_dir / "checkpoints",
                llm_mock_enabled=False,
                embedding_mock_enabled=False,
            )
            runtime = await start_service_runtime(
                settings=configured, work_dir=stack.work_dir
            )
            from evals.benchmark.llm_profile import run_llm_real_profile

            summary = await run_llm_real_profile(
                runtime, _DATASET_ROOT, evidence, provider_status=status, seed=args.seed
            )
            evidence.log(f"llm-real summary: {summary}")
    except Exception as caught:  # noqa: BLE001
        evidence.log(f"FATAL {type(caught).__name__}: {caught}")
        exit_code = 1
    finally:
        if runtime is not None:
            try:
                await runtime.stop()
            except Exception as caught:  # noqa: BLE001
                evidence.log(f"runtime stop failed: {caught}")
        if stack is not None:
            removed = isolated_stack.teardown_isolated_stack(stack)
            evidence.log(f"torn down containers: {removed}")
    evidence.finalize(
        repository_root=_REPO_ROOT,
        dataset_root=_DATASET_ROOT,
        stack=stack,
        settings_snapshot=_settings_snapshot(),
        run={"profile": "llm-real", "exit_code": exit_code},
        profile_runs=[],
    )
    print(f"EVIDENCE_DIR={evidence.directory}")
    print(f"PROFILE=llm-real EXIT_CODE={exit_code}")
    return exit_code


def _cmd_report(args: argparse.Namespace) -> int:
    from evals.benchmark.report import generate_report

    return generate_report(evidence_dir=Path(args.evidence_dir))


def _cmd_verify(args: argparse.Namespace) -> int:
    from evals.benchmark.evidence import verify_evidence_dir
    from evals.benchmark.report import evaluate_gates

    directory = Path(args.evidence_dir).resolve()
    problems = verify_evidence_dir(directory)
    gates = evaluate_gates(directory)
    for problem in problems:
        print(f"EVIDENCE_PROBLEM: {problem}")
    for gate, value in gates["gates"].items():
        print(f"GATE {gate} = {value}")
    print(f"READY_FOR_RESUME_CLAIM = {gates['ready']}")
    return 0 if not problems and gates["ready"] else 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--evidence-dir", type=Path, default=None)
        sub.add_argument("--seed", type=int, default=None)

    validate = subparsers.add_parser("validate")
    validate.set_defaults(handler=_cmd_validate)

    for name in ("workflow", "security", "recovery"):
        sub = subparsers.add_parser(name)
        add_common(sub)
        sub.set_defaults(handler=_make_profile_handler(name))

    rag = subparsers.add_parser("rag-real")
    add_common(rag)
    rag.set_defaults(handler=_cmd_rag_real)

    llm = subparsers.add_parser("llm-real")
    add_common(llm)
    llm.set_defaults(handler=_cmd_llm_real)

    report = subparsers.add_parser("report")
    report.add_argument("--evidence-dir", type=Path, required=True)
    report.set_defaults(handler=_cmd_report)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--evidence-dir", type=Path, required=True)
    verify.set_defaults(handler=_cmd_verify)
    return parser


def _make_profile_handler(profile: str):
    async def handler(args: argparse.Namespace) -> int:
        return await _run_isolated_profile(
            profile,
            args,
            llm_mock_enabled=True,
            embedding_mock_enabled=True,
        )

    return handler


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "seed", None) is None:
        args.seed = int(os.environ.get("BENCHMARK_SEED", "20261008"))
    handler = args.handler
    result = handler(args)
    if asyncio.iscoroutine(result):
        result = asyncio.run(result)
    return int(result)


if __name__ == "__main__":
    sys.exit(main())
