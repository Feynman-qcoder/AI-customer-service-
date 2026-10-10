"""Environment guards for the RESUME_AGENT_BENCHMARK_V1 CLI.

The benchmark must never touch the developer's demo database or leak real
credentials. ``assert_not_demo_database`` refuses any MySQL target that a
running ``dianshang-demo`` (or similarly named) container publishes, and
``real_provider_status`` reports whether a REAL LLM / embedding provider is
reachable WITHOUT ever printing the credentials themselves.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass

_DEMO_CONTAINER_PATTERN = re.compile(r"dianshang[-_]?demo", re.IGNORECASE)


class BenchmarkEnvironmentError(RuntimeError):
    """The environment is not safe for running the benchmark."""


def _docker_ps() -> str:
    try:
        completed = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}\t{{.Ports}}"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout


def _demo_published_ports() -> set[int]:
    ports: set[int] = set()
    for line in _docker_ps().splitlines():
        if not line.strip():
            continue
        name, _, port_part = line.partition("\t")
        if not _DEMO_CONTAINER_PATTERN.search(name):
            continue
        for match in re.finditer(r"0\.0\.0\.0:(\d+)->", port_part):
            ports.add(int(match.group(1)))
    return ports


def assert_not_demo_database(*, mysql_host: str, mysql_port: int) -> None:
    """Fail closed when the target MySQL port belongs to the demo stack."""

    demo_ports = _demo_published_ports()
    if mysql_port in demo_ports:
        raise BenchmarkEnvironmentError(
            f"refusing to run against the demo database: port {mysql_port} is "
            "published by a dianshang-demo container; the benchmark requires a "
            "freshly created isolated MySQL stack"
        )
    if mysql_port == 3306 and os.getenv("BENCHMARK_ALLOW_PORT_3306") != "1":
        raise BenchmarkEnvironmentError(
            "refusing default MySQL port 3306 (demo risk); start the isolated "
            "stack or set BENCHMARK_ALLOW_PORT_3306=1 for an explicitly "
            "verified non-demo target"
        )


@dataclass(frozen=True, slots=True)
class RealProviderStatus:
    """Reachability of real LLM / embedding providers (credentials excluded)."""

    llm_available: bool
    embedding_available: bool
    llm_provider: str
    llm_model: str
    embedding_provider: str
    embedding_model: str
    reason: str


async def _probe_http(base_url: str, timeout_seconds: float = 5.0) -> bool:
    try:
        import httpx
    except ImportError:
        return False
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            await client.get(base_url.rstrip("/"))
    except Exception:
        return False
    return True


async def real_provider_status(
    *,
    llm_mock_enabled: bool,
    embedding_mock_enabled: bool,
    llm_api_key: str,
    llm_base_url: str,
    llm_model_name: str,
    embedding_api_key: str,
    embedding_base_url: str,
    embedding_model_name: str,
) -> RealProviderStatus:
    reasons: list[str] = []
    llm_available = False
    embedding_available = False
    if llm_mock_enabled:
        reasons.append("LLM_MOCK_ENABLED=true")
    elif not llm_api_key:
        reasons.append("LLM_API_KEY missing")
    else:
        llm_available = await _probe_http(llm_base_url)
        if not llm_available:
            reasons.append("LLM provider unreachable")
    if embedding_mock_enabled:
        reasons.append("EMBEDDING_MOCK_ENABLED=true")
    elif not embedding_api_key:
        reasons.append("EMBEDDING_API_KEY missing")
    else:
        embedding_available = await _probe_http(embedding_base_url)
        if not embedding_available:
            reasons.append("embedding provider unreachable")
    return RealProviderStatus(
        llm_available=llm_available,
        embedding_available=embedding_available,
        llm_provider=("mock" if llm_mock_enabled else llm_base_url.split("//")[-1]),
        llm_model=llm_model_name,
        embedding_provider=(
            "mock" if embedding_mock_enabled else embedding_base_url.split("//")[-1]
        ),
        embedding_model=embedding_model_name,
        reason="; ".join(reasons) if reasons else "real providers available",
    )


__all__ = [
    "BenchmarkEnvironmentError",
    "RealProviderStatus",
    "assert_not_demo_database",
    "real_provider_status",
]
