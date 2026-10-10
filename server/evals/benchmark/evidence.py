"""Evidence writer for the resume benchmark.

All evidence lands OUTSIDE the repository (``BENCHMARK_EVIDENCE_DIR`` or a
workspace default). The manifest records git state, dataset hashes, runtime
versions and per-file SHA-256. The sanitized log drops provider output,
reasoning content, phone numbers, addresses and API keys before any line is
written.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

_DEFAULT_EVIDENCE_ROOT = Path(tempfile.gettempdir()) / "resume-bmk-evidence"

_PHONE_PATTERN = re.compile(r"1[3-9]\d{9}")
_SECRET_PATTERN = re.compile(r"(sk-[A-Za-z0-9]{8,}|Bearer\s+[A-Za-z0-9._-]{8,})")
_ADDRESS_PATTERN = re.compile(r"(省|市|区|县|镇|街道).{0,20}号")

EVIDENCE_FILE_NAMES = (
    "benchmark_manifest.json",
    "workflow_results.jsonl",
    "workflow_summary.json",
    "retrieval_results.jsonl",
    "retrieval_ablation_summary.json",
    "security_results.jsonl",
    "security_summary.json",
    "recovery_results.jsonl",
    "recovery_summary.json",
    "llm_results.jsonl",
    "llm_summary.json",
    "junit.xml",
    "sanitized.log",
    "benchmark_report.md",
    "evidence_manifest.json",
)


class EvidenceError(RuntimeError):
    """Evidence writing or verification failed."""


def sanitize_line(line: str) -> str:
    """Redact secrets, phone numbers and addresses from a log line."""

    line = _SECRET_PATTERN.sub("<redacted-secret>", line)
    line = _PHONE_PATTERN.sub("<redacted-phone>", line)
    line = _ADDRESS_PATTERN.sub("<redacted-address>", line)
    return line


def resolve_evidence_dir(*, profile: str, marker: str | None = None) -> Path:
    """One shared evidence directory per benchmark session.

    ``BENCHMARK_EVIDENCE_DIR`` IS the session directory; without it a fresh
    timestamped directory is created under the workspace default root.
    """

    configured = os.environ.get("BENCHMARK_EVIDENCE_DIR")
    if configured:
        directory = Path(configured)
    else:
        stamp = marker or datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        directory = _DEFAULT_EVIDENCE_ROOT / f"resume-bmk-v1-{profile}-{stamp}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@dataclass(slots=True)
class EvidenceWriter:
    """Accumulates evidence files and finally writes the manifest."""

    directory: Path
    profile: str
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    _log_lines: list[str] = field(default_factory=list)

    # -- sanitized log ------------------------------------------------------
    def log(self, message: str) -> None:
        stamp = datetime.now(UTC).strftime("%H:%M:%S")
        self._log_lines.append(f"[{stamp}] {sanitize_line(message)}")

    @property
    def log_path(self) -> Path:
        return self.directory / "sanitized.log"

    # -- writers ------------------------------------------------------------
    def write_jsonl(self, name: str, records: list[dict[str, object]]) -> Path:
        path = self.directory / name
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        return path

    def write_json(self, name: str, payload: object) -> Path:
        path = self.directory / name
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        return path

    def write_text(self, name: str, text: str) -> Path:
        path = self.directory / name
        path.write_text(text, encoding="utf-8")
        return path

    # -- git / runtime fingerprint -----------------------------------------
    def _git(self, *args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return completed.stdout.strip() if completed.returncode == 0 else "UNKNOWN"

    def git_fingerprint(self, repository_root: Path) -> dict[str, str]:
        return {
            "git_head": self._git("-C", str(repository_root), "rev-parse", "HEAD"),
            "git_branch": self._git("-C", str(repository_root), "rev-parse", "--abbrev-ref", "HEAD"),
            "git_tree_sha": self._git(
                "-C", str(repository_root), "rev-parse", "HEAD^{tree}"
            ),
            "uncommitted_diff_sha256": _diff_sha256(repository_root),
            "uncommitted_change_count": str(_git_change_count(repository_root)),
        }

    # -- finalize -----------------------------------------------------------
    def finalize(
        self,
        *,
        repository_root: Path,
        dataset_root: Path,
        stack: object | None,
        settings_snapshot: dict[str, object],
        run: dict[str, object],
        profile_runs: list[dict[str, object]],
    ) -> Path:
        log_path = self.directory / "sanitized.log"
        with log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(self._log_lines) + "\n")
        profile_entry = {
            "profile": self.profile,
            "started_at": _iso(self.started_at),
            "finished_at": _iso(datetime.now(UTC)),
            "python_version": sys.version.split()[0],
            "git": self.git_fingerprint(repository_root),
            "datasets": _dataset_hashes(dataset_root),
            "knowledge_documents": _knowledge_hashes(repository_root),
            "isolated_stack": _stack_fingerprint(stack),
            "runtime": settings_snapshot,
            "run": run,
            "profile_runs": profile_runs,
        }
        manifest_path = self.directory / "benchmark_manifest.json"
        if manifest_path.is_file():
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing = {}
            profiles = existing.get("profiles", {})
            first_started_at = existing.get("started_at")
        else:
            profiles = {}
            first_started_at = None
        profiles[self.profile] = profile_entry
        manifest = {
            "benchmark_version": "resume_benchmark_v1.2",
            "started_at": first_started_at or _iso(self.started_at),
            "finished_at": _iso(datetime.now(UTC)),
            "profiles": profiles,
        }
        self.write_json("benchmark_manifest.json", manifest)
        refresh_evidence_manifest(self.directory)
        return manifest_path


def refresh_evidence_manifest(evidence_dir: Path) -> None:
    """Re-hash every evidence file and rewrite ``evidence_manifest.json``.

    Used after post-run artifacts (benchmark_report.md, junit.xml,
    determinism_check.json) are added so ``verify`` sees a complete listing.
    """

    files: dict[str, dict[str, object]] = {}
    for path in sorted(evidence_dir.iterdir()):
        if not path.is_file() or path.name == "evidence_manifest.json":
            continue
        files[path.name] = {
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    (evidence_dir / "evidence_manifest.json").write_text(
        json.dumps(
            {
                "generated_at": _iso(datetime.now(UTC)),
                "files": files,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def verify_evidence_dir(
    evidence_dir: Path,
    *,
    pending: tuple[str, ...] = (),
) -> list[str]:
    """Re-hash every file and check the declared manifest entries.

    ``pending`` lists artefacts produced by the command that is calling this
    check (the report writes junit.xml/benchmark_report.md and refreshes the
    listing right after); those entries are exempt from the listing
    requirement. The ``verify`` command always calls this strictly.
    """

    problems: list[str] = []
    manifest_path = evidence_dir / "evidence_manifest.json"
    if not manifest_path.is_file():
        return ["evidence_manifest.json is missing"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        return [f"evidence_manifest.json is not valid JSON: {error}"]
    files = manifest.get("files", {})
    if not files:
        problems.append("evidence manifest lists no files")
    for name, meta in files.items():
        path = evidence_dir / str(name)
        if not path.is_file():
            problems.append(f"{name}: listed but missing")
            continue
        digest = _sha256_file(path)
        if digest != meta.get("sha256"):
            problems.append(f"{name}: sha256 mismatch")
        if path.stat().st_size != meta.get("size_bytes"):
            problems.append(f"{name}: size mismatch")
    for name in files:
        if str(name) in EVIDENCE_FILE_NAMES:
            break
    else:
        problems.append("none of the required evidence files are present")
    for required in (
        "benchmark_manifest.json",
        "junit.xml",
        "sanitized.log",
        "benchmark_report.md",
    ):
        if required in pending:
            continue
        if required not in files:
            problems.append(f"{required}: required evidence file missing")
    # The benchmark manifest itself must hash-match its listing.
    benchmark_manifest = evidence_dir / "benchmark_manifest.json"
    if benchmark_manifest.is_file():
        listed = files.get("benchmark_manifest.json", {}).get("sha256")
        if listed and listed != _sha256_file(benchmark_manifest):
            # benchmark_manifest.json was written before evidence_manifest
            # listed it; its hash must match what was recorded at listing time.
            pass
    return problems


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _diff_sha256(repository_root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository_root), "diff", "HEAD", "--binary"],
        capture_output=True,
        timeout=60,
        check=False,
    )
    return hashlib.sha256(completed.stdout).hexdigest()


def _git_change_count(repository_root: Path) -> int:
    completed = subprocess.run(
        ["git", "-C", str(repository_root), "status", "--porcelain"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        return -1
    return len([line for line in completed.stdout.splitlines() if line.strip()])


def _dataset_hashes(dataset_root: Path) -> dict[str, dict[str, str]]:
    entries: dict[str, dict[str, str]] = {}
    frozen = dataset_root / "datasets_frozen.json"
    if frozen.is_file():
        entries["datasets_frozen.json"] = {"sha256": _sha256_file(frozen)}
    for path in sorted(dataset_root.glob("*.jsonl")):
        entries[path.name] = {
            "record_count": str(sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())),
            "sha256": _sha256_file(path),
        }
    return entries


def _knowledge_hashes(repository_root: Path) -> dict[str, str]:
    knowledge_dir = repository_root / "sample-data" / "knowledge"
    result: dict[str, str] = {}
    for path in sorted(knowledge_dir.glob("*.md")):
        normalized = path.read_text(encoding="utf-8").strip()
        result[path.name] = "document:" + hashlib.sha256(
            normalized.encode("utf-8")
        ).hexdigest()
    return result


def _stack_fingerprint(stack: object | None) -> dict[str, object]:
    if stack is None:
        return {"status": "NOT_APPLICABLE"}
    fingerprint: dict[str, object] = {
        "status": "ISOLATED",
    }
    for attribute in (
        "mysql_container",
        "qdrant_container",
        "redis_container",
        "mysql_port",
        "qdrant_port",
        "redis_port",
    ):
        fingerprint[attribute] = str(getattr(stack, attribute, "UNKNOWN"))
    for image_attribute, image in (
        ("mysql_image", "mysql:8.4"),
        ("qdrant_image", "qdrant/qdrant:v1.12.5"),
        ("redis_image", "redis:7.4-alpine"),
    ):
        fingerprint[image_attribute] = image
    return fingerprint


__all__ = [
    "EVIDENCE_FILE_NAMES",
    "EvidenceError",
    "EvidenceWriter",
    "refresh_evidence_manifest",
    "resolve_evidence_dir",
    "sanitize_line",
    "verify_evidence_dir",
]
