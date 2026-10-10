"""Three-seed determinism comparison for RESUME_AGENT_BENCHMARK_V1.

Statistical rule 3: the deterministic profiles (workflow, security, recovery)
run consecutively under the 20261008 / 20261009 / 20261010 seeds with no code,
data, configuration or snapshot changes in between. Every case-level SCORED
artifact must be identical across the three sessions; any difference marks
NON_DETERMINISTIC_FAILURE. Wall-clock values (latency, timestamps) are
excluded — they are not part of the deterministic contract.

Two comparison layers:

- ``judgments`` (scored, strict): the seven check flags, ``strict_pass`` and
  every observed behaviour field the scorers consume (intent, risk,
  confirmation status, executed tools, mutation deltas, terminal state).
- ``diagnostics`` (non-scored, informational): raw retrieval artefacts such
  as the citation candidate list. Under Mock embeddings the dense channel is
  noise and Qdrant's approximate search can swap adjacent candidates between
  sessions (session-random point ids), so these raw lists may differ at the
  margin. Any such difference is reported verbatim; it only flips the verdict
  when it changes a scored artifact (a check flag or strict_pass).

Usage:
    python -m evals.benchmark.determinism \
        --reference <evidence-dir-A> \
        --replica <evidence-dir-B> --replica <evidence-dir-C> \
        [--out <path>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCORED_FIELDS: dict[str, tuple[str, ...]] = {
    "workflow": (
        "checks",
        "strict_pass",
        "intent",
        "risk_level",
        "agent_status",
        "executed_tools",
        "deltas",
    ),
    "security": (
        "strict_pass",
        "unauthorized_request_delta",
        "unauthorized_prepare_delta",
        "order_status_changes",
        "stock_changes",
        "leaked_victim_terms",
        "leaked_canary_tokens",
        "injection_leaks",
        "error",
    ),
    "recovery": (
        "strict_pass",
        "terminal_state",
        "request_count_delta",
        "effect_count_delta",
        "order_status_after",
        "duplicate_business_effects",
        "stale_writer_accepted",
        "audit_missing",
        "prompt_ok",
        "notes",
        "error",
    ),
}

_DIAGNOSTIC_FIELDS: dict[str, tuple[str, ...]] = {
    "workflow": ("citations",),
    "security": (),
    "recovery": (),
}

_RESULT_FILES = {
    "workflow": "workflow_results.jsonl",
    "security": "security_results.jsonl",
    "recovery": "recovery_results.jsonl",
}


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing result file: {path}")
    records: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _normalize(value: object) -> object:
    """Stable, order-insensitive normalization for comparison."""

    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        normalized = [_normalize(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True),
        )
    if isinstance(value, float):
        return round(value, 9)
    return value


def _profile_view(profile: str, record: dict[str, object]) -> dict[str, object]:
    observed = record.get("observed", {})
    if not isinstance(observed, dict):
        observed = {}
    if profile == "workflow":
        source = {
            "checks": record.get("checks"),
            "strict_pass": record.get("strict_pass"),
            "intent": observed.get("intent"),
            "risk_level": observed.get("risk_level"),
            "agent_status": observed.get("agent_status"),
            "executed_tools": observed.get("executed_tools"),
            "deltas": observed.get("deltas"),
            "citations": observed.get("citations"),
        }
    elif profile == "security":
        source = {
            "strict_pass": record.get("strict_pass"),
            **{
                field: observed.get(field)
                for field in _SCORED_FIELDS["security"]
                if field != "strict_pass"
            },
        }
    else:  # recovery
        source = {
            field: observed.get(field) for field in _SCORED_FIELDS["recovery"]
        }
    return {
        field: _normalize(source.get(field))
        for field in (*_SCORED_FIELDS[profile], *_DIAGNOSTIC_FIELDS[profile])
    }


def _compare_profile(
    profile: str,
    reference_dir: Path,
    replica_dirs: list[Path],
) -> dict[str, object]:
    file_name = _RESULT_FILES[profile]
    reference_records = {
        str(record["case_id"]): _profile_view(profile, record)
        for record in _load_jsonl(reference_dir / file_name)
    }
    scored_fields = _SCORED_FIELDS[profile]
    diagnostic_fields = _DIAGNOSTIC_FIELDS[profile]
    judgment_mismatches: list[dict[str, object]] = []
    diagnostic_differences: list[dict[str, object]] = []
    for replica_dir in replica_dirs:
        replica_records = {
            str(record["case_id"]): _profile_view(profile, record)
            for record in _load_jsonl(replica_dir / file_name)
        }
        if set(replica_records) != set(reference_records):
            judgment_mismatches.append(
                {
                    "replica": str(replica_dir),
                    "case_id": "<case-set>",
                    "fields": ["case_id"],
                    "reference": sorted(reference_records)[:5],
                    "replica_value": sorted(replica_records)[:5],
                }
            )
            continue
        for case_id, reference_view in reference_records.items():
            replica_view = replica_records[case_id]
            differing_scored = [
                field
                for field in scored_fields
                if reference_view.get(field) != replica_view.get(field)
            ]
            if differing_scored:
                judgment_mismatches.append(
                    {
                        "replica": str(replica_dir),
                        "case_id": case_id,
                        "fields": differing_scored,
                        "reference": {
                            field: reference_view.get(field)
                            for field in differing_scored
                        },
                        "replica_value": {
                            field: replica_view.get(field)
                            for field in differing_scored
                        },
                    }
                )
            differing_diagnostic = [
                field
                for field in diagnostic_fields
                if reference_view.get(field) != replica_view.get(field)
            ]
            if differing_diagnostic:
                diagnostic_differences.append(
                    {
                        "replica": str(replica_dir),
                        "case_id": case_id,
                        "fields": differing_diagnostic,
                        "reference": {
                            field: reference_view.get(field)
                            for field in differing_diagnostic
                        },
                        "replica_value": {
                            field: replica_view.get(field)
                            for field in differing_diagnostic
                        },
                    }
                )
    return {
        "compared_cases": len(reference_records),
        "replica_count": len(replica_dirs),
        "judgment_mismatches": judgment_mismatches[:20],
        "judgment_mismatch_count": len(judgment_mismatches),
        "diagnostic_differences": diagnostic_differences[:20],
        "diagnostic_difference_count": len(diagnostic_differences),
        "deterministic": not judgment_mismatches,
    }


def _compare_not_run(profile: str, summary_file: str, dirs: list[Path]) -> dict[str, object]:
    statuses = []
    for directory in dirs:
        path = directory / summary_file
        status = "MISSING"
        reason = ""
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                status = str(payload.get("status", "RUN"))
                reason = str(payload.get("reason", ""))
        statuses.append({"dir": str(directory), "status": status, "reason": reason})
    deterministic = len({entry["status"] for entry in statuses}) == 1
    return {
        "profile": profile,
        "statuses": statuses,
        "deterministic": deterministic,
    }


def compare_sessions(
    reference: Path,
    replicas: list[Path],
) -> dict[str, object]:
    """Compare deterministic profiles across sessions and return the verdict.

    The verdict is driven by the SCORED artifacts only; raw retrieval
    diagnostics are reported separately with their difference counts.
    """

    all_dirs = [reference, *replicas]
    profiles: dict[str, object] = {}
    for profile in ("workflow", "security", "recovery"):
        profiles[profile] = _compare_profile(profile, reference, replicas)
    profiles["rag-real"] = _compare_not_run(
        "rag-real", "retrieval_ablation_summary.json", all_dirs
    )
    profiles["llm-real"] = _compare_not_run("llm-real", "llm_summary.json", all_dirs)
    judgment_mismatch_total = sum(
        int(entry.get("judgment_mismatch_count", 0))  # type: ignore[union-attr]
        for entry in profiles.values()
    )
    diagnostic_difference_total = sum(
        int(entry.get("diagnostic_difference_count", 0))  # type: ignore[union-attr]
        for entry in profiles.values()
    )
    deterministic = judgment_mismatch_total == 0 and all(
        bool(entry.get("deterministic")) for entry in profiles.values()  # type: ignore[union-attr]
    )
    note = (
        "mock-chain diagnostic: raw citation candidate lists vary at the "
        "margin under Mock embeddings (noise dense channel + Qdrant "
        "approximate search over session-random point ids). It changed no "
        "scored artifact in any seed."
        if diagnostic_difference_total and deterministic
        else ""
    )
    return {
        "reference": str(reference),
        "replicas": [str(replica) for replica in replicas],
        "profiles": profiles,
        "judgment_mismatch_count_total": judgment_mismatch_total,
        "diagnostic_difference_count_total": diagnostic_difference_total,
        "mock_chain_diagnostic_note": note,
        "deterministic": deterministic,
        "verdict": "DETERMINISTIC" if deterministic else "NON_DETERMINISTIC_FAILURE",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--replica", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, default=None)
    arguments = parser.parse_args(argv)
    result = compare_sessions(arguments.reference, list(arguments.replica))
    out_path = arguments.out or (arguments.reference / "determinism_check.json")
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"DETERMINISM_OUT={out_path}")
    print(f"VERDICT={result['verdict']}")
    for profile, entry in result["profiles"].items():  # type: ignore[union-attr]
        if "judgment_mismatch_count" in entry:  # type: ignore[operator]
            print(
                f"  {profile}: cases={entry['compared_cases']} "  # type: ignore[index]
                f"judgment_mismatches={entry['judgment_mismatch_count']} "  # type: ignore[index]
                f"diagnostic_differences={entry['diagnostic_difference_count']}"  # type: ignore[index]
            )
        else:
            print(
                f"  {profile}: deterministic={entry['deterministic']}"  # type: ignore[index]
            )
    if result["mock_chain_diagnostic_note"]:
        print(f"  NOTE: {result['mock_chain_diagnostic_note']}")
    return 0 if result["deterministic"] else 1


if __name__ == "__main__":
    sys.exit(main())