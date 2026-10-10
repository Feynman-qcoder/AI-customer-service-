"""Contract tests for RESUME_AGENT_BENCHMARK_V1.

These tests never touch a database: they pin the frozen dataset contracts,
the de-duplication gate, the freeze manifest hashes, the statistics module
and the CLI validate entry point so the benchmark cannot silently drift.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.benchmark import generate_datasets as gen
from evals.benchmark.datasets import (
    LLM_REAL_CALL_COUNT,
    RAG_ANSWERED_COUNT,
    RAG_HOLDOUT_COUNT,
    RAG_NO_ANSWER_COUNT,
    RECOVERY_TRIAL_COUNT,
    SECURITY_HOLDOUT_COUNT,
    WORKFLOW_HOLDOUT_COUNT,
    BenchmarkDatasetError,
    load_llm_real,
    load_rag_holdout,
    load_recovery_matrix,
    load_security_holdout,
    load_workflow_holdout,
)
from evals.benchmark.evidence import sanitize_line, verify_evidence_dir
from evals.benchmark.stats import (
    bootstrap_mean_ci,
    percentile,
    ratio_metric,
    retrieval_metrics,
    wilson_interval,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATASET_ROOT = _REPO_ROOT / "server" / "evals" / "datasets" / "resume_benchmark_v1"


# ---------------------------------------------------------------------------
# frozen dataset contracts
# ---------------------------------------------------------------------------


def test_frozen_datasets_pass_validation() -> None:
    problems = gen.validate_frozen(_REPO_ROOT)
    assert problems == []


def test_workflow_holdout_shape() -> None:
    cases = load_workflow_holdout(_DATASET_ROOT)
    assert len(cases) == WORKFLOW_HOLDOUT_COUNT == 120
    categories: dict[str, int] = {}
    for case in cases:
        categories[case.category] = categories.get(case.category, 0) + 1
    assert set(categories.values()) == {10}
    assert len(categories) == 12
    multi_turn = [case for case in cases if case.conversation_history]
    assert len(multi_turn) == 20, "layered-memory ablation needs 20 multi-turn cases"
    action_cases = [
        case for case in cases if case.expected_requires_confirmation
    ]
    assert action_cases, "confirmation-gated cases must exist"
    for case in action_cases:
        assert case.expected_risk_level == "HIGH"
        assert case.expected_state_transition == "PROMPT_CONFIRMATION"
        assert case.expected_business_mutation_delta == "0"
        assert case.expected_tools == []


def test_rag_holdout_shape() -> None:
    cases = load_rag_holdout(_DATASET_ROOT)
    assert len(cases) == RAG_HOLDOUT_COUNT == 60
    answered = [case for case in cases if case.has_answer]
    no_answer = [case for case in cases if not case.has_answer]
    assert len(answered) == RAG_ANSWERED_COUNT == 48
    assert len(no_answer) == RAG_NO_ANSWER_COUNT == 12
    identities = [identity for case in answered for identity in case.gold_document_identities]
    assert all(identity.startswith("document:") and len(identity) == 73 for identity in identities)


def test_security_holdout_shape() -> None:
    cases = load_security_holdout(_DATASET_ROOT)
    assert len(cases) == SECURITY_HOLDOUT_COUNT == 60
    per_category: dict[str, int] = {}
    for case in cases:
        per_category[case.category] = per_category.get(case.category, 0) + 1
    assert set(per_category.values()) == {12}
    for case in cases:
        assert case.canary_tokens, "every security case carries canaries"
        assert all("CANARY" in token or token.isascii() for token in case.canary_tokens)


def test_recovery_matrix_shape() -> None:
    trials = load_recovery_matrix(_DATASET_ROOT)
    assert len(trials) == RECOVERY_TRIAL_COUNT == 40
    per_point: dict[str, int] = {}
    for trial in trials:
        per_point[trial.fault_point] = per_point.get(trial.fault_point, 0) + 1
    assert set(per_point.values()) == {5}
    assert len(per_point) == 8
    for trial in trials:
        assert trial.order_status == "WAITING_SHIPMENT"
        if trial.fault_point == "stop_before_user_confirmation":
            # recovery = the customer confirms, so one request/effect pair
            # exists after recovery and the terminal state is admin approval
            assert trial.expected_request_count_delta == 1
            assert trial.expected_terminal_state == "WAITING_ADMIN_APPROVAL"
            assert trial.expected_order_status_after == "WAITING_SHIPMENT"
        elif trial.fault_point == "conflicting_approval_or_invalid_order_state":
            assert trial.expected_terminal_state == "REJECTED_FAIL_CLOSED"
            assert trial.expected_order_status_after == "WAITING_SHIPMENT"
        else:
            assert trial.expected_terminal_state in {"WAITING_ADMIN_APPROVAL", "EXECUTED"}
            assert trial.expected_request_count_delta in {0, 1}


def test_llm_real_shape_and_call_budget() -> None:
    cases = load_llm_real(_DATASET_ROOT)
    assert len(cases) * 3 == LLM_REAL_CALL_COUNT + 30 == 90
    readonly = [case for case in cases if case.requires_real_llm]
    blocked = [case for case in cases if not case.requires_real_llm]
    assert len(readonly) == 20
    assert len(blocked) == 10
    for case in readonly:
        assert case.expected_provider_calls == 3
        assert case.allowed_evidence_ids, "readonly answers must be citable"
    for case in blocked:
        assert case.expected_provider_calls == 0
        assert case.forbidden_answer_facts


def test_llm_blocked_questions_stay_deterministic() -> None:
    """Blocked questions must route to fully deterministic answers (0 calls)."""

    from app.agent.routing import build_rule_based_plan

    for case in load_llm_real(_DATASET_ROOT):
        plan = build_rule_based_plan(case.question)
        deterministic = (
            plan.intent in {"CLARIFICATION", "CANCEL_ORDER", "REFUND_REQUEST"}
            or plan.risk_level != "LOW"
            or plan.requires_confirmation
            or plan.action_type is not None
        )
        if case.requires_real_llm:
            assert not deterministic, case.question
        else:
            assert deterministic, case.question


def test_freeze_manifest_hashes_match_files() -> None:
    manifest = json.loads(
        (_DATASET_ROOT / "datasets_frozen.json").read_text(encoding="utf-8")
    )
    for name, meta in manifest["datasets"].items():
        digest = gen._sha256_file(_DATASET_ROOT / name)  # noqa: SLF001
        assert digest == meta["sha256"], name


# ---------------------------------------------------------------------------
# de-duplication gate cannot be bypassed
# ---------------------------------------------------------------------------


def test_dedup_gate_rejects_near_duplicates() -> None:
    base = "订单ORD202609150001里的轻氧洗面巾C20还有库存吗"
    record = {"case_id": "wf_dup_01", "question": base, "conversation_history": []}
    violations = gen._dedup_check(  # noqa: SLF001
        [record], [base, "完全无关的另一个问题关于天气"]
    )
    assert any("duplicates after_sale_v2" in item for item in violations)


def test_generator_rejects_bad_template_counts() -> None:
    original = dict(gen._SECURITY_TEMPLATES)  # noqa: SLF001
    try:
        gen._SECURITY_TEMPLATES["user_input_injection"] = original[  # noqa: SLF001
            "user_input_injection"
        ][:5]
        with pytest.raises(BenchmarkDatasetError):
            gen.build_security_cases()
    finally:
        gen._SECURITY_TEMPLATES.clear()  # noqa: SLF001
        gen._SECURITY_TEMPLATES.update(original)


# ---------------------------------------------------------------------------
# statistics contracts
# ---------------------------------------------------------------------------


def test_wilson_interval_known_values() -> None:
    low, high = wilson_interval(0, 10)
    assert low == 0.0
    low, high = wilson_interval(10, 10)
    assert high > 0.99  # Wilson upper bound converges to, never equals, 1.0
    assert high <= 1.0
    assert 0.69 < low < 0.75
    low, high = wilson_interval(5, 10)
    assert 0.2 < low < 0.5 < high < 0.8


def test_ratio_metric_denominator_zero() -> None:
    metric = ratio_metric(0, 0)
    assert metric.value == 0.0


def test_bootstrap_mean_ci_reproducible_and_bounded() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    first = bootstrap_mean_ci(values, seed=20261008, iterations=1000)
    second = bootstrap_mean_ci(values, seed=20261008, iterations=1000)
    assert first == second
    mean, low, high = first
    assert abs(mean - 3.0) < 1e-9
    assert low <= mean <= high


def test_retrieval_metrics_hit_recall_mrr() -> None:
    # A: both gold items (|Gold|=2) retrieved at ranks 1 and 4
    # B: one of two gold items retrieved at rank 5
    # C: no gold in the top five
    flags = [
        [1, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
    ]
    metrics = retrieval_metrics(flags, gold_totals=[2, 2, 1])
    assert metrics.answered_query_count == 3
    assert metrics.hit_at_5.numerator == 2
    assert metrics.hit_at_5.denominator == 3
    assert abs(metrics.recall_at_5 - 0.5) < 1e-9  # (1.0 + 0.5 + 0.0) / 3
    assert abs(metrics.mrr_at_5 - 0.4) < 1e-9  # (1.0 + 0.2 + 0.0) / 3


def test_retrieval_metrics_denominator_falls_back_to_flags() -> None:
    metrics = retrieval_metrics([[1, 0, 0, 0, 0]])
    assert abs(metrics.recall_at_5 - 1.0) < 1e-9


def test_retrieval_metrics_rejects_mismatched_totals() -> None:
    import pytest as _pytest

    with _pytest.raises(ValueError):
        retrieval_metrics([[1, 0, 0, 0, 0]], gold_totals=[1, 1])


def test_percentile_edge_cases() -> None:
    # nearest-rank (lower) percentile method, as documented in stats.py
    assert percentile([], 0.5) == 0.0
    assert percentile([5.0], 0.95) == 5.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.95) == 3.0


# ---------------------------------------------------------------------------
# evidence sanitization
# ---------------------------------------------------------------------------


def test_sanitize_line_redacts_sensitive_content() -> None:
    # Synthetic fixtures only: the fake key is assembled at runtime so no
    # secret-shaped literal ever appears in the source tree.
    fake_key = "sk-" + "0" * 12
    line = f"user 13912345678 paid with {fake_key} at 示例市样例区1号"
    sanitized = sanitize_line(line)
    assert "13912345678" not in sanitized
    assert fake_key not in sanitized
    assert "<redacted-phone>" in sanitized
    assert "<redacted-secret>" in sanitized


def test_verify_evidence_dir_reports_missing_manifest(tmp_path: Path) -> None:
    problems = verify_evidence_dir(tmp_path)
    assert problems == ["evidence_manifest.json is missing"]


# ---------------------------------------------------------------------------
# CLI validate exit code
# ---------------------------------------------------------------------------


def test_cli_validate_returns_zero() -> None:
    from evals.benchmark import cli

    assert cli.main(["validate"]) == 0
