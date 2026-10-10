"""llm-real profile — 60 real provider calls (20 readonly x3 + 10 blocked x3).

Requires a REAL LLM provider (``LLM_MOCK_ENABLED=false`` + working key).
When unavailable the profile returns NOT_RUN — deterministic rule fallbacks
are never counted as real-LLM success. Thinking is disabled, temperature /
token limit / 12s timeout / single provider call per request stay fixed.

Scoring is structural: gold facts, forbidden facts and citation identities.
The model under test is never the judge.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

from evals.benchmark.datasets import LlmRealCase, load_llm_real
from evals.benchmark.harness import citation_identities, load_knowledge_identity_map
from evals.benchmark.stats import percentile, ratio_metric, ratio_metric_dict

_WARMUP_CALLS = 3


class _ProviderCallCounter:
    """Benchmark-only instrumentation over the real provider client."""

    def __init__(self) -> None:
        self.calls = 0
        self.failures: list[str] = []
        self.completion_tokens: list[int] = []
        self.latencies_ms: list[int] = []
        self.timeouts = 0
        self._patched: list[tuple[type, str, Any]] = []

    def install(self) -> None:
        from app.llm.openai_compatible_client import OpenAICompatibleLLMClient

        counter = self

        def make_wrapper(original: Any) -> Any:
            async def wrapper(self_client: Any, *args: Any, **kwargs: Any) -> Any:
                counter.calls += 1
                started = time.monotonic()
                try:
                    result = await original(self_client, *args, **kwargs)
                except TimeoutError:
                    counter.timeouts += 1
                    counter.failures.append("TIMEOUT")
                    raise
                except Exception as caught:  # noqa: BLE001
                    counter.failures.append(type(caught).__name__)
                    if "timeout" in str(caught).lower():
                        counter.timeouts += 1
                    raise
                counter.latencies_ms.append(int((time.monotonic() - started) * 1000))
                usage = getattr(result, "usage", None)
                if usage is not None:
                    counter.completion_tokens.append(int(usage.completion_tokens))
                return result

            return wrapper

        for method_name in ("answer_observed", "plan_observed"):
            original = getattr(OpenAICompatibleLLMClient, method_name)
            setattr(OpenAICompatibleLLMClient, method_name, make_wrapper(original))
            self._patched.append((OpenAICompatibleLLMClient, method_name, original))

    def reset(self) -> None:
        self.calls = 0
        self.failures.clear()
        self.completion_tokens.clear()
        self.latencies_ms.clear()
        self.timeouts = 0

    def restore(self) -> None:
        for owner, name, original in self._patched:
            setattr(owner, name, original)
        self._patched.clear()


async def _warm_up(runtime: Any, cases: list[LlmRealCase]) -> None:
    service = runtime.service
    maker = runtime.session_maker
    actor = runtime.actors.customer
    from evals.benchmark.harness import create_conversation

    warmup_done = 0
    index = 0
    while warmup_done < _WARMUP_CALLS and index < len(cases):
        case = cases[index]
        index += 1
        if not case.requires_real_llm:
            continue
        conversation_id = await create_conversation(
            maker, user_id=actor.user_id, marker=f"warmup-{warmup_done}"
        )
        try:
            await service.chat(actor, conversation_id, case.question)
        except Exception:  # noqa: BLE001 — warm-up errors are not measured
            pass
        warmup_done += 1


async def run_llm_real_profile(
    runtime: Any,
    dataset_root: Path,
    evidence: Any,
    *,
    provider_status: Any,
    seed: int,
) -> dict[str, Any]:
    if not provider_status.llm_available:
        summary = {
            "status": "NOT_RUN",
            "reason": provider_status.reason,
            "rule": "real LLM provider unavailable; deterministic fallbacks are "
            "never counted as real-LLM success",
        }
        evidence.write_json("llm_summary.json", summary)
        return summary

    cases = load_llm_real(dataset_root)
    identity_map = load_knowledge_identity_map(dataset_root)
    service = runtime.service
    maker = runtime.session_maker
    actor = runtime.actors.customer
    from evals.benchmark.harness import create_conversation

    counter = _ProviderCallCounter()
    counter.install()
    try:
        await _warm_up(runtime, cases)

        # randomize execution order deterministically (readonly only)
        readonly = [case for case in cases if case.requires_real_llm]
        blocked = [case for case in cases if not case.requires_real_llm]
        plan = [(case, repeat) for case in readonly for repeat in range(3)]
        rng = random.Random(seed)
        rng.shuffle(plan)

        records: list[dict[str, Any]] = []
        measured_calls = 0
        acceptance = 0
        fact_covered = 0
        fact_violations = 0
        citation_legal = 0
        citation_total = 0
        fallbacks = 0
        timeouts = 0
        e2e_latencies: list[float] = []
        completion_tokens: list[int] = []
        provider_calls_per_request: list[int] = []

        async def measure(case: LlmRealCase, tag: str) -> None:
            nonlocal measured_calls, acceptance, fact_covered
            nonlocal fact_violations, citation_legal, citation_total
            nonlocal fallbacks, timeouts
            conversation_id = await create_conversation(
                maker, user_id=actor.user_id, marker=f"{case.case_id}-{tag}"
            )
            counter.reset()
            started = time.monotonic()
            error: str | None = None
            response: Any = None
            try:
                response = await service.chat(actor, conversation_id, case.question)
            except Exception as caught:  # noqa: BLE001
                error = type(caught).__name__
            e2e = time.monotonic() - started
            calls = counter.calls
            e2e_latencies.append(e2e)
            provider_calls_per_request.append(calls)
            completion_tokens.extend(counter.completion_tokens)
            timeouts += counter.timeouts
            answer = str(getattr(response, "answer", "") or "")
            structured_accepted = (
                response is not None
                and error is None
                and (calls > 0 if case.requires_real_llm else calls == 0)
            )
            if structured_accepted:
                acceptance += 1
            if case.requires_real_llm and response is not None and calls == 0:
                fallbacks += 1
            if response is not None:
                for fact in case.required_answer_facts:
                    if fact in answer:
                        fact_covered += 1
                for fact in case.forbidden_answer_facts:
                    if fact in answer:
                        fact_violations += 1
            citations = (
                citation_identities(
                    [
                        str(getattr(source, "fileName", ""))
                        for source in (getattr(response, "sources", None) or [])
                    ],
                    identity_map,
                )
                if response is not None
                else []
            )
            doc_citations = [c for c in citations if c.startswith("document:")]
            allowed = set(case.allowed_evidence_ids)
            for identity in doc_citations:
                citation_total += 1
                if not allowed or identity in allowed:
                    citation_legal += 1
            measured_calls += 1
            records.append(
                {
                    "case_id": case.case_id,
                    "repeat_tag": tag,
                    "question": case.question,
                    "requires_real_llm": case.requires_real_llm,
                    "expected_provider_calls": case.expected_provider_calls,
                    "observed": {
                        "provider_calls": calls,
                        "e2e_latency_seconds": round(e2e, 4),
                        "answer_excerpt": answer[:160],
                        "citations": citations,
                        "error": error,
                        "structured_accepted": structured_accepted,
                    },
                }
            )

        for case, repeat in plan:
            await measure(case, f"r{repeat}")
        for case in blocked:
            for repeat in range(3):
                await measure(case, f"b{repeat}")

        required_fact_checks = sum(
            len(case.required_answer_facts) for case in readonly
        ) * 3
        summary = {
            "status": "RUN",
            "llm_provider": provider_status.llm_provider,
            "llm_model": provider_status.llm_model,
            "temperature": runtime.settings.llm_temperature,
            "max_completion_tokens": runtime.settings.llm_max_completion_tokens,
            "timeout_seconds": runtime.settings.llm_request_timeout_seconds,
            "seed": seed,
            "measured_calls": measured_calls,
            "structured_acceptance": ratio_metric_dict(ratio_metric(acceptance, measured_calls)),
            "required_fact_coverage": ratio_metric_dict(ratio_metric(fact_covered, required_fact_checks))
            if required_fact_checks
            else {"numerator": 0, "denominator": 0},
            "forbidden_fact_violation_rate": ratio_metric_dict(ratio_metric(
                fact_violations, measured_calls
            )),
            "citation_precision": (
                citation_legal / citation_total if citation_total else 0.0
            ),
            "deterministic_fallback": ratio_metric_dict(ratio_metric(fallbacks, measured_calls)),
            "timeout_rate": ratio_metric_dict(ratio_metric(timeouts, measured_calls)),
            "provider_calls_per_request": {
                "p50": percentile(
                    [float(value) for value in provider_calls_per_request], 0.5
                ),
                "max": max(provider_calls_per_request) if provider_calls_per_request else 0,
            },
            "completion_tokens": {
                "p50": percentile([float(v) for v in completion_tokens], 0.5),
                "p95": percentile([float(v) for v in completion_tokens], 0.95),
            },
            "e2e_latency_seconds": {
                "p50": round(percentile(e2e_latencies, 0.5), 4),
                "p95": round(percentile(e2e_latencies, 0.95), 4),
                "max": round(max(e2e_latencies), 4) if e2e_latencies else 0.0,
            },
        }
        evidence.write_jsonl("llm_results.jsonl", records)
        evidence.write_json("llm_summary.json", summary)
        return summary
    finally:
        counter.restore()


__all__ = ["run_llm_real_profile"]
