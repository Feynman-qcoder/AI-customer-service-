"""rag-real profile — 7-scheme retrieval ablation over 60 frozen queries.

Requires a REAL embedding provider: ``EMBEDDING_MOCK_ENABLED=false`` with a
working key. When the real provider is unavailable the profile returns
NOT_RUN — mock embedding results are never reported as real semantic
retrieval. The seven schemes are composed from the production retrieval
components (keyword / dense / structured channels, RRF, heuristic rerank,
score threshold) WITHOUT modifying production code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evals.benchmark.datasets import load_rag_holdout
from evals.benchmark.stats import ratio_metric_dict, retrieval_metrics

SCHEME_NAMES = (
    "keyword_only",
    "dense_only",
    "keyword_dense_merge",
    "keyword_dense_structured_merge",
    "three_channel_rrf",
    "three_channel_rrf_rerank",
    "three_channel_rrf_rerank_threshold",
)
_TOP_K = 5
_THRESHOLD = 0.35  # production rag_min_retrieval_score default


@dataclass(slots=True)
class RagRealStatus:
    status: str  # "RUN" | "NOT_RUN"
    reason: str


def _neutral_context() -> Any:
    from app.schemas.retrieval import RetrievalQueryContext

    return RetrievalQueryContext(
        product_category=None,
        order_status=None,
        payment_status=None,
        shipment_status=None,
        signed_days=None,
        after_sale_type=None,
        has_specific_order=False,
    )


def _document_identity_of(candidate: Any, identity_map: dict[str, str]) -> str | None:
    file_name = str(candidate.metadata.get("file_name", ""))
    return identity_map.get(file_name)


async def _scheme_results(
    knowledge: Any,
    session: Any,
    query: str,
    *,
    keyword: list[Any],
    dense: list[Any],
    structured: list[Any],
    scheme: str,
) -> list[Any]:
    from app.services.knowledge_service import heuristic_rerank, rrf_fuse

    if scheme == "keyword_only":
        return keyword[:_TOP_K]
    if scheme == "dense_only":
        return dense[:_TOP_K]
    if scheme == "keyword_dense_merge":
        merged = _merge_by_score([keyword, dense])
        return merged[:_TOP_K]
    if scheme == "keyword_dense_structured_merge":
        merged = _merge_by_score([keyword, dense, structured])
        return merged[:_TOP_K]
    if scheme == "three_channel_rrf":
        return rrf_fuse([keyword, dense, structured])[:_TOP_K]
    if scheme == "three_channel_rrf_rerank":
        fused = rrf_fuse([keyword, dense, structured])
        return heuristic_rerank(query, fused)[:_TOP_K]
    if scheme == "three_channel_rrf_rerank_threshold":
        fused = rrf_fuse([keyword, dense, structured])
        reranked = heuristic_rerank(query, fused)
        return [c for c in reranked if (c.rerank_score or 0) >= _THRESHOLD][:_TOP_K]
    raise ValueError(f"unknown scheme: {scheme}")


def _merge_by_score(result_sets: list[list[Any]]) -> list[Any]:
    """Naive score merge (no RRF): highest original score first, deduped."""

    seen: dict[str, Any] = {}
    for result_set in result_sets:
        for candidate in result_set:
            existing = seen.get(candidate.candidate_id)
            if existing is None or (candidate.original_score or 0) > (
                existing.original_score or 0
            ):
                seen[candidate.candidate_id] = candidate
    return sorted(
        seen.values(), key=lambda item: item.original_score or 0, reverse=True
    )


async def run_rag_real_profile(
    session_maker: Any,
    dataset_root: Path,
    evidence: Any,
    *,
    provider_status: Any,
) -> dict[str, Any]:
    if not provider_status.embedding_available:
        summary = {
            "status": "NOT_RUN",
            "reason": provider_status.reason,
            "rule": "real embedding provider unavailable; mock embeddings are "
            "never reported as real semantic retrieval",
        }
        evidence.write_json("retrieval_ablation_summary.json", summary)
        return summary

    from app.services.knowledge_service import knowledge_service
    from evals.benchmark.harness import load_knowledge_identity_map

    cases = load_rag_holdout(dataset_root)
    identity_map = load_knowledge_identity_map(dataset_root)
    context = _neutral_context()
    records: list[dict[str, Any]] = []
    per_scheme_gold_flags: dict[str, list[list[int]]] = {
        scheme: [] for scheme in SCHEME_NAMES
    }
    answered_gold_totals: list[int] = []
    no_answer_refusals: dict[str, int] = {scheme: 0 for scheme in SCHEME_NAMES}

    async with session_maker() as session:
        for case in cases:
            keyword = await knowledge_service.keyword_recall(session, case.question, 10)
            dense = await knowledge_service.dense_vector_recall(case.question, 10)
            structured = await knowledge_service.structured_rule_recall(
                session, case.question, 10, context
            )
            case_record: dict[str, Any] = {
                "case_id": case.case_id,
                "question": case.question,
                "has_answer": case.has_answer,
                "gold_document_identities": case.gold_document_identities,
                "schemes": {},
            }
            for scheme in SCHEME_NAMES:
                results = await _scheme_results(
                    knowledge_service,
                    session,
                    case.question,
                    keyword=keyword,
                    dense=dense,
                    structured=structured,
                    scheme=scheme,
                )
                identities = [
                    _document_identity_of(candidate, identity_map) for candidate in results
                ]
                if case.has_answer:
                    flags = [
                        1 if identity in case.gold_document_identities else 0
                        for identity in identities
                    ]
                    per_scheme_gold_flags[scheme].append(flags)
                    if scheme == SCHEME_NAMES[0]:
                        answered_gold_totals.append(len(case.gold_document_identities))
                else:
                    if not results:
                        no_answer_refusals[scheme] += 1
                case_record["schemes"][scheme] = {
                    "identities": identities,
                    "scores": [
                        float(
                            candidate.rerank_score
                            or candidate.fused_score
                            or candidate.original_score
                            or 0
                        )
                        for candidate in results
                    ],
                }
            records.append(case_record)
            evidence.log(f"rag {case.case_id} processed")

    answered_count = sum(1 for case in cases if case.has_answer)
    no_answer_count = sum(1 for case in cases if not case.has_answer)
    per_scheme_summary: dict[str, Any] = {}
    for scheme in SCHEME_NAMES:
        metrics = retrieval_metrics(
            per_scheme_gold_flags[scheme], answered_gold_totals
        )
        citation_total = sum(
            len(record["schemes"][scheme]["identities"]) for record in records
        )
        citation_legal = sum(
            1
            for record in records
            for identity in record["schemes"][scheme]["identities"]
            if not record["has_answer"]
            or identity in record["gold_document_identities"]
        )
        per_scheme_summary[scheme] = {
            "hit_at_5": ratio_metric_dict(metrics.hit_at_5),
            "recall_at_5": round(metrics.recall_at_5, 4),
            "recall_at_5_ci": list(metrics.recall_at_5_ci),
            "mrr_at_5": round(metrics.mrr_at_5, 4),
            "mrr_at_5_ci": list(metrics.mrr_at_5_ci),
            "no_answer_refusals": no_answer_refusals[scheme],
            "no_answer_denominator": no_answer_count,
            "citation_precision": (
                citation_legal / citation_total if citation_total else 0.0
            ),
        }
    keyword_hit = per_scheme_summary["keyword_only"]["hit_at_5"]
    full_hit = per_scheme_summary["three_channel_rrf_rerank_threshold"]["hit_at_5"]
    keyword_recall = per_scheme_summary["keyword_only"]["recall_at_5"]
    full_recall = per_scheme_summary["three_channel_rrf_rerank_threshold"]["recall_at_5"]
    keyword_mrr = per_scheme_summary["keyword_only"]["mrr_at_5"]
    full_mrr = per_scheme_summary["three_channel_rrf_rerank_threshold"]["mrr_at_5"]
    summary = {
        "status": "RUN",
        "embedding_provider": provider_status.embedding_provider,
        "embedding_model": provider_status.embedding_model,
        "answered_query_count": answered_count,
        "no_answer_query_count": no_answer_count,
        "schemes": per_scheme_summary,
        "full_scheme_vs_keyword_only": {
            "hit_at_5_absolute": round(full_hit["value"] - keyword_hit["value"], 4),
            "hit_at_5_percentage_points": round(
                100 * (full_hit["value"] - keyword_hit["value"]), 2
            ),
            "recall_at_5_absolute": round(full_recall - keyword_recall, 4),
            "mrr_at_5_absolute": round(full_mrr - keyword_mrr, 4),
        },
    }
    evidence.write_jsonl("retrieval_results.jsonl", records)
    evidence.write_json("retrieval_ablation_summary.json", summary)
    return summary


__all__ = ["SCHEME_NAMES", "run_rag_real_profile"]
