"""RESUME_AGENT_BENCHMARK_V1 — statistics.

All proportional metrics report numerator/denominator plus a Wilson 95%
interval; deltas, MRR and latency use a fixed-seed bootstrap 95% interval.
Nothing in here ever drops failures or retries samples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_BOOTSTRAP_DEFAULT_ITERATIONS = 10_000
_BOOTSTRAP_DEFAULT_SEED = 20261008
WILSON_Z = 1.959963984540054  # 95%


def wilson_interval(numerator: int, denominator: int) -> tuple[float, float]:
    if denominator <= 0:
        return (0.0, 0.0)
    p = numerator / denominator
    z2 = WILSON_Z * WILSON_Z
    denom = 1 + z2 / denominator
    center = (p + z2 / (2 * denominator)) / denom
    margin = (WILSON_Z / denom) * math.sqrt(
        (p * (1 - p) + z2 / (4 * denominator)) / denominator
    )
    return (max(0.0, center - margin), min(1.0, center + margin))


@dataclass(frozen=True, slots=True)
class RatioMetric:
    numerator: int
    denominator: int
    value: float
    wilson_low: float
    wilson_high: float


def ratio_metric(numerator: int, denominator: int) -> RatioMetric:
    low, high = wilson_interval(numerator, denominator)
    return RatioMetric(
        numerator=numerator,
        denominator=denominator,
        value=(numerator / denominator) if denominator else 0.0,
        wilson_low=low,
        wilson_high=high,
    )


def ratio_metric_dict(metric: RatioMetric) -> dict[str, float | int]:
    """JSON-friendly view of a RatioMetric (slots dataclass has no __dict__)."""

    return {
        "numerator": metric.numerator,
        "denominator": metric.denominator,
        "value": metric.value,
        "wilson_low": metric.wilson_low,
        "wilson_high": metric.wilson_high,
    }


def bootstrap_mean_ci(
    values: list[float],
    *,
    seed: int = _BOOTSTRAP_DEFAULT_SEED,
    iterations: int = _BOOTSTRAP_DEFAULT_ITERATIONS,
) -> tuple[float, float, float]:
    """Fixed-seed bootstrap mean with 95% interval (percentile method)."""
    if not values:
        return (0.0, 0.0, 0.0)
    state = seed & 0xFFFFFFFF
    samples: list[float] = []
    count = len(values)
    for _ in range(iterations):
        total = 0.0
        for _index in range(count):
            # xorshift32 keeps this reproducible across platforms.
            state ^= (state << 13) & 0xFFFFFFFF
            state ^= state >> 17
            state ^= (state << 5) & 0xFFFFFFFF
            total += values[state % count]
        samples.append(total / count)
    samples.sort()
    mean = sum(values) / count
    low = samples[int(0.025 * (iterations - 1))]
    high = samples[int(0.975 * (iterations - 1))]
    return (mean, low, high)


def bootstrap_delta_ci(
    baseline: list[float],
    treatment: list[float],
    *,
    seed: int = _BOOTSTRAP_DEFAULT_SEED,
    iterations: int = _BOOTSTRAP_DEFAULT_ITERATIONS,
) -> tuple[float, float, float]:
    """Fixed-seed bootstrap for mean(treatment) - mean(baseline)."""
    if not baseline or not treatment:
        return (0.0, 0.0, 0.0)
    state = seed & 0xFFFFFFFF
    count_base = len(baseline)
    count_treat = len(treatment)
    deltas: list[float] = []
    for _ in range(iterations):
        total_base = 0.0
        for _index in range(count_base):
            state ^= (state << 13) & 0xFFFFFFFF
            state ^= state >> 17
            state ^= (state << 5) & 0xFFFFFFFF
            total_base += baseline[state % count_base]
        total_treat = 0.0
        for _index in range(count_treat):
            state ^= (state << 13) & 0xFFFFFFFF
            state ^= state >> 17
            state ^= (state << 5) & 0xFFFFFFFF
            total_treat += treatment[state % count_treat]
        deltas.append(total_treat / count_treat - total_base / count_base)
    deltas.sort()
    observed = sum(treatment) / count_treat - sum(baseline) / count_base
    low = deltas[int(0.025 * (iterations - 1))]
    high = deltas[int(0.975 * (iterations - 1))]
    return (observed, low, high)


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    hit_at_5: RatioMetric
    recall_at_5: float
    recall_at_5_ci: tuple[float, float]
    mrr_at_5: float
    mrr_at_5_ci: tuple[float, float]
    answered_query_count: int


def retrieval_metrics(
    gold_hits_per_query: list[list[int]],
    gold_totals: list[int] | None = None,
) -> RetrievalMetrics:
    """Hit@5, Recall@5 and MRR@5 from per-query ranked gold flags.

    ``gold_hits_per_query[i][rank]`` is 1 when the result at ``rank`` (0-based)
    is a gold item for query ``i``. Hit@5 needs at least one gold in the top
    five; Recall@5 averages ``|Top5 ∩ Gold| / |Gold|`` where ``|Gold|`` comes
    from ``gold_totals[i]`` (falling back to the gold count inside the passed
    ranking); MRR@5 uses the reciprocal rank of the first gold result within
    the top five (0 when nothing hits).
    """
    answered = [flags for flags in gold_hits_per_query]
    hit_flags = [1 if any(flags[:5]) else 0 for flags in answered]
    totals = (
        list(gold_totals)
        if gold_totals is not None
        else [max(1, sum(flags)) for flags in answered]
    )
    if len(totals) != len(answered):
        raise ValueError("gold_totals length must match the query count")
    recalls = [
        (sum(flags[:5]) / max(1, total)) for flags, total in zip(answered, totals, strict=False)
    ]
    reciprocal_ranks = []
    for flags in answered:
        rank = next(
            (index + 1 for index, flag in enumerate(flags[:5]) if flag),
            0,
        )
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
    recall_mean, recall_low, recall_high = bootstrap_mean_ci(recalls)
    mrr_mean, mrr_low, mrr_high = bootstrap_mean_ci(reciprocal_ranks)
    return RetrievalMetrics(
        hit_at_5=ratio_metric(sum(hit_flags), len(hit_flags)),
        recall_at_5=recall_mean,
        recall_at_5_ci=(recall_low, recall_high),
        mrr_at_5=mrr_mean,
        mrr_at_5_ci=(mrr_low, mrr_high),
        answered_query_count=len(answered),
    )


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(fraction * (len(ordered) - 1))
    return ordered[index]
