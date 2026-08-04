"""Metrics for qrel-free multi-vector compression risk.

The functions in this module separate relevance-based effectiveness from
qrel-free ranking fidelity.  Positive regret always means that the candidate
is worse than the full-reference representation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from reprforge.heterogeneity_atlas import query_metrics, stable_ranks


def _aligned_vectors(
    reference: np.ndarray, candidate: np.ndarray, *, name: str
) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(reference, dtype=np.float64)
    right = np.asarray(candidate, dtype=np.float64)
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape or not len(left):
        raise ValueError(f"{name} must be aligned non-empty vectors")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError(f"{name} contains non-finite values")
    return left, right


def _aligned_score_matrices(
    reference: np.ndarray, candidate: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(reference, dtype=np.float64)
    right = np.asarray(candidate, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape or not left.size:
        raise ValueError("score surfaces must be aligned non-empty matrices")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("score surfaces contain non-finite values")
    return left, right


def bootstrap_mean_interval(
    values: np.ndarray,
    *,
    seed: int = 0,
    resamples: int = 4000,
) -> dict[str, float | int]:
    """Return two-sided 95% and one-sided 95% intervals for a mean."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("bootstrap values must be a finite non-empty vector")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(resamples, len(array)))
    means = array[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return {
        "mean": float(array.mean()),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "one_sided_95_lower": float(np.quantile(means, 0.05)),
        "one_sided_95_upper": float(np.quantile(means, 0.95)),
        "queries": int(len(array)),
        "resamples": int(resamples),
        "seed": int(seed),
    }


def regret_summary(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    catastrophic_threshold: float,
    seed: int = 0,
    resamples: int = 4000,
) -> dict[str, float | int | dict[str, float | int]]:
    """Summarize reference-minus-candidate per-query regret."""

    if catastrophic_threshold < 0:
        raise ValueError("catastrophic threshold must be non-negative")
    left, right = _aligned_vectors(reference, candidate, name="metric values")
    regret = left - right
    tail_count = max(1, math.ceil(0.05 * len(regret)))
    worst = np.sort(regret)[-tail_count:]
    return {
        "reference_mean": float(left.mean()),
        "candidate_mean": float(right.mean()),
        "mean_regret": float(regret.mean()),
        "median_regret": float(np.median(regret)),
        "p90_regret": float(np.quantile(regret, 0.90)),
        "p95_regret": float(np.quantile(regret, 0.95)),
        "worst_5pct_cvar": float(worst.mean()),
        "harmed_fraction": float(np.mean(regret > 0.0)),
        "catastrophic_threshold": float(catastrophic_threshold),
        "catastrophic_harm_fraction": float(
            np.mean(regret > catastrophic_threshold)
        ),
        "bootstrap": bootstrap_mean_interval(
            regret, seed=seed, resamples=resamples
        ),
    }


def rbo_ext(
    reference_order: Sequence[int],
    candidate_order: Sequence[int],
    *,
    p: float = 0.95,
) -> float:
    """Finite extrapolated Rank-Biased Overlap for two equal-depth rankings."""

    if not 0.0 < p < 1.0:
        raise ValueError("RBO persistence p must lie in (0, 1)")
    if len(reference_order) != len(candidate_order) or len(reference_order) == 0:
        raise ValueError("RBO rankings must have the same positive depth")
    reference_seen: set[int] = set()
    candidate_seen: set[int] = set()
    weighted = 0.0
    agreement = 0.0
    for depth, (left, right) in enumerate(
        zip(reference_order, candidate_order, strict=True), start=1
    ):
        reference_seen.add(int(left))
        candidate_seen.add(int(right))
        agreement = len(reference_seen & candidate_seen) / depth
        weighted += (1.0 - p) * agreement * p ** (depth - 1)
    return float(weighted + agreement * p ** len(reference_order))


def ranking_fidelity(
    reference_scores: np.ndarray,
    candidate_scores: np.ndarray,
    *,
    top_ks: Sequence[int] = (10, 100),
    rbo_depth: int = 100,
    rbo_p: float = 0.95,
) -> dict[str, np.ndarray]:
    """Compute qrel-free, per-query ranking-fidelity observations."""

    reference, candidate = _aligned_score_matrices(
        reference_scores, candidate_scores
    )
    corpus_size = reference.shape[1]
    ks = tuple(int(value) for value in top_ks)
    if not ks or any(value <= 0 or value > corpus_size for value in ks):
        raise ValueError("top-k depths must lie inside the corpus")
    if rbo_depth <= 0 or rbo_depth > corpus_size:
        raise ValueError("RBO depth must lie inside the corpus")

    reference_ranks = stable_ranks(reference)
    candidate_ranks = stable_ranks(candidate)
    output = {
        f"top_{k}_overlap": np.zeros(reference.shape[0], dtype=np.float64)
        for k in ks
    }
    output["full_top10_retained_at_candidate_top100"] = np.zeros(
        reference.shape[0], dtype=np.float64
    )
    output["full_top10_escape_candidate_top100"] = np.zeros(
        reference.shape[0], dtype=np.float64
    )
    output[f"rbo_ext_at_{rbo_depth}_p{rbo_p:g}"] = np.zeros(
        reference.shape[0], dtype=np.float64
    )
    positions = np.arange(corpus_size)
    full_depth = min(10, corpus_size)
    candidate_depth = min(100, corpus_size)
    for query_index in range(reference.shape[0]):
        reference_order = np.lexsort((positions, -reference[query_index]))
        candidate_order = np.lexsort((positions, -candidate[query_index]))
        for k in ks:
            output[f"top_{k}_overlap"][query_index] = len(
                set(reference_order[:k]) & set(candidate_order[:k])
            ) / k
        retained = np.mean(
            candidate_ranks[query_index, reference_order[:full_depth]]
            <= candidate_depth
        )
        output["full_top10_retained_at_candidate_top100"][query_index] = retained
        output["full_top10_escape_candidate_top100"][query_index] = 1.0 - retained
        output[f"rbo_ext_at_{rbo_depth}_p{rbo_p:g}"][query_index] = rbo_ext(
            reference_order[:rbo_depth],
            candidate_order[:rbo_depth],
            p=rbo_p,
        )
    return output


def ranking_safety_certificate(
    fidelity: Mapping[str, np.ndarray],
    *,
    minimum_top10_mean: float = 0.90,
    minimum_top10_lower: float = 0.88,
    minimum_top10_at_100_lower: float = 0.995,
    seed: int = 0,
    resamples: int = 4000,
) -> dict:
    """Apply a qrel-free configuration-level ranking assurance gate.

    The thresholds are frozen development criteria, not a distribution-free
    guarantee of relevance quality.  Transfer validity must be established on
    unopened collections.
    """

    required = (
        "top_10_overlap",
        "full_top10_retained_at_candidate_top100",
    )
    if any(name not in fidelity for name in required):
        raise ValueError("fidelity is missing a required ranking observation")
    if any(
        not 0.0 <= value <= 1.0
        for value in (
            minimum_top10_mean,
            minimum_top10_lower,
            minimum_top10_at_100_lower,
        )
    ):
        raise ValueError("ranking certificate thresholds must lie in [0, 1]")
    top10 = bootstrap_mean_interval(
        fidelity["top_10_overlap"], seed=seed, resamples=resamples
    )
    retained = bootstrap_mean_interval(
        fidelity["full_top10_retained_at_candidate_top100"],
        seed=seed,
        resamples=resamples,
    )
    result = {
        "status": "development_candidate_not_distribution_free_guarantee",
        "uses_qrels": False,
        "thresholds": {
            "minimum_top10_mean": float(minimum_top10_mean),
            "minimum_top10_one_sided_95_lower": float(minimum_top10_lower),
            "minimum_full_top10_at_candidate100_one_sided_95_lower": float(
                minimum_top10_at_100_lower
            ),
        },
        "observed": {
            "top10_mean": top10["mean"],
            "top10_one_sided_95_lower": top10["one_sided_95_lower"],
            "full_top10_at_candidate100_one_sided_95_lower": retained[
                "one_sided_95_lower"
            ],
        },
    }
    result["passes"] = bool(
        top10["mean"] >= minimum_top10_mean
        and top10["one_sided_95_lower"] >= minimum_top10_lower
        and retained["one_sided_95_lower"] >= minimum_top10_at_100_lower
    )
    return result


def evaluate_compression_pair(
    reference_scores: np.ndarray,
    candidate_scores: np.ndarray,
    relevance: Sequence[Mapping[int, float]],
    *,
    quality_ks: Sequence[int] = (5, 10, 100),
    safety_tolerance: float = 0.01,
    bootstrap_seed: int = 0,
    bootstrap_resamples: int = 4000,
) -> dict:
    """Evaluate one candidate representation against a full reference."""

    reference, candidate = _aligned_score_matrices(
        reference_scores, candidate_scores
    )
    if len(relevance) != reference.shape[0]:
        raise ValueError("relevance must be query-aligned")
    if safety_tolerance < 0:
        raise ValueError("safety tolerance must be non-negative")
    quality_reference = query_metrics(reference, relevance, ks=quality_ks)
    quality_candidate = query_metrics(candidate, relevance, ks=quality_ks)
    catastrophic = {
        "ndcg_at_10": 0.10,
        "recall_at_100": 0.05,
    }
    quality = {}
    for name in ("ndcg_at_5", "ndcg_at_10", "recall_at_100"):
        if name not in quality_reference:
            continue
        quality[name] = regret_summary(
            quality_reference[name],
            quality_candidate[name],
            catastrophic_threshold=catastrophic.get(name, 0.10),
            seed=bootstrap_seed,
            resamples=bootstrap_resamples,
        )
    required = ("ndcg_at_10", "recall_at_100")
    if not all(name in quality for name in required):
        raise ValueError("quality_ks must include 10 and 100 for the safety gate")
    safety = {
        "tolerance": float(safety_tolerance),
        "ndcg_at_10_upper_regret": quality["ndcg_at_10"]["bootstrap"][
            "one_sided_95_upper"
        ],
        "recall_at_100_upper_regret": quality["recall_at_100"]["bootstrap"][
            "one_sided_95_upper"
        ],
    }
    safety["passes"] = bool(
        safety["ndcg_at_10_upper_regret"] <= safety_tolerance
        and safety["recall_at_100_upper_regret"] <= safety_tolerance
    )
    fidelity = ranking_fidelity(reference, candidate)
    ranking_certificate = ranking_safety_certificate(
        fidelity,
        seed=bootstrap_seed,
        resamples=bootstrap_resamples,
    )
    return {
        "queries": int(reference.shape[0]),
        "corpus": int(reference.shape[1]),
        "regret_sign": "positive_means_candidate_worse_than_full",
        "quality": quality,
        "ranking_fidelity": {
            name: {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "p05": float(np.quantile(values, 0.05)),
                "bootstrap": bootstrap_mean_interval(
                    values,
                    seed=bootstrap_seed,
                    resamples=bootstrap_resamples,
                ),
                "per_query": values.tolist(),
            }
            for name, values in fidelity.items()
        },
        "qrel_free_ranking_certificate": ranking_certificate,
        "safety_gate": safety,
    }


def summarize_costs(
    *,
    reference_vector_bytes: int,
    candidate_vector_bytes: int,
    vector_bytes_kind: str,
    reference_token_work: int | None = None,
    candidate_token_work: int | None = None,
    token_work_kind: str | None = None,
) -> dict[str, float | int | str | None]:
    """Return explicit candidate/reference physical cost ratios."""

    if reference_vector_bytes <= 0 or candidate_vector_bytes < 0:
        raise ValueError("vector byte counts must be non-negative with positive reference")
    if not vector_bytes_kind:
        raise ValueError("vector bytes kind must be declared")
    result: dict[str, float | int | str | None] = {
        "vector_bytes_kind": vector_bytes_kind,
        "reference_vector_bytes": int(reference_vector_bytes),
        "candidate_vector_bytes": int(candidate_vector_bytes),
        "vector_byte_fraction": candidate_vector_bytes / reference_vector_bytes,
        "vector_byte_reduction": 1.0 - candidate_vector_bytes / reference_vector_bytes,
        "reference_token_work": None,
        "candidate_token_work": None,
        "token_work_fraction": None,
        "token_work_kind": None,
    }
    if (reference_token_work is None) != (candidate_token_work is None):
        raise ValueError("reference and candidate token work must be supplied together")
    if reference_token_work is not None and candidate_token_work is not None:
        if reference_token_work <= 0 or candidate_token_work < 0:
            raise ValueError("token work must be non-negative with positive reference")
        if not token_work_kind:
            raise ValueError("token work kind must be declared")
        result.update(
            {
                "reference_token_work": int(reference_token_work),
                "candidate_token_work": int(candidate_token_work),
                "token_work_fraction": candidate_token_work / reference_token_work,
                "token_work_kind": token_work_kind,
            }
        )
    return result
