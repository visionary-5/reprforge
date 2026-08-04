"""Phenomenon-first diagnostics for heterogeneous retrieval representations.

The atlas deliberately separates deployable, label-free portfolios from
label-using diagnostic ceilings.  Scores from different representations are
never compared directly: mixed-representation rankings use within-query
percentile ranks, while uniform routes retain their original rankings.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


def _finite_matrix(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not array.size:
        raise ValueError(f"{name} must be a non-empty 2-D matrix")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


@dataclass(frozen=True)
class ScoreCube:
    """Aligned retrieval surfaces and relevance judgments."""

    query_ids: tuple[str, ...]
    corpus_ids: tuple[str, ...]
    scores: Mapping[str, np.ndarray]
    relevance: tuple[Mapping[int, float], ...]
    split_roles: tuple[str, ...]

    def validate(self) -> None:
        if not self.query_ids or not self.corpus_ids or not self.scores:
            raise ValueError("score cube identifiers and routes cannot be empty")
        shape = (len(self.query_ids), len(self.corpus_ids))
        if len(set(self.query_ids)) != len(self.query_ids):
            raise ValueError("query identifiers must be unique")
        if len(set(self.corpus_ids)) != len(self.corpus_ids):
            raise ValueError("corpus identifiers must be unique")
        if len(self.relevance) != shape[0] or len(self.split_roles) != shape[0]:
            raise ValueError("relevance and split roles must be query-aligned")
        for route, values in self.scores.items():
            if _finite_matrix(values, name=f"scores[{route}]").shape != shape:
                raise ValueError(f"route {route!r} does not match cube shape")
        for row in self.relevance:
            if not row or any(index < 0 or index >= shape[1] for index in row):
                raise ValueError("every query needs in-corpus relevance labels")
            if any(not math.isfinite(value) or value <= 0 for value in row.values()):
                raise ValueError("relevance values must be finite and positive")

    @property
    def routes(self) -> tuple[str, ...]:
        return tuple(self.scores)


def deterministic_split_roles(
    query_ids: Sequence[str], *, eval_fraction: float = 1.0 / 3.0
) -> tuple[str, ...]:
    """Return a stable fit/eval partition without depending on row order."""

    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("eval_fraction must lie in (0, 1)")
    threshold = int(eval_fraction * (2**64))
    roles = []
    for query_id in query_ids:
        digest = hashlib.sha256(str(query_id).encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], "big")
        roles.append("eval" if bucket < threshold else "fit")
    if len(set(roles)) != 2:
        raise ValueError("deterministic split produced an empty partition")
    return tuple(roles)


def stable_ranks(scores: np.ndarray) -> np.ndarray:
    """Return one-based ranks with corpus position as the tie breaker."""

    values = _finite_matrix(scores, name="scores")
    ranks = np.empty(values.shape, dtype=np.int32)
    positions = np.arange(values.shape[1])
    for query_index, row in enumerate(values):
        order = np.lexsort((positions, -row))
        ranks[query_index, order] = np.arange(1, len(order) + 1)
    return ranks


def percentile_scores(scores: np.ndarray) -> np.ndarray:
    """Map each query row to a cross-route comparable rank percentile."""

    ranks = stable_ranks(scores)
    return 1.0 - (ranks.astype(np.float64) - 1.0) / ranks.shape[1]


def query_metrics(
    scores: np.ndarray,
    relevance: Sequence[Mapping[int, float]],
    *,
    ks: Sequence[int],
) -> dict[str, np.ndarray]:
    values = _finite_matrix(scores, name="scores")
    if values.shape[0] != len(relevance):
        raise ValueError("scores and relevance must be query-aligned")
    positions = np.arange(values.shape[1])
    output = {
        **{f"ndcg_at_{k}": np.zeros(values.shape[0]) for k in ks},
        **{f"recall_at_{k}": np.zeros(values.shape[0]) for k in ks},
    }
    for query_index, row in enumerate(values):
        order = np.lexsort((positions, -row))
        labels = relevance[query_index]
        denominator = sum(labels.values())
        ideal = sorted(labels.values(), reverse=True)
        for k in ks:
            top = order[:k]
            retrieved = sum(labels.get(int(index), 0.0) for index in top)
            output[f"recall_at_{k}"][query_index] = retrieved / denominator
            dcg = sum(
                labels.get(int(index), 0.0) / math.log2(rank + 1)
                for rank, index in enumerate(top, start=1)
            )
            idcg = sum(
                value / math.log2(rank + 1)
                for rank, value in enumerate(ideal[:k], start=1)
            )
            output[f"ndcg_at_{k}"][query_index] = dcg / idcg if idcg else 0.0
    return output


def _mask(cube: ScoreCube, role: str) -> np.ndarray:
    if role == "all":
        return np.ones(len(cube.query_ids), dtype=bool)
    aliases = {"dev": "fit", "holdout": "eval"}
    normalized = np.asarray(
        [aliases.get(value, value) for value in cube.split_roles], dtype=object
    )
    result = normalized == role
    if not result.any():
        raise ValueError(f"split role {role!r} selects no queries")
    return result


def _mean_metrics(
    per_query: Mapping[str, np.ndarray], mask: np.ndarray
) -> dict[str, float]:
    return {name: float(values[mask].mean()) for name, values in per_query.items()}


def paired_bootstrap_ci(
    left: np.ndarray,
    right: np.ndarray,
    *,
    seed: int = 0,
    resamples: int = 4000,
) -> dict[str, float | int]:
    """Paired query bootstrap for ``left - right``."""

    difference = np.asarray(left, dtype=np.float64) - np.asarray(
        right, dtype=np.float64
    )
    if difference.ndim != 1 or not len(difference):
        raise ValueError("bootstrap inputs must be aligned non-empty vectors")
    rng = np.random.default_rng(seed)
    draws = rng.choice(difference, size=(resamples, len(difference)), replace=True)
    means = draws.mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return {
        "mean_difference": float(difference.mean()),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "queries": int(len(difference)),
        "resamples": int(resamples),
    }


def _rrf_scores(ranks: Mapping[str, np.ndarray], *, constant: int = 60) -> np.ndarray:
    return sum(1.0 / (constant + values) for values in ranks.values())


def _best_rank_scores(ranks: Mapping[str, np.ndarray]) -> np.ndarray:
    return -np.minimum.reduce(tuple(ranks.values())).astype(np.float64)


def _static_relevance_rank_plan(
    cube: ScoreCube,
    ranks: Mapping[str, np.ndarray],
    *,
    baseline_route: str,
) -> tuple[np.ndarray, dict[str, int]]:
    """Fit one representation per document from fit-query relevant ranks.

    This is an intentionally optimistic diagnostic planner: it may use fit
    labels, but its selected route is static across all evaluation queries.
    Documents without fit relevance observations retain the global baseline.
    """

    fit = _mask(cube, "fit")
    route_names = cube.routes
    choices = np.full(len(cube.corpus_ids), route_names.index(baseline_route))
    observed = np.zeros(len(cube.corpus_ids), dtype=bool)
    rank_sums = np.zeros((len(route_names), len(cube.corpus_ids)))
    weights = np.zeros(len(cube.corpus_ids))
    for query_index in np.flatnonzero(fit):
        for corpus_index, relevance in cube.relevance[query_index].items():
            observed[corpus_index] = True
            weights[corpus_index] += relevance
            for route_index, route in enumerate(route_names):
                rank_sums[route_index, corpus_index] += (
                    relevance * ranks[route][query_index, corpus_index]
                )
    for corpus_index in np.flatnonzero(observed):
        mean_ranks = rank_sums[:, corpus_index] / weights[corpus_index]
        choices[corpus_index] = int(np.argmin(mean_ranks))
    counts = {
        route: int(np.sum(choices == route_index))
        for route_index, route in enumerate(route_names)
    }
    counts["fit_observed_documents"] = int(observed.sum())
    return choices, counts


def analyze_cube(
    cube: ScoreCube,
    *,
    ks: Sequence[int] = (5, 10),
    target_metric: str = "ndcg_at_10",
    costs: Mapping[str, Mapping[str, float]] | None = None,
) -> dict:
    """Build one atlas report with explicit deployability boundaries."""

    cube.validate()
    if any(k <= 0 or k > len(cube.corpus_ids) for k in ks):
        raise ValueError("ks must lie inside the corpus")
    ranks = {route: stable_ranks(values) for route, values in cube.scores.items()}
    percentiles = {
        route: 1.0 - (values.astype(np.float64) - 1.0) / len(cube.corpus_ids)
        for route, values in ranks.items()
    }
    metrics = {
        route: query_metrics(values, cube.relevance, ks=ks)
        for route, values in cube.scores.items()
    }
    if target_metric not in next(iter(metrics.values())):
        raise ValueError(f"unknown target metric: {target_metric}")

    fit = _mask(cube, "fit")
    evaluation = _mask(cube, "eval")
    best_global = max(
        cube.routes,
        key=lambda route: float(metrics[route][target_metric][fit].mean()),
    )
    rrf = query_metrics(_rrf_scores(ranks), cube.relevance, ks=ks)
    best_rank = query_metrics(_best_rank_scores(ranks), cube.relevance, ks=ks)

    static_choices, static_counts = _static_relevance_rank_plan(
        cube, ranks, baseline_route=best_global
    )
    mixed = np.empty((len(cube.query_ids), len(cube.corpus_ids)))
    for corpus_index, route_index in enumerate(static_choices):
        route = cube.routes[int(route_index)]
        # Different routes can assign the same integer rank to different
        # documents. A bounded best-global secondary key makes those ties
        # deterministic without overturning adjacent primary rank levels.
        mixed[:, corpus_index] = percentiles[route][:, corpus_index] + (
            percentiles[best_global][:, corpus_index]
            / (len(cube.corpus_ids) + 1.0)
        )
    static_metrics = query_metrics(mixed, cube.relevance, ks=ks)

    route_target = np.stack(
        [metrics[route][target_metric] for route in cube.routes], axis=0
    )
    oracle_route_indices = np.argmax(route_target, axis=0)
    query_oracle = route_target[oracle_route_indices, np.arange(route_target.shape[1])]
    eval_oracle_counts = {
        route: int(np.sum(oracle_route_indices[evaluation] == index))
        for index, route in enumerate(cube.routes)
    }

    route_pairs: dict[str, dict[str, float]] = {}
    for left_index, left in enumerate(cube.routes):
        for right in cube.routes[left_index + 1 :]:
            pair = f"{left}__{right}"
            per_k = {}
            for k in ks:
                intersections = []
                for query_index in np.flatnonzero(evaluation):
                    left_top = set(np.flatnonzero(ranks[left][query_index] <= k))
                    right_top = set(np.flatnonzero(ranks[right][query_index] <= k))
                    intersections.append(len(left_top & right_top) / len(left_top | right_top))
                per_k[f"top_{k}_jaccard"] = float(np.mean(intersections))
            route_pairs[pair] = per_k

    evidence_ceiling = {}
    rescue = {}
    minimum_ranks = np.minimum.reduce(tuple(ranks.values()))
    baseline_ranks = ranks[best_global]
    for k in ks:
        potential = []
        rescued_mass = []
        for query_index in np.flatnonzero(evaluation):
            labels = cube.relevance[query_index]
            denominator = sum(labels.values())
            potential.append(
                sum(
                    relevance
                    for corpus_index, relevance in labels.items()
                    if minimum_ranks[query_index, corpus_index] <= k
                )
                / denominator
            )
            rescued_mass.append(
                sum(
                    relevance
                    for corpus_index, relevance in labels.items()
                    if baseline_ranks[query_index, corpus_index] > k
                    and minimum_ranks[query_index, corpus_index] <= k
                )
                / denominator
            )
        evidence_ceiling[f"recall_at_{k}"] = float(np.mean(potential))
        rescue[f"relevant_mass_at_{k}"] = float(np.mean(rescued_mass))

    best_global_eval = metrics[best_global][target_metric][evaluation]
    report = {
        "schema_version": 1,
        "queries": len(cube.query_ids),
        "corpus": len(cube.corpus_ids),
        "routes": list(cube.routes),
        "split_counts": {
            "fit": int(fit.sum()),
            "eval": int(evaluation.sum()),
        },
        "target_metric": target_metric,
        "best_global_route_selected_on_fit": best_global,
        "uniform_routes": {
            route: {
                "fit": _mean_metrics(per_query, fit),
                "eval": _mean_metrics(per_query, evaluation),
                "all": _mean_metrics(per_query, _mask(cube, "all")),
                "cost": dict((costs or {}).get(route, {})),
            }
            for route, per_query in metrics.items()
        },
        "label_free_portfolios": {
            "rrf_all_routes": _mean_metrics(rrf, evaluation),
            "best_rank_pool": _mean_metrics(best_rank, evaluation),
        },
        "diagnostic_upper_bounds": {
            "query_route_oracle": {
                target_metric: float(query_oracle[evaluation].mean()),
                "selected_routes": eval_oracle_counts,
                "gap_over_best_global": paired_bootstrap_ci(
                    query_oracle[evaluation], best_global_eval
                ),
            },
            "relevant_evidence_best_rank": evidence_ceiling,
            "relevant_evidence_rescued_over_best_global": rescue,
        },
        "fit_label_static_document_plan": {
            "route_counts": static_counts,
            "eval": _mean_metrics(static_metrics, evaluation),
            "gap_over_best_global": paired_bootstrap_ci(
                static_metrics[target_metric][evaluation], best_global_eval
            ),
            "warning": (
                "Routes are learned from fit relevance and fixed per document; "
                "this diagnoses corpus-side heterogeneity, not a deployable policy."
            ),
        },
        "route_disagreement": route_pairs,
        "interpretation_guardrails": {
            "mixed_score_calibration": (
                "within-query rank percentile; best-global percentile is a "
                "bounded secondary tie-breaker"
            ),
            "query_route_oracle_uses_eval_labels": True,
            "best_rank_pool_is_coherent_label_free_ranking": True,
            "relevant_evidence_best_rank_is_not_coherent_ranking": True,
        },
    }
    return report
