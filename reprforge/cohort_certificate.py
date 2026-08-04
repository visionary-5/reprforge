"""Teacher-only cohort certificates for selective representation construction.

A certificate is a subset of expensive candidate representations that, under
a frozen completion rule, reproduces the complete teacher's Top-k set.  The
selector can inspect the full expensive surface, but never qrels.  It is an
oracle headroom diagnostic, not a deployable policy.
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from reprforge.heterogeneity_atlas import ScoreCube, paired_bootstrap_ci, query_metrics
from reprforge.landmark_probe import (
    _completed_rerank_scores,
    _fit_completion,
    _initial_order,
    _zscore,
)


@dataclass(frozen=True)
class QueryCertificate:
    candidate_indices: np.ndarray
    observed_positions: np.ndarray
    teacher_top_positions: np.ndarray
    certificate_top_positions: np.ndarray
    certificate_scores: np.ndarray
    removals: tuple[int, ...]

    @property
    def size(self) -> int:
        return int(len(self.observed_positions))


def _top_positions(values: np.ndarray, k: int) -> np.ndarray:
    positions = np.arange(len(values))
    return np.lexsort((positions, -values))[:k]


def _teacher_order_score(
    ranked: np.ndarray, teacher_top: np.ndarray
) -> float:
    relevance = {
        int(position): float(len(teacher_top) - rank)
        for rank, position in enumerate(teacher_top)
    }
    dcg = sum(
        relevance.get(int(position), 0.0) / np.log2(rank + 2)
        for rank, position in enumerate(ranked[: len(teacher_top)])
    )
    ideal = sum(
        value / np.log2(rank + 2)
        for rank, value in enumerate(sorted(relevance.values(), reverse=True))
    )
    return float(dcg / ideal) if ideal else 0.0


def _completed_values(
    x: np.ndarray, y: np.ndarray, observed: np.ndarray
) -> np.ndarray:
    return x + _zscore(_fit_completion(x, y, observed))


def _matches_teacher(
    ranked: np.ndarray,
    teacher_top: np.ndarray,
    objective: Literal["set", "order"],
) -> bool:
    if objective == "order":
        return bool(np.array_equal(ranked, teacher_top))
    if objective == "set":
        return set(int(value) for value in ranked) == set(
            int(value) for value in teacher_top
        )
    raise ValueError(f"unknown certificate objective: {objective}")


def greedy_certificate(
    base_row: np.ndarray,
    expensive_row: np.ndarray,
    *,
    candidate_k: int = 20,
    target_k: int = 10,
    anchors: int = 3,
    objective: Literal["set", "order"] = "set",
) -> QueryCertificate:
    """Return an inclusion-minimal backward-elimination certificate."""

    base = np.asarray(base_row, dtype=np.float64)
    expensive = np.asarray(expensive_row, dtype=np.float64)
    if base.ndim != 1 or base.shape != expensive.shape:
        raise ValueError("base and expensive rows must be aligned vectors")
    if not anchors >= 2 or not anchors <= target_k <= candidate_k <= len(base):
        raise ValueError("require 2 <= anchors <= target_k <= candidate_k")
    corpus_positions = np.arange(len(base))
    candidates = np.lexsort((corpus_positions, -base))[:candidate_k]
    x = _zscore(base[candidates])
    y = expensive[candidates]
    teacher_values = x + _zscore(y)
    teacher_top = _top_positions(teacher_values, target_k)
    anchor_positions = set(_initial_order(candidate_k)[:anchors])
    observed = np.ones(candidate_k, dtype=bool)
    removals: list[int] = []

    while int(observed.sum()) > anchors:
        choices: list[tuple[float, int, np.ndarray, np.ndarray]] = []
        for position in np.flatnonzero(observed):
            position = int(position)
            if position in anchor_positions:
                continue
            trial = observed.copy()
            trial[position] = False
            values = _completed_values(x, y, trial)
            ranked = _top_positions(values, target_k)
            if not _matches_teacher(ranked, teacher_top, objective):
                continue
            choices.append(
                (_teacher_order_score(ranked, teacher_top), -position, trial, ranked)
            )
        if not choices:
            break
        _, negative_position, observed, _ = max(
            choices, key=lambda value: (value[0], value[1])
        )
        removals.append(-negative_position)

    certificate_values = _completed_values(x, y, observed)
    certificate_top = _top_positions(certificate_values, target_k)
    if not _matches_teacher(certificate_top, teacher_top, objective):
        raise AssertionError(f"certificate lost the teacher Top-k {objective}")
    return QueryCertificate(
        candidate_indices=candidates,
        observed_positions=np.flatnonzero(observed).astype(np.int32),
        teacher_top_positions=teacher_top.astype(np.int32),
        certificate_top_positions=certificate_top.astype(np.int32),
        certificate_scores=_completed_rerank_scores(
            base, candidates, certificate_values
        ),
        removals=tuple(removals),
    )


def exact_certificate_size(
    base_row: np.ndarray,
    expensive_row: np.ndarray,
    *,
    candidate_k: int,
    target_k: int,
    upper_bound: int,
    anchors: int = 3,
    max_combinations: int = 200_000,
    objective: Literal["set", "order"] = "set",
) -> dict:
    """Audit the minimum certificate size below a greedy upper bound."""

    base = np.asarray(base_row, dtype=np.float64)
    expensive = np.asarray(expensive_row, dtype=np.float64)
    corpus_positions = np.arange(len(base))
    candidates = np.lexsort((corpus_positions, -base))[:candidate_k]
    x = _zscore(base[candidates])
    y = expensive[candidates]
    teacher_top = _top_positions(x + _zscore(y), target_k)
    anchor_positions = tuple(_initial_order(candidate_k)[:anchors])
    optional = tuple(
        position for position in range(candidate_k) if position not in anchor_positions
    )
    evaluated = 0
    for size in range(anchors, upper_bound):
        for chosen in itertools.combinations(optional, size - anchors):
            evaluated += 1
            if evaluated > max_combinations:
                return {
                    "exact_size": None,
                    "evaluated_combinations": evaluated - 1,
                    "exhausted": False,
                    "upper_bound": upper_bound,
                }
            observed = np.zeros(candidate_k, dtype=bool)
            observed[list(anchor_positions) + list(chosen)] = True
            ranked = _top_positions(
                _completed_values(x, y, observed), target_k
            )
            if _matches_teacher(ranked, teacher_top, objective):
                return {
                    "exact_size": size,
                    "evaluated_combinations": evaluated,
                    "exhausted": True,
                    "upper_bound": upper_bound,
                }
    return {
        "exact_size": upper_bound,
        "evaluated_combinations": evaluated,
        "exhausted": True,
        "upper_bound": upper_bound,
    }


def analyze_certificates(
    cube: ScoreCube,
    *,
    base_route: str,
    expensive_route: str,
    candidate_k: int = 20,
    target_k: int = 10,
    target_metric: str = "ndcg_at_10",
    per_item_build_ms: Sequence[float] | None = None,
    exact_audit_queries: int = 8,
    objective: Literal["set", "order"] = "set",
) -> dict:
    """Analyze greedy certificates and a deterministic exact audit subset."""

    cube.validate()
    base = cube.scores[base_route]
    expensive = cube.scores[expensive_route]
    certificates = [
        greedy_certificate(
            base[query_index],
            expensive[query_index],
            candidate_k=candidate_k,
            target_k=target_k,
            objective=objective,
        )
        for query_index in range(len(cube.query_ids))
    ]
    certificate_surface = np.stack(
        [certificate.certificate_scores for certificate in certificates]
    )
    teacher_surface = np.stack(
        [
            _completed_rerank_scores(
                base[query_index],
                certificate.candidate_indices,
                _zscore(base[query_index, certificate.candidate_indices])
                + _zscore(expensive[query_index, certificate.candidate_indices]),
            )
            for query_index, certificate in enumerate(certificates)
        ]
    )
    base_values = query_metrics(base, cube.relevance, ks=(target_k,))[target_metric]
    teacher_values = query_metrics(
        teacher_surface, cube.relevance, ks=(target_k,)
    )[target_metric]
    certificate_values = query_metrics(
        certificate_surface, cube.relevance, ks=(target_k,)
    )[target_metric]
    sizes = np.asarray([certificate.size for certificate in certificates])
    ordered_agreement = np.asarray(
        [
            np.mean(
                certificate.teacher_top_positions
                == certificate.certificate_top_positions
            )
            for certificate in certificates
        ]
    )
    all_candidate_docs = {
        int(corpus_index)
        for certificate in certificates
        for corpus_index in certificate.candidate_indices
    }
    certificate_docs = {
        int(certificate.candidate_indices[position])
        for certificate in certificates
        for position in certificate.observed_positions
    }
    costs = {}
    if per_item_build_ms is not None:
        item_costs = np.asarray(per_item_build_ms, dtype=np.float64)
        if item_costs.shape != (len(cube.corpus_ids),):
            raise ValueError("per-item build costs must be corpus-aligned")
        costs = {
            "certificate_unique_build_ms": float(item_costs[list(certificate_docs)].sum()),
            "all_candidate_unique_build_ms": float(item_costs[list(all_candidate_docs)].sum()),
            "full_corpus_build_ms": float(item_costs.sum()),
        }

    audit_order = sorted(
        range(len(cube.query_ids)),
        key=lambda index: hashlib.sha256(cube.query_ids[index].encode()).hexdigest(),
    )[:exact_audit_queries]
    audits = []
    for query_index in audit_order:
        result = exact_certificate_size(
            base[query_index],
            expensive[query_index],
            candidate_k=candidate_k,
            target_k=target_k,
            upper_bound=certificates[query_index].size,
            objective=objective,
        )
        audits.append({"query_id": cube.query_ids[query_index], **result})

    full_gain = float((teacher_values - base_values).mean())
    certificate_gain = float((certificate_values - base_values).mean())
    return {
        "schema_version": 1,
        "queries": len(cube.query_ids),
        "corpus": len(cube.corpus_ids),
        "candidate_k": candidate_k,
        "target_k": target_k,
        "base_route": base_route,
        "expensive_route": expensive_route,
        "selection_uses_qrels": False,
        "selection_uses_complete_teacher_surface": True,
        "certificate_objective": objective,
        "completion_contract": (
            "three fixed rank-coverage anchors plus query-local quadratic "
            "completion; selected items replace predictions with exact scores"
        ),
        "certificate_size": {
            "mean": float(sizes.mean()),
            "median": float(np.median(sizes)),
            "p90": float(np.quantile(sizes, 0.90)),
            "p95": float(np.quantile(sizes, 0.95)),
            "minimum": int(sizes.min()),
            "maximum": int(sizes.max()),
            "fraction_at_most_8": float(np.mean(sizes <= 8)),
            "fraction_at_most_12": float(np.mean(sizes <= 12)),
        },
        "quality": {
            "base": float(base_values.mean()),
            "full_candidate_teacher": float(teacher_values.mean()),
            "certificate": float(certificate_values.mean()),
            "certificate_vs_teacher": paired_bootstrap_ci(
                certificate_values, teacher_values
            ),
            "full_fusion_gain_recovery": (
                certificate_gain / full_gain if abs(full_gain) > 1e-12 else 0.0
            ),
            "mean_ordered_topk_position_agreement": float(ordered_agreement.mean()),
            "topk_set_agreement": 1.0,
            "topk_order_agreement": (
                1.0 if objective == "order" else float(ordered_agreement.mean())
            ),
        },
        "physical_reuse": {
            "certificate_unique_documents": len(certificate_docs),
            "all_candidate_unique_documents": len(all_candidate_docs),
            "full_corpus_documents": len(cube.corpus_ids),
            "certificate_fraction_of_all_candidate_documents": (
                len(certificate_docs) / len(all_candidate_docs)
            ),
            **costs,
        },
        "exact_audit": audits,
        "per_query": [
            {
                "query_id": cube.query_ids[index],
                "certificate_size": certificate.size,
                "ordered_topk_position_agreement": float(ordered_agreement[index]),
                "observed_corpus_ids": [
                    cube.corpus_ids[int(certificate.candidate_indices[position])]
                    for position in certificate.observed_positions
                ],
            }
            for index, certificate in enumerate(certificates)
        ],
    }
