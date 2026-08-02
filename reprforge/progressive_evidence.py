"""Progressive acquisition of expensive evidence for document retrieval.

The module deliberately separates two quantities that fixed candidate-depth
experiments conflate:

* ``candidate_limit`` is the largest first-stage cohort that defines the
  reference reranking decision; and
* ``acquired_pages`` is the amount of expensive visual evidence actually
  observed before the runtime can stop.

The initial policy is intentionally small.  It acquires BM25 candidates in
two-page increments and stops only when the Top-1 decision is unchanged from
the previous stage and its fused-score margin exceeds a stage-specific
threshold.  Thresholds are selected without qrels: a fully observed ranking
from historical queries acts as the teacher.  Relevance labels are reserved
for the final quality evaluation.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


DEFAULT_STAGES = (2, 4, 6, 8, 10)
DEFAULT_THRESHOLD_GRID: Mapping[int, tuple[float, ...]] = {
    4: (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0),
    6: (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0),
    8: (0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0),
}


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean()) / max(float(values.std()), 1e-12)


def _rank_row(scores: np.ndarray, identifiers: Sequence[str]) -> np.ndarray:
    return np.asarray(
        sorted(
            range(len(identifiers)),
            key=lambda position: (-float(scores[position]), identifiers[position]),
        ),
        dtype=np.int32,
    )


@dataclass(frozen=True)
class EvidenceStages:
    """Frozen progressive rankings and observable decision margins."""

    base_order: np.ndarray
    orders: Mapping[int, np.ndarray]
    top1_margins: Mapping[int, np.ndarray]
    stages: tuple[int, ...]
    candidate_limit: int


def build_evidence_stages(
    corpus_ids: Sequence[str],
    locator_scores: np.ndarray,
    expensive_scores: np.ndarray,
    *,
    stages: Sequence[int] = DEFAULT_STAGES,
    top_k: int = 20,
) -> EvidenceStages:
    """Replay rankings after progressively observing expensive page scores.

    Candidate membership always follows the deterministic locator ranking.
    At stage ``m``, only the first ``m`` candidates are fused using the same
    candidate-relative z-score rule as the deployed ReprForge prototype.  The
    untouched tail preserves locator order.
    """

    locator = np.asarray(locator_scores, dtype=np.float64)
    expensive = np.asarray(expensive_scores, dtype=np.float64)
    if locator.shape != expensive.shape or locator.ndim != 2:
        raise ValueError("locator and expensive scores must be equal 2-D matrices")
    if locator.shape[1] != len(corpus_ids):
        raise ValueError("score columns do not match corpus identifiers")
    normalized_stages = tuple(sorted({int(value) for value in stages}))
    if not normalized_stages or normalized_stages[0] < 2:
        raise ValueError("at least one stage >=2 is required")
    if any(left % 2 or right - left != 2 for left, right in zip(
        normalized_stages, normalized_stages[1:], strict=False
    )) or normalized_stages[0] != 2:
        raise ValueError("the reference policy requires contiguous two-page stages")
    candidate_limit = normalized_stages[-1]
    if candidate_limit > locator.shape[1]:
        raise ValueError("candidate limit exceeds the corpus")
    if top_k <= 0 or top_k > locator.shape[1]:
        raise ValueError("top_k must be within the corpus")

    base_order = np.stack(
        [_rank_row(row, corpus_ids) for row in locator],
        axis=0,
    )
    orders = {
        stage: np.empty((locator.shape[0], top_k), dtype=np.int32)
        for stage in normalized_stages
    }
    margins = {
        stage: np.empty(locator.shape[0], dtype=np.float64)
        for stage in normalized_stages[1:]
    }
    for query in range(locator.shape[0]):
        for stage in normalized_stages:
            candidates = base_order[query, :stage]
            fused = _zscore(locator[query, candidates]) + _zscore(
                expensive[query, candidates]
            )
            candidate_offsets = np.asarray(
                sorted(
                    range(stage),
                    key=lambda offset: (
                        -float(fused[offset]),
                        corpus_ids[int(candidates[offset])],
                    ),
                ),
                dtype=np.int32,
            )
            ranked_candidates = candidates[candidate_offsets]
            selected = set(int(value) for value in candidates)
            tail = [
                int(value)
                for value in base_order[query]
                if int(value) not in selected
            ]
            orders[stage][query] = np.asarray(
                [*ranked_candidates.tolist(), *tail][:top_k],
                dtype=np.int32,
            )
            if stage != normalized_stages[0]:
                sorted_fused = np.sort(fused)
                margins[stage][query] = float(
                    sorted_fused[-1] - sorted_fused[-2]
                )
    return EvidenceStages(
        base_order=base_order[:, :top_k],
        orders=orders,
        top1_margins=margins,
        stages=normalized_stages,
        candidate_limit=candidate_limit,
    )


def apply_progressive_policy(
    evidence: EvidenceStages,
    query_indices: Sequence[int],
    thresholds: Mapping[int, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return selected acquisition depths and corresponding ranked lists."""

    indices = np.asarray(query_indices, dtype=np.int32)
    selected = np.full(len(indices), evidence.candidate_limit, dtype=np.int16)
    active = np.ones(len(indices), dtype=bool)
    for previous, stage in zip(evidence.stages, evidence.stages[1:], strict=True):
        if stage == evidence.candidate_limit:
            break
        if stage not in thresholds:
            raise ValueError(f"missing stopping threshold for stage {stage}")
        stable = (
            evidence.orders[previous][indices, 0]
            == evidence.orders[stage][indices, 0]
        )
        confident = evidence.top1_margins[stage][indices] >= thresholds[stage]
        stop = active & stable & confident
        selected[stop] = stage
        active[stop] = False
    rankings = np.stack(
        [
            evidence.orders[int(depth)][int(query)]
            for query, depth in zip(indices, selected, strict=True)
        ]
    )
    return selected, rankings


def select_zero_error_thresholds(
    evidence: EvidenceStages,
    train_indices: Sequence[int],
    *,
    threshold_grid: Mapping[int, Sequence[float]] = DEFAULT_THRESHOLD_GRID,
) -> dict[int, float]:
    """Select the least-work thresholds with zero teacher disagreement.

    This is an empirical, deliberately conservative selector, not a formal
    finite-sample guarantee.  Its purpose is to test whether the observable
    stability signal survives a paper-disjoint evaluation before introducing
    a more elaborate risk-control method.
    """

    indices = np.asarray(train_indices, dtype=np.int32)
    if not len(indices):
        raise ValueError("training indices cannot be empty")
    decision_stages = evidence.stages[1:-1]
    if set(decision_stages) != set(threshold_grid):
        raise ValueError("threshold grid must cover every non-terminal stage")
    teacher = evidence.orders[evidence.candidate_limit][indices, 0]
    best: tuple[float, tuple[float, ...], dict[int, float]] | None = None
    grids = [tuple(float(value) for value in threshold_grid[stage]) for stage in decision_stages]
    for values in itertools.product(*grids):
        thresholds = dict(zip(decision_stages, values, strict=True))
        selected, rankings = apply_progressive_policy(
            evidence, indices, thresholds
        )
        if np.any(rankings[:, 0] != teacher):
            continue
        candidate = (
            float(selected.mean()),
            tuple(-value for value in values),
            thresholds,
        )
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        raise ValueError("no zero-error threshold configuration exists")
    return best[2]


def paper_disjoint_progressive_probe(
    evidence: EvidenceStages,
    groups: Sequence[str],
    *,
    threshold_grid: Mapping[int, Sequence[float]] = DEFAULT_THRESHOLD_GRID,
) -> dict[str, object]:
    """Run leave-one-source-group-out selection and return held-out rankings."""

    group_values = np.asarray([str(value) for value in groups])
    if len(group_values) != evidence.base_order.shape[0]:
        raise ValueError("group labels do not match the query count")
    rankings = np.empty_like(evidence.orders[evidence.candidate_limit])
    selected = np.empty(len(group_values), dtype=np.int16)
    folds: list[dict[str, object]] = []
    for group in sorted(set(group_values), key=lambda value: (len(value), value)):
        test = np.flatnonzero(group_values == group)
        train = np.flatnonzero(group_values != group)
        thresholds = select_zero_error_thresholds(
            evidence,
            train,
            threshold_grid=threshold_grid,
        )
        fold_selected, fold_rankings = apply_progressive_policy(
            evidence,
            test,
            thresholds,
        )
        selected[test] = fold_selected
        rankings[test] = fold_rankings
        teacher = evidence.orders[evidence.candidate_limit][test, 0]
        folds.append(
            {
                "held_out_group": group,
                "queries": int(len(test)),
                "thresholds": {str(key): value for key, value in thresholds.items()},
                "mean_acquired_pages": float(fold_selected.mean()),
                "top1_teacher_disagreements": int(
                    np.sum(fold_rankings[:, 0] != teacher)
                ),
            }
        )
    teacher = evidence.orders[evidence.candidate_limit][:, 0]
    return {
        "selected_depths": selected,
        "rankings": rankings,
        "folds": folds,
        "mean_acquired_pages": float(selected.mean()),
        "top1_teacher_disagreements": int(np.sum(rankings[:, 0] != teacher)),
        "top1_teacher_disagreement_rate": float(np.mean(rankings[:, 0] != teacher)),
    }


def paper_disjoint_bm25_margin_router(
    evidence: EvidenceStages,
    groups: Sequence[str],
    locator_scores: np.ndarray,
) -> dict[str, object]:
    """Strong cheap baseline: route between no visual work and the full cohort."""

    locator = np.asarray(locator_scores, dtype=np.float64)
    group_values = np.asarray([str(value) for value in groups])
    if locator.shape[0] != len(group_values):
        raise ValueError("locator scores and groups differ in query count")
    top_two = evidence.base_order[:, :2]
    margins = locator[np.arange(len(locator)), top_two[:, 0]] - locator[
        np.arange(len(locator)), top_two[:, 1]
    ]
    teacher = evidence.orders[evidence.candidate_limit][:, 0]
    selected = np.full(len(group_values), evidence.candidate_limit, dtype=np.int16)
    rankings = evidence.orders[evidence.candidate_limit].copy()
    thresholds: dict[str, float] = {}
    for group in sorted(set(group_values), key=lambda value: (len(value), value)):
        test = np.flatnonzero(group_values == group)
        train = np.flatnonzero(group_values != group)
        unsafe = train[evidence.base_order[train, 0] != teacher[train]]
        threshold = (
            float(np.nextafter(margins[unsafe].max(), np.inf))
            if len(unsafe)
            else float("-inf")
        )
        stop = margins[test] >= threshold
        stopped = test[stop]
        selected[stopped] = 0
        rankings[stopped] = evidence.base_order[stopped]
        thresholds[group] = threshold
    return {
        "selected_depths": selected,
        "rankings": rankings,
        "thresholds": thresholds,
        "mean_acquired_pages": float(selected.mean()),
        "top1_teacher_disagreements": int(np.sum(rankings[:, 0] != teacher)),
        "top1_teacher_disagreement_rate": float(np.mean(rankings[:, 0] != teacher)),
    }
