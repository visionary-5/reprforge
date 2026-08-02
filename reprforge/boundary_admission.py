"""Boundary-weighted admission for expensive document representations.

The admission policy treats a query batch as a workload episode.  Historical
fully observed rankings estimate how often a candidate at each locator rank
crosses the requested result boundary.  A page's admission value is the sum of
those rank risks over all of its appearances in the new episode.  This differs
from frequency-only admission: repeated appearances matter most when they are
near a decision boundary where expensive evidence can change membership.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean()) / max(float(values.std()), 1e-12)


@dataclass(frozen=True)
class BoundaryStatistics:
    flip_risk_by_rank: np.ndarray
    visual_prior_by_rank: np.ndarray


def fit_boundary_statistics(
    locator_zscores: np.ndarray,
    visual_zscores: np.ndarray,
    *,
    cutoff: int,
) -> BoundaryStatistics:
    """Fit qrel-free rank transition statistics from full-score logs."""

    locator = np.asarray(locator_zscores, dtype=np.float64)
    visual = np.asarray(visual_zscores, dtype=np.float64)
    if locator.shape != visual.shape or locator.ndim != 2:
        raise ValueError("locator and visual z-scores must be equal 2-D matrices")
    if cutoff <= 0 or cutoff >= locator.shape[1]:
        raise ValueError("cutoff must lie inside the candidate cohort")
    teacher_order = np.argsort(-(locator + visual), axis=1, kind="stable")
    teacher_membership = np.zeros_like(locator, dtype=bool)
    rows = np.arange(locator.shape[0])[:, None]
    teacher_membership[rows, teacher_order[:, :cutoff]] = True
    locator_membership = np.arange(locator.shape[1])[None, :] < cutoff
    return BoundaryStatistics(
        flip_risk_by_rank=np.mean(
            teacher_membership != locator_membership,
            axis=0,
        ),
        visual_prior_by_rank=np.mean(visual, axis=0),
    )


def select_episode_pages(
    candidate_pages: np.ndarray,
    *,
    budget_fraction: float,
    rank_weights: Sequence[float] | None,
) -> set[int]:
    """Select a page set by frequency or boundary-weighted frequency."""

    candidates = np.asarray(candidate_pages, dtype=np.int32)
    if candidates.ndim != 2:
        raise ValueError("candidate_pages must be a 2-D matrix")
    if not 0.0 <= budget_fraction <= 1.0:
        raise ValueError("budget_fraction must be in [0, 1]")
    weights = (
        np.ones(candidates.shape[1], dtype=np.float64)
        if rank_weights is None
        else np.asarray(rank_weights, dtype=np.float64)
    )
    if weights.shape != (candidates.shape[1],):
        raise ValueError("rank weights do not match the candidate cohort")
    scores: Counter[int] = Counter()
    for row in candidates:
        for rank, page in enumerate(row):
            scores[int(page)] += float(weights[rank])
    budget = math.floor(budget_fraction * len(scores))
    ordered = sorted(scores, key=lambda page: (-scores[page], page))
    return set(ordered[:budget])


def execute_boundary_plan(
    candidate_pages: np.ndarray,
    locator_zscores: np.ndarray,
    raw_visual_scores: np.ndarray,
    *,
    selected_pages: set[int],
    visual_prior_by_rank: Sequence[float],
    cutoff: int,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Execute a partial visual plan with train-only priors for missing pages.

    Actual visual scores are normalized only over the pages selected for the
    current query.  Missing candidates receive a rank-conditioned prior learned
    from other workload groups.  This makes the replay deployable without
    consulting the hidden full visual row at inference time.
    """

    candidates = np.asarray(candidate_pages, dtype=np.int32)
    locator = np.asarray(locator_zscores, dtype=np.float64)
    raw_visual = np.asarray(raw_visual_scores, dtype=np.float64)
    prior = np.asarray(visual_prior_by_rank, dtype=np.float64)
    if candidates.shape != locator.shape or candidates.shape != raw_visual.shape:
        raise ValueError("candidate, locator and visual matrices must align")
    if prior.shape != (candidates.shape[1],):
        raise ValueError("visual prior does not match candidate cohort")
    if cutoff <= 0 or cutoff > candidates.shape[1]:
        raise ValueError("cutoff must be within the candidate cohort")

    rankings = np.empty((len(candidates), cutoff), dtype=np.int32)
    events = 0
    for query, pages in enumerate(candidates):
        observed = np.asarray(
            [int(page) in selected_pages for page in pages],
            dtype=bool,
        )
        events += int(observed.sum())
        estimated_visual = prior.copy()
        if observed.sum() >= 2:
            estimated_visual[observed] = _zscore(raw_visual[query, observed])
        fused = locator[query] + estimated_visual
        order = sorted(
            range(len(pages)),
            key=lambda rank: (-float(fused[rank]), int(pages[rank])),
        )
        rankings[query] = pages[np.asarray(order[:cutoff], dtype=np.int32)]
    return rankings, {
        "selected_unique_pages": len(selected_pages),
        "visual_candidate_events": events,
        "mean_visual_pages_per_query": events / len(candidates),
    }
