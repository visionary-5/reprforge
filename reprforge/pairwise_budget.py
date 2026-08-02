"""Train-only budget calibration for complementary representation views."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from reprforge.boundary_admission import (
    execute_boundary_plan,
    select_episode_pages,
)
from reprforge.pairwise_view_admission import (
    build_boundary_pairs,
    select_pairwise_pages,
)


@dataclass(frozen=True)
class BudgetCalibration:
    baseline_fraction: float
    baseline_agreement: float
    selected_fraction: float
    selected_agreement: float
    grid_exhausted: bool


def _agreement(rankings: np.ndarray, teacher: np.ndarray) -> float:
    if rankings.shape != teacher.shape:
        raise ValueError("rankings and teacher must align")
    return float(
        np.mean(
            [
                set(row) == set(reference)
                for row, reference in zip(rankings, teacher, strict=True)
            ]
        )
    )


def calibrate_pair_budget(
    candidate_pages: np.ndarray,
    locator_zscores: np.ndarray,
    raw_visual_scores: np.ndarray,
    teacher: np.ndarray,
    *,
    rank_risk: Sequence[float],
    visual_prior_by_rank: Sequence[float],
    cutoff: int,
    baseline_fraction: float = 0.2,
    grid: Sequence[float] = (0.1, 0.12, 0.14, 0.16, 0.18, 0.2),
) -> BudgetCalibration:
    """Choose the smallest pair budget matching an additive train target.

    All inputs are historical score logs.  No qrels are accepted by the API.
    The returned fraction can then be applied to a future workload episode.
    """

    candidates = np.asarray(candidate_pages, dtype=np.int32)
    locator = np.asarray(locator_zscores, dtype=np.float64)
    visual = np.asarray(raw_visual_scores, dtype=np.float64)
    reference = np.asarray(teacher, dtype=np.int32)
    if candidates.shape != locator.shape or candidates.shape != visual.shape:
        raise ValueError("candidate and score matrices must align")
    if reference.shape != (len(candidates), cutoff):
        raise ValueError("teacher does not match the requested cutoff")
    if not 0.0 <= baseline_fraction <= 1.0:
        raise ValueError("baseline fraction must be in [0, 1]")
    fractions = tuple(float(value) for value in grid)
    if not fractions or any(not 0.0 <= value <= 1.0 for value in fractions):
        raise ValueError("budget grid must contain fractions in [0, 1]")
    if tuple(sorted(set(fractions))) != fractions:
        raise ValueError("budget grid must be strictly increasing")

    baseline_pages = select_episode_pages(
        candidates,
        budget_fraction=baseline_fraction,
        rank_weights=rank_risk,
    )
    baseline_rankings, _ = execute_boundary_plan(
        candidates,
        locator,
        visual,
        selected_pages=baseline_pages,
        visual_prior_by_rank=visual_prior_by_rank,
        cutoff=cutoff,
    )
    target = _agreement(baseline_rankings, reference)

    pairs = build_boundary_pairs(
        candidates,
        locator,
        cutoff=cutoff,
        rank_risk=rank_risk,
    )
    eligible = len(set(int(value) for value in candidates.flat))
    selected_fraction = fractions[-1]
    selected_agreement = 0.0
    exhausted = True
    for fraction in fractions:
        page_budget = math.floor(fraction * eligible)
        admission = select_pairwise_pages(pairs, page_budget=page_budget)
        rankings, _ = execute_boundary_plan(
            candidates,
            locator,
            visual,
            selected_pages=set(admission.selected_pages),
            visual_prior_by_rank=visual_prior_by_rank,
            cutoff=cutoff,
        )
        selected_agreement = _agreement(rankings, reference)
        selected_fraction = fraction
        if selected_agreement >= target:
            exhausted = False
            break
    return BudgetCalibration(
        baseline_fraction=baseline_fraction,
        baseline_agreement=target,
        selected_fraction=selected_fraction,
        selected_agreement=selected_agreement,
        grid_exhausted=exhausted,
    )

