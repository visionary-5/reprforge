"""Observable physical-cost model for an atomic representation plan.

The planner must not optimize a page-count proxy and later claim a latency
win.  This module models the work that the current ReprForge executor actually
performs: one atomic materialization submission, fixed-size encoder batches,
and candidate--representation scoring events.  Coefficients are fitted from
measured historical runs with a tiny non-negative least-squares solver so that
the model remains dependency-free and physically interpretable.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class AtomicCostObservation:
    pages: int
    score_events: int
    total_ms: float

    def __post_init__(self) -> None:
        if self.pages < 0 or self.score_events < 0:
            raise ValueError("physical work counters cannot be negative")
        if not math.isfinite(self.total_ms) or self.total_ms < 0:
            raise ValueError("measured time must be finite and non-negative")


@dataclass(frozen=True)
class AtomicCostEstimate:
    total_ms: float
    setup_ms: float
    page_ms: float
    batch_ms: float
    score_ms: float
    pages: int
    batches: int
    score_events: int


@dataclass(frozen=True)
class AtomicMaterializationCostModel:
    """A non-negative, batch-aware latency model.

    ``total = setup + page_ms * pages + batch_ms * ceil(pages / batch_size)
             + score_event_ms * score_events``

    It is intentionally small.  Image shape or representation type can be
    added as separate models when the executor admits heterogeneous routes;
    the current ColPali path resizes every page to one public processor shape.
    """

    batch_size: int
    setup_ms: float
    page_ms: float
    batch_ms: float
    score_event_ms: float

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch size must be positive")
        values = (self.setup_ms, self.page_ms, self.batch_ms, self.score_event_ms)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("physical cost coefficients must be finite and non-negative")

    def estimate(self, *, pages: int, score_events: int) -> AtomicCostEstimate:
        if pages < 0 or score_events < 0:
            raise ValueError("physical work counters cannot be negative")
        batches = math.ceil(pages / self.batch_size) if pages else 0
        page_cost = self.page_ms * pages
        batch_cost = self.batch_ms * batches
        score_cost = self.score_event_ms * score_events
        setup = self.setup_ms if pages or score_events else 0.0
        return AtomicCostEstimate(
            total_ms=setup + page_cost + batch_cost + score_cost,
            setup_ms=setup,
            page_ms=page_cost,
            batch_ms=batch_cost,
            score_ms=score_cost,
            pages=pages,
            batches=batches,
            score_events=score_events,
        )


def _design_row(pages: int, score_events: int, batch_size: int) -> list[float]:
    batches = math.ceil(pages / batch_size) if pages else 0
    return [1.0, float(pages), float(batches), float(score_events)]


def _non_negative_least_squares(
    design: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Solve a tiny NNLS problem by enumerating active coefficient sets."""

    features = design.shape[1]
    best: tuple[float, tuple[float, ...], np.ndarray] | None = None
    for size in range(1, features + 1):
        for active in itertools.combinations(range(features), size):
            coefficients = np.zeros(features, dtype=np.float64)
            solution, *_ = np.linalg.lstsq(design[:, active], target, rcond=None)
            if np.any(solution < -1e-10):
                continue
            coefficients[list(active)] = np.maximum(solution, 0.0)
            residual = target - design @ coefficients
            squared_error = float(residual @ residual)
            tie_break = tuple(float(value) for value in coefficients)
            candidate = (squared_error, tie_break, coefficients)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
    if best is None:
        return np.zeros(features, dtype=np.float64)
    return best[2]


def fit_atomic_cost_model(
    observations: Sequence[AtomicCostObservation] | Iterable[AtomicCostObservation],
    *,
    batch_size: int,
) -> AtomicMaterializationCostModel:
    rows = tuple(observations)
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    if len(rows) < 4:
        raise ValueError("at least four physical observations are required")
    design = np.asarray(
        [_design_row(row.pages, row.score_events, batch_size) for row in rows],
        dtype=np.float64,
    )
    target = np.asarray([row.total_ms for row in rows], dtype=np.float64)
    coefficients = _non_negative_least_squares(design, target)
    return AtomicMaterializationCostModel(
        batch_size=batch_size,
        setup_ms=float(coefficients[0]),
        page_ms=float(coefficients[1]),
        batch_ms=float(coefficients[2]),
        score_event_ms=float(coefficients[3]),
    )

