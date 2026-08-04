"""Causal bounded-window replay for progressive cohort construction.

The scheduler in this module never observes relevance labels, visual scores, or
queries that have not arrived.  Quality gains are passed only to the replay
accountant after a batch has been selected.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


Policy = str
POLICIES = (
    "fifo",
    "random",
    "history_popularity",
    "frontier",
    "frontier_fair",
)


def normalize_cohorts(
    cohorts: Sequence[Sequence[int]],
) -> tuple[frozenset[int], ...]:
    normalized = tuple(frozenset(int(page) for page in cohort) for cohort in cohorts)
    if not normalized or any(not cohort for cohort in normalized):
        raise ValueError("cohorts must be a non-empty sequence of non-empty sets")
    return normalized


def make_arrival_times(
    query_count: int,
    *,
    model: str,
    seed: int,
    burst_size: int = 32,
    burst_interval: float = 64.0,
    poisson_mean: float = 4.0,
) -> np.ndarray:
    """Return monotonically increasing arrival times in encoded-page work units."""

    if query_count <= 0:
        raise ValueError("query_count must be positive")
    if model == "burst":
        if burst_size <= 0 or burst_interval < 0:
            raise ValueError("burst size must be positive and interval non-negative")
        return (np.arange(query_count) // burst_size).astype(np.float64) * float(
            burst_interval
        )
    if model == "poisson":
        if poisson_mean <= 0:
            raise ValueError("poisson_mean must be positive")
        rng = np.random.default_rng(seed)
        gaps = rng.exponential(poisson_mean, size=query_count)
        gaps[0] = 0.0
        return np.cumsum(gaps, dtype=np.float64)
    raise ValueError(f"unknown arrival model: {model}")


@dataclass(frozen=True)
class ReplaySummary:
    policy: str
    window: int
    dispatch_order: tuple[int, ...]
    final_unique_pages: int
    completion_pages: tuple[float, ...]
    wait_work: tuple[float, ...]
    sojourn_work: tuple[float, ...]
    bypass_count: tuple[int, ...]
    quality_work_auc: float

    def as_dict(self) -> dict:
        completion = np.asarray(self.completion_pages, dtype=np.float64)
        wait = np.asarray(self.wait_work, dtype=np.float64)
        sojourn = np.asarray(self.sojourn_work, dtype=np.float64)
        bypass = np.asarray(self.bypass_count, dtype=np.int64)
        starved = bypass >= self.window
        return {
            "policy": self.policy,
            "window": self.window,
            "final_unique_pages": self.final_unique_pages,
            "completion_pages": {
                "mean": float(completion.mean()),
                "p50": float(np.quantile(completion, 0.50)),
                "p95": float(np.quantile(completion, 0.95)),
                "max": float(completion.max()),
            },
            "quality_work_auc": self.quality_work_auc,
            "wait_work": {
                "mean": float(wait.mean()),
                "p50": float(np.quantile(wait, 0.50)),
                "p95": float(np.quantile(wait, 0.95)),
                "max": float(wait.max()),
            },
            "sojourn_work": {
                "mean": float(sojourn.mean()),
                "p50": float(np.quantile(sojourn, 0.50)),
                "p95": float(np.quantile(sojourn, 0.95)),
                "max": float(sojourn.max()),
            },
            "starvation": {
                "definition": (
                    "a query is counted when at least window younger arrivals "
                    "are dispatched before it"
                ),
                "count": int(starved.sum()),
                "fraction": float(starved.mean()),
                "max_younger_bypass": int(bypass.max()),
            },
        }


def replay_windowed_arrivals(
    cohorts: Sequence[Sequence[int]],
    arrival_order: Sequence[int],
    arrival_times: Sequence[float] | np.ndarray,
    quality_gain: Sequence[float] | np.ndarray,
    *,
    base_mean_quality: float,
    corpus_pages: int,
    batch_size: int,
    window: int,
    policy: Policy,
    random_seed: int,
) -> ReplaySummary:
    """Replay one causal arrival stream with an atomic construction batch.

    ``window`` limits the scheduler to the oldest W arrived and pending
    requests.  The frontier can inspect cheap candidate membership inside that
    window, plus resident state and past arrivals.  Quality labels are consumed
    only after selection to integrate the frozen quality--work curve.
    """

    normalized = normalize_cohorts(cohorts)
    query_count = len(normalized)
    order = np.asarray(arrival_order, dtype=np.int64)
    times = np.asarray(arrival_times, dtype=np.float64)
    gains = np.asarray(quality_gain, dtype=np.float64)
    if sorted(order.tolist()) != list(range(query_count)):
        raise ValueError("arrival_order must be a query permutation")
    if times.shape != (query_count,) or not np.isfinite(times).all():
        raise ValueError("arrival_times must be a finite arrival-aligned vector")
    if np.any(np.diff(times) < 0) or np.any(times < 0):
        raise ValueError("arrival_times must be non-negative and monotonic")
    if gains.shape != (query_count,) or not np.isfinite(gains).all():
        raise ValueError("quality_gain must be a finite query-aligned vector")
    if batch_size <= 0 or window <= 0 or corpus_pages <= 0:
        raise ValueError("batch size, window, and corpus pages must be positive")
    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy}")

    rng = np.random.default_rng(random_seed)
    arrival_rank = np.empty(query_count, dtype=np.int64)
    arrival_rank[order] = np.arange(query_count)
    query_arrival = np.empty(query_count, dtype=np.float64)
    query_arrival[order] = times

    # A query's popularity priority is frozen when it arrives.  It only uses
    # candidate pages from earlier arrivals, never the unseen suffix.
    past_frequency: Counter[int] = Counter()
    history_priority = np.zeros(query_count, dtype=np.int64)
    for query in order:
        history_priority[query] = sum(
            past_frequency[page] for page in normalized[int(query)]
        )
        past_frequency.update(normalized[int(query)])

    resident: set[int] = set()
    pending: list[int] = []
    next_arrival = 0
    work_clock = 0.0
    dispatch_order: list[int] = []
    completion_pages = np.zeros(query_count, dtype=np.float64)
    wait_work = np.zeros(query_count, dtype=np.float64)
    sojourn_work = np.zeros(query_count, dtype=np.float64)
    dispatch_rank = np.empty(query_count, dtype=np.int64)
    deferrals = np.zeros(query_count, dtype=np.int64)
    current_quality = float(base_mean_quality)
    quality_points: list[tuple[int, float]] = [(0, current_quality)]

    while len(dispatch_order) < query_count:
        while next_arrival < query_count and times[next_arrival] <= work_clock:
            pending.append(int(order[next_arrival]))
            next_arrival += 1
        if not pending:
            work_clock = float(times[next_arrival])
            continue

        service_start = work_clock
        batch: list[int] = []
        staged: set[int] = set()
        while pending and len(batch) < batch_size:
            visible = pending[: min(window, len(pending))]
            if policy == "fifo":
                selected = visible[0]
            elif policy == "random":
                selected = visible[int(rng.integers(len(visible)))]
            elif policy == "history_popularity":
                selected = min(
                    visible,
                    key=lambda query: (
                        -int(history_priority[query]),
                        int(arrival_rank[query]),
                    ),
                )
            else:
                visible_frequency = Counter(
                    page for query in visible for page in normalized[query]
                )

                def frontier_key(query: int) -> tuple[int, int, int, int, int]:
                    cohort = normalized[query]
                    uncovered = cohort - resident - staged
                    return (
                        len(uncovered),
                        -len(cohort & resident),
                        -len(cohort & staged),
                        -sum(visible_frequency[page] for page in uncovered),
                        int(arrival_rank[query]),
                    )

                fairness_due = [
                    query
                    for query in visible
                    if policy == "frontier_fair"
                    and deferrals[query] >= window - 1
                ]
                selected = (
                    min(fairness_due, key=lambda query: int(arrival_rank[query]))
                    if fairness_due
                    else min(visible, key=frontier_key)
                )
            pending.remove(selected)
            if policy == "frontier_fair":
                for query in pending:
                    if arrival_rank[query] < arrival_rank[selected]:
                        deferrals[query] += 1
            batch.append(selected)
            staged.update(normalized[selected] - resident)

        new_pages = len(staged)
        resident.update(staged)
        work_clock += float(new_pages)
        for query in batch:
            dispatch_rank[query] = len(dispatch_order)
            dispatch_order.append(query)
            completion_pages[query] = len(resident)
            wait_work[query] = service_start - query_arrival[query]
            sojourn_work[query] = work_clock - query_arrival[query]
        current_quality += float(gains[batch].sum() / query_count)
        quality_points.append((len(resident), current_quality))

    if len(resident) > corpus_pages:
        raise ValueError("cohort union exceeds corpus size")

    # Left-continuous integration: a batch's improvement is visible after its
    # construction work is paid.  The denominator is full-corpus build work.
    quality_area = 0.0
    previous_pages = 0
    previous_quality = float(base_mean_quality)
    for page_count, quality in quality_points[1:]:
        quality_area += (page_count - previous_pages) * previous_quality
        previous_pages = page_count
        previous_quality = quality
    quality_area += (corpus_pages - previous_pages) * previous_quality

    bypass = np.zeros(query_count, dtype=np.int64)
    for query in range(query_count):
        younger = arrival_rank > arrival_rank[query]
        bypass[query] = int(np.sum(younger & (dispatch_rank < dispatch_rank[query])))

    return ReplaySummary(
        policy=policy,
        window=window,
        dispatch_order=tuple(dispatch_order),
        final_unique_pages=len(resident),
        completion_pages=tuple(completion_pages.tolist()),
        wait_work=tuple(wait_work.tolist()),
        sojourn_work=tuple(sojourn_work.tolist()),
        bypass_count=tuple(int(value) for value in bypass),
        quality_work_auc=float(quality_area / corpus_pages),
    )
