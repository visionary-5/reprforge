"""Strong qrel-free baselines for queued cohort construction.

Every policy in this module is allowed to observe only the query--page
candidate graph.  Relevance labels and visual scores are deliberately absent
from the scheduling API.  The policies reorder whole queries; publication is
still atomic at request-batch boundaries.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence


def normalize_cohorts(
    cohorts: Sequence[Sequence[int]],
) -> tuple[frozenset[int], ...]:
    normalized = tuple(frozenset(int(page) for page in cohort) for cohort in cohorts)
    if not normalized or any(not cohort for cohort in normalized):
        raise ValueError("cohorts must be a non-empty sequence of non-empty sets")
    return normalized


def _validate_batch_size(batch_size: int) -> None:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")


def _batched_greedy_order(
    cohorts: tuple[frozenset[int], ...],
    *,
    batch_size: int,
    select_key: Callable[
        [int, set[int], set[int], set[int], Counter[int]], tuple
    ],
) -> list[int]:
    """Run a deterministic atomic-batch greedy policy."""

    _validate_batch_size(batch_size)
    pending = set(range(len(cohorts)))
    resident: set[int] = set()
    frequency = Counter(page for cohort in cohorts for page in cohort)
    order: list[int] = []
    while pending:
        batch: list[int] = []
        staged: set[int] = set()
        while pending and len(batch) < batch_size:
            selected = min(
                pending,
                key=lambda query: select_key(
                    query, resident, staged, pending, frequency
                ),
            )
            pending.remove(selected)
            batch.append(selected)
            staged.update(cohorts[selected] - resident)
        order.extend(batch)
        resident.update(staged)
        for query in batch:
            for page in cohorts[query]:
                frequency[page] -= 1
    return order


def shortest_missing_order(
    cohorts: Sequence[Sequence[int]], *, batch_size: int
) -> list[int]:
    """Complete the cohort adding the fewest pages, with no reuse tie-break.

    Staged pages count as available while packing an atomic batch.  This is the
    simplest explanation for a frontier result: perhaps shortest remaining
    processing time alone is sufficient.
    """

    normalized = normalize_cohorts(cohorts)

    def key(
        query: int,
        resident: set[int],
        staged: set[int],
        pending: set[int],
        frequency: Counter[int],
    ) -> tuple[int, int]:
        del pending, frequency
        return (len(normalized[query] - resident - staged), query)

    return _batched_greedy_order(
        normalized, batch_size=batch_size, select_key=key
    )


def reuse_only_order(
    cohorts: Sequence[Sequence[int]], *, batch_size: int
) -> list[int]:
    """Prioritize future candidate demand while ignoring completion distance.

    This dynamic popularity baseline updates frequencies after each published
    batch.  It tests whether reuse popularity, rather than the resident
    completion frontier, explains the result.
    """

    normalized = normalize_cohorts(cohorts)

    def key(
        query: int,
        resident: set[int],
        staged: set[int],
        pending: set[int],
        frequency: Counter[int],
    ) -> tuple[int, int]:
        del resident, staged, pending
        return (-sum(frequency[page] for page in normalized[query]), query)

    return _batched_greedy_order(
        normalized, batch_size=batch_size, select_key=key
    )


def overlap_only_order(
    cohorts: Sequence[Sequence[int]], *, batch_size: int
) -> list[int]:
    """Form static high-overlap query groups, analogous to locality grouping.

    The policy never observes resident state.  A batch seed has the largest
    weighted overlap degree in the full queued workload; remaining positions
    maximize overlap with that batch's staged union.  This isolates CaGR-like
    query grouping from progressive representation construction.
    """

    normalized = normalize_cohorts(cohorts)
    _validate_batch_size(batch_size)
    frequency = Counter(page for cohort in normalized for page in cohort)
    degree = {
        query: sum(frequency[page] - 1 for page in cohort)
        for query, cohort in enumerate(normalized)
    }
    pending = set(range(len(normalized)))
    order: list[int] = []
    while pending:
        seed = min(pending, key=lambda query: (-degree[query], query))
        pending.remove(seed)
        batch = [seed]
        staged = set(normalized[seed])
        while pending and len(batch) < batch_size:
            selected = min(
                pending,
                key=lambda query: (
                    -len(normalized[query] & staged),
                    len(normalized[query] | staged),
                    -degree[query],
                    query,
                ),
            )
            pending.remove(selected)
            batch.append(selected)
            staged.update(normalized[selected])
        order.extend(batch)
    return order


def offline_work_greedy_order(
    cohorts: Sequence[Sequence[int]], *, batch_size: int
) -> list[int]:
    """Full-queue multi-start greedy refinement of completion page-work.

    The exact objective is the sum, over queries, of cumulative unique pages
    when their atomic batch completes.  We initialize from every deterministic
    qrel-free policy, retain the best, then repeatedly repartition adjacent
    batch pairs.  For a pair, the cumulative union after both batches is
    invariant, so reducing the first batch's missing union cannot worsen the
    objective.  All possible seeds and greedy packings of the 2B-query pair are
    evaluated.  This is a computable qrel-free offline greedy/lower envelope,
    not a claim of an exact combinatorial optimum.
    """

    normalized = normalize_cohorts(cohorts)
    _validate_batch_size(batch_size)

    # Local imports avoid making the basic baseline module a prerequisite of
    # the frozen frontier implementation.
    from reprforge.cohort_frontier_scheduler import (
        frontier_reuse_order,
        static_popularity_order,
    )

    starts = [
        list(range(len(normalized))),
        static_popularity_order(normalized),
        overlap_only_order(normalized, batch_size=batch_size),
        shortest_missing_order(normalized, batch_size=batch_size),
        reuse_only_order(normalized, batch_size=batch_size),
        frontier_reuse_order(normalized, batch_size=batch_size),
    ]

    def objective(order: Sequence[int]) -> int:
        resident: set[int] = set()
        total = 0
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            resident.update(
                set().union(*(normalized[query] for query in batch))
            )
            total += len(batch) * len(resident)
        return total

    best = min(starts, key=lambda order: (objective(order), tuple(order)))
    batches = [
        best[start : start + batch_size]
        for start in range(0, len(best), batch_size)
    ]
    for _ in range(4):
        changed = False
        resident: set[int] = set()
        for index in range(len(batches) - 1):
            left_size = len(batches[index])
            combined = batches[index] + batches[index + 1]
            existing = set().union(
                *(normalized[query] for query in batches[index])
            ) - resident
            best_left = list(batches[index])
            best_missing = len(existing)
            for seed in combined:
                selected = [seed]
                available = set(normalized[seed]) | resident
                remaining = set(combined) - {seed}
                while remaining and len(selected) < left_size:
                    query = min(
                        remaining,
                        key=lambda value: (
                            len(normalized[value] - available), value
                        ),
                    )
                    remaining.remove(query)
                    selected.append(query)
                    available.update(normalized[query])
                missing = len(available - resident)
                if (missing, tuple(selected)) < (
                    best_missing,
                    tuple(best_left),
                ):
                    best_missing = missing
                    best_left = selected
            if set(best_left) != set(batches[index]):
                changed = True
                right = [query for query in combined if query not in set(best_left)]
                batches[index] = best_left
                batches[index + 1] = right
            resident.update(
                set().union(*(normalized[query] for query in batches[index]))
            )
        if not changed:
            break
    refined = [query for batch in batches for query in batch]
    if objective(refined) > objective(best):
        raise AssertionError("offline adjacent refinement worsened its seed")
    return refined


POLICY_DESCRIPTIONS = {
    "fifo": "official frozen query order",
    "random": "deterministic random permutation; aggregate separately",
    "static_popularity": "full-stream page frequency, fixed once (offline)",
    "overlap_only": "static CaGR-style cohort overlap grouping",
    "shortest_missing": "fewest absent pages, without reuse tie-break",
    "reuse_only": "dynamic future candidate frequency, without distance",
    "frontier_reuse": "fewest absent pages with staged/resident and reuse ties",
    "offline_work_greedy": (
        "full-stream multi-start adjacent-batch work refinement (offline)"
    ),
}
