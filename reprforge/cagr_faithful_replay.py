"""Causal CaGR-RAG adaptation for visual-page construction replay.

The scheduler API contains candidate-page membership, arrival metadata, and
resident state only.  Quality gains are consumed after a request batch has
been selected and completed.
"""

from __future__ import annotations

from collections import Counter, OrderedDict, deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


POLICIES = (
    "fifo",
    "overlap_only",
    "history_popularity",
    "static_popularity",
    "cagr_faithful",
    "frontier",
)


def normalize_cohorts(
    cohorts: Sequence[Sequence[int]],
) -> tuple[frozenset[int], ...]:
    normalized = tuple(frozenset(int(page) for page in cohort) for cohort in cohorts)
    if not normalized or any(not cohort for cohort in normalized):
        raise ValueError("cohorts must be a non-empty sequence of non-empty sets")
    return normalized


def jaccard(left: frozenset[int], right: frozenset[int]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def form_cagr_groups(
    queries: Sequence[int],
    cohorts: Sequence[Sequence[int]],
    *,
    theta: float = 0.5,
    membership_rule: str = "max",
) -> tuple[tuple[int, ...], ...]:
    """Implement CaGR-RAG Algorithm 1 in deterministic arrival order.

    ``max`` follows Algorithm 1 line 10: a query joins the first group having
    at least one member above threshold. ``all`` is the stricter Equation 3
    diagnostic and is not the preregistered primary rule.
    """

    if not 0.0 <= theta <= 1.0:
        raise ValueError("theta must be in [0, 1]")
    if membership_rule not in {"max", "all"}:
        raise ValueError("membership_rule must be 'max' or 'all'")
    normalized = normalize_cohorts(cohorts)
    ordered = [int(query) for query in queries]
    if len(set(ordered)) != len(ordered):
        raise ValueError("queries must be unique")
    if any(query < 0 or query >= len(normalized) for query in ordered):
        raise ValueError("query index is out of range")

    groups: list[list[int]] = []
    for query in ordered:
        selected: list[int] | None = None
        for group in groups:
            similarities = [
                jaccard(normalized[query], normalized[member]) for member in group
            ]
            matches = (
                max(similarities) >= theta
                if membership_rule == "max"
                else min(similarities) >= theta
            )
            if matches:
                selected = group
                break
        if selected is None:
            groups.append([query])
        else:
            selected.append(query)
    return tuple(tuple(group) for group in groups)


def form_fixed_jaccard_groups(
    queries: Sequence[int],
    cohorts: Sequence[Sequence[int]],
    *,
    target_group_size: int,
) -> tuple[tuple[int, ...], ...]:
    """Greedy fixed-size Jaccard agglomeration for a strong CaGR adaptation.

    Seeds maximize total Jaccard degree over the remaining pool.  Each next
    member maximizes average Jaccard to the current group, then its strongest
    single link.  Input order is the deterministic final tie-break.
    """

    if target_group_size <= 0:
        raise ValueError("target_group_size must be positive")
    normalized = normalize_cohorts(cohorts)
    ordered = [int(query) for query in queries]
    if len(set(ordered)) != len(ordered):
        raise ValueError("queries must be unique")
    if any(query < 0 or query >= len(normalized) for query in ordered):
        raise ValueError("query index is out of range")
    rank = {query: index for index, query in enumerate(ordered)}
    similarity: dict[tuple[int, int], float] = {}

    def score(left: int, right: int) -> float:
        key = (min(left, right), max(left, right))
        if key not in similarity:
            similarity[key] = jaccard(normalized[left], normalized[right])
        return similarity[key]

    remaining = set(ordered)
    groups: list[tuple[int, ...]] = []
    while remaining:
        seed = min(
            remaining,
            key=lambda query: (
                -sum(score(query, other) for other in remaining if other != query),
                rank[query],
            ),
        )
        group = [seed]
        remaining.remove(seed)
        while remaining and len(group) < target_group_size:
            selected = min(
                remaining,
                key=lambda query: (
                    -sum(score(query, member) for member in group) / len(group),
                    -max(score(query, member) for member in group),
                    rank[query],
                ),
            )
            remaining.remove(selected)
            group.append(selected)
        groups.append(tuple(group))
    return tuple(groups)


@dataclass(frozen=True)
class ReplayResult:
    policy: str
    dispatch_order: tuple[int, ...]
    completion_pages: tuple[float, ...]
    completion_unit_cost: tuple[float, ...]
    wait_page_work: tuple[float, ...]
    sojourn_page_work: tuple[float, ...]
    bypass_count: tuple[int, ...]
    final_unique_pages: int
    quality_work_auc: float
    normalized_quality_regret_auc: float | None
    cache: dict[str, Any]
    prefetch: dict[str, Any]
    groups: dict[str, Any]
    request_batches: dict[str, Any]

    def as_dict(self, *, starvation_window: int) -> dict[str, Any]:
        completion = np.asarray(self.completion_pages, dtype=np.float64)
        cost = np.asarray(self.completion_unit_cost, dtype=np.float64)
        wait = np.asarray(self.wait_page_work, dtype=np.float64)
        sojourn = np.asarray(self.sojourn_page_work, dtype=np.float64)
        bypass = np.asarray(self.bypass_count, dtype=np.int64)
        starved = bypass >= starvation_window

        def distribution(values: np.ndarray) -> dict[str, float]:
            return {
                "mean": float(values.mean()),
                "p50": float(np.quantile(values, 0.50)),
                "p95": float(np.quantile(values, 0.95)),
                "max": float(values.max()),
            }

        return {
            "policy": self.policy,
            "final_unique_pages": self.final_unique_pages,
            "completion_pages": distribution(completion),
            "completion_unit_cost": distribution(cost),
            "wait_page_work": distribution(wait),
            "sojourn_page_work": distribution(sojourn),
            "quality_work_auc": self.quality_work_auc,
            "normalized_quality_regret_auc": self.normalized_quality_regret_auc,
            "starvation": {
                "count": int(starved.sum()),
                "fraction": float(starved.mean()),
                "max_younger_bypass": int(bypass.max()),
            },
            "cache": self.cache,
            "prefetch": self.prefetch,
            "groups": self.groups,
            "request_batches": self.request_batches,
        }


class _ActiveCache:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("cache capacity must be positive")
        self.capacity = capacity
        self.pages: OrderedDict[int, None] = OrderedDict()

    def contains(self, page: int) -> bool:
        return page in self.pages

    def touch(self, page: int) -> int | None:
        self.pages[page] = None
        self.pages.move_to_end(page)
        if len(self.pages) > self.capacity:
            evicted, _ = self.pages.popitem(last=False)
            return evicted
        return None


def _quality_metrics(
    points: list[tuple[int, float]],
    *,
    base_quality: float,
    final_quality: float,
    final_work: int,
    corpus_pages: int,
) -> tuple[float, float | None]:
    compressed: list[tuple[int, float]] = []
    for work, quality in points:
        if compressed and work == compressed[-1][0]:
            compressed[-1] = (work, quality)
        else:
            compressed.append((work, quality))

    corpus_area = 0.0
    final_area = 0.0
    previous_work = 0
    previous_quality = float(base_quality)
    for work, quality in compressed[1:]:
        width = work - previous_work
        corpus_area += width * previous_quality
        final_area += width * (previous_quality - base_quality)
        previous_work = work
        previous_quality = quality
    corpus_area += (corpus_pages - previous_work) * previous_quality
    if previous_work < final_work:
        final_area += (final_work - previous_work) * (
            previous_quality - base_quality
        )
    final_gain = final_quality - base_quality
    normalized_regret = (
        1.0 - final_area / (final_work * final_gain)
        if final_gain > 1e-12
        else None
    )
    return corpus_area / corpus_pages, normalized_regret


def replay_cagr_comparison(
    cohorts: Sequence[Sequence[int]],
    arrival_order: Sequence[int],
    arrival_times: Sequence[float] | np.ndarray,
    quality_gain: Sequence[float] | np.ndarray,
    *,
    base_mean_quality: float,
    corpus_pages: int,
    request_batch_size: int = 8,
    window: int = 64,
    policy: str,
    cache_capacity: int = 80,
    cagr_group_pool: int = 64,
    cagr_theta: float = 0.5,
    cagr_membership_rule: str = "max",
    cagr_grouping: str = "threshold",
    cagr_target_group_size: int = 8,
) -> ReplayResult:
    """Replay one policy with persistent compilation and equal active LRU."""

    normalized = normalize_cohorts(cohorts)
    query_count = len(normalized)
    order = np.asarray(arrival_order, dtype=np.int64)
    times = np.asarray(arrival_times, dtype=np.float64)
    gains = np.asarray(quality_gain, dtype=np.float64)
    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy}")
    if sorted(order.tolist()) != list(range(query_count)):
        raise ValueError("arrival_order must be a query permutation")
    if times.shape != (query_count,) or np.any(np.diff(times) < 0):
        raise ValueError("arrival_times must be a monotonic aligned vector")
    if gains.shape != (query_count,) or not np.isfinite(gains).all():
        raise ValueError("quality_gain must be a finite aligned vector")
    if request_batch_size <= 0 or window <= 0 or cagr_group_pool <= 0:
        raise ValueError("batch, window, and group pool must be positive")
    if cagr_grouping not in {"threshold", "fixed_jaccard"}:
        raise ValueError("cagr_grouping must be 'threshold' or 'fixed_jaccard'")
    if cagr_target_group_size <= 0:
        raise ValueError("cagr_target_group_size must be positive")
    if corpus_pages <= 0:
        raise ValueError("corpus_pages must be positive")

    arrival_rank = np.empty(query_count, dtype=np.int64)
    arrival_rank[order] = np.arange(query_count)
    query_arrival = np.empty(query_count, dtype=np.float64)
    query_arrival[order] = times

    past_frequency: Counter[int] = Counter()
    global_frequency = Counter(page for cohort in normalized for page in cohort)
    history_priority = np.zeros(query_count, dtype=np.int64)
    for query in order:
        history_priority[query] = sum(
            past_frequency[page] for page in normalized[int(query)]
        )
        past_frequency.update(normalized[int(query)])

    compiled: set[int] = set()
    cache = _ActiveCache(cache_capacity)
    pending: list[int] = []
    reserved: set[int] = set()
    next_arrival = 0
    page_clock = 0.0
    unit_cost_clock = 0.0
    dispatch: list[int] = []
    dispatch_rank = np.empty(query_count, dtype=np.int64)
    completion_pages = np.zeros(query_count, dtype=np.float64)
    completion_cost = np.zeros(query_count, dtype=np.float64)
    wait_page = np.zeros(query_count, dtype=np.float64)
    sojourn_page = np.zeros(query_count, dtype=np.float64)
    current_quality = float(base_mean_quality)
    quality_points: list[tuple[int, float]] = [(0, current_quality)]
    plan: deque[tuple[tuple[int, ...], tuple[int, ...] | None]] = deque()

    demand_events = demand_hits = demand_builds = demand_reloads = 0
    prefetch_events = prefetch_builds = prefetch_reloads = 0
    useful_prefetches = wasted_prefetches = 0
    wasted_prefetch_cost = 0.0
    outstanding_prefetch: dict[int, float] = {}
    group_sizes: list[int] = []
    query_batch_sizes: list[int] = []

    def insert_cache(page: int) -> None:
        nonlocal wasted_prefetches, wasted_prefetch_cost
        evicted = cache.touch(page)
        if evicted is not None and evicted in outstanding_prefetch:
            wasted_prefetches += 1
            wasted_prefetch_cost += outstanding_prefetch.pop(evicted)

    def demand(page: int) -> None:
        nonlocal page_clock, unit_cost_clock
        nonlocal demand_events, demand_hits, demand_builds, demand_reloads
        nonlocal useful_prefetches
        demand_events += 1
        if page in outstanding_prefetch:
            useful_prefetches += 1
            outstanding_prefetch.pop(page)
        if cache.contains(page):
            demand_hits += 1
            cache.touch(page)
            return
        if page in compiled:
            demand_reloads += 1
        else:
            demand_builds += 1
            compiled.add(page)
            page_clock += 1.0
        unit_cost_clock += 1.0
        insert_cache(page)

    def prefetch(pages: Sequence[int]) -> None:
        nonlocal page_clock, unit_cost_clock
        nonlocal prefetch_events, prefetch_builds, prefetch_reloads
        for page in sorted(set(int(value) for value in pages)):
            if cache.contains(page):
                continue
            prefetch_events += 1
            if page in compiled:
                prefetch_reloads += 1
            else:
                prefetch_builds += 1
                compiled.add(page)
                page_clock += 1.0
            unit_cost_clock += 1.0
            outstanding_prefetch[page] = 1.0
            insert_cache(page)
        quality_points.append((len(compiled), current_quality))

    def build_cagr_plan() -> None:
        visible_count = min(window, cagr_group_pool, len(pending))
        visible = pending[:visible_count]
        groups = (
            form_cagr_groups(
                visible,
                normalized,
                theta=cagr_theta,
                membership_rule=cagr_membership_rule,
            )
            if cagr_grouping == "threshold"
            else form_fixed_jaccard_groups(
                visible,
                normalized,
                target_group_size=cagr_target_group_size,
            )
        )
        for query in visible:
            pending.remove(query)
            reserved.add(query)
        group_sizes.extend(len(group) for group in groups)
        for group_index, group in enumerate(groups):
            next_pages = (
                tuple(sorted(normalized[groups[group_index + 1][0]]))
                if group_index + 1 < len(groups)
                else None
            )
            for start in range(0, len(group), request_batch_size):
                batch = tuple(group[start : start + request_batch_size])
                after = next_pages if start + request_batch_size >= len(group) else None
                plan.append((batch, after))

    def select_regular_batch() -> tuple[int, ...]:
        batch: list[int] = []
        staged: set[int] = set()
        while pending and len(batch) < request_batch_size:
            # Match the frozen windowed-arrival implementation exactly: after
            # selecting one request, the next-oldest pending request may enter
            # the W-sized visible prefix while the atomic batch is packed.
            visible = pending[: min(window, len(pending))]
            if policy == "fifo":
                selected = visible[0]
            elif policy == "history_popularity":
                selected = min(
                    visible,
                    key=lambda query: (
                        -int(history_priority[query]),
                        int(arrival_rank[query]),
                    ),
                )
            elif policy == "static_popularity":
                selected = min(
                    visible,
                    key=lambda query: (
                        -sum(global_frequency[page] for page in normalized[query]),
                        int(arrival_rank[query]),
                    ),
                )
            elif policy == "overlap_only":
                if not batch:
                    frequency = Counter(
                        page for query in visible for page in normalized[query]
                    )
                    selected = min(
                        visible,
                        key=lambda query: (
                            -sum(frequency[page] - 1 for page in normalized[query]),
                            int(arrival_rank[query]),
                        ),
                    )
                else:
                    selected = min(
                        visible,
                        key=lambda query: (
                            -len(normalized[query] & staged),
                            len(normalized[query] | staged),
                            int(arrival_rank[query]),
                        ),
                    )
            elif policy == "frontier":
                frequency = Counter(
                    page for query in visible for page in normalized[query]
                )
                selected = min(
                    visible,
                    key=lambda query: (
                        len(normalized[query] - compiled - staged),
                        -len(normalized[query] & compiled),
                        -len(normalized[query] & staged),
                        -sum(
                            frequency[page]
                            for page in normalized[query] - compiled - staged
                        ),
                        int(arrival_rank[query]),
                    ),
                )
            else:
                raise AssertionError("regular selection called for CaGR")
            pending.remove(selected)
            batch.append(selected)
            staged.update(normalized[selected] - compiled)
        return tuple(batch)

    while len(dispatch) < query_count:
        while next_arrival < query_count and times[next_arrival] <= page_clock:
            pending.append(int(order[next_arrival]))
            next_arrival += 1
        if policy == "cagr_faithful" and not plan and pending:
            build_cagr_plan()
        if policy == "cagr_faithful" and plan:
            batch, after_prefetch = plan.popleft()
        elif pending:
            batch = select_regular_batch()
            after_prefetch = None
        else:
            if next_arrival >= query_count:
                break
            idle = float(times[next_arrival] - page_clock)
            page_clock += idle
            unit_cost_clock += idle
            continue

        service_start = page_clock
        for query in batch:
            for page in sorted(normalized[query]):
                demand(page)
        query_batch_sizes.append(len(batch))
        for query in batch:
            if query in reserved:
                reserved.remove(query)
            dispatch_rank[query] = len(dispatch)
            dispatch.append(query)
            completion_pages[query] = len(compiled)
            completion_cost[query] = unit_cost_clock
            wait_page[query] = service_start - query_arrival[query]
            sojourn_page[query] = page_clock - query_arrival[query]
        current_quality += float(gains[list(batch)].sum() / query_count)
        quality_points.append((len(compiled), current_quality))
        if after_prefetch is not None:
            prefetch(after_prefetch)

    if reserved or plan or pending:
        raise AssertionError("replay ended with undispatched queries")
    if len(compiled) > corpus_pages:
        raise ValueError("cohort union exceeds corpus pages")
    final_union = len(set().union(*normalized))
    if len(compiled) != final_union:
        raise AssertionError("policy changed final candidate union")
    for cost in outstanding_prefetch.values():
        wasted_prefetches += 1
        wasted_prefetch_cost += cost

    bypass = np.zeros(query_count, dtype=np.int64)
    for query in range(query_count):
        younger = arrival_rank > arrival_rank[query]
        bypass[query] = int(np.sum(younger & (dispatch_rank < dispatch_rank[query])))

    final_quality = float(base_mean_quality + gains.mean())
    quality_auc, normalized_regret = _quality_metrics(
        quality_points,
        base_quality=base_mean_quality,
        final_quality=final_quality,
        final_work=final_union,
        corpus_pages=corpus_pages,
    )
    group_array = np.asarray(group_sizes, dtype=np.float64)
    batch_array = np.asarray(query_batch_sizes, dtype=np.float64)
    return ReplayResult(
        policy=policy,
        dispatch_order=tuple(dispatch),
        completion_pages=tuple(completion_pages.tolist()),
        completion_unit_cost=tuple(completion_cost.tolist()),
        wait_page_work=tuple(wait_page.tolist()),
        sojourn_page_work=tuple(sojourn_page.tolist()),
        bypass_count=tuple(int(value) for value in bypass),
        final_unique_pages=len(compiled),
        quality_work_auc=quality_auc,
        normalized_quality_regret_auc=normalized_regret,
        cache={
            "capacity_pages": cache_capacity,
            "demand_events": demand_events,
            "hits": demand_hits,
            "hit_fraction": demand_hits / demand_events if demand_events else 0.0,
            "builds": demand_builds,
            "reloads": demand_reloads,
        },
        prefetch={
            "events": prefetch_events,
            "builds": prefetch_builds,
            "reloads": prefetch_reloads,
            "useful": useful_prefetches,
            "wasted": wasted_prefetches,
            "precision": (
                useful_prefetches / prefetch_events if prefetch_events else None
            ),
            "unused_unit_work": wasted_prefetch_cost,
        },
        groups={
            "grouping": cagr_grouping if policy == "cagr_faithful" else None,
            "count": len(group_sizes),
            "singleton_count": sum(size == 1 for size in group_sizes),
            "singleton_fraction": (
                sum(size == 1 for size in group_sizes) / len(group_sizes)
                if group_sizes
                else None
            ),
            "size_mean": float(group_array.mean()) if len(group_array) else None,
            "size_p50": (
                float(np.quantile(group_array, 0.50)) if len(group_array) else None
            ),
            "size_p95": (
                float(np.quantile(group_array, 0.95)) if len(group_array) else None
            ),
            "size_max": int(group_array.max()) if len(group_array) else None,
        },
        request_batches={
            "count": len(query_batch_sizes),
            "query_slots_used_fraction": float(
                batch_array.sum() / (len(batch_array) * request_batch_size)
            ),
            "size_mean": float(batch_array.mean()),
            "size_p50": float(np.quantile(batch_array, 0.50)),
            "size_min": int(batch_array.min()),
            "size_max": int(batch_array.max()),
        },
    )
